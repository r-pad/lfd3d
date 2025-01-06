import math
from functools import partial
from typing import Optional

import spconv.pytorch as spconv
import torch
import torch_scatter
from addict import Dict
from timm.layers import DropPath
from torch import nn

from lfd3d.models.dit.models import TimestepEmbedder
from lfd3d.models.PointTransformerV3.model import (
    MLP,
    Embedding,
    PDNorm,
    Point,
    PointModule,
    PointSequential,
    SerializedAttention,
    SerializedUnpooling,
)


class DiPTv3SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before DiPTv3SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
            # For DiPTv3: keep track of these as well.
            t=point.t,
            N=point.N,
            t_emb=point.t_emb,
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class DiPTv3Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        t_hidden_size=128,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        assert pre_norm, "Only support pre-normalization"

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        # Adapted from the DiT paper - add adaLN modulation.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_hidden_size, 6 * channels, bias=True),
        )
        self.t_hidden_size = t_hidden_size

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, point: Point):
        # Time embedding should still be batched.
        assert len(point.t_emb.shape) == 2

        # First, compute adaLN modulation.
        adaLN_params = self.adaLN_modulation(point.t_emb)

        # Then, repeat with batching in mind.
        adaLN_params = adaLN_params[point.batch]

        # Compute the adaLN parameters.
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = adaLN_params.chunk(6, dim=1)

        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)

            # Apply adaLN modulation.
            point.feat = self.modulate(point.feat, shift_msa, scale_msa)

        # Attention.
        point = self.attn(point)

        # Modulation.
        point.feat = gate_msa * point.feat

        # Residual connection.
        point = self.drop_path(point)
        point.feat = shortcut + point.feat

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)

            # Apply adaLN modulation.
            point.feat = self.modulate(point.feat, shift_mlp, scale_mlp)

        # MLP.
        point = self.mlp(point)

        # Modulation.
        point.feat = gate_mlp * point.feat

        # Residual connection.
        point = self.drop_path(point)
        point.feat = shortcut + point.feat

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class DiPTv3(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        t_hidden_size=128,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders
        self.t_hidden_size = t_hidden_size

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        self.t_embedder = TimestepEmbedder(t_hidden_size)

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    DiPTv3SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    DiPTv3Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        t_hidden_size=t_hidden_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        DiPTv3Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            t_hidden_size=t_hidden_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        """
        A data_dict is a dictionary containing properties of a batched point cloud.
        It should contain the following properties for PTv3:
        1. "feat": feature of point cloud
        2. "grid_coord": discrete coordinate after grid sampling (voxelization) or "coord" + "grid_size"
        3. "offset" or "batch": https://github.com/Pointcept/Pointcept?tab=readme-ov-file#offset
        """
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point.t_emb = self.t_embedder(point.t)
        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)
        return point


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, t_hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(t_hidden_size, 2 * hidden_size, bias=True)
        )

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, c) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x  # type: ignore


class DiPTv3Adapter(nn.Module):
    def __init__(self, model, grid_size=0.001, final_dimension=6):
        super().__init__()
        self.model = model
        self.grid_size = grid_size

        text_dim = 384
        hidden_dim = 128
        self.text_projection_dim = 32
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.text_projection_dim),
        )

        final_input_size = model.dec[-1][-1].mlp[0].fc2.out_features
        self.final_layer = FinalLayer(
            final_input_size, model.t_hidden_size, final_dimension
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.model.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.model.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for enc in self.model.enc:
            for block in enc:
                if hasattr(block, "adaLN_modulation"):
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for dec in self.model.dec:
            for block in dec:
                if hasattr(block, "adaLN_modulation"):
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Denoise the input (a point cloud).
        """
        # Permute to (B, N, C)
        assert len(x.shape) == 3
        assert x.shape[1] == 3

        x = x.permute(0, 2, 1)  # type: ignore
        x0 = x0.permute(0, 2, 1)  # type: ignore
        if y is not None:
            y = y.permute(0, 2, 1)
        B, N, C_in = x.shape
        B0, N0, D = x0.shape
        B1, N1, D1 = y.shape if y is not None else (0, 0, 0)

        assert t.shape[0] == B

        if y is not None:
            # Concatenate y and x0 on the second-to-last dim (N)
            x0 = torch.cat([x0, y], dim=1)
            # Add zeros to the x tensor.
            x = torch.cat([x, torch.zeros(B, N1, C_in, device=x.device)], dim=1)

            # Create a labels tensor (one-hot encoded) for x0 and y.
            # This is a tensor of shape (B, N0 + N1, 2).
            labels = torch.cat(
                [
                    torch.zeros(B, N0, 1, device=x.device),
                    torch.ones(B, N1, 1, device=x.device),
                ],
                dim=1,
            )
            # Make one-hot
            labels = torch.cat([1 - labels, labels], dim=-1)
        else:
            labels = None

        if text_embed is not None:
            # Project text embedding
            text_emb = self.text_projector(text_embed)
            # Zero-pad tensor for anchor points
            text_emb = torch.cat(
                [
                    text_emb,
                    torch.zeros(B, N1, self.text_projection_dim, device=x.device),
                ],
                dim=1,
            )
            text_emb = text_emb.reshape(-1, self.text_projection_dim)

        # Reshape to (BxN, C), and create a batch vector with the indices.
        # Right now assuming that the batch has same number of points for each example.
        x0_flat = x0.reshape(-1, D)
        x_flat = x.reshape(-1, C_in)
        if labels is not None:
            labels = labels.reshape(-1, labels.shape[-1])
        batch_ixs = torch.repeat_interleave(
            torch.arange(B, device=x0.device), x0.shape[1]
        )

        cat_feat = torch.cat([x0_flat, x_flat], dim=-1)
        if y is not None:
            cat_feat = torch.cat([cat_feat, labels], dim=-1)
        if text_embed is not None:
            cat_feat = torch.cat([cat_feat, text_emb], dim=-1)

        data = Point(
            coord=x0_flat,
            feat=cat_feat,
            batch=batch_ixs,
            grid_size=self.grid_size,  # Not sure what to do here...
            t=t,
            N=N0 + N1,
        )
        pred = self.model(data)
        C_out = pred.feat.shape[-1]

        # Only need the action points.
        # Reshape back to (B, N, C)
        feats = pred.feat.reshape(B, -1, C_out)

        # Final linear layer to get the output.
        feats = self.final_layer(feats, pred.t_emb)

        feats = feats[:, :N, :]

        # Permute back to what we expect.
        feats = feats.permute(0, 2, 1)
        return feats  # type: ignore
