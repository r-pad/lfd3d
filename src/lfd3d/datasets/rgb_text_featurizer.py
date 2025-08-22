import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoProcessor

from lfd3d.datasets.rgb_text_feature_gen import (
    get_dinov2_image_embedding,
    get_siglip_text_embedding,
)


class RGBTextFeaturizer:
    def __init__(self, target_shape=224, rgb_feat=True):
        self.target_shape = target_shape
        self.rgb_feat = rgb_feat
        self.text_embeddings = {}

        # Initialize models
        self.siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(
            "cpu"
        )
        self.siglip_processor = AutoProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384"
        )
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").to(
            "cpu"
        )

    def compute_rgb_text_feat(self, rgb, text):
        """
        Compute RGB/text features generated with DINOv2 and SIGLIP
        """
        if text not in self.text_embeddings:
            # Compute features on CPU to avoid CUDA multiprocessing issues
            # We're only computing the features once and caching so its okay.
            self.text_embeddings[text] = get_siglip_text_embedding(
                text,
                siglip=self.siglip,
                siglip_processor=self.siglip_processor,
                device="cpu",
            )
        text_embed = self.text_embeddings[text]

        if self.rgb_feat:
            # Compress RGB features
            pca_n_components = 256
            rgb_embed = get_dinov2_image_embedding(
                Image.fromarray(rgb), dinov2=self.dinov2, device="cpu"
            )

            pca_model = PCA(n_components=pca_n_components)
            rgb_embed = pca_model.fit_transform(
                rgb_embed.reshape(-1, rgb_embed.shape[2])
            )
            rgb_embed = rgb_embed.reshape(
                self.target_shape, self.target_shape, pca_n_components
            )
        else:
            # Just return the (normalized) RGB values if features are not required.
            rgb_embed = (((rgb / 255.0) * 2) - 1).astype(np.float32)

        return rgb_embed, text_embed
