# from articubot -> model_invariant.py

# NOTE:
# Trying to implement PointNet++
# Borrowed from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import random
from collections import defaultdict
from time import time
from typing import Dict, List

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from diffusers import get_cosine_schedule_with_warmup
from pytorch3d.structures import Pointclouds
from torch import nn, optim

from lfd3d.models.dino_heatmap import calc_pix_metrics
from lfd3d.models.tax3d import calc_pcd_metrics
from lfd3d.utils.viz_utils import (
    get_action_anchor_pcd,
    get_img_and_track_pcd,
    invert_augmentation_and_normalization,
    project_pcd_on_image,
)


def timeit(tag, t):
    print(f"{tag}: {time() - t}s")
    return time()


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz_, npoint, keep_gripper_in_fps=False):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    if keep_gripper_in_fps:  # NOTE: assuming there are 4 gripper points
        xyz = xyz_[:, :-4, :]
        npoint = npoint - 4
    else:
        xyz = xyz_

    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    farthest = farthest * 0  # set to 0
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    if keep_gripper_in_fps:
        gripper_indices = torch.Tensor([N, N + 1, N + 2, N + 3]).long().to(device)
        gripper_indices = gripper_indices.unsqueeze(0).repeat(B, 1)
        centroids = torch.cat([centroids, gripper_indices], dim=1)
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = (
        torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    )
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points], dim=-1
        )  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(
        self,
        npoint,
        radius_list,
        nsample_list,
        in_channel,
        mlp_list,
        keep_gripper_in_fps=False,
    ):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.keep_gripper_in_fps = keep_gripper_in_fps
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(
            xyz, farthest_point_sample(xyz, S, self.keep_gripper_in_fps)
        )
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2
            )

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNet2(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2, self).__init__()
        # self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=3, mlp_list=[[16, 16, 32], [32, 32, 64]])
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=1024,
            radius_list=[0.05, 0.1],
            nsample_list=[16, 32],
            in_channel=0,
            mlp_list=[[16, 16, 32], [32, 32, 64]],
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=256,
            radius_list=[0.1, 0.2],
            nsample_list=[16, 32],
            in_channel=96,
            mlp_list=[[64, 64, 128], [64, 96, 128]],
        )
        self.sa3 = PointNetSetAbstractionMsg(
            64, [0.2, 0.4], [16, 32], 128 + 128, [[128, 196, 256], [128, 196, 256]]
        )
        self.sa4 = PointNetSetAbstractionMsg(
            16, [0.4, 0.8], [16, 32], 256 + 256, [[256, 256, 512], [256, 384, 512]]
        )
        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        # l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # (B, 3, 1024) (B, 96, 1024)
        l1_xyz, l1_points = self.sa1(l0_xyz, None)  # (B, 3, 1024) (B, 96, 1024)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B, 3, 256) (B, 256, 256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 3, 64) (B, 512, 64)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B, 3, 16) (B, 1024, 16)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B, 512, 64)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, 256, 256)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x  # x shape: B, N, num_classes


class PointNet2_super(nn.Module):
    def __init__(self, num_classes, input_channel=3, keep_gripper_in_fps=False):
        super(PointNet2_super, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=1024,
            radius_list=[0.025, 0.05],
            nsample_list=[16, 32],
            in_channel=input_channel - 3,
            mlp_list=[[16, 16, 32], [32, 32, 64]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.05, 0.1],
            nsample_list=[16, 32],
            in_channel=96,
            mlp_list=[[64, 64, 128], [64, 96, 128]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa3 = PointNetSetAbstractionMsg(
            256,
            [0.1, 0.2],
            [16, 32],
            128 + 128,
            [[128, 196, 256], [128, 196, 256]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa4 = PointNetSetAbstractionMsg(
            128,
            [0.2, 0.4],
            [16, 32],
            256 + 256,
            [[256, 256, 512], [256, 384, 512]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa5 = PointNetSetAbstractionMsg(
            64,
            [0.4, 0.8],
            [16, 32],
            512 + 512,
            [[512, 512, 512], [512, 512, 512]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa6 = PointNetSetAbstractionMsg(
            16,
            [0.8, 1.6],
            [16, 32],
            512 + 512,
            [[512, 512, 512], [512, 512, 512]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.fp6 = PointNetFeaturePropagation(512 + 512 + 512 + 512, [512, 512])
        self.fp5 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [512, 512])
        self.fp4 = PointNetFeaturePropagation(1024, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :])
        else:
            l1_xyz, l1_points = self.sa1(l0_xyz, None)  # (B, 3, 1024) (B, 96, 1024)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 3, 256) (B, 512, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B, 3, 128) (B, 1024, 16)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)  # (B, 3, 64) (B , 1024, 64)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points)  # (B, 3, 16) (B, 1024, 16)

        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points)  # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)  # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # (B, 128, num_point)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x  # x shape: B, N, num_classes


class PointNet2_superplus(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2_superplus, self).__init__()
        self.sa0 = PointNetSetAbstractionMsg(
            npoint=2048,
            radius_list=[0.0125, 0.025],
            nsample_list=[16, 32],
            in_channel=0,
            mlp_list=[[32, 32, 64], [64, 64, 128]],
        )
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=1024,
            radius_list=[0.025, 0.05],
            nsample_list=[16, 32],
            in_channel=64 + 128,
            mlp_list=[[64, 64, 128], [128, 196, 256]],
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.05, 0.1],
            nsample_list=[16, 32],
            in_channel=128 + 256,
            mlp_list=[[128, 196, 256], [128, 196, 256]],
        )
        self.sa3 = PointNetSetAbstractionMsg(
            256, [0.1, 0.2], [16, 32], 256 + 256, [[256, 384, 512], [256, 384, 512]]
        )
        self.sa4 = PointNetSetAbstractionMsg(
            128, [0.2, 0.4], [16, 32], 512 + 512, [[256, 384, 512], [256, 384, 512]]
        )
        self.sa5 = PointNetSetAbstractionMsg(
            64, [0.4, 0.8], [16, 32], 512 + 512, [[512, 512, 512], [512, 512, 512]]
        )
        self.sa6 = PointNetSetAbstractionMsg(
            16, [0.8, 1.6], [16, 32], 512 + 512, [[512, 512, 512], [512, 512, 512]]
        )
        self.fp6 = PointNetFeaturePropagation(512 + 512 + 512 + 512, [512, 512, 512])
        self.fp5 = PointNetFeaturePropagation(512 + 512 + 512, [512, 512, 512])
        self.fp4 = PointNetFeaturePropagation(512 + 512 + 512, [512, 384, 256])
        self.fp3 = PointNetFeaturePropagation(256 + 256 + 256, [256, 256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 256 + 128, [256, 128, 128])
        self.fp1 = PointNetFeaturePropagation(128 + 128 + 64, [128, 128, 128])
        self.fp0 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l01_xyz, l01_points = self.sa0(l0_xyz, None)  # (B, 3, 1024) (B, 96, 1024)
        l1_xyz, l1_points = self.sa1(l01_xyz, l01_points)  # (B, 3, 1024) (B, 96, 1024)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 3, 256) (B, 512, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B, 3, 128) (B, 1024, 16)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)  # (B, 3, 64) (B , 1024, 64)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points)  # (B, 3, 16) (B, 1024, 16)

        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points)  # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)  # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, 128, 1024)
        l01_points = self.fp1(
            l01_xyz, l1_xyz, l01_points, l1_points
        )  # (B, 128, num_point)
        l0_points = self.fp0(l0_xyz, l01_xyz, None, l01_points)  # (B, 128, num_point)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x  # x shape: B, N, num_classes


class ArticubotNetwork(nn.Module):
    """
    Modified version of PointNet2_super to work with this codebase
    """

    def __init__(self, model_cfg):
        super(ArticubotNetwork, self).__init__()

        num_classes = model_cfg.num_classes
        input_channel = model_cfg.in_channels
        keep_gripper_in_fps = model_cfg.keep_gripper_in_fps

        self.use_text_embedding = model_cfg.use_text_embedding
        self.use_dual_head = model_cfg.use_dual_head
        self.encoded_text_dim = 128  # Output dimension after encoding
        if self.use_text_embedding:
            self.text_encoder = nn.Linear(
                1152, self.encoded_text_dim
            )  # SIGLIP input dim
            self.film_predictor = nn.Sequential(
                nn.Linear(self.encoded_text_dim, 256),  # [B, 128] -> [B, 256]
                nn.ReLU(),
                nn.Linear(256, 1024 * 2),  # [B, 256] -> [B, 2048]
            )
            # Init as gamma=0 and beta=1
            self.film_predictor[-1].weight.data.zero_()
            self.film_predictor[-1].bias.data.copy_(
                torch.cat([torch.ones(1024), torch.zeros(1024)])
            )

        self.sa1 = PointNetSetAbstractionMsg(
            npoint=1024,
            radius_list=[0.025, 0.05],
            nsample_list=[16, 32],
            in_channel=input_channel - 3,
            mlp_list=[[16, 16, 32], [32, 32, 64]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.05, 0.1],
            nsample_list=[16, 32],
            in_channel=96,
            mlp_list=[[64, 64, 128], [64, 96, 128]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa3 = PointNetSetAbstractionMsg(
            256,
            [0.1, 0.2],
            [16, 32],
            128 + 128,
            [[128, 196, 256], [128, 196, 256]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa4 = PointNetSetAbstractionMsg(
            128,
            [0.2, 0.4],
            [16, 32],
            256 + 256,
            [[256, 256, 512], [256, 384, 512]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa5 = PointNetSetAbstractionMsg(
            64,
            [0.4, 0.8],
            [16, 32],
            512 + 512,
            [[512, 512, 512], [512, 512, 512]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa6 = PointNetSetAbstractionMsg(
            16,
            [0.8, 1.6],
            [16, 32],
            512 + 512,
            [[512, 512, 512], [512, 512, 512]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.fp6 = PointNetFeaturePropagation(512 + 512 + 512 + 512, [512, 512])
        self.fp5 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [512, 512])
        self.fp4 = PointNetFeaturePropagation(1024, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)

        # Dual head architecture
        if self.use_dual_head:
            # Human prediction head
            self.human_conv = nn.Conv1d(128, 128, 1)
            self.human_bn = nn.BatchNorm1d(128)
            self.human_head = nn.Conv1d(128, num_classes, 1)

            # Robot prediction head
            self.robot_conv = nn.Conv1d(128, 128, 1)
            self.robot_bn = nn.BatchNorm1d(128)
            self.robot_head = nn.Conv1d(128, num_classes, 1)
        else:
            # Single head
            self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, text_embedding=None, data_source=None):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :])
        else:
            l1_xyz, l1_points = self.sa1(l0_xyz, None)  # (B, 3, 1024) (B, 96, 1024)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 3, 256) (B, 512, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B, 3, 128) (B, 1024, 16)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)  # (B, 3, 64) (B , 1024, 64)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points)  # (B, 3, 16) (B, 1024, 16)

        # Apply FiLM conditioning at bottleneck
        if self.use_text_embedding:
            encoded_text = self.text_encoder(text_embedding)  # [B, 128]
            film_params = self.film_predictor(encoded_text)  # [B, 1024 * 2]
            gamma, beta = film_params.chunk(2, dim=1)  # [B, 1024] each
            gamma = gamma.unsqueeze(2)  # [B, 1024, 1] for broadcasting
            beta = beta.unsqueeze(2)  # [B, 1024, 1] for broadcasting
            l6_points = gamma * l6_points + beta  # FiLM modulation: [B, 1024, 16]

        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points)  # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)  # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # (B, 128, num_point)

        # Shared backbone features
        backbone_features = F.relu(self.bn1(self.conv1(l0_points)))  # (B, 128, N)

        if self.use_dual_head:
            assert data_source is not None
            # Dual head prediction - Compute both and mask out.
            human_features = F.relu(self.human_bn(self.human_conv(backbone_features)))
            robot_features = F.relu(self.robot_bn(self.robot_conv(backbone_features)))

            human_output = self.human_head(human_features)  # (B, num_classes, N)
            robot_output = self.robot_head(robot_features)  # (B, num_classes, N)

            # Create final output by selecting appropriate head for each batch item
            human_mask = torch.tensor(
                [ds == "human" for ds in data_source],
                device=human_output.device,
                dtype=torch.bool,
            )
            human_mask = human_mask.unsqueeze(1).unsqueeze(
                2
            )  # [B, 1, 1] for broadcasting
            x = torch.where(human_mask, human_output, robot_output)
        else:
            x = self.conv2(backbone_features)  # (B, num_classes, N)

        x = x.permute(0, 2, 1)  # (B, N, num_classes)
        return x


class ArticubotSmallNetwork(nn.Module):
    def __init__(self, model_cfg):
        super(ArticubotSmallNetwork, self).__init__()
        raise NotImplementedError("not up to date. refer articubotnetwork and update.")

        num_classes = model_cfg.num_classes
        input_channel = model_cfg.in_channels
        keep_gripper_in_fps = model_cfg.keep_gripper_in_fps

        # Reduced SA layers: Only sa1, sa2, sa3 with smaller MLPs
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,  # Reduced from 1024
            radius_list=[0.025, 0.05],
            nsample_list=[16, 32],
            in_channel=input_channel - 3,
            mlp_list=[[16, 16], [32, 32]],  # Smaller MLP
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=256,  # Reduced from 512
            radius_list=[0.05, 0.1],
            nsample_list=[16, 32],
            in_channel=48,  # Adjusted based on sa1 output
            mlp_list=[[32, 32, 64], [64, 64]],  # Smaller MLP
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=128,  # Reduced from 256
            radius_list=[0.1, 0.2],
            nsample_list=[16, 32],
            in_channel=128,  # Adjusted based on sa2 output
            mlp_list=[[64, 64, 128], [128, 128]],  # Smaller MLP
            keep_gripper_in_fps=keep_gripper_in_fps,
        )

        # Reduced FP layers: Only fp3, fp2, fp1
        self.fp3 = PointNetFeaturePropagation(
            128 + 128 + 128, [128, 128]
        )  # Adjusted input channels
        self.fp2 = PointNetFeaturePropagation(
            48 + 128, [128, 64]
        )  # Adjusted input channels
        self.fp1 = PointNetFeaturePropagation(64, [64, 64, 64])

        self.conv1 = nn.Conv1d(64, 64, 1)  # Reduced channels
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :])
        else:
            l1_xyz, l1_points = self.sa1(l0_xyz, None)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = nn.functional.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)  # [B, num_classes, N]
        return x.permute(0, 2, 1)  # [B, N, num_classes]


class GoalRegressionModule(pl.LightningModule):
    """
    A goal generation module that handles model training, inference, evaluation and visualization.
    Based on CrossDisplacementModule but reworked to use the Articubot high-level model.
    """

    def __init__(self, network, cfg) -> None:
        super().__init__()
        self.network = network
        self.model_cfg = cfg.model
        self.prediction_type = self.model_cfg.type  # flow or point
        self.mode = cfg.mode  # train or eval
        self.val_outputs: defaultdict[str, List[Dict]] = defaultdict(list)
        self.train_outputs: List[Dict] = []
        self.predict_outputs: defaultdict[str, List[Dict]] = defaultdict(list)
        self.predict_weighted_displacements: defaultdict[str, List[Dict]] = defaultdict(
            list
        )
        self.fixed_variance = cfg.model.fixed_variance
        self.uniform_weights_coeff = cfg.model.uniform_weights_coeff
        self.is_gmm = cfg.model.is_gmm

        if self.prediction_type != "cross_displacement":
            raise ValueError(f"Invalid prediction type: {self.prediction_type}")
        self.label_key = "cross_displacement"

        # mode-specific processing
        if self.mode == "train":
            self.run_cfg = cfg.training
            # training-specific params
            self.lr = self.run_cfg.lr
            self.weight_decay = self.run_cfg.weight_decay
            self.num_training_steps = self.run_cfg.num_training_steps
            self.lr_warmup_steps = self.run_cfg.lr_warmup_steps
            self.additional_train_logging_period = (
                self.run_cfg.additional_train_logging_period
            )
        elif self.mode == "eval":
            self.run_cfg = cfg.inference
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # data params
        self.batch_size = self.run_cfg.batch_size
        self.val_batch_size = self.run_cfg.val_batch_size
        self.max_depth = cfg.dataset.max_depth

        # TODO: Make config param
        self.weight_loss_weight = 10  # weight of the weighted displacement loss term

    def configure_optimizers(self):
        assert self.mode == "train", "Can only configure optimizers in training mode."
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",  # Step after every batch
            },
        }

    def extract_gt_4_points(self, batch):
        cross_displacement = batch[self.label_key].points_padded()
        initial_gripper = batch["action_pcd"].points_padded()
        ground_truth_gripper = initial_gripper + cross_displacement
        batch_indices = torch.arange(
            ground_truth_gripper.shape[0], device=ground_truth_gripper.device
        ).unsqueeze(1)

        # Select specific idxs to compute the loss over
        # TODO: Find a cleaner way to pass the idxs
        gt_primary_points = ground_truth_gripper[batch_indices, batch["gripper_idx"], :]
        # Assumes 0/1 are tips to be averaged
        gt_extra_point = (gt_primary_points[:, 0, :] + gt_primary_points[:, 1, :]) / 2
        gt = torch.cat([gt_primary_points, gt_extra_point[:, None, :]], dim=1)

        init_primary_points = initial_gripper[batch_indices, batch["gripper_idx"], :]
        init_extra_point = (
            init_primary_points[:, 0, :] + init_primary_points[:, 1, :]
        ) / 2
        init = torch.cat([init_primary_points, init_extra_point[:, None, :]], dim=1)
        return init, gt

    def project_3d_to_2d(self, points_3d, intrinsics, img_shape=(224, 224)):
        """
        Project 3D points to 2D pixel coordinates.

        Args:
            points_3d: (B, N, 3) or (B, N, M, 3) 3D points
            intrinsics: (B, 3, 3) camera intrinsics
            img_shape: (H, W) image shape for clamping

        Returns:
            pixel_coords: (B, N, 2) or (B, N, M, 2) pixel coordinates [x, y]
        """
        H, W = img_shape
        original_shape = points_3d.shape

        # Reshape to (B, -1, 3) for batch processing
        if len(original_shape) == 4:
            B, N, M, _ = original_shape
            points_3d = points_3d.reshape(B, N * M, 3)

        fx = intrinsics[:, 0, 0].unsqueeze(1)  # (B, 1)
        fy = intrinsics[:, 1, 1].unsqueeze(1)
        cx = intrinsics[:, 0, 2].unsqueeze(1)
        cy = intrinsics[:, 1, 2].unsqueeze(1)

        # Project: [x, y, z] -> [u, v]
        # Add epsilon to avoid division by zero
        z = points_3d[:, :, 2].clamp(min=1e-6)
        u = (points_3d[:, :, 0] * fx / z + cx).clamp(0, W - 1)  # (B, N*M)
        v = (points_3d[:, :, 1] * fy / z + cy).clamp(0, H - 1)  # (B, N*M)

        pixel_coords = torch.stack([u, v], dim=2)  # (B, N*M, 2)

        # Reshape back to original shape
        if len(original_shape) == 4:
            pixel_coords = pixel_coords.reshape(B, N, M, 2)

        return pixel_coords

    def get_weighted_displacement(self, scene_pcd, outputs, padding_mask):
        batch_size, num_points, _ = scene_pcd.shape
        scene_pcd = scene_pcd[:, :, None, :3]

        weights = outputs[:, :, -1]  # B, N
        weights = weights.masked_fill(
            ~padding_mask, float("-inf")
        )  # Set invalid to -inf
        # softmax the weights
        weights = torch.nn.functional.softmax(weights, dim=1)

        outputs = outputs[:, :, :-1]  # B, N, 12
        # sum the displacement of the predicted gripper point cloud according to the weights
        pred_points = weights[:, :, None, None] * (
            scene_pcd + outputs.reshape(batch_size, num_points, 4, 3)
        )
        pred_points = pred_points.sum(dim=1)
        return pred_points

    def prepare_scene_pcd(
        self,
        gripper_pcd,
        scene_pcd,
        scene_feat_pcd,
        scene_padding_mask,
        add_action_pcd_masked,
        use_rgb,
    ):
        batch_size, pcd_size, feat_dim = scene_feat_pcd.shape
        if add_action_pcd_masked:
            # Concat action pcd and scene pcd and add a mask for the same
            gripper_pcd = torch.cat(
                [
                    gripper_pcd,
                    torch.ones(
                        batch_size, gripper_pcd.shape[1], 1, device=scene_pcd.device
                    ),
                ],
                dim=2,
            )  # [B, K, 4]
            scene_pcd = torch.cat(
                [
                    scene_pcd,
                    torch.zeros(
                        batch_size, scene_pcd.shape[1], 1, device=scene_pcd.device
                    ),
                ],
                dim=2,
            )  # [B, N, 4]
            scene_pcd = torch.cat([gripper_pcd, scene_pcd], dim=1)
            pcd_size += gripper_pcd.shape[1]
            scene_padding_mask = torch.cat(
                [
                    torch.ones(
                        batch_size,
                        gripper_pcd.shape[1],
                        device=scene_pcd.device,
                        dtype=bool,
                    ),
                    scene_padding_mask,
                ],
                dim=1,
            )
            if use_rgb:
                gripper_feat = torch.zeros(
                    batch_size, gripper_pcd.shape[1], feat_dim, device=scene_pcd.device
                )
                scene_feat_pcd = torch.cat([gripper_feat, scene_feat_pcd], dim=1)

        if use_rgb:
            scene_pcd = torch.cat([scene_pcd, scene_feat_pcd], dim=2)
        return scene_pcd, pcd_size, scene_padding_mask

    def nll_loss(
        self,
        pred_displacement,
        gt_displacement,
        weights,
        scene_padding_mask,
        variance,
        use_weights=True,
    ):
        batch_size, pcd_size = pred_displacement.shape[:2]
        weights = weights.masked_fill(
            ~scene_padding_mask, float("-inf")
        )  # Set invalid to -inf
        if use_weights is False:
            # We overwrite with uniform weights
            weights = weights.masked_fill(scene_padding_mask, 1)

        diff = (pred_displacement - gt_displacement).reshape(
            batch_size, pcd_size, -1
        )  # Shape: (B, N, 12)
        exponent = -0.5 * torch.sum(
            (diff**2) / variance, dim=2
        )  # Shape: (B, N), sum over the guassian dimension
        log_gaussians = exponent

        # Compute log mixing coefficients
        log_mixing_coeffs = torch.log_softmax(
            weights, dim=1
        )  # softmax the weight along the per-point dimension, shape B, N
        log_mixing_coeffs = torch.clamp(
            log_mixing_coeffs, min=-10
        )  # Prevent extreme values

        masked_sum = log_gaussians + log_mixing_coeffs  # [B, N]
        masked_sum = masked_sum.masked_fill(
            ~scene_padding_mask, -1e9
        )  # In-place masking

        max_log = torch.max(
            masked_sum, dim=1, keepdim=True
        ).values  # get the per-batch max log along all the points, B, 1
        log_probs = max_log.squeeze(1) + torch.logsumexp(
            masked_sum - max_log, dim=1
        )  # B,

        nll_loss = -torch.mean(log_probs)  # mean of the negative log likelihood
        return nll_loss

    def forward(self, batch):
        initial_gripper = batch["action_pcd"].points_padded()

        scene_pcd = batch["anchor_pcd"].points_padded()
        scene_feat_pcd = batch["anchor_pcd"].features_padded()
        batch_size, max_points, *_ = scene_pcd.shape
        num_points = batch["anchor_pcd"].num_points_per_cloud()
        scene_padding_mask = (
            torch.arange(max_points, device=num_points.device)[None, :]
            < num_points[:, None]
        )

        text_embedding = batch["text_embed"]
        batch_size, pcd_size, _ = scene_pcd.shape

        scene_pcd, pcd_size, scene_padding_mask = self.prepare_scene_pcd(
            initial_gripper,
            scene_pcd,
            scene_feat_pcd,
            scene_padding_mask,
            self.model_cfg.add_action_pcd_masked,
            self.model_cfg.use_rgb,
        )
        data_sources = batch["data_source"]

        outputs = self.network(
            scene_pcd.permute(0, 2, 1),
            text_embedding=text_embedding,
            data_source=data_sources,
        )

        init, gt = self.extract_gt_4_points(batch)
        pred_displacement = outputs[:, :, :-1].reshape(batch_size, pcd_size, 4, 3)
        gt_displacement = gt[:, None, :, :] - scene_pcd[:, :, None, :3]
        weights = outputs[:, :, -1]  # B, N

        if self.is_gmm:
            loss = 0
            for var in self.fixed_variance:
                loss += self.nll_loss(
                    pred_displacement,
                    gt_displacement,
                    weights,
                    scene_padding_mask,
                    var,
                    use_weights=True,
                )
                loss += self.uniform_weights_coeff * self.nll_loss(
                    pred_displacement,
                    gt_displacement,
                    weights,
                    scene_padding_mask,
                    var,
                    use_weights=False,
                )
        else:
            pred_points = self.get_weighted_displacement(
                scene_pcd, outputs, scene_padding_mask
            )
            per_point_displacement_loss = F.mse_loss(
                pred_displacement[scene_padding_mask],
                gt_displacement[scene_padding_mask],
            )

            weighted_avg_loss = F.mse_loss(pred_points, gt)
            loss = (
                per_point_displacement_loss
                + self.weight_loss_weight * weighted_avg_loss
            )

        return None, loss

    @torch.no_grad()
    def predict(self, batch, progress=False):
        """
        Compute prediction for a given batch.
        NOTE: To maintain consistency with the codebase,
        this returns the displacement to the goal position,
        not the actual goal position itself.

        Args:
            batch: the input batch
            progress: whether to show progress bar
        """
        initial_gripper = batch["action_pcd"].points_padded()
        scene_pcd = batch["anchor_pcd"].points_padded()
        scene_feat_pcd = batch["anchor_pcd"].features_padded()
        batch_size, max_points, *_ = scene_pcd.shape
        num_points = batch["anchor_pcd"].num_points_per_cloud()
        scene_padding_mask = (
            torch.arange(max_points, device=num_points.device)[None, :]
            < num_points[:, None]
        )

        text_embedding = batch["text_embed"]
        batch_size, pcd_size, _ = scene_pcd.shape  # Matches forward

        scene_pcd, pcd_size, scene_padding_mask = self.prepare_scene_pcd(
            initial_gripper,
            scene_pcd,
            scene_feat_pcd,
            scene_padding_mask,
            self.model_cfg.add_action_pcd_masked,
            self.model_cfg.use_rgb,
        )

        data_sources = batch["data_source"]

        outputs = self.network(
            scene_pcd.permute(0, 2, 1),
            text_embedding=text_embedding,
            data_source=data_sources,
        )
        init, gt = self.extract_gt_4_points(batch)

        if self.is_gmm:
            pred = self.sample_from_gmm(scene_pcd, outputs, scene_padding_mask)
        else:
            pred = self.get_weighted_displacement(
                scene_pcd, outputs, scene_padding_mask
            )
        pred_displacement = pred - init
        return {self.prediction_type: {"pred": pred_displacement}}, outputs

    def sample_from_gmm(self, scene_pcd, outputs, padding_mask):
        batch_size, num_points, _ = scene_pcd.shape
        weights = outputs[:, :, -1]  # B, N
        weights = weights.masked_fill(
            ~padding_mask, float("-inf")
        )  # Set invalid to -inf
        weights = torch.nn.functional.softmax(weights, dim=1)

        # Sample point indices based on weights for each batch element
        sampled_indices = torch.multinomial(weights, num_samples=1)  # B, 1
        batch_indices = torch.arange(batch_size, device=outputs.device).unsqueeze(
            1
        )  # B, 1

        # Extract displacement predictions
        displacements = outputs[:, :, :-1].reshape(
            batch_size, num_points, 4, 3
        )  # B, N, 4, 3

        # Prepare scene points
        scene_points = scene_pcd[:, :, None, :3]  # B, N, 1, 3

        # Get the Gaussian means: scene_point + displacement
        # Broadcasting will expand scene_points from [B, N, 1, 3] to [B, N, 4, 3]
        means = scene_points + displacements  # B, N, 4, 3

        # Get sampled means
        sampled_means = means[batch_indices, sampled_indices].squeeze(1)  # B, 4, 3

        # NOTE: We can sample from these gaussians as well, but just using the mean for now.
        # noise = torch.randn_like(sampled_means) * (self.fixed_variance**0.5)
        # pred_points = sampled_means + noise

        pred_points = sampled_means

        return pred_points

    def log_viz_to_wandb(self, batch, pred_dict, weighted_displacement, tag):
        """
        Log visualizations to wandb.

        Args:
            batch: the input batch
            pred_dict: the prediction dictionary
            weighted_displacement: the output of the articubot model
            tag: the tag to use for logging
        """
        batch_size = batch[self.label_key].points_padded().shape[0]
        # pick a random sample in the batch to visualize
        viz_idx = np.random.randint(0, batch_size)
        RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)
        max_depth = self.max_depth

        all_pred = pred_dict[self.prediction_type]["all_pred"][viz_idx].cpu().numpy()
        N = all_pred.shape[0]
        end2start = np.linalg.inv(batch["start2end"][viz_idx].cpu().numpy())

        if N == 1:
            BLUES = [BLUE]
        else:
            # Multiple shades of blue for different samples
            BLUES = [
                (int(200 * (1 - i / (N - 1))), int(220 * (1 - i / (N - 1))), 255)
                for i in range(N)
            ]

        goal_text = batch["caption"][viz_idx]
        vid_name = batch["vid_name"][viz_idx]
        rmse = pred_dict["rmse"][viz_idx]
        anchor_pcd = batch["anchor_pcd"].points_padded()[viz_idx].cpu().numpy()
        weighted_displacement = weighted_displacement[viz_idx].cpu().numpy()

        pcd, gt = self.extract_gt_4_points(batch)
        pcd, gt = pcd.cpu().numpy()[viz_idx], gt.cpu().numpy()[viz_idx]
        all_pred_pcd = pcd + all_pred
        gt_pcd = gt
        padding_mask = torch.ones(gt.shape[0]).bool().numpy()

        # Move center back from action_pcd to the camera frame
        # and invert augmentation transforms before viz
        pcd_mean = batch["pcd_mean"][viz_idx].cpu().numpy()
        pcd_std = batch["pcd_std"][viz_idx].cpu().numpy()
        R = batch["augment_R"][viz_idx].cpu().numpy()
        t = batch["augment_t"][viz_idx].cpu().numpy()
        scene_centroid = batch["augment_C"][viz_idx].cpu().numpy()

        pcd = invert_augmentation_and_normalization(
            pcd, pcd_mean, pcd_std, R, t, scene_centroid
        )
        anchor_pcd = invert_augmentation_and_normalization(
            anchor_pcd, pcd_mean, pcd_std, R, t, scene_centroid
        )
        all_pred_pcd = invert_augmentation_and_normalization(
            all_pred_pcd, pcd_mean, pcd_std, R, t, scene_centroid
        )
        gt_pcd = invert_augmentation_and_normalization(
            gt_pcd, pcd_mean, pcd_std, R, t, scene_centroid
        )

        # Transform to end frame
        pcd_endframe = np.hstack((pcd, np.ones((pcd.shape[0], 1))))
        pcd_endframe = (end2start @ pcd_endframe.T).T[:, :3]
        all_pred_pcd_tmp = []
        for i in range(N):
            tmp_pcd = np.hstack((all_pred_pcd[i], np.ones((all_pred_pcd.shape[1], 1))))
            tmp_pcd = (end2start @ tmp_pcd.T).T[:, :3]
            all_pred_pcd_tmp.append(tmp_pcd)
        all_pred_pcd = np.stack(all_pred_pcd_tmp)
        gt_pcd = np.hstack((gt_pcd, np.ones((gt_pcd.shape[0], 1))))
        gt_pcd = (end2start @ gt_pcd.T).T[:, :3]

        # Transform from world frame to primary camera frame for projection
        # Primary camera extrinsics: T_world_from_cam, we need T_cam_from_world
        primary_extrinsics = batch["extrinsics"][viz_idx].cpu().numpy()  # (4, 4)
        T_cam_from_world = np.linalg.inv(primary_extrinsics)

        # Save world-frame versions for action_anchor_pcd visualization
        pcd_world = pcd.copy()
        anchor_pcd_world = anchor_pcd.copy()

        # Transform points to primary camera frame
        # Transform initial pcd (for init_rgb_proj)
        pcd_cam = np.hstack((pcd, np.ones((pcd.shape[0], 1))))
        pcd_cam = (T_cam_from_world @ pcd_cam.T).T[:, :3]

        pcd_endframe = np.hstack((pcd_endframe, np.ones((pcd_endframe.shape[0], 1))))
        pcd_endframe = (T_cam_from_world @ pcd_endframe.T).T[:, :3]

        all_pred_pcd_tmp = []
        for i in range(N):
            tmp_pcd = np.hstack((all_pred_pcd[i], np.ones((all_pred_pcd.shape[1], 1))))
            tmp_pcd = (T_cam_from_world @ tmp_pcd.T).T[:, :3]
            all_pred_pcd_tmp.append(tmp_pcd)
        all_pred_pcd = np.stack(all_pred_pcd_tmp)

        gt_pcd = np.hstack((gt_pcd, np.ones((gt_pcd.shape[0], 1))))
        gt_pcd = (T_cam_from_world @ gt_pcd.T).T[:, :3]

        # Also transform anchor_pcd to camera frame
        anchor_pcd = np.hstack((anchor_pcd, np.ones((anchor_pcd.shape[0], 1))))
        anchor_pcd = (T_cam_from_world @ anchor_pcd.T).T[:, :3]

        K = batch["intrinsics"][viz_idx].cpu().numpy()

        rgb_init, rgb_end = (
            batch["rgbs"][viz_idx, 0].cpu().numpy(),
            batch["rgbs"][viz_idx, 1].cpu().numpy(),
        )
        depth_init, depth_end = (
            batch["depths"][viz_idx, 0].cpu().numpy(),
            batch["depths"][viz_idx, 1].cpu().numpy(),
        )

        # Project tracks to image and save
        init_rgb_proj = project_pcd_on_image(pcd_cam, padding_mask, rgb_init, K, GREEN)
        end_rgb_proj = project_pcd_on_image(gt_pcd, padding_mask, rgb_end, K, RED)
        pred_rgb_proj = project_pcd_on_image(
            all_pred_pcd[-1], padding_mask, rgb_end, K, BLUE
        )
        rgb_proj_viz = cv2.hconcat([init_rgb_proj, end_rgb_proj, pred_rgb_proj])

        wandb_proj_img = wandb.Image(
            rgb_proj_viz,
            caption=f"Left: Initial Frame (GT Track)\n; Middle: Final Frame (GT Track)\n\
            ; Right: Final Frame (Pred Track)\n; Goal Description : {goal_text};\n\
            rmse={rmse};\nvideo path = {vid_name}; ",
        )
        ###

        # Visualize point cloud
        viz_pcd, _ = get_img_and_track_pcd(
            rgb_end,
            depth_end,
            K,
            padding_mask,
            pcd_endframe,
            gt_pcd,
            all_pred_pcd,
            GREEN,
            RED,
            BLUES,
            max_depth,
            anchor_pcd.shape[0],
        )
        ###

        # Visualize action/anchor point cloud
        action_anchor_pcd = get_action_anchor_pcd(
            pcd_world,
            anchor_pcd_world,
            GREEN,
            RED,
        )
        ###

        # _ = save_weighted_displacement_pcd_viz(anchor_pcd, weighted_displacement)

        viz_dict = {
            f"{tag}/track_projected_to_rgb": wandb_proj_img,
            f"{tag}/image_and_tracks_pcd": wandb.Object3D(viz_pcd),
            f"{tag}/action_anchor_pcd": wandb.Object3D(action_anchor_pcd),
            "trainer/global_step": self.global_step,
        }

        wandb.log(viz_dict)

    def training_step(self, batch, batch_idx):
        """
        Training step for the module. Logs training metrics and visualizations to wandb.
        """
        self.train()
        batch_size = batch[self.label_key].points_padded().shape[0]

        _, loss = self(batch)
        #########################################################
        # logging training metrics
        #########################################################
        train_metrics = {"loss": loss}

        # determine if additional logging should be done
        do_additional_logging = (
            self.global_step % self.additional_train_logging_period == 0
            and self.global_step != 0
        )

        # additional logging
        if do_additional_logging:
            n_samples_wta = self.run_cfg.n_samples_wta
            self.eval()
            with torch.no_grad():
                all_pred_dict = []
                if self.is_gmm:
                    for i in range(n_samples_wta):
                        all_pred_dict.append(self.predict(batch))
                else:
                    all_pred_dict = [self.predict(batch)]
                # Use one sample for computing other metrics
                pred_dict, weighted_displacement = all_pred_dict[0]
                # Store all sample preds for viz
                pred_dict[self.prediction_type]["all_pred"] = [
                    i[0][self.prediction_type]["pred"] for i in all_pred_dict
                ]
                pred_dict[self.prediction_type]["all_pred"] = torch.stack(
                    pred_dict[self.prediction_type]["all_pred"]
                ).permute(1, 0, 2, 3)
            self.train()  # Switch back to training mode

            init, gt = self.extract_gt_4_points(batch)
            gt_displacement = gt - init

            padding_mask = torch.ones(gt.shape[0], gt.shape[1]).bool()
            pcd_std = batch["pcd_std"]
            ground_truth = batch[self.label_key].to(self.device)
            pred_dict = calc_pcd_metrics(
                pred_dict,
                init,
                pred_dict[self.prediction_type]["all_pred"],
                gt_displacement,
                pcd_std,
                padding_mask,
            )

            # Calculate pixel metrics
            intrinsics = batch["intrinsics"]
            H, W = batch["rgbs"].shape[2:4]

            # Project GT to 2D (take first point only)
            gt_2d = (
                self.project_3d_to_2d(gt[:, :1, :], intrinsics, (H, W))
                .squeeze(1)
                .long()
            )  # (B, 2)

            # Project all predictions to 2D (take first point only)
            all_pred_3d = (
                init[:, None, :, :] + pred_dict[self.prediction_type]["all_pred"]
            )  # (B, N, 4, 3)
            all_pred_2d = (
                self.project_3d_to_2d(all_pred_3d[:, :, :1, :], intrinsics, (H, W))
                .squeeze(2)
                .long()
            )  # (B, N, 2)

            pred_dict = calc_pix_metrics(pred_dict, gt_2d, all_pred_2d, (H, W))
            train_metrics.update(pred_dict)

            if self.trainer.is_global_zero:
                ####################################################
                # logging visualizations
                ####################################################
                self.log_viz_to_wandb(batch, pred_dict, weighted_displacement, "train")

        self.train_outputs.append(train_metrics)
        return loss

    def on_train_epoch_end(self):
        if len(self.train_outputs) == 0:
            return

        log_dictionary = {}
        loss = torch.stack([x["loss"] for x in self.train_outputs]).mean()
        log_dictionary["train/loss"] = loss

        def mean_metric(metric_name):
            return torch.stack(
                [x[metric_name].mean() for x in self.train_outputs if metric_name in x]
            ).mean()

        if any("rmse" in x for x in self.train_outputs):
            log_dictionary["train/rmse"] = mean_metric("rmse")
            log_dictionary["train/wta_rmse"] = mean_metric("wta_rmse")
            log_dictionary["train/chamfer_dist"] = mean_metric("chamfer_dist")
            log_dictionary["train/wta_chamfer_dist"] = mean_metric("wta_chamfer_dist")
            log_dictionary["train/sample_std"] = mean_metric("sample_std")
            log_dictionary["train/pix_dist"] = mean_metric("pix_dist")
            log_dictionary["train/wta_pix_dist"] = mean_metric("wta_pix_dist")
            log_dictionary["train/normalized_pix_dist"] = mean_metric(
                "normalized_pix_dist"
            )
            log_dictionary["train/wta_normalized_pix_dist"] = mean_metric(
                "wta_normalized_pix_dist"
            )

        ####################################################
        # logging training metrics
        ####################################################
        self.log_dict(
            log_dictionary,
            add_dataloader_idx=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.train_outputs.clear()

    def on_validation_epoch_start(self):
        # Choose a random batch index for each validation epoch
        self.random_val_viz_idx = {
            k: random.randint(0, len(v) - 1)
            for k, v in self.trainer.val_dataloaders.items()
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation step for the module. Logs validation metrics and visualizations to wandb.
        """
        val_tag = self.trainer.datamodule.val_tags[dataloader_idx]
        n_samples_wta = self.run_cfg.n_samples_wta
        self.eval()
        with torch.no_grad():
            all_pred_dict = []
            if self.is_gmm:
                for i in range(n_samples_wta):
                    all_pred_dict.append(self.predict(batch))
            else:
                all_pred_dict = [self.predict(batch)]
            pred_dict, weighted_displacement = all_pred_dict[0]

            # Store all sample preds for viz
            pred_dict[self.prediction_type]["all_pred"] = [
                i[0][self.prediction_type]["pred"] for i in all_pred_dict
            ]
            pred_dict[self.prediction_type]["all_pred"] = torch.stack(
                pred_dict[self.prediction_type]["all_pred"]
            ).permute(1, 0, 2, 3)

        init, gt = self.extract_gt_4_points(batch)
        gt_displacement = gt - init

        padding_mask = torch.ones(gt.shape[0], gt.shape[1]).bool()
        pcd_std = batch["pcd_std"]
        ground_truth = batch[self.label_key].to(self.device)
        pred_dict = calc_pcd_metrics(
            pred_dict,
            init,
            pred_dict[self.prediction_type]["all_pred"],
            gt_displacement,
            pcd_std,
            padding_mask,
        )

        # Calculate pixel metrics
        intrinsics = batch["intrinsics"]
        H, W = batch["rgbs"].shape[2:4]

        # Project GT to 2D (take first point only)
        gt_2d = (
            self.project_3d_to_2d(gt[:, :1, :], intrinsics, (H, W)).squeeze(1).long()
        )  # (B, 2)

        # Project all predictions to 2D (take first point only)
        all_pred_3d = (
            init[:, None, :, :] + pred_dict[self.prediction_type]["all_pred"]
        )  # (B, N, 4, 3)
        all_pred_2d = (
            self.project_3d_to_2d(all_pred_3d[:, :, :1, :], intrinsics, (H, W))
            .squeeze(2)
            .long()
        )  # (B, N, 2)

        pred_dict = calc_pix_metrics(pred_dict, gt_2d, all_pred_2d, (H, W))
        self.val_outputs[val_tag].append(pred_dict)

        ####################################################
        # logging visualizations
        ####################################################
        if (
            batch_idx == self.random_val_viz_idx[val_tag]
            and self.trainer.is_global_zero
        ):
            self.log_viz_to_wandb(
                batch, pred_dict, weighted_displacement, f"val_{val_tag}"
            )
        return pred_dict

    def on_validation_epoch_end(self):
        log_dict = {}
        all_metrics = {
            "rmse": [],
            "wta_rmse": [],
            "chamfer_dist": [],
            "wta_chamfer_dist": [],
            "sample_std": [],
            "pix_dist": [],
            "wta_pix_dist": [],
            "normalized_pix_dist": [],
            "wta_normalized_pix_dist": [],
        }

        for val_tag in self.trainer.datamodule.val_tags:
            val_outputs = self.val_outputs[val_tag]
            tag_metrics = {}

            if len(val_outputs) == 0:
                continue

            for metric in all_metrics.keys():
                values = torch.stack([x[metric].mean() for x in val_outputs]).mean()
                tag_metrics[metric] = values
                all_metrics[metric].append(values)

            # Per dataset metrics
            for metric, value in tag_metrics.items():
                log_dict[f"val_{val_tag}/{metric}"] = value

        # Avg over all datasets
        for metric, values in all_metrics.items():
            log_dict[f"val/{metric}"] = torch.stack(values).mean()

        # Minimize the linear combination of RMSE (reconstruction error) and -std (i.e. maximize diversity)
        # TODO: Find a better metric, and dynamically configure this....
        alpha = 0.95
        log_dict["val/rmse_and_std_combi"] = alpha * log_dict["val/rmse"] + (
            1 - alpha
        ) * (-log_dict["val/sample_std"])

        self.log_dict(
            log_dict,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        self.val_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step for model evaluation.
        """

        assert "actual_caption" in batch, """We expect the actual caption of
        the subgoal to be present in the batch to calculate errors for each
        subgoal independently. The actual caption differs from the "caption"
        key only when use_full_text is True. """

        eval_tag = self.trainer.datamodule.eval_tags[dataloader_idx]
        n_samples_wta = self.trainer.datamodule.n_samples_wta

        all_pred_dict = []
        if self.is_gmm:
            for i in range(n_samples_wta):
                all_pred_dict.append(self.predict(batch))
        else:
            all_pred_dict = [self.predict(batch)]

        pred_dict, weighted_displacement = all_pred_dict[0]
        # Store all sample preds for viz
        pred_dict[self.prediction_type]["all_pred"] = [
            i[0][self.prediction_type]["pred"] for i in all_pred_dict
        ]
        pred_dict[self.prediction_type]["all_pred"] = torch.stack(
            pred_dict[self.prediction_type]["all_pred"]
        ).permute(1, 0, 2, 3)

        init, gt = self.extract_gt_4_points(batch)
        gt_displacement = gt - init

        padding_mask = torch.ones(gt.shape[0], gt.shape[1]).bool()
        pcd_std = batch["pcd_std"]
        ground_truth = batch[self.label_key].to(self.device)
        pred_dict = calc_pcd_metrics(
            pred_dict,
            init,
            pred_dict[self.prediction_type]["all_pred"],
            gt_displacement,
            pcd_std,
            padding_mask,
        )

        # Calculate pixel metrics
        intrinsics = batch["intrinsics"]
        H, W = batch["rgbs"].shape[2:4]

        # Project GT to 2D (take first point only)
        gt_2d = (
            self.project_3d_to_2d(gt[:, :1, :], intrinsics, (H, W)).squeeze(1).long()
        )  # (B, 2)

        # Project all predictions to 2D (take first point only)
        all_pred_3d = (
            init[:, None, :, :] + pred_dict[self.prediction_type]["all_pred"]
        )  # (B, N, 4, 3)
        all_pred_2d = (
            self.project_3d_to_2d(all_pred_3d[:, :, :1, :], intrinsics, (H, W))
            .squeeze(2)
            .long()
        )  # (B, N, 2)

        pred_dict = calc_pix_metrics(pred_dict, gt_2d, all_pred_2d, (H, W))
        self.predict_outputs[eval_tag].append(pred_dict)
        self.predict_weighted_displacements[eval_tag].append(
            weighted_displacement.cpu()
        )

        # Get pred_coord for visualization (first sample, first 3 points)
        pred_3d = (
            init + pred_dict[self.prediction_type]["pred"]
        )  # (B, 4, 3) absolute positions
        pred_3d_first3 = pred_3d[:, :3, :]  # (B, 3, 3) first 3 points
        pred_coord_viz = self.project_3d_to_2d(
            pred_3d_first3, intrinsics, (H, W)
        ).long()  # (B, 3, 2)

        return {
            "pred_coord": pred_coord_viz,
            "rmse": pred_dict["rmse"],
            "chamfer_dist": pred_dict["chamfer_dist"],
            "wta_rmse": pred_dict["wta_rmse"],
            "wta_chamfer_dist": pred_dict["wta_chamfer_dist"],
            "pix_dist": pred_dict["pix_dist"],
            "wta_pix_dist": pred_dict["wta_pix_dist"],
            "vid_name": batch["vid_name"],
            "caption": batch["caption"],
            "actual_caption": batch["actual_caption"],
        }

    def on_predict_epoch_end(self):
        """
        Visualize random 5 batches in the test sets.
        """
        save_wta_to_disk = self.trainer.datamodule.save_wta_to_disk
        for dataloader_idx, eval_tag in enumerate(self.trainer.datamodule.eval_tags):
            if "test" not in eval_tag:
                continue

            pred_outputs = self.predict_outputs[eval_tag]
            rmse = torch.cat([x["rmse"] for x in pred_outputs])
            chamfer_dist = torch.cat([x["chamfer_dist"] for x in pred_outputs])
            cross_displacement, all_cross_displacement = [], []
            weighted_displacements = []
            for i, pred in enumerate(pred_outputs):
                cross_displacement.extend(pred["cross_displacement"]["pred"])
                all_cross_displacement.extend(pred["cross_displacement"]["all_pred"])
                weighted_displacements.append(
                    self.predict_weighted_displacements[eval_tag][i]
                )

            dataloader = self.trainer.predict_dataloaders[dataloader_idx]
            total_batches = len(dataloader)
            random_indices = random.sample(range(total_batches), min(5, total_batches))

            for i, batch in enumerate(dataloader):
                if i not in random_indices:
                    continue

                batch_len = len(batch["caption"])
                weighted_displacement_batch = weighted_displacements[i]
                for idx in range(batch_len):
                    pred_dict = self.compose_pred_dict_for_viz(
                        rmse,
                        chamfer_dist,
                        cross_displacement,
                        all_cross_displacement,
                        idx,
                    )
                    viz_batch = self.compose_batch_for_viz(batch, idx)
                    self.log_viz_to_wandb(
                        viz_batch,
                        pred_dict,
                        weighted_displacement_batch[idx][None],
                        eval_tag,
                    )
        self.predict_outputs.clear()

    def compose_pred_dict_for_viz(
        self, rmse, chamfer_dist, cross_displacement, all_cross_displacement, idx
    ):
        return {
            "rmse": [rmse[idx]],
            "chamfer_dist": [chamfer_dist[idx]],
            "cross_displacement": {
                "pred": cross_displacement[idx][None],
                "all_pred": all_cross_displacement[idx][None],
            },
        }

    def compose_batch_for_viz(self, batch, idx):
        viz_batch = {}
        for key in batch.keys():
            if type(batch[key]) == Pointclouds:
                pcd = batch[key].points_padded()[idx]
                viz_batch[key] = Pointclouds(points=pcd[None])
            elif key in ["rgbs", "depths", "gripper_idx"]:
                viz_batch[key] = batch[key][idx][None]
            else:
                viz_batch[key] = [batch[key][idx]]
        return viz_batch


if __name__ == "__main__":
    model = PointNet2(num_classes=10).cuda()
    model.eval()
    # torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    # torch.backends.cudnn.deterministic = True
    inpput = torch.rand(1, 3, 2000).cuda()
    out = model(inpput)
    max_diff = -1
    for _ in range(1):
        inpput_translated = inpput + 50
        out_translated = model(inpput_translated)
        diff = torch.norm(out - out_translated)
        max_diff = max(max_diff, diff)
        print("difference: ", diff)
