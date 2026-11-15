"""
Basic PointNet++ operators: farthest point sampling (FPS), ball query, and the Set Abstraction layer.

Adapted from yanx27/Pointnet_Pointnet2_pytorch (MIT License)
https://github.com/yanx27/Pointnet_Pointnet2_pytorch
Original paper: Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets
in a Metric Space", NeurIPS 2017.

Pure PyTorch, no CUDA extensions to compile; runs on both Windows and Linux.
Tensor shape conventions: B=batch, N=number of points, S=number of sampled centers, K=points per neighborhood, C=feature dimension.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """Pairwise squared distances between two point sets. src: [B,N,C], dst: [B,M,C] -> [B,N,M]"""
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, -1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, -1).unsqueeze(1)
    return dist


def index_points(points, idx):
    """Gather points by index. points: [B,N,C], idx: [B,S] or [B,S,K] -> [B,S,C] or [B,S,K,C]"""
    B = points.shape[0]
    view_shape = [B] + [1] * (idx.dim() - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_idx, idx, :]


def farthest_point_sample(xyz, npoint):
    """Farthest point sampling: pick npoint centers that cover the whole cloud as evenly as possible. xyz: [B,N,3] -> [B,npoint]"""
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """Ball query: for each center, find up to nsample neighbors within the given radius. -> [B,S,nsample]"""
    device = xyz.device
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # If there are fewer than nsample neighbors, pad with the first neighbor (at least the center itself)
    group_first = group_idx[:, :, 0:1].repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """FPS to pick centers + ball query to group. Returns local coordinates (relative to the center) concatenated with point features."""
    B, N, C = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)                        # [B,S,3]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)                        # [B,S,K,3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)              # [B,S,K,D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """Treat the whole cloud as one group (used by the last SA layer, equivalent to one PointNet producing a global feature)."""
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=xyz.device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """Set Abstraction layer = sampling + grouping + a small PointNet (shared MLP + max pool)."""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint, self.radius, self.nsample, self.group_all = npoint, radius, nsample, group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """xyz: [B,3,N], points: [B,D,N] or None -> new_xyz: [B,3,S], new_points: [B,D',S]"""
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)             # [B,C,K,S]
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]                # max pool within each neighborhood
        return new_xyz.permute(0, 2, 1), new_points
