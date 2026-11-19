"""
PointNet++ (SSG, single-scale grouping) classification network.
Adapted from pointnet2_cls_ssg.py in yanx27/Pointnet_Pointnet2_pytorch (MIT License).

in_channels: number of extra per-point features besides xyz
  - ModelNet40: 3 with normals, 0 without
  - Coral point clouds with RGB: 3
"""
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_utils import PointNetSetAbstraction


class PointNet2Cls(nn.Module):
    def __init__(self, num_classes, in_channels=0, dropout=0.4):
        super().__init__()
        self.in_channels = in_channels
        # Three SA layers: 512 centers (radius 0.2) -> 128 centers (radius 0.4) -> global
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, 3 + in_channels, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], True)
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """x: [B, 3 + in_channels, N]; the first 3 channels are xyz. Returns logits [B, num_classes]."""
        B = x.shape[0]
        xyz = x[:, :3, :]
        feats = x[:, 3:, :] if self.in_channels > 0 else None
        l1_xyz, l1_points = self.sa1(xyz, feats)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        return self.head(l3_points.view(B, 1024))
