#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
import numpy as np
import torch.nn as nn
from loguru import logger
import torch.nn.functional as F

EPS = 1e-3

class TriPlane(nn.Module):
    def __init__(self, features=32, resX=256, resY=256, resZ=256):
        super().__init__()
        self.plane_xy = nn.Parameter(torch.randn(1, features, resX, resY))
        self.plane_xz = nn.Parameter(torch.randn(1, features, resX, resZ))
        self.plane_yz = nn.Parameter(torch.randn(1, features, resY, resZ))
        self.dim = features
        self.n_input_dims = 3
        self.n_output_dims = 3 * features
        self.center = 0.0
        self.scale = 2.0

    def forward(self, x):
        x = (x - self.center) / self.scale + 0.5

        assert x.max() <= 1 + EPS and x.min() >= -EPS, f"x must be in [0, 1], got {x.min()} and {x.max()}"
        x = x * 2 - 1
        shape = x.shape
        coords = x.reshape(1, -1, 1, 3)
        
        # DEBUG: Check plane shapes
        if hasattr(self, '_debug_printed') and not self._debug_printed:
            print(f"TriPlane DEBUG: plane_xy shape: {self.plane_xy.shape}")
            print(f"TriPlane DEBUG: plane_xz shape: {self.plane_xz.shape}")
            print(f"TriPlane DEBUG: plane_yz shape: {self.plane_yz.shape}")
            print(f"TriPlane DEBUG: coords shape: {coords.shape}")
            self._debug_printed = True
        
        # align_corners=True ==> the extrema (-1 and 1) considered as the center of the corner pixels
        # F.grid_sample: [1, C, H, W], [1, N, 1, 2] -> [1, C, N, 1]
        feat_xy = F.grid_sample(self.plane_xy, coords[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_xz = F.grid_sample(self.plane_xz, coords[..., [0, 2]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_yz = F.grid_sample(self.plane_yz, coords[..., [1, 2]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        
        # DEBUG: Check individual feature shapes
        if not hasattr(self, '_debug_printed') or not self._debug_printed:
            print(f"TriPlane DEBUG: feat_xy shape: {feat_xy.shape}")
            print(f"TriPlane DEBUG: feat_xz shape: {feat_xz.shape}")
            print(f"TriPlane DEBUG: feat_yz shape: {feat_yz.shape}")
            self._debug_printed = True
        
        feat = torch.cat([feat_xy, feat_xz, feat_yz], dim=1)
        
        # DEBUG: Check final concatenated features
        if not hasattr(self, '_debug_final_printed') or not self._debug_final_printed:
            print(f"TriPlane DEBUG: feat shape after cat: {feat.shape}")
            print(f"TriPlane DEBUG: expected output dim: {3 * self.dim}")
            self._debug_final_printed = True
        
        feat = feat.reshape(*shape[:-1], 3 * self.dim)
        
        # CRITICAL: Check for dimension mismatch
        expected_dim = 3 * self.dim
        actual_dim = feat.shape[-1]
        if actual_dim != expected_dim:
            print(f"TriPlane ERROR: Output dimension mismatch!")
            print(f"  Expected: {expected_dim}, Got: {actual_dim}")
            print(f"  self.dim: {self.dim}")
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {feat.shape}")
            # Force correct dimensions
            if actual_dim > expected_dim:
                print("  Truncating to expected dimensions...")
                feat = feat[..., :expected_dim]
        
        return feat