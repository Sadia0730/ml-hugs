#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
import torch
import torch.nn as nn
from loguru import logger
import torch.nn.functional as F

from .activation import SineActivation


act_fn_dict = {
    'softplus': torch.nn.Softplus(),
    'relu': torch.nn.ReLU(),
    'sine': SineActivation(omega_0=30),
    'gelu': torch.nn.GELU(),
    'tanh': torch.nn.Tanh(),
}


class AppearanceDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=64, act='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
            
        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.opacity = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        self.shs = nn.Linear(hidden_dim, 16*3)
        
    def forward(self, x):

        x = self.net(x)
        shs = self.shs(x)
        opacity = self.opacity(x)
        return {'shs': shs, 'opacity': opacity}
    

class DeformationDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=128, weight_norm=True, act='gelu', disable_posedirs=False):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.sine = SineActivation(omega_0=30)
        self.disable_posedirs = disable_posedirs
        
        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.skinning_linear = nn.Linear(hidden_dim, hidden_dim)
        self.skinning = nn.Linear(hidden_dim, 24)
        
        if weight_norm:
            self.skinning_linear = nn.utils.weight_norm(self.skinning_linear)
            
        # initialize blendshapes to be zero, and skinning weights to be equal for every bone (after softmax activation)
        if not disable_posedirs:
            self.blendshapes = nn.Linear(hidden_dim, 3 * 207)
            torch.nn.init.constant_(self.blendshapes.bias, 0.0)
            torch.nn.init.constant_(self.blendshapes.weight, 0.0)
        
    def forward(self, x):
        x = self.net(x)
        if not self.disable_posedirs:
            posedirs = self.blendshapes(x)
            posedirs = posedirs.reshape(207, -1)
            
        lbs_weights = self.skinning(F.gelu(self.skinning_linear(x)))
        lbs_weights = F.gelu(lbs_weights)
        
        return {
            'lbs_weights': lbs_weights,
            'posedirs': posedirs if not self.disable_posedirs else None,
        }
    

class GeometryDecoder(torch.nn.Module):
    def __init__(self, n_features, use_surface=False, hidden_dim=128, act='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.xyz = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 3))
        self.rotations = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 6))
        self.scales = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 2 if use_surface else 3))
        
    def forward(self, x):
        xyz = self.xyz(x)
        rotations = self.rotations(x)
        scales = F.gelu(self.scales(x))
                
        return {
            'xyz': xyz,
            'rotations': rotations,
            'scales': scales,
        }

# OLD FiLM IMPLEMENTATION - COMMENTED OUT
# Replaced with frame-based FiLM in the main model
# class FiLMModulation(nn.Module):
#     def __init__(self, pose_dim, triplane_dim, hidden_dim=96): 
#         super().__init__()
#         self.film_net = nn.Sequential(
#             nn.Linear(pose_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, triplane_dim * 2)  
#         )
#     
#     def forward(self, pose, triplane_feats):
#         # pose: (N, pose_dim), triplane_feats: (N, triplane_dim)
#         film_params = self.film_net(pose)
#         gamma, beta = film_params.chunk(2, dim=-1)
#         return gamma * triplane_feats + beta

class NonRigidDeformer(nn.Module):
    def __init__(self, input_dim=75, triplane_dim=96, hidden_dim=128, act='gelu', use_smoothness=False):
        super().__init__()
        self.pose_dim = input_dim - 3     # 72 for SMPL-23x3 (axis-angle), or 23x6 if using 6D
        self.xyz_dim  = 3
        self.triplane_dim = triplane_dim

        in_dim = input_dim + triplane_dim              # pose + xyz + triplane (keeps ckpt compat for fc1 shape)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln1 = nn.LayerNorm(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 3)

        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

        self.act = act_fn_dict[act]
        self.use_smoothness = use_smoothness

        # NEW: bounded, learnable deformation scale
        self.delta_scale = nn.Parameter(torch.tensor(0.05))

    # Fast path: pose is a single vector (P,), xyz is (N,3), tri is (N,F)
    def forward_from_parts(self, pose_vec, xyz, tri):
        W = self.fc1.weight                       # (H, P+3+F)
        b = self.fc1.bias                         # (H,)
        P = self.pose_dim
        Wp, Wr = W[:, :P], W[:, P:]               # (H,P), (H,3+F)

        # fold pose once → broadcast via bias; avoids repeating pose for every point
        pose_bias = pose_vec @ Wp.t() + b         # (H,)
        xr = torch.cat([xyz, tri], dim=-1)        # (N, 3+F)

        h = self.act(xr @ Wr.t() + pose_bias)     # (N, H)
        h = self.act(self.fc2(h))
        h = self.ln1(h)
        h = self.act(self.fc3(h))

        delta_raw = self.fc_out(h)                # (N,3)
        delta = torch.tanh(delta_raw) * self.delta_scale
        out = xyz + delta
        return out, delta                         # return both for loss/logging

    # Back-compat (slower): concatenated x = [pose | xyz | tri] per-point
    def forward(self, x):
        pose = x[:, :self.pose_dim]
        xyz  = x[:, self.pose_dim:self.pose_dim + 3]
        tri  = x[:, self.pose_dim + 3:]
        pose_vec = pose[0]                        # all rows share same pose per frame
        return self.forward_from_parts(pose_vec, xyz, tri)

    def smoothness_loss(self, delta, xyz=None, k=6, max_points=256):
        if not self.use_smoothness:
            return delta.new_zeros(())
        # Random-neighbor, vectorized (very fast)
        N = delta.shape[0]
        if N > max_points:
            idx = torch.randperm(N, device=delta.device)[:max_points]
            delta = delta[idx]; N = max_points

        with torch.no_grad():
            nn_idx = torch.randint(0, N, (N, k), device=delta.device)
            self_idx = torch.arange(N, device=delta.device).view(-1, 1)
            replace = torch.randint(0, N, (N, 1), device=delta.device)
            nn_idx = torch.where(nn_idx == self_idx, replace, nn_idx)

        neigh = delta[nn_idx]                     # (N,k,3)
        diff  = delta[:, None, :] - neigh         # (N,k,3)
        return diff.pow(2).sum(-1).mean()