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

class FiLMModulation(nn.Module):
    def __init__(self, pose_dim, triplane_dim, hidden_dim=128):
        super().__init__()
        self.film_net = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, triplane_dim * 2)  # gamma and beta
        )
    
    def forward(self, pose, triplane_feats):
        # pose: (N, pose_dim), triplane_feats: (N, triplane_dim)
        film_params = self.film_net(pose)
        gamma, beta = film_params.chunk(2, dim=-1)
        return gamma * triplane_feats + beta

class NonRigidDeformer(nn.Module):
    def __init__(self, input_dim=75, triplane_dim=96, hidden_dim=256, act='gelu', use_film=True):
        super().__init__()
        self.input_dim = input_dim + triplane_dim  # posevec + xyz + triplane_feats
        self.hidden_dim = hidden_dim
        self.act = act_fn_dict[act]
        self.use_film = use_film
        self.pose_dim = input_dim - 3  # Remove xyz from input_dim
        self.xyz_dim = 3
        self.triplane_dim = triplane_dim
        
        if use_film:
            self.film = FiLMModulation(self.pose_dim, triplane_dim)
        
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim + self.input_dim)
        self.fc3 = nn.Linear(hidden_dim + self.input_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim + self.input_dim)
        self.fc5 = nn.Linear(hidden_dim + self.input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 3)
        
        nn.init.constant_(self.fc_out.weight, 0.0)
        nn.init.constant_(self.fc_out.bias, 0.0)

    def forward(self, x):
        # x: [posevec | xyz | triplane_feats]
        # Extract dimensions: posevec (72), xyz (3), triplane_feats (96)
        pose = x[:, :self.pose_dim]  # First 72 dimensions
        xyz = x[:, self.pose_dim:self.pose_dim + self.xyz_dim]  # Next 3 dimensions  
        triplane_feats = x[:, self.pose_dim + self.xyz_dim:]  # Remaining dimensions
        
        if self.use_film:
            modulated_triplane = self.film(pose, triplane_feats)
            x = torch.cat([pose, xyz, modulated_triplane], dim=-1)
        
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        h = torch.cat([h, x], dim=-1)
        h = self.ln1(h)
        h = self.act(self.fc3(h))
        h = self.act(self.fc4(h))
        h = torch.cat([h, x], dim=-1)
        h = self.ln2(h)
        h = self.act(self.fc5(h))
        delta = self.fc_out(h)
        return xyz + delta  # residual connection

    def smoothness_loss(self, delta, xyz, k=8, max_points=1024):
        # delta: (N, 3), xyz: (N, 3)
        N = delta.shape[0]
        if N > 2048:
            idx = torch.randperm(N, device=delta.device)[:max_points]
            delta = delta[idx]
            xyz = xyz[idx]
            N = max_points
        with torch.no_grad():
            dists = torch.cdist(xyz, xyz)  # (N, N)
            knn_idx = dists.topk(k+1, largest=False).indices[:, 1:]  # (N, k), skip self
        loss = 0.0
        for i in range(delta.shape[0]):
            neighbors = delta[knn_idx[i]]
            loss = loss + ((delta[i] - neighbors) ** 2).sum(-1).mean()
        return loss / delta.shape[0]