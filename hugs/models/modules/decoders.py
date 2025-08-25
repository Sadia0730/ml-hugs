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
    def __init__(self, pose_dim, triplane_dim, hidden_dim=96): 
        super().__init__()
        self.film_net = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, triplane_dim * 2)  
        )
    
    def forward(self, pose, triplane_feats):
        # pose: (N, pose_dim), triplane_feats: (N, triplane_dim)
        film_params = self.film_net(pose)
        gamma, beta = film_params.chunk(2, dim=-1)
        return gamma * triplane_feats + beta

class NonRigidDeformer(nn.Module):
    def __init__(self, input_dim=75, triplane_dim=96, hidden_dim=192, act='gelu', use_film=True):
        super().__init__()
        self.input_dim = input_dim + triplane_dim  # posevec + xyz + triplane_feats
        self.hidden_dim = hidden_dim  
        self.act = act_fn_dict[act]
        self.use_film = use_film
        self.pose_dim = input_dim - 3  
        self.xyz_dim = 3
        self.triplane_dim = triplane_dim
        
        # Don't create FiLM here - we'll create it dynamically when we know the actual dimensions
        self.film = None
        
        # Add projection layer to handle dimension mismatches
        self.triplane_projection = None
        
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 3)
        
        nn.init.constant_(self.fc_out.weight, 0.0)
        nn.init.constant_(self.fc_out.bias, 0.0)

    def set_triplane_projection(self, actual_triplane_dim):
        """Set up projection layer if triplane dimensions don't match"""
        if actual_triplane_dim != self.triplane_dim:
            self.triplane_projection = nn.Linear(actual_triplane_dim, self.triplane_dim).to(self.fc1.weight.device)
            logger.info(f"Added triplane projection: {actual_triplane_dim} -> {self.triplane_dim}")
            
            # Create FiLM with the projected dimension
            if self.use_film and self.film is None:
                self.film = FiLMModulation(self.pose_dim, self.triplane_dim, hidden_dim=96).to(self.fc1.weight.device)
                logger.info(f"Created FiLM with projected triplane_dim: {self.triplane_dim}")

    def forward(self, x):
        pose = x[:, :self.pose_dim]  # First pose_dim dimensions
        xyz = x[:, self.pose_dim:self.pose_dim + self.xyz_dim] 
        triplane_feats = x[:, self.pose_dim + self.xyz_dim:]  
        
        # Apply projection if needed
        if self.triplane_projection is not None:
            triplane_feats = self.triplane_projection(triplane_feats)
        
        if self.use_film and self.film is not None:
            modulated_triplane = self.film(pose, triplane_feats)
            x = torch.cat([pose, xyz, modulated_triplane], dim=-1)
        else:
            x = torch.cat([pose, xyz, triplane_feats], dim=-1)
        
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        residual = h  # Store for residual connection
        h = self.ln1(h)
        h = self.act(self.fc3(h))
        h = h + residual  # Residual connection
        delta = self.fc_out(h)
        return xyz + delta  # residual connection

    def smoothness_loss(self, delta, xyz, k=6, max_points=768):
        # delta: (N, 3), xyz: (N, 3) - Reduced k and max_points for efficiency
        N = delta.shape[0]
        if N > max_points:
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