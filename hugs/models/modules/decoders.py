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
    def __init__(self, pose_dim, triplane_dim, hidden_dim=96, feat_dim=None):
        super().__init__()
        self.film_net = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, triplane_dim * 2)  # gamma and beta - removed middle layer
        )
        self.feat_dim = feat_dim  # Optionally set at init
        # Projection layers will be initialized lazily if feat_dim is not known at init
        self.film_proj_gamma = None
        self.film_proj_beta = None
    
    def forward(self, pose, triplane_feats):
        # pose: (B, pose_dim), triplane_feats: (B, C_feat, ...)
        film_params = self.film_net(pose)
        gamma, beta = film_params.chunk(2, dim=-1)  # (B, triplane_dim)
        C_feat = triplane_feats.shape[1]
        # Initialize projection layers if needed
        if (self.film_proj_gamma is None) or (self.film_proj_beta is None) or (self.feat_dim != C_feat):
            self.film_proj_gamma = nn.Linear(gamma.shape[1], C_feat).to(gamma.device)
            self.film_proj_beta = nn.Linear(beta.shape[1], C_feat).to(beta.device)
            self.feat_dim = C_feat
        gamma = self.film_proj_gamma(gamma)  # (B, C_feat)
        beta = self.film_proj_beta(beta)     # (B, C_feat)
        # Unsqueeze to match extra dims if needed
        extra_dims = triplane_feats.dim() - 2
        if extra_dims:
            shape = [gamma.size(0), gamma.size(1)] + [1] * extra_dims
            gamma = gamma.view(shape)
            beta = beta.view(shape)
        return gamma * triplane_feats + beta

class NonRigidDeformer(nn.Module):
    def __init__(self, input_dim=75, triplane_dim=96, hidden_dim=192, act='gelu', use_film=True):
        super().__init__()
        # Don't compute final input_dim here - we'll do it dynamically on first forward
        self.base_input_dim = input_dim  # pose + xyz
        self.triplane_dim = triplane_dim  # Expected triplane dimension
        self.hidden_dim = hidden_dim
        self.act = act_fn_dict[act]
        self.use_film = use_film
        self.pose_dim = input_dim - 3  # Remove xyz from input_dim
        self.xyz_dim = 3
        
        # We'll initialize linear layers on first forward pass when we know actual dimensions
        self.fc1 = None
        self.fc2 = None
        self.ln1 = None
        self.fc3 = None
        self.fc_out = None
        self.layers_initialized = False
        
        if use_film:
            self.film = FiLMModulation(self.pose_dim, triplane_dim, hidden_dim=96)

    def _initialize_layers(self, actual_input_dim):
        """Initialize linear layers with the actual input dimension"""
        self.input_dim = actual_input_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim).to(next(self.parameters()).device)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim).to(next(self.parameters()).device)
        self.ln1 = nn.LayerNorm(self.hidden_dim).to(next(self.parameters()).device)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim).to(next(self.parameters()).device)
        self.fc_out = nn.Linear(self.hidden_dim, 3).to(next(self.parameters()).device)
        
        # Initialize fc_out to output small residuals
        nn.init.constant_(self.fc_out.weight, 0.0)
        nn.init.constant_(self.fc_out.bias, 0.0)
        self.layers_initialized = True

    def forward(self, x):
        # x: [posevec | xyz | triplane_feats]
        # Extract dimensions: posevec (72), xyz (3), triplane_feats (actual_dim)
        pose = x[:, :self.pose_dim]  # First 72 dimensions
        xyz = x[:, self.pose_dim:self.pose_dim + self.xyz_dim]  # Next 3 dimensions  
        triplane_feats = x[:, self.pose_dim + self.xyz_dim:]  # Remaining dimensions
        
        # Write debug info to file since prints aren't showing
        try:
            with open("nonrigid_debug.txt", "w") as f:
                f.write(f"=== NonRigidDeformer Debug ===\n")
                f.write(f"Input x shape: {x.shape}\n")
                f.write(f"pose shape: {pose.shape}\n")
                f.write(f"xyz shape: {xyz.shape}\n")
                f.write(f"triplane_feats shape: {triplane_feats.shape}\n")
                f.write(f"self.pose_dim: {self.pose_dim}\n")
                f.write(f"self.xyz_dim: {self.xyz_dim}\n")
                f.write(f"self.triplane_dim (expected): {self.triplane_dim}\n")
                f.write(f"actual triplane_feats dim: {triplane_feats.shape[1] if len(triplane_feats.shape) > 1 else 'N/A'}\n")
                if hasattr(self, 'input_dim'):
                    f.write(f"self.input_dim (actual): {self.input_dim}\n")
                if self.fc1 is not None:
                    f.write(f"self.fc1 expects input size: {self.fc1.in_features}\n")
        except Exception as e:
            pass  # Don't crash if file write fails
        
        print("pose:", pose.shape, flush=True)
        print("xyz:", xyz.shape, flush=True)
        print("triplane_feats:", triplane_feats.shape, flush=True)
        print("x:", x.shape, flush=True)
        
        if self.use_film:
            modulated_triplane = self.film(pose, triplane_feats)
            x = torch.cat([pose, xyz, modulated_triplane], dim=-1)
        
        # Initialize layers on first forward pass
        if not self.layers_initialized:
            actual_input_dim = x.shape[1]
            self._initialize_layers(actual_input_dim)
        
        # Ensure layers are initialized
        if self.fc1 is None or self.fc2 is None or self.fc3 is None or self.fc_out is None or self.ln1 is None:
            raise RuntimeError("Layers not properly initialized")
        
        # Simplified forward pass - fewer layers, one residual connection
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