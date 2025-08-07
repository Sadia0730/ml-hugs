#!/usr/bin/env python3

import torch
import torch.nn as nn
from loguru import logger
import sys
sys.path.append('.')

from hugs.models.hugs_trimlp import HUGS_TRIMLP
from hugs.models.scene import SceneGS
from hugs.models.modules.triplane import TriPlane
from hugs.models.modules.decoders import AppearanceDecoder, GeometryDecoder, DeformationDecoder, NonRigidDeformer

def count_parameters(model):
    """Count total parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def estimate_flops(model, input_size=(110210, 3)):
    """Estimate FLOPs for the model"""
    # This is a rough estimation
    # In practice, you'd need to use tools like torchprofile or fvcore
    
    # For triplane: 3 planes × resolution × features
    triplane_flops = 3 * 256 * 256 * 32 * input_size[0]  # 3 planes, 256x256, 32 features
    
    # For decoders: input_size × hidden_dim × output_dim
    decoder_flops = input_size[0] * 96 * 192 * 3  # geometry decoder
    
    # For non-rigid deformer: input_size × hidden_dim × layers
    deformer_flops = input_size[0] * 171 * 192 * 4  # 4 layers
    
    total_flops = triplane_flops + decoder_flops + deformer_flops
    return total_flops

def analyze_model_complexity():
    """Analyze the computational complexity of HUGS model"""
    
    logger.info("=== HUGS Model Complexity Analysis ===")
    
    # Create model components
    triplane = TriPlane(features=32, resX=256, resY=256, resZ=256)
    appearance_dec = AppearanceDecoder(n_features=96)
    geometry_dec = GeometryDecoder(n_features=96)
    deformation_dec = DeformationDecoder(n_features=96)
    nonrigid_deformer = NonRigidDeformer(input_dim=171, triplane_dim=96, hidden_dim=192)
    
    # Count parameters for each component
    components = {
        'TriPlane': triplane,
        'AppearanceDecoder': appearance_dec,
        'GeometryDecoder': geometry_dec,
        'DeformationDecoder': deformation_dec,
        'NonRigidDeformer': nonrigid_deformer
    }
    
    total_params = 0
    total_trainable = 0
    
    logger.info("\n=== Parameter Count ===")
    for name, model in components.items():
        params, trainable = count_parameters(model)
        total_params += params
        total_trainable += trainable
        logger.info(f"{name}:")
        logger.info(f"  Total parameters: {params:,}")
        logger.info(f"  Trainable parameters: {trainable:,}")
    
    # Add triplane parameters (3 planes × 256 × 256 × 32)
    triplane_params = 3 * 256 * 256 * 32
    total_params += triplane_params
    total_trainable += triplane_params
    
    logger.info(f"\nTriPlane (3 planes):")
    logger.info(f"  Parameters: {triplane_params:,}")
    
    # Add Gaussian parameters (110,210 Gaussians)
    n_gaussians = 110210
    gaussian_params = {
        'xyz': n_gaussians * 3,
        'scaling': n_gaussians * 3,
        'rotation': n_gaussians * 4,
        'opacity': n_gaussians * 1,
        'features': n_gaussians * 16 * 3,  # SH coefficients
    }
    
    total_gaussian_params = sum(gaussian_params.values())
    total_params += total_gaussian_params
    total_trainable += total_gaussian_params
    
    logger.info(f"\nGaussian Parameters ({n_gaussians:,} Gaussians):")
    for name, count in gaussian_params.items():
        logger.info(f"  {name}: {count:,}")
    logger.info(f"  Total Gaussian parameters: {total_gaussian_params:,}")
    
    # Add SMPL parameters
    smpl_params = {
        'global_orient': 6,  # per frame
        'body_pose': 23 * 6,  # per frame
        'betas': 10,
        'transl': 3,  # per frame
    }
    total_smpl_params = sum(smpl_params.values())
    total_params += total_smpl_params
    total_trainable += total_smpl_params
    
    logger.info(f"\nSMPL Parameters:")
    for name, count in smpl_params.items():
        logger.info(f"  {name}: {count:,}")
    logger.info(f"  Total SMPL parameters: {total_smpl_params:,}")
    
    logger.info(f"\n=== SUMMARY ===")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {total_trainable:,}")
    
    # Estimate FLOPs
    estimated_flops = estimate_flops(None)
    logger.info(f"Estimated FLOPs per forward pass: {estimated_flops:,}")
    
    # Memory usage estimation
    param_memory = total_params * 4  # 4 bytes per float32
    gaussian_memory = n_gaussians * (3+3+4+1+16*3) * 4  # xyz + scaling + rotation + opacity + features
    triplane_memory = 3 * 256 * 256 * 32 * 4
    
    total_memory_mb = (param_memory + gaussian_memory + triplane_memory) / (1024 * 1024)
    
    logger.info(f"Estimated memory usage: {total_memory_mb:.1f} MB")
    
    return total_params, estimated_flops

if __name__ == '__main__':
    analyze_model_complexity() 