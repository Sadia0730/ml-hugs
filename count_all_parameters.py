#!/usr/bin/env python3
"""
Count ALL trainable parameters including Gaussians
"""
import sys
import torch
import glob
import json

sys.path.insert(0, '/home/ubx858/ml-hugs')

from hugs.models.modules.triplane import TriPlane
from hugs.models.modules.decoders import AppearanceDecoder, GeometryDecoder, DeformationDecoder

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_gaussian_params():
    """Estimate Gaussian parameters from typical training runs"""
    # Check actual trained model for Gaussian counts
    results_files = glob.glob('/home/ubx858/ml-hugs/output/**/results_train.json', recursive=True)
    
    print("\n" + "=" * 70)
    print("Checking Actual Trained Models for Gaussian Counts")
    print("=" * 70)
    
    # Parameters per Gaussian
    params_per_gaussian = {
        'xyz': 3,
        'rotation': 4,  # quaternion
        'scale': 3,
        'opacity': 1,
        'SH_coefficients': 16 * 3,  # 16 coefficients × 3 channels (for degree 0)
    }
    total_per_gaussian = sum(params_per_gaussian.values())
    
    print(f"\nParameters per Gaussian: {total_per_gaussian}")
    for k, v in params_per_gaussian.items():
        print(f"  - {k}: {v}")
    
    # Try to load a checkpoint to get actual Gaussian counts
    ckpt_files = glob.glob('/home/ubx858/ml-hugs/output/**/ckpt/final.pth', recursive=True)
    
    if ckpt_files:
        print(f"\nFound {len(ckpt_files)} checkpoint files. Loading one to inspect...")
        try:
            ckpt = torch.load(ckpt_files[0], map_location='cpu')
            
            body_xyz = ckpt.get('xyz', None)
            cloth_xyz = ckpt.get('cloth_xyz', None)
            
            num_body = body_xyz.shape[0] if body_xyz is not None else 0
            num_cloth = cloth_xyz.shape[0] if cloth_xyz is not None else 0
            
            print(f"\nActual Gaussian counts from checkpoint:")
            print(f"  Body Gaussians: {num_body:,}")
            print(f"  Cloth Gaussians: {num_cloth:,}")
            print(f"  Total Gaussians: {num_body + num_cloth:,}")
            
            total_gaussian_params = (num_body + num_cloth) * total_per_gaussian
            print(f"\nTotal Gaussian Parameters: {total_gaussian_params:,} ({total_gaussian_params/1e6:.2f}M)")
            
            return total_gaussian_params, num_body, num_cloth
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    # Fallback estimate
    print("\nUsing estimated values:")
    est_body = 10000
    est_cloth = 5000
    print(f"  Body Gaussians (estimated): {est_body:,}")
    print(f"  Cloth Gaussians (estimated): {est_cloth:,}")
    print(f"  Total Gaussians (estimated): {est_body + est_cloth:,}")
    
    total_gaussian_params = (est_body + est_cloth) * total_per_gaussian
    print(f"\nTotal Gaussian Parameters (estimated): {total_gaussian_params:,} ({total_gaussian_params/1e6:.2f}M)")
    
    return total_gaussian_params, est_body, est_cloth

def main():
    print("=" * 70)
    print("COMPLETE Cloth-HUGS Parameter Count")
    print("=" * 70)
    
    # Network parameters
    triplane_res = 256
    n_features = 32
    
    triplane = TriPlane(features=n_features, resX=triplane_res, resY=triplane_res, resZ=triplane_res)
    appearance_dec = AppearanceDecoder(n_features=3*n_features, hidden_dim=64, act='gelu')
    geometry_dec = GeometryDecoder(n_features=3*n_features, hidden_dim=128, act='gelu')
    deformation_dec = DeformationDecoder(n_features=3*n_features, hidden_dim=128, act='gelu', disable_posedirs=True)
    
    triplane_params = count_parameters(triplane)
    appearance_params = count_parameters(appearance_dec)
    geometry_params = count_parameters(geometry_dec)
    deformation_params = count_parameters(deformation_dec)
    
    total_network_params = triplane_params + appearance_params + geometry_params + deformation_params
    
    print("\n" + "-" * 70)
    print("1. Network Parameters (Triplane + Decoders)")
    print("-" * 70)
    print(f"  TriPlane (shared):      {triplane_params:>12,} ({triplane_params/1e6:.2f}M)")
    print(f"  Appearance Decoder:     {appearance_params:>12,} ({appearance_params/1e6:.2f}M)")
    print(f"  Geometry Decoder:       {geometry_params:>12,} ({geometry_params/1e6:.2f}M)")
    print(f"  Deformation Decoder:    {deformation_params:>12,} ({deformation_params/1e6:.2f}M)")
    print(f"  " + "-" * 66)
    print(f"  Network Total:          {total_network_params:>12,} ({total_network_params/1e6:.2f}M)")
    
    # Gaussian parameters
    gaussian_params, num_body, num_cloth = estimate_gaussian_params()
    
    # Total
    total_all_params = total_network_params + gaussian_params
    
    print("\n" + "=" * 70)
    print("TOTAL TRAINABLE PARAMETERS")
    print("=" * 70)
    print(f"  Network Parameters:     {total_network_params:>12,} ({total_network_params/1e6:.2f}M)")
    print(f"  Gaussian Parameters:    {gaussian_params:>12,} ({gaussian_params/1e6:.2f}M)")
    print(f"  " + "-" * 66)
    print(f"  GRAND TOTAL:            {total_all_params:>12,} ({total_all_params/1e6:.2f}M)")
    
    print("\n" + "=" * 70)
    print("FOR COMPUTE REPORT:")
    print("=" * 70)
    print(f"  Network only:  {total_network_params/1e6:.1f}M")
    print(f"  Total (with Gaussians): {total_all_params/1e6:.1f}M")
    print("\nRecommendation:")
    print(f"  Use '{total_all_params/1e6:.1f}M' for complete model size")
    print(f"  Or '{total_network_params/1e6:.1f}M' if reporting network parameters only")
    print("=" * 70)

if __name__ == "__main__":
    main()

