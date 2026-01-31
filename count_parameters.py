#!/usr/bin/env python3
"""
Count total number of parameters in Cloth-HUGS model
"""
import sys
import torch

# Add project to path
sys.path.insert(0, '/home/ubx858/ml-hugs')

from hugs.models.hugs_trimlp import HUGS_TRIMLP
from hugs.models.modules.triplane import TriPlane
from hugs.models.modules.decoders import AppearanceDecoder, GeometryDecoder, DeformationDecoder

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("=" * 70)
    print("Cloth-HUGS Model Parameter Count")
    print("=" * 70)
    
    # Configuration from hugs_human_scene.yaml
    triplane_res = 256
    n_features = 32
    sh_degree = 0
    
    print(f"\nConfiguration:")
    print(f"  Triplane resolution: {triplane_res}x{triplane_res}")
    print(f"  Feature dimension: {n_features}")
    print(f"  SH degree: {sh_degree}")
    
    # 1. TriPlane parameters
    print("\n" + "-" * 70)
    print("1. TriPlane Feature Volume")
    print("-" * 70)
    triplane = TriPlane(features=n_features, resX=triplane_res, resY=triplane_res, resZ=triplane_res)
    triplane_params = count_parameters(triplane)
    print(f"  Parameters: {triplane_params:,}")
    print(f"  Breakdown:")
    print(f"    - plane_xy: {n_features} × {triplane_res} × {triplane_res} = {n_features * triplane_res * triplane_res:,}")
    print(f"    - plane_xz: {n_features} × {triplane_res} × {triplane_res} = {n_features * triplane_res * triplane_res:,}")
    print(f"    - plane_yz: {n_features} × {triplane_res} × {triplane_res} = {n_features * triplane_res * triplane_res:,}")
    
    # 2. Appearance Decoder
    print("\n" + "-" * 70)
    print("2. Appearance Decoder")
    print("-" * 70)
    appearance_dec = AppearanceDecoder(n_features=3*n_features, hidden_dim=64, act='gelu')
    appearance_params = count_parameters(appearance_dec)
    print(f"  Input: {3*n_features} (concatenated triplane features)")
    print(f"  Hidden dim: 64")
    print(f"  Outputs: SH coefficients (16×3) + opacity (1)")
    print(f"  Parameters: {appearance_params:,}")
    
    # 3. Geometry Decoder
    print("\n" + "-" * 70)
    print("3. Geometry Decoder")
    print("-" * 70)
    geometry_dec = GeometryDecoder(n_features=3*n_features, hidden_dim=128, act='gelu')
    geometry_params = count_parameters(geometry_dec)
    print(f"  Input: {3*n_features}")
    print(f"  Hidden dim: 128")
    print(f"  Outputs: xyz (3) + rotations (6) + scales (3)")
    print(f"  Parameters: {geometry_params:,}")
    
    # 4. Deformation Decoder
    print("\n" + "-" * 70)
    print("4. Deformation Decoder")
    print("-" * 70)
    deformation_dec = DeformationDecoder(n_features=3*n_features, hidden_dim=128, act='gelu', disable_posedirs=True)
    deformation_params = count_parameters(deformation_dec)
    print(f"  Input: {3*n_features}")
    print(f"  Hidden dim: 128")
    print(f"  Outputs: LBS weights (24) + posedirs (disabled)")
    print(f"  Parameters: {deformation_params:,}")
    
    # Total
    total_network_params = triplane_params + appearance_params + geometry_params + deformation_params
    
    print("\n" + "=" * 70)
    print("TOTAL NETWORK PARAMETERS")
    print("=" * 70)
    print(f"  TriPlane:           {triplane_params:>12,} ({triplane_params/1e6:.2f}M)")
    print(f"  Appearance Decoder: {appearance_params:>12,} ({appearance_params/1e6:.2f}M)")
    print(f"  Geometry Decoder:   {geometry_params:>12,} ({geometry_params/1e6:.2f}M)")
    print(f"  Deformation Decoder:{deformation_params:>12,} ({deformation_params/1e6:.2f}M)")
    print(f"  " + "-" * 66)
    print(f"  TOTAL:              {total_network_params:>12,} ({total_network_params/1e6:.2f}M)")
    
    # Gaussian parameters (per Gaussian, not counted as model params typically)
    print("\n" + "-" * 70)
    print("Note: Gaussian Parameters (per Gaussian, not model params)")
    print("-" * 70)
    print("  Each Gaussian has:")
    print("    - Position (xyz): 3")
    print("    - Rotation (quaternion): 4")
    print("    - Scale: 3")
    print("    - Opacity: 1")
    print("    - SH coefficients: 16×3 = 48")
    print("    Total per Gaussian: 59 parameters")
    print("  Number of Gaussians varies during training (densification)")
    print("  These are optimized directly, not through the network")
    
    print("\n" + "=" * 70)
    print(f"FOR COMPUTE REPORT: {total_network_params/1e6:.1f}M")
    print("=" * 70)

if __name__ == "__main__":
    main()

