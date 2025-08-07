#!/usr/bin/env python3

import os
import sys
import torch
import glob
from loguru import logger

sys.path.append('.')

from hugs.models.hugs_trimlp import HUGS_TRIMLP

def verify_checkpoint(checkpoint_path):
    """Verify that a checkpoint contains all expected components"""
    logger.info(f"Verifying checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Check required components
    required_keys = [
        'active_sh_degree',
        'xyz', 
        'triplane',
        'appearance_dec',
        'geometry_dec', 
        'deformation_dec',
        'nonrigid_deformer',  # This is the key one we want to verify
        'scaling_multiplier',
        'max_radii2D',
        'xyz_gradient_accum',
        'denom',
        'optimizer',
        'spatial_lr_scale'
    ]
    
    missing_keys = []
    present_keys = []
    
    for key in required_keys:
        if key in state_dict:
            present_keys.append(key)
            logger.info(f"✅ {key}: {type(state_dict[key])}")
        else:
            missing_keys.append(key)
            logger.warning(f"❌ {key}: MISSING")
    
    # Check nonrigid_deformer specifically
    if 'nonrigid_deformer' in state_dict:
        nonrigid_state = state_dict['nonrigid_deformer']
        logger.info(f"✅ nonrigid_deformer contains {len(nonrigid_state)} parameters")
        for param_name, param_tensor in nonrigid_state.items():
            logger.info(f"   - {param_name}: {param_tensor.shape}")
    else:
        logger.error("❌ nonrigid_deformer is missing from checkpoint!")
    
    # Summary
    logger.info(f"\nSummary:")
    logger.info(f"✅ Present keys: {len(present_keys)}/{len(required_keys)}")
    logger.info(f"❌ Missing keys: {len(missing_keys)}")
    
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    
    return len(missing_keys) == 0

def main():
    # Check all checkpoints in the output directory
    output_dir = "output/human/neuman/citron/hugs_trimlp/20240303_1758_release_test-dataset.seq=citron/2025-08-04_15-58-56"
    ckpt_dir = os.path.join(output_dir, "ckpt")
    
    if not os.path.exists(ckpt_dir):
        logger.error(f"Checkpoint directory not found: {ckpt_dir}")
        return
    
    # Find all checkpoint files
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    ckpt_files.sort()
    
    logger.info(f"Found {len(ckpt_files)} checkpoint files")
    
    all_valid = True
    for ckpt_file in ckpt_files:
        logger.info(f"\n{'='*50}")
        is_valid = verify_checkpoint(ckpt_file)
        if not is_valid:
            all_valid = False
    
    if all_valid:
        logger.info(f"\n🎉 All checkpoints are valid!")
    else:
        logger.error(f"\n❌ Some checkpoints have issues!")

if __name__ == '__main__':
    main() 