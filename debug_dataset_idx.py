#!/usr/bin/env python3

import torch
import sys
sys.path.append('.')

def test_indexing():
    print("=== Testing Dataset Index Behavior ===")
    
    # Simulate your tensor shapes
    global_orient = torch.zeros(30, 6)  # 30 frames, 6 rot6d each
    body_pose = torch.zeros(30, 138)    # 30 frames, 138 rot6d each
    
    print(f"global_orient shape: {global_orient.shape}")
    print(f"body_pose shape: {body_pose.shape}")
    
    # Test different dataset_idx values
    for dataset_idx in [-1, 29, torch.tensor(-1), torch.tensor(29)]:
        print(f"\n--- Testing dataset_idx = {dataset_idx} (type: {type(dataset_idx)}) ---")
        
        try:
            go_selected = global_orient[dataset_idx]
            bp_selected = body_pose[dataset_idx]
            
            print(f"  global_orient[{dataset_idx}] shape: {go_selected.shape}")
            print(f"  body_pose[{dataset_idx}] shape: {bp_selected.shape}")
            
            # Test the rotation conversion
            go_6d = go_selected
            if go_6d.dim() > 1:
                go_6d = go_6d.flatten()
            print(f"  After flatten: {go_6d.shape}")
            
            # Test reshape
            try:
                go_reshaped = go_6d.reshape(1, 6)
                print(f"  Reshape to (1, 6): SUCCESS - {go_reshaped.shape}")
            except Exception as e:
                print(f"  Reshape to (1, 6): FAILED - {e}")
                
        except Exception as e:
            print(f"  Indexing failed: {e}")

if __name__ == "__main__":
    test_indexing() 