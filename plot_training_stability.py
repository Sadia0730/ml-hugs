#!/usr/bin/env python3
"""
Plot training stability graph showing when training collapsed.
Usage: python plot_training_stability.py <results_json_path>
"""

import json
import sys
import os

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy are required. Install with:")
    print("  pip install matplotlib numpy")
    sys.exit(1)


def plot_stability(json_path, output_path=None):
    """Plot training stability metrics over iterations."""
    
    # Load data
    with open(json_path) as f:
        data = json.load(f)
    
    # Extract metrics
    iterations = []
    lpips = []
    psnr = []
    ssim = []
    
    for k, v in data.items():
        if k != 'final':
            iterations.append(int(k))
            lpips.append(v['hugs_lpips'])
            psnr.append(v['hugs_psnr'])
            ssim.append(v['hugs_ssim'])
    
    iterations = np.array(iterations)
    lpips = np.array(lpips)
    psnr = np.array(psnr)
    ssim = np.array(ssim)
    
    # Sort by iteration
    sort_idx = np.argsort(iterations)
    iterations = iterations[sort_idx]
    lpips = lpips[sort_idx]
    psnr = psnr[sort_idx]
    ssim = ssim[sort_idx]
    
    # Identify stable and unstable regions
    stable_mask = iterations < 15000
    unstable_mask = ~stable_mask
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Training Stability Analysis - ZJU Subject 392', 
                 fontsize=16, fontweight='bold')
    
    # Plot LPIPS
    ax1 = axes[0]
    ax1.plot(iterations[stable_mask], lpips[stable_mask], 
             'b-', linewidth=2.5, label='Stable Training', alpha=0.8, marker='o', markersize=4)
    if unstable_mask.sum() > 0:
        ax1.plot(iterations[unstable_mask], lpips[unstable_mask], 
                 'r--', linewidth=2.5, label='Unstable/Collapsed', alpha=0.8, marker='x', markersize=6)
    ax1.axvline(x=15000, color='red', linestyle='--', linewidth=2, 
                label='Collapse Point (iter 15000)', alpha=0.7)
    ax1.set_ylabel('LPIPS\n(lower is better)', fontsize=12, fontweight='bold')
    ax1.set_title('LPIPS Over Training Iterations', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_xlim(left=0, right=max(iterations) * 1.05)
    
    # Plot PSNR
    ax2 = axes[1]
    ax2.plot(iterations[stable_mask], psnr[stable_mask], 
             'g-', linewidth=2.5, label='Stable Training', alpha=0.8, marker='o', markersize=4)
    if unstable_mask.sum() > 0:
        ax2.plot(iterations[unstable_mask], psnr[unstable_mask], 
                 'r--', linewidth=2.5, label='Unstable/Collapsed', alpha=0.8, marker='x', markersize=6)
    ax2.axvline(x=15000, color='red', linestyle='--', linewidth=2, 
                label='Collapse Point (iter 15000)', alpha=0.7)
    ax2.set_ylabel('PSNR (dB)\n(higher is better)', fontsize=12, fontweight='bold')
    ax2.set_title('PSNR Over Training Iterations', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower left', fontsize=10)
    ax2.set_xlim(left=0, right=max(iterations) * 1.05)
    
    # Plot SSIM
    ax3 = axes[2]
    ax3.plot(iterations[stable_mask], ssim[stable_mask], 
             'm-', linewidth=2.5, label='Stable Training', alpha=0.8, marker='o', markersize=4)
    if unstable_mask.sum() > 0:
        ax3.plot(iterations[unstable_mask], ssim[unstable_mask], 
                 'r--', linewidth=2.5, label='Unstable/Collapsed', alpha=0.8, marker='x', markersize=6)
    ax3.axvline(x=15000, color='red', linestyle='--', linewidth=2, 
                label='Collapse Point (iter 15000)', alpha=0.7)
    ax3.set_ylabel('SSIM\n(higher is better)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax3.set_title('SSIM Over Training Iterations', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='lower left', fontsize=10)
    ax3.set_xlim(left=0, right=max(iterations) * 1.05)
    
    # Add text annotation for stable region
    fig.text(0.5, 0.02, 
             f'Stable Region: Iterations 0 - 14,000 | Best Checkpoint: Iteration 12,000',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    # Save figure
    if output_path is None:
        output_path = os.path.join(os.path.dirname(json_path), 'training_stability.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Graph saved to: {output_path}")
    
    plt.close()
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING STABILITY SUMMARY")
    print("="*70)
    print(f"✓ Stable Region: Iterations 0 - 14,000 ({stable_mask.sum()} iterations)")
    print(f"✗ Collapse Point: Iteration 15,000")
    print(f"  Best Stable Checkpoint: Iteration 12,000")
    print(f"    LPIPS: {lpips[iterations == 12000][0]:.5f}")
    print(f"    PSNR:  {psnr[iterations == 12000][0]:.3f}")
    print(f"    SSIM:  {ssim[iterations == 12000][0]:.5f}")
    print("="*70)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Default path
        json_path = 'output/human/zju/392/hugs_trimlp/cloth_zju_392_train/2025-11-01_16-24-20/results_train.json'
        print(f"Using default path: {json_path}")
    else:
        json_path = sys.argv[1]
    
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    plot_stability(json_path)

