#!/usr/bin/env python3
"""
Extract and compare metrics from ablation study runs.
Useful for generating tables for paper.
"""

import json
import os
import glob
from pathlib import Path
import argparse

def extract_metrics_from_json(json_path):
    """Extract final metrics from results_train.json"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get final iteration (before collapse if it happened)
        iterations = [k for k in data.keys() if k.isdigit()]
        if not iterations:
            return None
        
        # Use iteration 19000 (before potential 20000 collapse)
        # or last valid iteration
        valid_iters = [int(k) for k in iterations if int(k) < 20000]
        if valid_iters:
            final_iter = str(max(valid_iters)).zfill(6)
        else:
            final_iter = max(iterations, key=lambda x: int(x))
        
        metrics = data.get(final_iter, {})
        
        return {
            'iteration': final_iter,
            'psnr': metrics.get('hugs_psnr', None),
            'ssim': metrics.get('hugs_ssim', None),
            'lpips': metrics.get('hugs_lpips', None),
            'fid': metrics.get('hugs_fid', None),
            'human_psnr': metrics.get('hugs_human_psnr', None),
            'human_ssim': metrics.get('hugs_human_ssim', None),
            'human_lpips': metrics.get('hugs_human_lpips', None),
            'human_fid': metrics.get('hugs_human_fid', None),
        }
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None


def find_ablation_runs(base_dir):
    """Find all ablation run directories"""
    pattern = os.path.join(base_dir, "**/abl_cloth_*/**/results_train.json")
    json_files = glob.glob(pattern, recursive=True)
    
    results = {}
    for json_path in json_files:
        # Extract config name from path
        # Expected: output_ablation/.../abl_cloth_*/.../results_train.json
        parts = Path(json_path).parts
        config_name = None
        for part in parts:
            if part.startswith('abl_cloth_'):
                config_name = part
                break
        
        if config_name:
            metrics = extract_metrics_from_json(json_path)
            if metrics:
                if config_name not in results:
                    results[config_name] = []
                results[config_name].append(metrics)
    
    return results


def print_table(results):
    """Print a formatted table of results"""
    # Configuration mapping
    config_display = {
        'abl_cloth_baseline': ('Baseline', '✓ (1000)', '✗', '✗', '✗'),
        'abl_cloth_sim_only': ('+ Sim', '✓ (1000)', '✓ (1.0)', '✗', '✗'),
        'abl_cloth_arap_only': ('+ ARAP', '✓ (1000)', '✗', '✓ (0.5)', '✗'),
        'abl_cloth_mask_only': ('+ Mask', '✓ (1000)', '✗', '✗', '✓ (1.0)'),
        'abl_cloth_sim_arap': ('+ Sim+ARAP', '✓ (1000)', '✓ (1.0)', '✓ (0.5)', '✗'),
        'abl_cloth_sim_mask': ('+ Sim+Mask', '✓ (1000)', '✓ (1.0)', '✗', '✓ (1.0)'),
        'abl_cloth_arap_mask': ('+ ARAP+Mask', '✓ (1000)', '✗', '✓ (0.5)', '✓ (1.0)'),
        'abl_cloth_full': ('Full (All)', '✓ (1000)', '✓ (1.0)', '✓ (0.5)', '✓ (1.0)'),
        'abl_cloth_full_no_lbs': ('Full (No LBS)', '✗', '✓ (1.0)', '✓ (0.5)', '✓ (1.0)'),
        'abl_cloth_sim_only_no_lbs': ('Sim (No LBS)', '✗', '✓ (1.0)', '✗', '✗'),
        'abl_cloth_full_lbs_weak': ('Full (LBS weak)', '✓ (100)', '✓ (1.0)', '✓ (0.5)', '✓ (1.0)'),
    }
    
    print("\n" + "="*120)
    print("ABLATION STUDY RESULTS TABLE")
    print("="*120)
    print(f"{'Method':<20} {'LBS Reg.':<12} {'L_sim':<10} {'L_ARAP':<10} {'L_mask':<10} {'PSNR ↑':<10} {'SSIM ↑':<10} {'LPIPS ↓':<10} {'Iter':<8}")
    print("-"*120)
    
    # Sort by config name for consistent output
    sorted_configs = sorted(results.keys())
    
    for config in sorted_configs:
        if config not in config_display:
            continue
            
        display_name, lbs, sim, arap, mask = config_display[config]
        
        # Average metrics if multiple runs
        config_results = results[config]
        if not config_results:
            continue
        
        avg_psnr = sum(r['psnr'] for r in config_results if r['psnr'] is not None) / len([r for r in config_results if r['psnr'] is not None])
        avg_ssim = sum(r['ssim'] for r in config_results if r['ssim'] is not None) / len([r for r in config_results if r['ssim'] is not None])
        avg_lpips = sum(r['lpips'] for r in config_results if r['lpips'] is not None) / len([r for r in config_results if r['lpips'] is not None])
        iter_str = config_results[0]['iteration']
        
        # Highlight baseline
        if config == 'abl_cloth_baseline':
            display_name = f"**{display_name}**"
        
        print(f"{display_name:<20} {lbs:<12} {sim:<10} {arap:<10} {mask:<10} "
              f"{avg_psnr:>8.2f}   {avg_ssim:>8.3f}   {avg_lpips:>8.4f}   {iter_str:<8}")
    
    print("="*120)
    print("\nNote: Metrics are averaged across runs. Baseline (LBS-only) is highlighted.")


def print_latex_table(results):
    """Print LaTeX table format"""
    config_display = {
        'abl_cloth_baseline': ('Baseline', '\\checkmark (1000)', '$\\times$', '$\\times$', '$\\times$'),
        'abl_cloth_sim_only': ('+ Sim', '\\checkmark (1000)', '\\checkmark (1.0)', '$\\times$', '$\\times$'),
        'abl_cloth_arap_only': ('+ ARAP', '\\checkmark (1000)', '$\\times$', '\\checkmark (0.5)', '$\\times$'),
        'abl_cloth_mask_only': ('+ Mask', '\\checkmark (1000)', '$\\times$', '$\\times$', '\\checkmark (1.0)'),
        'abl_cloth_sim_arap': ('+ Sim+ARAP', '\\checkmark (1000)', '\\checkmark (1.0)', '\\checkmark (0.5)', '$\\times$'),
        'abl_cloth_sim_mask': ('+ Sim+Mask', '\\checkmark (1000)', '\\checkmark (1.0)', '$\\times$', '\\checkmark (1.0)'),
        'abl_cloth_arap_mask': ('+ ARAP+Mask', '\\checkmark (1000)', '$\\times$', '\\checkmark (0.5)', '\\checkmark (1.0)'),
        'abl_cloth_full': ('Full (All)', '\\checkmark (1000)', '\\checkmark (1.0)', '\\checkmark (0.5)', '\\checkmark (1.0)'),
        'abl_cloth_full_no_lbs': ('Full (No LBS)', '$\\times$', '\\checkmark (1.0)', '\\checkmark (0.5)', '\\checkmark (1.0)'),
        'abl_cloth_sim_only_no_lbs': ('Sim (No LBS)', '$\\times$', '\\checkmark (1.0)', '$\\times$', '$\\times$'),
        'abl_cloth_full_lbs_weak': ('Full (LBS weak)', '\\checkmark (100)', '\\checkmark (1.0)', '\\checkmark (0.5)', '\\checkmark (1.0)'),
    }
    
    print("\n" + "="*120)
    print("LATEX TABLE FORMAT")
    print("="*120)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Ablation study: Impact of different loss components}")
    print("\\label{tab:ablation}")
    print("\\resizebox{\\textwidth}{!}{%")
    print("\\begin{tabular}{lccccccc}")
    print("\\toprule")
    print("Method & LBS Reg. & $L_\\text{sim}$ & $L_\\text{ARAP}$ & $L_\\text{mask}$ & PSNR $\\uparrow$ & SSIM $\\uparrow$ & LPIPS $\\downarrow$ \\\\")
    print("\\midrule")
    
    sorted_configs = sorted(results.keys())
    for config in sorted_configs:
        if config not in config_display:
            continue
        
        display_name, lbs, sim, arap, mask = config_display[config]
        config_results = results[config]
        if not config_results:
            continue
        
        avg_psnr = sum(r['psnr'] for r in config_results if r['psnr'] is not None) / len([r for r in config_results if r['psnr'] is not None])
        avg_ssim = sum(r['ssim'] for r in config_results if r['ssim'] is not None) / len([r for r in config_results if r['ssim'] is not None])
        avg_lpips = sum(r['lpips'] for r in config_results if r['lpips'] is not None) / len([r for r in config_results if r['lpips'] is not None])
        
        # Bold baseline
        if config == 'abl_cloth_baseline':
            print(f"\\textbf{{{display_name}}} & {lbs} & {sim} & {arap} & {mask} & "
                  f"\\textbf{{{avg_psnr:.2f}}} & \\textbf{{{avg_ssim:.3f}}} & \\textbf{{{avg_lpips:.4f}}} \\\\")
        else:
            print(f"{display_name} & {lbs} & {sim} & {arap} & {mask} & "
                  f"{avg_psnr:.2f} & {avg_ssim:.3f} & {avg_lpips:.4f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")
    print("\\end{table}")
    print("="*120)


def main():
    parser = argparse.ArgumentParser(description='Extract ablation study metrics')
    parser.add_argument('--base-dir', type=str, 
                       default='output_ablation',
                       help='Base directory containing ablation runs')
    parser.add_argument('--latex', action='store_true',
                       help='Print LaTeX table format')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_dir):
        print(f"Error: Directory {args.base_dir} does not exist")
        return
    
    print(f"Searching for ablation runs in: {args.base_dir}")
    results = find_ablation_runs(args.base_dir)
    
    if not results:
        print("No ablation results found!")
        return
    
    print(f"\nFound results for {len(results)} configurations:")
    for config, runs in results.items():
        print(f"  - {config}: {len(runs)} run(s)")
    
    if args.latex:
        print_latex_table(results)
    else:
        print_table(results)
        
        # Also save detailed results
        print("\n" + "="*120)
        print("DETAILED RESULTS (for reference)")
        print("="*120)
        for config, runs in sorted(results.items()):
            print(f"\n{config}:")
            for i, run in enumerate(runs, 1):
                print(f"  Run {i} (iter {run['iteration']}):")
                print(f"    PSNR: {run['psnr']:.2f}, SSIM: {run['ssim']:.3f}, LPIPS: {run['lpips']:.4f}")


if __name__ == '__main__':
    main()
