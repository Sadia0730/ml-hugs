#!/usr/bin/env python3
import json
import sys

def find_worst_iteration(json_file, target_psnr, target_ssim, target_lpips):
    """Find the worst iteration based on target metrics."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Remove "final" entry if present, as it's just a copy of the last iteration
    iterations = {k: v for k, v in data.items() if k != "final"}
    
    worst_score = float('-inf')
    worst_iter = None
    worst_details = {}
    
    print(f"Target metrics:")
    print(f"  Human PSNR: {target_psnr}")
    print(f"  Human SSIM: {target_ssim}")
    print(f"  Human LPIPS: {target_lpips}")
    print("\nAnalyzing iterations...")
    print("-" * 80)
    
    results = []
    worse_than_all = []  # Iterations that are worse than ALL three targets
    
    for iter_name, metrics in sorted(iterations.items()):
        psnr = metrics.get("hugs_human_psnr", 0)
        ssim = metrics.get("hugs_human_ssim", 0)
        lpips = metrics.get("hugs_human_lpips", 0)
        
        # Check if this iteration is worse than ALL three targets
        is_worse_psnr = psnr < target_psnr  # Lower PSNR is worse
        is_worse_ssim = ssim < target_ssim  # Lower SSIM is worse
        is_worse_lpips = lpips > target_lpips  # Higher LPIPS is worse
        
        is_worse_than_all = is_worse_psnr and is_worse_ssim and is_worse_lpips
        
        # Calculate deviations (higher is worse)
        # PSNR: lower is worse, so penalty = (target - actual) if actual < target
        psnr_penalty = max(0, target_psnr - psnr)
        
        # SSIM: lower is worse, so penalty = (target - actual) if actual < target
        ssim_penalty = max(0, target_ssim - ssim)
        
        # LPIPS: higher is worse, so penalty = (actual - target) if actual > target
        lpips_penalty = max(0, lpips - target_lpips)
        
        # Combined worstness score (weighted sum)
        # Normalize by relative importance (PSNR and SSIM typically in different scales)
        # Using simple weighted sum
        worstness = (psnr_penalty * 1.0) + (ssim_penalty * 10.0) + (lpips_penalty * 5.0)
        
        result = {
            'iteration': iter_name,
            'psnr': psnr,
            'ssim': ssim,
            'lpips': lpips,
            'psnr_penalty': psnr_penalty,
            'ssim_penalty': ssim_penalty,
            'lpips_penalty': lpips_penalty,
            'worstness': worstness,
            'is_worse_than_all': is_worse_than_all
        }
        
        results.append(result)
        
        if is_worse_than_all:
            worse_than_all.append(result)
        
        if worstness > worst_score:
            worst_score = worstness
            worst_iter = iter_name
            worst_details = {
                'psnr': psnr,
                'ssim': ssim,
                'lpips': lpips,
                'psnr_penalty': psnr_penalty,
                'ssim_penalty': ssim_penalty,
                'lpips_penalty': lpips_penalty,
                'is_worse_than_all': is_worse_than_all
            }
    
    # Filter and sort: Only consider iterations worse than ALL targets
    worse_than_all.sort(key=lambda x: x['worstness'], reverse=True)
    
    if len(worse_than_all) == 0:
        print("\n" + "=" * 80)
        print("NO ITERATIONS FOUND that are worse than ALL three targets!")
        print("=" * 80)
        print("\nIterations worse than individual targets:")
        worse_psnr = [r for r in results if r['psnr'] < target_psnr]
        worse_ssim = [r for r in results if r['ssim'] < target_ssim]
        worse_lpips = [r for r in results if r['lpips'] > target_lpips]
        print(f"  - Worse PSNR (< {target_psnr}): {len(worse_psnr)} iterations")
        print(f"  - Worse SSIM (< {target_ssim}): {len(worse_ssim)} iterations")
        print(f"  - Worse LPIPS (> {target_lpips}): {len(worse_lpips)} iterations")
        return None, None
    
    print(f"\nFound {len(worse_than_all)} iterations worse than ALL three targets:")
    print("=" * 80)
    print(f"{'Iteration':<12} {'PSNR':<10} {'< Target':<10} {'SSIM':<10} {'< Target':<10} {'LPIPS':<10} {'> Target':<10}")
    print("-" * 80)
    
    for r in worse_than_all:
        psnr_status = "✗" if r['psnr'] < target_psnr else "✓"
        ssim_status = "✗" if r['ssim'] < target_ssim else "✓"
        lpips_status = "✗" if r['lpips'] > target_lpips else "✓"
        print(f"{r['iteration']:<12} {r['psnr']:<10.3f} {psnr_status:<10} {r['ssim']:<10.3f} {ssim_status:<10} {r['lpips']:<10.3f} {lpips_status:<10}")
    
    worst_iter = worse_than_all[0]['iteration']
    worst_details = worse_than_all[0]
    
    print("\n" + "=" * 80)
    print(f"WORST ITERATION (worse than all targets): {worst_iter}")
    print("=" * 80)
    print(f"Human PSNR:  {worst_details['psnr']:.3f} (target: {target_psnr}, worse: {worst_details['psnr'] < target_psnr})")
    print(f"Human SSIM:  {worst_details['ssim']:.3f} (target: {target_ssim}, worse: {worst_details['ssim'] < target_ssim})")
    print(f"Human LPIPS: {worst_details['lpips']:.3f} (target: {target_lpips}, worse: {worst_details['lpips'] > target_lpips})")
    print(f"Combined Worstness Score: {worst_details['worstness']:.3f}")
    
    # Also find worst for each metric individually
    print("\n" + "=" * 80)
    print("Individual Worst Metrics:")
    print("=" * 80)
    
    worst_psnr_iter = min(iterations.items(), key=lambda x: x[1].get("hugs_human_psnr", float('inf')))
    worst_ssim_iter = min(iterations.items(), key=lambda x: x[1].get("hugs_human_ssim", float('inf')))
    worst_lpips_iter = max(iterations.items(), key=lambda x: x[1].get("hugs_human_lpips", float('-inf')))
    
    print(f"Lowest PSNR:  {worst_psnr_iter[0]} -> {worst_psnr_iter[1].get('hugs_human_psnr', 0):.3f}")
    print(f"Lowest SSIM:  {worst_ssim_iter[0]} -> {worst_ssim_iter[1].get('hugs_human_ssim', 0):.3f}")
    print(f"Highest LPIPS: {worst_lpips_iter[0]} -> {worst_lpips_iter[1].get('hugs_human_lpips', 0):.3f}")
    
    return worst_iter, worst_details

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_worst_iteration.py <results_json_file> [target_psnr] [target_ssim] [target_lpips]")
        print("Example: python find_worst_iteration.py results_train.json 19.48 0.66 0.15")
        sys.exit(1)
    
    json_file = sys.argv[1]
    target_psnr = float(sys.argv[2]) if len(sys.argv) > 2 else 19.48
    target_ssim = float(sys.argv[3]) if len(sys.argv) > 3 else 0.66
    target_lpips = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
    
    find_worst_iteration(json_file, target_psnr, target_ssim, target_lpips)
