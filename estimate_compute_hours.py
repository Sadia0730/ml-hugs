#!/usr/bin/env python3
"""
Estimate GPU+CPU compute hours from training logs or timing
"""
import os
import json
import glob
from datetime import datetime
import re

def parse_training_time_from_logs(log_file):
    """Parse training time from log files"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        start_time = None
        end_time = None
        
        for line in lines:
            # Look for timestamps in logs
            if 'Training' in line or 'Starting' in line:
                # Extract timestamp if available
                pass
        
        return start_time, end_time
    except:
        return None, None

def estimate_from_timestamps(output_dir):
    """Estimate training time from directory timestamps"""
    results = []
    
    # Find all experiment directories
    exp_dirs = glob.glob(os.path.join(output_dir, "**/results_train.json"), recursive=True)
    
    for json_file in exp_dirs:
        exp_dir = os.path.dirname(json_file)
        
        # Get creation and modification times
        try:
            stat_info = os.stat(json_file)
            creation_time = stat_info.st_ctime
            modification_time = stat_info.st_mtime
            
            # Training time in hours
            training_hours = (modification_time - creation_time) / 3600.0
            
            results.append({
                'experiment': exp_dir,
                'hours': training_hours
            })
        except:
            pass
    
    return results

def estimate_from_config():
    """Estimate based on configuration"""
    print("=" * 70)
    print("Compute Hour Estimation for Cloth-HUGS")
    print("=" * 70)
    
    # Training configuration
    num_iterations = 20000
    batch_size = 1
    
    # Typical iteration times for 3DGS-based methods
    # Based on 3DGS papers: ~100-200ms per iteration on modern GPUs
    seconds_per_iter = 0.15  # Conservative estimate
    
    estimated_seconds = num_iterations * seconds_per_iter
    estimated_hours = estimated_seconds / 3600.0
    
    print(f"\nBased on Configuration:")
    print(f"  Training iterations: {num_iterations:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Estimated time per iteration: {seconds_per_iter:.2f}s")
    print(f"  Estimated training time: {estimated_hours:.2f} hours ({estimated_hours * 60:.1f} minutes)")
    
    # Ablation studies
    num_ablations = 12  # From your ablation configs
    ablation_hours = estimated_hours * num_ablations
    
    print(f"\nAblation Studies:")
    print(f"  Number of ablation experiments: {num_ablations}")
    print(f"  Total ablation compute: {ablation_hours:.2f} hours")
    
    # Development and tuning (rough estimate)
    development_multiplier = 3  # Typical for research
    development_hours = estimated_hours * development_multiplier
    
    print(f"\nDevelopment & Hyperparameter Tuning:")
    print(f"  Estimated development runs: {development_multiplier}x main training")
    print(f"  Development compute: {development_hours:.2f} hours")
    
    # Total
    total_gpu_hours = estimated_hours + ablation_hours + development_hours
    
    print("\n" + "=" * 70)
    print("TOTAL ESTIMATED GPU HOURS")
    print("=" * 70)
    print(f"  Main training:       {estimated_hours:>8.2f} hours")
    print(f"  Ablation studies:    {ablation_hours:>8.2f} hours")
    print(f"  Development/tuning:  {development_hours:>8.2f} hours")
    print(f"  " + "-" * 66)
    print(f"  TOTAL:               {total_gpu_hours:>8.2f} hours")
    
    print("\n" + "=" * 70)
    print(f"FOR COMPUTE REPORT: {total_gpu_hours:.0f} GPU hours")
    print("=" * 70)
    
    return total_gpu_hours

def check_actual_training_logs():
    """Check actual training logs if available"""
    print("\n" + "=" * 70)
    print("Checking Actual Training Logs")
    print("=" * 70)
    
    output_dirs = [
        '/home/ubx858/ml-hugs/output',
        '/home/ubx858/ml-hugs/output_ablation'
    ]
    
    total_experiments = 0
    
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            results = estimate_from_timestamps(output_dir)
            if results:
                print(f"\nFound {len(results)} experiments in {output_dir}:")
                total_hours = 0
                for r in results[:5]:  # Show first 5
                    exp_name = os.path.basename(r['experiment'])
                    print(f"  {exp_name}: {r['hours']:.2f} hours")
                    total_hours += r['hours']
                
                if len(results) > 5:
                    remaining_hours = sum(r['hours'] for r in results[5:])
                    total_hours += remaining_hours
                    print(f"  ... and {len(results) - 5} more experiments: {remaining_hours:.2f} hours")
                
                print(f"  Subtotal: {total_hours:.2f} hours")
                total_experiments += len(results)
    
    if total_experiments == 0:
        print("\nNo training logs found. Using estimated values.")
        return None
    
    return total_experiments

if __name__ == "__main__":
    # Check for actual logs
    actual_count = check_actual_training_logs()
    
    # Provide estimate
    print("\n")
    estimate_from_config()
    
    print("\n" + "=" * 70)
    print("NOTES:")
    print("=" * 70)
    print("1. GPU hours include single GPU training time")
    print("2. CPU hours are typically minimal for GPU-accelerated training")
    print("3. For the report, you can combine as 'GPU+CPU Hours'")
    print("4. If you have exact timing logs, update the values accordingly")
    print("=" * 70)

