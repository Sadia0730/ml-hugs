#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import glob
import json
import os
import subprocess
import sys
import time
import argparse
import torch
from loguru import logger
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
import torch.multiprocessing as mp

sys.path.append('.')

from hugs.trainer import GaussianTrainer
from hugs.utils.config import get_cfg_items
from hugs.cfg.config import cfg as default_cfg
from hugs.utils.general import safe_state, find_cfg_diff
from hugs.utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    check_gpu_health,
    get_rank,
    get_world_size,
    set_random_seeds
)

def get_logger(cfg, rank=-1, time_str=None):
    if rank != -1 and not is_main_process():
        return
        
    output_path = cfg.output_path
    if time_str is None:
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    mode = 'eval' if cfg.eval else 'train'
    
    # Handle sequence being a list - use first sequence for logging
    seq = cfg.dataset.seq[0] if isinstance(cfg.dataset.seq, (list, ListConfig)) else cfg.dataset.seq
    
    if cfg.mode in ['human', 'human_scene']:
        logdir = os.path.join(
            output_path, cfg.mode, cfg.dataset.name,
            seq, cfg.human.name, cfg.exp_name, 
            time_str,
        )
    else:
        logdir = os.path.join(
            output_path, cfg.mode, cfg.dataset.name,
            seq, cfg.exp_name,
            time_str,
        )
    cfg.logdir = logdir
    cfg.logdir_ckpt = os.path.join(logdir, 'ckpt')
    
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(cfg.logdir_ckpt, exist_ok=True)
    os.makedirs(os.path.join(logdir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(logdir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(logdir, 'anim'), exist_ok=True)
    os.makedirs(os.path.join(logdir, 'meshes'), exist_ok=True)
    
    logger.add(os.path.join(logdir, f'{mode}.log'), level='INFO')
    logger.info(f'Logging to {logdir}')
    logger.info(OmegaConf.to_yaml(cfg))
    
    with open(os.path.join(logdir, f'config_{mode}.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

def train_worker(rank, world_size, cfg, devices):
    try:
        logger.info(f"Starting worker {rank}/{world_size} with device mapping: {devices}")
        
        # Setup distributed training FIRST before setting device
        if world_size > 1:
            setup_distributed(rank, world_size, devices)
        
        # The device index after CUDA_VISIBLE_DEVICES remapping
        device_id = devices[rank]  # This should be rank after remapping
        torch.cuda.set_device(device_id)
        
        logger.info(f"Worker {rank} using CUDA device {device_id}")
        
        # Generate shared timestamp for all processes
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Setup logging only on main process, but ensure all processes have logdir
        if rank == 0:  # Use rank instead of is_main_process() for initial setup
            get_logger(cfg, rank, time_str)
        else:
            # Non-main processes need logdir set for saving images
            output_path = cfg.output_path
            seq = cfg.dataset.seq[0] if isinstance(cfg.dataset.seq, (list, ListConfig)) else cfg.dataset.seq
            
            if cfg.mode in ['human', 'human_scene']:
                logdir = os.path.join(
                    output_path, cfg.mode, cfg.dataset.name,
                    seq, cfg.human.name, cfg.exp_name, 
                    time_str,
                )
            else:
                logdir = os.path.join(
                    output_path, cfg.mode, cfg.dataset.name,
                    seq, cfg.exp_name,
                    time_str,
                )
            cfg.logdir = logdir
            cfg.logdir_ckpt = os.path.join(logdir, 'ckpt')
        
        # Add device information to config
        cfg.device = device_id
        cfg.rank = rank
        cfg.world_size = world_size
        cfg.gpu_mapping = {rank: device_id for rank in range(world_size)}  # Add GPU mapping to config
        
        # Create trainer
        trainer = GaussianTrainer(cfg)
        
        # Train and save checkpoint only if not in eval mode
        if not cfg.eval:
            trainer.train()
            trainer.save_ckpt()
        
        # Run evaluation
        trainer.validate()
        
        # Save results.json and run animation only on main process
        if rank == 0:
            # Save results.json
            mode = 'eval' if cfg.eval else 'train'
            with open(os.path.join(cfg.logdir, f'results_{mode}.json'), 'w') as f:
                json.dump(trainer.eval_metrics, f, indent=4)
                
            # Run animation
            if cfg.mode in ['human', 'human_scene']:
                trainer.animate()
                trainer.render_canonical(pose_type='a_pose')
                trainer.render_canonical(pose_type='da_pose')
        
        # Cleanup
        if world_size > 1:
            cleanup_distributed()
            
    except Exception as e:
        logger.error(f"Process {rank} failed with error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        if world_size > 1:
            cleanup_distributed()
        raise e

def validate_gpus(gpu_list):
    """Validate GPU availability and health"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    available_devices = list(range(torch.cuda.device_count()))
    logger.info(f"Available CUDA devices: {available_devices}")
    
    # Check if requested GPUs exist
    for device in gpu_list:
        if device not in available_devices:
            raise ValueError(f"GPU {device} not found. Available devices: {available_devices}")
    
    # Check GPU health
    healthy_gpus = []
    for device in gpu_list:
        try:
            torch.cuda.set_device(device)
            name = torch.cuda.get_device_name(device)
            
            # Test basic GPU operation
            test_tensor = torch.randn(100, 100, device=device)
            result = torch.matmul(test_tensor, test_tensor.T)
            del test_tensor, result
            torch.cuda.empty_cache()
            
            healthy_gpus.append(device)
            logger.info(f"GPU {device} ({name}) is healthy and available")
            
        except Exception as e:
            logger.error(f"GPU {device} failed health check: {e}")
            raise RuntimeError(f"GPU {device} is not accessible: {e}")
    
    return healthy_gpus

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cfg_files/release/neuman/hugs_human.yaml')
    parser.add_argument('--gpus', type=str, default='0', 
                       help='Comma-separated list of GPU devices to use (e.g., "0,2,3,4")')
    parser.add_argument('overrides', nargs='*', help='Additional configuration overrides in the format key=value')
    args = parser.parse_args()
    
    # Parse and validate GPU devices
    original_devices = [int(x.strip()) for x in args.gpus.split(',')]
    world_size = len(original_devices)
    
    logger.info(f"Requested GPUs: {original_devices}")
    logger.info(f"World size: {world_size}")
    
    # Validate GPUs before proceeding
    try:
        healthy_gpus = validate_gpus(original_devices)
    except Exception as e:
        logger.error(f"GPU validation failed: {e}")
        sys.exit(1)
    
    # Set environment variables for distributed training
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    logger.info(f"Set CUDA_VISIBLE_DEVICES={args.gpus}")
    
    # Environment variables for debugging and stability
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    
    # After setting CUDA_VISIBLE_DEVICES, devices are remapped to 0, 1, 2, ...
    remapped_devices = list(range(world_size))
    
    logger.info(f"Remapped devices after CUDA_VISIBLE_DEVICES: {remapped_devices}")
    
    # Load config
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(default_cfg, cfg)
    
    # Apply any additional overrides
    if args.overrides:
        overrides = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, overrides)
    
    logger.info(f"Starting training with {world_size} GPUs")
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Launch distributed training if multiple GPUs
    if world_size > 1:
        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        logger.info("Launching distributed training with torch.multiprocessing.spawn")
        mp.spawn(
            train_worker,
            args=(world_size, cfg, remapped_devices),
            nprocs=world_size,
            join=True
        )
    else:
        logger.info("Launching single-GPU training")
        train_worker(0, 1, cfg, remapped_devices)

if __name__ == '__main__':
    main()
            