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
import torch.multiprocessing as mp

sys.path.append('.')

from hugs.trainer import GaussianTrainer
from hugs.utils.config import get_cfg_items
from hugs.cfg.config import cfg as default_cfg
from hugs.utils.general import safe_state, find_cfg_diff
from hugs.utils.distributed import setup_distributed, cleanup_distributed, is_main_process

def get_logger(cfg, rank=-1):
    if rank != -1 and not is_main_process():
        return
        
    output_path = cfg.output_path
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    mode = 'eval' if cfg.eval else 'train'
    
    if cfg.mode in ['human', 'human_scene']:
        logdir = os.path.join(
            output_path, cfg.mode, cfg.dataset.name,
            cfg.dataset.seq, cfg.human.name, cfg.exp_name, 
            time_str,
        )
    else:
        logdir = os.path.join(
            output_path, cfg.mode, cfg.dataset.name,
            cfg.dataset.seq, cfg.exp_name,
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
    # Setup distributed training
    if world_size > 1:
        setup_distributed(rank, world_size, devices)
        
    # Setup logging only on main process
    get_logger(cfg, rank)
    
    # Create trainer
    trainer = GaussianTrainer(cfg)
    
    # Train
    trainer.train()
    
    # Cleanup
    if world_size > 1:
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cfg_files/release/neuman/hugs_human.yaml')
    parser.add_argument('--gpus', type=str, default='0', 
                       help='Comma-separated list of GPU devices to use (e.g., "0,2,3,4")')
    args = parser.parse_args()
    
    # Parse GPU devices
    devices = [int(x.strip()) for x in args.gpus.split(',')]
    world_size = len(devices)
    
    # Validate devices
    available_devices = list(range(torch.cuda.device_count()))
    for device in devices:
        if device not in available_devices:
            raise ValueError(f"GPU {device} not found. Available devices: {available_devices}")
    
    # Load config
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(default_cfg, cfg)
    
    # Launch distributed training if multiple GPUs
    if world_size > 1:
        mp.spawn(
            train_worker,
            args=(world_size, cfg, devices),
            nprocs=world_size,
            join=True
        )
    else:
        train_worker(0, 1, cfg, devices)

if __name__ == '__main__':
    main()
            