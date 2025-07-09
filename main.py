#!/usr/bin/env python
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import os
import glob
import shutil
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
    get_rank,
    get_world_size,
    set_random_seeds
)

# ----------------------------------------------------------------------------
# Logger setup (copied from your previous get_logger)
# ----------------------------------------------------------------------------
def get_logger(cfg, rank=-1, time_str=None):
    if rank != -1 and not is_main_process():
        return

    output_path = cfg.output_path
    if time_str is None:
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    mode = 'eval' if cfg.eval else 'train'

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
    os.makedirs(cfg.logdir, exist_ok=True)
    os.makedirs(cfg.logdir_ckpt, exist_ok=True)
    for sub in ['val', 'train', 'anim', 'meshes']:
        os.makedirs(os.path.join(logdir, sub), exist_ok=True)

    logger.add(os.path.join(logdir, f'{mode}.log'), level='INFO')
    logger.info(f'Logging to {logdir}')
    logger.info(OmegaConf.to_yaml(cfg))

    with open(os.path.join(logdir, f'config_{mode}.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

# ----------------------------------------------------------------------------
# GPU validation helper
# ----------------------------------------------------------------------------
def validate_gpus(gpu_list):
    """Validate GPU availability and health"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    available = list(range(torch.cuda.device_count()))
    for d in gpu_list:
        if d not in available:
            raise ValueError(f"GPU {d} not found; available: {available}")
    print(f"Using GPUs {gpu_list}")

# ----------------------------------------------------------------------------
# Worker function: sets up DDP and runs training
# ----------------------------------------------------------------------------
def train_worker(rank, world_size, cfg, devices):
    try:
        # Initialize distributed once
        if world_size > 1:
            setup_distributed(rank, world_size, devices)

        # Bind this process to its GPU
        device_id = devices[rank]
        torch.cuda.set_device(device_id)

        # Create log dir
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        get_logger(cfg, rank, time_str)

        # Attach distributed info
        cfg.device      = device_id
        cfg.rank        = rank
        cfg.world_size  = world_size
        cfg.gpu_mapping = {r: devices[r] for r in range(world_size)}

        # Train
        trainer = GaussianTrainer(cfg)
        trainer.train()

    except Exception as e:
        logger.error(f"Worker {rank} failed: {e}")
        raise

    finally:
        if world_size > 1:
            cleanup_distributed()

# ----------------------------------------------------------------------------
# Top-level entrypoint for each spawned process
# ----------------------------------------------------------------------------
def _entry(rank, world_size, gpu_ids, args):
    # Only set the device here; train_worker will init DDP
    torch.cuda.set_device(gpu_ids[rank])

    # Load & merge config
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(default_cfg, cfg)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    # Run the worker
    train_worker(rank, world_size, cfg, gpu_ids)

# ----------------------------------------------------------------------------
# Main process: parse args and spawn processes
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='cfg_files/release/neuman/hugs_human.yaml')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Comma-separated list of GPU IDs')
    parser.add_argument('overrides', nargs='*',
                        help='Config overrides key=value')
    args = parser.parse_args()

    gpu_ids = list(map(int, args.gpus.split(',')))
    world_size = len(gpu_ids)

    validate_gpus(gpu_ids)
    set_random_seeds(42)

    if world_size > 1:
        mp.spawn(_entry,
                 args=(world_size, gpu_ids, args),
                 nprocs=world_size,
                 join=True)
    else:
        _entry(0, 1, gpu_ids, args)

if __name__ == '__main__':
    main()
