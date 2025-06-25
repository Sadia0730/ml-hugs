import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger

def setup_distributed(rank, world_size, devices):
    """
    Setup distributed training
    
    Args:
        rank: The rank of current process
        world_size: Total number of processes
        devices: List of GPU device indices to use
    """
    # Set master address - use localhost for single-machine
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Get the actual GPU device for this rank
    device = devices[rank]
    
    logger.info(f"Initializing process {rank}/{world_size} on GPU {device}")
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(device)

def cleanup_distributed():
    """
    Clean up distributed training
    """
    if dist.is_initialized():
        dist.destroy_process_group()

def convert_model_to_ddp(model, device):
    """
    Convert a model to DDP
    """
    model = model.to(device)
    model = DDP(model, device_ids=[device])
    return model

def is_main_process():
    """
    Check if this is the main process in DDP
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def get_world_size():
    """
    Get the number of processes in the distributed training
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def synchronize():
    """
    Synchronize all processes
    """
    if not dist.is_initialized():
        return
    dist.barrier() 