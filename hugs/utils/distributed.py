import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
import datetime
import numpy as np
import random

def setup_distributed(rank, world_size, devices):
    """
    Setup distributed training
    
    Args:
        rank: The rank of current process
        world_size: Total number of processes
        devices: List of remapped GPU device indices to use (after CUDA_VISIBLE_DEVICES is set)
    """
    try:
        # Set master address - use localhost for single-machine
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Get the actual GPU device for this rank - now using remapped index
        device = devices[rank]  # This is already the remapped index
        
        # Verify CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your installation")
            
        # Verify the requested GPU exists in the remapped space
        if device >= torch.cuda.device_count():
            raise ValueError(f"Remapped GPU {device} not found. Available devices after remapping: {list(range(torch.cuda.device_count()))}")
            
        # Set CUDA device before anything else
        torch.cuda.set_device(device)
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        logger.info(f"Initializing process {rank}/{world_size} on remapped GPU {device}")
        
        # Initialize process group with timeout and specific backend options
        timeout = datetime.timedelta(seconds=1800)  # 30 minutes timeout
        backend_opts = {}
        
        # Try NCCL first, fallback to Gloo if it fails
        backend = "nccl"
        try:
            if not dist.is_nccl_available():
                backend = "gloo"
                logger.warning("NCCL not available, using Gloo backend")
        except:
            backend = "gloo"
            logger.warning("Error checking NCCL, using Gloo backend")
            
        # Force Gloo if environment variable is set
        if os.environ.get('FORCE_GLOO', '0') == '1':
            backend = "gloo"
            logger.info("Forcing Gloo backend due to FORCE_GLOO=1")
        
        # Initialize the process group
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timeout
        )
        
        # Verify the initialization
        if not dist.is_initialized():
            raise RuntimeError("Failed to initialize process group")
            
        logger.info(f"Process {rank} initialized successfully on remapped GPU {device} with {backend} backend")
        
    except Exception as e:
        logger.error(f"Failed to setup distributed process {rank}: {str(e)}")
        raise e

def cleanup_distributed():
    """
    Clean up distributed training
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed")

def init_distributed(rank, world_size, device):
    """Initialize distributed training environment (legacy compatibility function)."""
    if world_size > 1:
        # Create device list for setup_distributed
        devices = [device]  # For single device per process
        setup_distributed(rank, world_size, devices)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def convert_model_to_ddp(model, device):
    """
    Convert a model to DistributedDataParallel
    
    Args:
        model: The model to convert to DDP
        device: The remapped device index (after CUDA_VISIBLE_DEVICES is set)
    """
    if dist.is_initialized():
        try:
            model = DDP(model, device_ids=[device], find_unused_parameters=True)
            logger.info(f"Model converted to DDP on device {device}")
        except Exception as e:
            logger.error(f"Failed to convert model to DDP: {e}")
            logger.warning("Falling back to single-GPU mode")
            return model
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

def get_rank():
    """
    Get the rank of current process
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def synchronize():
    """
    Synchronize all processes
    """
    if not dist.is_initialized():
        return
    dist.barrier()

def check_gpu_health():
    """Check health of all GPUs"""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available")
        return []
    
    healthy_gpus = []
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.set_device(i)
            name = torch.cuda.get_device_name(i)
            
            # Test basic operations
            test_tensor = torch.randn(1000, 1000, device=i)
            result = torch.matmul(test_tensor, test_tensor.T)
            del test_tensor, result
            torch.cuda.empty_cache()
            
            healthy_gpus.append(i)
            logger.info(f"GPU {i} ({name}) is healthy")
            
        except Exception as e:
            logger.error(f"GPU {i} failed health check: {e}")
    
    return healthy_gpus

def set_random_seeds(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False