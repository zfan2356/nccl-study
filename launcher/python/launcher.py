"""Common multi-process launcher utilities for nccl-study experiments."""

import os
import torch
import torch.distributed as dist


def init_dist(backend: str = "gloo"):
    """Initialize torch.distributed and set the CUDA device for this rank.

    Uses gloo backend by default to avoid conflicting with our custom
    GPU communication kernels.

    Returns (rank, world_size, local_rank).
    """
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def get_rank_info():
    """Return (rank, world_size, local_rank) after init."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world_size, local_rank


def log(rank: int, msg: str):
    """Print a message prefixed with rank."""
    print(f"[Rank {rank}] {msg}", flush=True)
