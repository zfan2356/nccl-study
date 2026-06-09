"""IPC handle exchange utilities using torch.distributed."""

from dataclasses import dataclass, field
from typing import Dict, List

import torch.distributed as dist


@dataclass
class IpcBuffer:
    """Represents a CUDA buffer with its IPC handle."""
    name: str
    ptr: int  # device pointer as integer
    handle: bytes  # cudaIpcMemHandle_t as bytes
    size: int
    device: int


def exchange_ipc_handles(
    local_buffers: Dict[str, IpcBuffer],
    rank: int,
    world_size: int,
) -> Dict[int, Dict[str, bytes]]:
    """Exchange IPC handles between all ranks.

    Args:
        local_buffers: dict of name -> IpcBuffer for this rank's buffers
        rank: current rank
        world_size: total number of ranks

    Returns:
        dict of rank -> {name: handle_bytes} for all ranks
    """
    # Prepare local handle info
    local_info = {name: buf.handle for name, buf in local_buffers.items()}

    # Gather from all ranks
    all_infos: List = [None] * world_size
    dist.all_gather_object(all_infos, local_info)

    # Convert to dict keyed by rank
    result = {}
    for r in range(world_size):
        if r != rank:
            result[r] = all_infos[r]

    return result
