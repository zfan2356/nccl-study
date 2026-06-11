"""cuMem (CUDA VMM) handle exchange utilities.

Uses POSIX file descriptors from cuMemExportToShareableHandle, passed between
processes via Unix domain sockets (SCM_RIGHTS). This mirrors NCCL's P2P_CUMEM
path in third-party/nccl/src/transport/p2p.cc.
"""

from dataclasses import dataclass
import os
import socket
import struct
import time
from typing import Dict, List

import torch.distributed as dist


@dataclass
class CuMemBuffer:
    """Represents a cuMem-allocated CUDA buffer exported as a POSIX FD."""
    name: str
    ptr: int
    fd: int
    size: int
    aligned_size: int
    device: int


def _send_fd(sock: socket.socket, fd: int) -> None:
    sock.sendmsg([b"x"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", fd))])


def _recv_fd(sock: socket.socket) -> int:
    _msg, ancdata, _flags, _addr = sock.recvmsg(1, socket.CMSG_SPACE(struct.calcsize("i")))
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            return struct.unpack("i", cmsg_data)[0]
    raise RuntimeError("Failed to receive file descriptor over UDS")


def _socket_path() -> str:
    port = os.environ.get("MASTER_PORT", "29500")
    return f"/tmp/nccl_study_cumem_{port}.sock"


def exchange_cumem_fds(
    local_buffers: Dict[str, CuMemBuffer],
    rank: int,
    world_size: int,
) -> Dict[int, Dict[str, int]]:
    """Exchange cuMem export FDs between all ranks via Unix domain sockets.

    Rank 0 acts as the UDS server. To avoid send/recv deadlocks on a single
    connection, rank 0 sends first then receives; other ranks receive first
    then send.

    Args:
        local_buffers: buffers to export from this rank (name -> CuMemBuffer)
        rank: current rank
        world_size: total ranks

    Returns:
        dict of remote_rank -> {buffer_name: import_fd} for ranks != self
    """
    if world_size < 2:
        return {}

    sock_path = _socket_path()
    send_fds = [buf.fd for buf in local_buffers.values()]

    all_counts = _gather_buffer_counts(local_buffers, rank, world_size)
    recv_count = sum(all_counts[r] for r in range(world_size) if r != rank)

    server = None
    if rank == 0:
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(sock_path)
        server.listen(world_size - 1)

    dist.barrier()

    if rank == 0:
        conn, _ = server.accept()

        for fd in send_fds:
            _send_fd(conn, fd)

        remote_fds = [_recv_fd(conn) for _ in range(recv_count)]

        conn.close()
        server.close()
        os.unlink(sock_path)
    else:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        for _ in range(100):
            try:
                client.connect(sock_path)
                break
            except ConnectionRefusedError:
                time.sleep(0.05)
        else:
            raise RuntimeError(f"cuMem UDS server not ready: {sock_path}")

        remote_fds = [_recv_fd(client) for _ in range(recv_count)]

        for fd in send_fds:
            _send_fd(client, fd)

        client.close()

    dist.barrier()

    return _assign_remote_fds(local_buffers, remote_fds, rank, world_size)


def _gather_buffer_counts(
    local_buffers: Dict[str, CuMemBuffer],
    rank: int,
    world_size: int,
) -> Dict[int, int]:
    """Gather {rank: num_buffers} from all ranks."""
    local_count = len(local_buffers)
    all_counts: List = [None] * world_size
    dist.all_gather_object(all_counts, local_count)
    return {r: all_counts[r] for r in range(world_size)}


def _assign_remote_fds(
    local_buffers: Dict[str, CuMemBuffer],
    remote_fds: List[int],
    rank: int,
    world_size: int,
) -> Dict[int, Dict[str, int]]:
    """Map received FDs to remote rank + buffer name using metadata from all_gather."""
    local_meta = {name: {"size": buf.size, "aligned_size": buf.aligned_size, "device": buf.device}
                  for name, buf in local_buffers.items()}
    all_meta: List = [None] * world_size
    dist.all_gather_object(all_meta, local_meta)

    result: Dict[int, Dict[str, int]] = {}
    fd_idx = 0
    for r in range(world_size):
        if r == rank:
            continue
        result[r] = {}
        for name, meta in all_meta[r].items():
            result[r][name] = {
                "fd": remote_fds[fd_idx],
                "size": meta["size"],
                "aligned_size": meta["aligned_size"],
                "device": meta["device"],
            }
            fd_idx += 1

    if fd_idx != len(remote_fds):
        raise RuntimeError(
            f"FD count mismatch: assigned {fd_idx}, received {len(remote_fds)}"
        )

    return result
