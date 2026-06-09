"""
Ring buffer P2P experiment with multi-process IPC.

Usage:
    torchrun --nproc_per_node=2 primitives/l20/python/run_ring_buffer_p2p.py [slot_mib] [num_slots] [total_mib]

Each rank controls one GPU. Rank 0 is sender, Rank 1 is receiver.
IPC handles are exchanged via torch.distributed (gloo backend).
"""

import sys

from launcher import init_dist, log
from ipc import IpcBuffer, exchange_ipc_handles
import ipc_utils
import ring_buffer_p2p_ipc


def main():
    # Parse args
    slot_mib = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    num_slots = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    total_mib = int(sys.argv[3]) if len(sys.argv) > 3 else 256

    slot_bytes = slot_mib * 1024 * 1024
    slot_count = slot_bytes // 4  # uint32_t elements per slot
    slot_bytes = slot_count * 4
    total_slots = (total_mib * 1024 * 1024) // slot_bytes
    if total_slots < 1:
        total_slots = 1
    total_count = total_slots * slot_count

    # Initialize distributed
    rank, world_size, local_rank = init_dist("gloo")
    assert world_size == 2, "This experiment requires exactly 2 ranks"

    ipc_utils.set_device(local_rank)
    log(rank, f"Device {local_rank}, slot={slot_mib}MiB, ring_depth={num_slots}, total={total_slots * slot_mib}MiB ({total_slots} slots)")

    seed = 0xDEAD0000

    # Allocate buffers per rank
    local_buffers = {}

    if rank == 0:
        # Sender: allocate srcBuf (local) and head (local, receiver will remote-write)
        src_ptr, src_handle = ipc_utils.alloc_and_export(total_count * 4)
        head_ptr, head_handle = ipc_utils.alloc_and_export(8)  # uint64_t
        local_buffers["head"] = IpcBuffer("head", head_ptr, head_handle, 8, local_rank)
        # Fill source data
        ring_buffer_p2p_ipc.fill_source(src_ptr, total_count, seed)
        log(rank, "Source buffer filled")
    else:
        # Receiver: allocate recvBuf (local ring buffer), tail (local), dstBuf (for verify)
        recv_ptr, recv_handle = ipc_utils.alloc_and_export(num_slots * slot_count * 4)
        tail_ptr, tail_handle = ipc_utils.alloc_and_export(8)  # uint64_t
        dst_ptr, dst_handle = ipc_utils.alloc_and_export(total_count * 4)
        local_buffers["recv_buf"] = IpcBuffer("recv_buf", recv_ptr, recv_handle, num_slots * slot_count * 4, local_rank)
        local_buffers["tail"] = IpcBuffer("tail", tail_ptr, tail_handle, 8, local_rank)
        log(rank, "Receiver buffers allocated")

    # Exchange IPC handles
    remote_handles = exchange_ipc_handles(local_buffers, rank, world_size)
    log(rank, "IPC handles exchanged")

    # Open remote handles
    if rank == 0:
        # Sender needs: remote recvBuf, remote tail
        remote_recv_ptr = ipc_utils.open_ipc(remote_handles[1]["recv_buf"])
        remote_tail_ptr = ipc_utils.open_ipc(remote_handles[1]["tail"])
        log(rank, "Opened receiver's IPC handles")
    else:
        # Receiver needs: remote head
        remote_head_ptr = ipc_utils.open_ipc(remote_handles[0]["head"])
        log(rank, "Opened sender's IPC handle")

    # Synchronize before launching kernels
    import torch.distributed as dist
    dist.barrier()
    log(rank, "Starting kernels")

    # Both ranks launch their kernel, then synchronize.
    # Receiver must be launched first (it spin-waits on tail).
    # We use dist.barrier() to ensure both processes are ready before launching.
    if rank == 0:
        start_evt = ipc_utils.create_event()
        stop_evt = ipc_utils.create_event()

        ipc_utils.record_event(start_evt)
        ring_buffer_p2p_ipc.run_sender(src_ptr, remote_recv_ptr, remote_tail_ptr, head_ptr,
                                       slot_count, num_slots, total_slots)
        ipc_utils.record_event(stop_evt)
        ipc_utils.sync_event(stop_evt)

        elapsed = ipc_utils.elapsed_ms(start_evt, stop_evt)
        total_bytes = total_count * 4
        gbps = total_bytes / (elapsed / 1000.0) / 1e9
        log(rank, f"Elapsed: {elapsed:.2f} ms, Bandwidth: {gbps:.2f} GB/s")

        ipc_utils.destroy_event(start_evt)
        ipc_utils.destroy_event(stop_evt)
    else:
        ring_buffer_p2p_ipc.run_receiver(dst_ptr, recv_ptr, tail_ptr, remote_head_ptr,
                                         slot_count, num_slots, total_slots)
        ipc_utils.synchronize()

        # Verify
        ok = ring_buffer_p2p_ipc.verify(dst_ptr, total_count, seed)
        log(rank, f"Verification: {'PASS' if ok else 'FAIL'}")

    # Cleanup
    dist.barrier()
    if rank == 0:
        ipc_utils.close_ipc(remote_recv_ptr)
        ipc_utils.close_ipc(remote_tail_ptr)
        ipc_utils.free_buffer(src_ptr)
        ipc_utils.free_buffer(head_ptr)
    else:
        ipc_utils.close_ipc(remote_head_ptr)
        ipc_utils.free_buffer(recv_ptr)
        ipc_utils.free_buffer(tail_ptr)
        ipc_utils.free_buffer(dst_ptr)

    dist.destroy_process_group()
    log(rank, "Done")


if __name__ == "__main__":
    main()
