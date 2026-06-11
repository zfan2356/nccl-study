"""
Ring buffer P2P experiment with multi-process cuMem (CUDA VMM).

This mirrors run_ring_buffer_p2p.py but uses cuMemCreate/Export/Import instead
of legacy cudaIpcMemHandle — the same path NCCL uses when NCCL_CUMEM_ENABLE=1.

Usage:
    torchrun --nproc_per_node=2 primitives/l20/python/run_ring_buffer_p2p_cumem.py [slot_mib] [num_slots] [total_mib]

Each rank controls one GPU. Rank 0 is sender, Rank 1 is receiver.
cuMem POSIX FDs are exchanged via Unix domain sockets (SCM_RIGHTS).
"""

import os
import sys

from launcher import init_dist, log
from cumem import CuMemBuffer, exchange_cumem_fds
import cumem_utils
import ipc_utils  # timing helpers only
import ring_buffer_p2p_ipc


def main():
    slot_mib = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    num_slots = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    total_mib = int(sys.argv[3]) if len(sys.argv) > 3 else 256

    slot_bytes = slot_mib * 1024 * 1024
    slot_count = slot_bytes // 4
    slot_bytes = slot_count * 4
    total_slots = (total_mib * 1024 * 1024) // slot_bytes
    if total_slots < 1:
        total_slots = 1
    total_count = total_slots * slot_count

    rank, world_size, local_rank = init_dist("gloo")
    assert world_size == 2, "This experiment requires exactly 2 ranks"

    if not cumem_utils.is_vmm_supported(local_rank):
        raise RuntimeError(
            f"cuMem VMM not supported on device {local_rank}. "
            "Requires CUDA driver >= 12.0 and VMM-capable GPU."
        )

    cumem_utils.set_device(local_rank)
    cumem_utils.enable_peer_access(local_rank, 1 - local_rank)

    log(rank, f"Device {local_rank} [cuMem], slot={slot_mib}MiB, "
        f"ring_depth={num_slots}, total={total_slots * slot_mib}MiB ({total_slots} slots)")

    seed = 0xDEAD0000
    local_buffers = {}

    if rank == 0:
        src_ptr, src_fd, src_aligned = cumem_utils.alloc_and_export_fd(total_count * 4, local_rank)
        head_ptr, head_fd, head_aligned = cumem_utils.alloc_and_export_fd(8, local_rank)
        local_buffers["head"] = CuMemBuffer(
            "head", head_ptr, head_fd, 8, head_aligned, local_rank
        )
        ring_buffer_p2p_ipc.fill_source(src_ptr, total_count, seed)
        log(rank, "Source buffer filled (cuMem)")
    else:
        recv_ptr, recv_fd, recv_aligned = cumem_utils.alloc_and_export_fd(
            num_slots * slot_count * 4, local_rank
        )
        tail_ptr, tail_fd, tail_aligned = cumem_utils.alloc_and_export_fd(8, local_rank)
        dst_ptr, dst_fd, dst_aligned = cumem_utils.alloc_and_export_fd(total_count * 4, local_rank)
        local_buffers["recv_buf"] = CuMemBuffer(
            "recv_buf", recv_ptr, recv_fd, num_slots * slot_count * 4, recv_aligned, local_rank
        )
        local_buffers["tail"] = CuMemBuffer(
            "tail", tail_ptr, tail_fd, 8, tail_aligned, local_rank
        )
        log(rank, "Receiver buffers allocated (cuMem)")

    remote_handles = exchange_cumem_fds(local_buffers, rank, world_size)
    for buf in local_buffers.values():
        os.close(buf.fd)
    log(rank, "cuMem FDs exchanged via UDS")

    if rank == 0:
        recv_info = remote_handles[1]["recv_buf"]
        tail_info = remote_handles[1]["tail"]
        remote_recv_ptr = cumem_utils.import_from_fd(
            recv_info["fd"], recv_info["aligned_size"], local_rank
        )
        remote_tail_ptr = cumem_utils.import_from_fd(
            tail_info["fd"], tail_info["aligned_size"], local_rank
        )
        os.close(recv_info["fd"])
        os.close(tail_info["fd"])
        log(rank, "Imported receiver cuMem buffers")
    else:
        head_info = remote_handles[0]["head"]
        remote_head_ptr = cumem_utils.import_from_fd(
            head_info["fd"], head_info["aligned_size"], local_rank
        )
        os.close(head_info["fd"])
        log(rank, "Imported sender cuMem buffer (head)")

    import torch.distributed as dist
    dist.barrier()

    # Warmup: run once to prime TLBs and page tables
    warmup_slots = min(num_slots, total_slots)
    log(rank, "Warmup...")
    if rank == 0:
        ring_buffer_p2p_ipc.run_sender(
            src_ptr, remote_recv_ptr, remote_tail_ptr, head_ptr,
            slot_count, num_slots, warmup_slots,
        )
        cumem_utils.synchronize()
    else:
        ring_buffer_p2p_ipc.run_receiver(
            dst_ptr, recv_ptr, tail_ptr, remote_head_ptr,
            slot_count, num_slots, warmup_slots,
        )
        cumem_utils.synchronize()

    dist.barrier()

    # Reset flags and buffers for timed run (each rank resets only its own local buffers)
    if rank == 0:
        ring_buffer_p2p_ipc.memset_buffer(head_ptr, 8)
    else:
        ring_buffer_p2p_ipc.memset_buffer(tail_ptr, 8)
        ring_buffer_p2p_ipc.memset_buffer(recv_ptr, num_slots * slot_count * 4)
        ring_buffer_p2p_ipc.memset_buffer(dst_ptr, total_count * 4)

    dist.barrier()
    log(rank, "Starting timed run")

    # Timed run
    if rank == 0:
        start_evt = ipc_utils.create_event()
        stop_evt = ipc_utils.create_event()

        ipc_utils.record_event(start_evt)
        ring_buffer_p2p_ipc.run_sender(
            src_ptr, remote_recv_ptr, remote_tail_ptr, head_ptr,
            slot_count, num_slots, total_slots,
        )
        ipc_utils.record_event(stop_evt)
        ipc_utils.sync_event(stop_evt)

        elapsed = ipc_utils.elapsed_ms(start_evt, stop_evt)
        total_bytes = total_count * 4
        gbps = total_bytes / (elapsed / 1000.0) / 1e9
        log(rank, f"Elapsed: {elapsed:.2f} ms, Bandwidth: {gbps:.2f} GB/s")

        ipc_utils.destroy_event(start_evt)
        ipc_utils.destroy_event(stop_evt)
    else:
        ring_buffer_p2p_ipc.run_receiver(
            dst_ptr, recv_ptr, tail_ptr, remote_head_ptr,
            slot_count, num_slots, total_slots,
        )
        cumem_utils.synchronize()

        ok = ring_buffer_p2p_ipc.verify(dst_ptr, total_count, seed)
        log(rank, f"Verification: {'PASS' if ok else 'FAIL'}")

    dist.barrier()
    if rank == 0:
        cumem_utils.close_imported(remote_recv_ptr, recv_info["aligned_size"])
        cumem_utils.close_imported(remote_tail_ptr, tail_info["aligned_size"])
        cumem_utils.free_local(src_ptr, src_aligned)
        cumem_utils.free_local(head_ptr, head_aligned)
    else:
        cumem_utils.close_imported(remote_head_ptr, head_info["aligned_size"])
        cumem_utils.free_local(recv_ptr, recv_aligned)
        cumem_utils.free_local(tail_ptr, tail_aligned)
        cumem_utils.free_local(dst_ptr, dst_aligned)

    dist.destroy_process_group()
    log(rank, "Done")


if __name__ == "__main__":
    main()
