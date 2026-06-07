# ring_buffer_p2p_2node Design

## Overview

This experiment implements a minimal version of NCCL's Simple protocol: two persistent kernels (sender on GPU0, receiver on GPU1) communicate through a ring buffer with head/tail flag synchronization.

## Protocol

```
GPU0 (sender kernel)                  GPU1 (receiver kernel)
┌────────────────────┐               ┌────────────────────┐
│ 1. read local src  │               │ 1. poll tail       │
│ 2. write to remote │──peer write──→│    (spin until     │
│    recvBuf on GPU1 │               │     data ready)    │
│ 3. __threadfence   │               │ 2. read local      │
│    _system()       │               │    recvBuf         │
│ 4. update tail     │──peer write──→│ 3. copy to dstBuf  │
│                    │               │ 4. __threadfence   │
│ 5. poll head       │←─peer write───│    _system()       │
│    (flow control)  │               │ 5. update head     │
└────────────────────┘               └────────────────────┘
```

## Memory Layout

- `srcBuf`: allocated on GPU0, holds all source data
- `recvBuf`: allocated on GPU1, ring buffer with `numSlots` slots (sender remote-writes here)
- `dstBuf`: allocated on GPU1, receiver copies here for verification
- `tail` (uint64): allocated on GPU1, sender remote-writes to signal data ready
- `head` (uint64): allocated on GPU0, receiver remote-writes to signal slot consumed

## Key Principles

- **Remote write, local read**: sender writes to receiver's memory via peer access; receiver reads locally. Remote writes hide latency better than remote reads (fire-and-forget).
- **`__threadfence_system()`** (PTX: `membar.sys`): ensures all preceding writes are visible to other GPUs before the flag is updated. Required because the observer is on a different GPU; `membar.gl` would only guarantee visibility within the same GPU.
- **Ring buffer flow control**: sender cannot advance more than `numSlots` ahead of receiver (prevents overwriting unread data).
- **Flag placement ("remote write, local read")**: `tail` is on GPU1 (receiver polls locally, sender remote-writes); `head` is on GPU0 (sender polls locally, receiver remote-writes). High-frequency polling must be local (~tens of ns); low-frequency updates can be remote writes.

## Why Use Remote Write + Ring Buffer (Not Remote Read)

For pure data transfer, SM remote read (12.7 GB/s) is actually faster than remote write (10.6 GB/s), and `cudaMemcpyPeerAsync` (26 GB/s) beats both. So the ring buffer + remote write pattern is **not** optimized for simple P2P copy.

It exists for the **reduce** scenario (AllReduce, ReduceScatter):

```
Remote read approach (receiver remote-reads from sender):
  result = remote_load(sender_buf[i]) + local_load(my_buf[i])
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~~~~~~
           slow (~12.7 GB/s)              fast (local)
  → reduce throughput bottlenecked by remote read latency

Remote write approach (sender writes to receiver's local buffer):
  result = local_load(recvBuf[i]) + local_load(my_buf[i])
           ~~~~~~~~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~~~~~~
           fast (local)              fast (local)
  → reduce is all local reads, maximum compute throughput
```

The remote write cost is paid by the sender (fire-and-forget, doesn't block), while the receiver performs reduce on two local operands at full memory bandwidth. This is why NCCL uses the "sender remote-writes, receiver reads locally" pattern -- it decouples the data transfer latency from the reduce computation.

| Scenario | Best approach |
|----------|--------------|
| Pure data transfer | CE (`cudaMemcpyPeerAsync`) or SM remote read |
| Collective with reduce | Remote write + local read + local reduce (this pattern) |

## Optimization Process

Starting from a naive implementation to the final version:

### V1: 256 threads, 32-bit stores

- Single persistent thread block with 256 threads
- Each thread copies `uint32_t` (4 bytes) per iteration
- Result: **1.8 GB/s**

Problem: too few threads, each issuing narrow (32-bit) remote stores. The outstanding write count is far too low to fill the PCIe pipeline. By Little's Law: `BW = outstanding_requests * request_size / latency`. With ~256 concurrent 4B writes and ~2us PCIe round-trip, maximum achievable BW is severely limited.

### V2: 1024 threads, 32-bit stores

- Increased to 1024 threads per block
- 4x more concurrent outstanding stores
- Result: **4.9 GB/s** (2.7x improvement)

Problem: still 32-bit stores. Each PCIe TLP (Transaction Layer Packet) has a fixed ~16-byte header overhead. With 4B payload per TLP the effective ratio is only 4/20 = 20%; with 16B payload (uint4) it becomes 16/32 = 50%. Wider stores amortize the per-packet header cost and push more useful data per PCIe transaction.

### V3: 1024 threads, 128-bit stores (uint4)

- Switched from `uint32_t` to `uint4` (16 bytes per store instruction)
- Each thread now issues a 128-bit `STG.E.128` instruction per loop iteration
- Result: **11.7 GB/s** (2.4x improvement)

This is the final version. Each store pushes 16B through the PCIe fabric, and with 1024 threads the SM can maintain enough outstanding writes to partially saturate the link.

## Results

Default parameters: 4 MiB slot, 4 ring slots, 256 MiB total transfer.

| Configuration | Bandwidth |
|---------------|-----------|
| 4 MiB slot, 4 ring depth | 11.7 GB/s |
| 8 MiB slot, 8 ring depth | 11.9 GB/s |
| 1 MiB slot, 8 ring depth | 8.7 GB/s |

## Comparison with Other Approaches

| Method | Bandwidth | Notes |
|--------|-----------|-------|
| `cudaMemcpyPeerAsync` (Copy Engine) | 26.1 GB/s | Dedicated DMA hardware, no SM involvement |
| SM remote read (`remote_memory_kernel`) | 12.7 GB/s | Kernel loads from peer, write local |
| SM remote write (`remote_memory_kernel`) | 10.6 GB/s | Kernel loads local, stores to peer |
| **ring_buffer_p2p_2node (this experiment)** | **11.7 GB/s** | SM remote write + flag sync overhead |

The ~12 GB/s ceiling for a single thread block is inherent to SM-initiated remote access on this PCIe SYS topology. The SM's outstanding memory request limit bounds achievable bandwidth when latency is high (Little's Law).

## Why NCCL Achieves Higher Bandwidth

NCCL uses multiple techniques beyond what this single-block experiment does:

1. **Multiple channels**: each channel is a separate kernel (or cooperative thread block), multiplying outstanding requests across multiple SMs.
2. **Cooperative kernel launch**: allows multiple thread blocks to coordinate as a persistent kernel, utilizing more SMs.
3. **Double buffering with overlap**: while one slot is being written, another can be read, pipelining sender and receiver.
4. **Tuned slot sizes**: NCCL auto-tunes buffer sizes and thread counts for each topology.

These optimizations allow NCCL to push SM-based transfers closer to the hardware limit, while also performing reduce operations during the copy — something the Copy Engine cannot do.
