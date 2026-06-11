# L20 Communication Primitives Labs

This directory contains NCCL-style communication primitive experiments on a two-GPU L20 node with PCIe SYS interconnect.

The goal is to implement the core building blocks that NCCL uses internally, starting from the lowest-level GPU-to-GPU synchronization and working up to collective communication patterns.

## Hardware Scope

Same as `interconnect/l20`:

- Machine: single node
- GPUs: 2x NVIDIA L20
- GPU-to-GPU path: `SYS` (PCIe + QPI/UPI across NUMA nodes)
- Peer access: available in both directions

## Build

```bash
cmake -S primitives/l20 -B build/primitives_l20
cmake --build build/primitives_l20
```

---

## Primitives

### 1. ring_buffer_p2p

Ring buffer based P2P communication with head/tail flag synchronization. This is the foundation of NCCL's Simple protocol — sender remote-writes data into receiver's local buffer, receiver polls a flag then reads locally.

Designed for **reduce scenarios** (AllReduce, ReduceScatter) where receiver needs to compute on incoming data. Not optimal for pure data transfer (use CE or direct_put instead).

| Experiment | Description | Status | Design Doc |
|------------|-------------|--------|------------|
| `ring_buffer_p2p_2node` | Single block, single channel, 2 GPUs | Done | [design/ring_buffer_p2p_2node.md](design/ring_buffer_p2p_2node.md) |
| `ring_buffer_p2p_2node_multi_channel` | Multi-channel: each block owns independent ring buffer + flags, data split across channels | TODO | |
| `ring_buffer_p2p_2node_cooperative` | Cooperative kernel: multiple blocks collaborate on same slot via `grid.sync()` | TODO | |

### 2. recv_reduce_send

Fuse reduce into the ring buffer copy — receive data, reduce with local buffer, then forward to next hop. This is the core compute primitive of NCCL and the basic unit of ring allreduce.

| Experiment | Description | Status | Design Doc |
|------------|-------------|--------|------------|
| `recv_reduce_send_2node` | Single channel recv+reduce+send on 2 GPUs | TODO | |

### 3. ll_protocol

Low-Latency protocol: inline data+flag in every 12-byte unit (8B data + 4B flag). No separate `__threadfence_system()` needed — the flag validity implies data validity. Trades bandwidth for minimum latency on small messages.

| Experiment | Description | Status | Design Doc |
|------------|-------------|--------|------------|
| `ll_protocol_2node` | LL send/recv on 2 GPUs, compare latency with ring_buffer_p2p | TODO | |

### 4. direct_put

For operations without reduce (AllGather, Broadcast, All-to-All): sender writes directly to receiver's final output buffer at the correct offset, skipping intermediate ring buffers entirely.

| Experiment | Description | Status | Design Doc |
|------------|-------------|--------|------------|
| `direct_put_2node` | Direct remote write to output buffer on 2 GPUs | TODO | |

---

## Running

### `ring_buffer_p2p_2node` (single process, 2 GPUs)

```bash
./build/primitives_l20/ring_buffer_p2p_2node [slot_MiB] [num_slots] [total_MiB]
```

Example:

```bash
./build/primitives_l20/ring_buffer_p2p_2node 4 4 256
```

### `ring_buffer_p2p` multi-process (legacy IPC)

```bash
torchrun --nproc_per_node=2 primitives/l20/python/run_ring_buffer_p2p.py [slot_MiB] [num_slots] [total_MiB]
```

Uses `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`.

### `ring_buffer_p2p` multi-process (cuMem / VMM)

```bash
torchrun --nproc_per_node=2 primitives/l20/python/run_ring_buffer_p2p_cumem.py [slot_MiB] [num_slots] [total_MiB]
```

Uses `cuMemCreate` / `cuMemExportToShareableHandle` (POSIX FD) / `cuMemImportFromShareableHandle` — the same path NCCL uses when `NCCL_CUMEM_ENABLE=1`. FDs are passed between processes via Unix domain sockets (`SCM_RIGHTS`).
