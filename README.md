# NCCL Study

This repository is a study workspace for understanding GPU communication from the bottom up, progressing from raw hardware primitives to production-grade libraries like NCCL, NVSHMEM, and DeepEP.

## Learning Roadmap

### Phase 1: Interconnect Fundamentals (`interconnect/`)

Understand the raw hardware: PCIe topology, peer access, DMA (Copy Engine) vs SM-initiated memory access.

- [x] Topology inspection (`nvidia-smi topo`, `lspci`)
- [x] CUDA peer access probe (`cudaDeviceCanAccessPeer`)
- [x] P2P copy bandwidth (`cudaMemcpyPeerAsync`, Copy Engine)
- [x] SM remote memory access (kernel load/store to peer GPU memory)

**Key takeaway**: Copy Engine achieves ~26 GB/s; SM remote access is ~12 GB/s on PCIe SYS, bounded by outstanding request limits (Little's Law).

### Phase 2: Communication Primitives (`primitives/`)

Implement the core building blocks that NCCL uses internally.

- [x] Ring buffer P2P 2-node (persistent kernel + head/tail ring buffer, NCCL Simple protocol core)
- [ ] LL (Low-Latency) protocol (data+flag inline, no separate fence)
- [ ] recvReduceSend (fuse reduce into the copy, the reason NCCL uses SM instead of CE)
- [ ] Multi-channel (multiple thread blocks/SMs to saturate bandwidth)

**Key takeaway**: NCCL doesn't use `cudaMemcpy`; it uses SM kernels for remote write + flag sync + fused reduce. Multiple channels scale bandwidth toward hardware limits.

### Phase 3: Collective Communication (`collectives/`)

Assemble primitives into collective operations.

- [ ] Ring AllReduce (chain recvReduceSend around a ring)
- [ ] Tree AllReduce (binary tree for latency-sensitive small messages)
- [ ] Compare with `ncclAllReduce` (same hardware, measure gap)

**Key takeaway**: Ring is bandwidth-optimal for large messages; Tree is latency-optimal for small messages. NCCL auto-selects based on message size.

### Phase 4: NVSHMEM & One-Sided Communication

Understand the PGAS (Partitioned Global Address Space) model — GPU-initiated, one-sided, fine-grained communication.

- [ ] NVSHMEM basics: symmetric memory, `nvshmem_put`/`nvshmem_get` from within kernels
- [ ] Signal-based synchronization (`nvshmem_signal`, `nvshmem_wait_until`)
- [ ] Compare NVSHMEM put/get with Phase 2 hand-written flag sync
- [ ] Irregular communication patterns (sparse all-to-all, hash table lookup)

**Key takeaway**: NCCL = collective (all participate); NVSHMEM = one-sided (any thread can independently read/write remote GPU memory). NVSHMEM is better for irregular/dynamic patterns.

### Phase 5: NCCL GIN (GPU-Initiated Networking)

Understand how NCCL extends to GPU-initiated inter-node communication without CPU coordination.

- [ ] NCCL Device API: device-side communicators, memory windows
- [ ] One-sided remote memory ops from CUDA kernels (similar to NVSHMEM but within NCCL ecosystem)
- [ ] Backend comparison: GDAKI (GPUDirect Async Kernel-Initiated, GPU→NIC RDMA) vs Proxy (lock-free GPU→CPU queue)
- [ ] Overlap communication with computation using GIN

**Key takeaway**: Traditional NCCL is host-initiated (CPU orchestrates). GIN makes it GPU-initiated — the GPU kernel directly triggers network transfers, eliminating CPU latency overhead.

### Phase 6: DeepEP & Expert Parallelism

Understand all-to-all communication optimized for MoE (Mixture of Experts) models.

- [ ] MoE dispatch/combine pattern: every GPU sends tokens to every other GPU's expert
- [ ] Intra-node: NVLink-domain all-to-all with custom kernels
- [ ] Inter-node: RDMA-domain transfers via NVSHMEM/GIN, overlapped with GPU computation
- [ ] Asymmetric-domain bandwidth forwarding (NVLink ↔ RDMA bridge)
- [ ] Low-latency inference kernels (pure-RDMA path for decode)
- [ ] Compare with naive NCCL `AlltoAll`

**Key takeaway**: DeepEP is purpose-built for MoE's all-to-all pattern. It bypasses NCCL, uses custom kernels + NVSHMEM for GPU-initiated RDMA, and optimizes the NVLink↔InfiniBand boundary.

## Dependency Graph

```
Phase 1 (interconnect)
    │
    ▼
Phase 2 (primitives)
    │
    ├──────────────────┐
    ▼                  ▼
Phase 3 (collectives)  Phase 4 (NVSHMEM)
    │                  │
    ▼                  ▼
Phase 5 (NCCL GIN) ←──┘
    │
    ▼
Phase 6 (DeepEP)
```

## Hardware Path

| Phase | Hardware Required |
|-------|-------------------|
| 1-3 | 2x L20, PCIe SYS (current setup) |
| 4 | Multi-GPU node with NVSHMEM support |
| 5 | Multi-node with InfiniBand / GPUDirect RDMA |
| 6 | Multi-node NVLink + InfiniBand (e.g., H800 cluster) |

## Quick Start

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/zfan2356/nccl-study.git
cd nccl-study

# Build everything and register Python paths (one-time setup)
bash scripts/prepare_env.sh

# Run multi-process ring buffer P2P experiment (legacy IPC)
torchrun --nproc_per_node=2 primitives/l20/python/run_ring_buffer_p2p.py

# Run multi-process ring buffer P2P experiment (cuMem VMM, CUDA 12+)
torchrun --nproc_per_node=2 primitives/l20/python/run_ring_buffer_p2p_cumem.py
```

## Directory Structure

```
nccl-study/
├── launcher/               # Public multi-process framework
│   ├── src/ipc_utils.cu    #   CUDA IPC pybind11 utilities
│   └── python/             #   Launcher & IPC handle exchange
├── scripts/
│   └── prepare_env.sh      # One-time env setup (build + register paths)
├── interconnect/
│   └── l20/                # Phase 1: raw hardware experiments
├── primitives/
│   └── l20/                # Phase 2: communication primitives
├── collectives/
│   └── l20/                # Phase 3: collective ops (TODO)
├── nvshmem/                # Phase 4: PGAS model (TODO, hardware dependent)
├── gin/                    # Phase 5: GPU-initiated networking (TODO)
├── deepep/                 # Phase 6: expert parallelism (TODO)
└── third-party/
    ├── DeepEP/             # DeepEP source (submodule)
    └── nccl/               # NCCL source (submodule)
```

## Third-Party Dependencies

- `third-party/DeepEP`: DeepEP, included as a Git submodule from `https://github.com/deepseek-ai/DeepEP.git`.
- `third-party/nccl`: NVIDIA NCCL, included as a Git submodule from `https://github.com/NVIDIA/nccl.git`.

To clone this repository with submodules:

```bash
git clone --recurse-submodules https://github.com/zfan2356/nccl-study.git
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```
