# L20 Interconnect Labs

This directory contains local single-node, two-GPU L20 experiments for learning how GPU interconnect communication works from the bottom up.

The goal is to build small C++/CUDA experiments first, then compare the same communication patterns with NCCL.

## Hardware Scope

- Machine: single node
- GPUs: 2x NVIDIA L20
- Observed GPU-to-GPU path: `SYS`
- Focus: CUDA peer access, peer copies, CUDA IPC, and toy collective communication

See `topology.md` for the captured `nvidia-smi topo -m` and `lspci -tv` outputs.

## Planned Experiments

1. Inspect topology with `nvidia-smi topo -m` and `lspci -tv`. Done: GPU0 <-> GPU1 is `SYS`, and the visible CUDA GPUs are on different PCIe root branches.
2. Check CUDA peer accessibility with `cudaDeviceCanAccessPeer`. Done: peer access is available in both directions.
3. Enable peer access and run GPU-to-GPU copies. Done: `p2p_copy` measures bidirectional `cudaMemcpyPeerAsync`.
4. Launch a kernel on GPU0 that directly reads or writes GPU1 memory. Done: `remote_memory_kernel` measures SM-initiated peer memory access.
5. Implement a minimal two-GPU all-reduce.
6. Compare the toy implementation with `ncclAllReduce`.

## CUDA Virtual Memory Notes

CUDA device pointers are virtual addresses in a CUDA device context, not CPU pointers. A host process can store pointers returned by `cudaMalloc` on multiple GPUs, but the CPU cannot dereference those pointers directly.

The key idea for P2P is: peer access lets one GPU's virtual address space resolve another GPU's device pointer.

With peer access disabled, a kernel running on GPU0 should only dereference GPU0-accessible device pointers. With peer access enabled by `cudaDeviceEnablePeerAccess(1, 0)`, CUDA maps peer GPU memory into GPU0's accessible virtual address space. Then a GPU0 kernel can use ordinary load/store instructions on a pointer allocated on GPU1.

This mapping is directional. Enabling GPU0 -> GPU1 access does not automatically enable GPU1 -> GPU0 access; the reverse direction needs its own `cudaDeviceEnablePeerAccess(0, 0)` call while GPU1 is the current device.

This also does not merge every GPU into one global memory space. It only makes peer allocations resolvable from the enabled GPU context.

That does not make peer memory local. The instruction is issued by GPU0 SMs, but the memory request travels over the GPU interconnect path, which is `SYS` on this machine. This is different from `cudaMemcpyPeerAsync`, which uses CUDA copy/DMA machinery for bulk transfers.

## Build

This directory is set up as a standalone CMake project for C++/CUDA experiments.

```bash
cmake -S interconnect/l20 -B build/l20
cmake --build build/l20
```

Note: this environment currently has CUDA 12.1 with conda GCC 14 first in `PATH`. CUDA 12.1 cannot compile against that libstdc++, so the CMake project prefers `/usr/bin/g++` as the CUDA host compiler when available.

## Experiments

### `p2p_capabilities`

Prints CUDA device metadata, checks peer accessibility for each GPU pair, and tries to enable peer access when supported.

```bash
./build/l20/p2p_capabilities
```

### `p2p_copy`

Allocates one buffer on GPU0 and one buffer on GPU1, initializes the source buffer with a CUDA kernel, copies data with `cudaMemcpyPeerAsync`, verifies correctness, and reports bandwidth in both directions.

```bash
./build/l20/p2p_copy
```

Optional arguments:

```bash
./build/l20/p2p_copy <MiB> <iterations>
```

Example:

```bash
./build/l20/p2p_copy 256 20
```

### `remote_memory_kernel`

Launches kernels on GPU0 and passes either GPU0-local or GPU1-remote pointers. It compares local SM load/store bandwidth with SM-initiated peer memory access.

```bash
./build/l20/remote_memory_kernel
```

Optional arguments:

```bash
./build/l20/remote_memory_kernel <MiB> <iterations>
```
