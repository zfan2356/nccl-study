# L20 Topology

Captured with:

```bash
nvidia-smi topo -m
```

Output:

```text
        GPU0    GPU1    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      SYS     0-191   0               N/A
GPU1    SYS      X      192-383 1               N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

## Notes

- `GPU0 <-> GPU1` is `SYS`, so communication crosses PCIe and the inter-socket CPU fabric.
- GPU0 is close to NUMA node 0, while GPU1 is close to NUMA node 1.
- This is a useful learning setup because it makes locality visible: CPU thread affinity, host staging, and peer access behavior can matter.
- Do not assume NVLink-style direct bandwidth. Measure each path explicitly.

## PCIe Tree

Captured with:

```bash
lspci -tv
```

Output:

```text
-+-[0000:a0]---00.0-[a1-a3]----00.0-[a2-a3]----00.0-[a3]----00.0  NVIDIA Corporation AD102GL [L20]
 +-[0000:80]---00.0-[81-84]----00.0-[82-84]--+-00.0-[83]----00.0  NVIDIA Corporation AD102GL [L20]
 |                                           \-01.0-[84]----00.0  NVIDIA Corporation AD102GL [L20]
 +-[0000:60]---00.0-[61-63]----00.0-[62-63]----00.0-[63]----00.0  NVIDIA Corporation AD102GL [L20]
 +-[0000:40]---00.0-[41-43]----00.0-[42-43]----00.0-[43]----00.0  NVIDIA Corporation AD102GL [L20]
 +-[0000:30]---00.0-[31-34]----00.0-[32-34]--+-00.0-[33]----00.0  NVIDIA Corporation AD102GL [L20]
 |                                           \-01.0-[34]----00.0  NVIDIA Corporation AD102GL [L20]
 +-[0000:20]---00.0-[21-23]----00.0-[22-23]----00.0-[23]----00.0  NVIDIA Corporation AD102GL [L20]
 \-[0000:00]-+-00.0  Intel Corporation 82G33/G31/P35/P31 Express DRAM Controller
             +-01.0-[01-02]----00.0-[02]--+-01.0  Cirrus Logic GD 5446
             |                            +-02.0  Red Hat, Inc. Virtio network device
             |                            +-03.0  Intel Corporation 82801FB/FBM/FR/FW/FRW (ICH6 Family) High Definition Audio Controller
             |                            +-04.0  NEC Corporation uPD720200 USB 3.0 Host Controller
             |                            +-05.0  Red Hat, Inc. Virtio block device
             |                            +-06.0  Red Hat, Inc. Virtio block device
             |                            +-07.0  Red Hat, Inc. Virtio block device
             |                            +-08.0  Red Hat, Inc. Virtio memory balloon
             |                            +-09.0  Red Hat, Inc. Virtio block device
             |                            +-0a.0  Red Hat, Inc. Virtio network device
             |                            +-0b.0  Red Hat, Inc. Virtio block device
             |                            +-0c.0  Red Hat, Inc. Virtio block device
             |                            +-0d.0  Red Hat, Inc. Virtio block device
             |                            +-0e.0  Red Hat, Inc. Virtio block device
             |                            +-0f.0  Red Hat, Inc. Virtio block device
             |                            +-10.0  Red Hat, Inc. Virtio block device
             |                            +-11.0  Red Hat, Inc. Virtio block device
             |                            +-12.0  Red Hat, Inc. Virtio block device
             |                            +-13.0  Red Hat, Inc. Virtio block device
             |                            \-15.0  Red Hat, Inc. Virtio block device
             +-02.0-[03-04]----00.0-[04]--
             +-03.0-[05-06]----00.0-[06]--
             +-04.0  Red Hat, Inc. QEMU PCIe Expander bridge
             +-05.0  Red Hat, Inc. QEMU PCIe Expander bridge
             +-06.0  Red Hat, Inc. QEMU PCIe Expander bridge
             +-07.0  Red Hat, Inc. QEMU PCIe Expander bridge
             +-08.0  Red Hat, Inc. QEMU PCIe Expander bridge
             +-09.0  Red Hat, Inc. QEMU PCIe Expander bridge
             +-1f.0  Intel Corporation 82801IB (ICH9) LPC Interface Controller
             +-1f.2  Intel Corporation 82801IR/IO/IH (ICH9R/DO/DH) 6 port SATA Controller [AHCI mode]
             \-1f.3  Intel Corporation 82801I (ICH9 Family) SMBus Controller
```

The CUDA-visible devices in this lab are:

- `GPU0`: `0000:43:00.0`
- `GPU1`: `0000:63:00.0`

Those two devices sit under different top-level PCIe branches:

```text
[0000:40] -> ... -> [43] -> 43:00.0  L20
[0000:60] -> ... -> [63] -> 63:00.0  L20
```

This matches the `SYS` relationship from `nvidia-smi topo -m`: the two visible GPUs are not siblings under the same nearby PCIe switch. They are exposed under separate PCIe root/expander branches, so traffic between them has to climb out of one branch and cross system-level fabric before reaching the other branch.

The full `lspci` tree shows more L20 devices than CUDA exposes to this process. That can happen in virtualized or partitioned environments where the host PCIe tree is visible, but the CUDA runtime only makes a subset of GPUs available to the current container/process.

### Reading `lspci -tv`

- `[0000:40]`, `[0000:60]`, `[0000:80]`, and similar labels are PCI domains/root buses.
- `00.0-[41-43]` means device `00.0` is a PCIe bridge whose downstream bus range is `41` through `43`.
- The final `00.0 NVIDIA Corporation AD102GL [L20]` is the GPU function at the leaf of that branch.
- Branches with two leaves, such as `[83]` and `[84]`, indicate two GPUs behind the same higher-level branch.
- `QEMU PCIe Expander bridge` and `Virtio` devices show that this environment is virtualized.

## Next Questions

1. Does CUDA report peer access support between GPU0 and GPU1? Yes, both directions.
2. If peer access is supported, what bandwidth does `cudaMemcpyPeerAsync` achieve?
3. If peer access is not supported, what path does a device-to-device copy fall back to?
4. How does NCCL choose transports for this topology?

## CUDA Peer Access Probe

Built and ran:

```bash
cmake -S interconnect/l20 -B build/l20
cmake --build build/l20
./build/l20/p2p_capabilities
```

Observed:

```text
CUDA device count: 2

GPU0
  name: NVIDIA L20
  pci bus id: 0000:43:00.0
  SMs: 92
  total global memory: 44.5274 GiB

GPU1
  name: NVIDIA L20
  pci bus id: 0000:63:00.0
  SMs: 92
  total global memory: 44.5274 GiB

GPU0 -> GPU1 peer access: yes
  enabled
GPU1 -> GPU0 peer access: yes
  enabled
```

## Peer Copy Baseline

Built and ran:

```bash
cmake --build build/l20
./build/l20/p2p_copy
```

Observed with the default 256 MiB buffer and 20 iterations:

```text
Peer copy size: 256 MiB
Iterations: 20
GPU0 -> GPU1: 26.0884 GB/s (256 MiB x 20 iterations)
GPU1 -> GPU0: 26.1118 GB/s (256 MiB x 20 iterations)
```

This is the baseline bandwidth for `cudaMemcpyPeerAsync` on the observed `SYS` GPU-to-GPU path.

## Remote Memory Kernel Baseline

Built and ran:

```bash
cmake --build build/l20
./build/l20/remote_memory_kernel
```

Observed with the default 256 MiB buffer and 20 iterations:

```text
Kernel copy size: 256 MiB
Iterations: 20
All kernels launch on GPU0 SMs.
GPU0 local read -> GPU0 local write: 177.791 GB/s
GPU1 remote read -> GPU0 local write: 12.7099 GB/s
GPU0 local read -> GPU1 remote write: 10.5906 GB/s
```

These numbers measure ordinary GPU0 kernel load/store instructions. They are lower than `cudaMemcpyPeerAsync` because the work is issued by GPU0 SMs as memory instructions rather than by the CUDA copy engine as a bulk transfer. The reported bandwidth uses the copied payload size; the local read/write case touches both a source and destination buffer.
