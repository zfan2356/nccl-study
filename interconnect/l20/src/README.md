# Source Layout

Place small C++/CUDA experiment programs in this directory.

Each experiment should stay focused on one mechanism, such as topology inspection, peer access, CUDA IPC, or a toy all-reduce.

- `p2p_capabilities.cu`: enumerates CUDA devices, checks `cudaDeviceCanAccessPeer`, and enables peer access when available.
- `p2p_copy.cu`: copies device memory between GPU0 and GPU1 with `cudaMemcpyPeerAsync`, verifies the copied data, and reports bandwidth.
- `remote_memory_kernel.cu`: launches kernels on GPU0 that use ordinary load/store instructions against GPU0-local and GPU1-remote pointers.
