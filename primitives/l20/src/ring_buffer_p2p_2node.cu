#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void checkCuda(cudaError_t result, const char* expr) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(expr) + ": " + cudaGetErrorString(result));
  }
}

#define CHECK_CUDA(expr) checkCuda((expr), #expr)

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct Config {
  size_t slotBytes = 4 * 1024 * 1024;  // 4 MiB per slot
  int numSlots = 4;                     // ring buffer depth
  size_t totalBytes = 256 * 1024 * 1024; // total data to transfer
};

Config parseArgs(int argc, char** argv) {
  Config cfg;
  if (argc >= 2) cfg.slotBytes = std::stoull(argv[1]) * 1024 * 1024;
  if (argc >= 3) cfg.numSlots = std::stoi(argv[2]);
  if (argc >= 4) cfg.totalBytes = std::stoull(argv[3]) * 1024 * 1024;
  return cfg;
}

// ---------------------------------------------------------------------------
// Shared structures between sender and receiver
//
// Layout:
//   recvBuf[numSlots][slotCount]  -- allocated on GPU1 (receiver local)
//   tail (volatile uint64_t*)     -- allocated on GPU1 (sender remote-writes)
//   head (volatile uint64_t*)     -- allocated on GPU0 (receiver remote-writes)
//
// Placement principle: "remote write, local read"
//   Flags are placed on the GPU that polls (spin-reads) them.
//   - tail on GPU1: receiver polls tail locally (fast, ~tens of ns per iteration),
//     sender updates tail via remote write (fire-and-forget, infrequent).
//   - head on GPU0: sender polls head locally (fast),
//     receiver updates head via remote write (infrequent).
//   If flags were placed on the remote side, every spin iteration would be a
//   remote read (~2us round-trip on PCIe SYS), making polling extremely slow.
//   Remote writes are acceptable for updates because they don't block the writer
//   and only happen once per slot (low frequency).
//
// Protocol:
//   Sender writes slot data into recvBuf[slot % numSlots] via peer access,
//   then issues __threadfence_system() and increments tail.
//   Receiver spins on tail, reads the slot locally, then increments head.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Sender kernel (runs on GPU0)
//
// Reads source data from GPU0-local srcBuf, writes it to recvBuf on GPU1
// (remote write via peer access). Uses tail/head for flow control.
// ---------------------------------------------------------------------------
__global__ void senderKernel(uint32_t* __restrict__ srcBuf,
                             uint32_t* __restrict__ recvBuf,
                             volatile uint64_t* tail,
                             volatile uint64_t* head,
                             size_t slotCount,
                             int numSlots,
                             int totalSlots) {
  // Use a single thread block as the persistent "channel"
  const int tid = threadIdx.x;
  const int numThreads = blockDim.x;

  // Use 128-bit (uint4) wide copies to maximize PCIe utilization
  size_t slotCount4 = slotCount / 4;  // number of uint4 elements per slot

  for (int slot = 0; slot < totalSlots; ++slot) {
    // Flow control: wait until receiver has consumed enough so we don't
    // overwrite an unread slot.  (tail - head < numSlots)
    if (tid == 0) {
      while (static_cast<int64_t>(slot) - static_cast<int64_t>(*head) >= numSlots) {
        // spin
      }
    }
    __syncthreads();

    // Compute pointers into the ring buffer slot
    int ringIdx = slot % numSlots;
    uint4* dst = reinterpret_cast<uint4*>(recvBuf + static_cast<size_t>(ringIdx) * slotCount);
    uint4* src = reinterpret_cast<uint4*>(srcBuf + static_cast<size_t>(slot) * slotCount);

    // Copy data from local srcBuf to remote recvBuf (peer write) using 128-bit stores
    for (size_t i = tid; i < slotCount4; i += numThreads) {
      dst[i] = src[i];
    }

    __syncthreads();

    // Ensure all data writes are visible to GPU1 before updating tail.
    //
    // __threadfence_system() compiles to PTX `membar.sys` (SASS: MEMBAR.SC.SYS).
    // It guarantees that all preceding stores are observable by every agent in
    // the system (other GPUs via peer access, and the CPU) before any subsequent
    // store becomes observable.
    //
    // Without this fence, the hardware may reorder stores: GPU1 could see the
    // tail update before the data arrives, reading stale/zero values.
    //
    // Fence levels:
    //   __threadfence_block() → membar.cta  (visible within this thread block)
    //   __threadfence()       → membar.gl   (visible to all SMs on this GPU)
    //   __threadfence_system()→ membar.sys  (visible to all GPUs + CPU)
    //
    // We need membar.sys here because the observer (receiver) is on a different
    // GPU. membar.gl would only guarantee visibility within GPU0's own SMs.
    // membar.sys waits for writes to propagate through PCIe/NVLink to remote
    // observers, which is more expensive but necessary for cross-GPU ordering.
    if (tid == 0) {
      __threadfence_system();
      *tail = static_cast<uint64_t>(slot + 1);
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Receiver kernel (runs on GPU1)
//
// Spins on tail to detect new data, reads from local recvBuf, writes to
// local dstBuf (for verification), then updates head to release the slot.
// ---------------------------------------------------------------------------
__global__ void receiverKernel(uint32_t* __restrict__ dstBuf,
                               uint32_t* __restrict__ recvBuf,
                               volatile uint64_t* tail,
                               volatile uint64_t* head,
                               size_t slotCount,
                               int numSlots,
                               int totalSlots) {
  const int tid = threadIdx.x;
  const int numThreads = blockDim.x;

  size_t slotCount4 = slotCount / 4;

  for (int slot = 0; slot < totalSlots; ++slot) {
    // Wait for sender to produce this slot
    if (tid == 0) {
      while (*tail <= static_cast<uint64_t>(slot)) {
        // spin
      }
    }
    __syncthreads();

    // Read from local recvBuf and write to dstBuf for verification (128-bit)
    int ringIdx = slot % numSlots;
    uint4* src = reinterpret_cast<uint4*>(recvBuf + static_cast<size_t>(ringIdx) * slotCount);
    uint4* dst = reinterpret_cast<uint4*>(dstBuf + static_cast<size_t>(slot) * slotCount);

    for (size_t i = tid; i < slotCount4; i += numThreads) {
      dst[i] = src[i];
    }

    __syncthreads();

    // Signal sender that this slot is now free.
    // membar.sys ensures our local reads from recvBuf are complete before we
    // update head -- otherwise sender might overwrite the slot while we're
    // still reading. Then the head update (remote write to GPU0) tells sender
    // this slot is reusable.
    if (tid == 0) {
      __threadfence_system();
      *head = static_cast<uint64_t>(slot + 1);
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

void enablePeerAccess(int device, int peer) {
  int canAccess = 0;
  CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccess, device, peer));
  if (!canAccess) {
    throw std::runtime_error("GPU" + std::to_string(device) +
                             " cannot access GPU" + std::to_string(peer));
  }

  CHECK_CUDA(cudaSetDevice(device));
  cudaError_t result = cudaDeviceEnablePeerAccess(peer, 0);
  if (result == cudaErrorPeerAccessAlreadyEnabled) {
    CHECK_CUDA(cudaGetLastError());
    return;
  }
  CHECK_CUDA(result);
}

void fillSource(uint32_t* buf, size_t count, uint32_t seed) {
  std::vector<uint32_t> host(count);
  for (size_t i = 0; i < count; ++i) {
    host[i] = seed ^ static_cast<uint32_t>(i);
  }
  CHECK_CUDA(cudaMemcpy(buf, host.data(), count * sizeof(uint32_t),
                         cudaMemcpyHostToDevice));
}

bool verifyDst(uint32_t* devBuf, size_t count, uint32_t seed) {
  std::vector<uint32_t> host(count);
  CHECK_CUDA(cudaMemcpy(host.data(), devBuf, count * sizeof(uint32_t),
                         cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < count; ++i) {
    uint32_t expected = seed ^ static_cast<uint32_t>(i);
    if (host[i] != expected) {
      std::cerr << "verification failed at element " << i
                << ": expected " << expected << ", got " << host[i] << "\n";
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
      std::cerr << "Need at least two CUDA devices.\n";
      return EXIT_FAILURE;
    }

    Config cfg = parseArgs(argc, argv);
    size_t slotCount = cfg.slotBytes / sizeof(uint32_t);
    cfg.slotBytes = slotCount * sizeof(uint32_t);
    int totalSlots = static_cast<int>(cfg.totalBytes / cfg.slotBytes);
    if (totalSlots < 1) totalSlots = 1;
    size_t totalCount = static_cast<size_t>(totalSlots) * slotCount;

    std::cout << "Slot size: " << cfg.slotBytes / (1024 * 1024) << " MiB\n";
    std::cout << "Num slots (ring depth): " << cfg.numSlots << "\n";
    std::cout << "Total transfer: " << totalSlots * (cfg.slotBytes / (1024 * 1024)) << " MiB"
              << " (" << totalSlots << " slots)\n";

    enablePeerAccess(0, 1);
    enablePeerAccess(1, 0);

    // Allocate source buffer on GPU0 (all data that will be sent)
    uint32_t* srcBuf = nullptr;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMalloc(&srcBuf, totalCount * sizeof(uint32_t)));

    // Allocate ring buffer on GPU1 (receiver local, sender remote-writes)
    uint32_t* recvBuf = nullptr;
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc(&recvBuf, static_cast<size_t>(cfg.numSlots) * slotCount * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(recvBuf, 0, static_cast<size_t>(cfg.numSlots) * slotCount * sizeof(uint32_t)));

    // Allocate destination buffer on GPU1 (receiver writes here for verification)
    uint32_t* dstBuf = nullptr;
    CHECK_CUDA(cudaMalloc(&dstBuf, totalCount * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(dstBuf, 0, totalCount * sizeof(uint32_t)));

    // Allocate synchronization counters
    // tail on GPU1 (sender remote-writes, receiver local-reads)
    volatile uint64_t* tail = nullptr;
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc((void**)&tail, sizeof(uint64_t)));
    CHECK_CUDA(cudaMemset((void*)tail, 0, sizeof(uint64_t)));

    // head on GPU0 (receiver remote-writes, sender local-reads)
    volatile uint64_t* head = nullptr;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMalloc((void**)&head, sizeof(uint64_t)));
    CHECK_CUDA(cudaMemset((void*)head, 0, sizeof(uint64_t)));

    // Fill source data
    uint32_t seed = 0xDEAD0000u;
    CHECK_CUDA(cudaSetDevice(0));
    fillSource(srcBuf, totalCount, seed);

    // Create streams
    cudaStream_t senderStream, receiverStream;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaStreamCreate(&senderStream));
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaStreamCreate(&receiverStream));

    // Warmup: run once to prime TLBs and page tables
    {
      CHECK_CUDA(cudaSetDevice(1));
      CHECK_CUDA(cudaMemset((void*)tail, 0, sizeof(uint64_t)));
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(cudaSetDevice(0));
      CHECK_CUDA(cudaMemset((void*)head, 0, sizeof(uint64_t)));
      CHECK_CUDA(cudaDeviceSynchronize());

      int warmupSlots = cfg.numSlots;  // just a few slots
      CHECK_CUDA(cudaSetDevice(1));
      receiverKernel<<<1, 1024, 0, receiverStream>>>(
          dstBuf, recvBuf, tail, head, slotCount, cfg.numSlots, warmupSlots);
      CHECK_CUDA(cudaGetLastError());

      CHECK_CUDA(cudaSetDevice(0));
      senderKernel<<<1, 1024, 0, senderStream>>>(
          srcBuf, recvBuf, tail, head, slotCount, cfg.numSlots, warmupSlots);
      CHECK_CUDA(cudaGetLastError());

      CHECK_CUDA(cudaSetDevice(0));
      CHECK_CUDA(cudaStreamSynchronize(senderStream));
      CHECK_CUDA(cudaSetDevice(1));
      CHECK_CUDA(cudaStreamSynchronize(receiverStream));
    }

    // Reset counters for timed run
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMemset((void*)tail, 0, sizeof(uint64_t)));
    CHECK_CUDA(cudaMemset(recvBuf, 0, static_cast<size_t>(cfg.numSlots) * slotCount * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(dstBuf, 0, totalCount * sizeof(uint32_t)));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMemset((void*)head, 0, sizeof(uint64_t)));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t startEvt, stopEvt;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventCreate(&startEvt));
    CHECK_CUDA(cudaEventCreate(&stopEvt));

    // Launch receiver first (it will spin-wait on tail)
    CHECK_CUDA(cudaSetDevice(1));
    receiverKernel<<<1, 1024, 0, receiverStream>>>(
        dstBuf, recvBuf, tail, head, slotCount, cfg.numSlots, totalSlots);
    CHECK_CUDA(cudaGetLastError());

    // Launch sender with timing
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventRecord(startEvt, senderStream));
    senderKernel<<<1, 1024, 0, senderStream>>>(
        srcBuf, recvBuf, tail, head, slotCount, cfg.numSlots, totalSlots);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stopEvt, senderStream));

    // Wait for both to finish
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaStreamSynchronize(senderStream));
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaStreamSynchronize(receiverStream));

    float elapsedMs = 0.0f;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, startEvt, stopEvt));

    double totalDataBytes = static_cast<double>(totalCount) * sizeof(uint32_t);
    double gbps = totalDataBytes / (static_cast<double>(elapsedMs) / 1000.0) / 1.0e9;

    std::cout << "\nResults:\n";
    std::cout << "  Elapsed: " << elapsedMs << " ms\n";
    std::cout << "  Bandwidth: " << gbps << " GB/s\n";

    // Verify correctness
    CHECK_CUDA(cudaSetDevice(1));
    bool ok = verifyDst(dstBuf, totalCount, seed);
    std::cout << "  Verification: " << (ok ? "PASS" : "FAIL") << "\n";

    // Cleanup
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventDestroy(stopEvt));
    CHECK_CUDA(cudaEventDestroy(startEvt));
    CHECK_CUDA(cudaStreamDestroy(senderStream));
    CHECK_CUDA(cudaFree(srcBuf));
    CHECK_CUDA(cudaFree((void*)head));

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaStreamDestroy(receiverStream));
    CHECK_CUDA(cudaFree(dstBuf));
    CHECK_CUDA(cudaFree(recvBuf));
    CHECK_CUDA(cudaFree((void*)tail));

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
