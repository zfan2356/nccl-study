#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

void checkCuda(cudaError_t result, const char* expr) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(expr) + ": " + cudaGetErrorString(result));
  }
}

#define CHECK_CUDA(expr) checkCuda((expr), #expr)

// ---------------------------------------------------------------------------
// Kernels (same as ring_buffer_p2p_2node.cu)
// ---------------------------------------------------------------------------

__global__ void senderKernel(uint32_t* __restrict__ srcBuf,
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
    if (tid == 0) {
      while (static_cast<int64_t>(slot) - static_cast<int64_t>(*head) >= numSlots) {
      }
    }
    __syncthreads();

    int ringIdx = slot % numSlots;
    uint4* dst = reinterpret_cast<uint4*>(recvBuf + static_cast<size_t>(ringIdx) * slotCount);
    uint4* src = reinterpret_cast<uint4*>(srcBuf + static_cast<size_t>(slot) * slotCount);

    for (size_t i = tid; i < slotCount4; i += numThreads) {
      dst[i] = src[i];
    }

    __syncthreads();

    if (tid == 0) {
      __threadfence_system();
      *tail = static_cast<uint64_t>(slot + 1);
    }
    __syncthreads();
  }
}

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
    if (tid == 0) {
      while (*tail <= static_cast<uint64_t>(slot)) {
      }
    }
    __syncthreads();

    int ringIdx = slot % numSlots;
    uint4* src = reinterpret_cast<uint4*>(recvBuf + static_cast<size_t>(ringIdx) * slotCount);
    uint4* dst = reinterpret_cast<uint4*>(dstBuf + static_cast<size_t>(slot) * slotCount);

    for (size_t i = tid; i < slotCount4; i += numThreads) {
      dst[i] = src[i];
    }

    __syncthreads();

    if (tid == 0) {
      __threadfence_system();
      *head = static_cast<uint64_t>(slot + 1);
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Fill source buffer with a pattern (for verification)
// ---------------------------------------------------------------------------

__global__ void fillPattern(uint32_t* data, size_t count, uint32_t seed) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    data[idx] = seed ^ static_cast<uint32_t>(idx);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host-callable functions exposed via pybind11
// ---------------------------------------------------------------------------

static void fill_source(uintptr_t ptr, size_t count, uint32_t seed) {
  uint32_t* buf = reinterpret_cast<uint32_t*>(ptr);
  int threads = 256;
  int blocks = static_cast<int>((count + threads - 1) / threads);
  fillPattern<<<blocks, threads>>>(buf, count, seed);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

static void run_sender(uintptr_t src_buf, uintptr_t recv_buf,
                       uintptr_t tail, uintptr_t head,
                       size_t slot_count, int num_slots, int total_slots) {
  senderKernel<<<1, 1024>>>(
      reinterpret_cast<uint32_t*>(src_buf),
      reinterpret_cast<uint32_t*>(recv_buf),
      reinterpret_cast<volatile uint64_t*>(tail),
      reinterpret_cast<volatile uint64_t*>(head),
      slot_count, num_slots, total_slots);
  CHECK_CUDA(cudaGetLastError());
}

static void run_receiver(uintptr_t dst_buf, uintptr_t recv_buf,
                         uintptr_t tail, uintptr_t head,
                         size_t slot_count, int num_slots, int total_slots) {
  receiverKernel<<<1, 1024>>>(
      reinterpret_cast<uint32_t*>(dst_buf),
      reinterpret_cast<uint32_t*>(recv_buf),
      reinterpret_cast<volatile uint64_t*>(tail),
      reinterpret_cast<volatile uint64_t*>(head),
      slot_count, num_slots, total_slots);
  CHECK_CUDA(cudaGetLastError());
}

static void memset_buffer(uintptr_t ptr, size_t bytes) {
  CHECK_CUDA(cudaMemset(reinterpret_cast<void*>(ptr), 0, bytes));
  CHECK_CUDA(cudaDeviceSynchronize());
}

static bool verify(uintptr_t ptr, size_t count, uint32_t seed) {
  std::vector<uint32_t> host(count);
  CHECK_CUDA(cudaMemcpy(host.data(), reinterpret_cast<void*>(ptr),
                         count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < count; ++i) {
    uint32_t expected = seed ^ static_cast<uint32_t>(i);
    if (host[i] != expected) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(ring_buffer_p2p_ipc, m) {
  m.doc() = "Ring buffer P2P kernel with IPC support";

  m.def("fill_source", &fill_source, "Fill buffer with verification pattern");
  m.def("run_sender", &run_sender, "Launch sender kernel on current device");
  m.def("run_receiver", &run_receiver, "Launch receiver kernel on current device");
  m.def("memset_buffer", &memset_buffer, "Zero out a device buffer");
  m.def("verify", &verify, "Verify destination buffer against expected pattern");
}
