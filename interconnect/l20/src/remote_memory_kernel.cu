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

__global__ void fillPattern(uint32_t* data, size_t count, uint32_t seed) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    data[idx] = seed ^ static_cast<uint32_t>(idx);
  }
}

__global__ void copyAddOne(uint32_t* dst, const uint32_t* src, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    dst[idx] = src[idx] + 1;
  }
}

size_t parseMiB(int argc, char** argv) {
  if (argc < 2) return 256;
  return std::stoull(argv[1]);
}

int parseIterations(int argc, char** argv) {
  if (argc < 3) return 20;
  return std::stoi(argv[2]);
}

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

void initializeSource(uint32_t* data, size_t count, int device, uint32_t seed) {
  CHECK_CUDA(cudaSetDevice(device));
  int threads = 256;
  int blocks = static_cast<int>((count + threads - 1) / threads);
  fillPattern<<<blocks, threads>>>(data, count, seed);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

void clearBuffer(uint32_t* data, size_t bytes, int device) {
  CHECK_CUDA(cudaSetDevice(device));
  CHECK_CUDA(cudaMemset(data, 0, bytes));
  CHECK_CUDA(cudaDeviceSynchronize());
}

void verifyCopyAddOne(uint32_t* dst, size_t count, int device, uint32_t seed) {
  CHECK_CUDA(cudaSetDevice(device));
  std::vector<uint32_t> host(count);
  CHECK_CUDA(cudaMemcpy(host.data(), dst, count * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < count; ++i) {
    uint32_t expected = (seed ^ static_cast<uint32_t>(i)) + 1;
    if (host[i] != expected) {
      throw std::runtime_error("verification failed at element " + std::to_string(i) +
                               ": expected " + std::to_string(expected) +
                               ", got " + std::to_string(host[i]));
    }
  }
}

double measureKernelCopy(uint32_t* dst, const uint32_t* src, size_t count,
                         size_t bytes, int iterations) {
  CHECK_CUDA(cudaSetDevice(0));

  cudaStream_t stream{};
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  int threads = 256;
  int blocks = static_cast<int>((count + threads - 1) / threads);

  copyAddOne<<<blocks, threads, 0, stream>>>(dst, src, count);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < iterations; ++i) {
    copyAddOne<<<blocks, threads, 0, stream>>>(dst, src, count);
  }
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaGetLastError());

  float elapsedMs = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));

  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaStreamDestroy(stream));

  double totalBytes = static_cast<double>(bytes) * static_cast<double>(iterations);
  return totalBytes / (static_cast<double>(elapsedMs) / 1000.0) / 1.0e9;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
      std::cerr << "Need at least two CUDA devices for remote memory access.\n";
      return EXIT_FAILURE;
    }

    size_t bytes = parseMiB(argc, argv) * 1024 * 1024;
    size_t count = bytes / sizeof(uint32_t);
    bytes = count * sizeof(uint32_t);
    int iterations = parseIterations(argc, argv);

    enablePeerAccess(0, 1);
    enablePeerAccess(1, 0);

    uint32_t* gpu0Src = nullptr;
    uint32_t* gpu0Dst = nullptr;
    uint32_t* gpu1Src = nullptr;
    uint32_t* gpu1Dst = nullptr;

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMalloc(&gpu0Src, bytes));
    CHECK_CUDA(cudaMalloc(&gpu0Dst, bytes));

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc(&gpu1Src, bytes));
    CHECK_CUDA(cudaMalloc(&gpu1Dst, bytes));

    uint32_t gpu0Seed = 0x12340000u;
    uint32_t gpu1Seed = 0x56780000u;

    initializeSource(gpu0Src, count, 0, gpu0Seed);
    initializeSource(gpu1Src, count, 1, gpu1Seed);

    std::cout << "Kernel copy size: " << bytes / (1024 * 1024) << " MiB\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "All kernels launch on GPU0 SMs.\n";

    clearBuffer(gpu0Dst, bytes, 0);
    double localReadWrite = measureKernelCopy(gpu0Dst, gpu0Src, count, bytes, iterations);
    verifyCopyAddOne(gpu0Dst, count, 0, gpu0Seed);
    std::cout << "GPU0 local read -> GPU0 local write: "
              << localReadWrite << " GB/s\n";

    clearBuffer(gpu0Dst, bytes, 0);
    double remoteRead = measureKernelCopy(gpu0Dst, gpu1Src, count, bytes, iterations);
    verifyCopyAddOne(gpu0Dst, count, 0, gpu1Seed);
    std::cout << "GPU1 remote read -> GPU0 local write: "
              << remoteRead << " GB/s\n";

    clearBuffer(gpu1Dst, bytes, 1);
    double remoteWrite = measureKernelCopy(gpu1Dst, gpu0Src, count, bytes, iterations);
    verifyCopyAddOne(gpu1Dst, count, 1, gpu0Seed);
    std::cout << "GPU0 local read -> GPU1 remote write: "
              << remoteWrite << " GB/s\n";

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaFree(gpu1Dst));
    CHECK_CUDA(cudaFree(gpu1Src));

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(gpu0Dst));
    CHECK_CUDA(cudaFree(gpu0Src));

    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
