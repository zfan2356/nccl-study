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

void verifyCopy(uint32_t* dst, size_t count, int device, uint32_t seed) {
  CHECK_CUDA(cudaSetDevice(device));
  std::vector<uint32_t> host(count);
  CHECK_CUDA(cudaMemcpy(host.data(), dst, count * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < count; ++i) {
    uint32_t expected = seed ^ static_cast<uint32_t>(i);
    if (host[i] != expected) {
      throw std::runtime_error("verification failed at element " + std::to_string(i) +
                               ": expected " + std::to_string(expected) +
                               ", got " + std::to_string(host[i]));
    }
  }
}

double measurePeerCopy(uint32_t* dst, int dstDevice, uint32_t* src, int srcDevice,
                       size_t bytes, int iterations) {
  CHECK_CUDA(cudaSetDevice(dstDevice));

  cudaStream_t stream{};
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, bytes, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < iterations; ++i) {
    CHECK_CUDA(cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, bytes, stream));
  }
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsedMs = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));

  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaStreamDestroy(stream));

  double totalBytes = static_cast<double>(bytes) * static_cast<double>(iterations);
  return totalBytes / (static_cast<double>(elapsedMs) / 1000.0) / 1.0e9;
}

void runDirection(uint32_t* buffers[2], size_t count, int srcDevice, int dstDevice,
                  int iterations) {
  size_t bytes = count * sizeof(uint32_t);
  uint32_t seed = 0x12340000u + static_cast<uint32_t>(srcDevice);

  initializeSource(buffers[srcDevice], count, srcDevice, seed);

  CHECK_CUDA(cudaSetDevice(dstDevice));
  CHECK_CUDA(cudaMemset(buffers[dstDevice], 0, bytes));
  CHECK_CUDA(cudaDeviceSynchronize());

  double gbps = measurePeerCopy(buffers[dstDevice], dstDevice, buffers[srcDevice],
                                srcDevice, bytes, iterations);
  verifyCopy(buffers[dstDevice], count, dstDevice, seed);

  std::cout << "GPU" << srcDevice << " -> GPU" << dstDevice << ": "
            << gbps << " GB/s"
            << " (" << bytes / (1024 * 1024) << " MiB x "
            << iterations << " iterations)\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
      std::cerr << "Need at least two CUDA devices for peer copy.\n";
      return EXIT_FAILURE;
    }

    size_t bytes = parseMiB(argc, argv) * 1024 * 1024;
    size_t count = bytes / sizeof(uint32_t);
    bytes = count * sizeof(uint32_t);
    int iterations = parseIterations(argc, argv);

    enablePeerAccess(0, 1);
    enablePeerAccess(1, 0);

    uint32_t* buffers[2]{};
    for (int device = 0; device < 2; ++device) {
      CHECK_CUDA(cudaSetDevice(device));
      CHECK_CUDA(cudaMalloc(&buffers[device], bytes));
    }

    std::cout << "Peer copy size: " << bytes / (1024 * 1024) << " MiB\n";
    std::cout << "Iterations: " << iterations << "\n";

    runDirection(buffers, count, 0, 1, iterations);
    runDirection(buffers, count, 1, 0, iterations);

    for (int device = 0; device < 2; ++device) {
      CHECK_CUDA(cudaSetDevice(device));
      CHECK_CUDA(cudaFree(buffers[device]));
    }

    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
