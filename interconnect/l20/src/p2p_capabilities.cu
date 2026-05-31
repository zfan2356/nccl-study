#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void checkCuda(cudaError_t result, const char* expr) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(expr) + ": " + cudaGetErrorString(result));
  }
}

#define CHECK_CUDA(expr) checkCuda((expr), #expr)

void printDeviceInfo(int device) {
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  char pciBusId[32]{};
  CHECK_CUDA(cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device));

  std::cout << "GPU" << device << "\n"
            << "  name: " << prop.name << "\n"
            << "  pci bus id: " << pciBusId << "\n"
            << "  pci domain:bus:device: " << prop.pciDomainID << ":"
            << prop.pciBusID << ":" << prop.pciDeviceID << "\n"
            << "  SMs: " << prop.multiProcessorCount << "\n"
            << "  total global memory: "
            << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0)
            << " GiB\n";
}

void checkPeerAccess(int src, int dst) {
  int canAccess = 0;
  CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccess, src, dst));

  std::cout << "GPU" << src << " -> GPU" << dst
            << " peer access: " << (canAccess ? "yes" : "no") << "\n";

  if (!canAccess) return;

  CHECK_CUDA(cudaSetDevice(src));
  cudaError_t enableResult = cudaDeviceEnablePeerAccess(dst, 0);
  if (enableResult == cudaErrorPeerAccessAlreadyEnabled) {
    CHECK_CUDA(cudaGetLastError());
    std::cout << "  already enabled\n";
    return;
  }

  CHECK_CUDA(enableResult);
  std::cout << "  enabled\n";
}

}  // namespace

int main() {
  try {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    std::cout << "CUDA device count: " << deviceCount << "\n\n";
    if (deviceCount < 2) {
      std::cerr << "Need at least two CUDA devices for L20 interconnect labs.\n";
      return EXIT_FAILURE;
    }

    for (int device = 0; device < deviceCount; ++device) {
      printDeviceInfo(device);
      std::cout << "\n";
    }

    for (int src = 0; src < deviceCount; ++src) {
      for (int dst = 0; dst < deviceCount; ++dst) {
        if (src == dst) continue;
        checkPeerAccess(src, dst);
      }
    }

    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
