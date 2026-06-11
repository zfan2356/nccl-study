#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

constexpr CUmemAllocationHandleType kHandleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

// Throw if a CUDA Runtime API call fails.
void checkCuda(cudaError_t result, const char* expr) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(expr) + ": " + cudaGetErrorString(result));
  }
}

// Throw if a CUDA Driver API call fails.
void checkCu(CUresult result, const char* expr) {
  if (result != CUDA_SUCCESS) {
    const char* err = nullptr;
    cuGetErrorString(result, &err);
    throw std::runtime_error(std::string(expr) + ": " + (err ? err : "unknown CUDA driver error"));
  }
}

#define CHECK_CUDA(expr) checkCuda((expr), #expr)
#define CHECK_CU(expr) checkCu((expr), #expr)

// Initialize the CUDA driver once before any cu* call.
void ensure_driver_init() {
  static bool initialized = false;
  if (!initialized) {
    CHECK_CU(cuInit(0));  // load libcuda and init the driver API
    initialized = true;
  }
}

// Round size up to cuMem's minimum allocation granularity for this device.
size_t align_size(size_t size, int device) {
  ensure_driver_init();
  CHECK_CUDA(cudaSetDevice(device));

  CUdevice cuDev = 0;
  CHECK_CU(cuDeviceGet(&cuDev, device));  // runtime dev id -> CUdevice handle

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;           // GPU device memory
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = cuDev;
  prop.requestedHandleTypes = kHandleType;             // exportable as POSIX fd

  int rdmaCapable = 0;
  CHECK_CU(cuDeviceGetAttribute(
      &rdmaCapable, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, cuDev));
  if (rdmaCapable) {
    prop.allocFlags.gpuDirectRDMACapable = 1;          // allow NIC direct access if supported
  }

  size_t granularity = 0;
  CHECK_CU(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  return (size + granularity - 1) / granularity * granularity;
}

// Map a physical allocation into a reserved VA and grant this GPU read/write access.
void map_and_set_access(CUdeviceptr ptr, size_t size, CUmemGenericAllocationHandle handle, int device) {
  CHECK_CUDA(cudaSetDevice(device));
  CUdevice cuDev = 0;
  CHECK_CU(cuDeviceGet(&cuDev, device));

  CHECK_CU(cuMemMap(ptr, size, 0, handle, 0));       // bind physical handle to virtual address

  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cuDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CHECK_CU(cuMemSetAccess(ptr, size, &accessDesc, 1)); // allow this GPU to load/store the mapping
}

// Tear down a VA mapping and release the underlying physical allocation.
void unmap_and_release(uintptr_t ptr, size_t aligned_size) {
  CUmemGenericAllocationHandle handle = {};
  CHECK_CU(cuMemRetainAllocationHandle(&handle, reinterpret_cast<void*>(ptr)));  // recover handle from ptr
  CHECK_CU(cuMemUnmap(static_cast<CUdeviceptr>(ptr), aligned_size));              // drop VA mapping
  CHECK_CU(cuMemRelease(handle));                                                  // free physical allocation
  CHECK_CU(cuMemAddressFree(static_cast<CUdeviceptr>(ptr), aligned_size));        // release reserved VA range
}

}  // namespace

// ---------------------------------------------------------------------------
// Capability probe
// ---------------------------------------------------------------------------

// Return true if this device supports CUDA VMM (cuMem API).
static bool is_vmm_supported(int device) {
  try {
    ensure_driver_init();
    int driverVersion = 0;
    CHECK_CUDA(cudaDriverGetVersion(&driverVersion));
    if (driverVersion < 12000) {
      return false;
    }

    CUdevice cuDev = 0;
    CHECK_CU(cuDeviceGet(&cuDev, device));
    int flag = 0;
    CHECK_CU(cuDeviceGetAttribute(
        &flag, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cuDev));
    return flag != 0;
  } catch (const std::exception&) {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Device management
// ---------------------------------------------------------------------------

// Set the active CUDA device for this thread.
static void set_device(int device) {
  CHECK_CUDA(cudaSetDevice(device));
}

// Block until all work on the current device completes.
static void synchronize() {
  CHECK_CUDA(cudaDeviceSynchronize());
}

// Enable P2P so kernels on device can dereference peer GPU pointers.
static void enable_peer_access(int device, int peer) {
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

// ---------------------------------------------------------------------------
// cuMem buffer management (POSIX file descriptor handle type)
// ---------------------------------------------------------------------------

// Allocate local GPU memory and export it as a POSIX fd for cross-process sharing.
static py::tuple alloc_and_export_fd(size_t size, int device) {
  ensure_driver_init();
  CHECK_CUDA(cudaSetDevice(device));

  CUdevice cuDev = 0;
  CHECK_CU(cuDeviceGet(&cuDev, device));

  const size_t alignedSize = align_size(size, device);

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = cuDev;
  prop.requestedHandleTypes = kHandleType;

  int rdmaCapable = 0;
  CHECK_CU(cuDeviceGetAttribute(
      &rdmaCapable, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, cuDev));
  if (rdmaCapable) {
    prop.allocFlags.gpuDirectRDMACapable = 1;
  }

  CUmemGenericAllocationHandle allocHandle = {};
  CHECK_CU(cuMemCreate(&allocHandle, alignedSize, &prop, 0));  // allocate physical GPU memory

  CUdeviceptr ptr = 0;
  CHECK_CU(cuMemAddressReserve(&ptr, alignedSize, 0, 0, 0));   // reserve a VA range (no backing yet)
  map_and_set_access(ptr, alignedSize, allocHandle, device);

  CHECK_CUDA(cudaMemset(reinterpret_cast<void*>(ptr), 0, alignedSize));

  int exportFd = -1;
  CHECK_CU(cuMemExportToShareableHandle(
      &exportFd, allocHandle, kHandleType, 0));                // produce fd for UDS transfer

  CHECK_CU(cuMemRelease(allocHandle));  // drop export ref; mapping keeps allocation alive

  return py::make_tuple(static_cast<uintptr_t>(ptr), exportFd, alignedSize);
}

// Import a peer's exported fd and map it into this process/GPU's address space.
static uintptr_t import_from_fd(int fd, size_t aligned_size, int importer_device) {
  ensure_driver_init();
  CHECK_CUDA(cudaSetDevice(importer_device));

  CUmemGenericAllocationHandle importHandle = {};
  CHECK_CU(cuMemImportFromShareableHandle(
      &importHandle, reinterpret_cast<void*>(static_cast<uintptr_t>(fd)), kHandleType));

  CUdeviceptr ptr = 0;
  CHECK_CU(cuMemAddressReserve(&ptr, aligned_size, 0, 0, 0));
  map_and_set_access(ptr, aligned_size, importHandle, importer_device);

  CHECK_CU(cuMemRelease(importHandle));  // mapping holds the only ref we need

  return static_cast<uintptr_t>(ptr);
}

// Free a buffer allocated by alloc_and_export_fd on this process.
static void free_local(uintptr_t ptr, size_t aligned_size) {
  if (ptr == 0 || aligned_size == 0) {
    return;
  }
  ensure_driver_init();
  unmap_and_release(ptr, aligned_size);
}

// Unmap a buffer imported by import_from_fd (does not free the peer's allocation).
static void close_imported(uintptr_t ptr, size_t aligned_size) {
  if (ptr == 0 || aligned_size == 0) {
    return;
  }
  ensure_driver_init();
  unmap_and_release(ptr, aligned_size);
}

// Query the cuMem-aligned size for a given raw byte count.
static size_t get_aligned_size(size_t size, int device) {
  return align_size(size, device);
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(cumem_utils, m) {
  m.doc() = "CUDA cuMem (VMM) utilities for multi-process GPU communication";

  m.def("is_vmm_supported", &is_vmm_supported,
        "Check if CUDA VMM / cuMem is supported on the given device");
  m.def("set_device", &set_device, "Set current CUDA device");
  m.def("synchronize", &synchronize, "Synchronize current device");
  m.def("enable_peer_access", &enable_peer_access,
        "Enable peer access between two CUDA devices");

  m.def("alloc_and_export_fd", &alloc_and_export_fd,
        "Allocate cuMem buffer on device; return (ptr, export_fd, aligned_size)");
  m.def("import_from_fd", &import_from_fd,
        "Import a cuMem buffer from a POSIX FD; return mapped device ptr");
  m.def("free_local", &free_local, "Free a locally allocated cuMem buffer");
  m.def("close_imported", &close_imported, "Unmap an imported cuMem buffer");
  m.def("get_aligned_size", &get_aligned_size,
        "Return cuMem minimum-granularity-aligned size for a device");
}
