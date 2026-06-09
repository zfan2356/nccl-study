#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

}  // namespace

// ---------------------------------------------------------------------------
// Device management
// ---------------------------------------------------------------------------

static void set_device(int device) {
  CHECK_CUDA(cudaSetDevice(device));
}

static void synchronize() {
  CHECK_CUDA(cudaDeviceSynchronize());
}

static int get_device_count() {
  int count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&count));
  return count;
}

// ---------------------------------------------------------------------------
// Buffer allocation with IPC handle export
// ---------------------------------------------------------------------------

// Allocate a buffer on the current device and return (ptr_as_int, ipc_handle_bytes)
static std::pair<uintptr_t, py::bytes> alloc_and_export(size_t size) {
  void* ptr = nullptr;
  CHECK_CUDA(cudaMalloc(&ptr, size));
  CHECK_CUDA(cudaMemset(ptr, 0, size));

  cudaIpcMemHandle_t handle;
  CHECK_CUDA(cudaIpcGetMemHandle(&handle, ptr));

  // Convert handle to bytes
  std::string handle_bytes(reinterpret_cast<char*>(&handle), sizeof(handle));
  return {reinterpret_cast<uintptr_t>(ptr), py::bytes(handle_bytes)};
}

// Open an IPC handle exported by another process
static uintptr_t open_ipc(py::bytes handle_bytes) {
  std::string data = handle_bytes;
  if (data.size() != sizeof(cudaIpcMemHandle_t)) {
    throw std::runtime_error("Invalid IPC handle size: expected " +
                             std::to_string(sizeof(cudaIpcMemHandle_t)) +
                             ", got " + std::to_string(data.size()));
  }

  cudaIpcMemHandle_t handle;
  std::memcpy(&handle, data.data(), sizeof(handle));

  void* ptr = nullptr;
  CHECK_CUDA(cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<uintptr_t>(ptr);
}

// Close an IPC-opened handle
static void close_ipc(uintptr_t ptr) {
  CHECK_CUDA(cudaIpcCloseMemHandle(reinterpret_cast<void*>(ptr)));
}

// Free a locally allocated buffer
static void free_buffer(uintptr_t ptr) {
  CHECK_CUDA(cudaFree(reinterpret_cast<void*>(ptr)));
}

// ---------------------------------------------------------------------------
// Timing utilities
// ---------------------------------------------------------------------------

static uintptr_t create_event() {
  cudaEvent_t evt;
  CHECK_CUDA(cudaEventCreate(&evt));
  return reinterpret_cast<uintptr_t>(evt);
}

static void record_event(uintptr_t evt) {
  CHECK_CUDA(cudaEventRecord(reinterpret_cast<cudaEvent_t>(evt)));
}

static void sync_event(uintptr_t evt) {
  CHECK_CUDA(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(evt)));
}

static float elapsed_ms(uintptr_t start, uintptr_t stop) {
  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms,
    reinterpret_cast<cudaEvent_t>(start),
    reinterpret_cast<cudaEvent_t>(stop)));
  return ms;
}

static void destroy_event(uintptr_t evt) {
  CHECK_CUDA(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(evt)));
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(ipc_utils, m) {
  m.doc() = "CUDA IPC utilities for multi-process GPU communication experiments";

  // Device management
  m.def("set_device", &set_device, "Set current CUDA device");
  m.def("synchronize", &synchronize, "Synchronize current device");
  m.def("get_device_count", &get_device_count, "Get number of CUDA devices");

  // IPC buffer management
  m.def("alloc_and_export", &alloc_and_export,
        "Allocate buffer on current device, return (ptr, ipc_handle_bytes)");
  m.def("open_ipc", &open_ipc,
        "Open an IPC handle from another process, return device ptr");
  m.def("close_ipc", &close_ipc, "Close an IPC-opened handle");
  m.def("free_buffer", &free_buffer, "Free a locally allocated buffer");

  // Timing
  m.def("create_event", &create_event, "Create a CUDA event");
  m.def("record_event", &record_event, "Record a CUDA event");
  m.def("sync_event", &sync_event, "Synchronize a CUDA event");
  m.def("elapsed_ms", &elapsed_ms, "Get elapsed time between two events in ms");
  m.def("destroy_event", &destroy_event, "Destroy a CUDA event");
}
