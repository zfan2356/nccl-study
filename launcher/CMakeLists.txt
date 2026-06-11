cmake_minimum_required(VERSION 3.18)

if(EXISTS "/usr/bin/g++")
  set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE FILEPATH "C++ compiler")
  set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++" CACHE FILEPATH "CUDA host compiler")
endif()

project(nccl_study_launcher LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(ipc_utils src/ipc_utils.cu)
target_include_directories(ipc_utils PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(ipc_utils PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

find_library(CUDA_DRIVER_LIBRARY cuda REQUIRED)
pybind11_add_module(cumem_utils src/cumem_utils.cu)
target_include_directories(cumem_utils PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cumem_utils PRIVATE ${CUDA_DRIVER_LIBRARY})
set_target_properties(cumem_utils PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
