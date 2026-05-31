cmake_minimum_required(VERSION 3.18)

# This lab environment puts conda GCC 14 first in PATH, but CUDA 12.1 cannot
# compile CUDA C++ against that libstdc++. Prefer the system compiler when it is
# available so a plain CMake configure works from an activated conda shell.
if(EXISTS "/usr/bin/g++")
  set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE FILEPATH "C++ compiler")
  set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++" CACHE FILEPATH "CUDA host compiler")
endif()

project(l20_interconnect_labs LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

function(add_l20_cuda_experiment target source)
  add_executable(${target} ${source})

  target_include_directories(${target} PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
endfunction()

add_l20_cuda_experiment(p2p_capabilities src/p2p_capabilities.cu)
add_l20_cuda_experiment(p2p_copy src/p2p_copy.cu)
add_l20_cuda_experiment(remote_memory_kernel src/remote_memory_kernel.cu)
