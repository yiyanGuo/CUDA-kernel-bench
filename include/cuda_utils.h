#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _cuda_status = (call);                                         \
    if (_cuda_status != cudaSuccess) {                                         \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(_cuda_status));                          \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

