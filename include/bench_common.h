#pragma once

#include "cuda_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

inline bool nearly_equal(float lhs, float rhs, float abs_tolerance = 1e-5f,
                         float rel_tolerance = 1e-5f) {
  float diff = std::fabs(lhs - rhs);
  if (diff <= abs_tolerance) {
    return true;
  }
  float scale = std::max(std::fabs(lhs), std::fabs(rhs));
  return diff <= rel_tolerance * scale;
}

template <typename T>
inline bool compare_arrays(const T *lhs, const T *rhs, int count,
                           float abs_tolerance = 1e-5f,
                           float rel_tolerance = 1e-5f) {
  for (int i = 0; i < count; ++i) {
    if (!nearly_equal(static_cast<float>(lhs[i]), static_cast<float>(rhs[i]),
                      abs_tolerance, rel_tolerance)) {
      std::fprintf(stderr, "Mismatch at index %d: lhs=%f rhs=%f\n", i,
                   static_cast<double>(lhs[i]), static_cast<double>(rhs[i]));
      return false;
    }
  }
  return true;
}

template <typename LaunchFn>
float measure_min_elapsed_ms(int repeat_count, LaunchFn &&launch) {
  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  float best_ms = std::numeric_limits<float>::max();
  for (int i = 0; i < repeat_count; ++i) {
    CUDA_CHECK(cudaEventRecord(start_event));
    launch();
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    best_ms = std::min(best_ms, elapsed_ms);
  }

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));
  return best_ms;
}

inline double to_giga_throughput(double units, float elapsed_ms) {
  return units / (static_cast<double>(elapsed_ms) * 1e6);
}

inline double to_bandwidth_gb(double bytes, float elapsed_ms) {
  return bytes / (static_cast<double>(elapsed_ms) * 1e6);
}

inline void print_benchmark_line(const char *op_name, const char *impl_name,
                                 float best_ms, double work_units,
                                 const char *work_unit_name, double bytes,
                                 bool passed) {
  std::printf("[%s/%s] best=%.4f ms, %.3f G%s/s, %.3f GB/s, verify=%s\n",
              op_name, impl_name, best_ms,
              to_giga_throughput(work_units, best_ms), work_unit_name,
              to_bandwidth_gb(bytes, best_ms), passed ? "PASS" : "FAIL");
}
