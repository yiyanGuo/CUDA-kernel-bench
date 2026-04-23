#include <bench_common.h>
#include "vector_add.h"
#include <vector>


#ifndef VECTOR_ADD_N
#define VECTOR_ADD_N (1 << 24)
#endif

#ifndef VECTOR_ADD_WARMUP
#define VECTOR_ADD_WARMUP 2
#endif

#ifndef VECTOR_ADD_REPEAT
#define VECTOR_ADD_REPEAT 5
#endif

void vector_add_naive(const float *a, const float *b, float *c, int n);

namespace {

using VectorAddLaunchFn = void (*)(const float *, const float *, float *, int);

struct VectorAddImplementation {
  const char *name;
  VectorAddLaunchFn launch;
};

struct VectorAddContext {
  int n;
  std::vector<float> host_a;
  std::vector<float> host_b;
  std::vector<float> host_output;
  std::vector<float> host_ref;
  float *device_a;
  float *device_b;
  float *device_c;
};

static const VectorAddImplementation kVectorAddImplementations[] = {
    {"naive", vector_add_naive},
    {"float4", vector_add_float4}
};

static void cpu_vector_add(const std::vector<float> &a,
                           const std::vector<float> &b, std::vector<float> &c) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    c[i] = a[i] + b[i];
  }
}

static void prepare_vector_add_context(VectorAddContext &context) {
  context.n = VECTOR_ADD_N;
  context.host_a.resize(context.n);
  context.host_b.resize(context.n);
  context.host_output.assign(context.n, 0.0f);
  context.host_ref.assign(context.n, 0.0f);

  for (int i = 0; i < context.n; ++i) {
    context.host_a[i] = static_cast<float>(i % 97) * 0.5f;
    context.host_b[i] = static_cast<float>(i % 53) * 0.25f;
  }

  cpu_vector_add(context.host_a, context.host_b, context.host_ref);

  context.device_a = nullptr;
  context.device_b = nullptr;
  context.device_c = nullptr;
  CUDA_CHECK(cudaMalloc(&context.device_a, sizeof(float) * context.n));
  CUDA_CHECK(cudaMalloc(&context.device_b, sizeof(float) * context.n));
  CUDA_CHECK(cudaMalloc(&context.device_c, sizeof(float) * context.n));

  CUDA_CHECK(cudaMemcpy(context.device_a, context.host_a.data(),
                        sizeof(float) * context.n, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(context.device_b, context.host_b.data(),
                        sizeof(float) * context.n, cudaMemcpyHostToDevice));
}

static void release_vector_add_context(VectorAddContext &context) {
  CUDA_CHECK(cudaFree(context.device_a));
  CUDA_CHECK(cudaFree(context.device_b));
  CUDA_CHECK(cudaFree(context.device_c));
}

static bool
run_one_vector_add_implementation(const VectorAddImplementation &implementation,
                                  VectorAddContext &context) {
  for (int i = 0; i < VECTOR_ADD_WARMUP; ++i) {
    implementation.launch(context.device_a, context.device_b, context.device_c,
                          context.n);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  float best_ms = measure_min_elapsed_ms(VECTOR_ADD_REPEAT, [&]() {
    implementation.launch(context.device_a, context.device_b, context.device_c,
                          context.n);
  });

  CUDA_CHECK(cudaMemcpy(context.host_output.data(), context.device_c,
                        sizeof(float) * context.n, cudaMemcpyDeviceToHost));
  bool passed = compare_arrays(context.host_output.data(),
                               context.host_ref.data(), context.n);

  const double work_units = static_cast<double>(context.n);
  const double bytes = static_cast<double>(context.n) * sizeof(float) * 3.0;
  print_benchmark_line("vector_add", implementation.name, best_ms, work_units,
                       "FLOP", bytes, passed);
  return passed;
}

} // namespace

bool run_vector_add_benchmark() {
  VectorAddContext context{};
  prepare_vector_add_context(context);

  bool all_passed = true;
  for (const auto &implementation : kVectorAddImplementations) {
    all_passed = run_one_vector_add_implementation(implementation, context) &&
                 all_passed;
  }

  release_vector_add_context(context);
  return all_passed;
}
