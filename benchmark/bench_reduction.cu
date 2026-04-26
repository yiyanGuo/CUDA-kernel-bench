#include <bench_common.h>
#include "reduction.h"

#include <vector>

#ifndef REDUCTION_N
#define REDUCTION_N (1 << 24)
#endif

#ifndef REDUCTION_WARMUP
#define REDUCTION_WARMUP 2
#endif

#ifndef REDUCTION_REPEAT
#define REDUCTION_REPEAT 5
#endif

namespace {

using ReductionLaunchFn = void (*)(const float *, float *, int);

struct ReductionImplementation {
  const char *name;
  ReductionLaunchFn launch;
};

struct ReductionContext {
  int n;
  std::vector<float> host_input;
  float host_output;
  float host_ref;
  float *device_input;
  float *device_output;
};


static const ReductionImplementation kReductionImplementations[] = {
    {"reduction_naive", reduction_naive},
    {"reduction_presum", reduction_presum},
    {"reduction_presum_float4", reduction_presum_float4},
    {"reduction_shuffle", reduction_shuffle},
    {"reduction_grid_stride", reduction_grid_stride},
    {"reduction_integrate", reduction_integrate}
};

static float cpu_reduction(const std::vector<float> &input) {
  double sum = 0.0;
  for (float value : input) {
    sum += static_cast<double>(value);
  }
  return static_cast<float>(sum);
}

static void prepare_reduction_context(ReductionContext &context) {
  context.n = REDUCTION_N;
  context.host_input.resize(context.n);
  context.host_output = 0.0f;
  context.host_ref = 0.0f;

  for (int i = 0; i < context.n; ++i) {
    context.host_input[i] = static_cast<float>((i % 113) - 56) * 0.03125f;
  }

  context.host_ref = cpu_reduction(context.host_input);

  context.device_input = nullptr;
  context.device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&context.device_input, sizeof(float) * context.n));
  CUDA_CHECK(cudaMalloc(&context.device_output, sizeof(float)));

  CUDA_CHECK(cudaMemcpy(context.device_input, context.host_input.data(),
                        sizeof(float) * context.n, cudaMemcpyHostToDevice));
}

static void release_reduction_context(ReductionContext &context) {
  CUDA_CHECK(cudaFree(context.device_input));
  CUDA_CHECK(cudaFree(context.device_output));
}

static bool
run_one_reduction_implementation(const ReductionImplementation &implementation,
                                 ReductionContext &context) {
  for (int i = 0; i < REDUCTION_WARMUP; ++i) {
    CUDA_CHECK(cudaMemset(context.device_output, 0, sizeof(float)));
    implementation.launch(context.device_input, context.device_output,
                          context.n);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  float best_ms = measure_min_elapsed_ms(REDUCTION_REPEAT, [&]() {
    CUDA_CHECK(cudaMemset(context.device_output, 0, sizeof(float)));
    implementation.launch(context.device_input, context.device_output,
                          context.n);
  });

  CUDA_CHECK(cudaMemcpy(&context.host_output, context.device_output,
                        sizeof(float), cudaMemcpyDeviceToHost));
  bool passed =
      nearly_equal(context.host_output, context.host_ref, 1e-2f, 1e-3f);

  const double work_units = static_cast<double>(context.n - 1);
  const double bytes =
      static_cast<double>(context.n) * sizeof(float) + sizeof(float);
  print_benchmark_line("reduction", implementation.name, best_ms, work_units,
                       "Add", bytes, passed);
  return passed;
}

} // namespace

bool run_reduction_benchmark() {
  ReductionContext context{};
  prepare_reduction_context(context);

  bool all_passed = true;
  for (const auto &implementation : kReductionImplementations) {
    all_passed =
        run_one_reduction_implementation(implementation, context) && all_passed;
  }

  release_reduction_context(context);
  return all_passed;
}
