#include "scan.h"
#include <bench_common.h>

#include <vector>

#ifndef SCAN_N
#define SCAN_N (1 << 24)
#endif

#ifndef SCAN_WARMUP
#define SCAN_WARMUP 2
#endif

#ifndef SCAN_REPEAT
#define SCAN_REPEAT 5
#endif

namespace {

using ScanLaunchFn = void (*)(const float *, float *, int);

struct ScanImplementation {
  const char *name;
  ScanLaunchFn launch;
};

struct ScanContext {
  int n;
  std::vector<float> host_input;
  std::vector<float> host_output;
  std::vector<float> host_ref;
  float *device_input;
  float *device_output;
};

static const ScanImplementation kScanImplementations[] = {
    {"naive", scan_naive}};

static void cpu_scan(const std::vector<float> &input,
                     std::vector<float> &output) {
  float running_sum = 0.0f;
  for (std::size_t i = 0; i < input.size(); ++i) {
    running_sum += input[i];
    output[i] = running_sum;
  }
}

static void prepare_scan_context(ScanContext &context) {
  context.n = SCAN_N;
  context.host_input.resize(context.n);
  context.host_output.assign(context.n, 0.0f);
  context.host_ref.assign(context.n, 0.0f);

  for (int i = 0; i < context.n; ++i) {
    context.host_input[i] = static_cast<float>((i % 37) - 18) * 0.125f;
  }

  cpu_scan(context.host_input, context.host_ref);

  context.device_input = nullptr;
  context.device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&context.device_input, sizeof(float) * context.n));
  CUDA_CHECK(cudaMalloc(&context.device_output, sizeof(float) * context.n));

  CUDA_CHECK(cudaMemcpy(context.device_input, context.host_input.data(),
                        sizeof(float) * context.n, cudaMemcpyHostToDevice));
}

static void release_scan_context(ScanContext &context) {
  CUDA_CHECK(cudaFree(context.device_input));
  CUDA_CHECK(cudaFree(context.device_output));
}

static bool
run_one_scan_implementation(const ScanImplementation &implementation,
                            ScanContext &context) {
  for (int i = 0; i < SCAN_WARMUP; ++i) {
    implementation.launch(context.device_input, context.device_output,
                          context.n);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  float best_ms = measure_min_elapsed_ms(SCAN_REPEAT, [&]() {
    implementation.launch(context.device_input, context.device_output,
                          context.n);
  });

  CUDA_CHECK(cudaMemcpy(context.host_output.data(), context.device_output,
                        sizeof(float) * context.n, cudaMemcpyDeviceToHost));
  bool passed =
      compare_arrays(context.host_output.data(), context.host_ref.data(),
                     context.n, 1e-3f, 1e-3f);

  const double work_units = static_cast<double>(context.n - 1);
  const double bytes = static_cast<double>(context.n) * sizeof(float) * 2.0;
  print_benchmark_line("scan", implementation.name, best_ms, work_units, "Add",
                       bytes, passed);
  return passed;
}

} // namespace

bool run_scan_benchmark() {
  ScanContext context{};
  prepare_scan_context(context);

  bool all_passed = true;
  for (const auto &implementation : kScanImplementations) {
    all_passed =
        run_one_scan_implementation(implementation, context) && all_passed;
  }

  release_scan_context(context);
  return all_passed;
}
