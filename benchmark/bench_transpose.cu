#include <bench_common.h>

#include <vector>

#ifndef TRANSPOSE_ROWS
#define TRANSPOSE_ROWS 2048
#endif

#ifndef TRANSPOSE_COLS
#define TRANSPOSE_COLS 2048
#endif

#ifndef TRANSPOSE_WARMUP
#define TRANSPOSE_WARMUP 2
#endif

#ifndef TRANSPOSE_REPEAT
#define TRANSPOSE_REPEAT 5
#endif

void transpose_naive(const float *input, float *output, int rows, int cols);

namespace {

using TransposeLaunchFn = void (*)(const float *, float *, int, int);

struct TransposeImplementation {
  const char *name;
  TransposeLaunchFn launch;
};

struct TransposeContext {
  int rows;
  int cols;
  int element_count;
  std::vector<float> host_input;
  std::vector<float> host_output;
  std::vector<float> host_ref;
  float *device_input;
  float *device_output;
};

static const TransposeImplementation kTransposeImplementations[] = {
    {"naive", transpose_naive},
};

static void cpu_transpose(const std::vector<float> &input,
                          std::vector<float> &output, int rows, int cols) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      output[col * rows + row] = input[row * cols + col];
    }
  }
}

static void prepare_transpose_context(TransposeContext &context) {
  context.rows = TRANSPOSE_ROWS;
  context.cols = TRANSPOSE_COLS;
  context.element_count = context.rows * context.cols;
  context.host_input.resize(context.element_count);
  context.host_output.assign(context.element_count, 0.0f);
  context.host_ref.assign(context.element_count, 0.0f);

  for (int i = 0; i < context.element_count; ++i) {
    context.host_input[i] = static_cast<float>((i * 17) % 113) * 0.125f;
  }

  cpu_transpose(context.host_input, context.host_ref, context.rows,
                context.cols);

  context.device_input = nullptr;
  context.device_output = nullptr;
  CUDA_CHECK(
      cudaMalloc(&context.device_input, sizeof(float) * context.element_count));
  CUDA_CHECK(cudaMalloc(&context.device_output,
                        sizeof(float) * context.element_count));

  CUDA_CHECK(cudaMemcpy(context.device_input, context.host_input.data(),
                        sizeof(float) * context.element_count,
                        cudaMemcpyHostToDevice));
}

static void release_transpose_context(TransposeContext &context) {
  CUDA_CHECK(cudaFree(context.device_input));
  CUDA_CHECK(cudaFree(context.device_output));
}

static bool
run_one_transpose_implementation(const TransposeImplementation &implementation,
                                 TransposeContext &context) {
  for (int i = 0; i < TRANSPOSE_WARMUP; ++i) {
    implementation.launch(context.device_input, context.device_output,
                          context.rows, context.cols);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  float best_ms = measure_min_elapsed_ms(TRANSPOSE_REPEAT, [&]() {
    implementation.launch(context.device_input, context.device_output,
                          context.rows, context.cols);
  });

  CUDA_CHECK(cudaMemcpy(context.host_output.data(), context.device_output,
                        sizeof(float) * context.element_count,
                        cudaMemcpyDeviceToHost));
  bool passed = compare_arrays(context.host_output.data(),
                               context.host_ref.data(), context.element_count);

  const double work_units = static_cast<double>(context.element_count);
  const double bytes =
      static_cast<double>(context.element_count) * sizeof(float) * 2.0;
  print_benchmark_line("transpose", implementation.name, best_ms, work_units,
                       "Elem", bytes, passed);
  return passed;
}

} // namespace

bool run_transpose_benchmark() {
  TransposeContext context{};
  prepare_transpose_context(context);

  bool all_passed = true;
  for (const auto &implementation : kTransposeImplementations) {
    all_passed =
        run_one_transpose_implementation(implementation, context) && all_passed;
  }

  release_transpose_context(context);
  return all_passed;
}
