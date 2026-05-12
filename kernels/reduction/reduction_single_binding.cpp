#include <torch/extension.h>

#include <limits>

#define STRINGIFY_DETAIL(x) #x
#define STRINGIFY(x) STRINGIFY_DETAIL(x)

void KERNEL_SYMBOL(const float* input, float* output, int n);

namespace {

void check_vector_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat,
              name, " must use float32 dtype.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1D.");
  TORCH_CHECK(tensor.numel() <= std::numeric_limits<int>::max(),
              name, " is too large for the current CUDA launcher.");
}

void check_scalar_output(const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.is_cuda(), "output must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat,
              "output must use float32 dtype.");
  TORCH_CHECK(tensor.is_contiguous(), "output must be contiguous.");
  TORCH_CHECK(tensor.numel() == 1, "output must contain exactly one element.");
}

void launch_binding(const torch::Tensor& input, torch::Tensor& output) {
  check_vector_tensor(input, "input");
  check_scalar_output(output);
  KERNEL_SYMBOL(input.data_ptr<float>(), output.data_ptr<float>(),
                static_cast<int>(input.numel()));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(STRINGIFY(KERNEL_SYMBOL), &launch_binding);
}
