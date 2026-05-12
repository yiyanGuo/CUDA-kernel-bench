#include <cuda_fp16.h>
#include <torch/extension.h>

#include <limits>

#define STRINGIFY_DETAIL(x) #x
#define STRINGIFY(x) STRINGIFY_DETAIL(x)

void KERNEL_SYMBOL(const half* input, const half* weight, half* output,
                   int num_tokens, int hidden_size, float eps);

namespace {

void check_matrix_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::kHalf, name,
              " must use float16 dtype.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  TORCH_CHECK(tensor.dim() == 2, name, " must be 2D (tokens x hidden).");
  TORCH_CHECK(tensor.numel() <= std::numeric_limits<int>::max(), name,
              " is too large for the current CUDA launcher.");
}

void check_vector_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::kHalf, name,
              " must use float16 dtype.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1D.");
  TORCH_CHECK(tensor.numel() <= std::numeric_limits<int>::max(), name,
              " is too large for the current CUDA launcher.");
}

void launch_binding(const torch::Tensor& input, const torch::Tensor& weight,
                    torch::Tensor& out, double eps) {
  check_matrix_tensor(input, "input");
  check_vector_tensor(weight, "weight");
  check_matrix_tensor(out, "out");

  TORCH_CHECK(input.size(1) == weight.size(0),
              "hidden size must match weight length.");
  TORCH_CHECK(input.sizes() == out.sizes(), "out must match input shape.");

  int num_tokens = static_cast<int>(input.size(0));
  int hidden_size = static_cast<int>(input.size(1));
  KERNEL_SYMBOL(reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
                reinterpret_cast<half*>(out.data_ptr<at::Half>()), num_tokens,
                hidden_size, static_cast<float>(eps));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(STRINGIFY(KERNEL_SYMBOL), &launch_binding);
}
