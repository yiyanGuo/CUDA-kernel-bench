#include <torch/extension.h>

#include <limits>

#include "transpose.h"

namespace {

void check_matrix_tensor(const torch::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat,
              name, " must use float32 dtype.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  TORCH_CHECK(tensor.dim() == 2, name, " must be 2D.");
  TORCH_CHECK(tensor.numel() <= std::numeric_limits<int>::max(),
              name, " is too large for the current CUDA launcher.");
}

void check_output_tensor(const torch::Tensor &input, const torch::Tensor &output) {
  check_matrix_tensor(output, "output");
  TORCH_CHECK(output.size(0) == input.size(1) && output.size(1) == input.size(0),
              "output must have transposed dimensions.");
}

} // namespace

void transpose_naive_binding(const torch::Tensor &input, torch::Tensor &output) {
  check_matrix_tensor(input, "input");
  check_output_tensor(input, output);
  transpose_naive(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      static_cast<int>(input.size(0)),
      static_cast<int>(input.size(1)));
}

void transpose_tile_float4_binding(const torch::Tensor &input,
                                   torch::Tensor &output) {
  check_matrix_tensor(input, "input");
  check_output_tensor(input, output);
  transpose_tile_float4(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      static_cast<int>(input.size(0)),
      static_cast<int>(input.size(1)));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transpose_naive", &transpose_naive_binding);
  m.def("transpose_tile_float4", &transpose_tile_float4_binding);
}
