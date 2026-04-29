#include <torch/extension.h>

#include <limits>

#include "vector_add.h"

namespace {

void check_vector_tensor(const torch::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat,
              name, " must use float32 dtype.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1D.");
  TORCH_CHECK(tensor.numel() <= std::numeric_limits<int>::max(),
              name, " is too large for the current CUDA launcher.");
}

} // namespace

void vector_add_naive_binding(const torch::Tensor &a, const torch::Tensor &b,
                              torch::Tensor &out) {
  check_vector_tensor(a, "a");
  check_vector_tensor(b, "b");
  check_vector_tensor(out, "out");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have identical shapes.");
  TORCH_CHECK(a.sizes() == out.sizes(), "out must match the input shape.");
  vector_add_naive(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
                   static_cast<int>(a.numel()));
}

void vector_add_float4_binding(const torch::Tensor &a, const torch::Tensor &b,
                               torch::Tensor &out) {
  check_vector_tensor(a, "a");
  check_vector_tensor(b, "b");
  check_vector_tensor(out, "out");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have identical shapes.");
  TORCH_CHECK(a.sizes() == out.sizes(), "out must match the input shape.");
  vector_add_float4(a.data_ptr<float>(), b.data_ptr<float>(),
                    out.data_ptr<float>(), static_cast<int>(a.numel()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_add_naive", &vector_add_naive_binding);
  m.def("vector_add_float4", &vector_add_float4_binding);
}
