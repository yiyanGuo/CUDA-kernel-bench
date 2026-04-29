#include <torch/extension.h>

#include <limits>

#include "reduction.h"

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

void check_scalar_output(const torch::Tensor &tensor) {
  TORCH_CHECK(tensor.is_cuda(), "output must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat,
              "output must use float32 dtype.");
  TORCH_CHECK(tensor.is_contiguous(), "output must be contiguous.");
  TORCH_CHECK(tensor.numel() == 1, "output must contain exactly one element.");
}

} // namespace

#define BIND_REDUCTION_FN(name)                                                \
  void name##_binding(const torch::Tensor &input, torch::Tensor &output) {     \
    check_vector_tensor(input, "input");                                        \
    check_scalar_output(output);                                                \
    name(input.data_ptr<float>(), output.data_ptr<float>(),                     \
         static_cast<int>(input.numel()));                                      \
  }

BIND_REDUCTION_FN(reduction_naive)
BIND_REDUCTION_FN(reduction_presum)
BIND_REDUCTION_FN(reduction_presum_float4)
BIND_REDUCTION_FN(reduction_shuffle)
BIND_REDUCTION_FN(reduction_grid_stride)
BIND_REDUCTION_FN(reduction_integrate)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reduction_naive", &reduction_naive_binding);
  m.def("reduction_presum", &reduction_presum_binding);
  m.def("reduction_presum_float4", &reduction_presum_float4_binding);
  m.def("reduction_shuffle", &reduction_shuffle_binding);
  m.def("reduction_grid_stride", &reduction_grid_stride_binding);
  m.def("reduction_integrate", &reduction_integrate_binding);
}
