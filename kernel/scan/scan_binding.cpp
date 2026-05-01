#include <torch/extension.h>

#include <limits>

#include "scan.h"

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

#define BIND_SCAN_FN(name)                                                     \
  void name##_binding(const torch::Tensor &input, torch::Tensor &output) {     \
    check_vector_tensor(input, "input");                                        \
    check_vector_tensor(output, "output");                                      \
    TORCH_CHECK(input.sizes() == output.sizes(),                                \
                "output must match the input shape.");                          \
    name(input.data_ptr<float>(), output.data_ptr<float>(),                     \
         static_cast<int>(input.numel()));                                      \
  }

BIND_SCAN_FN(scan_naive)
BIND_SCAN_FN(scan_one_block)
BIND_SCAN_FN(scan_multi_block)
BIND_SCAN_FN(scan_thrust_exclusive)
BIND_SCAN_FN(scan_warp)
BIND_SCAN_FN(scan_memory_buffer)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scan_naive", &scan_naive_binding);
  m.def("scan_one_block", &scan_one_block_binding);
  m.def("scan_multi_block", &scan_multi_block_binding);
  m.def("scan_thrust_exclusive", &scan_thrust_exclusive_binding);
  m.def("scan_warp", &scan_warp_binding);
  m.def("scan_memory_buffer", &scan_memory_buffer_binding);
}
