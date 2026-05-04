#include <torch/extension.h>

#include <limits>

#include "softmax.h"

namespace {

void check_softmax_tensor(const torch::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name,
              " must use float32 dtype.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  TORCH_CHECK(tensor.dim() == 4, name,
              " must be 4D (batch_size, num_heads, query_len, key_len).");
  TORCH_CHECK(tensor.numel() <= std::numeric_limits<int>::max(), name,
              " is too large for the current CUDA launcher.");
}

} // namespace

void softmax_naive_binding(const torch::Tensor &logits, torch::Tensor &output) {
  check_softmax_tensor(logits, "logits");
  check_softmax_tensor(output, "output");
  TORCH_CHECK(logits.sizes() == output.sizes(),
              "output must match logits shape.");

  softmax_naive(logits.data_ptr<float>(), output.data_ptr<float>(),
                static_cast<int>(logits.size(0)),
                static_cast<int>(logits.size(1)),
                static_cast<int>(logits.size(2)),
                static_cast<int>(logits.size(3)));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax_naive", &softmax_naive_binding);
}
