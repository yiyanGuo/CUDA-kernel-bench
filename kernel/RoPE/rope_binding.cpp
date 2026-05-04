#include <torch/extension.h>

#include <limits>

#include "rope.h"

namespace {

void check_input_tensor(const torch::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name,
              " must use float32 dtype.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  TORCH_CHECK(tensor.dim() == 4,
              name, " must be 4D (batch x seq_len x heads x head_dim).");
  TORCH_CHECK(tensor.numel() <= std::numeric_limits<int>::max(), name,
              " is too large for the current CUDA launcher.");
}

void check_freq_tensor(const torch::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name,
              " must use float32 dtype.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  TORCH_CHECK(tensor.dim() == 2,
              name, " must be 2D (positions x rotary_dim / 2).");
  TORCH_CHECK(tensor.numel() <= std::numeric_limits<int>::max(), name,
              " is too large for the current CUDA launcher.");
}

void check_position_ids_tensor(const torch::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::kInt, name,
              " must use int32 dtype.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  TORCH_CHECK(tensor.dim() == 2, name, " must be 2D (batch x seq_len).");
  TORCH_CHECK(tensor.numel() <= std::numeric_limits<int>::max(), name,
              " is too large for the current CUDA launcher.");
}

} // namespace

void rope_naive_binding(torch::Tensor &q, torch::Tensor &k,
                        const torch::Tensor &cos, const torch::Tensor &sin,
                        const torch::Tensor &position_ids, int rotary_dim,
                        int position_offset) {
  check_input_tensor(q, "q");
  check_input_tensor(k, "k");
  check_freq_tensor(cos, "cos");
  check_freq_tensor(sin, "sin");
  check_position_ids_tensor(position_ids, "position_ids");

  TORCH_CHECK(q.size(0) == k.size(0), "q and k batch sizes must match.");
  TORCH_CHECK(q.size(1) == k.size(1), "q and k seq_len must match.");
  TORCH_CHECK(q.size(3) == k.size(3), "q and k head_dim must match.");
  TORCH_CHECK(q.size(3) % 2 == 0, "head_dim must be even.");
  TORCH_CHECK(position_ids.size(0) == q.size(0),
              "position_ids batch size must match q/k.");
  TORCH_CHECK(position_ids.size(1) == q.size(1),
              "position_ids seq_len must match q/k.");
  TORCH_CHECK(cos.sizes() == sin.sizes(), "cos and sin must have the same shape.");
  TORCH_CHECK(rotary_dim > 0, "rotary_dim must be > 0.");
  TORCH_CHECK(rotary_dim <= q.size(3), "rotary_dim must be <= head_dim.");
  TORCH_CHECK(rotary_dim % 2 == 0, "rotary_dim must be even.");
  TORCH_CHECK(cos.size(1) == rotary_dim / 2,
              "cos/sin width must equal rotary_dim / 2.");
  TORCH_CHECK(position_offset >= 0, "position_offset must be >= 0.");
  TORCH_CHECK(position_offset < cos.size(0),
              "position_offset must fit cos/sin positions.");

  rope_naive(q.data_ptr<float>(), k.data_ptr<float>(), cos.data_ptr<float>(),
             sin.data_ptr<float>(), position_ids.data_ptr<int>(),
             static_cast<int>(q.size(0)), static_cast<int>(q.size(1)),
             static_cast<int>(q.size(2)), static_cast<int>(k.size(2)),
             static_cast<int>(q.size(3)), rotary_dim, position_offset);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_naive", &rope_naive_binding);
}
