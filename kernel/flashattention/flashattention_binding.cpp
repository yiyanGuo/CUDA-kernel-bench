#include <torch/extension.h>

#include <limits>

#include "ATen/core/TensorBody.h"
#include "flashattention.h"

namespace {

constexpr int kBlockM = 32;
constexpr int kBlockN = 32;
constexpr int kMmaBlockM = 64;
constexpr int kMmaBlockN = 64;
constexpr int kHeadDim = 64;

void check_attention_tensor(const torch::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name,
              " must use float32 dtype.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  TORCH_CHECK(tensor.dim() == 4, name,
              " must be 4D (batch_size, num_heads, seq_len, head_dim).");
  TORCH_CHECK(tensor.size(3) == kHeadDim,
              name, " must use head_dim == 64.");
  TORCH_CHECK(tensor.numel() <= std::numeric_limits<int>::max(), name,
              " is too large for the current CUDA launcher.");
}

void check_attention_half_tensor(const torch::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::kHalf, name,
              " must use float16 dtype.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  TORCH_CHECK(tensor.dim() == 4, name,
              " must be 4D (batch_size, num_heads, seq_len, head_dim).");
  TORCH_CHECK(tensor.size(3) == kHeadDim,
              name, " must use head_dim == 64.");
  TORCH_CHECK(tensor.numel() <= std::numeric_limits<int>::max(), name,
              " is too large for the current CUDA launcher.");
}

} // namespace

void flash_attention_binding(const torch::Tensor &q,
                             const torch::Tensor &k,
                             const torch::Tensor &v,
                             torch::Tensor &output) {
  check_attention_tensor(q, "q");
  check_attention_tensor(k, "k");
  check_attention_tensor(v, "v");
  check_attention_tensor(output, "output");

  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
              "q, k, and v must have the same batch_size.");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1),
              "q, k, and v must have the same num_heads.");
  TORCH_CHECK(k.size(2) == v.size(2),
              "k and v must have the same key_len.");
  TORCH_CHECK(output.sizes() == q.sizes(), "output must match q shape.");
  TORCH_CHECK(q.size(2) % kBlockM == 0,
              "query_len must be a multiple of 32 for flashattention_naive.");
  TORCH_CHECK(k.size(2) % kBlockN == 0,
              "key_len must be a multiple of 32 for flashattention_naive.");

  TORCH_CHECK(q.size(2) / kBlockM <= std::numeric_limits<int>::max(),
              "grid x dimension is too large for flashattention_naive.");

  flash_attention(q.data_ptr<float>(), k.data_ptr<float>(),
                  v.data_ptr<float>(), output.data_ptr<float>(),
                  static_cast<int>(q.size(0)),
                  static_cast<int>(q.size(1)),
                  static_cast<int>(q.size(2)),
                  static_cast<int>(k.size(2)));
}

void flash_attention_mma_binding(const torch::Tensor &q,
                                 const torch::Tensor &k,
                                 const torch::Tensor &v,
                                 torch::Tensor &output) {
  check_attention_half_tensor(q, "q");
  check_attention_half_tensor(k, "k");
  check_attention_half_tensor(v, "v");
  check_attention_half_tensor(output, "output");

  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
              "q, k, and v must have the same batch_size.");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1),
              "q, k, and v must have the same num_heads.");
  TORCH_CHECK(k.size(2) == v.size(2),
              "k and v must have the same key_len.");
  TORCH_CHECK(output.sizes() == q.sizes(), "output must match q shape.");
  TORCH_CHECK(q.size(2) % kMmaBlockM == 0,
              "query_len must be a multiple of 64 for flashattention_mma.");
  TORCH_CHECK(k.size(2) % kMmaBlockN == 0,
              "key_len must be a multiple of 64 for flashattention_mma.");

  TORCH_CHECK(q.size(2) / kMmaBlockM <= std::numeric_limits<int>::max(),
              "grid x dimension is too large for flashattention_mma.");

  flash_attention_mma(reinterpret_cast<const half *>(q.data_ptr<at::Half>()),
                      reinterpret_cast<const half *>(k.data_ptr<at::Half>()),
                      reinterpret_cast<const half *>(v.data_ptr<at::Half>()),
                      reinterpret_cast<half *>(output.data_ptr<at::Half>()),
                      static_cast<int>(q.size(0)),
                      static_cast<int>(q.size(1)),
                      static_cast<int>(q.size(2)),
                      static_cast<int>(k.size(2)));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_attention", &flash_attention_binding,
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("output"));
  m.def("flash_attention_mma", &flash_attention_mma_binding,
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("output"));
}
