#include <cuda_fp16.h>
#include <torch/extension.h>

#include <limits>

#define STRINGIFY_DETAIL(x) #x
#define STRINGIFY(x) STRINGIFY_DETAIL(x)

#ifdef USE_HALF_INPUT
using scalar_t = half;
static constexpr at::ScalarType kTorchDtype = at::kHalf;
#else
using scalar_t = float;
static constexpr at::ScalarType kTorchDtype = at::kFloat;
#endif

void KERNEL_SYMBOL(const scalar_t* q, const scalar_t* k, const scalar_t* v,
                   scalar_t* output, int batch_size, int num_heads,
                   int query_len, int key_len);

namespace {

constexpr int kHeadDim = 64;

void check_attention_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == kTorchDtype, name,
              " has unexpected dtype.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  TORCH_CHECK(tensor.dim() == 4, name,
              " must be 4D (batch_size, num_heads, seq_len, head_dim).");
  TORCH_CHECK(tensor.size(3) == kHeadDim,
              name, " must use head_dim == 64.");
  TORCH_CHECK(tensor.numel() <= std::numeric_limits<int>::max(), name,
              " is too large for the current CUDA launcher.");
}

void launch_binding(const torch::Tensor& q, const torch::Tensor& k,
                    const torch::Tensor& v, torch::Tensor& output) {
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
#ifdef USE_HALF_INPUT
  TORCH_CHECK(q.size(2) % 64 == 0,
              "query_len must be a multiple of 64 for this implementation.");
  TORCH_CHECK(k.size(2) % 64 == 0,
              "key_len must be a multiple of 64 for this implementation.");
#else
  TORCH_CHECK(q.size(2) % 32 == 0,
              "query_len must be a multiple of 32 for this implementation.");
  TORCH_CHECK(k.size(2) % 32 == 0,
              "key_len must be a multiple of 32 for this implementation.");
#endif

  KERNEL_SYMBOL(reinterpret_cast<const scalar_t*>(q.data_ptr()),
                reinterpret_cast<const scalar_t*>(k.data_ptr()),
                reinterpret_cast<const scalar_t*>(v.data_ptr()),
                reinterpret_cast<scalar_t*>(output.data_ptr()),
                static_cast<int>(q.size(0)), static_cast<int>(q.size(1)),
                static_cast<int>(q.size(2)), static_cast<int>(k.size(2)));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(STRINGIFY(KERNEL_SYMBOL), &launch_binding,
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("output"));
}
