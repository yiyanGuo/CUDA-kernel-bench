#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <limits>

namespace {

void check_cublas(cublasStatus_t status) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLAS call failed: ", status);
}

cublasHandle_t get_cublas_handle() {
  static cublasHandle_t handle = [] {
    cublasHandle_t h;
    check_cublas(cublasCreate(&h));
    check_cublas(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));
    return h;
  }();
  return handle;
}

void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  TORCH_CHECK(tensor.scalar_type() == torch::kHalf, name, " must be float16.");
  TORCH_CHECK(tensor.dim() == 2, name, " must be a 2D tensor.");
}

}  // namespace

void hgemm_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  check_tensor(a, "a");
  check_tensor(b, "b");
  check_tensor(out, "out");

  const int64_t m64 = a.size(0);
  const int64_t k64 = a.size(1);
  const int64_t n64 = b.size(0);
  TORCH_CHECK(b.size(1) == k64, "b must have shape [N, K].");
  TORCH_CHECK(out.size(0) == m64 && out.size(1) == n64,
              "out must have shape [M, N].");
  TORCH_CHECK(m64 <= std::numeric_limits<int>::max(), "M is too large.");
  TORCH_CHECK(n64 <= std::numeric_limits<int>::max(), "N is too large.");
  TORCH_CHECK(k64 <= std::numeric_limits<int>::max(), "K is too large.");

  const int m = static_cast<int>(m64);
  const int n = static_cast<int>(n64);
  const int k = static_cast<int>(k64);
  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasHandle_t handle = get_cublas_handle();
  check_cublas(cublasSetStream(handle, at::cuda::getCurrentCUDAStream()));

  // Row-major out[M, N] is column-major out^T[N, M]. Compute out^T = b * a^T.
  check_cublas(cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N,
      n, m, k,
      &alpha,
      b.data_ptr(), CUDA_R_16F, k,
      a.data_ptr(), CUDA_R_16F, k,
      &beta,
      out.data_ptr(), CUDA_R_16F, n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hgemm_cublas", &hgemm_cublas, "hgemm_cublas");
}
