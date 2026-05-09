#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <iostream>
#include <limits>
#include <stdexcept>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_VECTOR_ADD_TENSOR(T)                                             \
  TORCH_CHECK((T).is_cuda(), #T " must be a CUDA tensor.");                    \
  TORCH_CHECK((T).is_contiguous(), #T " must be contiguous.");                 \
  TORCH_CHECK((T).numel() <= std::numeric_limits<int>::max(),                  \
              #T " is too large for the current CUDA launcher.");

#define CHECK_VECTOR_ADD_INPUTS(a, b, out)                                     \
  CHECK_VECTOR_ADD_TENSOR(a)                                                   \
  CHECK_VECTOR_ADD_TENSOR(b)                                                   \
  CHECK_VECTOR_ADD_TENSOR(out)                                                 \
  TORCH_CHECK((a).sizes() == (b).sizes(), "a and b must have identical shapes."); \
  TORCH_CHECK((a).sizes() == (out).sizes(), "out must match the input shape."); \
  TORCH_CHECK((a).scalar_type() == (b).scalar_type(), "a and b dtype mismatch."); \
  TORCH_CHECK((a).scalar_type() == (out).scalar_type(), "out dtype mismatch.");

#define TORCH_BINDING_VECTOR_ADD_TYPED(func, th_type, element_type)            \
  void func(torch::Tensor a, torch::Tensor b, torch::Tensor out) {             \
    CHECK_VECTOR_ADD_INPUTS(a, b, out)                                         \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(out, (th_type))                                   \
    const int n = static_cast<int>(out.numel());                               \
    func##_launcher(reinterpret_cast<element_type*>(a.data_ptr()),             \
                    reinterpret_cast<element_type*>(b.data_ptr()),             \
                    reinterpret_cast<element_type*>(out.data_ptr()), n);       \
  }

namespace {

constexpr int kThreadsPerBlock = 256;

template <typename scalar_t>
__global__ void vector_add_kernel(const scalar_t* a, const scalar_t* b,
                                  scalar_t* out, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    out[index] = a[index] + b[index];
  }
}

__global__ void vector_add_f32x4_kernel(const float* a, const float* b,
                                        float* out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int index = tid * 4;

  if (index + 3 < n) {
    float4 av = reinterpret_cast<const float4*>(a)[tid];
    float4 bv = reinterpret_cast<const float4*>(b)[tid];
    reinterpret_cast<float4*>(out)[tid] =
        make_float4(av.x + bv.x, av.y + bv.y, av.z + bv.z, av.w + bv.w);
    return;
  }

  for (int offset = 0; offset < 4 && index + offset < n; ++offset) {
    out[index + offset] = a[index + offset] + b[index + offset];
  }
}

__global__ void vector_add_f16x2_kernel(const half* a, const half* b,
                                        half* out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int index = tid * 2;

  if (index + 1 < n) {
    half2 av = reinterpret_cast<const half2*>(a)[tid];
    half2 bv = reinterpret_cast<const half2*>(b)[tid];
    reinterpret_cast<half2*>(out)[tid] = __hadd2(av, bv);
    return;
  }

  if (index < n) {
    out[index] = __hadd(a[index], b[index]);
  }
}

template <typename scalar_t>
void vector_add_launcher(const scalar_t* a, const scalar_t* b, scalar_t* out,
                         int n) {
  dim3 block(kThreadsPerBlock);
  dim3 grid((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
  vector_add_kernel<scalar_t><<<grid, block>>>(a, b, out, n);
}

void vector_add_f32x4_launcher(float* a, float* b, float* out, int n) {
  dim3 block(kThreadsPerBlock);
  dim3 grid((n + kThreadsPerBlock * 4 - 1) / (kThreadsPerBlock * 4));
  vector_add_f32x4_kernel<<<grid, block>>>(a, b, out, n);
}

void vector_add_f16x2_launcher(half* a, half* b, half* out, int n) {
  dim3 block(kThreadsPerBlock);
  dim3 grid((n + kThreadsPerBlock * 2 - 1) / (kThreadsPerBlock * 2));
  vector_add_f16x2_kernel<<<grid, block>>>(a, b, out, n);
}

}  // namespace

void vector_add(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  CHECK_VECTOR_ADD_INPUTS(a, b, out)

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      a.scalar_type(), "vector_add", [&] {
        const int n = static_cast<int>(out.numel());
        vector_add_launcher<scalar_t>(a.data_ptr<scalar_t>(),
                                      b.data_ptr<scalar_t>(),
                                      out.data_ptr<scalar_t>(), n);
      });
}

TORCH_BINDING_VECTOR_ADD_TYPED(vector_add_f32x4, torch::kFloat32, float)
TORCH_BINDING_VECTOR_ADD_TYPED(vector_add_f16x2, torch::kHalf, half)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(vector_add)
  TORCH_BINDING_COMMON_EXTENSION(vector_add_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(vector_add_f16x2)
}
