#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cute/tensor.hpp>

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

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kNumElemPerThread = 4;

template <int kElemsPerThread, typename scalar_t>
__global__ void vector_add_cute_kernel(const scalar_t* a, const scalar_t* b,
                                       scalar_t* out, int n) {
  using namespace cute;

  const int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int elem_begin = tile_idx * kElemsPerThread;
  if (elem_begin >= n) {
    return;
  }

  auto tensor_shape = make_shape(n);
  auto ga = make_tensor(make_gmem_ptr(a), tensor_shape);
  auto gb = make_tensor(make_gmem_ptr(b), tensor_shape);
  auto gout = make_tensor(make_gmem_ptr(out), tensor_shape);

  auto tile_shape = make_shape(Int<kElemsPerThread>{});
  auto ta = local_tile(ga, tile_shape, make_coord(tile_idx));
  auto tb = local_tile(gb, tile_shape, make_coord(tile_idx));
  auto tout = local_tile(gout, tile_shape, make_coord(tile_idx));

#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    if (elem_begin + i < n) {
      tout(i) = ta(i) + tb(i);
    }
  }
}

template <typename scalar_t>
void vector_add_cute_x4_launcher(const scalar_t* a, const scalar_t* b,
                                 scalar_t* out, int n) {
  dim3 block(kThreadsPerBlock);
  dim3 grid((n + kThreadsPerBlock * kNumElemPerThread - 1) /
            (kThreadsPerBlock * kNumElemPerThread));
  vector_add_cute_kernel<kNumElemPerThread, scalar_t>
      <<<grid, block>>>(a, b, out, n);
}

}  // namespace

void vector_add_cute_x4(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  CHECK_VECTOR_ADD_INPUTS(a, b, out)

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      a.scalar_type(), "vector_add_cute_x4", [&] {
        const int n = static_cast<int>(out.numel());
        vector_add_cute_x4_launcher<scalar_t>(a.data_ptr<scalar_t>(),
                                              b.data_ptr<scalar_t>(),
                                              out.data_ptr<scalar_t>(), n);
      });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(vector_add_cute_x4)
}
