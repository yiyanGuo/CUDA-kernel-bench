#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/numeric/numeric_types.hpp>
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

namespace {

constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileK = 32;

template <typename T, int TileM, int TileN, int TileK, typename TiledMMA>
__global__ void gemm_simple(T* Cptr, const T* Aptr, const T* Bptr, int m,
                            int n, int k) {
  using namespace cute;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n),
                         make_stride(n, Int<1>{}));

  int tile_n = blockIdx.x;
  int tile_m = blockIdx.y;

  Tensor gA = local_tile(A, make_tile(Int<TileM>{}, Int<TileK>{}),
                         make_coord(tile_m, _));
  Tensor gB = local_tile(B, make_tile(Int<TileN>{}, Int<TileK>{}),
                         make_coord(tile_n, _));
  Tensor gC = local_tile(C, make_tile(Int<TileM>{}, Int<TileN>{}),
                         make_coord(tile_m, tile_n));

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tAgA = thr_mma.partition_A(gA);
  auto tBgB = thr_mma.partition_B(gB);
  auto tCgC = thr_mma.partition_C(gC);

  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));
  auto tCrC = thr_mma.make_fragment_C(tCgC);

  clear(tCrC);

  int num_tile_k = size<2>(gA);
#pragma unroll
  for (int itile = 0; itile < num_tile_k; itile++) {
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);
    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  cute::copy(tCrC, tCgC);
}

using HgemmMma = decltype(cute::make_tiled_mma(
    cute::SM80_16x8x16_F32F16F16F32_TN{},
    cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>{},
    cute::Tile<cute::_32, cute::_32, cute::_16>{}));

void hgemm_simple_launcher(const cute::half_t* a, const cute::half_t* b,
                           cute::half_t* out, int m, int n, int k) {
  dim3 block(cute::size(HgemmMma{}));
  dim3 grid(n / kTileN, m / kTileM);
  gemm_simple<cute::half_t, kTileM, kTileN, kTileK, HgemmMma>
      <<<grid, block>>>(out, a, b, m, n, k);
}

}  // namespace

void hgemm_simple(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor.");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor.");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor.");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous.");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous.");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous.");
  TORCH_CHECK(a.dim() == 2, "a must be a 2D tensor.");
  TORCH_CHECK(b.dim() == 2, "b must be a 2D tensor.");
  TORCH_CHECK(out.dim() == 2, "out must be a 2D tensor.");
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(out, torch::kHalf)

  const int64_t m64 = a.size(0);
  const int64_t k64 = a.size(1);
  const int64_t n64 = b.size(0);
  TORCH_CHECK(b.size(1) == k64, "b must have shape [N, K].");
  TORCH_CHECK(out.size(0) == m64 && out.size(1) == n64,
              "out must have shape [M, N].");
  TORCH_CHECK(m64 > 0 && n64 > 0 && k64 > 0, "M, N, and K must be positive.");
  TORCH_CHECK(m64 % kTileM == 0, "M must be a multiple of 128.");
  TORCH_CHECK(n64 % kTileN == 0, "N must be a multiple of 128.");
  TORCH_CHECK(k64 % kTileK == 0, "K must be a multiple of 32.");
  TORCH_CHECK(m64 <= std::numeric_limits<int>::max(), "M is too large.");
  TORCH_CHECK(n64 <= std::numeric_limits<int>::max(), "N is too large.");
  TORCH_CHECK(k64 <= std::numeric_limits<int>::max(), "K is too large.");

  hgemm_simple_launcher(reinterpret_cast<const cute::half_t*>(a.data_ptr()),
                        reinterpret_cast<const cute::half_t*>(b.data_ptr()),
                        reinterpret_cast<cute::half_t*>(out.data_ptr()),
                        static_cast<int>(m64), static_cast<int>(n64),
                        static_cast<int>(k64));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(hgemm_simple)
}
