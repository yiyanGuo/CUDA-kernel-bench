#include "cute/arch/copy_sm80.hpp"
#include "cute/arch/mma_sm80.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/tensor.hpp"
#include "cutlass/half.h"

#include <torch/extension.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

using namespace cute;

template <typename T, int TileM, int TileN, int TileK, 
          typename sALayout, typename sBLayout,
          typename TiledMMA, typename TiledCopyA, typename TiledCopyB>
__global__ void hgemm_shared_memory_kernel(
    const T* Aptr,
    const T* Bptr,
    T* Cptr,
    int M, int N, int K,
    sALayout sA_layout, sBLayout sB_layout
) {
    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(N, K), make_stride(K, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(M, N), make_stride(N, Int<1>{}));   

    int tile_n = blockIdx.x;
    int tile_m = blockIdx.y;

    Tensor gA = local_tile(A, make_tile(Int<TileM>{}, Int<TileK>{}), make_coord(tile_m, _));
    Tensor gB = local_tile(B, make_tile(Int<TileN>{}, Int<TileK>{}), make_coord(tile_n, _));
    Tensor gC = local_tile(C, make_tile(Int<TileM>{}, Int<TileN>{}), make_coord(tile_m, tile_n));

    // shared memory
    __shared__ T smem_A[TileM * TileK];
    __shared__ T smem_B[TileN * TileK];

    Tensor sA = make_tensor(make_smem_ptr(smem_A), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smem_B), sB_layout);

    // global memory to shared memory
    TiledCopyA copyA;
    auto thr_copyA = copyA.get_slice(threadIdx.x);
    auto tAgA = thr_copyA.partition_S(gA);
    auto tAsA = thr_copyA.partition_D(sA);
    
    TiledCopyB copyB;
    auto thr_copyB = copyB.get_slice(threadIdx.x);
    auto tBgB = thr_copyB.partition_S(gB);
    auto tBsB = thr_copyB.partition_D(sB);

    // shared memory to register

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    auto tAsA_mma = thr_mma.partition_A(sA);
    auto tBsB_mma = thr_mma.partition_B(sB);

    auto tCgC = thr_mma.partition_C(gC);

    auto tCrA = thr_mma.partition_fragment_A(sA);
    auto tCrB = thr_mma.partition_fragment_B(sB);
    auto tCrC = thr_mma.make_fragment_C(tCgC);

    clear(tCrC);

    const int ktiles = K / TileK;
#pragma unroll
    for(int k = 0; k < ktiles; k++) {
        // global memory to shared memory
        cute::copy(copyA, tAgA(_, _, _, k), tAsA);
        cute::copy(copyB, tBgB(_, _, _, k), tBsB);

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // shared memory to registers
        cute::copy(tAsA_mma, tCrA);
        cute::copy(tBsB_mma, tCrB);

        cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
        __syncthreads();
    }

    cute::copy(tCrC, tCgC);
}



template<int kTileM, int kTileN, int kTileK>
static void launcher(
    const half_t* A,
    const half_t* B,
    half_t* C,
    int M,
    int N,
    int K
) {
    using mma_op = cute::SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = cute::MMA_Traits<mma_op>;
    using mma_atom = cute::MMA_Atom<mma_traits>;

    static constexpr int kMmaAtomRepeatM = 2;
    static constexpr int kMmaAtomRepeatN = 2;
    static constexpr int kMmaAtomRepeatK = 1;

    static constexpr int kPermutationM = 32;
    static constexpr int kPermutationN = 32;
    static constexpr int kPermutationK = 16;

    using MMA = decltype(cute::make_tiled_mma(
        mma_atom{},
        cute::make_layout(
            cute::make_shape(Int<kMmaAtomRepeatM>{}, Int<kMmaAtomRepeatN>{}, Int<kMmaAtomRepeatK>{})
        ),
        cute::make_tile(Int<kPermutationM>{}, Int<kPermutationN>{}, Int<kPermutationK>{})
    ));

    auto sA = make_layout(make_shape(Int<kTileM>{}, Int<kTileK>{}), make_stride(Int<kTileK>{}, Int<1>{})); // row-major
    auto sB = make_layout(make_shape(Int<kTileN>{}, Int<kTileK>{}), make_stride(Int<kTileK>{}, Int<1>{})); // row-major

    using copy_op = cute::UniversalCopy<uint128_t>;
    using copy_traits = cute::Copy_Traits<copy_op>;
    using copy_atom = cute::Copy_Atom<copy_traits, half_t>;

    using TiledCopyA = decltype(cute::make_tiled_copy(
        copy_atom{},
        cute::make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})), 
        cute::make_layout(make_shape(Int<1>{}, Int<8>{})) // 在 row 方向上 8 个 half
    ));
    using TiledCopyB = TiledCopyA;

    dim3 block(cute::size(MMA{}));
    dim3 grid(N / kTileN, M / kTileM);
    hgemm_shared_memory_kernel<
        cute::half_t, kTileM, kTileN, kTileK,
        decltype(sA), decltype(sB), 
        MMA, TiledCopyA, TiledCopyB
    >
    <<<grid, block>>>(
        A, B, C, M, N, K, sA, sB
    );

}

void hgemm_shared_memory(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
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
    TORCH_CHECK(m64 <= std::numeric_limits<int>::max(), "M is too large.");
    TORCH_CHECK(n64 <= std::numeric_limits<int>::max(), "N is too large.");
    TORCH_CHECK(k64 <= std::numeric_limits<int>::max(), "K is too large.");

    launcher<128, 128, 32>(reinterpret_cast<const cute::half_t*>(a.data_ptr()),
                reinterpret_cast<const cute::half_t*>(b.data_ptr()),
                reinterpret_cast<cute::half_t*>(out.data_ptr()),
                static_cast<int>(m64), static_cast<int>(n64),
                static_cast<int>(k64)
    );
} 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(hgemm_shared_memory)
}