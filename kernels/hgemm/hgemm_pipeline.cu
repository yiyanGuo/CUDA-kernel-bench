#include "cute/tensor.hpp"
#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/arch/mma_sm80.hpp"
#include "cute/numeric/int.hpp"
#include "cute/numeric/numeric_types.hpp"
#include <limits>
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

template <int Stages>
__device__ __forceinline__ void cp_async_wait_for_current(int groups_to_keep) {
    static_assert(Stages >= 2 && Stages <= 4, "Stages must be 2, 3, or 4.");
    if constexpr (Stages == 2) {
        cp_async_wait<0>();
    } else if constexpr (Stages == 3) {
        if (groups_to_keep > 0) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
    } else {
        if (groups_to_keep > 1) {
            cp_async_wait<2>();
        } else if (groups_to_keep > 0) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
    }
}

template <
    int TileM, int TileN, int TileK, int Stages,
    typename SmemLayoutA, typename SmemLayoutB,
    typename TiledMMA,
    typename G2SCopyA, typename G2SCopyB,
    typename S2RCopyAtom>
__global__ void hgemm_pipeline_kernel(
    const half_t* Aptr,
    const half_t* Bptr,
    half_t* Cptr,
    int M,
    int N,
    int K,
    SmemLayoutA sA_layout,
    SmemLayoutB sB_layout
) {
    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_layout(
        make_shape(M, K),
        make_stride(K, Int<1>{})
    ));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_layout(
        make_shape(N, K),
        make_stride(K, Int<1>{})
    ));
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_layout(
        make_shape(M, N),
        make_stride(N, Int<1>{})
    ));

    int ix = blockIdx.x;
    int iy = blockIdx.y;
    Tensor gA = local_tile(A, make_tile(Int<TileM>{}, Int<TileK>{}), make_coord(iy, _)); //(TileM, TileK, nk)
    Tensor gB = local_tile(B, make_tile(Int<TileN>{}, Int<TileK>{}), make_coord(ix, _)); //(TileN, TileK, nk)
    Tensor gC = local_tile(C, make_tile(Int<TileM>{}, Int<TileN>{}), make_coord(iy, ix));//(TileM, TileN)

    // shared memory
    extern __shared__ char shared_memory[];
    half_t* smemA = reinterpret_cast<half_t*>(shared_memory);
    half_t* smemB = smemA + cosize(sA_layout);

    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);
    
    // g2s tile
    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(threadIdx.x);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CP, CP_M, CP_K, nk)
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA); // (CP, CP_M, CP_K, stages)

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(threadIdx.x);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CP, CP_N, CP_K, nk)
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB); // (CP, CP_N, CP_K, stages)

    // s2r tile
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    // auto tAsA_mma = thr_mma.partition_A(sA); // (MMA, MMA_M, MMA_K, stages)
    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); //(MMA, MMA_M, MMA_K)
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); //(MMA, MMA_N, MMA_K)
    auto tCrC = thr_mma.partition_fragment_C(gC); //(MMA, MMA_M, MMA_N)

    clear(tCrC);
    
    auto tCgC = thr_mma.partition_C(gC);

    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtom{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(threadIdx.x);
    auto tAsA = s2r_thr_copy_a.partition_S(sA); //(CP, CP_M, CP_K, stages)
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); //(CP, CP_M, CP_K) copy 和 mma 必须兼容

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtom{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(threadIdx.x);
    auto tBsB = s2r_thr_copy_b.partition_S(sB); //(CP, CP_M, CP_K, stages)
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); //(CP, CP_N, CP_K)

    // submit the first pipeline stages
    int ntile = K / TileK;
    int itile_to_read = 0;
#pragma unroll
    for (int istage = 0; istage < Stages - 1; ++istage) {
        if (itile_to_read < ntile) {
            copy(g2s_tiled_copy_a,
                 tAgA_copy(_, _, _, itile_to_read),
                 tAsA_copy(_, _, _, istage));
            copy(g2s_tiled_copy_b,
                 tBgB_copy(_, _, _, itile_to_read),
                 tBsB_copy(_, _, _, istage));
            cp_async_fence();
            ++itile_to_read;
        }
    }

    // K loop
#pragma unroll 
    for(int itile = 0; itile < ntile; itile++) {
        int istage = itile % Stages;

        int groups_to_keep = itile_to_read - itile - 1;
        if (groups_to_keep < 0) {
            groups_to_keep = 0;
        }
        if (groups_to_keep > Stages - 2) {
            groups_to_keep = Stages - 2;
        }
        cp_async_wait_for_current<Stages>(groups_to_keep);
        __syncthreads();

        // next stage g2s
        if(itile_to_read < ntile) {
            int iwrite_stage = itile_to_read % Stages;
            copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read), tAsA_copy(_, _, _, iwrite_stage));
            copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read), tBsB_copy(_, _, _, iwrite_stage));
            cp_async_fence();
            itile_to_read++;
        }

        // current stage s2r
        copy(s2r_tiled_copy_a, tAsA(_, _, _, istage), tCrA_view);
        copy(s2r_tiled_copy_b, tBsB(_, _, _, istage), tCrB_view);

        // mma
        gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

        // sync
        if(itile + 1 < ntile) {
            __syncthreads();
        }
    }
    __syncthreads();

    copy(tCrC, tCgC);
}

template<int kTileM, int kTileN, int kTileK, int kStages>
static void launcher(
    const half_t* A,
    const half_t* B,
    half_t* C,
    int M,
    int N,
    int K
) {
    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int kMmaAtomRepeatM = 2;
    static constexpr int kMmaAtomRepeatN = 2;
    static constexpr int kMmaAtomRepeatK = 1;

    static constexpr int kPermutationM = 32;
    static constexpr int kPermutationN = 32;
    static constexpr int kPermutationK = 16;

    using TiledMMA = decltype(make_tiled_mma(
        mma_atom{},
        make_layout(
            make_shape(Int<kMmaAtomRepeatM>{}, Int<kMmaAtomRepeatN>{}, Int<kMmaAtomRepeatK>{})
        ),
        make_tile(Int<kPermutationM>{}, Int<kPermutationN>{}, Int<kPermutationK>{})
    ));

    auto sA_layout = make_layout(
        make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStages>{}),
        make_stride(Int<kTileK>{}, Int<1>{}, Int<kTileM * kTileK>{})
    );
    auto sB_layout = make_layout(
        make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStages>{}),
        make_stride(Int<kTileK>{}, Int<1>{}, Int<kTileN * kTileK>{})
    );

    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, half_t>;

    using G2SCopyA = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));
    using G2SCopyB = G2SCopyA;

    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, half_t>;

    dim3 block(size(TiledMMA{}));
    dim3 grid((N + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM);

    constexpr int smem_size =
        sizeof(half_t) *
        (cosize(decltype(sA_layout){}) + cosize(decltype(sB_layout){}));
    auto kernel = hgemm_pipeline_kernel<
        kTileM, kTileN, kTileK, kStages,
        decltype(sA_layout), decltype(sB_layout),
        TiledMMA, G2SCopyA, G2SCopyB, s2r_copy_atom>;
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<grid, block, smem_size>>>(
        A, B, C, M, N, K, sA_layout, sB_layout
    );
}

template <int kStages>
static void hgemm_pipeline(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor.");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor.");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor.");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous.");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous.");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous.");
    TORCH_CHECK(a.dim() == 2, "a must be a 2D tensor.");
    TORCH_CHECK(b.dim() == 2, "b must be a 2D tensor.");
    TORCH_CHECK(out.dim() == 2, "out must be a 2D tensor.");
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf);
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf);
    CHECK_TORCH_TENSOR_DTYPE(out, torch::kHalf);


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

    launcher<128, 128, 32, kStages>(
        reinterpret_cast<const half_t*>(a.data_ptr()),
        reinterpret_cast<const half_t*>(b.data_ptr()),
        reinterpret_cast<half_t*>(out.data_ptr()),
        static_cast<int>(m64), static_cast<int>(n64),
        static_cast<int>(k64)
    );
} 

void hgemm_2_pipeline(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    hgemm_pipeline<2>(a, b, out);
}

void hgemm_3_pipeline(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    hgemm_pipeline<3>(a, b, out);
}

void hgemm_4_pipeline(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    hgemm_pipeline<4>(a, b, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(hgemm_2_pipeline)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_3_pipeline)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_4_pipeline)
}
