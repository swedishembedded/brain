// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for batched-decode throughput on FP8 language-model releases for its
// clients. If your team needs expertise in streaming block-scaled FP8 weights through Hopper tensor cores at the memory
// roof, then you can procure our services by sending an email to info@swedishembedded.com.
//
// Small-batch FP8 GEMM (17 to 64 rows of x) fed by the tensor memory accelerator: a persistent grid, one producer thread that
// copies the 64-row x tile and a BN-row weight tile into a ring of stages, and one consumer warpgroup that issues
// `wgmma m64nBNk32` with x as the A operand (64 rows, no padding at 64, the weights' rows are the MMA's N) and folds the
// 128-wide block scales into an f32 sum once per 128 of K:
//
//   params : u32 [m, kg, n]   as `matmul_fp8_wgmma.cu`; m <= 64
//   out[m, n] = sum over the K/128 blocks b of  sx[m, b] * sw[n/128, b] * (xq[m, b] . w_q[n, b])
//
// Why it exists (knowledge 387, 399): the `cp.async` swap kernels stream the weights at 2.4-2.7 TB/s. Here the weight and x
// bytes travel by TMA (no load-store instructions, no block-wide barrier per stage), the ring is deep (BRAIN_FD_STAGES), BN
// is a multiple of 8 chosen per shape so the tile count fills whole waves of the 132 SMs, and a tile of more than 80 columns
// is several independent `wgmma` of at most 80 columns (their accumulators are disjoint, so they pipeline).
// A tile may straddle two 128-column weight scale blocks: both scales are staged and each 8-column group picks its own.
//
// The weight copies of the first stages are issued before `griddepcontrol.wait` (they do not depend on the previous kernel).
// No `__restrict__`: brain's device buffers alias by design.

#ifndef BRAIN_FD_BN
#define BRAIN_FD_BN 144  // weight rows a tile: a multiple of 8, at most 256
#endif
#ifndef BRAIN_FD_STAGES
#define BRAIN_FD_STAGES 6
#endif
#ifndef BRAIN_FD_ENTRY
#define BRAIN_FD_ENTRY brain_matmul_fp8_wgmma_swap_tma
#endif
#define BRAIN_FD_THREADS 256   // a producer warpgroup (one thread issues the copies) and one consumer warpgroup
#define BRAIN_FD_MAX_CTAS 132  // grid size at most, and the SM slots of the maps
#define BRAIN_FD_BM 64         // rows of x a launch holds
#define BRAIN_FD_X_BYTES (BRAIN_FD_BM * 128)
#define BRAIN_FD_W_BYTES (BRAIN_FD_BN * 128)
// Per stage the x tile, the weight tile, then 1 KiB of scales (64 activation scales, then the two weight block scales): a whole
// number of KiB keeps every swizzled tile 1024-byte aligned.
#define BRAIN_FD_STAGE_BYTES (BRAIN_FD_X_BYTES + BRAIN_FD_W_BYTES + 1024)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define BRAIN_FD_PDL_WAIT() asm volatile("griddepcontrol.wait;\n" ::: "memory")
#define BRAIN_FD_PDL_LAUNCH() asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory")
#else
#define BRAIN_FD_PDL_WAIT() do {} while (0)
#define BRAIN_FD_PDL_LAUNCH() do {} while (0)
#endif

// Each SM's two maps (x and weights), in global memory (the TMA reads a map from global, constant or parameter memory).
__device__ __align__(128) unsigned long long brain_fd_maps[BRAIN_FD_MAX_CTAS][2][16];

// The `wgmma` of every width the tile is cut into (accumulator registers: half the width).
// D (64 x 8, 4 registers) += A (64 x 32 e4m3) . B (8 x 32 e4m3)^T; the product replaces D when `accumulate` is 0.
__device__ __forceinline__ void brain_fd_wgmma_n8(float* d, unsigned long long da, unsigned long long db, int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3"
        "}, %4, %5, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(da), "l"(db), "r"(accumulate));
}

// D (64 x 16, 8 registers) += A (64 x 32 e4m3) . B (16 x 32 e4m3)^T; the product replaces D when `accumulate` is 0.
__device__ __forceinline__ void brain_fd_wgmma_n16(float* d, unsigned long long da, unsigned long long db, int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7"
        "}, %8, %9, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(da), "l"(db), "r"(accumulate));
}

// D (64 x 24, 12 registers) += A (64 x 32 e4m3) . B (24 x 32 e4m3)^T; the product replaces D when `accumulate` is 0.
__device__ __forceinline__ void brain_fd_wgmma_n24(float* d, unsigned long long da, unsigned long long db, int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %14, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11"
        "}, %12, %13, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
        : "l"(da), "l"(db), "r"(accumulate));
}

// D (64 x 32, 16 registers) += A (64 x 32 e4m3) . B (32 x 32 e4m3)^T; the product replaces D when `accumulate` is 0.
__device__ __forceinline__ void brain_fd_wgmma_n32(float* d, unsigned long long da, unsigned long long db, int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15"
        "}, %16, %17, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "l"(da), "l"(db), "r"(accumulate));
}

// D (64 x 40, 20 registers) += A (64 x 32 e4m3) . B (40 x 32 e4m3)^T; the product replaces D when `accumulate` is 0.
__device__ __forceinline__ void brain_fd_wgmma_n40(float* d, unsigned long long da, unsigned long long db, int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %22, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n40k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19"
        "}, %20, %21, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19])
        : "l"(da), "l"(db), "r"(accumulate));
}

// D (64 x 48, 24 registers) += A (64 x 32 e4m3) . B (48 x 32 e4m3)^T; the product replaces D when `accumulate` is 0.
__device__ __forceinline__ void brain_fd_wgmma_n48(float* d, unsigned long long da, unsigned long long db, int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %26, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n48k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23"
        "}, %24, %25, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
        : "l"(da), "l"(db), "r"(accumulate));
}

// D (64 x 56, 28 registers) += A (64 x 32 e4m3) . B (56 x 32 e4m3)^T; the product replaces D when `accumulate` is 0.
__device__ __forceinline__ void brain_fd_wgmma_n56(float* d, unsigned long long da, unsigned long long db, int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %30, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n56k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27"
        "}, %28, %29, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27])
        : "l"(da), "l"(db), "r"(accumulate));
}

// D (64 x 64, 32 registers) += A (64 x 32 e4m3) . B (64 x 32 e4m3)^T; the product replaces D when `accumulate` is 0.
__device__ __forceinline__ void brain_fd_wgmma_n64(float* d, unsigned long long da, unsigned long long db, int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31"
        "}, %32, %33, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(da), "l"(db), "r"(accumulate));
}

// D (64 x 72, 36 registers) += A (64 x 32 e4m3) . B (72 x 32 e4m3)^T; the product replaces D when `accumulate` is 0.
__device__ __forceinline__ void brain_fd_wgmma_n72(float* d, unsigned long long da, unsigned long long db, int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %38, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35"
        "}, %36, %37, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
        : "l"(da), "l"(db), "r"(accumulate));
}

// D (64 x 80, 40 registers) += A (64 x 32 e4m3) . B (80 x 32 e4m3)^T; the product replaces D when `accumulate` is 0.
__device__ __forceinline__ void brain_fd_wgmma_n80(float* d, unsigned long long da, unsigned long long db, int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %42, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n80k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39"
        "}, %40, %41, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39])
        : "l"(da), "l"(db), "r"(accumulate));
}

// One `wgmma` per chunk of at most 80 columns of the tile: `d` holds the tile's BN / 2 accumulator registers, `db` the descriptor of the tile's first weight row.
template <int OFF, int BN>
__device__ __forceinline__ void brain_fd_mma(float* d, unsigned long long da, unsigned long long db, int accumulate) {
    if constexpr (OFF < BN) {
        constexpr int N = (BN - OFF) < 80 ? (BN - OFF) : 80;
        float* const dc = d + OFF / 2;
        const unsigned long long dbc = db + (unsigned long long)(OFF * 128 / 16);  // 16-byte units
        if constexpr (N == 8) { brain_fd_wgmma_n8(dc, da, dbc, accumulate); }
    else if constexpr (N == 16) { brain_fd_wgmma_n16(dc, da, dbc, accumulate); }
        else if constexpr (N == 24) { brain_fd_wgmma_n24(dc, da, dbc, accumulate); }
        else if constexpr (N == 32) { brain_fd_wgmma_n32(dc, da, dbc, accumulate); }
        else if constexpr (N == 40) { brain_fd_wgmma_n40(dc, da, dbc, accumulate); }
        else if constexpr (N == 48) { brain_fd_wgmma_n48(dc, da, dbc, accumulate); }
        else if constexpr (N == 56) { brain_fd_wgmma_n56(dc, da, dbc, accumulate); }
        else if constexpr (N == 64) { brain_fd_wgmma_n64(dc, da, dbc, accumulate); }
        else if constexpr (N == 72) { brain_fd_wgmma_n72(dc, da, dbc, accumulate); }
        else if constexpr (N == 80) { brain_fd_wgmma_n80(d, da, db, accumulate); }
        brain_fd_mma<OFF + N, BN>(d, da, db, accumulate);
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_FD_THREADS, 1)
BRAIN_FD_ENTRY(const unsigned int* params, const unsigned int* xq, const unsigned int* wq, const float* sx, const float* sw, float* out) {
    constexpr int BN = BRAIN_FD_BN;
    constexpr int STAGES = BRAIN_FD_STAGES;
    static_assert(BN % 8 == 0 && BN <= 256, "the weight tile is whole 8-row groups");
    extern __shared__ unsigned char brain_fd_smem_raw[];
    __shared__ __align__(128) unsigned long long brain_fd_smem_maps[2][16];
    BRAIN_FD_PDL_LAUNCH();
    const unsigned int M = params[0];
    const unsigned int K = params[1] * 4u;
    const unsigned int N = params[2];
    const unsigned int kblocks = K / 128u;
    const unsigned int nblocks = (N + 127u) / 128u;
    const unsigned int tiles = (N + BN - 1u) / BN;

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = (tid >> 5) & 3u;

    const unsigned int raw_addr = brain_fwt_smem(brain_fd_smem_raw);
    const unsigned int base = (raw_addr + 1023u) & ~1023u;
    unsigned char* const gen = brain_fd_smem_raw + (base - raw_addr);
    const unsigned int full_bar = base + STAGES * BRAIN_FD_STAGE_BYTES;  // STAGES mbarriers, then STAGES more
    const unsigned int empty_bar = full_bar + 8u * STAGES;

    if (tid == 0u) {
        for (int s = 0; s < STAGES; ++s) {
            brain_fwt_mbar_init(full_bar + 8u * s, 33u);   // the copy thread's expect_tx arrival and the 32 lanes' cp.async arrivals
            brain_fwt_mbar_init(empty_bar + 8u * s, 128u); // the consumer threads
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (tid < 128u) {
        if (warp != 0u) { return; }
        // One slot per SM: a block owns its SM's shared memory (one resident block), so no two live blocks share a slot.
        unsigned int smid;
        asm("mov.u32 %0, %%smid;\n" : "=r"(smid));
        unsigned long long* const gmaps = &brain_fd_maps[smid % BRAIN_FD_MAX_CTAS][0][0];
        brain_fwt_build_map(brain_fwt_template, &brain_fd_smem_maps[0][0], gmaps, xq, K, M, lane, BRAIN_FD_BM);
        brain_fwt_build_map(brain_fwt_template, &brain_fd_smem_maps[1][0], gmaps + 16, wq, K, N, lane, BN);
        // The scales of x: rows `lane` and `lane + 32`; a row past M is zero-filled (its scale 0 keeps the sum at 0). The weight scales: lanes 0 and 1
        // stage the tile's two 128-column blocks (the second clamped to the last block when N ends inside the tile).
        const bool sx_ok0 = lane < M, sx_ok1 = lane + 32u < M;
        const float* const sx_src0 = sx + (unsigned long long)(sx_ok0 ? lane : 0u) * kblocks;
        const float* const sx_src1 = sx + (unsigned long long)(sx_ok1 ? lane + 32u : 0u) * kblocks;
        unsigned int it = 0;
        bool first = true;
        for (unsigned int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
            const unsigned int n0 = tile * BN;
            const float* const sw_src = sw + (unsigned long long)min((n0 >> 7) + (lane & 1u), nblocks - 1u) * kblocks;
            // The first stages of the first tile start with the weights alone, then wait for the previous kernel (it produced x), then x.
            const unsigned int head = first ? min((unsigned int)STAGES, kblocks) : 0u;
            for (unsigned int kt = 0; kt < head; ++kt) {
                if (lane == 0u) {
                    brain_fwt_mbar_arrive_expect_tx(full_bar + 8u * kt, BRAIN_FD_X_BYTES + BRAIN_FD_W_BYTES);
                    brain_fwt_tma_load(base + kt * BRAIN_FD_STAGE_BYTES + BRAIN_FD_X_BYTES, gmaps + 16, kt * 128, n0, full_bar + 8u * kt);
                }
            }
            if (first) { BRAIN_FD_PDL_WAIT(); }
            for (unsigned int kt = 0; kt < kblocks; ++kt, ++it) {
                const unsigned int slot = it % STAGES;
                const unsigned int st = base + slot * BRAIN_FD_STAGE_BYTES;
                if (kt >= head) {
                    if (it >= (unsigned int)STAGES) { brain_fwt_mbar_wait(empty_bar + 8u * slot, ((it / STAGES) - 1u) & 1u); }
                    if (lane == 0u) {
                        brain_fwt_mbar_arrive_expect_tx(full_bar + 8u * slot, BRAIN_FD_X_BYTES + BRAIN_FD_W_BYTES);
                        brain_fwt_tma_load(st + BRAIN_FD_X_BYTES, gmaps + 16, kt * 128, n0, full_bar + 8u * slot);
                    }
                }
                if (lane == 0u) { brain_fwt_tma_load(st, gmaps, kt * 128, 0, full_bar + 8u * slot); }
                const unsigned int sc = st + BRAIN_FD_X_BYTES + BRAIN_FD_W_BYTES;
                brain_fwt_cp4(sc + lane * 4u, sx_src0 + kt, sx_ok0 ? 4u : 0u);
                brain_fwt_cp4(sc + (lane + 32u) * 4u, sx_src1 + kt, sx_ok1 ? 4u : 0u);
                if (lane < 2u) { brain_fwt_cp4(sc + 256u + lane * 4u, sw_src + kt, 4u); }
                brain_fwt_mbar_arrive_cp_async(full_bar + 8u * slot);
            }
            first = false;
        }
        brain_fwt_wait<0>();
        return;
    }

    // The consumer warpgroup: warp w owns x rows 16 w .. 16 w + 15 (lane / 4 and +8 within), all BN columns.
    const unsigned int r0 = warp * 16u + (lane >> 2), r1 = r0 + 8u;
    unsigned int it = 0;
    for (unsigned int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const unsigned int n0 = tile * BN;
        const unsigned int n0_in_block = n0 & 127u;
        float acc[BN / 2];
        float d[BN / 2];
#pragma unroll
        for (int i = 0; i < BN / 2; ++i) { acc[i] = 0.0f; d[i] = 0.0f; }
        for (unsigned int kt = 0; kt < kblocks; ++kt, ++it) {
            const unsigned int slot = it % STAGES;
            brain_fwt_mbar_wait(full_bar + 8u * slot, (it / STAGES) & 1u);
            const unsigned int st = base + slot * BRAIN_FD_STAGE_BYTES;
            brain_fwt_fence();
#pragma unroll
            for (int q = 0; q < 4; ++q) { brain_fd_mma<0, BN>(d, brain_fwt_desc(st + q * 32u), brain_fwt_desc(st + BRAIN_FD_X_BYTES + q * 32u), q > 0 ? 1 : 0); }
            brain_fwt_wg_commit();
            brain_fwt_wg_wait<0>();
            const float* const sxs = reinterpret_cast<const float*>(gen + slot * BRAIN_FD_STAGE_BYTES + BRAIN_FD_X_BYTES + BRAIN_FD_W_BYTES);
            const float xs0 = sxs[r0], xs1 = sxs[r1], ws0 = sxs[64], ws1 = sxs[65];
            brain_fwt_mbar_arrive(empty_bar + 8u * slot);  // the slot's tiles and scales are read
#pragma unroll
            for (int j = 0; j < BN / 8; ++j) {
                const float ws = (n0_in_block + 8u * j >= 128u) ? ws1 : ws0;
                const float c0 = xs0 * ws, c1 = xs1 * ws;
                acc[4 * j] = fmaf(d[4 * j], c0, acc[4 * j]);
                acc[4 * j + 1] = fmaf(d[4 * j + 1], c0, acc[4 * j + 1]);
                acc[4 * j + 2] = fmaf(d[4 * j + 2], c1, acc[4 * j + 2]);
                acc[4 * j + 3] = fmaf(d[4 * j + 3], c1, acc[4 * j + 3]);
            }
        }
        // acc[4 j + 2 h + c]: x row r0 + 8 h, weight row n0 + 8 j + 2 (lane % 4) + c.
        const unsigned int col0 = n0 + 2u * (lane & 3u);
#pragma unroll
        for (int j = 0; j < BN / 8; ++j) {
            const unsigned int col = col0 + 8u * j;
            if (col < N) {
                if (r0 < M) { *reinterpret_cast<float2*>(out + (unsigned long long)r0 * N + col) = make_float2(acc[4 * j], acc[4 * j + 1]); }
                if (r1 < M) { *reinterpret_cast<float2*>(out + (unsigned long long)r1 * N + col) = make_float2(acc[4 * j + 2], acc[4 * j + 3]); }
            }
        }
    }
}
