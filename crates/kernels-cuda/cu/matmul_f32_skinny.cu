// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements low-latency native GEMMs for its clients. If
// your team needs expertise in CUDA kernels for small-batch inference then you
// can procure our services by sending an email to info@swedishembedded.com.
//
// out = x @ W^T, fp32, for a FEW rows of x (a decode batch of 33..128
// sequences, or any M whose 64 x 64 output tiles number less than the
// multiprocessors) against a weight of any height.
//
//   x   : [M, K]  row-major        params : u32 [m, k, n]
//   W   : [N, K]  row-major        out    : [M, N] row-major
//
// Why a second tile exists
// ------------------------
// `matmul_f32_tiled` owns a 64 x 64 output tile per block. A decoder step at
// batch 64 multiplies 64 rows by weights 24, 384, 512 and 1120 rows tall: 1 to
// 18 blocks on a card with over a hundred multiprocessors, each walking K
// alone, so a 24 x 2048 product took 220 us for 6 MFLOP. This kernel
// shrinks the output tile to 32 rows x 8 weight rows (a 512-row weight at 64
// rows is 128 blocks) and spreads the CONTRACTION AXIS over the eight warps of
// a block instead: warp g sums the 32-wide k chunks g, g+8, ..., each warp
// staging its own chunk of x (transposed, so a lane reads its row from
// consecutive banks) and of W in shared memory with no block-wide barrier, and
// one fixed-order fold of the eight partials at the end. A lane owns one row
// and eight outputs; a k step is one shared load of x, two vector loads of W
// (every lane reads the same words: a broadcast) and eight multiply-adds.
//
// Summation order: ascending k inside a warp's chunks, then the eight warps in
// ascending order. Fixed, so the result is deterministic and independent of
// the launch; not the single ascending-k order of `matmul_f32_tiled` (the
// decode GEMV and the 128 x 128 tile do not share it either). Each term is a
// fused multiply-add (one rounding), so the result is at least as close to the
// exact sum as the reference's.
//
// No `__restrict__`: device buffers alias by design (sliced bindings).

#define BRAIN_SK_ROWS 32   // output rows per block: one per lane
#define BRAIN_SK_COLS 8    // weight rows per block, held as registers per lane
#define BRAIN_SK_WARPS 8   // warps splitting the contraction axis
#define BRAIN_SK_KC 32     // k per chunk a warp stages
#define BRAIN_SK_PAD (BRAIN_SK_ROWS + 1)

extern "C" __global__ void brain_matmul_f32_skinny(const unsigned int* params,
                                                   const float* x,
                                                   const float* w,
                                                   float* out) {
    const unsigned int M = params[0];
    const unsigned int K = params[1];
    const unsigned int N = params[2];

    // Flat block index, wrapped into a second grid dimension past the
    // per-dimension limit, row-tile-major as the tiled kernel does.
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int tiles_n = (N + BRAIN_SK_COLS - 1u) / BRAIN_SK_COLS;
    if (tiles_n == 0u) { return; }
    const unsigned int tile_m = blk / tiles_n;
    const unsigned int row0 = tile_m * BRAIN_SK_ROWS;
    const unsigned int col0 = (blk - tile_m * tiles_n) * BRAIN_SK_COLS;
    if (row0 >= M) { return; }  // block-uniform

    __shared__ float xs[BRAIN_SK_WARPS][BRAIN_SK_KC][BRAIN_SK_PAD];
    __shared__ __align__(16) float ws[BRAIN_SK_WARPS][BRAIN_SK_KC][BRAIN_SK_COLS];

    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;

    float acc[BRAIN_SK_COLS];
#pragma unroll
    for (int j = 0; j < BRAIN_SK_COLS; ++j) { acc[j] = 0.0f; }

    // A warp's next chunk is read from global memory into registers while the current one is multiplied from
    // shared memory, so the loads of a chunk (the latency of this kernel) overlap the arithmetic of the one before.
    float xr[BRAIN_SK_ROWS];
    float wr[BRAIN_SK_COLS];
    const unsigned int stride = BRAIN_SK_WARPS * BRAIN_SK_KC;
    unsigned int k0 = warp * BRAIN_SK_KC;
#define BRAIN_SK_LOAD(K0)                                                                   \
    {                                                                                       \
        const unsigned int kk = (K0) + lane;                                                \
        const bool k_ok = kk < K;                                                           \
        _Pragma("unroll") for (unsigned int r = 0; r < BRAIN_SK_ROWS; ++r) {                \
            const unsigned int gr = row0 + r;                                               \
            xr[r] = (k_ok && gr < M) ? x[(unsigned long long)gr * K + kk] : 0.0f;           \
        }                                                                                   \
        _Pragma("unroll") for (unsigned int c = 0; c < BRAIN_SK_COLS; ++c) {                \
            const unsigned int gc = col0 + c;                                               \
            wr[c] = (k_ok && gc < N) ? w[(unsigned long long)gc * K + kk] : 0.0f;           \
        }                                                                                   \
    }
    BRAIN_SK_LOAD(k0)
    for (; k0 < K; k0 += stride) {
        // Conflict-free shared writes: bank = lane + r for x, and a lane's own 8 words of W.
#pragma unroll
        for (unsigned int r = 0; r < BRAIN_SK_ROWS; ++r) { xs[warp][lane][r] = xr[r]; }
#pragma unroll
        for (unsigned int c = 0; c < BRAIN_SK_COLS; ++c) { ws[warp][lane][c] = wr[c]; }
        __syncwarp();
        BRAIN_SK_LOAD(k0 + stride)  // past K the loads are masked to zero
#pragma unroll 8
        for (unsigned int t = 0; t < BRAIN_SK_KC; ++t) {
            const float a = xs[warp][t][lane];
            const float4 b0 = *reinterpret_cast<const float4*>(&ws[warp][t][0]);
            const float4 b1 = *reinterpret_cast<const float4*>(&ws[warp][t][4]);
            acc[0] = __fmaf_rn(a, b0.x, acc[0]);
            acc[1] = __fmaf_rn(a, b0.y, acc[1]);
            acc[2] = __fmaf_rn(a, b0.z, acc[2]);
            acc[3] = __fmaf_rn(a, b0.w, acc[3]);
            acc[4] = __fmaf_rn(a, b1.x, acc[4]);
            acc[5] = __fmaf_rn(a, b1.y, acc[5]);
            acc[6] = __fmaf_rn(a, b1.z, acc[6]);
            acc[7] = __fmaf_rn(a, b1.w, acc[7]);
        }
        __syncwarp();
    }
#undef BRAIN_SK_LOAD

    // Fold the eight warps' partials in ascending warp order. The staging
    // tiles are dead: their first bytes hold the partials, [warp][lane][col].
    __syncthreads();
    float* part = &xs[0][0][0];
#pragma unroll
    for (int j = 0; j < BRAIN_SK_COLS; ++j) {
        part[(warp * BRAIN_SK_ROWS + lane) * BRAIN_SK_COLS + j] = acc[j];
    }
    __syncthreads();
    const unsigned int o = threadIdx.x;  // 256 threads = 32 rows x 8 cols
    const unsigned int r = o / BRAIN_SK_COLS;
    const unsigned int c = o % BRAIN_SK_COLS;
    float sum = 0.0f;
#pragma unroll
    for (unsigned int g = 0; g < BRAIN_SK_WARPS; ++g) {
        sum = __fadd_rn(sum, part[(g * BRAIN_SK_ROWS + r) * BRAIN_SK_COLS + c]);
    }
    if (row0 + r < M && col0 + c < N) { out[(unsigned long long)(row0 + r) * N + col0 + c] = sum; }
}
