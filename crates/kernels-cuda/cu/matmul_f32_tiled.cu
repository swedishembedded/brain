// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// out = x @ W^T, fp32 - the hand-written CUDA form of the portable
// `matmul.wgsl` reference, with the identical buffer layout, the identical
// uniform (`m, k, n`) and the identical REDUCTION ORDER.
//
//   x   : [M, K]  row-major
//   W   : [N, K]  row-major   (W[n, k] is weight row n = output feature n)
//   out : [M, N]  row-major   out[m, n] = sum_k x[m, k] * W[n, k]
//
// Why this kernel is faster, stated as the thing it fixes
// ------------------------------------------------------
// The reference gives one thread one output element and walks K in global
// memory. Neighbouring threads differ in `n`, so their `W[n, k]` addresses
// are K floats apart: every lane of a warp touches a different cache line
// for every one of the K iterations, and each element of W is re-fetched
// once per row of x. This kernel stages both operands through
// `__shared__` in 64x16 tiles that the whole block cooperates to load
// coalesced, and gives each thread a 4x4 register block of outputs - so
// each staged value is reused four times from registers and the global
// traffic drops by the tile width rather than relying on the L2 to hide it.
//
// Why it still agrees with the reference to the last bit
// -----------------------------------------------------
// The accumulation walks k ASCENDING in a single fp32 register per output,
// exactly as the reference's `for (i = 0..K) acc = acc + x[..]*w[..]` does -
// the tiling changes where the operands are READ from, never the order they
// are summed in. Combined with the backend's `--fmad=false` (which the
// generated tier is also compiled with, so neither side contracts a
// multiply-add), the only remaining difference would be reassociation, and
// there is none. Out-of-range lanes of the K tile are zero-filled, and
// `acc + 0.0f * 0.0f` is exact, so the tail of a K that is not a multiple
// of the tile is not a source of drift either.
//
// No `__restrict__` anywhere, deliberately: brain's device buffers alias by
// design (a sliced step binds ranges of one allocation), and the registry's
// own invariant check refuses a no-alias promise that has not been proven at
// the call sites.
//
// Nothing here names a card. The tile geometry is a property of this source;
// whether a given device can host it (threads per block, shared bytes per
// block) is asked of the driver at run time by the provider that dispatches
// it, and the compute capability it is compiled for is read back from the
// device that will run it.

#define BRAIN_MM_BM 64  // output rows per block
#define BRAIN_MM_BN 64  // output columns per block
#define BRAIN_MM_BK 16  // reduction depth staged per iteration
#define BRAIN_MM_TM 4   // output rows per thread
#define BRAIN_MM_TN 4   // output columns per thread
// 16x16 = 256 threads, each owning a 4x4 register block -> 64x64 per block.
#define BRAIN_MM_THREADS ((BRAIN_MM_BM / BRAIN_MM_TM) * (BRAIN_MM_BN / BRAIN_MM_TN))
// One float of padding per staged row: the stores below write a column of
// the staged tile (consecutive threads share a bank at stride 64), and an
// odd stride spreads them across all 32 banks instead of serialising 16-way.
#define BRAIN_MM_PAD (BRAIN_MM_BM + 1)
// Rows of the staged tile one full pass of the block covers, so the staging
// loop below has a compile-time trip count.
#define BRAIN_MM_ROWS_PER_PASS (BRAIN_MM_THREADS / BRAIN_MM_BK)

extern "C" __global__ void brain_matmul_f32_tiled(const unsigned int* params,
                                                  const float* x,
                                                  const float* w,
                                                  float* out) {
    const unsigned int M = params[0];
    const unsigned int K = params[1];
    const unsigned int N = params[2];

    // Blocks are launched as a flat 1-D count that the host may wrap into a
    // second grid dimension past the driver's per-dimension limit - the same
    // `gid.y * (nwg.x * WG) + gid.x` reconstruction every WGSL kernel in this
    // engine does, one level up (blocks, not invocations).
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int tiles_n = (N + BRAIN_MM_BN - 1u) / BRAIN_MM_BN;
    if (tiles_n == 0u) { return; }
    const unsigned int tile_m = blk / tiles_n;
    const unsigned int tile_n = blk - tile_m * tiles_n;
    const unsigned int row0 = tile_m * BRAIN_MM_BM;
    const unsigned int col0 = tile_n * BRAIN_MM_BN;
    if (row0 >= M) { return; }  // block-uniform: no barrier is skipped per-lane

    __shared__ float xs[BRAIN_MM_BK][BRAIN_MM_PAD];
    __shared__ float ws[BRAIN_MM_BK][BRAIN_MM_PAD];

    const unsigned int tid = threadIdx.x;
    const unsigned int ty = tid / (BRAIN_MM_BN / BRAIN_MM_TN);  // 0..15
    const unsigned int tx = tid % (BRAIN_MM_BN / BRAIN_MM_TN);  // 0..15

    // Staging index: 16 consecutive threads read 16 consecutive k of one
    // row, so each quarter-warp issues one contiguous 64-byte request.
    const unsigned int lk = tid % BRAIN_MM_BK;          // 0..15, the k lane
    const unsigned int lr = tid / BRAIN_MM_BK;          // 0..15, the first row

    float acc[BRAIN_MM_TM][BRAIN_MM_TN];
#pragma unroll
    for (int i = 0; i < BRAIN_MM_TM; ++i) {
#pragma unroll
        for (int j = 0; j < BRAIN_MM_TN; ++j) { acc[i][j] = 0.0f; }
    }

    for (unsigned int k0 = 0; k0 < K; k0 += BRAIN_MM_BK) {
        const unsigned int kk = k0 + lk;
        const bool k_ok = kk < K;
#pragma unroll
        for (unsigned int p = 0; p < BRAIN_MM_BM / BRAIN_MM_ROWS_PER_PASS; ++p) {
            const unsigned int r = lr + p * BRAIN_MM_ROWS_PER_PASS;
            const unsigned int gr = row0 + r;
            xs[lk][r] = (k_ok && gr < M) ? x[(unsigned long long)gr * K + kk] : 0.0f;
            const unsigned int gc = col0 + r;
            ws[lk][r] = (k_ok && gc < N) ? w[(unsigned long long)gc * K + kk] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (unsigned int t = 0; t < BRAIN_MM_BK; ++t) {
            float a[BRAIN_MM_TM];
            float b[BRAIN_MM_TN];
#pragma unroll
            for (int i = 0; i < BRAIN_MM_TM; ++i) { a[i] = xs[t][ty * BRAIN_MM_TM + i]; }
#pragma unroll
            for (int j = 0; j < BRAIN_MM_TN; ++j) { b[j] = ws[t][tx * BRAIN_MM_TN + j]; }
#pragma unroll
            for (int i = 0; i < BRAIN_MM_TM; ++i) {
#pragma unroll
                for (int j = 0; j < BRAIN_MM_TN; ++j) { acc[i][j] = acc[i][j] + a[i] * b[j]; }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < BRAIN_MM_TM; ++i) {
        const unsigned int r = row0 + ty * BRAIN_MM_TM + i;
        if (r >= M) { continue; }
#pragma unroll
        for (int j = 0; j < BRAIN_MM_TN; ++j) {
            const unsigned int c = col0 + tx * BRAIN_MM_TN + j;
            if (c < N) { out[(unsigned long long)r * N + c] = acc[i][j]; }
        }
    }
}
