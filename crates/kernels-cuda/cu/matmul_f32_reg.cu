// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bit-exact native GEMMs for its clients. If
// your team needs expertise in hand-written CUDA kernels that reproduce a
// portable reference to the last bit then you can procure our services by
// sending an email to info@swedishembedded.com.
//
// Register-tiled fp32 matmul, out = x @ W^T - the hand-written CUDA form of
// `matmul_reg3.wgsl`, with the identical buffer layout, the identical uniform
// and the identical RESULT TO THE LAST BIT.
//
//   params : u32 [m, k, n]
//   x      : [M, K] f32
//   w      : [N, K] f32
//   out    : [M, N] f32
//
// What it computes, exactly
// -------------------------
// The WGSL kernel keeps one fp32 accumulator per output, starting at +0, and
// adds `a * b` for k ascending, the product and the sum each rounded
// (`--fmad=false`), over K padded with zero terms to a whole number of 8-wide
// chunks. This kernel does the same with `__fmul_rn`/`__fadd_rn`. It pads K
// to a whole number of 16-wide chunks instead: the extra terms are `+0 * +0`,
// and adding +0 to an accumulator leaves it unchanged unless it is -0, which
// one that starts at +0 and only ever adds rounded values can never be (an
// exact cancellation rounds to +0).
//
// Tiling
// ------
// 128 x 128 outputs per 256-thread block, each thread an 8 x 8 register block
// (rows ty*4 + {0..3} and 64 + ty*4 + {0..3}, columns likewise from tx), so
// a k-step is four 16-byte shared loads for 64 multiply-adds. Shared tiles are
// k-MAJOR, double-buffered, one barrier per 16-wide chunk; the next chunk's
// global loads are issued before the current chunk is consumed. Four
// consecutive threads stage the 64 contiguous bytes of one row's chunk.
// Bindings that are not 16-byte aligned, or a K that is not a multiple of
// four, take the scalar-load instantiation, which computes the same bits.
//
// The register budget is capped so TWO blocks fit per SM (`__launch_bounds__`
// minimum 2, 128 registers, no spills): with one block, every chunk's barrier
// and staging stalls the whole SM; with two, one block's barrier hides behind
// the other's arithmetic. That cap is worth more than any change to the inner
// loop - measured as the difference between a little over half and nine
// tenths of the separately-rounded multiply-add rate.
//
// No `__restrict__` anywhere, deliberately: brain's device buffers alias by
// design (a sliced step binds ranges of one allocation).

#define BRAIN_MMR_BM 128
#define BRAIN_MMR_BN 128
#define BRAIN_MMR_BK 16
#define BRAIN_MMR_THREADS 256
// Row stride padded by four floats: the transposing stores of the four
// threads that share a row spread over the banks, and every 16-byte read of
// the inner loop stays aligned.
#define BRAIN_MMR_SP (BRAIN_MMR_BM + 4)

struct BrainMmrShared {
    float a[2][BRAIN_MMR_BK][BRAIN_MMR_SP];
    float b[2][BRAIN_MMR_BK][BRAIN_MMR_SP];
};

// Four consecutive values of a row starting at `k`, zero past K or for a row
// outside the operand.
template <bool VEC>
__device__ __forceinline__ float4 brain_mmr_load(const float* row, bool ok, unsigned int k, unsigned int kk) {
    if (VEC) {
        if (!ok || k >= kk) { return make_float4(0.f, 0.f, 0.f, 0.f); }
        return __ldg(reinterpret_cast<const float4*>(row + k));
    }
    float4 v;
    v.x = (ok && k + 0 < kk) ? __ldg(row + k + 0) : 0.f;
    v.y = (ok && k + 1 < kk) ? __ldg(row + k + 1) : 0.f;
    v.z = (ok && k + 2 < kk) ? __ldg(row + k + 2) : 0.f;
    v.w = (ok && k + 3 < kk) ? __ldg(row + k + 3) : 0.f;
    return v;
}

__device__ __forceinline__ void brain_mmr_store(float (*dst)[BRAIN_MMR_SP], unsigned int q, unsigned int r, float4 v) {
    dst[q + 0][r] = v.x;
    dst[q + 1][r] = v.y;
    dst[q + 2][r] = v.z;
    dst[q + 3][r] = v.w;
}

template <bool VEC>
__device__ __forceinline__ void brain_mmr_block(BrainMmrShared& sh, const unsigned int* params, const float* x, const float* w, float* out) {
    const unsigned int m = params[0];
    const unsigned int kk = params[1];
    const unsigned int n = params[2];
    const unsigned int tiles_n = (n + BRAIN_MMR_BN - 1) / BRAIN_MMR_BN;
    const unsigned int tiles = ((m + BRAIN_MMR_BM - 1) / BRAIN_MMR_BM) * tiles_n;
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (blk >= tiles) { return; }
    const unsigned int row0 = (blk / tiles_n) * BRAIN_MMR_BM;
    const unsigned int col0 = (blk % tiles_n) * BRAIN_MMR_BN;

    const unsigned int tid = threadIdx.x;
    const unsigned int ty = tid / 16;
    const unsigned int tx = tid % 16;

    // Staging: thread t covers vectors t and t + 256 of a chunk's 512 per
    // operand (128 rows x 4 vectors of 4 k); four consecutive threads share a row.
    const unsigned int sr0 = tid / 4;
    const unsigned int sr1 = sr0 + 64;
    const unsigned int sq = (tid % 4) * 4;
    const bool a0_ok = row0 + sr0 < m;
    const bool a1_ok = row0 + sr1 < m;
    const bool b0_ok = col0 + sr0 < n;
    const bool b1_ok = col0 + sr1 < n;
    const float* a0_row = x + (size_t)(a0_ok ? row0 + sr0 : 0) * kk;
    const float* a1_row = x + (size_t)(a1_ok ? row0 + sr1 : 0) * kk;
    const float* b0_row = w + (size_t)(b0_ok ? col0 + sr0 : 0) * kk;
    const float* b1_row = w + (size_t)(b1_ok ? col0 + sr1 : 0) * kk;

    float c[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) { c[i][j] = 0.0f; }
    }

    const unsigned int nchunks = (kk + BRAIN_MMR_BK - 1) / BRAIN_MMR_BK;

    brain_mmr_store(sh.a[0], sq, sr0, brain_mmr_load<VEC>(a0_row, a0_ok, sq, kk));
    brain_mmr_store(sh.a[0], sq, sr1, brain_mmr_load<VEC>(a1_row, a1_ok, sq, kk));
    brain_mmr_store(sh.b[0], sq, sr0, brain_mmr_load<VEC>(b0_row, b0_ok, sq, kk));
    brain_mmr_store(sh.b[0], sq, sr1, brain_mmr_load<VEC>(b1_row, b1_ok, sq, kk));
    __syncthreads();

    for (unsigned int ch = 0; ch < nchunks; ++ch) {
        const unsigned int buf = ch & 1u;
        const bool has_next = ch + 1 < nchunks;
        float4 na0, na1, nb0, nb1;
        if (has_next) {
            const unsigned int k1 = (ch + 1) * BRAIN_MMR_BK + sq;
            na0 = brain_mmr_load<VEC>(a0_row, a0_ok, k1, kk);
            na1 = brain_mmr_load<VEC>(a1_row, a1_ok, k1, kk);
            nb0 = brain_mmr_load<VEC>(b0_row, b0_ok, k1, kk);
            nb1 = brain_mmr_load<VEC>(b1_row, b1_ok, k1, kk);
        }
#pragma unroll
        for (int k = 0; k < BRAIN_MMR_BK; ++k) {
            const float4 a0 = *reinterpret_cast<const float4*>(&sh.a[buf][k][ty * 4]);
            const float4 a1 = *reinterpret_cast<const float4*>(&sh.a[buf][k][64 + ty * 4]);
            const float4 b0 = *reinterpret_cast<const float4*>(&sh.b[buf][k][tx * 4]);
            const float4 b1 = *reinterpret_cast<const float4*>(&sh.b[buf][k][64 + tx * 4]);
            const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < 8; ++j) { c[i][j] = __fadd_rn(c[i][j], __fmul_rn(av[i], bv[j])); }
            }
        }
        if (has_next) {
            const unsigned int nb = buf ^ 1u;
            brain_mmr_store(sh.a[nb], sq, sr0, na0);
            brain_mmr_store(sh.a[nb], sq, sr1, na1);
            brain_mmr_store(sh.b[nb], sq, sr0, nb0);
            brain_mmr_store(sh.b[nb], sq, sr1, nb1);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned int r = row0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + (i - 4));
        if (r >= m) { continue; }
        float* orow = out + (size_t)r * n;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned int cbase = col0 + h * 64 + tx * 4;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (cbase + j < n) { orow[cbase + j] = c[i][h * 4 + j]; }
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_MMR_THREADS, 2)
brain_matmul_f32_reg(const unsigned int* params, const float* x, const float* w, float* out) {
    // One shared allocation for both instantiations: a `__shared__` inside
    // the templated body would be one per instantiation, twice the footprint.
    __shared__ BrainMmrShared sh;
    const bool vec = (params[1] % 4 == 0) && ((reinterpret_cast<size_t>(x) | reinterpret_cast<size_t>(w)) & 15u) == 0;
    if (vec) {
        brain_mmr_block<true>(sh, params, x, w, out);
    } else {
        brain_mmr_block<false>(sh, params, x, w, out);
    }
}
