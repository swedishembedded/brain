// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bit-exact native GEMMs for its clients. If
// your team needs expertise in hand-written CUDA kernels that reproduce a
// portable reference to the last bit then you can procure our services by
// sending an email to info@swedishembedded.com.
//
// Register-tiled fp32 matmuls, each the hand-written CUDA form of a WGSL kernel
// with the identical buffer layout, the identical uniform and the identical
// RESULT TO THE LAST BIT:
//
//   brain_matmul_f32_reg     `matmul_reg3.wgsl`, out = x @ W^T
//     params : u32 [m, k, n]
//     x : [M, K]   w : [N, K]   out : [M, N]
//
//   brain_matmul_f32_dx_reg  `matmul_dx_reg.wgsl`, its input gradient
//     params : u32 [m, k, n, accumulate]
//     dy : [M, N]  w : [N, K]   out : [M, K] = dy @ W (added to `out` when
//     `accumulate` is non-zero, as one rounded addition after the sum)
//
//   brain_matmul_f32_dw_reg  `matmul_dw_reg.wgsl`, a weight gradient
//     params : u32 [m, k, n]
//     a : [M, N]   b : [M, K]   out : [N, K] += a^T @ b (always accumulated)
//
// the same three with fused multiply-adds - `brain_matmul_f32_fma`,
// `brain_matmul_f32_dx_fma`, `brain_matmul_f32_dw_fma`, identical contracts, one
// rounding per term instead of two, for a trainer that asks for them by name
// (`gpu_core::Fused::{F32Fma, F32DxFma, F32DwFma}`) - and one training kernel
// with no WGSL twin, held to a host oracle instead of to bits:
//
//   brain_matmul_i8w_dx      the input gradient through an int8 weight
//     params : u32 [m, k, n, accumulate]   (K a multiple of 32)
//     dy : [M, N]  wq : [N, K/4] u32  sw : [N, K/32] f32
//     out : [M, K] = dy @ deq(W), deq(W)[n, c] = q[n, c] * sw[n, c/32]
//     The weight is dequantised exactly while it is staged, and the terms
//     accumulate with fused multiply-adds.
//
// All three are one GEMM over a contraction axis of length L (K, N and M
// respectively); they differ only in whether each operand holds a row per
// output or a row per contraction step, and in the epilogue.
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
// Bindings that are not 16-byte aligned, or a row length that is not a
// multiple of four, take the scalar-load instantiation, which computes the
// same bits.
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

// Four consecutive floats of `row` starting at `i`, zero at and past `lim` or
// for a row outside the operand.
template <bool VEC>
__device__ __forceinline__ float4 brain_mmr_load(const float* row, bool ok, unsigned int i, unsigned int lim) {
    if (VEC) {
        if (!ok || i >= lim) { return make_float4(0.f, 0.f, 0.f, 0.f); }
        return __ldg(reinterpret_cast<const float4*>(row + i));
    }
    float4 v;
    v.x = (ok && i + 0 < lim) ? __ldg(row + i + 0) : 0.f;
    v.y = (ok && i + 1 < lim) ? __ldg(row + i + 1) : 0.f;
    v.z = (ok && i + 2 < lim) ? __ldg(row + i + 2) : 0.f;
    v.w = (ok && i + 3 < lim) ? __ldg(row + i + 3) : 0.f;
    return v;
}

// One operand's share of a 16-step chunk, staged into a k-major shared tile.
// ROW_MAJOR: the operand is [outputs, L] (a row per output, contraction
// contiguous) - four consecutive threads read one row's 16 steps and the
// store transposes. Otherwise it is [L, outputs] (contraction rows, outputs
// contiguous) - a thread reads four consecutive outputs of one step and the
// store is a straight copy.
template <bool VEC, bool ROW_MAJOR>
struct BrainMmrStage;

// Row per output: four consecutive threads read one row's 16 steps (64
// contiguous bytes) and the store transposes into the k-major tile. A
// thread's two rows are 64 apart, so one row pointer serves both.
template <bool VEC>
struct BrainMmrStage<VEC, true> {
    const float* row;
    unsigned int r;
    unsigned int q;
    unsigned int len;
    bool ok0;
    bool ok1;
    float4 v[2];

    // `op` is the operand, `o0` its first output in this tile, `outs` the
    // output count, `len` the contraction length L.
    __device__ __forceinline__ void init(const float* op, unsigned int o0, unsigned int outs, unsigned int l, unsigned int tid) {
        q = (tid % 4) * 4;
        r = tid / 4;
        len = l;
        ok0 = o0 + r < outs;
        ok1 = o0 + r + BRAIN_MMR_THREADS / 4 < outs;
        // A row past the operand is never dereferenced (its flag is false).
        row = op + (size_t)(ok0 ? o0 + r : 0) * len;
    }
    __device__ __forceinline__ void load(unsigned int k0, unsigned int, unsigned int, unsigned int, unsigned int) {
        v[0] = brain_mmr_load<VEC>(row, ok0, k0 + q, len);
        v[1] = brain_mmr_load<VEC>(row + (size_t)(BRAIN_MMR_THREADS / 4) * len, ok1, k0 + q, len);
    }
    __device__ __forceinline__ void store(float (*dst)[BRAIN_MMR_SP]) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const unsigned int rr = r + p * (BRAIN_MMR_THREADS / 4);
            dst[q + 0][rr] = v[p].x;
            dst[q + 1][rr] = v[p].y;
            dst[q + 2][rr] = v[p].z;
            dst[q + 3][rr] = v[p].w;
        }
    }
};

// Row per contraction step: a thread reads four consecutive outputs of a step,
// and its two vectors are the same four outputs eight steps apart - one column
// pointer serves both, and the store is a straight copy.
template <bool VEC>
struct BrainMmrStage<VEC, false> {
    const float* col;
    unsigned int r;
    unsigned int q;
    unsigned int live;
    float4 v[2];

    __device__ __forceinline__ void init(const float* op, unsigned int o0, unsigned int outs, unsigned int, unsigned int tid) {
        r = (tid % 32) * 4;
        q = tid / 32;
        col = op + o0 + r;
        live = outs - min(outs, o0 + r);
    }
    __device__ __forceinline__ void load(unsigned int k0, unsigned int, unsigned int len, unsigned int, unsigned int stride) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const unsigned int l = k0 + q + 8 * p;
            v[p] = brain_mmr_load<VEC>(col + (size_t)l * stride, l < len, 0, live);
        }
    }
    __device__ __forceinline__ void store(float (*dst)[BRAIN_MMR_SP]) {
#pragma unroll
        for (int p = 0; p < 2; ++p) { *reinterpret_cast<float4*>(&dst[q + 8 * p][r]) = v[p]; }
    }
};

// The second operand of the int8-weight input gradient: a packed int8 matrix
// `[L, cols/4]` u32 with one fp32 scale per 32 of its columns (`[L, cols/32]`,
// `model::int8`'s group-wise layout), staged as the exact fp32 values
// `q * scale` - the dequantised weight, formed on the way into shared memory
// so the matrix is never resident as fp32. A thread's four outputs are one
// packed word, inside one scale group.
struct BrainMmrStageI8 {
    const unsigned int* q;
    const float* s;
    unsigned int r;
    unsigned int k0;
    float4 v[2];

    __device__ __forceinline__ void init(const unsigned int* wq, const float* sw, unsigned int o0, unsigned int tid) {
        q = wq;
        s = sw;
        r = o0 + (tid % 32) * 4;
        k0 = tid / 32;
    }
    __device__ __forceinline__ void load(unsigned int l0, unsigned int cols, unsigned int len) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const unsigned int l = l0 + k0 + 8 * p;
            if (l < len && r < cols) {
                const unsigned int w = __ldg(q + (size_t)l * (cols / 4) + r / 4);
                const float sc = __ldg(s + (size_t)l * (cols / 32) + r / 32);
                v[p] = make_float4(__fmul_rn((float)(signed char)(w & 0xffu), sc), __fmul_rn((float)(signed char)((w >> 8) & 0xffu), sc),
                                   __fmul_rn((float)(signed char)((w >> 16) & 0xffu), sc), __fmul_rn((float)(signed char)(w >> 24), sc));
            } else {
                v[p] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
    }
    __device__ __forceinline__ void store(float (*dst)[BRAIN_MMR_SP], unsigned int o0) {
#pragma unroll
        for (int p = 0; p < 2; ++p) { *reinterpret_cast<float4*>(&dst[k0 + 8 * p][r - o0]) = v[p]; }
    }
};

// One 128 x 128 output tile of out[r, c] = sum_l A(r, l) * B(c, l), each
// operand either [outputs, L] (`*_ROW_MAJOR`) or [L, outputs]. `rows`/`cols`/
// `len` are the output height and width and L. `FUSED` accumulates with one
// fused multiply-add per term instead of the reference's two roundings - for
// the training-only kernels, which have no WGSL twin to reproduce.
template <bool VEC, bool A_ROW_MAJOR, bool B_ROW_MAJOR, bool FUSED = false, bool B_I8 = false>
__device__ __forceinline__ void brain_mmr_block(BrainMmrShared& sh, unsigned int rows, unsigned int cols, unsigned int len, const float* a,
                                                const float* b, float* out, bool accumulate, const unsigned int* bq = nullptr,
                                                const float* bs = nullptr) {
    const unsigned int tiles_n = (cols + BRAIN_MMR_BN - 1) / BRAIN_MMR_BN;
    const unsigned int tiles = ((rows + BRAIN_MMR_BM - 1) / BRAIN_MMR_BM) * tiles_n;
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (blk >= tiles) { return; }
    const unsigned int row0 = (blk / tiles_n) * BRAIN_MMR_BM;
    const unsigned int col0 = (blk % tiles_n) * BRAIN_MMR_BN;

    const unsigned int tid = threadIdx.x;
    const unsigned int ty = tid / 16;
    const unsigned int tx = tid % 16;

    BrainMmrStage<VEC, A_ROW_MAJOR> sa;
    BrainMmrStage<VEC, B_ROW_MAJOR> sb;
    BrainMmrStageI8 sq;
    sa.init(a, row0, rows, len, tid);
    if (B_I8) {
        sq.init(bq, bs, col0, tid);
    } else {
        sb.init(b, col0, cols, len, tid);
    }

    float c[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) { c[i][j] = 0.0f; }
    }

    const unsigned int nchunks = (len + BRAIN_MMR_BK - 1) / BRAIN_MMR_BK;
    sa.load(0, rows, len, row0, rows);
    if (B_I8) { sq.load(0, cols, len); } else { sb.load(0, cols, len, col0, cols); }
    sa.store(sh.a[0]);
    if (B_I8) { sq.store(sh.b[0], col0); } else { sb.store(sh.b[0]); }
    __syncthreads();

    for (unsigned int ch = 0; ch < nchunks; ++ch) {
        const unsigned int buf = ch & 1u;
        const bool has_next = ch + 1 < nchunks;
        if (has_next) {
            sa.load((ch + 1) * BRAIN_MMR_BK, rows, len, row0, rows);
            if (B_I8) { sq.load((ch + 1) * BRAIN_MMR_BK, cols, len); } else { sb.load((ch + 1) * BRAIN_MMR_BK, cols, len, col0, cols); }
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
                for (int j = 0; j < 8; ++j) { c[i][j] = FUSED ? __fmaf_rn(av[i], bv[j], c[i][j]) : __fadd_rn(c[i][j], __fmul_rn(av[i], bv[j])); }
            }
        }
        if (has_next) {
            sa.store(sh.a[buf ^ 1u]);
            if (B_I8) { sq.store(sh.b[buf ^ 1u], col0); } else { sb.store(sh.b[buf ^ 1u]); }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned int r = row0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + (i - 4));
        if (r >= rows) { continue; }
        float* orow = out + (size_t)r * cols;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned int cbase = col0 + h * 64 + tx * 4;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (cbase + j < cols) {
                    const float v = c[i][h * 4 + j];
                    orow[cbase + j] = accumulate ? __fadd_rn(orow[cbase + j], v) : v;
                }
            }
        }
    }
}

// Whether 16-byte loads are legal: both operand bases aligned and both row
// lengths whole numbers of four floats.
__device__ __forceinline__ bool brain_mmr_vec(const float* a, const float* b, unsigned int la, unsigned int lb) {
    return la % 4 == 0 && lb % 4 == 0 && ((reinterpret_cast<size_t>(a) | reinterpret_cast<size_t>(b)) & 15u) == 0;
}

// The register budget is capped so two blocks share an SM (see the header).
// One shared allocation per entry serves both instantiations: a `__shared__`
// inside the templated body would be one per instantiation.
// The three GEMM shapes, each at both accumulations: the reference's two
// roundings (the bit-identical redirects) or one fused multiply-add per term
// (the `_fma` entries a trainer asks for by name, where bit identity with a
// WGSL twin is not the contract and the fused form runs at the FMA rate).
template <bool FUSED>
__device__ __forceinline__ void brain_mmr_fwd(BrainMmrShared& sh, const unsigned int* params, const float* x, const float* w, float* out) {
    const unsigned int m = params[0], k = params[1], n = params[2];
    if (brain_mmr_vec(x, w, k, k)) {
        brain_mmr_block<true, true, true, FUSED>(sh, m, n, k, x, w, out, false);
    } else {
        brain_mmr_block<false, true, true, FUSED>(sh, m, n, k, x, w, out, false);
    }
}

template <bool FUSED>
__device__ __forceinline__ void brain_mmr_dx(BrainMmrShared& sh, const unsigned int* params, const float* dy, const float* w, float* out) {
    const unsigned int m = params[0], k = params[1], n = params[2];
    const bool accumulate = params[3] != 0;
    if (brain_mmr_vec(dy, w, n, k)) {
        brain_mmr_block<true, true, false, FUSED>(sh, m, k, n, dy, w, out, accumulate);
    } else {
        brain_mmr_block<false, true, false, FUSED>(sh, m, k, n, dy, w, out, accumulate);
    }
}

template <bool FUSED>
__device__ __forceinline__ void brain_mmr_dw(BrainMmrShared& sh, const unsigned int* params, const float* a, const float* b, float* out) {
    const unsigned int m = params[0], k = params[1], n = params[2];
    if (brain_mmr_vec(a, b, n, k)) {
        brain_mmr_block<true, false, false, FUSED>(sh, n, k, m, a, b, out, true);
    } else {
        brain_mmr_block<false, false, false, FUSED>(sh, n, k, m, a, b, out, true);
    }
}

#define BRAIN_MMR_ENTRY(NAME, BODY, FUSED)                                                                              \
    extern "C" __global__ void __launch_bounds__(BRAIN_MMR_THREADS, 2)                                                 \
    NAME(const unsigned int* params, const float* a, const float* b, float* out) {                                     \
        __shared__ BrainMmrShared sh;                                                                                  \
        BODY<FUSED>(sh, params, a, b, out);                                                                            \
    }

BRAIN_MMR_ENTRY(brain_matmul_f32_reg, brain_mmr_fwd, false)
BRAIN_MMR_ENTRY(brain_matmul_f32_dx_reg, brain_mmr_dx, false)
BRAIN_MMR_ENTRY(brain_matmul_f32_dw_reg, brain_mmr_dw, false)
BRAIN_MMR_ENTRY(brain_matmul_f32_fma, brain_mmr_fwd, true)
BRAIN_MMR_ENTRY(brain_matmul_f32_dx_fma, brain_mmr_dx, true)
BRAIN_MMR_ENTRY(brain_matmul_f32_dw_fma, brain_mmr_dw, true)

extern "C" __global__ void __launch_bounds__(BRAIN_MMR_THREADS, 2)
brain_matmul_i8w_dx(const unsigned int* params, const float* dy, const unsigned int* wq, const float* sw, float* out) {
    __shared__ BrainMmrShared sh;
    const unsigned int m = params[0], k = params[1], n = params[2];
    const bool accumulate = params[3] != 0;
    if (brain_mmr_vec(dy, dy, n, n)) {
        brain_mmr_block<true, true, false, true, true>(sh, m, k, n, dy, nullptr, out, accumulate, wq, sw);
    } else {
        brain_mmr_block<false, true, false, true, true>(sh, m, k, n, dy, nullptr, out, accumulate, wq, sw);
    }
}
