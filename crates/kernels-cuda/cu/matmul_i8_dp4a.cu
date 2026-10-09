// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements int8 GEMMs for diffusion transformers and
// LLM prefill on GPUs without int8 tensor cores for its clients. If your team
// needs expertise in getting quantised models onto the packed-dot-product
// roof of the hardware they already own then you can procure our services by
// sending an email to info@swedishembedded.com.
//
// Tiled INT8 GEMM, out = dequant(x_q @ W_q^T) - the hand-written CUDA form of
// `matmul_i8_dyn.wgsl`, with the identical buffer layout, the identical
// uniform and the identical RESULT TO THE LAST BIT.
//
//   params : u32 [m, kg, n]   kg = K/4 packed words per row, K a multiple of 32
//   x_q    : [M, kg] u32      activations, 4 signed int8 per word
//   w_q    : [N, kg] u32      weights, 4 signed int8 per word
//   sx     : [M]     f32      per-row activation scale
//   sw     : [N, kg/8] f32    per-32-element-group weight scale
//   out    : [M, N]  f32      out[m, n] = sx[m] * sum_g f32(sum_{k in g} x*w) * sw[n, g]
//
// What it computes, exactly
// -------------------------
// The WGSL kernel sums each 32-element weight-scale group in int32 (exact in
// any order), then folds the groups into one fp32 running total per output
// in ascending group order as `d = d + f32(c) * sw` with the product and the
// sum individually rounded, and multiplies by `sx` last. This kernel does
// the same operations in the same order with explicit round-to-nearest
// intrinsics, so no contraction can move a bit; the backend also compiles
// with `--fmad=false`.
//
// The one trick is in how `f32(c)` is formed. Each int32 accumulator starts
// at the bit pattern of the float 1.5 * 2^23 (0x4B400000); `__dp4a` adds the
// group's integer sum to that bit pattern, and since a group's sum is bounded
// by 32 * 128 * 128 = 2^19 < 2^22 the result is the float 1.5 * 2^23 + c
// EXACTLY. One float subtraction recovers `c` as a float - exact - where an
// int-to-float conversion runs at a quarter of the arithmetic rate on parts
// whose conversion unit is narrow. The fold therefore costs three full-rate
// operations per eight `__dp4a`.
//
// Tiling
// ------
// 128 x 128 outputs per 256-thread block, each thread an 8 x 8 register block
// (rows ty*4 + {0..3} and 64 + ty*4 + {0..3}, columns likewise from tx), so
// a thread reads its eight activation words and eight weight words of one
// k-step as four 16-byte shared loads and issues 64 `__dp4a` on them - a
// 16:1 math-to-load ratio, the balance a 4-wide packed dot needs (a fp32
// tile's 4:1 leaves the load pipe as the limit; see the kernel rules, C7).
// Shared tiles are k-MAJOR (`[k-word][row]`), so those loads are contiguous.
// One k-chunk is 16 words (two weight-scale groups), double-buffered: the
// next chunk's global loads are issued before the current chunk is consumed
// and land in the other buffer, one barrier per chunk.
//
// Global loads: four consecutive threads read the 64 contiguous bytes of one
// row's chunk, so a warp covers eight rows' chunks in whole 32-byte sectors.
// The transposing shared stores are padded to keep them at most two-way
// conflicted; the inner loop's loads are conflict-free.
//
// Edges: rows of x past `m` and weight rows past `n` stage zeros and are
// never stored; a chunk whose second group lies past `kg` stages zeros and
// does not fold it (folding a zero group would turn a -0.0 running total into
// +0.0, which the WGSL kernel never does). Bindings that are not 16-byte
// aligned take the scalar-load instantiation, which computes the same bits.
//
// No `__restrict__` anywhere, deliberately: brain's device buffers alias by
// design (a sliced step binds ranges of one allocation).

#define BRAIN_I8D_BM 128
#define BRAIN_I8D_BN 128
// Packed words per k-chunk: two 8-word weight-scale groups.
#define BRAIN_I8D_BKW 16
#define BRAIN_I8D_GW 8
#define BRAIN_I8D_THREADS 256
// Shared row stride in words: the tile width plus four, so the transposing
// stores of four threads that share a row spread over the banks while every
// 16-byte read of the inner loop stays aligned.
#define BRAIN_I8D_SP (BRAIN_I8D_BM + 4)
// 1.5 * 2^23: the float whose low mantissa bits hold an int32 group sum.
#define BRAIN_I8D_BIAS_BITS 0x4B400000
#define BRAIN_I8D_BIAS_F 12582912.0f

struct BrainI8dShared {
    unsigned int a[2][BRAIN_I8D_BKW][BRAIN_I8D_SP];
    unsigned int b[2][BRAIN_I8D_BKW][BRAIN_I8D_SP];
    float s[2][2][BRAIN_I8D_BN];
};

// Four consecutive words of row `row` starting at word `k`, zero past the
// row's end or for a row outside the operand.
template <bool VEC>
__device__ __forceinline__ uint4 brain_i8d_load(const unsigned int* base, bool row_ok, unsigned int k, unsigned int kg) {
    if (!row_ok || k >= kg) { return make_uint4(0u, 0u, 0u, 0u); }
    if (VEC) { return __ldg(reinterpret_cast<const uint4*>(base + k)); }
    return make_uint4(__ldg(base + k), __ldg(base + k + 1), __ldg(base + k + 2), __ldg(base + k + 3));
}

template <bool VEC>
__device__ __forceinline__ void brain_i8d_block(BrainI8dShared& sh, const unsigned int* params, const unsigned int* xq,
                                                const unsigned int* wq, const float* sx, const float* sw, float* out) {

    const unsigned int m = params[0];
    const unsigned int kg = params[1];
    const unsigned int n = params[2];
    const unsigned int ng = kg / BRAIN_I8D_GW;
    const unsigned int tiles_n = (n + BRAIN_I8D_BN - 1) / BRAIN_I8D_BN;
    const unsigned int tiles = ((m + BRAIN_I8D_BM - 1) / BRAIN_I8D_BM) * tiles_n;
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (blk >= tiles) { return; }
    const unsigned int row0 = (blk / tiles_n) * BRAIN_I8D_BM;
    const unsigned int col0 = (blk % tiles_n) * BRAIN_I8D_BN;

    const unsigned int tid = threadIdx.x;
    const unsigned int ty = tid / 16;
    const unsigned int tx = tid % 16;

    // Staging: thread t covers vectors t and t + 256 of a chunk's 512
    // (128 rows x 4 vectors of 4 words); four consecutive threads share a row.
    const unsigned int sr0 = tid / 4;
    const unsigned int sr1 = sr0 + 64;
    const unsigned int sq = (tid % 4) * 4;
    const bool a0_ok = row0 + sr0 < m;
    const bool a1_ok = row0 + sr1 < m;
    const bool b0_ok = col0 + sr0 < n;
    const bool b1_ok = col0 + sr1 < n;
    const unsigned int* a0_base = xq + (size_t)(a0_ok ? row0 + sr0 : 0) * kg;
    const unsigned int* a1_base = xq + (size_t)(a1_ok ? row0 + sr1 : 0) * kg;
    const unsigned int* b0_base = wq + (size_t)(b0_ok ? col0 + sr0 : 0) * kg;
    const unsigned int* b1_base = wq + (size_t)(b1_ok ? col0 + sr1 : 0) * kg;
    // Scale staging: thread t stages group (t / 128) of the chunk for column t % 128.
    const unsigned int sg = tid / BRAIN_I8D_BN;
    const unsigned int scol = tid % BRAIN_I8D_BN;
    const bool s_ok = col0 + scol < n;

    int c[8][8];
    float d[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            c[i][j] = 0;
            d[i][j] = 0.0f;
        }
    }

    const unsigned int nchunks = (kg + BRAIN_I8D_BKW - 1) / BRAIN_I8D_BKW;

    // Prime chunk 0 into buffer 0.
    {
        const uint4 va0 = brain_i8d_load<VEC>(a0_base, a0_ok, sq, kg);
        const uint4 va1 = brain_i8d_load<VEC>(a1_base, a1_ok, sq, kg);
        const uint4 vb0 = brain_i8d_load<VEC>(b0_base, b0_ok, sq, kg);
        const uint4 vb1 = brain_i8d_load<VEC>(b1_base, b1_ok, sq, kg);
        sh.a[0][sq + 0][sr0] = va0.x; sh.a[0][sq + 1][sr0] = va0.y; sh.a[0][sq + 2][sr0] = va0.z; sh.a[0][sq + 3][sr0] = va0.w;
        sh.a[0][sq + 0][sr1] = va1.x; sh.a[0][sq + 1][sr1] = va1.y; sh.a[0][sq + 2][sr1] = va1.z; sh.a[0][sq + 3][sr1] = va1.w;
        sh.b[0][sq + 0][sr0] = vb0.x; sh.b[0][sq + 1][sr0] = vb0.y; sh.b[0][sq + 2][sr0] = vb0.z; sh.b[0][sq + 3][sr0] = vb0.w;
        sh.b[0][sq + 0][sr1] = vb1.x; sh.b[0][sq + 1][sr1] = vb1.y; sh.b[0][sq + 2][sr1] = vb1.z; sh.b[0][sq + 3][sr1] = vb1.w;
        sh.s[0][sg][scol] = (s_ok && sg < ng) ? __ldg(sw + (size_t)(col0 + scol) * ng + sg) : 0.0f;
    }
    __syncthreads();

    for (unsigned int ch = 0; ch < nchunks; ++ch) {
        const unsigned int buf = ch & 1u;
        const bool has_next = ch + 1 < nchunks;
        uint4 na0, na1, nb0, nb1;
        float ns = 0.0f;
        if (has_next) {
            const unsigned int k1 = (ch + 1) * BRAIN_I8D_BKW + sq;
            na0 = brain_i8d_load<VEC>(a0_base, a0_ok, k1, kg);
            na1 = brain_i8d_load<VEC>(a1_base, a1_ok, k1, kg);
            nb0 = brain_i8d_load<VEC>(b0_base, b0_ok, k1, kg);
            nb1 = brain_i8d_load<VEC>(b1_base, b1_ok, k1, kg);
            const unsigned int g1 = (ch + 1) * 2 + sg;
            ns = (s_ok && g1 < ng) ? __ldg(sw + (size_t)(col0 + scol) * ng + g1) : 0.0f;
        }

#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int kk = 0; kk < BRAIN_I8D_GW; ++kk) {
                const int k = half * BRAIN_I8D_GW + kk;
                const uint4 a0 = *reinterpret_cast<const uint4*>(&sh.a[buf][k][ty * 4]);
                const uint4 a1 = *reinterpret_cast<const uint4*>(&sh.a[buf][k][64 + ty * 4]);
                const uint4 b0 = *reinterpret_cast<const uint4*>(&sh.b[buf][k][tx * 4]);
                const uint4 b1 = *reinterpret_cast<const uint4*>(&sh.b[buf][k][64 + tx * 4]);
                const unsigned int av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                const unsigned int bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        // A group's first step seeds its accumulator with the
                        // bias itself, so no separate reset is issued.
                        c[i][j] = __dp4a(static_cast<int>(av[i]), static_cast<int>(bv[j]), kk == 0 ? BRAIN_I8D_BIAS_BITS : c[i][j]);
                    }
                }
            }
            // Fold this group, in ascending group order, unless it lies past
            // the operand (a uniform branch: every thread sees the same `g`).
            const unsigned int g = ch * 2 + half;
            if (g < ng) {
                const float4 s0 = *reinterpret_cast<const float4*>(&sh.s[buf][half][tx * 4]);
                const float4 s1 = *reinterpret_cast<const float4*>(&sh.s[buf][half][64 + tx * 4]);
                const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float cf = __fsub_rn(__int_as_float(c[i][j]), BRAIN_I8D_BIAS_F);
                        d[i][j] = __fadd_rn(d[i][j], __fmul_rn(cf, sv[j]));
                    }
                }
            }
        }

        if (has_next) {
            const unsigned int nb = buf ^ 1u;
            sh.a[nb][sq + 0][sr0] = na0.x; sh.a[nb][sq + 1][sr0] = na0.y; sh.a[nb][sq + 2][sr0] = na0.z; sh.a[nb][sq + 3][sr0] = na0.w;
            sh.a[nb][sq + 0][sr1] = na1.x; sh.a[nb][sq + 1][sr1] = na1.y; sh.a[nb][sq + 2][sr1] = na1.z; sh.a[nb][sq + 3][sr1] = na1.w;
            sh.b[nb][sq + 0][sr0] = nb0.x; sh.b[nb][sq + 1][sr0] = nb0.y; sh.b[nb][sq + 2][sr0] = nb0.z; sh.b[nb][sq + 3][sr0] = nb0.w;
            sh.b[nb][sq + 0][sr1] = nb1.x; sh.b[nb][sq + 1][sr1] = nb1.y; sh.b[nb][sq + 2][sr1] = nb1.z; sh.b[nb][sq + 3][sr1] = nb1.w;
            sh.s[nb][sg][scol] = ns;
        }
        __syncthreads();
    }

    // Epilogue: the per-row activation scale, then guarded stores. A thread's
    // four consecutive columns are one 16-byte store where the row allows it.
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned int r = row0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + (i - 4));
        if (r >= m) { continue; }
        const float sv = sx[r];
        float* orow = out + (size_t)r * n;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned int cbase = col0 + h * 64 + tx * 4;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (cbase + j < n) { orow[cbase + j] = __fmul_rn(d[i][h * 4 + j], sv); }
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_I8D_THREADS, 1)
brain_matmul_i8_dp4a(const unsigned int* params, const unsigned int* xq, const unsigned int* wq, const float* sx,
                     const float* sw, float* out) {
    // Every row starts on a 32-byte boundary of its operand (K % 32 == 0), so
    // the base pointers alone decide whether 16-byte loads are legal.
    // One shared allocation for both instantiations: a `__shared__` inside
    // the templated body would be one per instantiation, twice the footprint.
    __shared__ BrainI8dShared sh;
    const bool vec = ((reinterpret_cast<size_t>(xq) | reinterpret_cast<size_t>(wq)) & 15u) == 0;
    if (vec) {
        brain_i8d_block<true>(sh, params, xq, wq, sx, sw, out);
    } else {
        brain_i8d_block<false>(sh, params, xq, wq, sx, sw, out);
    }
}
