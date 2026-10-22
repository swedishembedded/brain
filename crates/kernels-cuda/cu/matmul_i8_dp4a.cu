// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements int8 GEMMs for diffusion transformers and
// LLM prefill on GPUs without int8 tensor cores for its clients. If your team
// needs expertise in getting quantised models onto the packed-dot-product
// roof of the hardware they already own then you can procure our services by
// sending an email to info@swedishembedded.com.
//
// Tiled INT8 GEMM, out = dequant(x_q @ W_q^T) - the native CUDA form of
// `matmul_i8_dyn.wgsl`, with the identical buffer layout, the identical
// uniform and the identical RESULT TO THE LAST BIT.
//
//   params : u32 [m, kg, n]   kg = K/4 packed words per row, K a multiple of 32
//   x_q    : [M, kg] u32      activations, 4 signed int8 per word
//   w_q    : [N, kg] u32      weights, 4 signed int8 per word
//   sx     : [M, kg/8] f32    per-32-element-group activation scale
//   sw     : [N, kg/8] f32    per-32-element-group weight scale
//   out    : [M, N]  f32      out[m, n] = sum_g f32(sum_{k in g} x*w) * (sw[n, g] * sx[m, g])
//
// What it computes, exactly
// -------------------------
// The WGSL kernel sums each 32-element weight-scale group in int32 (exact in
// any order), then folds the groups into one fp32 running total per output
// in ascending group order as `d = d + f32(c) * (sw * sx)` with each product
// and the sum individually rounded. This kernel does
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
//
// Affine K-quant weights (GGUF Q4_K, Q5_K; `matmul_kq_dyn.wgsl`'s layout)
// --------------------------------------------------------------------------
// The same block serves them (`brain_matmul_kq4_dp4a`, `brain_matmul_kq8_dp4a`):
//
//   params : u32 [m, k, n]        the RAW K, a multiple of 32
//   wq     : [N, K*BITS/32] u32   unsigned codes, low bits first
//   wsm    : [N, ceil(K/64)] u32  per-group (sc, m) byte pairs, two groups a word
//   wd     : [N, ceil(K/256)] u32 per-super-block (d, dmin) f16 pair
//   xgs    : bound, not read      (the block sums its staged activations)
//   out[m, n] = sum_g sx[m,g] * ( ds[n,g]*A[m,n,g] - dm[n,g]*S[m,g] )
//
// Staging turns a chunk's codes into the int8 words the inner loop already
// multiplies (4-bit codes unpacked by two byte permutes; 8-bit codes are
// already words), so the dp4a loop is the int8 one, unchanged. It also forms
// each staged row's group sums S - exact int32 dp4a sums against a word of
// ones, two threads per group joined by a shuffle - and each column's ds and
// dm. The fold adds `(ds*A - dm*S) * sx` with fused multiply-adds: the K-quant
// instances are held to the fp32 summation bound against an f64 oracle
// (`tests/kq_native.rs`), not to the WGSL tier's bits.

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

// Four consecutive words of row `row` starting at word `k`, zero past the
// row's end or for a row outside the operand.
template <bool VEC>
__device__ __forceinline__ uint4 brain_i8d_load(const unsigned int* base, bool row_ok, unsigned int k, unsigned int kg) {
    if (!row_ok || k >= kg) { return make_uint4(0u, 0u, 0u, 0u); }
    if (VEC) { return __ldg(reinterpret_cast<const uint4*>(base + k)); }
    return make_uint4(__ldg(base + k), __ldg(base + k + 1), __ldg(base + k + 2), __ldg(base + k + 3));
}

// The weight format a block instance reads: int8 codes with an f32 scale per
// group, or affine K-quant codes of 4 or 8 bits.
#define BRAIN_QF_I8 0
#define BRAIN_QF_KQ4 4
#define BRAIN_QF_KQ8 8

struct BrainI8dShared {
    unsigned int a[2][BRAIN_I8D_BKW][BRAIN_I8D_SP];
    unsigned int b[2][BRAIN_I8D_BKW][BRAIN_I8D_SP];
    float s[2][2][BRAIN_I8D_BN];
    float ax[2][2][BRAIN_I8D_BM];
};

// The K-quant block's tiles: the int8 ones plus each column's group mins and
// each row's activation group sums.
struct BrainKqdShared {
    unsigned int a[2][BRAIN_I8D_BKW][BRAIN_I8D_SP];
    unsigned int b[2][BRAIN_I8D_BKW][BRAIN_I8D_SP];
    float s[2][2][BRAIN_I8D_BN];
    float dm[2][2][BRAIN_I8D_BN];
    float xs[2][2][BRAIN_I8D_BM];
    float ax[2][2][BRAIN_I8D_BM];
};

// An f16 bit pattern as f32 (exact).
__device__ __forceinline__ float brain_i8d_f16(unsigned int bits) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(static_cast<unsigned short>(bits)));
    return f;
}

// Four int8 words of weight row `row` starting at element word `k` (4
// elements a word) for format FMT, zero past the row's end or for a row
// outside the operand. A 4-bit row stores two elements words per code word.
template <int FMT, bool VEC>
__device__ __forceinline__ uint4 brain_i8d_load_w(const unsigned int* base, bool row_ok, unsigned int k, unsigned int kg) {
    if (FMT != BRAIN_QF_KQ4) { return brain_i8d_load<VEC>(base, row_ok, k, kg); }
    if (!row_ok || k >= kg) { return make_uint4(0u, 0u, 0u, 0u); }
    const unsigned int* p = base + k / 2;
    const uint2 w = VEC ? __ldg(reinterpret_cast<const uint2*>(p)) : make_uint2(__ldg(p), __ldg(p + 1));
    const unsigned int e0 = w.x & 0x0f0f0f0fu, o0 = (w.x >> 4) & 0x0f0f0f0fu;
    const unsigned int e1 = w.y & 0x0f0f0f0fu, o1 = (w.y >> 4) & 0x0f0f0f0fu;
    return make_uint4(__byte_perm(e0, o0, 0x5140u), __byte_perm(e0, o0, 0x7362u), __byte_perm(e1, o1, 0x5140u), __byte_perm(e1, o1, 0x7362u));
}

// The sum of the sixteen int8 values in `v`, exact.
__device__ __forceinline__ int brain_i8d_sum4(const uint4& v) {
    return __dp4a(static_cast<int>(v.w), 0x01010101, __dp4a(static_cast<int>(v.z), 0x01010101, __dp4a(static_cast<int>(v.y), 0x01010101, __dp4a(static_cast<int>(v.x), 0x01010101, 0))));
}

// One 128 x 128 tile of the GEMM for weight format FMT. `kg` is K/4.
// Scale staging and folding differ by format; the dp4a loop does not.
template <int FMT, bool VEC, class Shared>
__device__ __forceinline__ void brain_i8d_block(Shared& sh, unsigned int m, unsigned int kg, unsigned int n, const unsigned int* xq,
                                                const unsigned int* wq, const float* sx, const float* sw, const unsigned int* wsm,
                                                const unsigned int* wd, float* out) {
    constexpr bool KQ = FMT != BRAIN_QF_I8;
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
    // A weight row is K*BITS/32 words: kg for int8 and 8-bit codes, kg/2 for 4-bit.
    const unsigned int wrow_words = FMT == BRAIN_QF_KQ4 ? kg / 2 : kg;
    const unsigned int* a0_base = xq + (size_t)(a0_ok ? row0 + sr0 : 0) * kg;
    const unsigned int* a1_base = xq + (size_t)(a1_ok ? row0 + sr1 : 0) * kg;
    const unsigned int* b0_base = wq + (size_t)(b0_ok ? col0 + sr0 : 0) * wrow_words;
    const unsigned int* b1_base = wq + (size_t)(b1_ok ? col0 + sr1 : 0) * wrow_words;
    // Scale staging: thread t stages group (t / 128) of the chunk for column t % 128.
    const unsigned int sg = tid / BRAIN_I8D_BN;
    const unsigned int scol = tid % BRAIN_I8D_BN;
    const bool s_ok = col0 + scol < n;
    const unsigned int wsm_words = (ng + 1) / 2;
    const unsigned int wd_words = (ng + 7) / 8;

    // Column `scol`'s scale for group `g` and, for a K-quant, its min: read
    // ahead of the chunk's math like the codes, stored with them.
    auto fetch_scales = [&](unsigned int g) {
        float2 v = make_float2(0.0f, 0.0f);
        if (s_ok && g < ng) {
            if (KQ) {
                const unsigned int smw = __ldg(wsm + (size_t)(col0 + scol) * wsm_words + g / 2) >> ((g & 1u) * 16u);
                const unsigned int dw = __ldg(wd + (size_t)(col0 + scol) * wd_words + g / 8);
                v = make_float2(__fmul_rn(brain_i8d_f16(dw & 0xffffu), static_cast<float>(smw & 0xffu)),
                                __fmul_rn(brain_i8d_f16(dw >> 16), static_cast<float>((smw >> 8) & 0xffu)));
            } else {
                v.x = __ldg(sw + (size_t)(col0 + scol) * ng + g);
            }
        }
        return v;
    };
    auto store_scales = [&](unsigned int b, float2 v) {
        sh.s[b][sg][scol] = v.x;
        if constexpr (KQ) { sh.dm[b][sg][scol] = v.y; }
    };
    // Row `row0 + scol`'s activation scale for group `g`, staged by the same
    // threads as the weight scales.
    const bool x_ok = row0 + scol < m;
    auto fetch_act_scale = [&](unsigned int g) { return (x_ok && g < ng) ? __ldg(sx + (size_t)(row0 + scol) * ng + g) : 0.0f; };
    // The staged rows' activation group sums: the two threads holding a
    // row's half-groups join their exact partial sums.
    auto stage_sums = [&](unsigned int b, const uint4& va0, const uint4& va1) {
        if constexpr (KQ) {
            int p0 = brain_i8d_sum4(va0);
            int p1 = brain_i8d_sum4(va1);
            p0 += __shfl_xor_sync(0xffffffffu, p0, 1);
            p1 += __shfl_xor_sync(0xffffffffu, p1, 1);
            if ((tid & 1u) == 0u) {
                sh.xs[b][(tid % 4) / 2][sr0] = static_cast<float>(p0);
                sh.xs[b][(tid % 4) / 2][sr1] = static_cast<float>(p1);
            }
        }
    };
    auto store = [&](unsigned int b, const uint4& va0, const uint4& va1, const uint4& vb0, const uint4& vb1) {
        sh.a[b][sq + 0][sr0] = va0.x; sh.a[b][sq + 1][sr0] = va0.y; sh.a[b][sq + 2][sr0] = va0.z; sh.a[b][sq + 3][sr0] = va0.w;
        sh.a[b][sq + 0][sr1] = va1.x; sh.a[b][sq + 1][sr1] = va1.y; sh.a[b][sq + 2][sr1] = va1.z; sh.a[b][sq + 3][sr1] = va1.w;
        sh.b[b][sq + 0][sr0] = vb0.x; sh.b[b][sq + 1][sr0] = vb0.y; sh.b[b][sq + 2][sr0] = vb0.z; sh.b[b][sq + 3][sr0] = vb0.w;
        sh.b[b][sq + 0][sr1] = vb1.x; sh.b[b][sq + 1][sr1] = vb1.y; sh.b[b][sq + 2][sr1] = vb1.z; sh.b[b][sq + 3][sr1] = vb1.w;
    };

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
        const uint4 vb0 = brain_i8d_load_w<FMT, VEC>(b0_base, b0_ok, sq, kg);
        const uint4 vb1 = brain_i8d_load_w<FMT, VEC>(b1_base, b1_ok, sq, kg);
        store(0, va0, va1, vb0, vb1);
        stage_sums(0, va0, va1);
        store_scales(0, fetch_scales(sg));
        sh.ax[0][sg][scol] = fetch_act_scale(sg);
    }
    __syncthreads();

    for (unsigned int ch = 0; ch < nchunks; ++ch) {
        const unsigned int buf = ch & 1u;
        const bool has_next = ch + 1 < nchunks;
        uint4 na0, na1, nb0, nb1;
        float2 ns = make_float2(0.0f, 0.0f);
        float nx = 0.0f;
        if (has_next) {
            const unsigned int k1 = (ch + 1) * BRAIN_I8D_BKW + sq;
            na0 = brain_i8d_load<VEC>(a0_base, a0_ok, k1, kg);
            na1 = brain_i8d_load<VEC>(a1_base, a1_ok, k1, kg);
            nb0 = brain_i8d_load_w<FMT, VEC>(b0_base, b0_ok, k1, kg);
            nb1 = brain_i8d_load_w<FMT, VEC>(b1_base, b1_ok, k1, kg);
            ns = fetch_scales((ch + 1) * 2 + sg);
            nx = fetch_act_scale((ch + 1) * 2 + sg);
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
                const float4 ax0 = *reinterpret_cast<const float4*>(&sh.ax[buf][half][ty * 4]);
                const float4 ax1 = *reinterpret_cast<const float4*>(&sh.ax[buf][half][64 + ty * 4]);
                const float axv[8] = {ax0.x, ax0.y, ax0.z, ax0.w, ax1.x, ax1.y, ax1.z, ax1.w};
                if constexpr (KQ) {
                    const float4 m0 = *reinterpret_cast<const float4*>(&sh.dm[buf][half][tx * 4]);
                    const float4 m1 = *reinterpret_cast<const float4*>(&sh.dm[buf][half][64 + tx * 4]);
                    const float4 x0 = *reinterpret_cast<const float4*>(&sh.xs[buf][half][ty * 4]);
                    const float4 x1 = *reinterpret_cast<const float4*>(&sh.xs[buf][half][64 + ty * 4]);
                    const float mv[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
                    const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const float cf = __fsub_rn(__int_as_float(c[i][j]), BRAIN_I8D_BIAS_F);
                            d[i][j] = __fmaf_rn(__fmaf_rn(-mv[j], xv[i], __fmul_rn(cf, sv[j])), axv[i], d[i][j]);
                        }
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const float cf = __fsub_rn(__int_as_float(c[i][j]), BRAIN_I8D_BIAS_F);
                            d[i][j] = __fadd_rn(d[i][j], __fmul_rn(cf, __fmul_rn(sv[j], axv[i])));
                        }
                    }
                }
            }
        }

        if (has_next) {
            const unsigned int nb = buf ^ 1u;
            store(nb, na0, na1, nb0, nb1);
            stage_sums(nb, na0, na1);
            store_scales(nb, ns);
            sh.ax[nb][sg][scol] = nx;
        }
        __syncthreads();
    }

    // Epilogue: guarded stores (both scales are already in).
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
                if (cbase + j < n) { orow[cbase + j] = d[i][h * 4 + j]; }
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
        brain_i8d_block<BRAIN_QF_I8, true>(sh, params[0], params[1], params[2], xq, wq, sx, sw, nullptr, nullptr, out);
    } else {
        brain_i8d_block<BRAIN_QF_I8, false>(sh, params[0], params[1], params[2], xq, wq, sx, sw, nullptr, nullptr, out);
    }
}

// The affine K-quant instances: `params` carries the RAW K.
template <int FMT>
__device__ __forceinline__ void brain_kqd_entry(const unsigned int* params, const unsigned int* xq, const unsigned int* wq, const float* sx,
                                                const unsigned int* wsm, const unsigned int* wd, float* out) {
    __shared__ BrainKqdShared sh;
    const bool vec = ((reinterpret_cast<size_t>(xq) | reinterpret_cast<size_t>(wq)) & 15u) == 0;
    const unsigned int kg = params[1] / 4;
    if (vec) {
        brain_i8d_block<FMT, true>(sh, params[0], kg, params[2], xq, wq, sx, nullptr, wsm, wd, out);
    } else {
        brain_i8d_block<FMT, false>(sh, params[0], kg, params[2], xq, wq, sx, nullptr, wsm, wd, out);
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_I8D_THREADS, 1)
brain_matmul_kq4_dp4a(const unsigned int* params, const unsigned int* xq, const unsigned int* wq, const float* sx,
                      const unsigned int* wsm, const unsigned int* wd, const float* xgs, float* out) {
    (void)xgs;
    brain_kqd_entry<BRAIN_QF_KQ4>(params, xq, wq, sx, wsm, wd, out);
}

extern "C" __global__ void __launch_bounds__(BRAIN_I8D_THREADS, 1)
brain_matmul_kq8_dp4a(const unsigned int* params, const unsigned int* xq, const unsigned int* wq, const float* sx,
                      const unsigned int* wsm, const unsigned int* wd, const float* xgs, float* out) {
    (void)xgs;
    brain_kqd_entry<BRAIN_QF_KQ8>(params, xq, wq, sx, wsm, wd, out);
}
