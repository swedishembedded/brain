// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving GGUF K-quant models at
// the speed of the memory system for its clients. If your team needs expertise
// in native CUDA kernels for 4-bit LLM decode on the GPUs it already owns,
// you can procure our services by sending an email to info@swedishembedded.com.
//
// Skinny-M affine K-quant matmul (GGUF Q4_K / Q5_K, `model::kquant`'s device
// layout) - the native CUDA form of `matmul_kq_gemv.wgsl`, with its buffer
// layout and uniform:
//
//   params : u32 [m, k, n]       the RAW logical K, a multiple of 32
//   xq     : [M, K/4] u32        activations, 4 signed int8 per word
//   wq     : [N, K*BITS/32] u32  unsigned codes, K-contiguous, low bits first
//                                (code b of word w is element w*(32/BITS) + b)
//   sx     : [M] f32             per-row activation scale
//   wsm    : [N, ceil(K/64)] u32 per-group (sc, m) byte pairs, two groups a
//                                word: group g at bits 16*(g%2), sc low byte
//   wd     : [N, ceil(K/256)] u32 per-super-block (d, dmin) f16 pair, d low
//   xgs    : [M, K/32] f32       activation group sums (bound, not read: the
//                                kernel sums the same int8 bytes itself)
//   out    : [M, N] f32          out[m, n] = sx[m] * sum_g( ds*A[m,n,g] - dm*S[m,g] )
//
// with ds = d*sc and dm = dmin*m of weight row n's group g, A the int8 dot of
// the group's 32 activations with its 32 codes, and S the sum of the group's
// 32 activations.
//
// Decode reads every weight byte once per token, so the kernel is built to
// keep the memory system busy: each lane loads one whole 32-element group of
// codes per 16 (Q4_K) or 32 (Q5_K) bytes with `ld.global.nc` vector loads,
// sixteen lanes share a weight row, sixteen weight rows share a block and the
// slice of x every row reads is staged once per block in shared memory,
// double-buffered (the structure of `matmul_q4_gemv.cu`).
//
// Arithmetic: A and S of a group are exact int32 sums (dp4a; S as a dp4a
// against a word of ones), so a group costs two roundings - `fma(ds, A, acc)`
// then `fma(-dm, S, acc)` - where the WGSL kernel rounds once per 4-element
// quad. The two tiers therefore agree to the fp32 summation bound, not
// bitwise; the gate is `tests/kq_native.rs`, against an f64 oracle.
//
// No `__restrict__` anywhere, deliberately: brain's device buffers alias by
// design (a sliced step binds ranges of one allocation).

// Output columns (weight rows) one block covers, and threads per weight row.
#define BRAIN_KQG_COLS 16
#define BRAIN_KQG_LANES 16
#define BRAIN_KQG_THREADS (BRAIN_KQG_COLS * BRAIN_KQG_LANES)
// Rows of x one block covers; larger m is split over blocks by the grid.
#define BRAIN_KQG_ROWS 8
// Groups per thread in one stage of loads.
#define BRAIN_KQG_UNROLL 2
// Blocks per SM the register budget is sized for.
#define BRAIN_KQG_MIN_BLOCKS 3
// Weight-scale groups per `wd` entry (a 256-element super-block).
#define BRAIN_KQG_GPS 8

// 16-byte load that does not allocate in L1 where the part has the qualifier
// (the weights stream through once), plain read-only load elsewhere.
__device__ __forceinline__ uint4 brain_kqg_ld_stream(const uint4* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    uint4 v;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p));
    return v;
#else
    return __ldg(p);
#endif
}

template <bool VEC>
__device__ __forceinline__ uint4 brain_kqg_load_w(const unsigned int* p) {
    if (VEC) { return brain_kqg_ld_stream(reinterpret_cast<const uint4*>(p)); }
    return make_uint4(__ldg(p), __ldg(p + 1), __ldg(p + 2), __ldg(p + 3));
}

template <bool VEC>
__device__ __forceinline__ uint4 brain_kqg_load_x(const unsigned int* p) {
    if (VEC) { return *reinterpret_cast<const uint4*>(p); }
    return make_uint4(p[0], p[1], p[2], p[3]);
}

// One group's codes: 32 elements of BITS bits, BITS/4 vectors of 16 bytes.
template <int BITS>
struct BrainKqCodes {
    uint4 v[BITS / 4];
};

// What one thread has in flight for `UNROLL` groups of its row: the codes and
// the two words their scales are packed in.
template <int BITS>
struct BrainKqStage {
    BrainKqCodes<BITS> w[BRAIN_KQG_UNROLL];
    unsigned int sm[BRAIN_KQG_UNROLL];
    unsigned int d[BRAIN_KQG_UNROLL];
};

// A group's 32 codes as the eight dp4a operands x's eight words meet: operand
// j holds elements 4j..4j+3. A 4-bit word's eight nibbles (element 8c + b at
// bits 4b) split into its even and odd nibbles and interleave by two byte
// permutes; an 8-bit word already is an operand. Codes are unsigned and below
// 128, so dp4a's signed reading of a byte is the code.
template <int BITS>
__device__ __forceinline__ void brain_kqg_operands(const BrainKqCodes<BITS>& c, unsigned int (&op)[8]) {
    if (BITS == 4) {
        const unsigned int words[4] = {c.v[0].x, c.v[0].y, c.v[0].z, c.v[0].w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned int even = words[j] & 0x0f0f0f0fu;
            const unsigned int odd = (words[j] >> 4) & 0x0f0f0f0fu;
            op[2 * j] = __byte_perm(even, odd, 0x5140u);
            op[2 * j + 1] = __byte_perm(even, odd, 0x7362u);
        }
    } else {
        const uint4& a = c.v[0];
        const uint4& b = c.v[BITS / 4 - 1];
        op[0] = a.x; op[1] = a.y; op[2] = a.z; op[3] = a.w;
        op[4] = b.x; op[5] = b.y; op[6] = b.z; op[7] = b.w;
    }
}

// An f16 bit pattern as f32 (exact).
__device__ __forceinline__ float brain_kqg_f16(unsigned int bits) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(static_cast<unsigned short>(bits)));
    return f;
}

// `(ds, dm)` of group `g`, from its `wsm` and `wd` words.
__device__ __forceinline__ float2 brain_kqg_scales(unsigned int g, unsigned int sm, unsigned int dw) {
    const unsigned int shift = (g & 1u) * 16u;
    const float d = brain_kqg_f16(dw & 0xffffu);
    const float dmin = brain_kqg_f16(dw >> 16);
    return make_float2(__fmul_rn(d, static_cast<float>((sm >> shift) & 0xffu)), __fmul_rn(dmin, static_cast<float>((sm >> (shift + 8u)) & 0xffu)));
}

// One group against one row's 32 activations (two 16-byte vectors).
__device__ __forceinline__ float brain_kqg_group(float acc, const uint4& xa, const uint4& xb, const unsigned int (&op)[8], float2 s) {
    const unsigned int x[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    int a = 0;
    int sum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        a = __dp4a(static_cast<int>(x[j]), static_cast<int>(op[j]), a);
        sum = __dp4a(static_cast<int>(x[j]), 0x01010101, sum);
    }
    return __fmaf_rn(-s.y, static_cast<float>(sum), __fmaf_rn(s.x, static_cast<float>(a), acc));
}

// Vectors of x (16 bytes) per row of x in one stage of the shared tile: two
// per group.
#define BRAIN_KQG_XVECS (2 * BRAIN_KQG_UNROLL * BRAIN_KQG_LANES)
typedef uint4 BrainKqXTile[2][BRAIN_KQG_ROWS][BRAIN_KQG_XVECS];

// `MR` rows of x against one weight row per 16 threads; lane j of a row owns
// groups j, j + 16, ...
template <int BITS, int MR, bool VEC>
__device__ __forceinline__ void brain_kqg_tile(BrainKqXTile xs, const unsigned int* xq, const unsigned int* wq, const float* sx,
                                               const unsigned int* wsm, const unsigned int* wd, float* out, unsigned int mrows,
                                               unsigned int m0, unsigned int k, unsigned int n, unsigned int col) {
    constexpr int UNROLL = BRAIN_KQG_UNROLL;
    static_assert(MR <= BRAIN_KQG_ROWS, "the shared tile is sized for the widest instance");
    const unsigned int lane = threadIdx.x % BRAIN_KQG_LANES;
    const bool live = col < n;
    const unsigned int groups = k >> 5;
    const unsigned int xvecs = groups * 2u;
    const unsigned int iters = (groups + BRAIN_KQG_LANES - 1) / BRAIN_KQG_LANES;
    const unsigned long long row = live ? col : 0u;
    const unsigned int* wrow = wq + row * ((unsigned long long)k * BITS / 32u);
    const unsigned int* smrow = wsm + row * ((groups + 1u) / 2u);
    const unsigned int* drow = wd + row * ((groups + BRAIN_KQG_GPS - 1u) / BRAIN_KQG_GPS);

    float acc[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) { acc[r] = 0.0f; }

    int buf = 0;
    for (unsigned int i0 = 0; i0 < iters; i0 += UNROLL, buf ^= 1) {
        // Issue every weight load of the stage before any is consumed.
        BrainKqStage<BITS> st;
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const unsigned int g = (i0 + u) * BRAIN_KQG_LANES + lane;
            if (live && g < groups) {
#pragma unroll
                for (int v = 0; v < BITS / 4; ++v) { st.w[u].v[v] = brain_kqg_load_w<VEC>(wrow + (unsigned long long)g * BITS + 4u * v); }
                st.sm[u] = __ldg(smrow + g / 2u);
                st.d[u] = __ldg(drow + g / BRAIN_KQG_GPS);
            }
        }
        // Rows of x past `mrows` are not loaded: their outputs are discarded.
#pragma unroll
        for (int e0 = 0; e0 < MR * BRAIN_KQG_XVECS; e0 += BRAIN_KQG_THREADS) {
            const int e = e0 + threadIdx.x;
            const int r = e / BRAIN_KQG_XVECS;
            const int v = e - r * BRAIN_KQG_XVECS;
            const unsigned int gv = 2u * i0 * BRAIN_KQG_LANES + v;
            if (static_cast<unsigned int>(r) < mrows && gv < xvecs) {
                xs[buf][r][v] = brain_kqg_load_x<VEC>(xq + (unsigned long long)(m0 + r) * (k >> 2) + 4ull * gv);
            }
        }
        __syncthreads();
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const unsigned int g = (i0 + u) * BRAIN_KQG_LANES + lane;
            if (live && g < groups) {
                const int v = 2 * (u * BRAIN_KQG_LANES + static_cast<int>(lane));
                unsigned int op[8];
                brain_kqg_operands<BITS>(st.w[u], op);
                const float2 s = brain_kqg_scales(g, st.sm[u], st.d[u]);
#pragma unroll
                for (int r = 0; r < MR; ++r) { acc[r] = brain_kqg_group(acc[r], xs[buf][r][v], xs[buf][r][v + 1], op, s); }
            }
        }
    }
    // Fold the row's sixteen lanes; every lane runs the shuffles.
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        float total = acc[r];
#pragma unroll
        for (int off = BRAIN_KQG_LANES / 2; off > 0; off >>= 1) { total += __shfl_xor_sync(0xffffffffu, total, off, BRAIN_KQG_LANES); }
        if (lane == 0 && live && static_cast<unsigned int>(r) < mrows) { out[(unsigned long long)(m0 + r) * n + col] = __fmul_rn(total, sx[m0 + r]); }
    }
}

template <int BITS, bool VEC>
__device__ __forceinline__ void brain_kqg_select(BrainKqXTile xs, const unsigned int* xq, const unsigned int* wq, const float* sx,
                                                 const unsigned int* wsm, const unsigned int* wd, float* out, unsigned int mrows,
                                                 unsigned int m0, unsigned int k, unsigned int n, unsigned int col) {
    if (mrows <= 1u)      { brain_kqg_tile<BITS, 1, VEC>(xs, xq, wq, sx, wsm, wd, out, mrows, m0, k, n, col); }
    else if (mrows <= 2u) { brain_kqg_tile<BITS, 2, VEC>(xs, xq, wq, sx, wsm, wd, out, mrows, m0, k, n, col); }
    else if (mrows <= 4u) { brain_kqg_tile<BITS, 4, VEC>(xs, xq, wq, sx, wsm, wd, out, mrows, m0, k, n, col); }
    else                  { brain_kqg_tile<BITS, 8, VEC>(xs, xq, wq, sx, wsm, wd, out, mrows, m0, k, n, col); }
}

// One block's share: the `tile_m`-th group of up to BRAIN_KQG_ROWS rows of x
// against the `tile_n`-th group of BRAIN_KQG_COLS weight rows.
template <int BITS>
__device__ __forceinline__ void brain_kqg_block(const unsigned int* params, const unsigned int* xq, const unsigned int* wq, const float* sx,
                                                const unsigned int* wsm, const unsigned int* wd, float* out) {
    const unsigned int m = params[0];
    const unsigned int k = params[1];
    const unsigned int n = params[2];
    // Flat 1-D block count the host may wrap into a second grid dimension.
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    // Row tiles of one weight tile are adjacent in launch order, so they
    // run together and the weights they share are read from memory once.
    const unsigned int tiles_m = (m + BRAIN_KQG_ROWS - 1u) / BRAIN_KQG_ROWS;
    if (tiles_m == 0u || n == 0u) { return; }
    const unsigned int tile_n = blk / tiles_m;
    const unsigned int tile_m = blk - tile_n * tiles_m;
    const unsigned int m0 = tile_m * BRAIN_KQG_ROWS;
    if (m0 >= m) { return; }  // block-uniform
    const unsigned int mrows = min(m - m0, (unsigned int)BRAIN_KQG_ROWS);
    const unsigned int col = tile_n * BRAIN_KQG_COLS + threadIdx.x / BRAIN_KQG_LANES;

    __shared__ BrainKqXTile xs;
    const bool aligned = ((reinterpret_cast<unsigned long long>(xq) | reinterpret_cast<unsigned long long>(wq)) & 15ull) == 0ull;
    if (aligned) {
        brain_kqg_select<BITS, true>(xs, xq, wq, sx, wsm, wd, out, mrows, m0, k, n, col);
    } else {
        brain_kqg_select<BITS, false>(xs, xq, wq, sx, wsm, wd, out, mrows, m0, k, n, col);
    }
}

// Programmatic dependent launch (compute capability 9.0): wait for the kernel
// before this one, then let the one after it become resident. A no-op below
// 9.0 and on a launch without the attribute.
#ifndef BRAIN_PDL_ENTRY
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define BRAIN_PDL_ENTRY() do { asm volatile("griddepcontrol.wait;\n" ::: "memory"); asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); } while (0)
#else
#define BRAIN_PDL_ENTRY() do {} while (0)
#endif
#endif

// Q4_K: 4-bit codes.
extern "C" __global__ void __launch_bounds__(BRAIN_KQG_THREADS, BRAIN_KQG_MIN_BLOCKS)
brain_matmul_kq4_gemv(const unsigned int* params, const unsigned int* xq, const unsigned int* wq, const float* sx,
                      const unsigned int* wsm, const unsigned int* wd, const float* xgs, float* out) {
    BRAIN_PDL_ENTRY();
    (void)xgs;
    brain_kqg_block<4>(params, xq, wq, sx, wsm, wd, out);
}

// Q5_K: 5-bit codes held in 8-bit slots.
extern "C" __global__ void __launch_bounds__(BRAIN_KQG_THREADS, BRAIN_KQG_MIN_BLOCKS)
brain_matmul_kq8_gemv(const unsigned int* params, const unsigned int* xq, const unsigned int* wq, const float* sx,
                      const unsigned int* wsm, const unsigned int* wd, const float* xgs, float* out) {
    BRAIN_PDL_ENTRY();
    (void)xgs;
    brain_kqg_block<8>(params, xq, wq, sx, wsm, wd, out);
}
