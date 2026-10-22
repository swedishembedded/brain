// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for memory-bandwidth-bound LLM
// decode on GPUs for its clients. If your team needs expertise in native
// CUDA kernels that stream 4-bit weights at the speed of the memory system
// then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Skinny-M W4A8 matmul, out = dequant(x_q8 @ W_q4^T) - the native CUDA form
// of `matmul_q4_gemv_reg.wgsl`, with the identical buffer layout and the
// identical uniform, bit-identical to it.
//
//   params : u32 [m, k, n]     the RAW logical K, a multiple of 32
//   x_q    : [M, K/4] u32      activations, 4 signed int8 per word
//   w_q    : [N, K/8] u32      weights, 8 signed int4 per word (nibble b of
//                              word g is element 8g + b)
//   sx     : [M, K/32] f32     per-group activation scale
//   sw     : [N, K/32] f32     per-32-element-group weight scale
//   out    : [M, N]   f32      out[m, n] = sum_k x_q[m,k] * w_q[n,k] * (sw[n, k/32] * sx[m, k/32])
//
// Why this kernel exists
// ----------------------
// Decode reads every weight byte once per token, so a 4-bit weight should
// cost half the time of an 8-bit one. The WGSL kernel reads one 32-bit word
// per lane per step, which keeps too few bytes of a row in flight for the
// memory system, and so ran the 4-bit tier SLOWER than the native int8 GEMV
// beside it. This kernel reads 16 bytes of weights per lane
// (`ld.global.nc.v4`, no L1 allocation where the part has the qualifier), the
// structure of `matmul_i8_gemv.cu`. A 16-byte weight vector is exactly one
// 32-element scale group, so each vector meets one scale and 32 bytes of x.
//
// Unpacking. A weight word's eight nibbles become two dp4a operands: the low
// nibble of every byte (`w & 0x0f0f0f0f`, elements 0, 2, 4, 6) and the high
// nibble (`(w >> 4) & 0x0f0f0f0f`, elements 1, 3, 5, 7) are sign-extended in
// place (a byte of 8..15 gains 0xf0) and interleaved by two byte permutes into
// elements 0..3 and 4..7, the bytes `dot4I8Packed` reads in the WGSL kernel.
//
// How it stays bit-identical to the WGSL kernel
// ---------------------------------------------
// The WGSL kernel gives each of 64 lanes the weight words g = t, t+64, ...,
// adds `f32(dot4(x[2g], lo(w)) + dot4(x[2g+1], hi(w))) * (sw[g/4] * sx[m, g/4])`
// to one f32 accumulator per lane in ascending g, then folds the 64 partials
// in ascending lane order. Here 16 threads serve one weight
// row and thread j owns virtual lanes 4j..4j+3: in iteration i the words
// 64i+4j .. 64i+4j+3 are one 16-byte vector, so the vector load IS the four
// virtual lanes' words, and their group (16i + j) is one scale. The two dot
// products of a word are summed as integers (exact, |sum| <= 8 * 128 * 8)
// before the single conversion, as the WGSL kernel does; every float
// operation is an explicit round-to-nearest one and the fold walks the 16
// threads in ascending order, four accumulators each.
//
// Rows of x beyond `m` in a partial tile are pointed at the tile's first row
// and discarded, as the WGSL kernel points rows past `p.m` at row 0.
//
// Alignment: the vector path needs x_q and w_q 16-byte aligned; row strides
// are multiples of 16 bytes (K % 32 == 0). A binding that starts elsewhere
// takes the scalar-load path, which computes the same bits.
//
// No `__restrict__` anywhere, deliberately: brain's device buffers alias by
// design (a sliced step binds ranges of one allocation).

// Output columns (weight rows) one block covers, and threads per weight row.
#define BRAIN_Q4G_COLS 16
#define BRAIN_Q4G_LANES 16
#define BRAIN_Q4G_THREADS (BRAIN_Q4G_COLS * BRAIN_Q4G_LANES)
// Rows of x one block covers; larger m is split over blocks by the grid.
#define BRAIN_Q4G_ROWS 8
// Blocks per SM the register budget is sized for.
#ifndef BRAIN_Q4G_MIN_BLOCKS
#define BRAIN_Q4G_MIN_BLOCKS 3
#endif
// Weight vectors per thread in one stage of loads.
#ifndef BRAIN_Q4G_UNROLL
#define BRAIN_Q4G_UNROLL 2
#endif

// 16-byte weight load that does not allocate in L1 where the part has the
// qualifier (the weights stream through once), plain read-only load elsewhere.
__device__ __forceinline__ uint4 brain_q4g_ld_stream(const uint4* p) {
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
__device__ __forceinline__ uint4 brain_q4g_load_w(const unsigned int* p) {
    if (VEC) { return brain_q4g_ld_stream(reinterpret_cast<const uint4*>(p)); }
    return make_uint4(__ldg(p), __ldg(p + 1), __ldg(p + 2), __ldg(p + 3));
}

template <bool VEC>
__device__ __forceinline__ uint4 brain_q4g_load_x(const unsigned int* p) {
    if (VEC) { return *reinterpret_cast<const uint4*>(p); }
    return make_uint4(p[0], p[1], p[2], p[3]);
}

// Sign-extend the low nibble of each byte (values 0..15) to a full int8.
__device__ __forceinline__ unsigned int brain_q4g_sext(unsigned int v) {
    return v | ((v & 0x08080808u) * 0x1eu);
}

// A weight vector (four words, one scale group) unpacked once into the dp4a
// operands every row of x multiplies: `lo[c]` holds elements 0..3 of word c,
// `hi[c]` elements 4..7. The low nibble of every byte (`w & 0x0f0f0f0f`,
// elements 0, 2, 4, 6) and the high nibble (elements 1, 3, 5, 7) are
// sign-extended in place and interleaved by two byte permutes.
struct BrainQ4W {
    unsigned int lo[4];
    unsigned int hi[4];
};

__device__ __forceinline__ BrainQ4W brain_q4g_unpack(const uint4& w) {
    BrainQ4W u;
    const unsigned int words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const unsigned int even = brain_q4g_sext(words[c] & 0x0f0f0f0fu);
        const unsigned int odd = brain_q4g_sext((words[c] >> 4) & 0x0f0f0f0fu);
        u.lo[c] = __byte_perm(even, odd, 0x5140u);
        u.hi[c] = __byte_perm(even, odd, 0x7362u);
    }
    return u;
}

// One weight word against its two words of x, added to a virtual lane's
// accumulator: `f32(dot4(x0, lo) + dot4(x1, hi)) * s`, the integer sum exact
// and each float operation rounded as the reference does.
__device__ __forceinline__ float brain_q4g_term(float acc, unsigned int x0, unsigned int x1, unsigned int lo, unsigned int hi, float s) {
    const int d = __dp4a(static_cast<int>(x1), static_cast<int>(hi), __dp4a(static_cast<int>(x0), static_cast<int>(lo), 0));
    return __fadd_rn(acc, __fmul_rn(__int2float_rn(d), s));
}

// One unpacked weight vector against one row's 32 bytes of x.
__device__ __forceinline__ void brain_q4g_vec(float (&acc)[4], const uint4& xa, const uint4& xb, const BrainQ4W& w, float s) {
    acc[0] = brain_q4g_term(acc[0], xa.x, xa.y, w.lo[0], w.hi[0], s);
    acc[1] = brain_q4g_term(acc[1], xa.z, xa.w, w.lo[1], w.hi[1], s);
    acc[2] = brain_q4g_term(acc[2], xb.x, xb.y, w.lo[2], w.hi[2], s);
    acc[3] = brain_q4g_term(acc[3], xb.z, xb.w, w.lo[3], w.hi[3], s);
}

// What one thread has in flight for `UNROLL` consecutive weight vectors of
// its row: the weights and their scales.
template <int UNROLL>
struct BrainQ4Stage {
    uint4 w[UNROLL];
    float s[UNROLL];
};

// Issue every weight load of the stage `i0 .. i0 + UNROLL` before any is
// consumed. Vector `g` of a row is scale group `g`.
template <int UNROLL, bool VEC>
__device__ __forceinline__ void brain_q4g_load_stage(BrainQ4Stage<UNROLL>& st, unsigned int i0, unsigned int lane, bool live,
                                                     unsigned int groups, const unsigned int* wrow, const float* srow) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        const unsigned int g = (i0 + u) * BRAIN_Q4G_LANES + lane;
        if (live && g < groups) {
            st.w[u] = brain_q4g_load_w<VEC>(wrow + 4ull * g);
            st.s[u] = srow[g];
        }
    }
}

// Fold the 64 virtual lanes in ascending order and write row `r`'s output.
// Every lane of the warp runs the shuffles; only the first thread of a live
// row writes.
template <int MR>
__device__ __forceinline__ void brain_q4g_fold(const float (&acc)[MR][4], float* out, unsigned int lane,
                                               bool live, unsigned int mrows, unsigned int m0, unsigned int n, unsigned int col) {
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        float total = 0.0f;
#pragma unroll
        for (int j = 0; j < BRAIN_Q4G_LANES; ++j) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                total = __fadd_rn(total, __shfl_sync(0xffffffffu, acc[r][c], j, BRAIN_Q4G_LANES));
            }
        }
        if (lane == 0 && live && static_cast<unsigned int>(r) < mrows) {
            out[(unsigned long long)(m0 + r) * n + col] = total;
        }
    }
}

// Vectors of x (16 bytes) per row of x in one stage of the shared tile: two
// per weight vector.
#define BRAIN_Q4G_XVECS (2 * BRAIN_Q4G_UNROLL * BRAIN_Q4G_LANES)
typedef uint4 BrainQ4XTile[2][BRAIN_Q4G_ROWS][BRAIN_Q4G_XVECS];

// `MR` rows of x against one weight row per 16 threads. Every weight row of
// the block wants the same stretch of x, so the block loads each stage's
// slice of x into shared memory once and the threads read it from there:
// read per thread from global memory, x is TWICE the weights' own bytes (an
// int8 activation against a 4-bit weight) and held the kernel to well under
// the memory system's bandwidth at every row count, one row included. Each
// weight vector is unpacked once and multiplied by every row. Double-
// buffered: one barrier per stage orders the fill against this stage's reads
// and the previous stage's reads against this fill. Every thread runs every
// stage (the barrier needs them).
template <int MR, bool VEC>
__device__ __forceinline__ void brain_q4g_tile(BrainQ4XTile xs, const unsigned int* xq, const unsigned int* wq,
                                               const float* sx, const float* sw, float* out, unsigned int mrows,
                                               unsigned int m0, unsigned int k, unsigned int n, unsigned int col) {
    constexpr int UNROLL = BRAIN_Q4G_UNROLL;
    static_assert(MR <= BRAIN_Q4G_ROWS, "the shared tile is sized for the widest instance");
    const unsigned int lane = threadIdx.x % BRAIN_Q4G_LANES;
    const bool live = col < n;
    const unsigned int groups = k >> 5;
    const unsigned int xvecs = groups * 2u;  // 16-byte vectors per row of x
    const unsigned int iters = (groups + BRAIN_Q4G_LANES - 1) / BRAIN_Q4G_LANES;

    const unsigned int* wrow = wq + (unsigned long long)(live ? col : 0u) * (k >> 3);
    const float* srow = sw + (unsigned long long)(live ? col : 0u) * groups;
    const float* sxrow[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        const unsigned int row = (static_cast<unsigned int>(r) < mrows) ? m0 + r : m0;
        sxrow[r] = sx + (unsigned long long)row * groups;
    }

    float acc[MR][4];
#pragma unroll
    for (int r = 0; r < MR; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) { acc[r][c] = 0.0f; }
    }

    int buf = 0;
    for (unsigned int i0 = 0; i0 < iters; i0 += UNROLL, buf ^= 1) {
        BrainQ4Stage<UNROLL> st;
        brain_q4g_load_stage<UNROLL, VEC>(st, i0, lane, live, groups, wrow, srow);
        // Rows of x past `mrows` are not loaded: their outputs are discarded.
#pragma unroll
        for (int e0 = 0; e0 < MR * BRAIN_Q4G_XVECS; e0 += BRAIN_Q4G_THREADS) {
            const int e = e0 + threadIdx.x;
            const int r = e / BRAIN_Q4G_XVECS;
            const int v = e - r * BRAIN_Q4G_XVECS;
            const unsigned int gv = 2u * i0 * BRAIN_Q4G_LANES + v;
            if (static_cast<unsigned int>(r) < mrows && gv < xvecs) {
                xs[buf][r][v] = brain_q4g_load_x<VEC>(xq + (unsigned long long)(m0 + r) * (k >> 2) + 4ull * gv);
            }
        }
        __syncthreads();
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const unsigned int g = (i0 + u) * BRAIN_Q4G_LANES + lane;
            if (live && g < groups) {
                const int v = 2 * (u * BRAIN_Q4G_LANES + static_cast<int>(lane));
                const BrainQ4W w = brain_q4g_unpack(st.w[u]);
#pragma unroll
                for (int r = 0; r < MR; ++r) { brain_q4g_vec(acc[r], xs[buf][r][v], xs[buf][r][v + 1], w, __fmul_rn(st.s[u], __ldg(sxrow[r] + g))); }
            }
        }
    }
    brain_q4g_fold<MR>(acc, out, lane, live, mrows, m0, n, col);
}

template <bool VEC>
__device__ __forceinline__ void brain_q4g_select(BrainQ4XTile xs, const unsigned int* xq, const unsigned int* wq,
                                                 const float* sx, const float* sw, float* out, unsigned int mrows,
                                                 unsigned int m0, unsigned int k, unsigned int n, unsigned int col) {
    if (mrows <= 1u)      { brain_q4g_tile<1, VEC>(xs, xq, wq, sx, sw, out, mrows, m0, k, n, col); }
    else if (mrows <= 2u) { brain_q4g_tile<2, VEC>(xs, xq, wq, sx, sw, out, mrows, m0, k, n, col); }
    else if (mrows <= 4u) { brain_q4g_tile<4, VEC>(xs, xq, wq, sx, sw, out, mrows, m0, k, n, col); }
    else                  { brain_q4g_tile<8, VEC>(xs, xq, wq, sx, sw, out, mrows, m0, k, n, col); }
}

// One block's share of a GEMV: the `tile_m`-th group of up to BRAIN_Q4G_ROWS
// rows of x against the `tile_n`-th group of BRAIN_Q4G_COLS weight rows.
// Shared by the single-matrix kernel and `matmul_q4_gemv_multi`, so the two
// produce the same bits by construction.
__device__ __forceinline__ void brain_q4g_block(const unsigned int* xq, const unsigned int* wq, const float* sx,
                                                const float* sw, float* out, unsigned int m, unsigned int k,
                                                unsigned int n, unsigned int tile_m, unsigned int tile_n) {
    const unsigned int m0 = tile_m * BRAIN_Q4G_ROWS;
    if (m0 >= m) { return; }  // block-uniform
    const unsigned int mrows = min(m - m0, (unsigned int)BRAIN_Q4G_ROWS);
    const unsigned int col = tile_n * BRAIN_Q4G_COLS + threadIdx.x / BRAIN_Q4G_LANES;

    __shared__ BrainQ4XTile xs;
    const bool aligned = ((reinterpret_cast<unsigned long long>(xq) | reinterpret_cast<unsigned long long>(wq)) & 15ull) == 0ull;
    if (aligned) {
        brain_q4g_select<true>(xs, xq, wq, sx, sw, out, mrows, m0, k, n, col);
    } else {
        brain_q4g_select<false>(xs, xq, wq, sx, sw, out, mrows, m0, k, n, col);
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

extern "C" __global__ void __launch_bounds__(BRAIN_Q4G_THREADS, BRAIN_Q4G_MIN_BLOCKS)
brain_matmul_q4_gemv(const unsigned int* params, const unsigned int* xq, const unsigned int* wq,
                     const float* sx, const float* sw, float* out) {
    BRAIN_PDL_ENTRY();
    const unsigned int m = params[0];
    const unsigned int k = params[1];
    const unsigned int n = params[2];

    // Flat 1-D block count the host may wrap into a second grid dimension.
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    // Row tiles of one weight tile are adjacent in launch order, so they
    // run together and the weights they share are read from memory once.
    const unsigned int tiles_m = (m + BRAIN_Q4G_ROWS - 1u) / BRAIN_Q4G_ROWS;
    if (tiles_m == 0u || n == 0u) { return; }
    const unsigned int tile_n = blk / tiles_m;
    const unsigned int tile_m = blk - tile_n * tiles_m;
    brain_q4g_block(xq, wq, sx, sw, out, m, k, n, tile_m, tile_n);
}

// Up to four q4 matrices that read ONE activation, in a single launch (the
// `matmul_q4_gemv_multi` registry entry) - the q4 twin of
// `brain_matmul_i8_gemv_multi`, with the same block layout: set 0's blocks
// first, then set 1's, each running `brain_q4g_block` on its own matrix, so
// every output is bit-identical to a separate `brain_matmul_q4_gemv`.
//
//   params : u32 [m, k, n0, n1, n2, n3]    n_i = 0 leaves set i unused
//   xq, sx : the shared packed activation and its group scales
//   wq_i, sw_i, out_i : set i's packed weights, group scales and output
extern "C" __global__ void __launch_bounds__(BRAIN_Q4G_THREADS, BRAIN_Q4G_MIN_BLOCKS)
brain_matmul_q4_gemv_multi(const unsigned int* params, const unsigned int* xq, const float* sx,
                           const unsigned int* wq0, const float* sw0, float* out0,
                           const unsigned int* wq1, const float* sw1, float* out1,
                           const unsigned int* wq2, const float* sw2, float* out2,
                           const unsigned int* wq3, const float* sw3, float* out3) {
    BRAIN_PDL_ENTRY();
    const unsigned int m = params[0];
    const unsigned int k = params[1];
    const unsigned int tiles_m = (m + BRAIN_Q4G_ROWS - 1u) / BRAIN_Q4G_ROWS;

    unsigned int local = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int* wq = wq0;
    const float* sw = sw0;
    float* out = out0;
    unsigned int n = 0u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned int ni = params[2 + i];
        const unsigned int blocks = tiles_m * ((ni + BRAIN_Q4G_COLS - 1u) / BRAIN_Q4G_COLS);
        if (n == 0u && local < blocks) {
            n = ni;
            wq = (i == 0) ? wq0 : (i == 1) ? wq1 : (i == 2) ? wq2 : wq3;
            sw = (i == 0) ? sw0 : (i == 1) ? sw1 : (i == 2) ? sw2 : sw3;
            out = (i == 0) ? out0 : (i == 1) ? out1 : (i == 2) ? out2 : out3;
        } else if (n == 0u) {
            local -= blocks;
        }
    }
    if (n == 0u) { return; }  // past the last matrix's blocks; block-uniform

    const unsigned int tile_n = local / tiles_m;
    const unsigned int tile_m = local - tile_n * tiles_m;
    brain_q4g_block(xq, wq, sx, sw, out, m, k, n, tile_m, tile_n);
}
