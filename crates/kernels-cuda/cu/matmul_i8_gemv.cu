// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for memory-bandwidth-bound LLM
// decode on GPUs for its clients. If your team needs expertise in hand-written
// CUDA kernels that stream quantised weights at the speed of the memory system
// then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Skinny-M INT8 matmul, out = dequant(x_q @ W_q^T) - the hand-written CUDA
// form of `matmul_i8_gemv_reg.wgsl`, with the identical buffer layout, the
// identical uniform and the identical RESULT TO THE LAST BIT.
//
//   params : u32 [m, kg, n]   kg = K/4 packed words per row, K a multiple of 32
//   x_q    : [M, kg] u32      activations, 4 signed int8 per word
//   w_q    : [N, kg] u32      weights, 4 signed int8 per word
//   sx     : [M]     f32      per-row activation scale
//   sw     : [N, kg/8] f32    per-32-element-group weight scale
//   out    : [M, N]  f32      out[m, n] = sx[m] * sum_k x_q[m,k] * w_q[n,k] * sw[n, k/32]
//
// Why this kernel exists
// ----------------------
// Decode streams every weight byte once per token, so it is a memory-bandwidth
// problem and the only figure of merit is the fraction of HBM bandwidth a
// weight row is read at. The WGSL kernel reads one 32-bit word per lane per
// step, so a warp has at most 128 bytes of a row in flight and the memory
// system sits mostly idle. This kernel reads 16 bytes per lane
// (`ld.global.nc.v4`, L1 no-allocate on parts that have the qualifier - the
// weights are used exactly once and must not evict the activations that every
// row re-reads) and computes with `__dp4a`.
//
// What the load structure is, and why
// -----------------------------------
// Sixteen threads serve a row, so a narrow projection runs few threads and
// each thread's latency chain is the kernel's critical path. Two things were
// measured to matter on a GH200 and are built in:
//  * a thread issues ALL the loads of a stage - weights, scales and the
//    matching vectors of x - before it consumes any. Loading x inside the
//    consume loop serialises one L2 round trip per vector and cost about a
//    quarter of the kernel's time on a 5120x6144 projection;
//  * the register budget is capped (`__launch_bounds__`) so five blocks fit
//    per SM. The widest row count (8 rows of x) wants more registers than the
//    decode row count needs; left uncapped it set the occupancy of every
//    launch, including the single-row one that is the whole point.
// For more than two rows of x the staged vectors of x would not fit that
// budget, so those rows are read at the point of use (served from L1: all the
// weight rows of a block read the same x).
//
// How it stays bit-identical to the WGSL kernel
// ---------------------------------------------
// The WGSL kernel gives each of 64 lanes the words g = t, t+64, t+128, ...,
// keeps one f32 accumulator per lane that adds `f32(dot4(x, w)) * sw[g/8]`
// in ascending g, then folds the 64 partials in ascending lane order and
// multiplies by sx. f32 addition is not associative, so matching it means
// keeping those 64 virtual accumulators and that fold order exactly. Here 16
// threads serve one weight row and thread j owns virtual lanes 4j..4j+3: in
// iteration i the words 64i+4j .. 64i+4j+3 are one 16-byte vector, so the
// vector load IS the four virtual lanes' words. The fold walks the 16 threads
// in ascending order with warp shuffles, four accumulators each - ascending
// virtual lane. `--fmad=false` is the backend's default and the products and
// sums are written as explicit round-to-nearest operations, so no contraction
// can change a bit either.
//
// Rows of x beyond `m` in a partial tile are pointed at the tile's first row
// and discarded, exactly as the WGSL kernel points rows past `p.m` at row 0:
// no read ever leaves the binding.
//
// Alignment: the vector path needs x_q and w_q 16-byte aligned; row strides
// are multiples of 32 bytes (K % 32 == 0). A binding that starts at a word
// offset which is not a multiple of four takes the scalar-load path, which
// issues the same loads word by word and computes the same bits.
//
// No `__restrict__` anywhere, deliberately: brain's device buffers alias by
// design (a sliced step binds ranges of one allocation).

// Output columns (weight rows) one block covers, and threads per weight row.
#define BRAIN_I8G_COLS 8
#define BRAIN_I8G_LANES 16
#define BRAIN_I8G_THREADS (BRAIN_I8G_COLS * BRAIN_I8G_LANES)
// Rows of x one block covers; larger m is split over blocks by the grid.
#define BRAIN_I8G_ROWS 8
// Blocks per SM the register budget is sized for.
#define BRAIN_I8G_MIN_BLOCKS 5
// 16-byte vectors per thread in one stage of loads. Five divides the 20
// vectors a thread reads of a 5120-wide row, the hottest decode width.
#define BRAIN_I8G_UNROLL_NARROW 5
#define BRAIN_I8G_UNROLL_WIDE 4
// Up to this many rows of x are staged in registers with the weights.
#define BRAIN_I8G_STAGED_ROWS 2

// 16-byte weight load that does not allocate in L1 where the part has the
// qualifier (the weights stream through once), plain read-only load elsewhere.
__device__ __forceinline__ uint4 brain_ld_stream(const uint4* p) {
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

// Four consecutive words from `p`, as one vector when `VEC` and word by word
// otherwise.
template <bool VEC>
__device__ __forceinline__ uint4 brain_load_w(const unsigned int* p) {
    if (VEC) { return brain_ld_stream(reinterpret_cast<const uint4*>(p)); }
    return make_uint4(__ldg(p), __ldg(p + 1), __ldg(p + 2), __ldg(p + 3));
}

template <bool VEC>
__device__ __forceinline__ uint4 brain_load_x(const unsigned int* p) {
    if (VEC) { return *reinterpret_cast<const uint4*>(p); }
    return make_uint4(p[0], p[1], p[2], p[3]);
}

// One `f32(dot4(x, w)) * s` term added to a virtual lane's accumulator, with
// every operation individually rounded as the reference does.
__device__ __forceinline__ float brain_term(float acc, unsigned int x, unsigned int w, float s) {
    const float d = __int2float_rn(__dp4a(static_cast<int>(x), static_cast<int>(w), 0));
    return __fadd_rn(acc, __fmul_rn(d, s));
}

// What one thread has in flight for `UNROLL` consecutive 16-byte vectors of its
// weight row: the weights, their scales and - for the rows of x that are
// staged (`XROWS` of them) - the matching vectors of x.
template <int XROWS, int UNROLL>
struct BrainStage {
    uint4 w[UNROLL];
    float s[UNROLL];
    // A zero-length array is not standard C++, so an unstaged build keeps one
    // unused row that the compiler drops.
    uint4 x[XROWS > 0 ? XROWS : 1][UNROLL];
};

// Issue every load of the stage `i0 .. i0 + UNROLL` before any is consumed.
// Out-of-range vectors (the row tail, a dead row) load nothing.
template <int XROWS, int UNROLL, bool VEC>
__device__ __forceinline__ void brain_load_stage(BrainStage<XROWS, UNROLL>& st, unsigned int i0, unsigned int lane,
                                                 bool live, unsigned int granules, const unsigned int* wrow,
                                                 const float* srow, const unsigned int* const* xrow) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        const unsigned int g = (i0 + u) * BRAIN_I8G_LANES + lane;
        if (live && g < granules) {
            st.w[u] = brain_load_w<VEC>(wrow + 4ull * g);
            st.s[u] = srow[g >> 1];
#pragma unroll
            for (int r = 0; r < XROWS; ++r) { st.x[r][u] = brain_load_x<VEC>(xrow[r] + 4ull * g); }
        }
    }
}

// `MR` rows of x against one weight row per 16 threads.
template <int MR, int UNROLL, bool VEC>
__device__ __forceinline__ void brain_gemv_tile(const unsigned int* xq, const unsigned int* wq,
                                                const float* sx, const float* sw, float* out,
                                                unsigned int mrows, unsigned int m0,
                                                unsigned int kg, unsigned int n, unsigned int col) {
    constexpr int XROWS = (MR <= BRAIN_I8G_STAGED_ROWS) ? MR : 0;
    const unsigned int lane = threadIdx.x % BRAIN_I8G_LANES;
    const bool live = col < n;
    const unsigned int granules = kg >> 2;                         // 16-byte vectors per row
    const unsigned int iters = (granules + BRAIN_I8G_LANES - 1) / BRAIN_I8G_LANES;

    const unsigned int* wrow = wq + (unsigned long long)(live ? col : 0u) * kg;
    const float* srow = sw + (unsigned long long)(live ? col : 0u) * (kg >> 3);
    const unsigned int* xrow[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        const unsigned int row = (static_cast<unsigned int>(r) < mrows) ? m0 + r : m0;
        xrow[r] = xq + (unsigned long long)row * kg;
    }

    float acc[MR][4];
#pragma unroll
    for (int r = 0; r < MR; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) { acc[r][c] = 0.0f; }
    }

    for (unsigned int i0 = 0; i0 < iters; i0 += UNROLL) {
        BrainStage<XROWS, UNROLL> st;
        brain_load_stage<XROWS, UNROLL, VEC>(st, i0, lane, live, granules, wrow, srow, xrow);
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const unsigned int g = (i0 + u) * BRAIN_I8G_LANES + lane;
            if (live && g < granules) {
#pragma unroll
                for (int r = 0; r < MR; ++r) {
                    const uint4 xv = (r < XROWS) ? st.x[r < XROWS ? r : 0][u] : brain_load_x<VEC>(xrow[r] + 4ull * g);
                    acc[r][0] = brain_term(acc[r][0], xv.x, st.w[u].x, st.s[u]);
                    acc[r][1] = brain_term(acc[r][1], xv.y, st.w[u].y, st.s[u]);
                    acc[r][2] = brain_term(acc[r][2], xv.z, st.w[u].z, st.s[u]);
                    acc[r][3] = brain_term(acc[r][3], xv.w, st.w[u].w, st.s[u]);
                }
            }
        }
    }

    // Fold the 64 virtual lanes in ascending order. Every lane of the warp
    // runs the shuffles (a dead row's lanes simply fold zeros); only the
    // first thread of a live row writes.
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        float total = 0.0f;
#pragma unroll
        for (int j = 0; j < BRAIN_I8G_LANES; ++j) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                total = __fadd_rn(total, __shfl_sync(0xffffffffu, acc[r][c], j, BRAIN_I8G_LANES));
            }
        }
        if (lane == 0 && live && static_cast<unsigned int>(r) < mrows) {
            out[(unsigned long long)(m0 + r) * n + col] = __fmul_rn(total, sx[m0 + r]);
        }
    }
}

template <int MR, bool VEC>
__device__ __forceinline__ void brain_gemv_launch(const unsigned int* xq, const unsigned int* wq,
                                                  const float* sx, const float* sw, float* out,
                                                  unsigned int mrows, unsigned int m0,
                                                  unsigned int kg, unsigned int n, unsigned int col) {
    brain_gemv_tile<MR, (MR <= BRAIN_I8G_STAGED_ROWS) ? BRAIN_I8G_UNROLL_NARROW : BRAIN_I8G_UNROLL_WIDE, VEC>(
        xq, wq, sx, sw, out, mrows, m0, kg, n, col);
}

template <bool VEC>
__device__ __forceinline__ void brain_gemv_select(const unsigned int* xq, const unsigned int* wq,
                                                  const float* sx, const float* sw, float* out,
                                                  unsigned int mrows, unsigned int m0,
                                                  unsigned int kg, unsigned int n, unsigned int col) {
    if (mrows <= 1u)      { brain_gemv_launch<1, VEC>(xq, wq, sx, sw, out, mrows, m0, kg, n, col); }
    else if (mrows <= 2u) { brain_gemv_launch<2, VEC>(xq, wq, sx, sw, out, mrows, m0, kg, n, col); }
    else if (mrows <= 4u) { brain_gemv_launch<4, VEC>(xq, wq, sx, sw, out, mrows, m0, kg, n, col); }
    else                  { brain_gemv_launch<8, VEC>(xq, wq, sx, sw, out, mrows, m0, kg, n, col); }
}

// One block's share of a GEMV: the `tile_m`-th group of up to BRAIN_I8G_ROWS rows
// of x against the `tile_n`-th group of BRAIN_I8G_COLS weight rows. Shared by
// the single-matrix kernel below and by `matmul_i8_gemv_multi`, which runs it
// for whichever of several matrices a block belongs to - so the two produce
// the same bits by construction.
__device__ __forceinline__ void brain_gemv_block(const unsigned int* xq, const unsigned int* wq,
                                                 const float* sx, const float* sw, float* out,
                                                 unsigned int m, unsigned int kg, unsigned int n,
                                                 unsigned int tile_m, unsigned int tile_n) {
    const unsigned int m0 = tile_m * BRAIN_I8G_ROWS;
    if (m0 >= m) { return; }  // block-uniform
    const unsigned int mrows = min(m - m0, (unsigned int)BRAIN_I8G_ROWS);
    const unsigned int col = tile_n * BRAIN_I8G_COLS + threadIdx.x / BRAIN_I8G_LANES;

    const bool aligned = ((reinterpret_cast<unsigned long long>(xq) |
                           reinterpret_cast<unsigned long long>(wq)) & 15ull) == 0ull;
    if (aligned) {
        brain_gemv_select<true>(xq, wq, sx, sw, out, mrows, m0, kg, n, col);
    } else {
        brain_gemv_select<false>(xq, wq, sx, sw, out, mrows, m0, kg, n, col);
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_I8G_THREADS, BRAIN_I8G_MIN_BLOCKS)
brain_matmul_i8_gemv(const unsigned int* params, const unsigned int* xq, const unsigned int* wq,
                     const float* sx, const float* sw, float* out) {
    const unsigned int m = params[0];
    const unsigned int kg = params[1];
    const unsigned int n = params[2];

    // Flat 1-D block count the host may wrap into a second grid dimension,
    // reconstructed the way every kernel in this tree does.
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int tiles_n = (n + BRAIN_I8G_COLS - 1u) / BRAIN_I8G_COLS;
    if (tiles_n == 0u) { return; }
    const unsigned int tile_m = blk / tiles_n;
    const unsigned int tile_n = blk - tile_m * tiles_n;
    brain_gemv_block(xq, wq, sx, sw, out, m, kg, n, tile_m, tile_n);
}

// Up to four int8 matrices that read ONE activation, multiplied in a single
// launch (the `matmul_i8_gemv_multi` registry entry, a second entry point of
// this file).
//
//   params : u32 [m, kg, n0, n1, n2, n3]    n_i = 0 leaves set i unused
//   xq, sx : the shared packed activation and its per-row scales
//   wq_i, sw_i, out_i : set i's packed weights, group scales and output,
//                       exactly the buffers a single `matmul_i8_gemv` takes
//
// Why it exists. A decode layer multiplies the same activation by several
// projections - a GDN layer's qkv, b, a and z, a GQA layer's q, k and v, the
// SwiGLU's gate and up - and each is its own launch whose last wave is mostly
// empty: the 48-row b and a projections run six blocks. Merged, the blocks of
// all the matrices fill the card together and the launch count drops by two
// thirds, which on a replayed graph is also a node boundary each.
//
// Bit identity. A block runs `brain_gemv_block` - the very function the
// single-matrix kernel runs - on the block's own matrix, so every output
// element is the one a separate launch would have produced, bit for bit. The
// blocks of set 0 come first, then set 1, and so on; within a set the mapping
// is the single kernel's (`tile_m` major, `tile_n` minor).

extern "C" __global__ void __launch_bounds__(BRAIN_I8G_THREADS, BRAIN_I8G_MIN_BLOCKS)
brain_matmul_i8_gemv_multi(const unsigned int* params, const unsigned int* xq, const float* sx,
                           const unsigned int* wq0, const float* sw0, float* out0,
                           const unsigned int* wq1, const float* sw1, float* out1,
                           const unsigned int* wq2, const float* sw2, float* out2,
                           const unsigned int* wq3, const float* sw3, float* out3) {
    const unsigned int m = params[0];
    const unsigned int kg = params[1];
    const unsigned int tiles_m = (m + BRAIN_I8G_ROWS - 1u) / BRAIN_I8G_ROWS;

    // Which matrix this block belongs to: set i owns tiles_m * ceil(n_i / COLS)
    // consecutive blocks.
    unsigned int local = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int* wq = wq0;
    const float* sw = sw0;
    float* out = out0;
    unsigned int n = 0u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned int ni = params[2 + i];
        const unsigned int blocks = tiles_m * ((ni + BRAIN_I8G_COLS - 1u) / BRAIN_I8G_COLS);
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

    const unsigned int tiles_n = (n + BRAIN_I8G_COLS - 1u) / BRAIN_I8G_COLS;
    const unsigned int tile_m = local / tiles_n;
    const unsigned int tile_n = local - tile_m * tiles_n;
    brain_gemv_block(xq, wq, sx, sw, out, m, kg, n, tile_m, tile_n);
}
