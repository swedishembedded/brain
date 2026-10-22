// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for memory-bandwidth-bound LLM
// decode on GPUs for its clients. If your team needs expertise in native
// CUDA kernels that stream quantised weights at the speed of the memory system
// then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Skinny-M INT8 matmul, out = dequant(x_q @ W_q^T) - the native CUDA
// form of `matmul_i8_gemv_reg.wgsl`, with the identical buffer layout and the
// identical uniform. Two blocks share the file:
//  * the dp4a block, bit-identical to the WGSL kernel (the result to the last
//    bit), serves every row count below BRAIN_I8G_MMA_MIN_ROWS and every row
//    count on a device without int8 tensor cores (compute capability < 8.0);
//  * the tensor-core block (below `brain_gemv_block_mma`) serves the larger
//    batches from compute capability 8.0 and agrees with the WGSL kernel to
//    f32 rounding of the running sum, not bit for bit.
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
//  * the register budget is capped (`__launch_bounds__`) so three 256-thread
//    blocks fit per SM. The widest row count (8 rows of x) wants more registers than the
//    decode row count needs; left uncapped it set the occupancy of every
//    launch, including the single-row one that is the whole point.
// For more than two rows of x the staged vectors of x would not fit that
// budget, so the block loads each stage's slice of x into shared memory once
// and every weight row of the block reads it from there
// (`brain_gemv_tile_shared_x`). Read per thread from global memory, those loads
// of x were 4x the weights' own traffic at eight rows and held the kernel to
// well under the memory system's bandwidth, whatever the arithmetic cost.
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
#define BRAIN_I8G_COLS 16
#define BRAIN_I8G_LANES 16
#define BRAIN_I8G_THREADS (BRAIN_I8G_COLS * BRAIN_I8G_LANES)
// Rows of x one block covers; larger m is split over blocks by the grid.
#define BRAIN_I8G_ROWS 8
// Blocks per SM the register budget is sized for.
#ifndef BRAIN_I8G_MIN_BLOCKS
#define BRAIN_I8G_MIN_BLOCKS 3
#endif
// From compute capability 8.0 a batch of at least this many rows of x takes the tensor-core block.
#ifndef BRAIN_I8G_MMA_MIN_ROWS
#define BRAIN_I8G_MMA_MIN_ROWS 5
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(BRAIN_I8G_NO_MMA)
#define BRAIN_I8G_USE_MMA 1
#endif
// 16-byte vectors per thread in one stage of loads. Five divides the 20
// vectors a thread reads of a 5120-wide row, the hottest decode width.
#define BRAIN_I8G_UNROLL_NARROW 5
#ifndef BRAIN_I8G_UNROLL_NARROW2
#define BRAIN_I8G_UNROLL_NARROW2 3
#endif
#define BRAIN_I8G_UNROLL_WIDE 2
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

// `MR` > BRAIN_I8G_STAGED_ROWS rows of x against one weight row per 16 threads,
// with the rows of x shared through shared memory.
//
// Why. Every thread of a block walks the same stretch of K, so the sixteen
// weight rows of a block all want the SAME vectors of x, once per row of x. Read
// by each thread from global memory that is MR loads of x per vector of weights,
// 4x the weights' own traffic at eight rows through the read-only cache, and it
// (not the arithmetic, which was measured to be free) held the kernel to half
// the memory system's bandwidth. Here the block loads each stage's slice of x
// once, cooperatively, and the threads read it from shared memory, where the
// two weight rows of a warp (the same 16 vectors twice) are a broadcast.
//
// The arithmetic, the order of every accumulation and the fold are those of
// `brain_gemv_tile`: only where x comes from changes, so the result is bit for
// bit the same. The tile is double-buffered, so one barrier per stage orders
// the fill against the reads of the same stage and the reads of the previous
// stage against this stage's fill. Every thread of a block runs every stage
// (the barrier needs them), live or not.
template <int MR, int UNROLL, bool VEC>
__device__ __forceinline__ void brain_gemv_tile_shared_x(uint4 (*xs)[BRAIN_I8G_ROWS][BRAIN_I8G_UNROLL_WIDE * BRAIN_I8G_LANES],
                                                         const unsigned int* xq, const unsigned int* wq,
                                                         const float* sx, const float* sw, float* out,
                                                         unsigned int mrows, unsigned int m0,
                                                         unsigned int kg, unsigned int n, unsigned int col) {
    constexpr int VECS = UNROLL * BRAIN_I8G_LANES;  // vectors of x per row in one stage
    static_assert(UNROLL == BRAIN_I8G_UNROLL_WIDE && MR <= BRAIN_I8G_ROWS, "the shared tile is sized for the widest instance");
    const unsigned int lane = threadIdx.x % BRAIN_I8G_LANES;
    const bool live = col < n;
    const unsigned int granules = kg >> 2;
    const unsigned int iters = (granules + BRAIN_I8G_LANES - 1) / BRAIN_I8G_LANES;

    const unsigned int* wrow = wq + (unsigned long long)(live ? col : 0u) * kg;
    const float* srow = sw + (unsigned long long)(live ? col : 0u) * (kg >> 3);

    float acc[MR][4];
#pragma unroll
    for (int r = 0; r < MR; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) { acc[r][c] = 0.0f; }
    }

    int buf = 0;
    for (unsigned int i0 = 0; i0 < iters; i0 += UNROLL, buf ^= 1) {
        BrainStage<0, UNROLL> st;
        brain_load_stage<0, UNROLL, VEC>(st, i0, lane, live, granules, wrow, srow, nullptr);
        // Rows of x past `mrows` are not loaded: their outputs are discarded.
#pragma unroll
        for (int e0 = 0; e0 < MR * VECS; e0 += BRAIN_I8G_THREADS) {
            const int e = e0 + threadIdx.x;
            const int r = e / VECS;
            const int v = e - r * VECS;
            const unsigned int gv = i0 * BRAIN_I8G_LANES + v;
            if (static_cast<unsigned int>(r) < mrows && gv < granules) {
                xs[buf][r][v] = brain_load_x<VEC>(xq + (unsigned long long)(m0 + r) * kg + 4ull * gv);
            }
        }
        __syncthreads();
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const unsigned int g = (i0 + u) * BRAIN_I8G_LANES + lane;
            if (live && g < granules) {
#pragma unroll
                for (int r = 0; r < MR; ++r) {
                    const uint4 xv = xs[buf][r][u * BRAIN_I8G_LANES + lane];
                    acc[r][0] = brain_term(acc[r][0], xv.x, st.w[u].x, st.s[u]);
                    acc[r][1] = brain_term(acc[r][1], xv.y, st.w[u].y, st.s[u]);
                    acc[r][2] = brain_term(acc[r][2], xv.z, st.w[u].z, st.s[u]);
                    acc[r][3] = brain_term(acc[r][3], xv.w, st.w[u].w, st.s[u]);
                }
            }
        }
    }

    // Fold the 64 virtual lanes in ascending order, as `brain_gemv_tile` does.
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

typedef uint4 BrainXTile[2][BRAIN_I8G_ROWS][BRAIN_I8G_UNROLL_WIDE * BRAIN_I8G_LANES];

template <int MR, bool VEC>
__device__ __forceinline__ void brain_gemv_launch(BrainXTile xs, const unsigned int* xq, const unsigned int* wq,
                                                  const float* sx, const float* sw, float* out,
                                                  unsigned int mrows, unsigned int m0,
                                                  unsigned int kg, unsigned int n, unsigned int col) {
    if (MR <= BRAIN_I8G_STAGED_ROWS) {
        brain_gemv_tile<MR, (MR == 1 ? BRAIN_I8G_UNROLL_NARROW : BRAIN_I8G_UNROLL_NARROW2), VEC>(xq, wq, sx, sw, out, mrows, m0, kg, n, col);
    } else {
        brain_gemv_tile_shared_x<MR, BRAIN_I8G_UNROLL_WIDE, VEC>(xs, xq, wq, sx, sw, out, mrows, m0, kg, n, col);
    }
}

template <bool VEC>
__device__ __forceinline__ void brain_gemv_select(BrainXTile xs, const unsigned int* xq, const unsigned int* wq,
                                                  const float* sx, const float* sw, float* out,
                                                  unsigned int mrows, unsigned int m0,
                                                  unsigned int kg, unsigned int n, unsigned int col) {
    if (mrows <= 1u)      { brain_gemv_launch<1, VEC>(xs, xq, wq, sx, sw, out, mrows, m0, kg, n, col); }
    else if (mrows <= 2u) { brain_gemv_launch<2, VEC>(xs, xq, wq, sx, sw, out, mrows, m0, kg, n, col); }
    else if (mrows <= 4u) { brain_gemv_launch<4, VEC>(xs, xq, wq, sx, sw, out, mrows, m0, kg, n, col); }
    else                  { brain_gemv_launch<8, VEC>(xs, xq, wq, sx, sw, out, mrows, m0, kg, n, col); }
}

#ifdef BRAIN_I8G_USE_MMA
// ---------------------------------------------------------------------------
// The tensor-core block (compute capability 8.0 and up).
//
// Why. From five rows of x the dp4a block no longer streams the weights at
// the memory system's rate: each weight byte meets every row of x on the
// arithmetic pipes, and the rows of x are read back from shared memory at
// as many bytes as the weights themselves (knowledge 335). A tensor core does
// a whole 32-element group of 16 weight rows against eight rows of x in one
// instruction, with the activation held in registers.
//
// How. One block covers BRAIN_I8G_COLS (16) weight rows and up to
// BRAIN_I8G_ROWS (8) rows of x. Its warps split K between them (each takes one
// contiguous stretch, so its loads of a weight row are adjacent) and every
// warp runs `mma.m16n8k32` (s8 x s8 -> s32) per scale group with the 16
// weight rows as the rows of A and the rows of x as the eight columns of B. A
// group is exactly one MMA, so the group scale is folded per MMA: the
// accumulator is seeded with 1.5 * 2^23 so the integer sum comes back as a
// float bit pattern (exact for the group's |sum| <= 32 * 128 * 128), one FADD
// removes the bias and one FFMA applies the scale to the running f32 total -
// the construction of `matmul_i8_mma.cu`.
//
// The MMA fixes which k element sits in which register only up to a
// permutation shared by A and B, so a thread takes the eight bytes
// [8 * t4, 8 * t4 + 8) of a group from each of its two weight rows (one 8-byte
// load each) and the same eight bytes of its x row. Weights stream through
// with no L1 allocation; x (a few KiB per row) is read through L1.
//
// The warps' totals are added in warp order through shared memory, so the
// result does not depend on scheduling. It is not bit-identical to the dp4a
// block: the group scale is applied once per 32 products, as in the prefill
// GEMM, with a fused multiply-add.
#define BRAIN_I8G_MMA_WARPS (BRAIN_I8G_THREADS / 32)
#ifndef BRAIN_I8G_MMA_UNROLL
#define BRAIN_I8G_MMA_UNROLL 4
#endif

__device__ __forceinline__ void brain_mma_s8_m16n8k32(unsigned int (&d)[4], unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3, unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0x4B400000u));
}

__device__ __forceinline__ uint2 brain_ld_stream2(const uint2* p) {
    uint2 v;
    asm("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

template <bool VEC>
__device__ __forceinline__ uint2 brain_mma_ld_w(const unsigned int* p) {
    if (VEC) { return brain_ld_stream2(reinterpret_cast<const uint2*>(p)); }
    return make_uint2(__ldg(p), __ldg(p + 1));
}

template <bool VEC>
__device__ __forceinline__ uint2 brain_mma_ld_x(const unsigned int* p) {
    if (VEC) { return *reinterpret_cast<const uint2*>(p); }
    return make_uint2(p[0], p[1]);
}

template <bool VEC>
__device__ __forceinline__ void brain_gemv_block_mma(const unsigned int* xq, const unsigned int* wq,
                                                     const float* sx, const float* sw, float* out,
                                                     unsigned int m0, unsigned int mrows,
                                                     unsigned int kg, unsigned int n, unsigned int n0) {
    constexpr int U = BRAIN_I8G_MMA_UNROLL;
    __shared__ float red[BRAIN_I8G_MMA_WARPS][32][4];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int g = lane >> 2;    // MMA groupID: weight rows g and g + 8 of A, the row of x of B
    const unsigned int t4 = lane & 3u;
    const unsigned int groups = kg >> 3;
    // Rows past n read the last row; their outputs are dropped.
    const unsigned int rlo = min(n0 + g, n - 1u);
    const unsigned int rhi = min(n0 + g + 8u, n - 1u);
    const unsigned int xrow = m0 + (g < mrows ? g : 0u);

    const unsigned int* wlo = wq + (unsigned long long)rlo * kg + 2u * t4;
    const unsigned int* whi = wq + (unsigned long long)rhi * kg + 2u * t4;
    const float* slo = sw + (unsigned long long)rlo * groups;
    const float* shi = sw + (unsigned long long)rhi * groups;
    const unsigned int* xp = xq + (unsigned long long)xrow * kg + 2u * t4;

    float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
    // Each warp takes one contiguous stretch of K, so its loads of a weight row are adjacent.
    const unsigned int chunk = (groups + BRAIN_I8G_MMA_WARPS - 1u) / BRAIN_I8G_MMA_WARPS;
    const unsigned int first = min(warp * chunk, groups);
    const unsigned int per_warp = min(chunk, groups - first);
    for (unsigned int i0 = 0; i0 < per_warp; i0 += U) {
        uint2 wl[U], wh[U], x[U];
        float sl[U], sh[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const unsigned int gi = first + i0 + u;
            if (i0 + u < per_warp) {
                wl[u] = brain_mma_ld_w<VEC>(wlo + 8ull * gi);
                wh[u] = brain_mma_ld_w<VEC>(whi + 8ull * gi);
                x[u] = brain_mma_ld_x<VEC>(xp + 8ull * gi);
                sl[u] = slo[gi];
                sh[u] = shi[gi];
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (i0 + u < per_warp) {
                unsigned int d[4];
                brain_mma_s8_m16n8k32(d, wl[u].x, wh[u].x, wl[u].y, wh[u].y, x[u].x, x[u].y);
                f0 = fmaf(__uint_as_float(d[0]) - 12582912.0f, sl[u], f0);  // weight row g, row of x 2*t4
                f1 = fmaf(__uint_as_float(d[1]) - 12582912.0f, sl[u], f1);  // weight row g, row of x 2*t4 + 1
                f2 = fmaf(__uint_as_float(d[2]) - 12582912.0f, sh[u], f2);  // weight row g + 8
                f3 = fmaf(__uint_as_float(d[3]) - 12582912.0f, sh[u], f3);
            }
        }
    }
    red[warp][lane][0] = f0;
    red[warp][lane][1] = f1;
    red[warp][lane][2] = f2;
    red[warp][lane][3] = f3;
    __syncthreads();
    if (threadIdx.x < 128u) {
        const unsigned int l = threadIdx.x >> 2, j = threadIdx.x & 3u;
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < BRAIN_I8G_MMA_WARPS; ++w) { total += red[w][l][j]; }
        const unsigned int orow = n0 + (l >> 2) + 8u * (j >> 1);
        const unsigned int ocol = 2u * (l & 3u) + (j & 1u);
        if (orow < n && ocol < mrows) { out[(unsigned long long)(m0 + ocol) * n + orow] = total * sx[m0 + ocol]; }
    }
}
#endif

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
#ifdef BRAIN_I8G_USE_MMA
    if (m >= BRAIN_I8G_MMA_MIN_ROWS) {
        // Eight-byte loads when x_q and w_q are 8-byte aligned, word by word otherwise.
        const bool aligned8 = ((reinterpret_cast<unsigned long long>(xq) |
                                reinterpret_cast<unsigned long long>(wq)) & 7ull) == 0ull;
        if (aligned8) {
            brain_gemv_block_mma<true>(xq, wq, sx, sw, out, m0, mrows, kg, n, tile_n * BRAIN_I8G_COLS);
        } else {
            brain_gemv_block_mma<false>(xq, wq, sx, sw, out, m0, mrows, kg, n, tile_n * BRAIN_I8G_COLS);
        }
        return;
    }
#endif
    const unsigned int col = tile_n * BRAIN_I8G_COLS + threadIdx.x / BRAIN_I8G_LANES;

    // One staging tile for whichever row-count instance the block runs.
    __shared__ BrainXTile xs;
    const bool aligned = ((reinterpret_cast<unsigned long long>(xq) |
                           reinterpret_cast<unsigned long long>(wq)) & 15ull) == 0ull;
    if (aligned) {
        brain_gemv_select<true>(xs, xq, wq, sx, sw, out, mrows, m0, kg, n, col);
    } else {
        brain_gemv_select<false>(xs, xq, wq, sx, sw, out, mrows, m0, kg, n, col);
    }
}

// Programmatic dependent launch (compute capability 9.0): wait until the kernel before this one in the stream has
// finished and its writes are visible, then let the kernel after this one become resident at once. A launch without
// the programmatic attribute has nothing to wait for and no early dependents, so both are no-ops there.
#ifndef BRAIN_PDL_ENTRY
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define BRAIN_PDL_ENTRY() do { asm volatile("griddepcontrol.wait;\n" ::: "memory"); asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); } while (0)
#else
#define BRAIN_PDL_ENTRY() do {} while (0)
#endif
#endif

extern "C" __global__ void __launch_bounds__(BRAIN_I8G_THREADS, BRAIN_I8G_MIN_BLOCKS)
brain_matmul_i8_gemv(const unsigned int* params, const unsigned int* xq, const unsigned int* wq,
                     const float* sx, const float* sw, float* out) {
    BRAIN_PDL_ENTRY();
    const unsigned int m = params[0];
    const unsigned int kg = params[1];
    const unsigned int n = params[2];

    // Flat 1-D block count the host may wrap into a second grid dimension,
    // reconstructed the way every kernel in this tree does.
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    // Row tiles of one weight tile are adjacent in launch order, so they
    // run together and the weights they share are read from memory once.
    const unsigned int tiles_m = (m + BRAIN_I8G_ROWS - 1u) / BRAIN_I8G_ROWS;
    if (tiles_m == 0u || n == 0u) { return; }
    const unsigned int tile_n = blk / tiles_m;
    const unsigned int tile_m = blk - tile_n * tiles_m;
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
// Bit identity between the launches. A block runs `brain_gemv_block` - the very
// function the single-matrix kernel runs - on the block's own matrix, so every
// output element is the one a separate launch would have produced, bit for bit. The
// blocks of set 0 come first, then set 1, and so on; within a set the mapping
// is the single kernel's (`tile_m` major, `tile_n` minor).

extern "C" __global__ void __launch_bounds__(BRAIN_I8G_THREADS, BRAIN_I8G_MIN_BLOCKS)
brain_matmul_i8_gemv_multi(const unsigned int* params, const unsigned int* xq, const float* sx,
                           const unsigned int* wq0, const float* sw0, float* out0,
                           const unsigned int* wq1, const float* sw1, float* out1,
                           const unsigned int* wq2, const float* sw2, float* out2,
                           const unsigned int* wq3, const float* sw3, float* out3) {
    BRAIN_PDL_ENTRY();
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

    const unsigned int tile_n = local / tiles_m;
    const unsigned int tile_m = local - tile_n * tiles_m;
    brain_gemv_block(xq, wq, sx, sw, out, m, kg, n, tile_m, tile_n);
}
