// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for memory-bandwidth-bound LLM
// decode on GPUs for its clients. If your team needs expertise in native
// CUDA kernels that stream fp32 weights at the speed of the memory system
// then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Skinny-M fp32 matmul, out = x @ W^T - the native CUDA form of
// `matmul_gemv_reg.wgsl` (and so of `matmul_gemv.wgsl`), with the identical
// buffer layout, the identical uniform and the identical result bits.
//
//   params : u32 [m, k, n]
//   x      : [M, K] f32     row-major
//   w      : [N, K] f32     row-major
//   out    : [M, N] f32     out[m, n] = sum_k x[m, k] * w[n, k]
//
// Why this kernel exists
// ----------------------
// Decode streams every weight once per step, so the figure of merit is the
// fraction of DRAM bandwidth a weight row is read at. The WGSL kernel gives a
// 64-lane work-group one weight row and reads one word per lane per step, so a
// lane has a single load in flight at a time; and with several rows of x every
// lane reads one word of EACH row of x per word of weight, so at sixteen rows
// the activation reads are sixteen times the weights' own traffic and the
// kernel is bound by the cache, not by DRAM. Here:
//  * a lane reads 16-byte vectors and issues every weight load of a stage
//    before it consumes any, so a weight row has many loads in flight;
//  * a block serves BRAIN_F32G_COLS weight rows, and from two rows of x the
//    block loads each stage's slice of x into shared memory ONCE for all of
//    them, so the activation traffic from the cache is divided by the block's
//    weight rows. One row of x has an entry point of its own that reads x
//    straight from the read-only cache and declares no shared tile, so the
//    tile cannot cap how many of its blocks an SM holds.
//
// How it stays bit-identical to the WGSL kernel
// ---------------------------------------------
// The WGSL kernel gives lane t of 64 the elements k = t, t+64, t+128, ..., keeps
// one f32 accumulator per lane that adds `x * w` in ascending k (product and
// sum each rounded: the backend compiles with `--fmad=false`), then folds the
// 64 partials in ascending lane order starting from 0.0. Here 16 threads serve
// one weight row and thread j owns virtual lanes 4j..4j+3: in iteration i the
// elements 64i+4j .. 64i+4j+3 are one 16-byte vector, so the load IS the four
// virtual lanes' elements. The fold walks the 16 threads in ascending order
// with warp shuffles, four accumulators each - ascending virtual lane. Every
// product and sum is an explicit round-to-nearest operation, so no
// contraction can change a bit.
//
// Rows of x beyond `m` in a partial tile are pointed at the tile's first row
// and discarded, exactly as the WGSL kernel points rows past `p.m` at row 0:
// no read ever leaves the binding.
//
// Alignment: the vector path needs x and w 16-byte aligned and K a multiple of
// four (so every row starts aligned). Otherwise the scalar path issues the
// same loads element by element, each bounds-checked, and computes the same
// bits.
//
// No `__restrict__` anywhere, deliberately: brain's device buffers alias by
// design (a sliced step binds ranges of one allocation).

// Weight rows one block covers, threads per weight row, and the virtual lanes
// of the reference each thread owns.
#define BRAIN_F32G_COLS 16
#define BRAIN_F32G_LANES 16
#define BRAIN_F32G_VLANES 4
#define BRAIN_F32G_THREADS (BRAIN_F32G_COLS * BRAIN_F32G_LANES)
// Elements of K one iteration of every lane of a row covers: the reference's lane count.
#define BRAIN_F32G_STRIDE (BRAIN_F32G_LANES * BRAIN_F32G_VLANES)
// Rows of x one block of the batched entry covers; a larger m is split over blocks by the grid.
#define BRAIN_F32G_ROWS 4
// Iterations of a row's loads issued before any is consumed.
#define BRAIN_F32G_UNROLL 4
// Up to this many rows of x are read straight from the read-only cache; more go through shared memory.
#define BRAIN_F32G_DIRECT_ROWS 1

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

// Weight vector load that does not allocate in L1 where the part has the qualifier (the weights stream through once
// and must not evict the activations every weight row re-reads), plain read-only load elsewhere.
__device__ __forceinline__ float4 brain_f32g_ld_stream(const float* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    float4 v;
    asm("ld.global.nc.L1::no_allocate.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
#else
    return __ldg(reinterpret_cast<const float4*>(p));
#endif
}

// The four elements [k, k + 4) of a row of `len` elements: one vector when `VEC` (the row is aligned and `len` a
// multiple of four, so the vector is whole whenever `k < len`), element by element with each bounds-checked
// otherwise. Elements past the row are 0.0 and never consumed.
template <bool VEC, bool STREAM>
__device__ __forceinline__ float4 brain_f32g_load(const float* row, unsigned int k, unsigned int len) {
    if (VEC) {
        if (STREAM) { return brain_f32g_ld_stream(row + k); }
        return __ldg(reinterpret_cast<const float4*>(row + k));
    }
    return make_float4(k < len ? __ldg(row + k) : 0.0f, k + 1u < len ? __ldg(row + k + 1u) : 0.0f,
                       k + 2u < len ? __ldg(row + k + 2u) : 0.0f, k + 3u < len ? __ldg(row + k + 3u) : 0.0f);
}

// The terms of one vector into virtual lanes 4j .. 4j+3 of one row, each product and sum rounded as the reference
// rounds them. `valid` elements of the vector are inside the row; the rest are skipped, as the reference never visits
// them.
__device__ __forceinline__ void brain_f32g_terms(float (&acc)[BRAIN_F32G_VLANES], const float4& x, const float4& w, unsigned int valid) {
    if (valid >= 4u) {
        acc[0] = __fadd_rn(acc[0], __fmul_rn(x.x, w.x));
        acc[1] = __fadd_rn(acc[1], __fmul_rn(x.y, w.y));
        acc[2] = __fadd_rn(acc[2], __fmul_rn(x.z, w.z));
        acc[3] = __fadd_rn(acc[3], __fmul_rn(x.w, w.w));
        return;
    }
    if (valid > 0u) { acc[0] = __fadd_rn(acc[0], __fmul_rn(x.x, w.x)); }
    if (valid > 1u) { acc[1] = __fadd_rn(acc[1], __fmul_rn(x.y, w.y)); }
    if (valid > 2u) { acc[2] = __fadd_rn(acc[2], __fmul_rn(x.z, w.z)); }
}

// Fold row r's 64 virtual lanes in ascending order from 0.0: thread j's four in turn, for j ascending. Every lane
// runs the shuffles; only the row's first thread of a live row writes.
template <int MR>
__device__ __forceinline__ void brain_f32g_fold(const float (&acc)[MR][BRAIN_F32G_VLANES], float* out, unsigned int lane, bool live,
                                               unsigned int mrows, unsigned int m0, unsigned int n, unsigned int col) {
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        float total = 0.0f;
#pragma unroll
        for (int j = 0; j < BRAIN_F32G_LANES; ++j) {
#pragma unroll
            for (int c = 0; c < BRAIN_F32G_VLANES; ++c) {
                total = __fadd_rn(total, __shfl_sync(0xffffffffu, acc[r][c], j, BRAIN_F32G_LANES));
            }
        }
        if (lane == 0 && live && static_cast<unsigned int>(r) < mrows) {
            out[(unsigned long long)(m0 + r) * n + col] = total;
        }
    }
}

// The shared tile of x: per buffer, BRAIN_F32G_ROWS rows of one stage's BRAIN_F32G_UNROLL * BRAIN_F32G_STRIDE elements,
// as vectors.
#define BRAIN_F32G_XVECS (BRAIN_F32G_UNROLL * BRAIN_F32G_LANES)
typedef float4 BrainF32gXTile[2][BRAIN_F32G_ROWS][BRAIN_F32G_XVECS];

// `MR` rows of x against one weight row per BRAIN_F32G_LANES threads. Rows of x come from the read-only cache
// (`SHARED` false) or from the block's shared tile, filled once per stage for every weight row of the block
// (`SHARED` true, double-buffered: one barrier per stage orders the fill against the reads of the same stage and the
// reads of the previous stage against this stage's fill). Every thread of a block runs every stage, live or not: the
// barrier needs them all. The arithmetic and the fold do not depend on where x comes from.
template <int MR, bool VEC, bool SHARED>
__device__ __forceinline__ void brain_f32g_tile(BrainF32gXTile xs, const float* x, const float* w, float* out,
                                               unsigned int mrows, unsigned int m0, unsigned int k, unsigned int n, unsigned int col) {
    const unsigned int lane = threadIdx.x % BRAIN_F32G_LANES;
    const bool live = col < n;
    const unsigned int iters = (k + BRAIN_F32G_STRIDE - 1u) / BRAIN_F32G_STRIDE;
    const float* wrow = w + (unsigned long long)(live ? col : 0u) * k;
    const float* xrow[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        const unsigned int row = (static_cast<unsigned int>(r) < mrows) ? m0 + r : m0;
        xrow[r] = x + (unsigned long long)row * k;
    }

    float acc[MR][BRAIN_F32G_VLANES];
#pragma unroll
    for (int r = 0; r < MR; ++r) {
#pragma unroll
        for (int c = 0; c < BRAIN_F32G_VLANES; ++c) { acc[r][c] = 0.0f; }
    }

    int buf = 0;
    for (unsigned int i0 = 0; i0 < iters; i0 += BRAIN_F32G_UNROLL, buf ^= 1) {
        // Every weight load of the stage before any is consumed.
        float4 wv[BRAIN_F32G_UNROLL];
#pragma unroll
        for (int u = 0; u < BRAIN_F32G_UNROLL; ++u) {
            const unsigned int kk = (i0 + u) * BRAIN_F32G_STRIDE + lane * BRAIN_F32G_VLANES;
            wv[u] = (live && kk < k) ? brain_f32g_load<VEC, true>(wrow, kk, k) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        if (SHARED) {
            // The stage's slice of x, once for the block. Rows past `mrows` are not loaded: their outputs are
            // discarded.
#pragma unroll
            for (int e0 = 0; e0 < MR * BRAIN_F32G_XVECS; e0 += BRAIN_F32G_THREADS) {
                const int e = e0 + static_cast<int>(threadIdx.x);
                const int r = e / BRAIN_F32G_XVECS;
                const int v = e - r * BRAIN_F32G_XVECS;
                const unsigned int kk = i0 * BRAIN_F32G_STRIDE + static_cast<unsigned int>(v) * BRAIN_F32G_VLANES;
                if (r < MR && static_cast<unsigned int>(r) < mrows && kk < k) {
                    xs[buf][r][v] = brain_f32g_load<VEC, false>(x + (unsigned long long)(m0 + r) * k, kk, k);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int u = 0; u < BRAIN_F32G_UNROLL; ++u) {
            const unsigned int kk = (i0 + u) * BRAIN_F32G_STRIDE + lane * BRAIN_F32G_VLANES;
            if (live && kk < k) {
                const unsigned int valid = min(k - kk, 4u);
#pragma unroll
                for (int r = 0; r < MR; ++r) {
                    const float4 xv = SHARED ? xs[buf][r][u * BRAIN_F32G_LANES + lane] : brain_f32g_load<VEC, false>(xrow[r], kk, k);
                    brain_f32g_terms(acc[r], xv, wv[u], valid);
                }
            }
        }
    }
    brain_f32g_fold<MR>(acc, out, lane, live, mrows, m0, n, col);
}

template <int MR, bool VEC>
__device__ __forceinline__ void brain_f32g_launch(BrainF32gXTile xs, const float* x, const float* w, float* out,
                                                 unsigned int mrows, unsigned int m0, unsigned int k, unsigned int n, unsigned int col) {
    brain_f32g_tile<MR, VEC, (MR > BRAIN_F32G_DIRECT_ROWS)>(xs, x, w, out, mrows, m0, k, n, col);
}

template <bool VEC>
__device__ __forceinline__ void brain_f32g_select(BrainF32gXTile xs, const float* x, const float* w, float* out,
                                                 unsigned int mrows, unsigned int m0, unsigned int k, unsigned int n, unsigned int col) {
    if (mrows <= 2u)      { brain_f32g_launch<2, VEC>(xs, x, w, out, mrows, m0, k, n, col); }
    else                  { brain_f32g_launch<BRAIN_F32G_ROWS, VEC>(xs, x, w, out, mrows, m0, k, n, col); }
}

// Whether x and w can be read as aligned 16-byte vectors: K a multiple of four (every row starts aligned) and both
// bindings aligned.
__device__ __forceinline__ bool brain_f32g_aligned(const float* x, const float* w, unsigned int k) {
    return (k % 4u) == 0u && ((reinterpret_cast<unsigned long long>(x) | reinterpret_cast<unsigned long long>(w)) & 15ull) == 0ull;
}

// One row of x (`m == 1`): x straight from the read-only cache, no shared tile, so nothing but registers limits how
// many blocks an SM holds.
extern "C" __global__ void __launch_bounds__(BRAIN_F32G_THREADS)
brain_matmul_f32_gemv(const unsigned int* params, const float* x, const float* w, float* out) {
    BRAIN_PDL_ENTRY();
    const unsigned int k = params[1];
    const unsigned int n = params[2];
    if (params[0] == 0u || n == 0u || k == 0u) { return; }
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int col = blk * BRAIN_F32G_COLS + threadIdx.x / BRAIN_F32G_LANES;
    if (brain_f32g_aligned(x, w, k)) {
        brain_f32g_tile<1, true, false>(nullptr, x, w, out, 1u, 0u, k, n, col);
    } else {
        brain_f32g_tile<1, false, false>(nullptr, x, w, out, 1u, 0u, k, n, col);
    }
}

// Two rows of x and more: BRAIN_F32G_ROWS rows of x per block, the rows of x shared through the block's tile, and
// row tiles of one weight tile are adjacent in launch order, so they run together and the weights they
// share are read from memory once.
extern "C" __global__ void __launch_bounds__(BRAIN_F32G_THREADS)
brain_matmul_f32_gemv_rows(const unsigned int* params, const float* x, const float* w, float* out) {
    BRAIN_PDL_ENTRY();
    const unsigned int m = params[0];
    const unsigned int k = params[1];
    const unsigned int n = params[2];

    // Flat 1-D block count the host may wrap into a second grid dimension, reconstructed the way every kernel in this
    // tree does.
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int tiles_m = (m + BRAIN_F32G_ROWS - 1u) / BRAIN_F32G_ROWS;
    if (tiles_m == 0u || n == 0u || k == 0u) { return; }
    const unsigned int tile_n = blk / tiles_m;
    const unsigned int tile_m = blk - tile_n * tiles_m;
    const unsigned int m0 = tile_m * BRAIN_F32G_ROWS;
    if (m0 >= m) { return; }  // block-uniform
    const unsigned int mrows = min(m - m0, (unsigned int)BRAIN_F32G_ROWS);
    const unsigned int col = tile_n * BRAIN_F32G_COLS + threadIdx.x / BRAIN_F32G_LANES;

    __shared__ BrainF32gXTile xs;
    if (brain_f32g_aligned(x, w, k)) {
        brain_f32g_select<true>(xs, x, w, out, mrows, m0, k, n, col);
    } else {
        brain_f32g_select<false>(xs, x, w, out, mrows, m0, k, n, col);
    }
}
