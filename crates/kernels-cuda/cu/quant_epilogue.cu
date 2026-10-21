// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for latency-bound LLM decode on
// GPUs for its clients. If your team needs expertise in collapsing the chain of
// tiny kernels between two weight streams into one launch, then you can
// procure our services by sending an email to info@swedishembedded.com.
//
// The activation back end of a decode step's mixer or MLP, in a single launch:
// produce the activation that feeds the next int8 linear AND quantise it.
//
//   y  = f(a, b)                                   f picked by `mode`
//   sx = max(max|y|, 1e-8) / 127                   (per-row activation scale)
//   xq = pack4(clamp(rint(y / sx), -127, 127))     (four int8 per u32)
//
//   mode 0 : y = a                    (a plain activation that only needs quantising)
//   mode 1 : y = silu(a) * b          (the SwiGLU product feeding `down_proj`)
//   mode 2 : y = a * sigmoid(b)       (attention output times its sigmoid gate)
//
// Each replaces a chain of kernels a decode token runs dozens of times
// (`silu_mul` / `sigmoid` + `mul`, then `max_abs_rows`, then `quant_pack`) and
// produces what that chain produces to the last bit.
//
//   params : u32 [k, rows, mode, 0]
//   a, b   : [rows, k] f32   (b is not read in mode 0)
//   y      : [rows, k] f32   the activation itself (an fp32 linear reads it)
//   xq     : [rows, k/4] u32 packed int8 activation
//   sx     : [rows] f32      per-row scale
//
// The arithmetic is the reference kernels' own expressions: silu is
// `x / (1.0 + exp(-x))` and sigmoid `1.0 / (1.0 + exp(-x))` with `expf` (the
// WGSL translator's `exp`), products and sums are explicit round-to-nearest,
// the max is exact in any order, `/` and `rint` are correctly rounded (WGSL
// `round` is half-to-even, i.e. `rintf`). Nothing here is order-sensitive, so
// the result is the chain's, bit for bit, however the row is split over threads.
//
// A cluster of 1024-thread blocks per row (below); a thread keeps its elements in registers from
// the single read to the quantiser, and the four int8 of an output word, which
// belong to four different threads, are exchanged through shared memory so
// every store is a whole coalesced word. A row wider than
// BRAIN_QE_THREADS * BRAIN_QE_MAX_ELEMS is not served; the host keeps the WGSL
// chain for it.
//
// No `__restrict__` anywhere, deliberately: brain's device buffers alias by
// design (a sliced step binds ranges of one allocation).

#define BRAIN_QE_THREADS 1024
// Elements one thread keeps in registers: rows up to 1024 * this wide. 18 covers
// 18432, which holds Qwen3.8-27B's widest decode activation, the 17408-wide MLP
// hidden. A thousand threads rather than a few hundred because the work is
// issue- and latency-bound (an `expf` and two IEEE divisions an element): the
// 17408-wide product took 8.8 us on 256 threads and 5.7 us on 1024, the extra
// warps hiding the dependent chains, at the same instruction count.
#define BRAIN_QE_MAX_ELEMS 18

// The row is split over a CLUSTER of BRAIN_QE_CLUSTER blocks (one per SM): block r of a row owns the slices
// i = r, r + CLUSTER, ... of the element map c = t + THREADS * i, so a packed output word (four consecutive elements of
// one slice) belongs to one block. Each block reduces its own max, the cluster exchanges the per-block maxima through
// distributed shared memory, and every block quantises its slices against the row's max - the same numbers as one
// block per row, since the max is exact in any order. One 1024-thread block per row took 5-6 us on the 17408-wide MLP
// product, issue-bound on its exponentials and divisions on one SM.
// 1 (the default, any device) is one block per row; the registry's `quant_epilogue_cluster` compiles 4 (Hopper).
#ifndef BRAIN_QE_CLUSTER
#define BRAIN_QE_CLUSTER 1
#endif
#define BRAIN_QE_SLICES ((BRAIN_QE_MAX_ELEMS + BRAIN_QE_CLUSTER - 1) / BRAIN_QE_CLUSTER)

#if BRAIN_QE_CLUSTER > 1
__device__ __forceinline__ unsigned int brain_cluster_rank() {
    unsigned int r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
    return r;
}

__device__ __forceinline__ void brain_cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

// `addr` (this block's shared memory) as seen in block `rank` of the cluster, read as f32.
__device__ __forceinline__ float brain_cluster_load(const float* addr, unsigned int rank) {
    const unsigned int local = static_cast<unsigned int>(__cvta_generic_to_shared(addr));
    unsigned int remote;
    float v;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote) : "r"(local), "r"(rank));
    asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(v) : "r"(remote) : "memory");
    return v;
}
#else
__device__ __forceinline__ unsigned int brain_cluster_rank() { return 0u; }
__device__ __forceinline__ void brain_cluster_sync() { __syncthreads(); }
__device__ __forceinline__ float brain_cluster_load(const float* addr, unsigned int) { return *addr; }
#endif

extern "C" __global__ void __launch_bounds__(BRAIN_QE_THREADS)
brain_quant_epilogue(const unsigned int* params, const float* a, const float* b, float* y,
                     unsigned int* xq, float* sx) {
    const unsigned int k = params[0];
    const unsigned int rows = params[1];
    const unsigned int mode = params[2];

    // The grid is `rows * CLUSTER` blocks, a row's cluster consecutive, so every block of a cluster takes the same
    // early return below (none, or all of them).
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int row = blk / BRAIN_QE_CLUSTER;
    const unsigned int rank = brain_cluster_rank();
    if (row >= rows) { return; }  // cluster-uniform
    const unsigned int t = threadIdx.x;
    const unsigned long long base = (unsigned long long)row * k;

    __shared__ float warp_max[BRAIN_QE_THREADS / 32];
    __shared__ float block_max;
    __shared__ unsigned int qwords[BRAIN_QE_THREADS * BRAIN_QE_MAX_ELEMS / 4];
    signed char* qbytes = reinterpret_cast<signed char*>(qwords);

    // Phase 1: every load, nothing else - a load that has to wait for an
    // earlier store through a possibly aliasing pointer, or whose dependent
    // arithmetic sits right behind it, is a serialised round trip to L2 (the
    // lesson `add_rms_quant.cu` records). Slice j of this block is element slice
    // i = rank + CLUSTER * j of the row.
    float v[BRAIN_QE_SLICES];
    float w[BRAIN_QE_SLICES];
    const bool two = mode != 0u;
#pragma unroll
    for (int j = 0; j < BRAIN_QE_SLICES; ++j) {
        const unsigned int i = rank + BRAIN_QE_CLUSTER * j;
        const unsigned int c = t + BRAIN_QE_THREADS * i;
        const bool live = i < BRAIN_QE_MAX_ELEMS && c < k;
        v[j] = live ? a[base + c] : 0.0f;
        w[j] = (live && two) ? b[base + c] : 0.0f;
    }

    // Phase 2: the producer and the running max, in registers.
    float mx = 0.0f;
#pragma unroll
    for (int j = 0; j < BRAIN_QE_SLICES; ++j) {
        float r = v[j];
        if (mode == 1u) {
            // silu(a) * b:  a / (1 + exp(-a)) * b
            r = __fmul_rn(__fdiv_rn(r, __fadd_rn(1.0f, expf(-r))), w[j]);
        } else if (mode == 2u) {
            // a * sigmoid(b):  a * (1 / (1 + exp(-b)))
            r = __fmul_rn(r, __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-w[j]))));
        }
        v[j] = r;
        mx = fmaxf(mx, fabsf(r));
    }

    // Row max: exact in any order - the warps, then the block, then the cluster's blocks.
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o)); }
    if ((t & 31u) == 0u) { warp_max[t >> 5] = mx; }
    __syncthreads();
    if (t == 0) {
        float bm = warp_max[0];
#pragma unroll
        for (int i = 1; i < BRAIN_QE_THREADS / 32; ++i) { bm = fmaxf(bm, warp_max[i]); }
        block_max = bm;
    }
    brain_cluster_sync();  // every block's max is in place
    float rowmax = 0.0f;
#pragma unroll
    for (unsigned int r = 0; r < BRAIN_QE_CLUSTER; ++r) { rowmax = fmaxf(rowmax, brain_cluster_load(&block_max, r)); }
    brain_cluster_sync();  // no block leaves while another may still read its max
    const float scale = __fdiv_rn(fmaxf(rowmax, 1e-8f), 127.0f);
    if (t == 0 && rank == 0) { sx[row] = scale; }
    const float rscale = __fdiv_rn(1.0f, scale);

    // Phase 3: the stores - the product, the quantised bytes into shared
    // memory, then whole packed words of this block's slices.
#pragma unroll
    for (int j = 0; j < BRAIN_QE_SLICES; ++j) {
        const unsigned int i = rank + BRAIN_QE_CLUSTER * j;
        const unsigned int c = t + BRAIN_QE_THREADS * i;
        if (i < BRAIN_QE_MAX_ELEMS && c < k) {
            if (two) { y[base + c] = v[j]; }
            const float q = fminf(fmaxf(rintf(__fmul_rn(v[j], rscale)), -127.0f), 127.0f);
            qbytes[c] = static_cast<signed char>(static_cast<int>(q));
        }
    }
    __syncthreads();
    const unsigned int words = k >> 2;
    constexpr unsigned int SLICE_WORDS = BRAIN_QE_THREADS / 4;
    for (unsigned int j = 0; j < BRAIN_QE_SLICES; ++j) {
        const unsigned int i = rank + BRAIN_QE_CLUSTER * j;
        const unsigned int g = i * SLICE_WORDS + t;
        if (t < SLICE_WORDS && g < words) { xq[(unsigned long long)row * words + g] = qwords[g]; }
    }
}
