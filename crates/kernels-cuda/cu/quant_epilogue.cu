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
//   sx = max(max|y|, 1e-8) / 127 per 32 elements   (the activation group scales)
//   xq = pack4(clamp(rint(y / sx), -127, 127))     (four int8 per u32, each by its group's scale)
//
//   mode 0 : y = a                    (a plain activation that only needs quantising)
//   mode 1 : y = silu(a) * b          (the SwiGLU product feeding `down_proj`)
//   mode 2 : y = a * sigmoid(b)       (attention output times its sigmoid gate)
//
// Each replaces a chain of kernels a decode token runs dozens of times
// (`silu_mul` / `sigmoid` + `mul`, then `max_abs_rows`, then `quant_pack`) and
// produces what that chain produces to the last bit.
//
//   params : u32 [k, rows, mode, flags]   flags bit 0: produce `y` only - no int8 packing (`xq`, `sx` are not written);
//                                         a consumer that reads bf16 or E4M3 rows has no use for them; bit 1 (with bit 0):
//                                         `xq` receives the activation rounded to bf16 instead ([rows, k/2] u32, the rows
//                                         `cast_f32_bf16` would make), so a bf16-reading product needs no cast launch
//   a, b   : [rows, k] f32   (b is not read in mode 0)
//   y      : [rows, k] f32   the activation itself (an fp32 linear reads it)
//   xq     : [rows, k/4] u32 packed int8 activation
//   sx     : [rows, k/32] f32 per-group scale
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
// one slice) belongs to one block, and a 32-element scale group (one warp's elements of one slice) to one warp. Each
// group's max is that warp's reduction - exact in any order - so no block needs another's. One 1024-thread block per
// row took 5-6 us on the 17408-wide MLP product, issue-bound on its exponentials and divisions on one SM.
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

#else
__device__ __forceinline__ unsigned int brain_cluster_rank() { return 0u; }
#endif

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

// `v` rounded to bf16, round to nearest even, NaN to the canonical quiet NaN: `cast_f32_bf16`'s own rule.
__device__ __forceinline__ unsigned int brain_qe_bf16_bits(float v) {
    const unsigned int b = __float_as_uint(v);
    if ((b & 0x7fffffffu) > 0x7f800000u) { return 0x7fc0u; }
    return (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;
}

extern "C" __global__ void __launch_bounds__(BRAIN_QE_THREADS)
brain_quant_epilogue(const unsigned int* params, const float* a, const float* b, float* y,
                     unsigned int* xq, float* sx) {
    BRAIN_PDL_ENTRY();
    const unsigned int k = params[0];
    const unsigned int rows = params[1];
    const unsigned int mode = params[2];
    const bool pack = (params[3] & 1u) == 0u;
    const bool bf16_out = !pack && (params[3] & 2u) != 0u;

    // The grid is `rows * CLUSTER` blocks, a row's cluster consecutive, so every block of a cluster takes the same
    // early return below (none, or all of them).
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int row = blk / BRAIN_QE_CLUSTER;
    const unsigned int rank = brain_cluster_rank();
    if (row >= rows) { return; }  // cluster-uniform
    const unsigned int t = threadIdx.x;
    const unsigned long long base = (unsigned long long)row * k;

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

    // Phase 2: the producer, then each slice's group scale: element c = t + THREADS * i lies in group c / 32, the
    // same group as slice i of every lane of its warp, so the group's max is a warp reduction.
    const unsigned int groups = k >> 5;
    float rscale[BRAIN_QE_SLICES];
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
        if (!pack) { continue; }
        float mx = fabsf(r);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o)); }
        const float scale = __fdiv_rn(fmaxf(mx, 1e-8f), 127.0f);
        const unsigned int i = rank + BRAIN_QE_CLUSTER * j;
        const unsigned int c = t + BRAIN_QE_THREADS * i;
        if ((t & 31u) == 0u && i < BRAIN_QE_MAX_ELEMS && c < k) { sx[(unsigned long long)row * groups + (c >> 5)] = scale; }
        rscale[j] = __fdiv_rn(1.0f, scale);
    }

    // Phase 3: the stores - the product, the quantised bytes into shared
    // memory, then whole packed words of this block's slices.
#pragma unroll
    for (int j = 0; j < BRAIN_QE_SLICES; ++j) {
        const unsigned int i = rank + BRAIN_QE_CLUSTER * j;
        const unsigned int c = t + BRAIN_QE_THREADS * i;
        if (i < BRAIN_QE_MAX_ELEMS && c < k) {
            if (two) { y[base + c] = v[j]; }
            if (bf16_out) { reinterpret_cast<unsigned short*>(xq)[base + c] = static_cast<unsigned short>(brain_qe_bf16_bits(v[j])); }
            if (!pack) { continue; }
            const float q = fminf(fmaxf(rintf(__fmul_rn(v[j], rscale[j])), -127.0f), 127.0f);
            qbytes[c] = static_cast<signed char>(static_cast<int>(q));
        }
    }
    if (!pack) { return; }  // cluster-uniform
    __syncthreads();
    const unsigned int words = k >> 2;
    constexpr unsigned int SLICE_WORDS = BRAIN_QE_THREADS / 4;
    for (unsigned int j = 0; j < BRAIN_QE_SLICES; ++j) {
        const unsigned int i = rank + BRAIN_QE_CLUSTER * j;
        const unsigned int g = i * SLICE_WORDS + t;
        if (t < SLICE_WORDS && g < words) { xq[(unsigned long long)row * words + g] = qwords[g]; }
    }
}
