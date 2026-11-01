// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for prompt-processing throughput on FP8 language-model releases for its clients. If
// your team needs expertise in cutting the memory passes between the matrix products of an FP8 forward then you can procure
// our services by sending an email to info@swedishembedded.com.
//
// The activation front end of an FP8 linear that takes E4M3 rows, in a single launch:
//
//   sum = a + b                                    (optional residual add)
//   xn  = rmsnorm(sum) * w
//   s   = amax(|xn| over 128) / 448                (one scale per row per 128 elements)
//   q   = e4m3(xn / s)                             (round to nearest even, saturating)
//
// It replaces `add_rms_quant` followed by `quant_e4m3_128` (the pair wrote the fp32 `xn`, read it back and launched again) and is
// the SwiGLU kernel's counterpart for the norm: the bytes and scales are what `quant_e4m3_128` makes of the stored `xn`, bit for
// bit, and `xn` itself is stored only when something reads it as fp32.
//
//   params : u32 [d, rows, eps (f32 bits), flags]   d a multiple of 128, at most 5120. flags bit 0: add `b` first; bit 4: `xn`
//                                                   is not stored
//   a, b   : [rows, d] f32     b is not read (and sum is not written) without bit 0
//   w      : [d] f32           the norm gain
//   sum    : [rows, d] f32     a + b, the residual stream's next value
//   xn     : [rows, d] f32     the normalised rows (not written with bit 4)
//   xq     : [rows, d/4] u32   E4M3 bytes, four to a word
//   sx     : [rows, d/128] f32 the scales
//
// One block of 256 threads a row, a thread four consecutive elements of each 1024-element span: the loads, the stores and the E4M3
// words are 16- and 4-byte coalesced, a 128-element scale block is exactly one warp's span (amax by shuffles, no shared memory), and
// the sum of squares is a shuffle reduction and eight partials. That is not the 64-lane order `add_rms_quant` keeps for the decode
// token's bit-identity with the WGSL chain - the rows here are a mixed round's or a prefill's, whose product is a tensor-core GEMM -
// so `xn` differs from that chain's in the last bits of the sum of squares and nowhere else.
//
// The cluster form (`BRAIN_ARE_CLUSTER` 5, Hopper: the registry's `add_rms_quant_e4m3_cluster`) gives each of five blocks one of
// a row's five 1024-element spans, so a thread holds one float4 and no chain of twenty elements is on the critical path of a
// decode step's 16 rows. The sum of squares keeps the order of the one-block form bit for bit: the old thread t added the squares
// of its five float4s (one per span, in ascending order, element by element) into one accumulator, so the five blocks put their
// squares in shared memory, and after one cluster barrier thread t of every block reads thread t's five float4s from the five
// blocks and adds them in that order; then the warp's shuffle tree and the eight warps' partials in ascending order, as before.
// Each block computes the row's sum of squares itself (twenty adds), so no value crosses the cluster twice. The block stays small
// (256 threads, 4 KiB of shared memory) on purpose: a decode step's FP8 product is launched while this kernel runs and keeps
// 210 KiB of an SM's shared memory for its weight ring, so a norm block that is not small cannot be resident beside it
// (knowledge 420).
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_ARE_THREADS 256
#define BRAIN_ARE_SPAN (BRAIN_ARE_THREADS * 4)  // elements one pass of the block covers
#define BRAIN_ARE_PASSES 5                      // d <= 5120

// Blocks of one row: 1 (the default, any device) runs every pass in one block, 5 runs one pass in each block of a cluster.
#ifndef BRAIN_ARE_CLUSTER
#define BRAIN_ARE_CLUSTER 1
#endif
static_assert(BRAIN_ARE_PASSES % BRAIN_ARE_CLUSTER == 0, "a cluster takes a whole number of passes a block");
#define BRAIN_ARE_LOCAL (BRAIN_ARE_PASSES / BRAIN_ARE_CLUSTER)  // passes one block runs

#if BRAIN_ARE_CLUSTER > 1
__device__ __forceinline__ unsigned int brain_are_cluster_rank() {
    unsigned int r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
    return r;
}
// All blocks of the cluster are past this point, and what each wrote to its shared memory before it is visible to the others.
__device__ __forceinline__ void brain_are_cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
// This block is done reading the cluster's shared memory; `brain_are_cluster_wait` is the point after which it may exit. The
// release makes the peer loads before it complete first: with a relaxed arrive a peer could pass its wait and exit, and free its
// shared memory, while a load of it was still in flight (an unspecified launch failure, and only when a launch had more
// clusters than ran at once).
__device__ __forceinline__ void brain_are_cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void brain_are_cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
// The float4 at the same offset of block `rank`'s shared memory as `local` is of this block's.
__device__ __forceinline__ float4 brain_are_peer_float4(const float4* local, unsigned int rank) {
    const unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(local));
    unsigned int remote;
    float4 v;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote) : "r"(addr), "r"(rank));
    asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(remote) : "memory");
    return v;
}
#else
__device__ __forceinline__ unsigned int brain_are_cluster_rank() { return 0u; }
#endif

#ifndef BRAIN_PDL_ENTRY
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define BRAIN_PDL_ENTRY() do { asm volatile("griddepcontrol.wait;\n" ::: "memory"); asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); } while (0)
#else
#define BRAIN_PDL_ENTRY() do {} while (0)
#endif
#endif

extern "C" __global__ void __launch_bounds__(BRAIN_ARE_THREADS)
brain_add_rms_quant_e4m3(const unsigned int* params, const float* a, const float* b, const float* w, float* sum, float* xn, unsigned int* xq, float* sx) {
    BRAIN_PDL_ENTRY();
    const unsigned int d = params[0];
    const unsigned int rows = params[1];
    const float eps = __uint_as_float(params[2]);
    const bool add = (params[3] & 1u) != 0u;
    const bool store_xn = (params[3] & 16u) == 0u;

    // The grid is `rows * CLUSTER` blocks, a row's cluster consecutive, so every block of a cluster takes the same early return.
    const unsigned int row = (blockIdx.y * gridDim.x + blockIdx.x) / BRAIN_ARE_CLUSTER;
    const unsigned int rank = brain_are_cluster_rank();
    if (row >= rows) { return; }  // cluster-uniform
    const unsigned int tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned long long base = (unsigned long long)row * d;
    __shared__ float partial[BRAIN_ARE_THREADS / 32];
#if BRAIN_ARE_CLUSTER > 1
    __shared__ float4 squares[BRAIN_ARE_THREADS];  // the squares of this block's span: thread t's float4
#endif

    // Every load first, nothing between them (brain's buffers alias, so a store would order the loads behind it). Pass p of this
    // block is span k = rank + CLUSTER * p of the row.
    float4 v[BRAIN_ARE_LOCAL], g[BRAIN_ARE_LOCAL];
#pragma unroll
    for (int p = 0; p < BRAIN_ARE_LOCAL; ++p) {
        const unsigned int c = (rank + BRAIN_ARE_CLUSTER * p) * BRAIN_ARE_SPAN + 4u * tid;
        const bool live = c < d;
        v[p] = live ? *reinterpret_cast<const float4*>(a + base + c) : make_float4(0.f, 0.f, 0.f, 0.f);
        g[p] = live ? *reinterpret_cast<const float4*>(w + c) : make_float4(0.f, 0.f, 0.f, 0.f);
        if (add && live) {
            const float4 u = *reinterpret_cast<const float4*>(b + base + c);
            v[p] = make_float4(__fadd_rn(v[p].x, u.x), __fadd_rn(v[p].y, u.y), __fadd_rn(v[p].z, u.z), __fadd_rn(v[p].w, u.w));
        }
    }

    float ss = 0.0f;
#if BRAIN_ARE_CLUSTER > 1
    squares[tid] = make_float4(__fmul_rn(v[0].x, v[0].x), __fmul_rn(v[0].y, v[0].y), __fmul_rn(v[0].z, v[0].z), __fmul_rn(v[0].w, v[0].w));
    brain_are_cluster_sync();
    float4 q[BRAIN_ARE_PASSES];
#pragma unroll
    for (int k = 0; k < BRAIN_ARE_PASSES; ++k) { q[k] = brain_are_peer_float4(&squares[tid], k); }
    brain_are_cluster_arrive();
#pragma unroll
    for (int k = 0; k < BRAIN_ARE_PASSES; ++k) {
        ss = __fadd_rn(ss, q[k].x);
        ss = __fadd_rn(ss, q[k].y);
        ss = __fadd_rn(ss, q[k].z);
        ss = __fadd_rn(ss, q[k].w);
    }
#else
#pragma unroll
    for (int k = 0; k < BRAIN_ARE_PASSES; ++k) {
        ss = __fadd_rn(ss, __fmul_rn(v[k].x, v[k].x));
        ss = __fadd_rn(ss, __fmul_rn(v[k].y, v[k].y));
        ss = __fadd_rn(ss, __fmul_rn(v[k].z, v[k].z));
        ss = __fadd_rn(ss, __fmul_rn(v[k].w, v[k].w));
    }
#endif
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o)); }
    if (lane == 0u) { partial[warp] = ss; }
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < BRAIN_ARE_THREADS / 32; ++i) { total = __fadd_rn(total, partial[i]); }
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(total, static_cast<float>(d)), eps)));

#pragma unroll
    for (int p = 0; p < BRAIN_ARE_LOCAL; ++p) {
        const unsigned int c = (rank + BRAIN_ARE_CLUSTER * p) * BRAIN_ARE_SPAN + 4u * tid;
        const bool live = c < d;
        if (add && live) { *reinterpret_cast<float4*>(sum + base + c) = v[p]; }
        const float4 y = make_float4(__fmul_rn(__fmul_rn(v[p].x, inv), g[p].x), __fmul_rn(__fmul_rn(v[p].y, inv), g[p].y), __fmul_rn(__fmul_rn(v[p].z, inv), g[p].z),
                                     __fmul_rn(__fmul_rn(v[p].w, inv), g[p].w));
        if (store_xn && live) { *reinterpret_cast<float4*>(xn + base + c) = y; }
        // The warp's 32 lanes x 4 elements are one 128-element scale block.
        float amax = fmaxf(fmaxf(fabsf(y.x), fabsf(y.y)), fmaxf(fabsf(y.z), fabsf(y.w)));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o)); }
        const float scale = amax / 448.0f;
        unsigned int packed = 0u;
        if (scale > 0.0f) {
            packed = brain_e4m3x4(y, scale);
        }
        if (live) {
            xq[(unsigned long long)row * (d >> 2) + (c >> 2)] = packed;
            if (lane == 0u) { sx[(unsigned long long)row * (d >> 7) + (c >> 7)] = scale; }
        }
    }
#if BRAIN_ARE_CLUSTER > 1
    brain_are_cluster_wait();  // no block exits while a peer may still be reading its squares
#endif
}
