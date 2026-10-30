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
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_ARE_THREADS 256
#define BRAIN_ARE_SPAN (BRAIN_ARE_THREADS * 4)  // elements one pass of the block covers
#define BRAIN_ARE_PASSES 5                      // d <= 5120

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

    const unsigned int row = blockIdx.y * gridDim.x + blockIdx.x;
    if (row >= rows) { return; }  // block-uniform
    const unsigned int tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned long long base = (unsigned long long)row * d;
    __shared__ float partial[BRAIN_ARE_THREADS / 32];

    // Every load first, nothing between them (brain's buffers alias, so a store would order the loads behind it).
    float4 v[BRAIN_ARE_PASSES], g[BRAIN_ARE_PASSES];
#pragma unroll
    for (int k = 0; k < BRAIN_ARE_PASSES; ++k) {
        const unsigned int c = k * BRAIN_ARE_SPAN + 4u * tid;
        const bool live = c < d;
        v[k] = live ? *reinterpret_cast<const float4*>(a + base + c) : make_float4(0.f, 0.f, 0.f, 0.f);
        g[k] = live ? *reinterpret_cast<const float4*>(w + c) : make_float4(0.f, 0.f, 0.f, 0.f);
        if (add && live) {
            const float4 u = *reinterpret_cast<const float4*>(b + base + c);
            v[k] = make_float4(__fadd_rn(v[k].x, u.x), __fadd_rn(v[k].y, u.y), __fadd_rn(v[k].z, u.z), __fadd_rn(v[k].w, u.w));
        }
    }

    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < BRAIN_ARE_PASSES; ++k) {
        ss = __fadd_rn(ss, __fmul_rn(v[k].x, v[k].x));
        ss = __fadd_rn(ss, __fmul_rn(v[k].y, v[k].y));
        ss = __fadd_rn(ss, __fmul_rn(v[k].z, v[k].z));
        ss = __fadd_rn(ss, __fmul_rn(v[k].w, v[k].w));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o)); }
    if (lane == 0u) { partial[warp] = ss; }
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < BRAIN_ARE_THREADS / 32; ++i) { total = __fadd_rn(total, partial[i]); }
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(total, static_cast<float>(d)), eps)));

#pragma unroll
    for (int k = 0; k < BRAIN_ARE_PASSES; ++k) {
        const unsigned int c = k * BRAIN_ARE_SPAN + 4u * tid;
        const bool live = c < d;
        if (add && live) { *reinterpret_cast<float4*>(sum + base + c) = v[k]; }
        const float4 y = make_float4(__fmul_rn(__fmul_rn(v[k].x, inv), g[k].x), __fmul_rn(__fmul_rn(v[k].y, inv), g[k].y), __fmul_rn(__fmul_rn(v[k].z, inv), g[k].z),
                                     __fmul_rn(__fmul_rn(v[k].w, inv), g[k].w));
        if (store_xn && live) { *reinterpret_cast<float4*>(xn + base + c) = y; }
        // The warp's 32 lanes x 4 elements are one 128-element scale block.
        float amax = fmaxf(fmaxf(fabsf(y.x), fabsf(y.y)), fmaxf(fabsf(y.z), fabsf(y.w)));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o)); }
        const float scale = amax / 448.0f;
        unsigned int packed = 0u;
        if (scale > 0.0f) {
            unsigned short lo, hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(__fdiv_rn(y.y, scale)), "f"(__fdiv_rn(y.x, scale)));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(__fdiv_rn(y.w, scale)), "f"(__fdiv_rn(y.z, scale)));
            packed = (unsigned int)lo | ((unsigned int)hi << 16);
        }
        if (live) {
            xq[(unsigned long long)row * (d >> 2) + (c >> 2)] = packed;
            if (lane == 0u) { sx[(unsigned long long)row * (d >> 7) + (c >> 7)] = scale; }
        }
    }
}
