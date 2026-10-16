// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements QLoRA training stacks for its clients. If
// your team needs expertise in running a 4-bit frozen base on tensor cores
// without holding a second copy of it, you can procure our services by
// sending an email to info@swedishembedded.com.
//
// Decode a window of an NF4 weight into a bf16 scratch that a tensor-core GEMM
// then reads: the weight is stored eight 4-bit codes to a 32-bit word (code b
// at bits 4b, `model::lut4`'s layout) with one fp32 scale per 32 consecutive
// weights, and decodes to codebook[code] * scale, rounded to bf16 with
// round-to-nearest-even. The codebook is `kernels::template::NF4_CODEBOOK`;
// a device test pins the two together value by value.
//
// One thread per packed word: one 4-byte load of eight codes, one scale load
// shared by the four words of a group, one 16-byte store of eight bf16 values.
// The element count is a multiple of 8 and the output is 16-byte aligned (the
// caller guarantees both).
//
// params: [words] (packed words to decode).
// Bindings: uniform, packed codes, scales (f32), bf16 output.
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_NF4_THREADS 256

__device__ __forceinline__ float brain_nf4_value(unsigned code) {
    // The ascending NF4 codebook, as a select tree on the code bits (no
    // indexed constant array, which would put a local-memory table in every
    // thread).
    const float c0 = -1.0f, c1 = -0.6961928f, c2 = -0.52507305f, c3 = -0.3949175f;
    const float c4 = -0.28444138f, c5 = -0.18477343f, c6 = -0.091050036f, c7 = 0.0f;
    const float c8 = 0.0795803f, c9 = 0.1609302f, c10 = 0.2461123f, c11 = 0.33791524f;
    const float c12 = 0.44070983f, c13 = 0.562617f, c14 = 0.72295684f, c15 = 1.0f;
    const float q0 = (code & 1u) ? c1 : c0, q1 = (code & 1u) ? c3 : c2, q2 = (code & 1u) ? c5 : c4, q3 = (code & 1u) ? c7 : c6;
    const float q4 = (code & 1u) ? c9 : c8, q5 = (code & 1u) ? c11 : c10, q6 = (code & 1u) ? c13 : c12, q7 = (code & 1u) ? c15 : c14;
    const float r0 = (code & 2u) ? q1 : q0, r1 = (code & 2u) ? q3 : q2, r2 = (code & 2u) ? q5 : q4, r3 = (code & 2u) ? q7 : q6;
    const float s0 = (code & 4u) ? r1 : r0, s1 = (code & 4u) ? r3 : r2;
    return (code & 8u) ? s1 : s0;
}

__device__ __forceinline__ unsigned brain_nf4_bf16_bits(float v) {
    const unsigned b = __float_as_uint(v);
    return (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;
}

extern "C" __global__ void __launch_bounds__(BRAIN_NF4_THREADS) brain_nf4_dequant_bf16(const unsigned int* params, const unsigned int* packed, const float* scales, unsigned int* out) {
    const unsigned long long block = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned long long w = block * BRAIN_NF4_THREADS + threadIdx.x;
    if (w >= params[0]) { return; }
    const unsigned word = packed[w];
    const float scale = scales[w >> 2];
    uint4 r;
    unsigned* o = reinterpret_cast<unsigned*>(&r);
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const float lo = brain_nf4_value((word >> (8 * p)) & 15u) * scale;
        const float hi = brain_nf4_value((word >> (8 * p + 4)) & 15u) * scale;
        o[p] = brain_nf4_bf16_bits(lo) | (brain_nf4_bf16_bits(hi) << 16);
    }
    *reinterpret_cast<uint4*>(out + 4ull * w) = r;
}
