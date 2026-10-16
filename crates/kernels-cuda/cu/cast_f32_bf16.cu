// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements mixed-precision training stacks for its
// clients. If your team needs expertise in moving a model's GEMMs onto tensor
// cores without giving up fp32 master activations, you can procure our
// services by sending an email to info@swedishembedded.com.
//
// Narrow an fp32 activation to bf16 for a tensor-core GEMM that reads bf16
// operands: round-to-nearest-even on the top 16 bits, NaN kept a (quiet) NaN,
// two bf16 values to a 32-bit word with the lower index in the low half (the
// layout the bf16 weights are stored in). Four elements a thread with 16-byte
// loads and an 8-byte store; the element count is a multiple of four and both
// buffers are 16-byte aligned (the caller guarantees it, as the provider that
// issues this kernel only accepts such shapes).
//
// params: [count]. Bindings: uniform, src (f32), dst (packed bf16).
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_CAST_THREADS 256
#define BRAIN_CAST_PER_BLOCK (BRAIN_CAST_THREADS * 4)

__device__ __forceinline__ unsigned brain_cast_bf16_bits(float v) {
    const unsigned b = __float_as_uint(v);
    if ((b & 0x7fffffffu) > 0x7f800000u) { return 0x7fc0u; }
    return (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;
}

extern "C" __global__ void __launch_bounds__(BRAIN_CAST_THREADS) brain_cast_f32_bf16(const unsigned int* params, const float* src, unsigned int* dst) {
    const unsigned count = params[0];
    const unsigned long long block = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned long long i = block * BRAIN_CAST_PER_BLOCK + 4ull * threadIdx.x;
    if (i >= count) { return; }
    const float4 v = *reinterpret_cast<const float4*>(src + i);
    uint2 out;
    out.x = brain_cast_bf16_bits(v.x) | (brain_cast_bf16_bits(v.y) << 16);
    out.y = brain_cast_bf16_bits(v.z) | (brain_cast_bf16_bits(v.w) << 16);
    *reinterpret_cast<uint2*>(dst + (i >> 1)) = out;
}
