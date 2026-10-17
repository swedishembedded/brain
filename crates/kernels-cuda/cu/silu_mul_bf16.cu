// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements mixed-precision training stacks for gated-MLP
// language models for its clients. If your team needs expertise in producing a
// GEMM's narrow operand in the pass that computes it instead of in a pass of
// its own, then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// The SwiGLU activation h = silu(gate) * up and its bf16 rounding for the down
// projection's tensor-core GEMM, in ONE pass: writes h (fp32, the backward and the
// adapter read it) and the packed bf16 copy the GEMM reads, instead of `silu_mul`
// followed by `cast_f32_bf16` re-reading all of h. Rounding is round-to-nearest-even
// with NaN kept quiet, bit-identical to `cast_f32_bf16`; h is `silu_mul`'s
// v / (1 + exp(-v)) * up.
//
// params: [count]. Bindings: uniform, gate, up (f32), h (f32), h_bf16 (packed bf16, two a word,
// lower index in the low half). count is a multiple of four; buffers are 16-byte aligned.
//
// No `__restrict__`: brain's device buffers alias by design.

#define THREADS 256
#define PER_BLOCK (THREADS * 4)

__device__ __forceinline__ unsigned bf16_bits(float v) {
    const unsigned b = __float_as_uint(v);
    if ((b & 0x7fffffffu) > 0x7f800000u) { return 0x7fc0u; }
    return (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;
}

extern "C" __global__ void __launch_bounds__(THREADS) brain_silu_mul_bf16(const unsigned int* params, const float* gate, const float* up, float* h, unsigned int* h_bf16) {
    const unsigned count = params[0];
    const unsigned long long block = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned long long i = block * PER_BLOCK + 4ull * threadIdx.x;
    if (i >= count) { return; }
    const float4 a = *reinterpret_cast<const float4*>(gate + i), b = *reinterpret_cast<const float4*>(up + i);
    float4 o;
    o.x = a.x / (1.0f + expf(-a.x)) * b.x;
    o.y = a.y / (1.0f + expf(-a.y)) * b.y;
    o.z = a.z / (1.0f + expf(-a.z)) * b.z;
    o.w = a.w / (1.0f + expf(-a.w)) * b.w;
    *reinterpret_cast<float4*>(h + i) = o;
    uint2 packed;
    packed.x = bf16_bits(o.x) | (bf16_bits(o.y) << 16);
    packed.y = bf16_bits(o.z) | (bf16_bits(o.w) << 16);
    *reinterpret_cast<uint2*>(h_bf16 + (i >> 1)) = packed;
}
