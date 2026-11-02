// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for FP8 activation quantisation ahead of tensor-core GEMMs for its clients.
// If your team needs expertise in serving 4-bit and FP8 checkpoints with FP8 tensor cores then you can procure our
// services by sending an email to info@swedishembedded.com.
//
// Quantise fp32 activation rows to E4M3 with ONE f32 scale per row: `amax / 448` over the whole row (448 is the largest
// finite E4M3FN), the byte the round-to-nearest-even, saturating E4M3 of `x / scale` (the arithmetic of `quant_e4m3_128`,
// `e4m3_quant_common.cuh`). The activation operand of `matmul_nvfp4_wgmma`, whose accumulators are not rescaled along K.
//
//   params : u32 [rows, k]    k a multiple of 4
//   x      : [rows, k] f32    (16-byte aligned)
//   q      : [rows, k] bytes  E4M3, four to a word
//   s      : [rows] f32       an all-zero row gets the scale 0 and zero bytes
//
// A block of 256 threads a row: the row's amax by a block reduction, then the bytes (the second read of the row is an L2 hit).
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_QR_THREADS 256

extern "C" __global__ void __launch_bounds__(BRAIN_QR_THREADS) brain_quant_e4m3_row(const unsigned int* params, const float* x, unsigned int* q, float* s) {
    __shared__ float partial[BRAIN_QR_THREADS / 32];
    const unsigned long long k = params[1];
    const float4* row = reinterpret_cast<const float4*>(x + static_cast<unsigned long long>(blockIdx.x) * k);
    const unsigned int quads = static_cast<unsigned int>(k / 4ull);
    float amax = 0.0f;
    for (unsigned int i = threadIdx.x; i < quads; i += BRAIN_QR_THREADS) {
        const float4 v = row[i];
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o)); }
    if ((threadIdx.x & 31u) == 0u) { partial[threadIdx.x >> 5] = amax; }
    __syncthreads();
    float total = partial[0];
#pragma unroll
    for (int w = 1; w < BRAIN_QR_THREADS / 32; ++w) { total = fmaxf(total, partial[w]); }
    const float scale = total / 448.0f;
    unsigned int* const out = q + static_cast<unsigned long long>(blockIdx.x) * quads;
    for (unsigned int i = threadIdx.x; i < quads; i += BRAIN_QR_THREADS) { out[i] = scale > 0.0f ? brain_e4m3x4(row[i], scale) : 0u; }
    if (threadIdx.x == 0u) { s[blockIdx.x] = scale; }
}
