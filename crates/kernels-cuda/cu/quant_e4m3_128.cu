// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for FP8 activation quantisation ahead of tensor-core GEMMs for its clients.
// If your team needs expertise in serving FP8 checkpoints with FP8 tensor cores then you can procure our services by
// sending an email to info@swedishembedded.com.
//
// Quantise fp32 activation rows to E4M3 with one f32 scale per 128 elements (the "dynamic activation scheme" of a
// block-FP8 release, per-token groups of 128): for each (row, 128-block) the scale is `amax / 448` (448 is the largest
// finite E4M3FN) and the byte is the round-to-nearest-even, saturating E4M3 of `x / scale`.
//
//   params : u32 [rows, k]    k a multiple of 128
//   x      : [rows, k] f32    (16-byte aligned)
//   q      : [rows, k] bytes  E4M3, four to a word (4-byte aligned)
//   s      : [rows, k / 128] f32
//
// One warp per (row, block): every lane takes four consecutive elements (a 16-byte load), the block's amax is a warp
// shuffle reduction, and each lane writes its four bytes as one word. An all-zero block gets the scale 0 and zero bytes.
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_QE_THREADS 256
#define BRAIN_QE_WARPS (BRAIN_QE_THREADS / 32)

extern "C" __global__ void __launch_bounds__(BRAIN_QE_THREADS) brain_quant_e4m3_128(const unsigned int* params, const float* x, unsigned int* q, float* s) {
    const unsigned long long rows = params[0];
    const unsigned long long k = params[1];
    const unsigned long long blocks_per_row = k / 128ull;
    const unsigned long long block = ((unsigned long long)blockIdx.y * gridDim.x + blockIdx.x) * BRAIN_QE_WARPS + (threadIdx.x >> 5);
    if (block >= rows * blocks_per_row) { return; }
    const unsigned lane = threadIdx.x & 31u;
    const float4 v = *reinterpret_cast<const float4*>(x + block * 128ull + 4ull * lane);
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o)); }
    const float scale = amax / 448.0f;
    q[block * 32ull + lane] = scale > 0.0f ? brain_e4m3x4(v, scale) : 0u;
    if (lane == 0u) { s[block] = scale; }
}
