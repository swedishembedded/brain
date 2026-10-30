// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for FP8 serving of LLMs on GPUs for its clients. If your team needs expertise
// in cutting the memory passes between two FP8 tensor-core GEMMs then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// The SwiGLU product feeding an FP8 down projection, quantised to E4M3 in the same pass:
//
//   y = silu(a) * b                       (kept: an fp32 consumer of the same rows reads it)
//   s = amax(y over 128) / 448            (one scale per row per 128 elements)
//   q = e4m3(y / s)                       (round to nearest even, saturating)
//
// It is `quant_epilogue` (mode 1, no packing) followed by `quant_e4m3_128` with the fp32 product never read back: the
// separate pair moved 2 x 4 bytes in, 4 + 4 bytes out and 4 bytes in again per element; this moves 8 in, 4 + 1 out. The
// arithmetic is those two kernels' own expressions (`x / (1 + expf(-x))` then `* b`, explicit round-to-nearest; the scale
// and the bytes as `quant_e4m3_128` makes them), so the result equals the pair's bit for bit.
//
//   params : u32 [rows, k]    k a multiple of 128
//   a, b   : [rows, k] f32    gate and up rows (16-byte aligned)
//   y      : [rows, k] f32
//   q      : [rows, k] bytes  E4M3, four to a word
//   s      : [rows, k / 128] f32
//
// One warp per (row, 128-block), every lane four consecutive elements (16-byte accesses), 256 threads a block with no shared
// memory: many resident blocks hide the exponential and the two IEEE divisions an element costs, which one 1024-thread block
// an SM could not (knowledge 385 item 3). An all-zero block gets the scale 0 and zero bytes.
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_SMQ_THREADS 256
#define BRAIN_SMQ_WARPS (BRAIN_SMQ_THREADS / 32)

extern "C" __global__ void __launch_bounds__(BRAIN_SMQ_THREADS)
brain_silu_mul_quant_e4m3_128(const unsigned int* params, const float* a, const float* b, float* y, unsigned int* q, float* s) {
    const unsigned long long rows = params[0];
    const unsigned long long k = params[1];
    const unsigned long long blocks_per_row = k / 128ull;
    const unsigned long long block = ((unsigned long long)blockIdx.y * gridDim.x + blockIdx.x) * BRAIN_SMQ_WARPS + (threadIdx.x >> 5);
    if (block >= rows * blocks_per_row) { return; }
    const unsigned lane = threadIdx.x & 31u;
    const unsigned long long at = block * 128ull + 4ull * lane;
    const float4 g = *reinterpret_cast<const float4*>(a + at);
    const float4 u = *reinterpret_cast<const float4*>(b + at);
    float4 v;
    v.x = __fmul_rn(__fdiv_rn(g.x, __fadd_rn(1.0f, expf(-g.x))), u.x);
    v.y = __fmul_rn(__fdiv_rn(g.y, __fadd_rn(1.0f, expf(-g.y))), u.y);
    v.z = __fmul_rn(__fdiv_rn(g.z, __fadd_rn(1.0f, expf(-g.z))), u.z);
    v.w = __fmul_rn(__fdiv_rn(g.w, __fadd_rn(1.0f, expf(-g.w))), u.w);
    *reinterpret_cast<float4*>(y + at) = v;
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o)); }
    const float scale = amax / 448.0f;
    unsigned int packed = 0u;
    if (scale > 0.0f) {
        unsigned short lo, hi;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(__fdiv_rn(v.y, scale)), "f"(__fdiv_rn(v.x, scale)));
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(__fdiv_rn(v.w, scale)), "f"(__fdiv_rn(v.z, scale)));
        packed = (unsigned int)lo | ((unsigned int)hi << 16);
    }
    q[block * 32ull + lane] = packed;
    if (lane == 0u) { s[block] = scale; }
}
