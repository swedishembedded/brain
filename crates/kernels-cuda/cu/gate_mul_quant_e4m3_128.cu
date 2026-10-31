// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for FP8 serving of LLMs on GPUs for its clients. If your team needs expertise
// in cutting the memory passes between a gated attention and its FP8 output projection then you can procure our services
// by sending an email to info@swedishembedded.com.
//
// The sigmoid output gate of a gated attention layer feeding an FP8 `o_proj`, quantised to E4M3 in the same pass:
//
//   y = ctx * sigmoid(gate)               (kept only when flags bit 0 asks: an fp32 consumer of the same rows reads it)
//   s = amax(y over 128) / 448            (one scale per row per 128 elements)
//   q = e4m3(y / s)                       (round to nearest even, saturating)
//
// It is `quant_epilogue` (mode 2) followed by `quant_e4m3_128` with the fp32 product never read back, and with the gate read in
// place: the query projection's output interleaves, per head, `hd` query values and `hd` gate values, so the gate of element
// `c` of a row is at `q_full[row * 2 * k + (c / hd) * 2 * hd + hd + c % hd]` and no copy of it is made. The arithmetic is those
// kernels' own expressions (`1 / (1 + expf(-g))`, then `ctx * that`, explicit round-to-nearest; the scale and the bytes as
// `quant_e4m3_128` makes them), so the result equals the pair's bit for bit.
//
//   params : u32 [rows, k, flags, hd]   k a multiple of 128 and of hd, hd a multiple of 128 (a warp's 128 elements are one head's).
//                                       flags bit 0: store the product `y`
//   ctx    : [rows, k] f32              the attention output
//   gate   : [rows, 2 k] f32            the query projection's output (query and gate halves of every head)
//   y      : [rows, k] f32              written with bit 0 only
//   q      : [rows, k] bytes E4M3, four to a word
//   s      : [rows, k / 128] f32
//
// One warp per (row, 128-block), every lane four consecutive elements (16-byte accesses), 256 threads a block with no shared
// memory, as `silu_mul_quant_e4m3_128`. An all-zero block gets the scale 0 and zero bytes.
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_GMQ_THREADS 256
#define BRAIN_GMQ_WARPS (BRAIN_GMQ_THREADS / 32)

extern "C" __global__ void __launch_bounds__(BRAIN_GMQ_THREADS)
brain_gate_mul_quant_e4m3_128(const unsigned int* params, const float* ctx, const float* gate, float* y, unsigned int* q, float* s) {
    const unsigned long long rows = params[0];
    const unsigned long long k = params[1];
    const bool keep = (params[2] & 1u) != 0u;
    const unsigned long long hd = params[3];
    const unsigned long long blocks_per_row = k / 128ull;
    const unsigned long long block = ((unsigned long long)blockIdx.y * gridDim.x + blockIdx.x) * BRAIN_GMQ_WARPS + (threadIdx.x >> 5);
    if (block >= rows * blocks_per_row) { return; }
    const unsigned lane = threadIdx.x & 31u;
    const unsigned long long row = block / blocks_per_row;
    const unsigned long long c0 = (block % blocks_per_row) * 128ull;  // first element of this block within the row
    const unsigned long long at = block * 128ull + 4ull * lane;       // the same element in `ctx` and `y`
    const unsigned long long gat = row * 2ull * k + (c0 / hd) * 2ull * hd + hd + (c0 % hd) + 4ull * lane;
    const float4 a = *reinterpret_cast<const float4*>(ctx + at);
    const float4 g = *reinterpret_cast<const float4*>(gate + gat);
    float4 v;
    v.x = __fmul_rn(a.x, __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g.x))));
    v.y = __fmul_rn(a.y, __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g.y))));
    v.z = __fmul_rn(a.z, __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g.z))));
    v.w = __fmul_rn(a.w, __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g.w))));
    if (keep) { *reinterpret_cast<float4*>(y + at) = v; }
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o)); }
    const float scale = amax / 448.0f;
    q[block * 32ull + lane] = scale > 0.0f ? brain_e4m3x4(v, scale) : 0u;
    if (lane == 0u) { s[block] = scale; }
}
