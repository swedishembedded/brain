// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bound serving kernels for linear-attention language models for its clients. If
// your team needs expertise in collapsing the elementwise glue around a recurrent layer into one pass over its
// activations, then you can procure our services by sending an email to info@swedishembedded.com.
//
// Gated Delta Net's output gate in one pass: the RMSNorm of each head's recurrence output times its gain, multiplied by
// SiLU of the gate projection z ("norm before gate"):
//
//   out[r, :] = (x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * w[:]) * silu(z[r, :])      r = (token row, value head), 128 wide
//
// The chain it replaces is three launches over the whole round (a cooperative RMSNorm, a SiLU and a multiply), each a
// full read and write of the [rows, heads * 128] tensors; here the three tensors are read once and the result written
// once. One warp per head row, four values a lane, the mean by a shuffle reduction.
//
// A head row is one 128-element group of the E4M3 activation quantisation, so with flag bit 0 the same pass also writes
// what `quant_e4m3_128` makes of `out` for the output projection: the bytes `q` (four to a word) and one scale per head row
// in `s` (`amax / 448`, round-to-nearest-even saturating bytes of `out / scale`).
//
// Contract: params [head rows in all, first head row, eps (f32 bits), flags]; read x, z [head rows, 128], w [128]; write out
// [head rows, 128] for rows `first ..` (rows before it, a mixed round's decode rows, are left alone), and with flag bit 0
// q [head rows, 32] u32 and s [head rows] f32. dv = 128. Flag bit 1: x is bf16 rows ([head rows, 64] words, the recurrence's bf16
// output); bit 2: z is bf16 rows (the FP8 GEMM's bf16 output); bit 3 (with bit 0): `out` is not stored - nothing reads the fp32
// product when the E4M3 rows are what the output projection takes. Every value is widened exactly and every sum is f32, so for
// inputs that are bf16 values the E4M3 bytes and scales equal the fp32 form's bit for bit.
// No `__restrict__`: brain's device buffers alias by design.

#define D 128
#define WARPS 4

// Four consecutive bf16 of `p` (element index `at`, a multiple of 4) widened to f32: a bf16 is the high half of the f32 of its value.
__device__ __forceinline__ float4 brain_gn_load_bf16x4(const float* p, unsigned long long at) {
    const uint2 w = *reinterpret_cast<const uint2*>(reinterpret_cast<const unsigned short*>(p) + at);
    return make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xffff0000u), __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xffff0000u));
}

extern "C" __global__ void __launch_bounds__(WARPS * 32) brain_gdn_gated_norm(const unsigned int* params, const float* x, const float* w, const float* z, float* out, unsigned int* q, float* s) {
    const unsigned int total = params[0], first = params[1];
    const float eps = __uint_as_float(params[2]);
    const bool e4m3 = (params[3] & 1u) != 0u;
    const bool x_bf16 = (params[3] & 2u) != 0u, z_bf16 = (params[3] & 4u) != 0u, store = !e4m3 || (params[3] & 8u) == 0u;
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int row = first + blk * WARPS + (threadIdx.x >> 5);
    if (row >= total) { return; }
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned long long at = (unsigned long long)row * D + lane * 4;
    const float4 v = x_bf16 ? brain_gn_load_bf16x4(x, at) : *reinterpret_cast<const float4*>(x + at);
    float ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { ss += __shfl_xor_sync(0xffffffffu, ss, o); }
    const float inv = 1.0f / sqrtf(ss / (float)D + eps);
    const float4 g = *reinterpret_cast<const float4*>(w + lane * 4);
    const float4 zv = z_bf16 ? brain_gn_load_bf16x4(z, at) : *reinterpret_cast<const float4*>(z + at);
    float4 r;
    r.x = (v.x * inv * g.x) * (zv.x / (1.0f + expf(-zv.x)));
    r.y = (v.y * inv * g.y) * (zv.y / (1.0f + expf(-zv.y)));
    r.z = (v.z * inv * g.z) * (zv.z / (1.0f + expf(-zv.z)));
    r.w = (v.w * inv * g.w) * (zv.w / (1.0f + expf(-zv.w)));
    if (store) { *reinterpret_cast<float4*>(out + at) = r; }
    if (e4m3) {
        float amax = fmaxf(fmaxf(fabsf(r.x), fabsf(r.y)), fmaxf(fabsf(r.z), fabsf(r.w)));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o)); }
        const float scale = amax / 448.0f;
        unsigned int packed = 0u;
        if (scale > 0.0f) {
            unsigned short lo, hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(__fdiv_rn(r.y, scale)), "f"(__fdiv_rn(r.x, scale)));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(__fdiv_rn(r.w, scale)), "f"(__fdiv_rn(r.z, scale)));
            packed = (unsigned int)lo | ((unsigned int)hi << 16);
        }
        q[(unsigned long long)row * 32ull + lane] = packed;
        if (lane == 0u) { s[row] = scale; }
    }
}
