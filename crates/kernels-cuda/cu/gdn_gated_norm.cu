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
// once. A half-warp a head row, eight values a lane (16-byte loads of bf16 rows), the mean by a shuffle reduction; a warp walks ROWS rows.
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
#define PAIRS 2  // pairs of head rows a warp walks (a half-warp a row), all loaded before the first is used
#define ROWS (2 * PAIRS)

// Eight consecutive elements of `p` (element index `at`, a multiple of 8) widened to f32: bf16 rows are the high halves of the f32 values.
__device__ __forceinline__ void brain_gn_load8(const float* p, unsigned long long at, bool bf16, float (&o)[8]) {
    if (bf16) {
        const uint4 w = *reinterpret_cast<const uint4*>(reinterpret_cast<const unsigned short*>(p) + at);
        o[0] = __uint_as_float(w.x << 16); o[1] = __uint_as_float(w.x & 0xffff0000u); o[2] = __uint_as_float(w.y << 16); o[3] = __uint_as_float(w.y & 0xffff0000u);
        o[4] = __uint_as_float(w.z << 16); o[5] = __uint_as_float(w.z & 0xffff0000u); o[6] = __uint_as_float(w.w << 16); o[7] = __uint_as_float(w.w & 0xffff0000u);
    } else {
        const float4 a = *reinterpret_cast<const float4*>(p + at), b = *reinterpret_cast<const float4*>(p + at + 4);
        o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
    }
}

// SiLU from the hardware exponential and quotient: one MUFU each instead of the library's argument reduction and the
// IEEE division's range check, which made the kernel instruction-bound. The error is a few ulp of f32, far below the
// E4M3 step the result is rounded to.
__device__ __forceinline__ float brain_gn_silu(float z) { return __fdividef(z, 1.0f + __expf(-z)); }

extern "C" __global__ void __launch_bounds__(WARPS * 32) brain_gdn_gated_norm(const unsigned int* params, const float* x, const float* w, const float* z, float* out, unsigned int* q, float* s) {
    const unsigned int total = params[0], first = params[1];
    const float eps = __uint_as_float(params[2]);
    const bool e4m3 = (params[3] & 1u) != 0u;
    const bool x_bf16 = (params[3] & 2u) != 0u, z_bf16 = (params[3] & 4u) != 0u, store = !e4m3 || (params[3] & 8u) == 0u;
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int hl = threadIdx.x & 15u;  // the eight columns 8 hl .. of this half-warp's row
    const unsigned int row0 = first + (blk * WARPS + (threadIdx.x >> 5)) * ROWS + ((threadIdx.x >> 4) & 1u);
    float g[8];
    {
        const float4 a = *reinterpret_cast<const float4*>(w + hl * 8), b = *reinterpret_cast<const float4*>(w + hl * 8 + 4);
        g[0] = a.x; g[1] = a.y; g[2] = a.z; g[3] = a.w; g[4] = b.x; g[5] = b.y; g[6] = b.z; g[7] = b.w;
    }
    // All the rows' loads are issued before any is used: the rows of a warp are independent, and the loads of the next row
    // hide the latency of this row's reductions. A row past the end reads the last row and writes nothing.
    float v[PAIRS][8], zv[PAIRS][8];
#pragma unroll
    for (unsigned int i = 0; i < PAIRS; ++i) {
        const unsigned long long at = (unsigned long long)min(row0 + 2 * i, total - 1u) * D + hl * 8;
        brain_gn_load8(x, at, x_bf16, v[i]);
        brain_gn_load8(z, at, z_bf16, zv[i]);
    }
#pragma unroll
    for (unsigned int i = 0; i < PAIRS; ++i) {
        const unsigned int row = row0 + 2 * i;
        float ss = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) { ss = fmaf(v[i][k], v[i][k], ss); }
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) { ss += __shfl_xor_sync(0xffffffffu, ss, o); }
        const float inv = rsqrtf(ss / (float)D + eps);
        float r[8];
        float amax = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            r[k] = (v[i][k] * inv * g[k]) * brain_gn_silu(zv[i][k]);
            amax = fmaxf(amax, fabsf(r[k]));
        }
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o)); }
        if (row >= total) { continue; }
        if (store) {
            float4* const dst = reinterpret_cast<float4*>(out + (unsigned long long)row * D + hl * 8);
            dst[0] = make_float4(r[0], r[1], r[2], r[3]);
            dst[1] = make_float4(r[4], r[5], r[6], r[7]);
        }
        if (e4m3) {
            const float scale = amax / 448.0f;
            const unsigned int lo = scale > 0.0f ? brain_e4m3x4(make_float4(r[0], r[1], r[2], r[3]), scale) : 0u;
            const unsigned int hi = scale > 0.0f ? brain_e4m3x4(make_float4(r[4], r[5], r[6], r[7]), scale) : 0u;
            *reinterpret_cast<uint2*>(q + (unsigned long long)row * 32ull + hl * 2) = make_uint2(lo, hi);
            if (hl == 0u) { s[row] = scale; }
        }
    }
}
