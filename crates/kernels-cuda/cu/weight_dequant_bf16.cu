// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training over quantized
// checkpoints for its clients. If your team needs expertise in feeding
// tensor-core GEMMs from packed 4- and 8-bit weights without a full-precision
// copy of the model, you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Decode a window of rows of any packed weight tier into a bf16 scratch that a
// tensor-core GEMM then reads. One kernel for every tier: `fmt` selects the
// stored layout, the same numbering as `kernels::template::DecodeFormat` and
// `wgsl/weight_decode.wgsl` (which decodes to f32 and is the portable
// reference):
//
//   0 bf16 (copy)   1 f16   2 int8 (group 32)   3 int4 (group 32)   4 NF4 (group 32)
//   5 fp8 e4m3   6 fp8 e5m2   (block x block f32 scales)
//   7 Q4_K  8 Q5_K  (affine, packed sub-scale / super-scale planes)   9 Q6_K (group 16)
//   10 NVFP4 (E2M1, E4M3 byte per 16, f32 global)  11 MXFP4 (E2M1, E8M0 byte per 32)
//   12 / 13 MXFP8 E4M3 / E5M2 (E8M0 byte per 32); the byte scales are bound as the f32 plane `s` read as bytes
//
// A decoded value is the exact f32 `decode * scale` of the reference rounded to
// bf16 with round-to-nearest-even; a device test compares it with the host
// decode rounded the same way.
//
// One thread per eight consecutive weights of a row (K is a multiple of 8, so
// the eight share a row, a scale group and a block) writing one 16-byte store.
//
// params: [fmt, row0, rows, k, block, cb, sm_per_row, wd_per_row, global scale bits (NVFP4)]
// Bindings: uniform, packed codes, f32 scales, packed sub-scales (wsm), packed
// super-scales (wd), bf16 output; a tier leaves the planes it does not use
// bound to any buffer.
//
// No `__restrict__` anywhere: brain's device buffers alias by design.

#define BRAIN_WD_THREADS 256

__device__ __forceinline__ float brain_wd_nf4(unsigned code) {
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

__device__ __forceinline__ float brain_wd_with_sign(float magnitude, unsigned sign_bit) {
    return __uint_as_float(__float_as_uint(magnitude) | sign_bit);
}

__device__ __forceinline__ float brain_wd_f16(unsigned h) {
    const unsigned sign = (h & 0x8000u) << 16;
    const unsigned exp = (h >> 10) & 31u;
    const unsigned man = h & 0x3ffu;
    if (exp == 0u) { return brain_wd_with_sign((float)man * 5.9604645e-8f, sign); }
    if (exp == 31u) { return __uint_as_float(sign | 0x7f800000u | (man << 13)); }
    return __uint_as_float(sign | ((exp + 112u) << 23) | (man << 13));
}

__device__ __forceinline__ float brain_wd_e4m3(unsigned b) {
    const unsigned sign = (b & 0x80u) << 24;
    if ((b & 0x7fu) == 0x7fu) { return __uint_as_float(sign | 0x7fc00000u); }
    const unsigned e = (b >> 3) & 15u, m = b & 7u;
    const float mag = e == 0u ? (float)m * 0.001953125f : __uint_as_float(((e + 120u) << 23) | (m << 20));
    return brain_wd_with_sign(mag, sign);
}

__device__ __forceinline__ float brain_wd_e5m2(unsigned b) {
    const unsigned sign = (b & 0x80u) << 24;
    const unsigned e = (b >> 2) & 31u, m = b & 3u;
    if (e == 31u) { return __uint_as_float(sign | 0x7f800000u | (m << 21)); }
    const float mag = e == 0u ? (float)m * 1.52587890625e-5f : __uint_as_float(((e + 112u) << 23) | (m << 21));
    return brain_wd_with_sign(mag, sign);
}

__device__ __forceinline__ float brain_wd_e2m1(unsigned c) {
    const unsigned m = c & 7u;
    const float mag = m < 2u ? 0.5f * (float)m : __uint_as_float(((m >> 1) + 126u) << 23 | ((m & 1u) << 22));
    return brain_wd_with_sign(mag, (c & 8u) << 28);
}

__device__ __forceinline__ float brain_wd_e8m0(unsigned b) {
    if (b == 255u) { return __uint_as_float(0x7fc00000u); }
    return b == 0u ? __uint_as_float(0x00400000u) : __uint_as_float(b << 23);
}

__device__ __forceinline__ unsigned brain_wd_plane_byte(const float* s, unsigned long long i) {
    return (__float_as_uint(s[i >> 2]) >> (8 * (unsigned)(i & 3ull))) & 0xffu;
}

__device__ __forceinline__ float brain_wd_sext(unsigned v, int bits) {
    return (float)((int)(v << (32 - bits)) >> (32 - bits));
}

__device__ __forceinline__ unsigned brain_wd_bf16(float v) {
    const unsigned b = __float_as_uint(v);
    if (v != v) { return (b >> 16) | 0x40u; }
    return (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;
}

extern "C" __global__ void __launch_bounds__(BRAIN_WD_THREADS) brain_weight_dequant_bf16(
    const unsigned int* params, const unsigned int* w, const float* s, const unsigned int* wsm, const unsigned int* wd, unsigned int* out) {
    const unsigned long long t = ((unsigned long long)blockIdx.y * gridDim.x + blockIdx.x) * BRAIN_WD_THREADS + threadIdx.x;
    const unsigned fmt = params[0], row0 = params[1], rows = params[2], k = params[3], block = params[4], cb = params[5];
    const unsigned sm_per_row = params[6], wd_per_row = params[7];
    const unsigned long long e0 = t * 8ull;
    if (e0 >= (unsigned long long)rows * k) { return; }
    const unsigned row = row0 + (unsigned)(e0 / k);
    const unsigned c0 = (unsigned)(e0 % k);
    const unsigned long long wi = (unsigned long long)row * k + c0;
    float v[8];
    if (fmt == 0u) {
        const uint4 r = *reinterpret_cast<const uint4*>(w + (wi >> 1));
        *reinterpret_cast<uint4*>(out + 4ull * t) = r;
        return;
    } else if (fmt == 1u) {
        const uint4 r = *reinterpret_cast<const uint4*>(w + (wi >> 1));
        const unsigned* q = reinterpret_cast<const unsigned*>(&r);
        #pragma unroll
        for (int i = 0; i < 4; ++i) { v[2 * i] = brain_wd_f16(q[i] & 0xffffu); v[2 * i + 1] = brain_wd_f16(q[i] >> 16); }
    } else if (fmt == 2u || fmt == 9u) {
        const uint2 r = *reinterpret_cast<const uint2*>(w + (wi >> 2));
        const float sc = s[fmt == 2u ? (wi >> 5) : (wi >> 4)];
        #pragma unroll
        for (int i = 0; i < 8; ++i) { v[i] = brain_wd_sext(((i < 4 ? r.x : r.y) >> (8 * (i & 3))) & 0xffu, 8) * sc; }
    } else if (fmt == 3u || fmt == 4u) {
        const unsigned word = w[wi >> 3];
        const float sc = s[wi >> 5];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned nib = (word >> (4 * i)) & 15u;
            v[i] = (fmt == 3u ? brain_wd_sext(nib, 4) : brain_wd_nf4(nib)) * sc;
        }
    } else if (fmt == 5u || fmt == 6u) {
        const uint2 r = *reinterpret_cast<const uint2*>(w + (wi >> 2));
        const float sc = s[(row / block) * cb + c0 / block];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned byte = ((i < 4 ? r.x : r.y) >> (8 * (i & 3))) & 0xffu;
            v[i] = (fmt == 5u ? brain_wd_e4m3(byte) : brain_wd_e5m2(byte)) * sc;
        }
    } else if (fmt == 10u || fmt == 11u) {
        const unsigned word = w[wi >> 3];
        const unsigned sb = brain_wd_plane_byte(s, fmt == 10u ? (wi >> 4) : (wi >> 5));
        const float g = fmt == 10u ? __uint_as_float(params[8]) : 1.0f;
        const float sc = fmt == 10u ? brain_wd_e4m3(sb) : brain_wd_e8m0(sb);
        #pragma unroll
        for (int i = 0; i < 8; ++i) { v[i] = brain_wd_e2m1((word >> (4 * i)) & 15u) * g * sc; }
    } else if (fmt == 12u || fmt == 13u) {
        const uint2 r = *reinterpret_cast<const uint2*>(w + (wi >> 2));
        const float sc = brain_wd_e8m0(brain_wd_plane_byte(s, wi >> 5));
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned byte = ((i < 4 ? r.x : r.y) >> (8 * (i & 3))) & 0xffu;
            v[i] = (fmt == 12u ? brain_wd_e4m3(byte) : brain_wd_e5m2(byte)) * sc;
        }
    } else {
        // Affine K-quant: ds * code - dm per 32-element group.
        const unsigned g = c0 / 32u;
        const unsigned wword = wd[(unsigned long long)row * wd_per_row + g / 8u];
        const unsigned sword = wsm[(unsigned long long)row * sm_per_row + g / 2u];
        const unsigned shift = (g & 1u) * 16u;
        const float ds = brain_wd_f16(wword & 0xffffu) * (float)((sword >> shift) & 0xffu);
        const float dm = brain_wd_f16(wword >> 16) * (float)((sword >> (shift + 8u)) & 0xffu);
        if (fmt == 7u) {
            const unsigned word = w[wi >> 3];
            #pragma unroll
            for (int i = 0; i < 8; ++i) { v[i] = ds * (float)((word >> (4 * i)) & 15u) - dm; }
        } else {
            const uint2 r = *reinterpret_cast<const uint2*>(w + (wi >> 2));
            #pragma unroll
            for (int i = 0; i < 8; ++i) { v[i] = ds * (float)(((i < 4 ? r.x : r.y) >> (8 * (i & 3))) & 0xffu) - dm; }
        }
    }
    uint4 o;
    o.x = brain_wd_bf16(v[0]) | (brain_wd_bf16(v[1]) << 16);
    o.y = brain_wd_bf16(v[2]) | (brain_wd_bf16(v[3]) << 16);
    o.z = brain_wd_bf16(v[4]) | (brain_wd_bf16(v[5]) << 16);
    o.w = brain_wd_bf16(v[6]) | (brain_wd_bf16(v[7]) << 16);
    *reinterpret_cast<uint4*>(out + 4ull * t) = o;
}
