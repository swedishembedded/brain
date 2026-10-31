// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for FP8 serving of LLMs on GPUs for its clients. If your team needs expertise
// in cutting the instruction and memory cost of activation quantisation then you can procure our services by sending an
// email to info@swedishembedded.com.
//
// The E4M3 half of every activation quantiser (`quant_e4m3_128`, `silu_mul_quant_e4m3_128`, `add_rms_quant_e4m3`,
// `gdn_gated_norm`): four elements that share one scale become four bytes, `e4m3(RN(v / scale))`, round to nearest even,
// saturating.
//
// `v / scale` is the IEEE quotient (`__fdiv_rn`). The compiler's division is a reciprocal, one Newton step, a product, a
// residual and a correction, plus a range check and a call to a slow path for operands whose exponents leave the
// range where that sequence is exact: about 13 instructions an element, where the scale is the same for 128 elements.
// This takes the reciprocal (and its Newton step) once and spends three fused operations an element - the sequence's
// own operations in its own order, so the quotient is the same bit pattern - and keeps the division itself for a
// scale outside [2^-60, 2^60]. Inside that range the sequence is exact for every element: a quotient that underflows
// is far below half of the smallest E4M3 subnormal (2^-10) and rounds to a zero, whose sign the last step restores.

__device__ __forceinline__ unsigned int brain_e4m3x4(const float4 v, const float scale) {
    float q0, q1, q2, q3;
    if (scale >= 0x1p-60f && scale <= 0x1p60f) {
        float r;
        asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(scale));
        r = fmaf(r, fmaf(-scale, r, 1.0f), r);
        const float a[4] = {v.x, v.y, v.z, v.w};
        float q[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float first = fmaf(r, a[i], 0.0f);
            // A zero or underflowed quotient comes out +0 whatever the numerator's sign; the IEEE quotient keeps it.
            q[i] = copysignf(fmaf(r, fmaf(-scale, first, a[i]), first), a[i]);
        }
        q0 = q[0]; q1 = q[1]; q2 = q[2]; q3 = q[3];
    } else {
        q0 = __fdiv_rn(v.x, scale); q1 = __fdiv_rn(v.y, scale); q2 = __fdiv_rn(v.z, scale); q3 = __fdiv_rn(v.w, scale);
    }
    unsigned short lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(q1), "f"(q0));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(q3), "f"(q2));
    return (unsigned int)lo | ((unsigned int)hi << 16);
}
