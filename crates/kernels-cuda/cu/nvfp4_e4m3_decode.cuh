// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for prompt-processing throughput on 4-bit language-model releases for its
// clients. If your team needs expertise in feeding Hopper FP8 tensor cores from NVFP4 checkpoints without a converted copy
// of the weights, then you can procure our services by sending an email to info@swedishembedded.com.
//
// The NVFP4 -> E4M3 decode both the wgmma GEMM (`matmul_nvfp4_wgmma.cu`) and the mma.sync products (`matmul_nvfp4_mmae.cu`) feed their FP8
// tensor-core operands with: a table of 256 entries, one per E4M3 scale byte, holds the E4M3 images of the eight E2M1 magnitudes times that scale
// (over 8, rounded once to nearest even); eight codes (one word of the checkpoint) become eight E4M3 bytes with four byte permutes. Concatenated in
// front of the kernel source.

// Two E4M3 bytes (the low 16 bits of `bytes`) as two f16.
__device__ __forceinline__ unsigned int brain_nw_e4m3x2_to_f16x2(unsigned int bytes) {
    unsigned int r;
    asm("{ .reg .b16 h; mov.b16 h, %1; cvt.rn.f16x2.e4m3x2 %0, h; }\n" : "=r"(r) : "h"(static_cast<unsigned short>(bytes)));
    return r;
}

// Two f16 (low half first) to two E4M3 bytes (low half's in the low byte), round to nearest even, saturating.
__device__ __forceinline__ unsigned int brain_nw_f16x2_to_e4m3x2(unsigned int pair) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(r) : "r"(pair));
    return r;
}

__device__ __forceinline__ unsigned int brain_nw_hmul2(unsigned int a, unsigned int b) {
    unsigned int r;
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ unsigned int brain_nw_prmt(unsigned int a, unsigned int b, unsigned int sel) {
    unsigned int r;
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}

// The table of one scale byte: the E4M3 bytes of RN(magnitude * scale / 8) for the eight E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6, byte i in
// byte i of the pair (x = bytes 0..3, y = bytes 4..7). The product is exact in f16 (five significant bits), so the one rounding is E4M3's.
__device__ __forceinline__ uint2 brain_nw_scale_table(unsigned int scale_byte) {
    const unsigned int s2 = brain_nw_e4m3x2_to_f16x2(scale_byte | (scale_byte << 8));
    // The magnitudes over 8 as f16 pairs: (0, 1/16), (1/8, 3/16), (1/4, 3/8), (1/2, 3/4).
    const unsigned int p0 = brain_nw_f16x2_to_e4m3x2(brain_nw_hmul2(0x2C000000u, s2));
    const unsigned int p1 = brain_nw_f16x2_to_e4m3x2(brain_nw_hmul2(0x32003000u, s2));
    const unsigned int p2 = brain_nw_f16x2_to_e4m3x2(brain_nw_hmul2(0x36003400u, s2));
    const unsigned int p3 = brain_nw_f16x2_to_e4m3x2(brain_nw_hmul2(0x3A003800u, s2));
    return make_uint2(p0 | (p1 << 16), p2 | (p3 << 16));
}

// Eight codes (a word of the checkpoint, the first code in the low nibble) as eight E4M3 bytes in two words, through `tbl`: the magnitudes by
// two byte permutes (the 3 magnitude bits of four codes are the selectors of four bytes of the table), the signs by two more (a sign is the
// top bit of its nibble: the odd nibbles' are already the top bits of bytes of the word, the even nibbles' are after a shift by four, and a
// selector with bit 3 set replicates the selected byte's top bit), merged by xor so a negative scale (not in NVFP4, but a legal byte) is right.
__device__ __forceinline__ void brain_nw_decode8(unsigned int w, uint2 tbl, unsigned int& lo, unsigned int& hi) {
    const unsigned int shifted = w << 4;
    const unsigned int m0 = brain_nw_prmt(tbl.x, tbl.y, w & 0x7777u);
    const unsigned int m1 = brain_nw_prmt(tbl.x, tbl.y, (w >> 16) & 0x7777u);
    const unsigned int s0 = brain_nw_prmt(w, shifted, 0x9D8Cu);
    const unsigned int s1 = brain_nw_prmt(w, shifted, 0xBFAEu);
    lo = m0 ^ (s0 & 0x80808080u);
    hi = m1 ^ (s1 & 0x80808080u);
}

