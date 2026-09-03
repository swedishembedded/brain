// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-(row,group) exact integer sum of a packed int8 activation - S[m,g] = Σ_{k in g} xq[m,k], the activation-only piece of the affine K-quant correction term
// @how   one thread per output element, 8x dot4I8Packed against 0x01010101u (exact int8 lane sum, no rounding)
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// Swedish Embedded AB implements quantized inference kernels for edge and
// embedded GPUs for its clients. If your team needs expertise in shipping
// affine K-quant (GGUF Q4_K/Q5_K-class) inference on commodity GPU hardware
// then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Prepass for the affine K-quant GEMM correction term. An affine weight group
// reconstructs as `ds*code - dm` rather than the symmetric family's `ds*code`,
// so the GEMM's per-output value needs a second reduction alongside the usual
// int8 dot product:
//
//   out[m,n] = sx[m] * Σ_g( ds[n,g]*A[m,n,g] - dm[n,g]*S[m,g] )
//
// A[m,n,g] = Σ_{k in g}( q[n,k]*xq[m,k] ) is the existing dot4I8Packed-shaped
// term every int8 GEMM already computes in its k-loop. S[m,g] = Σ_{k in g}(
// xq[m,k] ) is NEW and, critically, independent of n - a property of the
// activation alone. Computing it inside the GEMM's k-loop would redo the same
// sum once per output COLUMN (N times); this kernel computes it ONCE per
// activation instead, at 1/N the cost, and the GEMM's epilogue then just reads
// S[m,g] back.
//
// The sum is taken over the packed int8 activation xq (model::int8's existing
// max_abs_row -> quant_pack pipeline output), never over the f32 activation
// directly - summing the f32 values and dividing by sx would look similar but
// is NOT the same term: xq is already rounded, and the correction has to
// match exactly what the GEMM's A term consumes, or the mismatch shows up as
// a bias proportional to dm. dot4I8Packed against 0x01010101u (four 0x01
// lanes) sums a packed word's four signed int8 lanes with no rounding - the
// same builtin the GEMM itself uses for its dot product, just with an all-ones
// second operand - and the result needs no scaling: it is already the exact
// integer Σ xq.
//
//   xq  : [M, K/4]  u32 - 4 int8 activations packed along K per u32
//   xgs : [M, K/32] f32 - one exact integer sum per 32-element group, per row
//
// One weight-scale group is 32 int8 = 8 packed u32 words (model::int8::GROUP
// = 32, model::int8::WORDS_PER_GROUP = 8), so one thread reads exactly 8
// words. |S| <= 32*127 = 4064, far inside f32's exact-integer range (2^24),
// so `f32(sum)` loses nothing - the whole kernel is bit-exact by construction,
// no tolerance ever applies to it (crates/model/tests/quant_group_sum.rs
// gates that with assert_eq! against a host-computed integer oracle).
//
// K must be a multiple of 32 (one thread's whole group). One invocation per
// OUTPUT f32 (M*K/32).

struct Params { m: u32, k: u32 };  // k = full K (multiple of 32)

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       xq:  array<u32>;
@group(0) @binding(2) var<storage, read_write> xgs: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let gs = p.k / 32u;
    let total = p.m * gs;
    if (idx >= total) { return; }
    let m = idx / gs;
    let g = idx % gs;
    let base = m * (p.k / 4u) + g * 8u;  // 8 packed words per 32-element group
    var s: i32 = 0;
    for (var w: u32 = 0u; w < 8u; w = w + 1u) {
        s = s + dot4I8Packed(xq[base + w], 0x01010101u);
    }
    xgs[idx] = f32(s);
}
