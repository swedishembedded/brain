// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Peak packed-int8 (DP4A) rate probe - the INT8 half of the device roofline
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// Peak `dot4I8Packed` rate — the int8 sibling of `roof_fma`.
//
// WHY THIS IS NEEDED, and it is not a nicety. On a served Qwen3-0.6B step the
// profile graded `matmul_i8_dyn` at about a third of roof - against the **fp32**
// roof, because that was the only one measured. Pascal's
// DP4A path is several times the fp32 rate, so grading an int8 kernel against
// fp32 flatters it by exactly that factor, and a kernel that looks like it is
// at a third of the machine may be at a tenth. The whole method here is "rank
// against the roof", so ranking an int8 kernel needs an int8 roof.
//
// Structure mirrors `roof_fma`: eight independent accumulators so the pipeline
// is never dependency-stalled, values seeded from the uniform so nothing
// constant-folds, no memory traffic in the loop, and the result written out so
// nothing is dead code. Unlike an FMA chain, `dot4I8Packed(a, b) + acc` with a
// loop-invariant `a, b` is a loop-invariant dot plus an add - a compiler that
// sees through the dot (one that inlines a byte-wise emulation of it does)
// hoists it and the loop times additions, reporting several times the
// silicon's int8 rate. So each dot reads its OWN accumulator as the packed
// operand: every dot depends on the last, and none can leave the loop.
//
// INT_OPS = n * iters * 8 * 8: eight `dot4I8Packed` per iteration, each a
// 4-wide dot = 4 multiplies + 4 adds. Counted the same way `gpu_core::cost`
// counts the int8 GEMMs, so the two are directly comparable.

struct Params {
    n: u32,      // active threads
    iters: u32,  // loop trip count (the caller calibrates for duration)
    a: u32,      // packed 4x int8 operand — via the uniform so it cannot fold
    b: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n) { return; }

    let b = p.b;
    var a0 = i32(p.a); var a1 = i32(p.a ^ 1u); var a2 = i32(p.a ^ 2u); var a3 = i32(p.a ^ 3u);
    var a4 = i32(p.a ^ 4u); var a5 = i32(p.a ^ 5u); var a6 = i32(p.a ^ 6u); var a7 = i32(p.a ^ 7u);

    for (var i: u32 = 0u; i < p.iters; i = i + 1u) {
        a0 = dot4I8Packed(bitcast<u32>(a0), b) + a0;
        a1 = dot4I8Packed(bitcast<u32>(a1), b) + a1;
        a2 = dot4I8Packed(bitcast<u32>(a2), b) + a2;
        a3 = dot4I8Packed(bitcast<u32>(a3), b) + a3;
        a4 = dot4I8Packed(bitcast<u32>(a4), b) + a4;
        a5 = dot4I8Packed(bitcast<u32>(a5), b) + a5;
        a6 = dot4I8Packed(bitcast<u32>(a6), b) + a6;
        a7 = dot4I8Packed(bitcast<u32>(a7), b) + a7;
    }

    let s = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
    out[idx] = inp[idx] + f32(s);
}
