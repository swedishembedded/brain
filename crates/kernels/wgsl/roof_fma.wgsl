// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Peak fp32 FMA-rate probe — the COMPUTE half of the device roofline
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Peak fp32 FMA-rate probe — the COMPUTE half of the device roofline.
//
// Why this exists. Every "% of peak" number in this engine was a P40 literal
// (`PEAK_TFLOPS = 11.76`) copied into each bench, so the moment anything ran on
// other hardware the utilisation column reported a fiction. The roof has to be a
// property of the device that ran, and no compute API reports it — so it is
// measured, by this kernel and by `axpy` (the bandwidth half).
//
// Why not measure it with `matmul_reg3`. That would report the *best kernel we
// have written*, not the *silicon*, and grading brain's kernels against brain's
// best kernel hides exactly the gap this workstream exists to close. This kernel
// therefore does no useful work: it is a dependency-free FMA chain in registers
// with no memory traffic in the loop at all, which is what a peak-rate probe is.
//
// Design notes, each load-bearing:
//   * EIGHT independent accumulators, so the FMA pipeline is never stalled on a
//     dependency however deep the device's FMA latency is. One would suffice at
//     high occupancy; eight makes the probe insensitive to occupancy.
//   * `c` and `d` arrive through the uniform (bitcast from u32), so no compiler
//     can constant-fold the chain. A folded loop measures nothing and reports a
//     spectacular number, which is the failure mode to design against.
//   * `c = 0.5, d = 0.5` has fixed point 1.0: the accumulators converge in a few
//     iterations and then stay exactly 1.0 — no overflow, no denormals (which
//     are slow on some hardware and would understate the roof), no NaNs.
//   * The result is written out, so nothing is dead code.
//
// FLOPs = n * iters * 8 * 2 (one FMA is a multiply and an add).

struct Params {
    n: u32,      // active threads
    iters: u32,  // FMA-loop trip count (the caller calibrates this for duration)
    c: u32,      // bitcast<f32> multiplier — via the uniform so it cannot fold
    d: u32,      // bitcast<f32> addend
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n) { return; }

    let c = bitcast<f32>(p.c);
    let d = bitcast<f32>(p.d);
    let s = inp[idx];

    var a0 = s;
    var a1 = s + 1.0;
    var a2 = s + 2.0;
    var a3 = s + 3.0;
    var a4 = s + 4.0;
    var a5 = s + 5.0;
    var a6 = s + 6.0;
    var a7 = s + 7.0;

    for (var i: u32 = 0u; i < p.iters; i = i + 1u) {
        a0 = a0 * c + d;
        a1 = a1 * c + d;
        a2 = a2 * c + d;
        a3 = a3 * c + d;
        a4 = a4 * c + d;
        a5 = a5 * c + d;
        a6 = a6 * c + d;
        a7 = a7 * c + d;
    }

    out[idx] = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
}
