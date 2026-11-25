// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Vector-quantization nearest-codebook assignment (Euclidean), one WORKGROUP per query - the cooperative form of `vq_argmin.wgsl`
// @how   64-thread workgroup, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `vq_argmin.wgsl`'s assignment, one WORKGROUP per query vector:
//
//   x : [m, d]   cb: [k, d]   out: [2m]   params: m, k, d
//   out[2m]   = f32(argmin_k ||x[m] - cb[k]||^2)   (index, tie -> LOWEST k)
//   out[2m+1] = the minimum squared distance
//
// `vq_argmin` gives thread `t` query `t` and walks all `k * d` codebook values
// from that one invocation: at a residual quantizer's 2048 x 256 codebook that
// is half a million dependent loads per thread, with only as many threads as
// there are query frames. Here lane `t` of a query's workgroup takes codebook
// rows `t, t + 64, ...`, then the 64 partial minima fold after one barrier.
// Dispatch: m * 64 invocations; `gpu_core::Gpu` swaps this kernel in for
// `vq_argmin` and scales the thread count (the kernel-upgrade table and
// `backend_api::select::Op::VqAssign`), so a caller inherits it unchanged.
//
// NUMERICS: bit-identical to `vq_argmin`: every distance is summed over `d` in
// the same order, and the minimum with its lowest index is exact whatever
// order the lanes fold in (a lane's first row starts at f32::MAX, so a row of
// non-finite distances keeps index 0 and f32::MAX exactly as the reference).

struct Params {
    m: u32,
    k: u32,
    d: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       cb:  array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> pd: array<f32, 64>;
var<workgroup> pk: array<u32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear workgroup index (identity for a 1D dispatch).
    let m = wg.y * nwg.x + wg.x;
    if (m >= p.m) { return; }
    let t = li.x;
    let xb = m * p.d;
    var best_k = 0u;
    var best_d = 3.402823e38; // f32::MAX
    for (var k = t; k < p.k; k = k + 64u) {
        let cbb = k * p.d;
        var dist = 0.0;
        for (var i: u32 = 0u; i < p.d; i = i + 1u) {
            let diff = x[xb + i] - cb[cbb + i];
            dist = dist + diff * diff;
        }
        if (dist < best_d) {
            best_d = dist;
            best_k = k;
        }
    }
    pd[t] = best_d;
    pk[t] = best_k;
    workgroupBarrier();
    if (t == 0u) {
        var gd = pd[0];
        var gk = pk[0];
        for (var i = 1u; i < 64u; i = i + 1u) {
            if (pd[i] < gd || (pd[i] == gd && pk[i] < gk)) {
                gd = pd[i];
                gk = pk[i];
            }
        }
        out[2u * m] = f32(gk);
        out[2u * m + 1u] = gd;
    }
}
