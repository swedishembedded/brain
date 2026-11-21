// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  RMSNorm with no gain: y = x * rsqrt(mean(x^2) + eps), one workgroup per row
// @how   64-thread workgroup per row, strided partial sums, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The gamma-free RMSNorm (the value normalisation of an attention that scales its values by nothing else, the
// embedding norm of a model whose gain is folded elsewhere): y_c = x_c * r, r = 1/sqrt(mean_c(x^2) + eps). The same
// operator as rmsnorm_rows with a gain of one, without the gain buffer. Backward in rmsnorm_nogain_dx.wgsl.
//
//   x, out: [rows, d]   params: d, rows, eps (f32 bits)
// Dispatch: rows * 64 invocations (one workgroup per row).

struct Params {
    d: u32,
    rows: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

var<workgroup> part: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    let live = row < p.rows;
    let base = row * p.d;
    var ss = 0.0;
    if (live) {
        for (var c = t; c < p.d; c = c + 64u) {
            let v = x[base + c];
            ss = ss + v * v;
        }
    }
    part[t] = ss;
    workgroupBarrier();
    var tot = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) { tot = tot + part[i]; }
    let r = inverseSqrt(tot / f32(p.d) + p.eps);
    if (live) {
        for (var c = t; c < p.d; c = c + 64u) {
            out[base + c] = x[base + c] * r;
        }
    }
}
