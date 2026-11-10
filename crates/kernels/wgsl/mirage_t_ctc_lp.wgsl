// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training CTC log-softmax per frame: lp[t, k] = x[t, k] - logsumexp(x[t])
// @how   64-thread workgroup per frame, online (max, sum) partials, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Params: frames, classes. Dispatch: frames workgroups.

struct Params { frames: u32, classes: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> lp: array<f32>;

var<workgroup> p_max: array<f32, 64>;
var<workgroup> p_sum: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.frames) { return; }
    let base = row * p.classes;
    var m = -3.0e38;
    var s = 0.0;
    for (var c = t; c < p.classes; c = c + 64u) {
        let val = x[base + c];
        if (val > m) {
            s = s * exp(m - val) + 1.0;
            m = val;
        } else {
            s = s + exp(val - m);
        }
    }
    p_max[t] = m;
    p_sum[t] = s;
    workgroupBarrier();
    var big = -3.0e38;
    for (var i = 0u; i < 64u; i = i + 1u) {
        if (p_max[i] > big) { big = p_max[i]; }
    }
    var total = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        total = total + p_sum[i] * exp(p_max[i] - big);
    }
    let l = big + log(total);
    for (var c = t; c < p.classes; c = c + 64u) {
        lp[base + c] = x[base + c] - l;
    }
}
