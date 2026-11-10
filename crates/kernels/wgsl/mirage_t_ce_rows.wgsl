// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training token cross-entropy rows: rowloss[r] = w[r] (logsumexp(x[r]) - x[r, tgt[r]]) and lse[r]
// @how   64-thread workgroup per row, online (max, sum) partials, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// A row with weight 0 is ignored (its tgt is not read). The caller sums rowloss with mirage_t_sum.
// Params: rows, v. Dispatch: rows workgroups.

struct Params { rows: u32, v: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       tgt: array<u32>;
@group(0) @binding(3) var<storage, read>       w: array<f32>;
@group(0) @binding(4) var<storage, read_write> rowloss: array<f32>;
@group(0) @binding(5) var<storage, read_write> lse: array<f32>;

var<workgroup> p_max: array<f32, 64>;
var<workgroup> p_sum: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.rows) { return; }
    let base = row * p.v;
    var m = -3.0e38;
    var s = 0.0;
    for (var c = t; c < p.v; c = c + 64u) {
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
    if (t == 0u) {
        var big = -3.0e38;
        for (var i = 0u; i < 64u; i = i + 1u) {
            if (p_max[i] > big) { big = p_max[i]; }
        }
        var total = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            total = total + p_sum[i] * exp(p_max[i] - big);
        }
        let l = big + log(total);
        lse[row] = l;
        if (w[row] != 0.0) {
            rowloss[row] = w[row] * (l - x[base + tgt[row]]);
        } else {
            rowloss[row] = 0.0;
        }
    }
}
