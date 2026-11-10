// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training CenterNet focal loss per cell: forward term or logit gradient
// @how   one thread per cell; accumulates in the backward mode
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// CenterNet focal loss per cell (mode 0 writes the term, mode 1 accumulates the logit gradient scaled by g[0] / norm).
// A cell is positive where tgt >= 1: -alpha w (1-p)^2 ln(p+f); otherwise -(1-alpha) w (1-t)^4 p^2 ln(1-p+f).
// Params: n, alpha, norm (f32 bits), mode. Dispatch: n threads.

struct Params { n: u32, alpha: f32, norm: f32, mode: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       tgt: array<f32>;
@group(0) @binding(3) var<storage, read>       weight: array<f32>;
@group(0) @binding(4) var<storage, read>       g: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n) { return; }
    let w = weight[idx];
    if (w == 0.0) {
        if (p.mode == 0u) { out[idx] = 0.0; }
        return;
    }
    let fl = 1.0e-12;
    let pr = 1.0 / (1.0 + exp(-x[idx]));
    let t = tgt[idx];
    let dp = pr * (1.0 - pr);
    var term = 0.0;
    var d = 0.0;
    if (t >= 1.0) {
        let a = p.alpha * w;
        term = -a * (1.0 - pr) * (1.0 - pr) * log(pr + fl);
        d = -a * (-2.0 * (1.0 - pr) * log(pr + fl) + (1.0 - pr) * (1.0 - pr) / (pr + fl)) * dp;
    } else {
        let omt = 1.0 - t;
        let b = (1.0 - p.alpha) * w * omt * omt * omt * omt;
        term = -b * pr * pr * log(1.0 - pr + fl);
        d = -b * (2.0 * pr * log(1.0 - pr + fl) - pr * pr / (1.0 - pr + fl)) * dp;
    }
    if (p.mode == 0u) {
        out[idx] = term;
    } else {
        out[idx] = out[idx] + d * g[0] / p.norm;
    }
}
