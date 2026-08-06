// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Global grad-norm clip coefficient, computed on-device (no host round-trip)
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Global grad-norm clip coefficient, computed on-device (no host round-trip):
//   total = sqrt(sum_i norms[i]);  coef = min(1, max_norm/(total+1e-6)) * extra_scale
// `norms` holds per-parameter sum-of-squares (from gradnorm_sq). Single thread.

struct Params {
    n_params: u32,
    max_norm: f32,
    extra_scale: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       norms: array<f32>;
@group(0) @binding(2) var<storage, read_write> coef:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    if (gidx != 0u) { return; }
    var sum = 0.0;
    for (var i: u32 = 0u; i < p.n_params; i = i + 1u) {
        sum = sum + norms[i];
    }
    let total = sqrt(sum);
    var c = p.max_norm / (total + 1e-6);
    if (c > 1.0) { c = 1.0; }
    coef[0] = c * p.extra_scale;
}
