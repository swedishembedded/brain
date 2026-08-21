// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  ELU backward - gradient w.r.t
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// ELU backward - gradient w.r.t. the pre-activation `x`.
//   dx = dy               if x > 0
//   dx = dy*alpha*exp(x)  otherwise (since y = alpha*(exp(x)-1), y'(x) =
//   alpha*exp(x) there). Elementwise; must stay consistent with elu.wgsl for
//   the gradient check.

struct Params {
    total: u32,
    alpha: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.total) { return; }
    let v = x[idx];
    if (v > 0.0) { dx[idx] = dy[idx]; } else { dx[idx] = dy[idx] * p.alpha * exp(v); }
}
