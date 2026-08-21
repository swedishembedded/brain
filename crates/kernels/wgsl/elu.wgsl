// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  ELU (Exponential Linear Unit) forward:  y = x  if x > 0, y = alpha*(exp(x)-1) otherwise
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// ELU forward:  y = x                if x > 0
//               y = alpha*(exp(x)-1) otherwise.
// `alpha` is passed as a bit-cast f32 in the uniform (PyTorch `nn.ELU()`
// default alpha = 1.0). Matching derivative in elu_bwd.wgsl.

struct Params {
    total: u32,
    alpha: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.total) { return; }
    let v = x[idx];
    if (v > 0.0) { out[idx] = v; } else { out[idx] = p.alpha * (exp(v) - 1.0); }
}
