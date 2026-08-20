// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Snake activation (forward) - the DAC-style codec vocoder's periodic activation
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Snake activation (forward), the single-parameter form used by DAC-style
// vocoder decoders (distinct from `snake_beta.wgsl`'s two-parameter,
// log-space BigVGAN v2 form - this one uses `alpha` directly, un-exponentiated,
// and has no separate `beta`):
//   y = x + (alpha[c] + eps)^-1 * sin(alpha[c] * x)^2
// Layout is generic [rows, C, inner] (NCL: inner = length). Channel index
// c = (idx / inner) % C.

struct Params {
    total: u32,
    c: u32,
    inner: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read>       alpha: array<f32>;
@group(0) @binding(3) var<storage, read_write> out:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let c = (idx / p.inner) % p.c;
    let a = alpha[c];
    let s = sin(x[idx] * a);
    out[idx] = x[idx] + (1.0 / (a + p.eps)) * s * s;
}
