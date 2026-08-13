// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  SnakeBeta activation (forward) - the periodic activation in the codec SEANet decoder / BigVGAN-style vocoder
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// SnakeBeta activation (forward) — the periodic activation in the codec SEANet
// decoder / BigVGAN-style vocoder. Per-channel learnable alpha/beta stored in
// log space:
//   a = exp(alpha[c]) ;  b = exp(beta[c]) + eps
//   y = x + (1/b) * sin(a*x)^2
// Layout is generic [rows, C, inner] (NCL: inner = length; per-feature on a
// [.,D] tensor: inner = 1, C = D). Channel index c = (idx / inner) % C.

struct Params {
    total: u32,
    c: u32,
    inner: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read>       alpha: array<f32>;
@group(0) @binding(3) var<storage, read>       beta:  array<f32>;
@group(0) @binding(4) var<storage, read_write> out:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let c = (idx / p.inner) % p.c;
    let a = exp(alpha[c]);
    let b = exp(beta[c]) + p.eps;
    let s = sin(x[idx] * a);
    out[idx] = x[idx] + (1.0 / b) * s * s;
}
