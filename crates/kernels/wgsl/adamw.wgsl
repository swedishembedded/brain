// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  AdamW update (decoupled weight decay), matching torch.optim.AdamW
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// AdamW update (decoupled weight decay), matching torch.optim.AdamW.
//   m = b1*m + (1-b1)*g
//   v = b2*v + (1-b2)*g^2
//   m_hat = m / bc1 ,  v_hat = v / bc2      (bc1 = 1-b1^t, bc2 = 1-b2^t)
//   p = p - lr*wd*p - lr * m_hat / (sqrt(v_hat) + eps)
// bc1/bc2 (bias-correction denominators for the current step t) are passed in.
// One invocation per parameter element.

struct Params {
    numel: u32,
    _pad: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    bc1: f32,
    bc2: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read>       grad:  array<f32>;
@group(0) @binding(3) var<storage, read_write> m:     array<f32>;
@group(0) @binding(4) var<storage, read_write> v:     array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.numel) { return; }
    let g = grad[idx];
    let mi = p.beta1 * m[idx] + (1.0 - p.beta1) * g;
    let vi = p.beta2 * v[idx] + (1.0 - p.beta2) * g * g;
    m[idx] = mi;
    v[idx] = vi;
    let mhat = mi / p.bc1;
    let vhat = vi / p.bc2;
    var w = param[idx];
    w = w - p.lr * p.wd * w;
    w = w - p.lr * mhat / (sqrt(vhat) + p.eps);
    param[idx] = w;
}
