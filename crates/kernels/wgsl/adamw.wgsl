// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  AdamW update (decoupled weight decay) fused with grad unscale/clip, matching torch.optim.AdamW + clip_grad_norm_
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// AdamW update (decoupled weight decay), matching torch.optim.AdamW, with the
// grad pre-scale (`grad_scale`/`grad_scale_buf`'s job) folded directly in:
//   g     = grad[i] * scale * coef[0]
//   m     = b1*m + (1-b1)*g
//   v     = b2*v + (1-b2)*g^2
//   m_hat = m / bc1 ,  v_hat = v / bc2      (bc1 = 1-b1^t, bc2 = 1-b2^t)
//   p     = p - lr*wd*p - lr * m_hat / (sqrt(v_hat) + eps)
//
// `coef` is a DEVICE-resident scalar: the on-device global grad-norm clip
// coefficient (`clip_coef`/`clip_coef_wg`) when clipping is active, or a
// caller-owned constant `[1.0]` otherwise - either way this kernel never
// branches on which, it just reads `coef[0]`. `scale` is a host-known
// constant (e.g. a 1/n_accum averaging factor) folded in only when there is
// no device-computed coefficient to fold it into instead; see `optim::Optim`.
// This removes the separate `grad_scale`/`grad_scale_buf` dispatch stage
// (and the buffer round-trip it cost) entirely: `optim::Optim` no longer
// dispatches either kernel.
//
// `desc[0]` is this dispatch's element count, written ONCE when the
// optimizer's dispatch graph is built and never again - every OTHER field
// this kernel reads (`p`, `coef`) is shared across every parameter tensor's
// AdamW dispatch in a step, so `p` is a single uniform buffer written once
// per step regardless of parameter count, not once per tensor. See
// `optim::Optim::step`.

struct Params {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    bc1: f32,
    bc2: f32,
    scale: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read>       grad:  array<f32>;
@group(0) @binding(3) var<storage, read_write> m:     array<f32>;
@group(0) @binding(4) var<storage, read_write> v:     array<f32>;
@group(0) @binding(5) var<storage, read>       desc:  array<u32>;
@group(0) @binding(6) var<storage, read>       coef:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= desc[0]) { return; }
    let g = grad[idx] * p.scale * coef[0];
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
