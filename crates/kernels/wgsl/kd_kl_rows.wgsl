// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements knowledge distillation of large models into small ones for its clients. If your team
// needs expertise in model compression and distillation, you can procure our services by sending an email to
// info@swedishembedded.com.

// @what  Per-row tempered KL(teacher || student) of two logit rows and its gradient with respect to the student's logits
// @how   one thread per row, serial passes over the row's columns, the gradient written over the student's logits
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// For row r < rows, with p = softmax(s[r] * inv_t) and q = softmax(t[r] * inv_t):
//
//   kl[r]   = sum_c q_c * (log q_c - log p_c)            (KL(teacher || student) at temperature 1 / inv_t)
//   s[r, c] = coef * (p_c - q_c)                          (overwrites the student's logits)
//
// `coef` is the host's scale of the loss's derivative: the gradient of `w * T^2 * mean_r kl[r]` with respect to
// the student's logits is `w * T * (p - q) / rows`, so the host passes `coef = w * T / rows`. The value is returned
// unscaled. Columns are summed in ascending order (a fixed order: a step is reproducible). Exactly 3 storage buffers.
//
//   params: rows, cols, inv_t (f32 bits), coef (f32 bits)

struct Params {
    rows: u32,
    cols: u32,
    inv_t: u32,
    coef: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> s:  array<f32>;
@group(0) @binding(2) var<storage, read>       t:  array<f32>;
@group(0) @binding(3) var<storage, read_write> kl: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let r = gid.y * (nwg.x * 64u) + gid.x;
    if (r >= p.rows) { return; }
    let inv_t = bitcast<f32>(p.inv_t);
    let coef = bitcast<f32>(p.coef);
    let base = r * p.cols;

    var ms = -3.4e38;
    var mt = -3.4e38;
    for (var c = 0u; c < p.cols; c = c + 1u) {
        ms = max(ms, s[base + c] * inv_t);
        mt = max(mt, t[base + c] * inv_t);
    }
    var zs = 0.0;
    var zt = 0.0;
    for (var c = 0u; c < p.cols; c = c + 1u) {
        zs = zs + exp(s[base + c] * inv_t - ms);
        zt = zt + exp(t[base + c] * inv_t - mt);
    }
    let lse_s = ms + log(zs);
    let lse_t = mt + log(zt);
    var acc = 0.0;
    for (var c = 0u; c < p.cols; c = c + 1u) {
        let ls = s[base + c] * inv_t - lse_s;
        let lt = t[base + c] * inv_t - lse_t;
        let q = exp(lt);
        acc = acc + q * (lt - ls);
        s[base + c] = coef * (exp(ls) - q);
    }
    kl[r] = acc;
}
