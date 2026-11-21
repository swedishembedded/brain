// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for serving and training mixture-of-experts models with non-standard expert
// activations for its clients. If your team needs expertise in gated expert FFNs then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Gated expert activation h = act(gate, up) over separate, concatenated or row-interleaved gate/up values: SwiGLU or the softcapped-sigmoid gate
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The activation between an expert's gate/up projection and its down projection, with the activation function and
// the layout of the two projections declared by parameters, so one kernel serves every expert FFN:
//
//   act 0, SwiGLU:            h = silu(g) * u,   silu(g) = g / (1 + exp(-g))      (silu_mul's expression, bit for bit)
//   act 1, softcap-sigmoid:   h = softcap_cg(g) * sigmoid(g) * softcap_cu(u),     softcap_c(v) = c * tanh(v / c)
//                             the sigmoid sees the RAW gate; the caps only bound the linear factors
//
//   mode 0, separate:    g = a[row, j], u = b[row, j]                  (two projections, today's expert layout)
//   mode 1, concat:      g = a[row, j], u = a[row, ff + j]             (one fused projection of 2*ff outputs)
//   mode 2, interleaved: g = a[row, 2j], u = a[row, 2j + 1]            (rows g0,u0,g1,u1,... of a fused weight)
//
// The fused layouts read the output of ONE projection with n = 2*ff straight from its buffer: a checkpoint that stores
// the fused weight (and its block scales) interleaved is served as published, no de-interleave pass exists. `b` is read
// only for mode 0 (bind any buffer otherwise). The backward is glu_act_bwd.wgsl.
//
//   params : u32 [rows, ff, mode, act, cap_gate (f32 bits), cap_up (f32 bits)]   caps are read only for act 1
//   a      : [rows, ff] (separate) or [rows, 2*ff] f32
//   b      : [rows, ff] f32, mode 0 only
//   h      : [rows, ff] f32

struct Params {
    rows: u32,
    ff: u32,
    mode: u32,
    act: u32,
    cap_gate: f32,
    cap_up: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a: array<f32>;
@group(0) @binding(2) var<storage, read>       b: array<f32>;
@group(0) @binding(3) var<storage, read_write> h: array<f32>;

fn capped(v: f32, c: f32) -> f32 {
    return c * tanh(clamp(v / c, -15.0, 15.0));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.ff) { return; }
    let row = idx / p.ff;
    let j = idx % p.ff;
    var g: f32;
    var u: f32;
    if (p.mode == 0u) {
        g = a[idx];
        u = b[idx];
    } else if (p.mode == 1u) {
        g = a[row * 2u * p.ff + j];
        u = a[row * 2u * p.ff + p.ff + j];
    } else {
        g = a[row * 2u * p.ff + 2u * j];
        u = a[row * 2u * p.ff + 2u * j + 1u];
    }
    if (p.act == 0u) {
        let silu = g / (1.0 + exp(-g));
        h[idx] = silu * u;
    } else {
        let sig = 1.0 / (1.0 + exp(-g));
        h[idx] = capped(g, p.cap_gate) * sig * capped(u, p.cap_up);
    }
}
