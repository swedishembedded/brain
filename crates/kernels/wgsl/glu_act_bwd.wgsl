// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for serving and training mixture-of-experts models with non-standard expert
// activations for its clients. If your team needs expertise in gated expert FFNs then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Backward of glu_act: gradients of the gate and up values from dh, in the same separate, concatenated or interleaved layout
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Backward of glu_act.wgsl (same params and layouts), from the saved gate/up values (the inputs, not h):
//
//   SwiGLU:          sg = sigmoid(g);  dg = dh * u * (sg + g*sg*(1-sg));            du = dh * silu(g)
//                    (silu_bwd_da / silu_bwd_db's expressions, bit for bit)
//   softcap-sigmoid: tg = tanh(g/cg), sg = sigmoid(g), tu = tanh(u/cu)
//                    f(g)  = cg*tg*sg;  f'(g) = (1 - tg^2)*sg + cg*tg*sg*(1 - sg)
//                    dg = dh * f'(g) * cu*tu;                                        du = dh * f(g) * (1 - tu^2)
//
// Mode 0 writes dg to `da` and du to `db`, both [rows, ff]. Modes 1 and 2 write both into `da`, [rows, 2*ff], in
// the layout of the forward's input, so it feeds the fused projection's dX and dW directly; `b` and `db` are then
// untouched (bind any buffer).
//
//   params : u32 [rows, ff, mode, act, cap_gate (f32 bits), cap_up (f32 bits)]
//   a, b   : the forward's inputs        dh : [rows, ff]        da, db : the gradients

struct Params {
    rows: u32,
    ff: u32,
    mode: u32,
    act: u32,
    cap_gate: f32,
    cap_up: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a:  array<f32>;
@group(0) @binding(2) var<storage, read>       b:  array<f32>;
@group(0) @binding(3) var<storage, read>       dh: array<f32>;
@group(0) @binding(4) var<storage, read_write> da: array<f32>;
@group(0) @binding(5) var<storage, read_write> db: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.ff) { return; }
    let row = idx / p.ff;
    let j = idx % p.ff;
    var gi = idx;
    var ui = idx;
    if (p.mode == 1u) {
        gi = row * 2u * p.ff + j;
        ui = gi + p.ff;
    } else if (p.mode == 2u) {
        gi = row * 2u * p.ff + 2u * j;
        ui = gi + 1u;
    }
    var g: f32;
    var u: f32;
    if (p.mode == 0u) {
        g = a[gi];
        u = b[ui];
    } else {
        g = a[gi];
        u = a[ui];
    }
    let d = dh[idx];
    var dg: f32;
    var du: f32;
    if (p.act == 0u) {
        let sig = 1.0 / (1.0 + exp(-g));
        let dsilu = sig + g * sig * (1.0 - sig);
        let silu = g / (1.0 + exp(-g));
        dg = d * u * dsilu;
        du = d * silu;
    } else {
        let sig = 1.0 / (1.0 + exp(-g));
        let tg = tanh(clamp(g / p.cap_gate, -15.0, 15.0));
        let tu = tanh(clamp(u / p.cap_up, -15.0, 15.0));
        let f = p.cap_gate * tg * sig;
        let fprime = (1.0 - tg * tg) * sig + p.cap_gate * tg * sig * (1.0 - sig);
        dg = d * fprime * (p.cap_up * tu);
        du = d * f * (1.0 - tu * tu);
    }
    da[gi] = dg;
    if (p.mode == 0u) {
        db[ui] = du;
    } else {
        da[ui] = du;
    }
}
