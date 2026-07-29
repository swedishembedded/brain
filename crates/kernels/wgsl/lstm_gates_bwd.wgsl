// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Backward of lstm_gates.wgsl. Given upstream grads dh (wrt h_out) and dc_next
// (wrt c_out, from the next timestep), the pre-activations `pre`, previous cell
// `c_prev` and new cell `c_out`, produce the grad wrt the pre-activations and wrt
// the previous cell state:
//   tc   = tanh(c);   do = dh*tc;   dc = dc_next + dh*o*(1-tc^2)
//   di = dc*g;  df = dc*c_prev;  dg = dc*i;  d_cprev = dc*f
//   d_pre[i] = di*i*(1-i);  d_pre[f] = df*f*(1-f);
//   d_pre[g] = dg*(1-g^2);  d_pre[o] = do*o*(1-o)
// i,f,g,o recomputed from `pre`. One thread per (row, unit).

struct Params {
    rows: u32,
    h: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dh:      array<f32>;
@group(0) @binding(2) var<storage, read>       dc_next: array<f32>;
@group(0) @binding(3) var<storage, read>       pre:     array<f32>;
@group(0) @binding(4) var<storage, read>       c_prev:  array<f32>;
@group(0) @binding(5) var<storage, read>       c_out:   array<f32>;
@group(0) @binding(6) var<storage, read_write> d_pre:   array<f32>;
@group(0) @binding(7) var<storage, read_write> d_cprev: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.rows * p.h;
    if (idx >= total) { return; }
    let r = idx / p.h;
    let j = idx % p.h;
    let b = r * 4u * p.h;
    let ii = 1.0 / (1.0 + exp(-pre[b + j]));
    let ff = 1.0 / (1.0 + exp(-pre[b + p.h + j]));
    let gg = tanh(pre[b + 2u * p.h + j]);
    let oo = 1.0 / (1.0 + exp(-pre[b + 3u * p.h + j]));
    let tc = tanh(c_out[idx]);
    let doo = dh[idx] * tc;
    let dc = dc_next[idx] + dh[idx] * oo * (1.0 - tc * tc);
    let di = dc * gg;
    let df = dc * c_prev[idx];
    let dg = dc * ii;
    d_cprev[idx] = dc * ff;
    d_pre[b + j] = di * ii * (1.0 - ii);
    d_pre[b + p.h + j] = df * ff * (1.0 - ff);
    d_pre[b + 2u * p.h + j] = dg * (1.0 - gg * gg);
    d_pre[b + 3u * p.h + j] = doo * oo * (1.0 - oo);
}
