// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Fused LSTM cell gate activation (PyTorch nn.LSTM layout). Given the summed
// pre-activations `pre = W_ih·x + b_ih + W_hh·h + b_hh` (computed with matmul) and
// the previous cell state, produce the new cell and hidden states.
//   pre    : [rows, 4H]  gate order i, f, g, o (PyTorch)
//   c_prev : [rows, H]
//   c_out  : [rows, H]   c = f*c_prev + i*g
//   h_out  : [rows, H]   h = o*tanh(c)
// with i=sigmoid, f=sigmoid, g=tanh, o=sigmoid. One thread per (row, unit).

struct Params {
    rows: u32,
    h: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pre:    array<f32>;
@group(0) @binding(2) var<storage, read>       c_prev: array<f32>;
@group(0) @binding(3) var<storage, read_write> c_out:  array<f32>;
@group(0) @binding(4) var<storage, read_write> h_out:  array<f32>;

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
    let ct = ff * c_prev[idx] + ii * gg;
    c_out[idx] = ct;
    h_out[idx] = oo * tanh(ct);
}
