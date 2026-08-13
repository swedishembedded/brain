// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse-MoE expert linear backward w.r.t. W: matmul_dw.wgsl, gated
// @how   one thread per output element, in-loop skip on the gate
// @opt   2
// @cpu   native
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Backward of moe_linear_gated.wgsl's `out = x @ W^T` w.r.t. W, for ONE
// expert. Bit-identical to the dense matmul_dw.wgsl over the SAME `dy`
// (never an approximation) for the same reason moe_linear_gated_dx.wgsl's
// doc gives: a non-routed row's `dy` is already exactly 0.0, so summing it
// in changes nothing -- skipping it only removes FLOPs.
//
// UNLIKE moe_linear_gated_dx.wgsl, the gated dimension here is the SUMMED
// (contraction) one: `dW[n, k] = sum_m dy[m, n] * x[m, k]`, and a single
// output element (n, k) draws from every row `m`, routed or not. There is no
// whole-thread early exit -- each thread must still visit every OTHER
// routed row for this expert, so a non-routed row is a loop `continue`, not
// a return.

struct Params {
    m: u32,
    k: u32,
    n: u32,
    n_experts: u32,
    e_idx: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:   array<f32>;
@group(0) @binding(2) var<storage, read>       x:    array<f32>;
@group(0) @binding(3) var<storage, read>       gate: array<f32>;
@group(0) @binding(4) var<storage, read_write> dw:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.n * p.k;
    if (idx >= total) { return; }
    let nn = idx / p.k;   // n
    let col = idx % p.k;  // k
    var acc = 0.0;
    for (var mm: u32 = 0u; mm < p.m; mm = mm + 1u) {
        if (gate[mm * p.n_experts + p.e_idx] <= 0.0) { continue; }
        acc = acc + dy[mm * p.n + nn] * x[mm * p.k + col];
    }
    dw[idx] = dw[idx] + acc;
}
