// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse-MoE expert linear: matmul.wgsl, but skips non-routed rows
// @how   one thread per output element, serial inner reduction, early exit
// @opt   2
// @cpu   native
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Same contract as matmul.wgsl (`out = x @ W^T`) EXCEPT a row whose gate
// weight for this expert is zero writes 0 and returns before the K-reduction,
// instead of computing it and relying on a later multiply-by-zero to discard
// it. That is the entire difference from evaluating every expert densely
// (`crates/glm/src/model.rs`'s `Mlp::Moe` arm, whose own `scale_add.wgsl`
// combine step already tolerates a garbage non-selected row because it always
// computed a real, finite one) — here the row is genuinely never reduced, so
// the FLOP cost of an expert's gate/up/down projections scales with the
// number of rows actually routed to it, not with `seq_len`.
//
// One kernel serves gate_proj, up_proj AND down_proj: they differ only in
// (k, n) and which expert's weight is bound, not in this row-gating logic.
// The `gate` buffer stays the dense `[m, n_experts]` matrix `router_gate.wgsl`
// already produces — down_proj re-reads the SAME row's gate value rather than
// inspecting whether its input `h` row is exactly zero, so all three
// projections agree on which rows are live even under float rounding.
//
//   x    : [m, k]           row-major
//   w    : [n, k]           row-major (w[j, l] is weight row j)
//   gate : [m, n_experts]   dense per-token-per-expert weight (0 = not routed)
//   out  : [m, n]           row-major; out[row, :] = 0 for a non-routed row

struct Params {
    m: u32,
    k: u32,
    n: u32,
    n_experts: u32,
    e_idx: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read>       w:    array<f32>;
@group(0) @binding(3) var<storage, read>       gate: array<f32>;
@group(0) @binding(4) var<storage, read_write> out:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.m * p.n;
    if (idx >= total) { return; }
    let row = idx / p.n;
    let col = idx % p.n;
    if (gate[row * p.n_experts + p.e_idx] <= 0.0) {
        out[idx] = 0.0;
        return;
    }
    let x_base = row * p.k;
    let w_base = col * p.k;
    var acc = 0.0;
    for (var i: u32 = 0u; i < p.k; i = i + 1u) {
        acc = acc + x[x_base + i] * w[w_base + i];
    }
    out[idx] = acc;
}
