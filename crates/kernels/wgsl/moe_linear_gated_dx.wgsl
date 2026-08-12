// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse-MoE expert linear backward w.r.t. x: matmul_dx.wgsl, gated
// @how   one thread per output element, pre-reduction early exit on the gate
// @opt   2
// @cpu   native
// @gpu   yes
// @npu   no
// @quant none
//
// Backward of moe_linear_gated.wgsl's `out = x @ W^T` w.r.t. x, for ONE
// expert. Bit-identical to the dense matmul_dx.wgsl over the SAME `dy`
// (never an approximation): with the gated forward, a non-routed row's `dy`
// is already exactly 0.0 end to end (`d_expert_out`'s own scale_add_dexp
// write is gated the same way), so skipping that row's reduction changes
// nothing about the sum -- it only removes FLOPs matmul_dx would spend
// computing an exact zero.
//
// The output row IS the token row here (dX is `[m, k]`, same shape as the
// forward's `x`), so a non-routed row can be resolved with a single
// pre-reduction check -- unlike moe_linear_gated_dw.wgsl, where the gated
// dimension is the summed (contraction) one, not the output one.
//
// `accumulate` mirrors matmul_dx.wgsl's contract, WITH the gated asymmetry
// the forward's early exit demands: accumulate=1 on a non-routed row must
// leave dx untouched (adding an exact zero is a no-op -- touching the buffer
// at all would be a spurious write, not a correctness bug, but skipping it
// is the entire point of gating); accumulate=0 must still zero it (the first
// writer for this row establishes the value).

struct Params {
    m: u32,
    k: u32,
    n: u32,
    n_experts: u32,
    e_idx: u32,
    accumulate: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:   array<f32>;
@group(0) @binding(2) var<storage, read>       w:    array<f32>;
@group(0) @binding(3) var<storage, read>       gate: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.m * p.k;
    if (idx >= total) { return; }
    let row = idx / p.k;   // m
    let col = idx % p.k;   // k
    if (gate[row * p.n_experts + p.e_idx] <= 0.0) {
        if (p.accumulate == 0u) { dx[idx] = 0.0; }
        return;
    }
    var acc = 0.0;
    for (var nn: u32 = 0u; nn < p.n; nn = nn + 1u) {
        acc = acc + dy[row * p.n + nn] * w[nn * p.k + col];
    }
    if (p.accumulate == 0u) { dx[idx] = acc; }
    else                    { dx[idx] = dx[idx] + acc; }
}
