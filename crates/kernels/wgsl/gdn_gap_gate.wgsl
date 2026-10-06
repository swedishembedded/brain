// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gated DeltaNet gates over irregular visits: decay set by the physical time since the previous token, write strength beta
// @how   one thread per (row, head) element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The gap-aware sibling of `gdn_decay_gate`: where that kernel's decay is a
// learned function of the token (a step size the model chooses), here the
// time between tokens is DATA, so the decay is the exponential of the time
// that actually passed, exactly as a continuous-time state forgets:
//   g[row,h]    = -softplus(rate[h]) * dt[row]        (raw log-decay, <= 0)
//   beta[row,h] = sigmoid(b_pre[row,h])               (write strength)
// A row with dt[row] < 0 is padding (an unused visit slot, a row past the
// sequence): it neither decays nor writes, g = 0 and beta = 0, so the
// recurrence passes the state through it unchanged.
// `rate` is the per-head raw rate, one per unit of time in `dt`.
//
// `softplus` is the stable shifted form (see `gdn_decay_gate`).

struct Params {
    rows: u32,
    heads: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       b_pre: array<f32>;
@group(0) @binding(2) var<storage, read>       dt:    array<f32>;
@group(0) @binding(3) var<storage, read>       rate:  array<f32>;
@group(0) @binding(4) var<storage, read_write> g:     array<f32>;
@group(0) @binding(5) var<storage, read_write> beta:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.heads) { return; }
    let row = idx / p.heads;
    let h = idx % p.heads;
    let gap = dt[row];
    if (gap < 0.0) {
        g[idx] = 0.0;
        beta[idx] = 0.0;
        return;
    }
    let x = rate[h];
    let softplus = max(x, 0.0) + log(1.0 + exp(-abs(x)));
    g[idx] = -softplus * gap;
    beta[idx] = 1.0 / (1.0 + exp(-b_pre[idx]));
}
