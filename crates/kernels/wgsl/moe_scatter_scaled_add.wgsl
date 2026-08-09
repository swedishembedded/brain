// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Scatter a compacted expert's output back into the dense MoE accumulator, gate-scaled
// @how   one thread per output element, scatter-add (no atomics: rows are unique per call)
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// The scatter half of row-compacted sparse MoE (see `model::moe`'s
// `expert_fwd_compact`): `acc[idx[i], c] += gate[idx[i], e_idx] * src[i, c]`
// (or `=` instead of `+=` when `accumulate` is 0, matching `scale_add.wgsl`'s
// own set-vs-add contract for the first expert in a layer's loop).
//
// `idx` names `n_idx` DISTINCT rows of `acc` (this expert's compacted-batch
// membership, computed host-side — see the module doc for why: WGSL kernels
// here may not use atomics, and a top-k>1 row can receive a `+=` from
// SEVERAL DIFFERENT experts' calls, so uniqueness only needs to hold WITHIN
// one call, not across the whole layer). `gate` stays the FULL dense
// `[m_full, n_experts]` matrix `router_gate.wgsl` already produces — reading
// `gate[idx[i], e_idx]` here (not a pre-scaled `src`) means the caller never
// needs a compacted gate buffer, only the compacted row-index and
// expert-output buffers `moe_gather`'s `embed.wgsl` reuse already produces.

struct Params {
    n_idx: u32,
    d: u32,
    n_experts: u32,
    e_idx: u32,
    accumulate: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       idx:  array<u32>;
@group(0) @binding(2) var<storage, read>       gate: array<f32>;
@group(0) @binding(3) var<storage, read>       src:  array<f32>;
@group(0) @binding(4) var<storage, read_write> out:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.n_idx * p.d;
    if (gidx >= total) { return; }
    let i = gidx / p.d;
    let c = gidx % p.d;
    let row = idx[i];
    let g = gate[row * p.n_experts + p.e_idx];
    let val = g * src[i * p.d + c];
    if (p.accumulate != 0u) {
        out[row * p.d + c] = out[row * p.d + c] + val;
    } else {
        out[row * p.d + c] = val;
    }
}
