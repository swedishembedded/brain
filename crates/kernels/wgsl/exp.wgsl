// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Elementwise exponential
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Elementwise exponential:  y[i] = exp(x[i]).
// Added for Gated DeltaNet's `exp(cumulative log-decay)` terms
// (`torch_chunk_gated_delta_rule`'s `g.cumsum(-1).exp()`): materialised once
// into its own buffer and reused everywhere the forward needs `exp(g_cs)`
// alone, rather than recomputed per consumer — see `crates/model/src/gdn.rs`'s
// module doc for which consumers reuse this buffer and which instead compute
// an exp-of-a-DIFFERENCE inline (to avoid dividing by a possibly-underflowed
// `exp(g_cs)`).

struct Params { total: u32, };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    out[idx] = exp(x[idx]);
}
