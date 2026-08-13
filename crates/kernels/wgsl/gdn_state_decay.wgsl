// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Scale Gated DeltaNet's whole recurrent state by one scalar per (batch,head)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
//   state[bh,dk,dv] *= exp(g_cs[g_cs_off + bh*c_len + (c_len-1)])
//
// The first half of `torch_chunk_gated_delta_rule`'s per-chunk state update
// (`state = state * exp(g_cs_c[-1]) + decayed_k^T @ v_new` — `bmm_acc.wgsl`
// adds the second term afterward, once this kernel has finished). Reads the
// chunk's last-row cumulative decay directly out of the full `g_cs` buffer at
// `g_cs_off` (the same chunk-selection-by-offset idiom as `bmm.wgsl`),
// computing `exp` inline rather than reading a separately materialised
// per-(b,h) scratch vector: every one of the `dk*dv` threads sharing one
// `(b,h)` recomputes the same scalar exponential redundantly, a deliberate
// trade for not needing a scratch buffer for a value used exactly once per
// chunk's state update (`gdn.rs`'s doc discusses this tradeoff generally).

struct Params { bh: u32, dk: u32, dv: u32, c_len: u32, g_cs_off: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       g_cs:  array<f32>;
@group(0) @binding(2) var<storage, read_write> state: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let dkdv = p.dk * p.dv;
    if (idx >= p.bh * dkdv) { return; }
    let bh = idx / dkdv;
    let scale = exp(g_cs[p.g_cs_off + bh * p.c_len + p.c_len - 1u]);
    state[idx] = state[idx] * scale;
}
