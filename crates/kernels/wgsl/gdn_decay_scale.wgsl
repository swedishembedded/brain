// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-row decay scale for Gated DeltaNet's state update: exp(g_cs_last - g_cs_i)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
//   decay_scale[bh,i] = exp(g_cs[g_cs_off + bh*c_len + (c_len-1)]
//                          - g_cs[g_cs_off + bh*c_len + i])
//
// The per-row factor `torch_chunk_gated_delta_rule`'s recurrent-state update
// scales `key` by (`k_c * exp(g_cs_c[-1] - g_cs_c)`) before accumulating
// `decayed_k^T @ v_new` into the state. Computed as `exp(a - b)` directly —
// never as `exp(a) / exp(b)` — because `g_cs` is a cumulative SUM of
// log-decays over a whole chunk and can be very negative, so a separately
// materialised `exp(g_cs)` (`exp.wgsl`'s own output buffer) can underflow to
// 0 long before the DIFFERENCE does; dividing by that 0 would be wrong where
// this direct form is exact. This is why `gdn.rs` does NOT reuse the
// `exp_g_cs` buffer here despite it existing for other consumers — see that
// module's doc for the full scratch-vs-recompute tradeoff.
//
// `g_cs` is `[bhc, c_len]` row-major; `g_cs_off` selects one chunk's
// contiguous `[bh, c_len]` slice (`bh = B*H`) out of the full `[bhc, c_len]`
// buffer — the same chunk-selection idiom as `bmm.wgsl`'s `*_off` params.
// Output `decay_scale` is a fresh dense `[bh, c_len]` buffer (no offset).

struct Params { bh: u32, c_len: u32, g_cs_off: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       g_cs: array<f32>;
@group(0) @binding(2) var<storage, read_write> decay_scale: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.bh * p.c_len) { return; }
    let bh = idx / p.c_len;
    let i = idx % p.c_len;
    let base = p.g_cs_off + bh * p.c_len;
    decay_scale[idx] = exp(g_cs[base + p.c_len - 1u] - g_cs[base + i]);
}
