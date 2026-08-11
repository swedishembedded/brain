// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward of Gated DeltaNet's whole-state decay scalar
// @how   one thread per (batch,head), serial reduction over a Params-bounded axis
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Backward of `gdn_state_decay.wgsl`'s `state[bh,dk,dv] *= decay_last` where
// `decay_last = exp(g_cs[g_cs_off + bh*c_len + c_len-1])`. Two gradients flow
// out of this multiply: `d_state_in[p,q] = d_state[p,q] * decay_last` (a
// scale by the SAME scalar `gdn_state_decay.wgsl` itself applies, so that
// kernel is reused unchanged as its own backward for this half — see
// `model::gdn::gdn_chunk_bwd`'s doc); and the scalar `decay_last`'s own
// gradient, which THIS kernel computes:
//   d_decay_last = sum_{p,q} d_state[p,q] * state_in[p,q]
// then, via `decay_last = exp(g_cs[...])`, chained straight into a single
// cell of `g_cs`'s gradient (no separate multiply-by-`exp` step needed since
// `decay_last` IS that exp, already available from `g_cs` directly):
//   d_g_cs[g_cs_off + bh*c_len + c_len-1] += d_decay_last * decay_last
//
// One thread per `(b,h)`, serial loop over `dk*dv` (correctness-first tier —
// a cooperative reduction would help at real model scale, but `dk`/`dv` are
// tens to low hundreds here, matching every
// other GDN reduction's tier). `state_in` is the forward-saved
// PRE-this-chunk's-update state (`GdnScratchTrain::state_history` at index
// `ci`, i.e. `state_history[ci]` — NOT `state_history[ci+1]`, which already
// has this chunk's decay/update baked in) — `state_off` selects that slice
// (`ci * bh*dk*dv`), a DIFFERENT stride than `g_cs_off` (`ci * bh*c_len`)
// since `state_history` and `g_cs` have unrelated per-chunk element counts;
// two offset params, not one shared like `gdn_decay_scale_bwd.wgsl`'s
// `g_cs_off` (which works there only because `decay_scale`'s history shares
// `g_cs`'s own `[bhc,c_len]` shape). `d_state` is the caller's running
// gradient BEFORE this chunk's backward touches it (the "cur" half of
// `gdn_chunk_bwd`'s ping-pong state accumulator), a dedicated `[bh,dk,dv]`
// buffer needing no offset of its own. `d_g_cs` is a genuine multi-source
// accumulator — see `gdn_decay_mask_bwd.wgsl`'s header for the full
// contributor list; zero it once, before any contributor runs.
//
// Dispatch: `threads = bh`.

struct Params { bh: u32, dk: u32, dv: u32, c_len: u32, g_cs_off: u32, state_off: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_state:  array<f32>;
@group(0) @binding(2) var<storage, read>       state_in: array<f32>;
@group(0) @binding(3) var<storage, read>       g_cs:     array<f32>;
@group(0) @binding(4) var<storage, read_write> d_g_cs:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let bh = gid.y * (nwg.x * 64u) + gid.x;
    if (bh >= p.bh) { return; }
    let dkdv = p.dk * p.dv;
    let base = bh * dkdv;
    let state_base = p.state_off + base;
    var acc = 0.0;
    var t: u32 = 0u;
    loop {
        if (t >= dkdv) { break; }
        acc = acc + d_state[base + t] * state_in[state_base + t];
        t = t + 1u;
    }
    let last = p.g_cs_off + bh * p.c_len + p.c_len - 1u;
    let decay_last = exp(g_cs[last]);
    d_g_cs[last] = d_g_cs[last] + acc * decay_last;
}
