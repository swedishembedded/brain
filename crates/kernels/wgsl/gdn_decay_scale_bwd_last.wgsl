// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward of Gated DeltaNet's per-row state-update decay scale, last-row reduction half
// @how   one thread per (batch,head), serial reduction over a Params-bounded axis
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The other half of `gdn_decay_scale_bwd.wgsl` (see that file's header for the
// forward formula and the full split rationale): the `+g_cs[bh,c_len-1]` term
// of `decay_scale[bh,i] = exp(g_cs[bh,c_len-1] - g_cs[bh,i])` feeds EVERY `i`
// in the chunk through the same shared scalar, so its gradient is a
// reduction over the whole row, landing in a single cell:
//   d_g_cs[g_cs_off + bh*c_len + c_len-1] +=
//       sum_{i=0}^{c_len-1} d_decay_scale[bh,i] * decay_scale[bh,i]
//
// One thread per `(b,h)`, serial loop over `c_len` (at most a few dozen for
// this model family — correctness-first tier, matching
// `gdn_state_decay_bwd_dscale.wgsl`'s identical shape). `decay_scale` is the
// forward-saved value (`GdnScratchTrain::decay_scale_hist`), read at this
// chunk's `g_cs_off` slice, same as `gdn_decay_scale_bwd.wgsl`. `d_g_cs` is a
// genuine multi-source accumulator — see `gdn_decay_mask_bwd.wgsl`'s header
// for the full contributor list; zero it once, before any contributor runs.
//
// Dispatch: `threads = bh`.

struct Params { bh: u32, c_len: u32, g_cs_off: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_decay_scale: array<f32>;
@group(0) @binding(2) var<storage, read>       decay_scale:   array<f32>;
@group(0) @binding(3) var<storage, read_write> d_g_cs:        array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let bh = gid.y * (nwg.x * 64u) + gid.x;
    if (bh >= p.bh) { return; }
    let base = bh * p.c_len;
    var acc = 0.0;
    var i: u32 = 0u;
    loop {
        if (i >= p.c_len) { break; }
        let cell = p.g_cs_off + base + i;
        acc = acc + d_decay_scale[base + i] * decay_scale[cell];
        i = i + 1u;
    }
    let last = p.g_cs_off + base + p.c_len - 1u;
    d_g_cs[last] = d_g_cs[last] + acc;
}
