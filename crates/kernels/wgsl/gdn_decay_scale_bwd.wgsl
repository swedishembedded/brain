// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward of Gated DeltaNet's per-row state-update decay scale, elementwise half
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Backward of `gdn_decay_scale.wgsl`:
//   decay_scale[bh,i] = exp(g_cs[bh,c_len-1] - g_cs[bh,i])
// The `-g_cs[bh,i]` term's gradient is the elementwise half of this kernel's
// job (the `+g_cs[bh,c_len-1]` term's gradient is a REDUCTION into a single
// cell per `(b,h)` — a single thread cannot do both without either racing on
// the last cell or leaving every other cell undone, so that half is a
// separate kernel, `gdn_decay_scale_bwd_last.wgsl`):
//   d_g_cs[g_cs_off + bh*c_len + i] -= d_decay_scale[bh,i] * decay_scale[bh,i]
//
// `d_decay_scale` is the caller's freshly computed (this chunk, `[bh,c_len]`
// dense) gradient of `decay_scale`'s row-scale use in
// `decayed_k = key_c * decay_scale` (`gdn_row_scale_off.wgsl`'s own
// backward, a `row_dot.wgsl` call). `decay_scale` here is the FORWARD value,
// read from `gdn_chunk_fwd_train`'s saved per-chunk history
// (`GdnScratchTrain::decay_scale_hist`, promoted to `[bhc,c_len]`) at this
// chunk's own `g_cs_off` slice — NOT recomputed, since it was already
// computed once by the forward pass and materialised precisely so backward
// does not need `state_history`/`g_cs` to recompute it a second time.
// `d_g_cs` is a genuine multi-source accumulator (see
// `gdn_decay_mask_bwd.wgsl`'s header for the full list of contributors) —
// the caller must zero it exactly once, before ANY of its contributors run.
//
// Dispatch: `threads = bh * c_len`.

struct Params { bh: u32, c_len: u32, g_cs_off: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_decay_scale: array<f32>;
@group(0) @binding(2) var<storage, read>       decay_scale:   array<f32>;
@group(0) @binding(3) var<storage, read_write> d_g_cs:        array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.bh * p.c_len) { return; }
    let cell = p.g_cs_off + idx;
    d_g_cs[cell] = d_g_cs[cell] - d_decay_scale[idx] * decay_scale[cell];
}
