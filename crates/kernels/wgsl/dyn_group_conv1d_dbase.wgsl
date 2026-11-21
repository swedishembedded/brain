// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for training and serving models with short causal convolutions for its
// clients. If your team needs expertise in sequence-mixing convolutions then you can procure our services by sending an
// email to info@swedishembedded.com.

// @what  Backward of dyn_group_conv1d w.r.t. its static per-channel taps, accumulated
// @how   one thread per (tap, channel), serial sum over the rows
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Backward of dyn_group_conv1d.wgsl (same params and flags) w.r.t. `base`:
//
//   dbase[j,c] += sum_t [j <= t % block, and (FLAG_SEG off or seg[t-j] == seg[t])] gg[t,c] * x[t-j,c]
//
// with gg = dy * res_scale (dy without FLAG_RES_SCALE). The gradient is ADDED into `dbase` (pre-zeroed by the caller, like
// rmsnorm_dw), at `base_off` and in the layout the forward reads (`[taps, c]`, or `[c, taps]` with FLAG_BASE_CK).
// The residual path has no tap and contributes nothing here.
//
//   dy, x : [n, c]   dbase : the base tensor's gradient
//   bindings: 1 dy, 2 x, 3 dbase (read_write), 4 seg

const FLAG_RES_SCALE: u32 = 1u;
const FLAG_SEG: u32 = 2u;
const FLAG_BASE_CK: u32 = 16u;

struct Params {
    n: u32,
    c: u32,
    taps: u32,
    group_size: u32,
    groups: u32,
    dyn_stride: u32,
    dyn_off: u32,
    base_off: u32,
    block: u32,
    flags: u32,
    res_scale: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:    array<f32>;
@group(0) @binding(2) var<storage, read>       x:     array<f32>;
@group(0) @binding(3) var<storage, read_write> dbase: array<f32>;
@group(0) @binding(4) var<storage, read>       seg:   array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.taps * p.c) { return; }
    let j = idx / p.c;
    let ci = idx % p.c;
    var gs = 1.0;
    if ((p.flags & FLAG_RES_SCALE) != 0u) { gs = p.res_scale; }
    let use_seg = (p.flags & FLAG_SEG) != 0u;

    var acc = 0.0;
    for (var t: u32 = j; t < p.n; t = t + 1u) {
        if (t % p.block < j) { continue; }
        if (use_seg && seg[t - j] != seg[t]) { continue; }
        acc = acc + dy[t * p.c + ci] * gs * x[(t - j) * p.c + ci];
    }
    var at = p.base_off + j * p.c + ci;
    if ((p.flags & FLAG_BASE_CK) != 0u) { at = p.base_off + ci * p.taps + (p.taps - 1u - j); }
    dbase[at] = dbase[at] + acc;
}
