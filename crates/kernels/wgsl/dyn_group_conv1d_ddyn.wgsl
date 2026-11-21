// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for training and serving models with short causal convolutions for its
// clients. If your team needs expertise in sequence-mixing convolutions then you can procure our services by sending an
// email to info@swedishembedded.com.

// @what  Backward of dyn_group_conv1d w.r.t. the per-token taps (the dynamic projection's output)
// @how   one thread per (row, tap, group), serial sum over the group's channels
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Backward of dyn_group_conv1d.wgsl (same params and flags, FLAG_NO_DYN off) w.r.t. `dyn`:
//
//   ddyn[t, j, g] = [j <= t % block, and (FLAG_SEG off or seg[t-j] == seg[t])] * sum_{c in group g} gg[t,c] * x[t-j,c]
//
// with gg = dy * res_scale (dy without FLAG_RES_SCALE). The gradient of the `taps * groups` window at `dyn_off` of
// row t is OVERWRITTEN (other columns of the row, e.g. a second convolution's window, are not touched), at
// `ddyn[t * dyn_stride + dyn_off + j * groups + g]`. Feeding the projection's own dX / dW is the caller's.
//
//   dy, x : [n, c]   ddyn : [n, dyn_stride]
//   bindings: 1 dy, 2 x, 3 ddyn (read_write), 4 seg

const FLAG_RES_SCALE: u32 = 1u;
const FLAG_SEG: u32 = 2u;

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
@group(0) @binding(1) var<storage, read>       dy:   array<f32>;
@group(0) @binding(2) var<storage, read>       x:    array<f32>;
@group(0) @binding(3) var<storage, read_write> ddyn: array<f32>;
@group(0) @binding(4) var<storage, read>       seg:  array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let per_row = p.taps * p.groups;
    if (idx >= p.n * per_row) { return; }
    let t = idx / per_row;
    let r = idx % per_row;
    let j = r / p.groups;
    let g = r % p.groups;
    var gs = 1.0;
    if ((p.flags & FLAG_RES_SCALE) != 0u) { gs = p.res_scale; }

    var acc = 0.0;
    let reach = j <= t % p.block;
    if (reach && ((p.flags & FLAG_SEG) == 0u || seg[t - j] == seg[t])) {
        for (var k: u32 = 0u; k < p.group_size; k = k + 1u) {
            let ci = g * p.group_size + k;
            acc = acc + dy[t * p.c + ci] * gs * x[(t - j) * p.c + ci];
        }
    }
    ddyn[t * p.dyn_stride + p.dyn_off + r] = acc;
}
