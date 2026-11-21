// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for training and serving models with short causal convolutions for its
// clients. If your team needs expertise in sequence-mixing convolutions then you can procure our services by sending an
// email to info@swedishembedded.com.

// @what  Backward of dyn_group_conv1d w.r.t. its input x (the residual path included)
// @how   one thread per (source row, channel), serial sum over the K taps that read it
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Backward of dyn_group_conv1d.wgsl (same params, buffers and flags) w.r.t. x. With gg[t,c] = dy[t,c] * res_scale
// (or dy when FLAG_RES_SCALE is off) the gradient of source row s is the sum over the taps j that read it:
//
//   dx[s,c] = sum_j [row t = s+j is in s's block, and (FLAG_SEG off or seg[t] == seg[s])] gg[t,c] * w(t,j,c)
//             + (FLAG_RES_SCALE ? gg[s,c] : 0)
//   w(t,j,c) = base tap j of c + (FLAG_NO_DYN off ? dyn[t, j, group(c)] : 0)
//
// A carried history (FLAG_HIST) is an input of the forward only and receives no gradient: training does not carry one.
//
//   dy, dx : [n, c]     x is not read.
//   bindings: 1 dy, 2 dyn, 3 base, 4 dx (read_write), 5 seg

const FLAG_RES_SCALE: u32 = 1u;
const FLAG_SEG: u32 = 2u;
const FLAG_NO_DYN: u32 = 8u;
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
@group(0) @binding(1) var<storage, read>       dy:   array<f32>;
@group(0) @binding(2) var<storage, read>       dyn:  array<f32>;
@group(0) @binding(3) var<storage, read>       base: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx:   array<f32>;
@group(0) @binding(5) var<storage, read>       seg:  array<u32>;

fn grad_scale() -> f32 {
    if ((p.flags & FLAG_RES_SCALE) != 0u) { return p.res_scale; }
    return 1.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n * p.c) { return; }
    let s = idx / p.c;
    let ci = idx % p.c;
    let g = ci / p.group_size;
    let gs = grad_scale();
    let use_seg = (p.flags & FLAG_SEG) != 0u;

    var acc = 0.0;
    for (var j: u32 = 0u; j < p.taps; j = j + 1u) {
        let t = s + j;
        if (t >= p.n) { break; }
        // The reader row t must be at least j rows into its block, i.e. s is in the same block.
        if (t % p.block < j) { continue; }
        if (use_seg && seg[t] != seg[s]) { continue; }
        var w = 0.0;
        if ((p.flags & FLAG_BASE_CK) != 0u) { w = base[p.base_off + ci * p.taps + (p.taps - 1u - j)]; }
        else { w = base[p.base_off + j * p.c + ci]; }
        if ((p.flags & FLAG_NO_DYN) == 0u) { w = w + dyn[t * p.dyn_stride + p.dyn_off + j * p.groups + g]; }
        acc = acc + dy[t * p.c + ci] * gs * w;
    }
    if ((p.flags & FLAG_RES_SCALE) != 0u) { acc = acc + dy[idx] * gs; }
    dx[idx] = acc;
}
