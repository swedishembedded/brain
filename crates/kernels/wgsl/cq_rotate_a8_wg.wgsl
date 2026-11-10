// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for running aggressively quantised networks on constrained devices for its
// clients. If your team needs expertise in rotated-codebook weight formats and int8 activation paths then you can procure
// our services by sending an email to info@swedishembedded.com.

// @what  Orthonormal natural-order Walsh transform of every 128-lane group of f32 activation rows followed by int8 quantisation with one scale per group - the cooperative form of cq_rotate_a8
// @how   one 64-thread workgroup per (row, group): the group lives in workgroup memory, seven butterfly stages behind barriers, a tree absmax, then each thread rounds two lanes and four threads' bytes pack into a word
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant cq
// @dtype f32
//
// The decode-regime sibling of `cq_rotate_a8` (see `cq_rotate_wg` for why the groups of a row are separate workgroups).
// Same operands, layout and result as `cq_rotate_a8`: the maximum is order independent, so the scale and every byte
// equal the barrier-free kernel's.
//
//   params : m, k, kpad, xr0
//   x      : [xr0 + m, k] f32
//   q      : [m, kpad / 4] u32
//   s      : [m, kpad / 128] f32

struct Params {
    m: u32,
    k: u32,
    kpad: u32,
    xr0: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> q: array<u32>;
@group(0) @binding(3) var<storage, read_write> s: array<f32>;

const GROUP: u32 = 128u;
// The f32 nearest 1/127 (bits 0x3c010204), the engine's scale step.
const INV_127: f32 = 0.007874016;

var<workgroup> lane: array<f32, 128>;
var<workgroup> reduce: array<f32, 64>;
var<workgroup> bytes: array<u32, 128>;

// Round half away from zero, exact for every f32 (floor(|v| + 0.5) is not: it rounds 0.49999997 up).
fn round_away(v: f32) -> f32 {
    let a = abs(v);
    let t = trunc(a);
    let r = t + select(0.0, 1.0, a - t >= 0.5);
    return select(r, -r, v < 0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = li.x;
    let groups = p.kpad / GROUP;
    let id = wg.y * nwg.x + wg.x;
    let row = id / groups;
    let g = id % groups;
    let base = g * GROUP;
    let live = row < p.m;
    for (var j = t; j < GROUP; j = j + 64u) {
        var v = 0.0;
        if (live && base + j < p.k) { v = x[(p.xr0 + row) * p.k + base + j]; }
        lane[j] = v;
    }
    workgroupBarrier();
    for (var h = 1u; h < GROUP; h = h * 2u) {
        let a = (t / h) * 2u * h + (t % h);
        let l = lane[a];
        let r = lane[a + h];
        workgroupBarrier();
        lane[a] = l + r;
        lane[a + h] = l - r;
        workgroupBarrier();
    }
    let v0 = lane[t] * 0.08838834764831845;
    let v1 = lane[t + 64u] * 0.08838834764831845;
    reduce[t] = max(abs(v0), abs(v1));
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (t < stride) { reduce[t] = max(reduce[t], reduce[t + stride]); }
        workgroupBarrier();
    }
    let absmax = reduce[0];
    var scale = 1.0;
    if (absmax > 0.0) { scale = absmax * INV_127; }
    let inv = 1.0 / scale;
    bytes[t] = bitcast<u32>(i32(clamp(round_away(v0 * inv), -128.0, 127.0))) & 0xFFu;
    bytes[t + 64u] = bitcast<u32>(i32(clamp(round_away(v1 * inv), -128.0, 127.0))) & 0xFFu;
    workgroupBarrier();
    if (live) {
        if (t < 32u) {
            q[(row * p.kpad + base) / 4u + t] = bytes[t * 4u] | (bytes[t * 4u + 1u] << 8u) | (bytes[t * 4u + 2u] << 16u) | (bytes[t * 4u + 3u] << 24u);
        }
        if (t == 0u) { s[row * groups + g] = scale; }
    }
}
