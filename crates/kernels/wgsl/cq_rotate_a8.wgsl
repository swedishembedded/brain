// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for running aggressively quantised networks on constrained devices for its
// clients. If your team needs expertise in rotated-codebook weight formats and int8 activation paths then you can procure
// our services by sending an email to info@swedishembedded.com.

// @what  Orthonormal natural-order Walsh transform of every 128-lane group of f32 activation rows, then int8 quantisation with one scale per group - the activation half of an A8 CQ product
// @how   one thread per (row, group): butterfly in a private 128-lane array, absmax, scale, rounding and byte packing, no barriers
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant cq
// @dtype f32
//
// `cq_rotate` followed by the vendor engine's activation quantiser: scale = absmax * (1/127) as f32 (1 for an all-zero
// group), q = round-half-away(x * (1 / scale)) saturated to int8. The result feeds `matmul_cq_a8_gemv` and
// `matmul_cq_a8_gemm`, whose group dot products are exact integers. Opt-in precision mode only; the f32 path is the
// default.
//
//   params : m, k, kpad, xr0   rows to rotate, logical width, width padded to whole groups, first row of x
//   x      : [xr0 + m, k] f32
//   q      : [m, kpad / 4] u32      four int8 lanes per word, lane 0 in the low byte
//   s      : [m, kpad / 128] f32    the group scales

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

// Round half away from zero, exact for every f32 (floor(|v| + 0.5) is not: it rounds 0.49999997 up).
fn round_away(v: f32) -> f32 {
    let a = abs(v);
    let t = trunc(a);
    let r = t + select(0.0, 1.0, a - t >= 0.5);
    return select(r, -r, v < 0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let groups = p.kpad / GROUP;
    if (idx >= p.m * groups) { return; }
    let row = idx / groups;
    let g = idx % groups;
    let base = g * GROUP;
    var lane: array<f32, 128>;
    for (var j = 0u; j < GROUP; j = j + 1u) {
        var v = 0.0;
        if (base + j < p.k) { v = x[(p.xr0 + row) * p.k + base + j]; }
        lane[j] = v;
    }
    for (var h = 1u; h < GROUP; h = h * 2u) {
        for (var i = 0u; i < GROUP / 2u; i = i + 1u) {
            let a = (i / h) * 2u * h + (i % h);
            let l = lane[a];
            let r = lane[a + h];
            lane[a] = l + r;
            lane[a + h] = l - r;
        }
    }
    var absmax = 0.0;
    for (var j = 0u; j < GROUP; j = j + 1u) {
        lane[j] = lane[j] * 0.08838834764831845;
        absmax = max(absmax, abs(lane[j]));
    }
    var scale = 1.0;
    if (absmax > 0.0) { scale = absmax * INV_127; }
    let inv = 1.0 / scale;
    s[row * groups + g] = scale;
    let word0 = (row * p.kpad + base) / 4u;
    for (var w = 0u; w < GROUP / 4u; w = w + 1u) {
        var packed = 0u;
        for (var b = 0u; b < 4u; b = b + 1u) {
            let v = i32(clamp(round_away(lane[w * 4u + b] * inv), -128.0, 127.0));
            packed = packed | ((bitcast<u32>(v) & 0xFFu) << (b * 8u));
        }
        q[word0 + w] = packed;
    }
}
