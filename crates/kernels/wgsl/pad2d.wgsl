// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Asymmetric constant- or edge-pad on NCHW (gather) - spec
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Asymmetric pad on NCHW (gather). Params carry the UNPADDED
// dims h, w and pad amounts l/r/t/b (left/right/top/bottom, u32, each may be
// 0); padded dims are derived: hp = h+t+b, wp = w+l+r. Batch and channel are
// combined (p = image index). `mode` picks the border (word 8), `value` is the
// constant fill as f32 bits (word 9, read with bitcast). One thread per OUTPUT
// element idx < total (total = NC*hp*wp):
//   p = idx/(hp*wp); r0 = idx % (hp*wp); ho = r0/wp; wo = r0 % wp   (u32 ops)
//   inside = ho >= t && ho < t+h && wo >= l && wo < l+w
//   mode 0 (constant): y[idx] = inside ? x[p*h*w + (ho-t)*w + (wo-l)] : value
//   mode 1 (edge):     the source row/col is the output row/col clamped into
//                      [t, t+h-1] / [l, l+w-1]; every border pixel replicates
//                      the nearest edge pixel (value is ignored)
// With mode 0 and value 0 this is the linear map y = P x (interior injection);
// crop2d with the SAME offsets is then its exact transpose/adjoint AND its
// backward: dx = crop2d(dy). Required: <pad2d(x), y> == <x, crop2d(y)>;
// crop2d(pad2d(x)) == x bitwise (for every mode and value, the interior is x).
// A nonzero value or mode 1 makes the map affine, and crop2d is then only the
// interior's inverse, not an adjoint.
//

struct Params {
    total: u32,
    h: u32,
    w: u32,
    l: u32,
    r: u32,
    t: u32,
    b: u32,
    mode: u32,
    value: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let hp = p.h + p.t + p.b;
    let wp = p.w + p.l + p.r;
    let img = idx / (hp * wp);
    let r0 = idx % (hp * wp);
    let ho = r0 / wp;
    let wo = r0 % wp;
    if (ho >= p.t && ho < p.t + p.h && wo >= p.l && wo < p.l + p.w) {
        y[idx] = x[img * p.h * p.w + (ho - p.t) * p.w + (wo - p.l)];
    } else if (p.mode == 1u) {
        let hs = min(max(ho, p.t), p.t + p.h - 1u) - p.t;
        let ws = min(max(wo, p.l), p.l + p.w - 1u) - p.l;
        y[idx] = x[img * p.h * p.w + hs * p.w + ws];
    } else {
        y[idx] = bitcast<f32>(p.value);
    }
}
