// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-row mean and std over each item's valid span of a segment-packed batch, uniformly or softmax-weighted
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Statistics pooling (ECAPA's squeeze-excitation mean and its attentive
// statistics pooling) over a `[c, n, s]` segment-packed batch, row `(ch, b)`
// valid on `[off, off + lens[b])` (the layout `seg_reflect.wgsl` documents):
//
//   weighted == 0:  w_t = 1 / L
//   weighted == 1:  w_t = softmax_t(a[row, off + t])  over the valid span only
//   mean = sum_t w_t x_t;  std = sqrt(max(sum_t w_t (x_t - mean)^2, eps))
//
// written token-major for the item: mean[b*c + ch], sdev[b*c + ch], so the
// pooled `[n, c]` vectors feed a linear layer directly. `a` is ignored (bind
// any buffer) when `weighted == 0`. Params: c, n, s, off, eps (f32 bits),
// weighted. Dispatch: c*n invocations, one per row; the margins are never
// read. The reductions are serial in t, in index order.

struct Params {
    c: u32,
    n: u32,
    s: u32,
    off: u32,
    eps: f32,
    weighted: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read>       a:    array<f32>;
@group(0) @binding(3) var<storage, read>       lens: array<u32>;
@group(0) @binding(4) var<storage, read_write> mean: array<f32>;
@group(0) @binding(5) var<storage, read_write> sdev: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let row = gid.y * (nwg.x * 64u) + gid.x;
    if (row >= p.c * p.n) { return; }
    let ch = row / p.n;
    let b = row % p.n;
    let len = lens[b];
    let base = row * p.s + p.off;
    var mx = -3.402823e38;
    var z = 0.0;
    if (p.weighted != 0u) {
        for (var t = 0u; t < len; t = t + 1u) {
            mx = max(mx, a[base + t]);
        }
        for (var t = 0u; t < len; t = t + 1u) {
            z = z + exp(a[base + t] - mx);
        }
    }
    let uniform_w = 1.0 / f32(len);
    var m = 0.0;
    for (var t = 0u; t < len; t = t + 1u) {
        var w = uniform_w;
        if (p.weighted != 0u) { w = exp(a[base + t] - mx) / z; }
        m = m + w * x[base + t];
    }
    var v = 0.0;
    for (var t = 0u; t < len; t = t + 1u) {
        var w = uniform_w;
        if (p.weighted != 0u) { w = exp(a[base + t] - mx) / z; }
        let dlt = x[base + t] - m;
        v = v + w * dlt * dlt;
    }
    mean[b * p.c + ch] = m;
    sdev[b * p.c + ch] = sqrt(max(v, p.eps));
}
