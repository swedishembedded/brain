// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-channel bias gradient over NCL
// @how   one thread per channel, serial reduction over rows*inner
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Per-channel bias gradient over NCL (`[rows, C, inner]`, e.g. `[N, C, L]`
// for a 1D conv's channel bias): `dbias[c] = sum over rows,inner of
// dy[row,c,l]`. `bias_grad.wgsl` does not fit here - it assumes the
// feature axis is the FASTEST-varying one (`dy[m,n]`, one row per sample),
// which is the opposite convention from NCL's channel-then-length layout.
// WRITES (not accumulates) `dbias` - same convention as
// `snake1d_bwd_dalpha.wgsl`.

struct Params {
    rows: u32,
    c: u32,
    inner: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:    array<f32>;
@group(0) @binding(2) var<storage, read_write> dbias: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let c = gid.y * (nwg.x * 64u) + gid.x;
    if (c >= p.c) { return; }
    var acc = 0.0;
    for (var row: u32 = 0u; row < p.rows; row = row + 1u) {
        let base = row * (p.c * p.inner) + c * p.inner;
        for (var l: u32 = 0u; l < p.inner; l = l + 1u) {
            acc = acc + dy[base + l];
        }
    }
    dbias[c] = acc;
}
