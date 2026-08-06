// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-channel scale (forward) — the codec decoder's LayerScale and any elementwise per-channel gain
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Per-channel scale (forward) — the codec decoder's LayerScale and any
// elementwise per-channel gain. y = x * scale[c], generic [rows, C, inner]
// layout (channel c = (idx / inner) % C; per-feature on [.,D] => inner=1, C=D).

struct Params {
    total: u32,
    c: u32,
    inner: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read>       scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> out:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let c = (idx / p.inner) % p.c;
    out[idx] = x[idx] * scale[c];
}
