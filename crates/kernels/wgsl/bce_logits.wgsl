// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Binary cross-entropy with logits, per (anchor, class), against a soft target
// t in [0,1]. Numerically stable form:
//   loss = max(z,0) - z*t + log(1 + exp(-|z|))
// Output out[total] (host sums). One thread per element.

struct Params {
    total: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:    array<f32>;
@group(0) @binding(3) var<storage, read_write> out:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.total) { return; }
    let z = logits[idx];
    let t = tgt[idx];
    out[idx] = max(z, 0.0) - z * t + log(1.0 + exp(-abs(z)));
}
