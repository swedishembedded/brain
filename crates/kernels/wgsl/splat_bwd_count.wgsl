// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// 3DGS backward, stage 1: per-pixel count of gradient-contributing gaussians
// (same walk as the forward compositing: alpha >= 1/255, stop at T <= 1e-4).
// The counts are prefix-scanned into record offsets for splat_bwd_emit.
// One invocation per pixel.

struct Params {
    width: u32,
    height: u32,
    tiles_x: u32,
    tiles_y: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       proj:   array<f32>; // N*9
@group(0) @binding(2) var<storage, read>       vals:   array<u32>; // sorted ids
@group(0) @binding(3) var<storage, read>       ranges: array<u32>; // n_tiles*2
@group(0) @binding(4) var<storage, read_write> counts: array<u32>; // W*H

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.width * p.height) { return; }
    let px = idx % p.width;
    let py = idx / p.width;
    let fx = f32(px) + 0.5;
    let fy = f32(py) + 0.5;
    let tile = (py / 16u) * p.tiles_x + (px / 16u);
    let start = ranges[tile * 2u];
    let end = ranges[tile * 2u + 1u];
    var t = 1.0;
    var n = 0u;
    for (var j = start; j < end; j = j + 1u) {
        let o = vals[j] * 9u;
        let dx = proj[o] - fx;
        let dy = proj[o + 1u] - fy;
        let sigma = 0.5 * (proj[o + 2u] * dx * dx + proj[o + 4u] * dy * dy)
            + proj[o + 3u] * dx * dy;
        if (sigma < 0.0) { continue; }
        let alpha = min(0.99, proj[o + 5u] * exp(-sigma));
        if (alpha < 1.0 / 255.0) { continue; }
        let next_t = t * (1.0 - alpha);
        if (next_t <= 1e-4) { break; }
        n = n + 1u;
        t = next_t;
    }
    counts[idx] = n;
}
