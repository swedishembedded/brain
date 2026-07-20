// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Tiled 3DGS, stage 1: per-gaussian overlapped-tile count. The bbox
// [mean2d ± radius] is intersected with the screen's 16×16 tile grid; culled
// gaussians (radius 0) count 0. The counts buffer is then prefix-scanned in
// place into emission offsets.

struct Params {
    n: u32,
    tiles_x: u32,
    tiles_y: u32,
    tile: u32, // 16
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       proj:   array<f32>; // N*9
@group(0) @binding(2) var<storage, read_write> counts: array<u32>; // N

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    let o = i * 9u;
    let rx = proj[o + 7u];
    if (rx <= 0.0) {
        counts[i] = 0u;
        return;
    }
    let ry = proj[o + 8u];
    let px = proj[o];
    let py = proj[o + 1u];
    let t = f32(p.tile);
    let tx0 = clamp(i32(floor((px - rx) / t)), 0, i32(p.tiles_x) - 1);
    let tx1 = clamp(i32(floor((px + rx) / t)), 0, i32(p.tiles_x) - 1);
    let ty0 = clamp(i32(floor((py - ry) / t)), 0, i32(p.tiles_y) - 1);
    let ty1 = clamp(i32(floor((py + ry) / t)), 0, i32(p.tiles_y) - 1);
    counts[i] = u32(tx1 - tx0 + 1) * u32(ty1 - ty0 + 1);
}
