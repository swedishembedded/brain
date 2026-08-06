// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Tiled 3DGS, stage 2: expand each visible gaussian into per-tile sort instances at its scanned offset
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Tiled 3DGS, stage 2: expand each visible gaussian into per-tile sort
// instances at its scanned offset. Key = tile_id << depth_bits | depth_q,
// where depth_q is the top `depth_bits` of the raw IEEE bits of the positive
// camera depth (monotonic under truncation), so one 32-bit LSD radix sort
// yields tile-major, front-to-back order. Value = gaussian index.
// Writes past `cap` are dropped (host clamps n_isects and warns).

struct Params {
    n: u32,
    tiles_x: u32,
    tiles_y: u32,
    tile: u32,       // 16
    depth_bits: u32, // 32 - ceil_log2(n_tiles)
    cap: u32,        // keys/vals capacity
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       proj:    array<f32>; // N*9
@group(0) @binding(2) var<storage, read>       offsets: array<u32>; // N (scanned counts)
@group(0) @binding(3) var<storage, read_write> keys:    array<u32>;
@group(0) @binding(4) var<storage, read_write> vals:    array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    let o = i * 9u;
    let rx = proj[o + 7u];
    if (rx <= 0.0) { return; }
    let ry = proj[o + 8u];
    let px = proj[o];
    let py = proj[o + 1u];
    let t = f32(p.tile);
    let tx0 = clamp(i32(floor((px - rx) / t)), 0, i32(p.tiles_x) - 1);
    let tx1 = clamp(i32(floor((px + rx) / t)), 0, i32(p.tiles_x) - 1);
    let ty0 = clamp(i32(floor((py - ry) / t)), 0, i32(p.tiles_y) - 1);
    let ty1 = clamp(i32(floor((py + ry) / t)), 0, i32(p.tiles_y) - 1);
    let depth_q = bitcast<u32>(proj[o + 6u]) >> (32u - p.depth_bits);
    var pos = offsets[i];
    for (var ty = ty0; ty <= ty1; ty = ty + 1) {
        for (var tx = tx0; tx <= tx1; tx = tx + 1) {
            if (pos < p.cap) {
                let tile_id = u32(ty) * p.tiles_x + u32(tx);
                keys[pos] = (tile_id << p.depth_bits) | depth_q;
                vals[pos] = i;
            }
            pos = pos + 1u;
        }
    }
}
