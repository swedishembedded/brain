// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  ray-evaluated splat backward, stage 2: one work group per tile reduces its pixels' logged pair gradients into per-instance records
// @how   128-thread workgroup tile, 4 barriers
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import camera
// @import splat_view
// @import splat_ray_pair
// @import splat_ray_window
//
// The ray renderer's backward as the GPU runs it, second stage. Together with
// `splat_ray_bwd_walk.wgsl` it computes exactly the records of
// `splat_ray_bwd_slots.wgsl` + `splat_bwd_tile_reduce.wgsl` - the per-pixel
// reference the CPU runs - without their slot grid: that path writes 17
// floats per (instance, pixel) to device memory and reads them all back,
// 17 * 256 * 4 bytes per instance whatever the gaussian covers, which at a
// few million instances per view is tens of gigabytes of traffic for what
// ends as 68 bytes per instance. The walk stage logs 16 bytes per pair a
// pixel actually composited instead.
//
// One workgroup per tile, 128 threads, two pixels per thread (rows y and
// y + 8 of the tile). A pixel's log lists its pairs in list order, so all
// threads walk the list in lockstep, each pixel with a cursor into its log:
// per instance, a pixel whose next logged pair is that instance's gaussian
// (a gaussian is in a tile's list once) re-evaluates
// the pair's geometry along its ray and expands the logged dL/dalpha, dL/dt*
// and w into the 17 channels (`ray_pair_grad`); the rest contribute zeros.
// Each thread sums its two pixels' channels, the workgroup reduces the 128
// partial sums in two stages (two barriers per instance: the uniform load of
// the live count and the one between the stages), and the record lands at
// the instance's emission slot (`order`). Once no pixel of the tile has a
// logged pair left, the remaining instances get zero records without being
// visited.
//
// The reduction: stage 1 has 18 * 7 threads each summing 19 (or fewer) of a
// channel's 128 values - the 18th channel counts the tile's pixels with
// pairs left, which is how the early stop is decided without an atomic - and
// stage 2 has 18 threads each summing a channel's 7 stage-1 sums. The row
// stride of the staging array is 129, not 128, so the 32 lanes of a warp
// reading a segment each hit a different bank.

@group(0) @binding(0) var<uniform> p: View;
@group(0) @binding(1) var<storage, read>       ray:    array<vec4<f32>>; // N*4
@group(0) @binding(2) var<storage, read>       vals:   array<u32>; // sorted instance -> gaussian id
@group(0) @binding(3) var<storage, read>       order:  array<u32>; // sorted instance -> emission slot
@group(0) @binding(4) var<storage, read>       ranges: array<u32>; // n_tiles*2
@group(0) @binding(5) var<storage, read>       offs:   array<u32>; // W*H + 1
@group(0) @binding(6) var<storage, read>       pairs:  array<u32>; // splat_ray_bwd_walk's log
@group(0) @binding(7) var<storage, read_write> recs:   array<f32>; // n_isects*17

const CH: u32 = 17u;
const LANES: u32 = 128u;
const ROW: u32 = 129u;  // staging row stride, padded against bank conflicts
const SEGS: u32 = 7u;   // stage-1 segments per channel
const SEG: u32 = 19u;   // values per segment, SEGS * SEG >= LANES
const NONE: u32 = 0xffffffffu;

var<workgroup> stage: array<f32, 2322>; // (CH + 1) * ROW
var<workgroup> sums: array<f32, 126>;   // (CH + 1) * SEGS
var<workgroup> live_px: f32;
var<workgroup> span: array<u32, 2>;

fn ray_record(g: u32) -> RayRec {
    return ray_rec4(ray[g * 4u], ray[g * 4u + 1u], ray[g * 4u + 2u], ray[g * 4u + 3u]);
}

// A pixel of the tile: its ray, its upstream gradient and a cursor into its
// logged pairs - `g` the gaussian of the next one, NONE when there is none.
struct Cursor {
    o: vec3<f32>,
    d: vec3<f32>,
    gc: vec3<f32>,
    gn: vec3<f32>,
    e: u32,   // word of the next logged pair
    end: u32, // one past the last
    g: u32,
};

fn peek(s: Cursor) -> Cursor {
    var c = s;
    c.g = NONE;
    if (c.e < c.end) { c.g = pairs[c.e]; }
    return c;
}

fn cursor(tile: u32, local: u32) -> Cursor {
    var c: Cursor;
    let tiles_x = (p.width + 15u) / 16u;
    let px = (tile % tiles_x) * 16u + local % 16u;
    let py = (tile / tiles_x) * 16u + local / 16u;
    c.e = 0u;
    c.end = 0u;
    if (px < p.width && py < p.height) {
        let pix = py * p.width + px;
        let pr = pixel_ray(p, vec2<f32>(f32(px) + 0.5, f32(py) + 0.5));
        c.o = pr.o;
        c.d = pr.d;
        let n_px = p.width * p.height;
        let h = 6u * (n_px + pix);
        c.gc = vec3<f32>(bitcast<f32>(pairs[h]), bitcast<f32>(pairs[h + 1u]), bitcast<f32>(pairs[h + 2u]));
        c.gn = vec3<f32>(bitcast<f32>(pairs[h + 3u]), bitcast<f32>(pairs[h + 4u]), bitcast<f32>(pairs[h + 5u]));
        let base = 12u * n_px;
        c.e = base + 4u * offs[pix];
        c.end = base + 4u * offs[pix + 1u];
    }
    return peek(c);
}

struct Taken {
    c: Cursor,
    part: array<f32, 17>,
};

// This pixel's 17 channels of the list's gaussian g (record r), advancing
// past it.
fn take(s: Cursor, g: u32, r: RayRec) -> Taken {
    var out: Taken;
    out.c = s;
    for (var c = 0u; c < CH; c = c + 1u) { out.part[c] = 0.0; }
    if (s.g != g) { return out; }
    let g_alpha = bitcast<f32>(pairs[s.e + 1u]);
    let g_t = bitcast<f32>(pairs[s.e + 2u]);
    let w = bitcast<f32>(pairs[s.e + 3u]);
    if (w != 0.0) {
        out.part = ray_pair_grad(r, s.o, s.d, s.gc, s.gn, g_alpha, g_t, w).part;
    }
    out.c.e = s.e + 4u;
    out.c = peek(out.c);
    return out;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_index) lane: u32,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let tiles_x = (p.width + 15u) / 16u;
    let n_tiles = tiles_x * ((p.height + 15u) / 16u);
    let tile = wg.y * nwg.x + wg.x;
    if (tile >= n_tiles) { return; }
    if (lane == 0u) {
        span[0] = ranges[tile * 2u];
        span[1] = ranges[tile * 2u + 1u];
        live_px = 1.0;
    }
    let start = workgroupUniformLoad(&span[0]);
    let end = workgroupUniformLoad(&span[1]);

    var c0 = cursor(tile, lane);
    var c1 = cursor(tile, lane + LANES);
    var j = start;
    loop {
        if (j >= end) { break; }
        let g = vals[j];
        let r = ray_record(g);
        let s0 = take(c0, g, r);
        let s1 = take(c1, g, r);
        c0 = s0.c;
        c1 = s1.c;
        for (var c = 0u; c < CH; c = c + 1u) {
            stage[c * ROW + lane] = s0.part[c] + s1.part[c];
        }
        stage[CH * ROW + lane] = f32(c0.g != NONE) + f32(c1.g != NONE);
        // No pixel had a pair left after the previous instance: this one's
        // partials, and those of everything after it, are zero.
        if (workgroupUniformLoad(&live_px) == 0.0) { break; }
        if (lane < (CH + 1u) * SEGS) {
            let c = lane / SEGS;
            let k0 = (lane % SEGS) * SEG;
            let k1 = min(k0 + SEG, LANES);
            var acc = 0.0;
            for (var k = k0; k < k1; k = k + 1u) {
                acc = acc + stage[c * ROW + k];
            }
            sums[lane] = acc;
        }
        workgroupBarrier();
        if (lane <= CH) {
            var acc = 0.0;
            for (var k = 0u; k < SEGS; k = k + 1u) {
                acc = acc + sums[lane * SEGS + k];
            }
            if (lane < CH) {
                recs[order[j] * CH + lane] = acc;
            } else {
                live_px = acc;
            }
        }
        j = j + 1u;
    }
    // the instances after the tile's last logged pair
    let rest = (end - j) * CH;
    for (var k = lane; k < rest; k = k + LANES) {
        recs[order[j + k / CH] * CH + k % CH] = 0.0;
    }
}
