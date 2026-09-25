// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  ray-evaluated splat backward: one work group per tile, gradient records reduced in workgroup memory
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
//
// The ray renderer's backward as the GPU runs it. It computes exactly the
// records of `splat_ray_bwd_slots.wgsl` + `splat_bwd_tile_reduce.wgsl` - the
// per-pixel reference the CPU runs - without their slot grid: that path
// writes 17 floats per (instance, pixel) to device memory and reads them all
// back, 17 * 256 * 4 bytes per instance whatever the gaussian covers, which
// at a few million instances per view is tens of gigabytes of traffic for
// what ends as 68 bytes per instance.
//
// One workgroup per tile, 128 threads, two pixels per thread (rows y and
// y + 8 of the tile). Each thread first walks its pixels' lists for the
// totals (no barriers), then all threads walk the list in lockstep: per
// instance each thread sums its two pixels' 17 channels, the workgroup
// reduces the 128 partial sums in two stages (two barriers per instance: the
// uniform load of the live count and the one between the stages), and the
// record lands at the instance's emission slot (`order`). A pixel whose walk has stopped
// contributes zeros; once no pixel of the tile is live the remaining
// instances get zero records without being evaluated.
//
// The reduction: stage 1 has 18 * 7 threads each summing 19 (or fewer) of a
// channel's 128 values - the 18th channel counts the tile's live pixels,
// which is how the early stop is decided without an atomic - and stage 2 has
// 18 threads each summing a channel's 7 stage-1 sums. The row stride of the
// staging array is 129, not 128, so the 32 lanes of a warp reading a
// segment each hit a different bank.
//
// With `want_ray` set, every pixel also writes dL/d(its ray origin) and
// dL/d(its ray direction) to dray[pix*6] for `splat_ray_camera_grad`.

struct Tail {
    want_ray: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Params {
    v: View,
    t: Tail,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       ray:    array<f32>; // N*16
@group(0) @binding(2) var<storage, read>       vals:   array<u32>; // sorted instance -> gaussian id
@group(0) @binding(3) var<storage, read>       order:  array<u32>; // sorted instance -> emission slot
@group(0) @binding(4) var<storage, read>       ranges: array<u32>; // n_tiles*2
@group(0) @binding(5) var<storage, read>       dimg:   array<f32>; // W*H*4
@group(0) @binding(6) var<storage, read>       daux:   array<f32>; // W*H*5
@group(0) @binding(7) var<storage, read_write> recs:   array<f32>; // n_isects*17
@group(0) @binding(8) var<storage, read_write> dray:   array<f32>; // W*H*6

const CH: u32 = 17u;
const LANES: u32 = 128u;
const ROW: u32 = 129u;  // staging row stride, padded against bank conflicts
const SEGS: u32 = 7u;   // stage-1 segments per channel
const SEG: u32 = 19u;   // values per segment, SEGS * SEG >= LANES

var<workgroup> stage: array<f32, 2322>; // (CH + 1) * ROW
var<workgroup> sums: array<f32, 126>;   // (CH + 1) * SEGS
var<workgroup> live_px: f32;
var<workgroup> span: array<u32, 2>;

fn rec(g: u32) -> RayRec {
    let r = g * 16u;
    return RayRec(vec3<f32>(ray[r], ray[r + 1u], ray[r + 2u]),
                  vec3<f32>(ray[r + 3u], ray[r + 4u], ray[r + 5u]),
                  vec3<f32>(ray[r + 6u], ray[r + 7u], ray[r + 8u]),
                  vec3<f32>(ray[r + 9u], ray[r + 10u], ray[r + 11u]),
                  ray[r + 12u],
                  vec3<f32>(ray[r + 13u], ray[r + 14u], ray[r + 15u]));
}

// A pixel of the tile: whether it is in the frame, and its ray.
struct TilePx {
    pix: u32,
    inside: bool,
    ok: bool,
    o: vec3<f32>,
    d: vec3<f32>,
};

fn tile_px(tile: u32, local: u32) -> TilePx {
    var t: TilePx;
    let tiles_x = (p.v.width + 15u) / 16u;
    let px = (tile % tiles_x) * 16u + local % 16u;
    let py = (tile / tiles_x) * 16u + local / 16u;
    t.inside = px < p.v.width && py < p.v.height;
    t.pix = py * p.v.width + px;
    let pr = pixel_ray(p.v, vec2<f32>(f32(px) + 0.5, f32(py) + 0.5));
    t.ok = t.inside && pr.ok != 0.0;
    t.o = pr.o;
    t.d = pr.d;
    return t;
}

fn back_of(t: TilePx, w: RayWalk) -> RayBack {
    var dimg4 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var daux4 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var gl = 0.0;
    if (t.inside) {
        let i = t.pix;
        dimg4 = vec4<f32>(dimg[i * 4u], dimg[i * 4u + 1u], dimg[i * 4u + 2u], dimg[i * 4u + 3u]);
        daux4 = vec4<f32>(daux[i * 5u], daux[i * 5u + 1u], daux[i * 5u + 2u], daux[i * 5u + 3u]);
        gl = daux[i * 5u + 4u];
    }
    return ray_back(w, t.ok, dimg4, daux4, gl, p.v.bg.xyz);
}

fn write_ray(t: TilePx, b: RayBack) {
    if (p.t.want_ray == 0u || !t.inside) { return; }
    let i = t.pix * 6u;
    dray[i] = b.g_o.x;
    dray[i + 1u] = b.g_o.y;
    dray[i + 2u] = b.g_o.z;
    dray[i + 3u] = b.g_d.x;
    dray[i + 4u] = b.g_d.y;
    dray[i + 5u] = b.g_d.z;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_index) lane: u32,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let tiles_x = (p.v.width + 15u) / 16u;
    let n_tiles = tiles_x * ((p.v.height + 15u) / 16u);
    let tile = wg.y * nwg.x + wg.x;
    if (tile >= n_tiles) { return; }
    if (lane == 0u) {
        span[0] = ranges[tile * 2u];
        span[1] = ranges[tile * 2u + 1u];
        live_px = 1.0;
    }
    let start = workgroupUniformLoad(&span[0]);
    let end = workgroupUniformLoad(&span[1]);

    let t0 = tile_px(tile, lane);
    let t1 = tile_px(tile, lane + LANES);
    var w0 = ray_walk(t0.o, t0.d, t0.ok);
    var w1 = ray_walk(t1.o, t1.d, t1.ok);
    for (var j = start; j < end && (w0.live || w1.live); j = j + 1u) {
        let r = rec(vals[j]);
        w0 = ray_walk_step(w0, r);
        w1 = ray_walk_step(w1, r);
    }
    var b0 = back_of(t0, w0);
    var b1 = back_of(t1, w1);

    var j = start;
    loop {
        if (j >= end) { break; }
        let r = rec(vals[j]);
        let s0 = ray_back_step(b0, r);
        let s1 = ray_back_step(b1, r);
        b0 = s0.b;
        b1 = s1.b;
        for (var c = 0u; c < CH; c = c + 1u) {
            stage[c * ROW + lane] = s0.part[c] + s1.part[c];
        }
        stage[CH * ROW + lane] = f32(b0.live) + f32(b1.live);
        // Every pixel was dead before this instance: its partials, and those
        // of everything after it, are zero.
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
    // the instances after the tile went dark
    let rest = (end - j) * CH;
    for (var k = lane; k < rest; k = k + LANES) {
        recs[order[j + k / CH] * CH + k % CH] = 0.0;
    }
    write_ray(t0, b0);
    write_ray(t1, b1);
}
