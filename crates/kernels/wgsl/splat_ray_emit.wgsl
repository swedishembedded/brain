// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pop-free differentiable splat renderers for
// its clients. If your team needs expertise in gaussian splatting then you
// can procure our services by sending an email to info@swedishembedded.com.

// @what  Ray-evaluated splatting: per-tile sort instances keyed by each gaussian's range along the tile's ray nearest its centre
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import camera
// @import splat_view
// @import splat_ray_pair
//
// The ray renderer's `splat_emit.wgsl`: the same instances at the same
// scanned offsets, the same values (emission index) and `ids`, but a key
// whose depth part is the gaussian's t* - the range of its maximum response
// along a ray - through the point of THAT tile nearest the gaussian's
// projected centre, rather than the range to its mean, one number for every
// tile. A pixel composites in order of its own t* (`splat_ray_window`), so
// the closer the list comes to each pixel's order, the less its window has
// to repair. The range to the mean misplaces a large or tilted gaussian,
// whose mean can lie far from where the tile's rays meet it; so does the
// tile's central ray for a gaussian that only clips the tile, which that ray
// passes wide of. The nearest point stands for the pixels that see the
// gaussian most. A host model of the ordering over 12 tiles of a trained
// 500k-gaussian scene at 816x612, with a window of 16, left 104 of their 3072
// pixels more than 0.02 in colour from the exact order when keyed through
// the central ray, 37 by the range to the mean and 24 through the nearest
// point.
//
// That point's ray comes from an affine model of the lens around the
// projected centre - the rays through it and one pixel to its right and
// below - so a lens that needs an iterative inverse (Brown, fisheye) is
// inverted three times per gaussian rather than once per instance, and the
// per-instance loop reads no memory. A key only has to order the list well,
// not exactly: the pixels' windows and the compositing use the exact rays.
//
// Key = tile_id << depth_bits | the top `depth_bits` of t*'s IEEE bits
// (non-negative floats order as their bits). Where the lens casts no ray
// near the centre, or a ray meets the gaussian behind its origin, the key is
// the range to the mean. Writes past `cap` are dropped (host clamps n_isects
// and warns).

struct Tail {
    tiles_x: u32,
    tiles_y: u32,
    depth_bits: u32, // 32 - ceil_log2(n_tiles)
    cap: u32,        // keys/vals capacity
};

struct Params {
    v: View,
    t: Tail,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       proj:    array<f32>; // N*9
@group(0) @binding(2) var<storage, read>       ray:     array<vec4<f32>>; // N*4
@group(0) @binding(3) var<storage, read>       offsets: array<u32>; // N (scanned counts)
@group(0) @binding(4) var<storage, read_write> keys:    array<u32>;
@group(0) @binding(5) var<storage, read_write> vals:    array<u32>; // emission index
@group(0) @binding(6) var<storage, read_write> ids:     array<u32>; // emission index -> gaussian

const TILE: f32 = 16.0;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.v.n) { return; }
    let o = i * 9u;
    let rx = proj[o + 7u];
    if (rx <= 0.0) { return; }
    let ry = proj[o + 8u];
    let px = proj[o];
    let py = proj[o + 1u];
    let range = proj[o + 6u];
    let tx0 = clamp(i32(floor((px - rx) / TILE)), 0, i32(p.t.tiles_x) - 1);
    let tx1 = clamp(i32(floor((px + rx) / TILE)), 0, i32(p.t.tiles_x) - 1);
    let ty0 = clamp(i32(floor((py - ry) / TILE)), 0, i32(p.t.tiles_y) - 1);
    let ty1 = clamp(i32(floor((py + ry) / TILE)), 0, i32(p.t.tiles_y) - 1);
    // only the mean and the precision take part: t* = d.A(m - o) / d.A d
    let zero = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let r = ray_rec4(ray[i * 4u], ray[i * 4u + 1u], ray[i * 4u + 2u], zero);
    // the lens around the centre: the ray at (px, py) + (dx, dy) pixels is
    // c + dx * (right - c) + dy * (below - c), its direction normalized
    let c = pixel_ray(p.v, vec2<f32>(px, py));
    let right = pixel_ray(p.v, vec2<f32>(px + 1.0, py));
    let below = pixel_ray(p.v, vec2<f32>(px, py + 1.0));
    let lens_ok = c.ok * right.ok * below.ok != 0.0;
    let dx_d = right.d - c.d;
    let dy_d = below.d - c.d;
    let dx_o = right.o - c.o;
    let dy_o = below.o - c.o;
    let shift = 32u - p.t.depth_bits;
    var pos = offsets[i];
    for (var ty = ty0; ty <= ty1; ty = ty + 1) {
        for (var tx = tx0; tx <= tx1; tx = tx + 1) {
            if (pos < p.t.cap) {
                let x0 = f32(tx) * TILE;
                let y0 = f32(ty) * TILE;
                let dx = clamp(px, x0 + 0.5, x0 + TILE - 0.5) - px;
                let dy = clamp(py, y0 + 0.5, y0 + TILE - 0.5) - py;
                var depth = range;
                if (lens_ok) {
                    let d = normalize(c.d + dx * dx_d + dy * dy_d);
                    let ro = c.o + dx * dx_o + dy * dy_o;
                    let ad = ray_rec_sym(r, d);
                    let vv = dot(d, ad);
                    if (vv > 0.0) {
                        let t = dot(r.m - ro, ad) / vv;
                        if (t > 0.0) { depth = t; }
                    }
                }
                let tile_id = u32(ty) * p.t.tiles_x + u32(tx);
                keys[pos] = (tile_id << p.t.depth_bits) | (bitcast<u32>(depth) >> shift);
                vals[pos] = pos;
                ids[pos] = i;
            }
            pos = pos + 1u;
        }
    }
}
