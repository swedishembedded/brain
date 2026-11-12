// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// The one material texture, generated on the GPU at start-up into every mip
// level of a 512 x 512 RGBA8 texture: nothing is stored on disk, and the
// whole set is 1.4 MB of GPU memory.
//
//   r  height: tileable value-noise fBm, 8 to 256 cycles per tile
//   g  blotches: low-frequency fBm, 2 to 16 cycles per tile (rust patches,
//      mountain ridges, clouds)
//   b  dr/du, a dr/dv: the height's slope, so a surface is bump mapped with
//      one fetch instead of three
//
// Each mip level is generated, not downsampled: octaves above the level's
// Nyquist frequency are replaced by their mean, so a level is the exactly
// band-limited version of level 0 and distant surfaces do not shimmer. The
// renderer draws one full-screen triangle per level, passing the level as
// the instance index.

const SIZE: u32 = 512u;
// slopes are stored as 0.5 + slope / (2 * SLOPE_RANGE)
const SLOPE_RANGE: f32 = 64.0;

struct BakeOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) level: u32,
};

@vertex
fn vs_bake(@builtin(vertex_index) i: u32, @builtin(instance_index) level: u32) -> BakeOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return BakeOut(vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0), level);
}

// A lattice value, periodic in `period` so the texture tiles.
fn lattice(c: vec2<i32>, period: i32, seed: u32) -> f32 {
    let w = ((c % period) + period) % period;
    return f32(hash_u(u32(w.x) ^ hash_u(u32(w.y) ^ hash_u(seed)))) / 4294967295.0;
}

fn tile_noise(uv: vec2<f32>, f: i32, seed: u32) -> f32 {
    let p = uv * f32(f);
    let i = vec2<i32>(floor(p));
    let t = fract(p);
    let s = t * t * (3.0 - 2.0 * t);
    let a = mix(lattice(i, f, seed), lattice(i + vec2<i32>(1, 0), f, seed), s.x);
    let b = mix(lattice(i + vec2<i32>(0, 1), f, seed), lattice(i + vec2<i32>(1, 1), f, seed), s.x);
    return mix(a, b, s.y);
}

// fBm from `f0` cycles per tile, `octaves` doublings, band-limited at `nyquist`.
fn tile_fbm(uv: vec2<f32>, f0: i32, octaves: i32, nyquist: f32, seed: u32) -> f32 {
    var s = 0.0;
    var a = 0.5;
    var f = f0;
    for (var k = 0; k < octaves; k = k + 1) {
        s = s + a * select(0.5, tile_noise(uv, f, seed + u32(k)), f32(f) <= nyquist);
        f = f * 2;
        a = a * 0.5;
    }
    return s;
}

// fBm clusters around its mean; stretch it to use the 8 bits.
fn height(uv: vec2<f32>, nyquist: f32) -> f32 {
    return (tile_fbm(uv, 8, 6, nyquist, 11u) - 0.49) * 2.4 + 0.5;
}

@fragment
fn fs_bake(v: BakeOut) -> @location(0) vec4<f32> {
    let size = f32(SIZE >> v.level);
    let uv = v.clip.xy / size;
    let nyq = size * 0.5;
    let e = 1.0 / size;
    let du = (height(uv + vec2<f32>(e, 0.0), nyq) - height(uv - vec2<f32>(e, 0.0), nyq)) / (2.0 * e);
    let dv = (height(uv + vec2<f32>(0.0, e), nyq) - height(uv - vec2<f32>(0.0, e), nyq)) / (2.0 * e);
    let blotch = (tile_fbm(uv, 2, 4, nyq, 41u) - 0.47) * 2.2 + 0.5;
    return vec4<f32>(height(uv, nyq), blotch, 0.5 + du / (2.0 * SLOPE_RANGE), 0.5 + dv / (2.0 * SLOPE_RANGE));
}

