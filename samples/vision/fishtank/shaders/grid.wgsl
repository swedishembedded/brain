// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// The tank: floor, ceiling and side walls behind the glass, covered in an
// analytic neon grid. Lines are anti-aliased with screen-space derivatives,
// so they stay one pixel crisp at every depth; their convergence toward the
// open back is the strongest perspective cue in the scene.

struct TankIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct TankOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_tank(v: TankIn) -> TankOut {
    var o: TankOut;
    o.world = v.position;
    o.normal = v.normal;
    o.clip = frame.view_proj * vec4<f32>(v.position, 1.0);
    return o;
}

// Anti-aliased grid lines `width` pixels wide. Where a cell shrinks below
// about two pixels (grazing angles, far away) the lines would merge into a
// solid smear, so they fade out instead: a grid finer than the pixel grid
// has nothing left to show.
fn grid_line(c: vec2<f32>, spacing: f32, width: f32) -> f32 {
    let q = c / spacing;
    let fw = max(fwidth(q), vec2<f32>(1e-6));
    let g = abs(fract(q - 0.5) - 0.5) / fw;
    let fade = 1.0 - smoothstep(0.25, 0.55, max(fw.x, fw.y));
    return (1.0 - min(min(g.x, g.y) / width, 1.0)) * fade;
}

@fragment
fn fs_tank(v: TankOut) -> @location(0) vec4<f32> {
    let p = v.world;
    let n = normalize(v.normal);
    let depth = frame.tank.z;
    let horizontal = abs(n.y) > 0.5;
    let c = select(vec2<f32>(p.y, p.z), vec2<f32>(p.x, p.z), horizontal);
    var tint = vec3<f32>(0.55, 0.25, 1.0);
    if (horizontal) {
        tint = select(vec3<f32>(0.75, 0.3, 1.0), vec3<f32>(0.1, 0.85, 1.0), n.y > 0.0);
    }
    let along = clamp(-p.z / depth, 0.0, 1.0);
    let minor = grid_line(c, 0.02, 1.0);
    let major = grid_line(c, 0.10, 1.6);
    let grid = (minor * 0.35 + major * 1.4) * frame.tank.w;
    // the glass rim and the open back glow brightest
    let rim = exp(-abs(p.z) * 900.0) * 2.5 + exp(-abs(p.z + depth) * 600.0) * 3.0;
    var col = vec3<f32>(0.006, 0.008, 0.02) + crystal_light(p, n) * 0.3;
    col = col + tint * (grid * (1.1 - 0.7 * along) + rim);
    // a pool of the crystal's light on the floor
    if (horizontal && n.y > 0.0) {
        let dxz = p.xz - frame.light.xz;
        let pool = exp(-dot(dxz, dxz) / 0.0016);
        col = col + vec3<f32>(0.15, 0.6, 1.0) * pool * (0.8 + 0.4 * frame.light.w);
    }
    return vec4<f32>(fogged(col, p), 1.0);
}
