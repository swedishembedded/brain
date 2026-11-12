// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Neon Observatory: procedural geometry placed in the SCREEN frame
//! (metres, origin at the display centre, +Z toward the viewer), so it stays
//! fixed to the laptop while the viewer moves.
//!
//! The layout is chosen for depth cues, not decoration: a tank whose front
//! opening IS the glass (its walls converge correctly only if the projection
//! is right), pillars at several depths that occlude each other differently
//! from every eye, rings and a crystal floating mid-tank, and a few orbs IN
//! FRONT of the glass that move against the viewer. Beyond the tank's open
//! back is a starfield at infinity, which shows no parallax at all - the
//! reference every other layer moves against.

use bytemuck::{Pod, Zeroable};

/// One mesh vertex: position and normal, model space.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// One object instance: its model matrix (uniform scale and rotation only,
/// so normals transform by the same matrix), a colour, and a material.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Instance {
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
    /// `x` material (see `scene.wgsl`), `y` emission, `z` phase, `w` unused.
    pub material: [f32; 4],
}

/// Materials the object shader understands.
pub const CRYSTAL: f32 = 0.0;
pub const PILLAR: f32 = 1.0;
pub const RING: f32 = 2.0;
pub const ORB: f32 = 3.0;

/// A mesh's vertex range in the shared vertex buffer.
#[derive(Clone, Copy, Debug)]
pub struct Range {
    pub first: u32,
    pub count: u32,
}

/// All geometry, as one non-indexed triangle list (flat-shaded meshes need
/// per-face normals, which indexing would only duplicate anyway).
pub struct Meshes {
    pub vertices: Vec<Vertex>,
    pub tank: Range,
    pub crystal: Range,
    pub pillar: Range,
    pub ring: Range,
    pub orb: Range,
}

/// The tank's size: it fills the display and runs back `depth` metres.
#[derive(Clone, Copy, Debug)]
pub struct Tank {
    pub half_w: f32,
    pub half_h: f32,
    pub depth: f32,
}

impl Tank {
    pub fn for_display(width_m: f64, height_m: f64) -> Tank {
        Tank { half_w: width_m as f32 / 2.0, half_h: height_m as f32 / 2.0, depth: (width_m * 0.9) as f32 }
    }
}

fn push_tri(v: &mut Vec<Vertex>, a: [f32; 3], b: [f32; 3], c: [f32; 3]) {
    let u = sub(b, a);
    let w = sub(c, a);
    let n = normalize(cross(u, w));
    for p in [a, b, c] {
        v.push(Vertex { position: p, normal: n });
    }
}

fn push_quad(v: &mut Vec<Vertex>, a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) {
    push_tri(v, a, b, c);
    push_tri(v, a, c, d);
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn normalize(a: [f32; 3]) -> [f32; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-12);
    [a[0] / l, a[1] / l, a[2] / l]
}

/// The tank's four inner faces, normals pointing into the tank; the front is
/// the glass and the back is open to the sky.
fn tank(v: &mut Vec<Vertex>, t: Tank) {
    let (x, y, z) = (t.half_w, t.half_h, -t.depth);
    // counter-clockwise as seen from inside
    push_quad(v, [-x, -y, 0.0], [x, -y, 0.0], [x, -y, z], [-x, -y, z]); // floor
    push_quad(v, [-x, y, z], [x, y, z], [x, y, 0.0], [-x, y, 0.0]); // ceiling
    push_quad(v, [-x, -y, z], [-x, y, z], [-x, y, 0.0], [-x, -y, 0.0]); // left wall
    push_quad(v, [x, -y, 0.0], [x, y, 0.0], [x, y, z], [x, -y, z]); // right wall
}

/// A faceted gem: an eight-sided bipyramid with a crown, unit height.
fn crystal(v: &mut Vec<Vertex>) {
    let n = 8;
    let top = [0.0, 0.5, 0.0];
    let bottom = [0.0, -0.5, 0.0];
    let ring = |k: usize, r: f32, y: f32, twist: f32| {
        let a = (k as f32 + twist) / n as f32 * std::f32::consts::TAU;
        [r * a.cos(), y, r * a.sin()]
    };
    for k in 0..n {
        let (a, b) = (ring(k, 0.30, 0.12, 0.0), ring(k + 1, 0.30, 0.12, 0.0));
        let (c, d) = (ring(k, 0.16, 0.30, 0.5), ring(k + 1, 0.16, 0.30, 0.5));
        // crown: girdle to upper ring, then to the apex
        push_tri(v, a, c, b);
        push_tri(v, b, c, d);
        push_tri(v, c, top, d);
        // pavilion
        push_tri(v, b, bottom, a);
    }
}

/// A hexagonal prism of unit height and unit radius, open ends.
fn pillar(v: &mut Vec<Vertex>) {
    let n = 6;
    for k in 0..n {
        let a0 = k as f32 / n as f32 * std::f32::consts::TAU;
        let a1 = (k + 1) as f32 / n as f32 * std::f32::consts::TAU;
        let (p0, p1) = ([a0.cos(), 0.0, a0.sin()], [a1.cos(), 0.0, a1.sin()]);
        push_quad(v, [p0[0], -0.5, p0[2]], [p0[0], 0.5, p0[2]], [p1[0], 0.5, p1[2]], [p1[0], -0.5, p1[2]]);
    }
}

/// A torus in the XY plane, major radius 1, minor radius `r`, smooth normals.
fn torus(v: &mut Vec<Vertex>, r: f32) {
    let (nu, nv) = (48, 12);
    let at = |i: usize, j: usize| {
        let u = i as f32 / nu as f32 * std::f32::consts::TAU;
        let w = j as f32 / nv as f32 * std::f32::consts::TAU;
        let c = [u.cos(), u.sin(), 0.0];
        let n = [c[0] * w.cos(), c[1] * w.cos(), w.sin()];
        Vertex { position: [c[0] + r * n[0], c[1] + r * n[1], r * n[2]], normal: n }
    };
    for i in 0..nu {
        for j in 0..nv {
            let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
            v.extend([a, b, c, a, c, d]);
        }
    }
}

/// A unit sphere, smooth normals.
fn sphere(v: &mut Vec<Vertex>) {
    let (nu, nv) = (24, 12);
    let at = |i: usize, j: usize| {
        let u = i as f32 / nu as f32 * std::f32::consts::TAU;
        let w = j as f32 / nv as f32 * std::f32::consts::PI;
        let n = [w.sin() * u.cos(), w.cos(), w.sin() * u.sin()];
        Vertex { position: n, normal: n }
    };
    for i in 0..nu {
        for j in 0..nv {
            let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
            v.extend([a, c, b, a, d, c]);
        }
    }
}

impl Meshes {
    pub fn build(t: Tank) -> Meshes {
        let mut v = Vec::new();
        let range = |v: &mut Vec<Vertex>, f: &dyn Fn(&mut Vec<Vertex>)| {
            let first = v.len() as u32;
            f(v);
            Range { first, count: v.len() as u32 - first }
        };
        let tank = range(&mut v, &|v| tank(v, t));
        let crystal = range(&mut v, &crystal);
        let pillar = range(&mut v, &pillar);
        let ring = range(&mut v, &|v| torus(v, 0.09));
        let orb = range(&mut v, &sphere);
        Meshes { vertices: v, tank, crystal, pillar, ring, orb }
    }
}

/// Column-major model matrix: rotate about Y by `yaw`, then about X by
/// `pitch`, scale uniformly by `s`, translate to `p`.
fn model(p: [f32; 3], s: f32, yaw: f32, pitch: f32) -> [[f32; 4]; 4] {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    // R = Ry * Rx, columns
    let c0 = [cy, 0.0, -sy];
    let c1 = [sy * sp, cp, cy * sp];
    let c2 = [sy * cp, -sp, cy * cp];
    [
        [c0[0] * s, c0[1] * s, c0[2] * s, 0.0],
        [c1[0] * s, c1[1] * s, c1[2] * s, 0.0],
        [c2[0] * s, c2[1] * s, c2[2] * s, 0.0],
        [p[0], p[1], p[2], 1.0],
    ]
}

/// Per-mesh instance lists for time `t` seconds.
pub struct Instances {
    pub crystal: Vec<Instance>,
    pub pillar: Vec<Instance>,
    pub ring: Vec<Instance>,
    pub orb: Vec<Instance>,
}

/// Where the crystal floats, for the floor's glow pool and the point light.
pub fn crystal_position(tank: Tank, t: f32) -> [f32; 3] {
    [0.0, 0.006 * (t * 0.8).sin(), -tank.depth * 0.42]
}

pub fn instances(tank: Tank, t: f32) -> Instances {
    let (w, h, d) = (tank.half_w, tank.half_h, tank.depth);
    let inst = |m: [[f32; 4]; 4], c: [f32; 3], mat: f32, e: f32, ph: f32| Instance { model: m, color: [c[0], c[1], c[2], 1.0], material: [mat, e, ph, 0.0] };

    let cp = crystal_position(tank, t);
    let crystal = vec![inst(model(cp, h * 0.62, t * 0.35, 0.0), [0.25, 0.85, 1.0], CRYSTAL, 1.0, 0.0)];

    // Pillars stand floor to ceiling. The one non-uniform scale in the scene
    // is safe: stretching an upright prism along Y leaves its side normals
    // horizontal, so transforming them by the model matrix and normalizing
    // stays correct.
    let pillar_r = w * 0.03;
    let mut pillar = Vec::new();
    for (k, &(x, z, hue)) in [(-0.72f32, 0.55f32, 0usize), (0.72, 0.55, 1), (-0.42, 0.86, 1), (0.42, 0.86, 0), (0.0, 0.97, 2)].iter().enumerate() {
        let c = [[0.2, 0.9, 1.0], [0.85, 0.3, 1.0], [1.0, 0.55, 0.2]][hue];
        let mut m = model([x * w, 0.0, -z * d], pillar_r, 0.0, 0.0);
        m[1] = [0.0, 2.0 * h, 0.0, 0.0];
        pillar.push(inst(m, c, PILLAR, 1.0, k as f32));
    }

    let ring = vec![
        inst(model([-w * 0.5, h * 0.35, -d * 0.22], h * 0.22, t * 0.5, 0.35), [1.0, 0.3, 0.8], RING, 2.0, 0.0),
        inst(model([w * 0.55, -h * 0.25, -d * 0.7], h * 0.32, -t * 0.3, -0.6), [0.3, 0.9, 1.0], RING, 2.0, 1.0),
        inst(model([w * 0.32, h * 0.45, d * 0.08], h * 0.11, t * 0.8, 1.1), [1.0, 0.75, 0.3], RING, 2.5, 2.0),
    ];

    let bob = |ph: f32| 0.004 * (t * 1.3 + ph).sin();
    let orb = vec![
        inst(model([-w * 0.28, -h * 0.5 + bob(0.0), d * 0.07], h * 0.035, 0.0, 0.0), [0.4, 1.0, 0.9], ORB, 4.0, 0.0),
        inst(model([w * 0.1, h * 0.62 + bob(1.7), d * 0.12], h * 0.03, 0.0, 0.0), [1.0, 0.5, 1.0], ORB, 4.0, 1.0),
        inst(model([w * 0.62, -h * 0.55 + bob(3.1), d * 0.04], h * 0.04, 0.0, 0.0), [1.0, 0.8, 0.4], ORB, 4.0, 2.0),
        inst(model([-w * 0.66, h * 0.1 + bob(4.4), -d * 0.35], h * 0.03, 0.0, 0.0), [0.5, 0.7, 1.0], ORB, 3.0, 3.0),
    ];
    Instances { crystal, pillar, ring, orb }
}
