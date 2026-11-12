// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// The scene, ray traced per pixel in one full-screen pass: every ray starts
// at the tracked eye and passes through its pixel on the physical glass, so
// the off-axis projection is exact by construction.
//
// Everything is in the SCREEN frame: metres, origin at the display centre,
// +X right, +Y up, +Z out of the glass toward the viewer. The layout spans
// the depths fish-tank VR has to sell:
//
//   z = +0.26 m   a rusty pipe whose open end reaches out of the glass
//   z = +0.12 m   a steel ball floating in front of the glass
//   0 .. -D       a riveted, rusted steel embrasure (the "tank"): the glass
//                 is its front, its back is open
//   -D            a buoy and a crate on its floor, lit by a low sun
//   0.7 - 9 km    a lake and three mountain ridges under a hazy sky
//
// All intersections are analytic (spheres, a hollow capped cylinder, an
// oriented box, planes), so a pixel costs a fixed handful of tests rather
// than a ray march. Lighting is physically based: GGX specular with Smith
// shadowing and Schlick Fresnel, Lambert diffuse, a sun whose shadows are
// exact, sky and room light entering through the two openings by their
// exact Lambertian form factors, and one traced mirror bounce for smooth
// steel and water.

struct Frame {
    inv_view_proj: mat4x4<f32>,
    // xyz the eye, w seconds
    eye: vec4<f32>,
    // half width, half height, depth of the embrasure; w grid on (1) / off (0)
    tank: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var mat_tex: texture_2d<f32>;
@group(0) @binding(2) var wrap: sampler;

const PI: f32 = 3.14159265;
const EXPOSURE: f32 = 0.8;
const BIG: f32 = 1e9;
const EPS: f32 = 1e-4;

// Light. The sun is low, behind the scene and to the left: light enters
// through the back opening only, toward the viewer.
const SUN: vec3<f32> = vec3<f32>(-0.3305, 0.2804, -0.9013);
const SUN_E: vec3<f32> = vec3<f32>(7.0, 5.0, 3.1);
// the cosine of the sun's angular radius (0.27 degrees)
const SUN_COS: f32 = 0.999989;
const ZENITH: vec3<f32> = vec3<f32>(0.10, 0.22, 0.55);
const HORIZON: vec3<f32> = vec3<f32>(0.85, 0.62, 0.48);
// The viewer's own room: what reflections toward the viewer show, and the
// average light it sends through the glass. A box around the laptop: the
// display wall at z = 0, a window behind the viewer and a lamp above them.
const ROOM: vec3<f32> = vec3<f32>(0.22, 0.17, 0.13);
const ROOM_MIN: vec3<f32> = vec3<f32>(-1.6, -0.75, 0.0);
const ROOM_MAX: vec3<f32> = vec3<f32>(1.6, 1.4, 2.6);
// the irradiance on the room's own walls, as radiance per unit albedo
const ROOM_LIGHT: vec3<f32> = vec3<f32>(0.9, 0.85, 0.75);
// one bounce off the sunlit rust inside the embrasure
const BOUNCE: vec3<f32> = vec3<f32>(0.075, 0.042, 0.024);

// The far scene.
const WATER_Y: f32 = -15.0;
const HAZE_M: f32 = 9000.0;
const RIDGE_Z: array<f32, 3> = array<f32, 3>(-700.0, -2500.0, -9000.0);
const RIDGE_W: array<f32, 3> = array<f32, 3>(1800.0, 6500.0, 24000.0);
const RIDGE_H: array<f32, 3> = array<f32, 3>(45.0, 260.0, 1300.0);

// Materials: the texture tiles every 1 / TILES_PER_M metres.
const TILES_PER_M: f32 = 4.0;
const TEXELS: f32 = 512.0;
const SLOPE_RANGE: f32 = 64.0;

const PIPE_R: f32 = 0.011;
const PIPE_BORE: f32 = 0.0082;
const CRATE_YAW: f32 = 0.6;

// Hit ids.
const NONE: u32 = 0u;
const WALL: u32 = 1u;
const PIPE: u32 = 2u;
const BORE: u32 = 3u;
const BUOY: u32 = 4u;
const CRATE: u32 = 5u;
const BALL: u32 = 6u;
const WATER: u32 = 7u;
const RIDGE: u32 = 8u;
const SKY: u32 = 9u;
const ROOM_HIT: u32 = 10u;
const EXIT: u32 = 11u;

struct Hit {
    t: f32,
    n: vec3<f32>,
    id: u32,
};

fn miss() -> Hit {
    return Hit(BIG, vec3<f32>(0.0, 0.0, 1.0), NONE);
}

// ---- layout, sized from the display ----------------------------------------

fn tank() -> vec3<f32> {
    return frame.tank.xyz;
}
fn buoy() -> vec4<f32> {
    let t = tank();
    let r = 0.42 * t.y;
    return vec4<f32>(-0.45 * t.x, -t.y + r, -0.55 * t.z, r);
}
fn ball() -> vec4<f32> {
    let t = tank();
    return vec4<f32>(-0.5 * t.x, 0.45 * t.y, 0.1, 0.13 * t.y);
}
fn crate_box() -> vec4<f32> {
    let t = tank();
    let s = 0.3 * t.y;
    return vec4<f32>(0.5 * t.x, -t.y + s, -0.75 * t.z, s);
}
// the pipe runs out of the left wall, through the glass, toward the viewer
fn pipe_a() -> vec3<f32> {
    let t = tank();
    return vec3<f32>(-t.x - 0.03, -0.35 * t.y, -0.35 * t.z);
}
fn pipe_b() -> vec3<f32> {
    let t = tank();
    return vec3<f32>(0.2 * t.x, -0.4 * t.y, 0.2);
}
// world <-> crate frame (a turn about Y)
fn to_crate(v: vec3<f32>) -> vec3<f32> {
    let c = cos(CRATE_YAW);
    let s = sin(CRATE_YAW);
    return vec3<f32>(c * v.x - s * v.z, v.y, s * v.x + c * v.z);
}
fn from_crate(v: vec3<f32>) -> vec3<f32> {
    let c = cos(CRATE_YAW);
    let s = sin(CRATE_YAW);
    return vec3<f32>(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

// ---- intersections ---------------------------------------------------------

fn sphere(ro: vec3<f32>, rd: vec3<f32>, s: vec4<f32>, id: u32) -> Hit {
    let oc = ro - s.xyz;
    let b = dot(oc, rd);
    let h = b * b - dot(oc, oc) + s.w * s.w;
    if (h < 0.0) {
        return miss();
    }
    let t = -b - sqrt(h);
    if (t < EPS) {
        return miss();
    }
    return Hit(t, (oc + rd * t) / s.w, id);
}

// A hollow capped cylinder: outer wall, annular end caps, and the bore.
fn pipe(ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    let a = pipe_a();
    let ba = pipe_b() - a;
    let oc = ro - a;
    let baba = dot(ba, ba);
    let bard = dot(ba, rd);
    let baoc = dot(ba, oc);
    let k2 = baba - bard * bard;
    let k1 = baba * dot(oc, rd) - baoc * bard;
    let k0 = baba * dot(oc, oc) - baoc * baoc - PIPE_R * PIPE_R * baba;
    var d = k1 * k1 - k2 * k0;
    if (d < 0.0) {
        return miss();
    }
    d = sqrt(d);
    var t = (-k1 - d) / k2;
    let y = baoc + t * bard;
    if (t > EPS && y > 0.0 && y < baba) {
        return Hit(t, (oc + rd * t - ba * y / baba) / PIPE_R, PIPE);
    }
    let cap = select(0.0, baba, y >= baba);
    t = (cap - baoc) / bard;
    let q = oc + rd * t - ba * cap / baba;
    if (t < EPS || dot(q, q) > PIPE_R * PIPE_R) {
        return miss();
    }
    if (dot(q, q) > PIPE_BORE * PIPE_BORE) {
        return Hit(t, normalize(ba) * select(-1.0, 1.0, cap > 0.0), PIPE);
    }
    // into the bore: its wall, seen from inside, is the far root
    let ki = baba * dot(oc, oc) - baoc * baoc - PIPE_BORE * PIPE_BORE * baba;
    let ti = (-k1 + sqrt(max(k1 * k1 - k2 * ki, 0.0))) / k2;
    let yi = baoc + ti * bard;
    if (yi > 0.0 && yi < baba) {
        return Hit(ti, -(oc + rd * ti - ba * yi / baba) / PIPE_BORE, BORE);
    }
    return Hit(t, -rd, BORE);
}

fn crate_hit(ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    let c = crate_box();
    let o = to_crate(ro - c.xyz);
    let d = to_crate(rd);
    let m = 1.0 / select(d, vec3<f32>(1e-9), abs(d) < vec3<f32>(1e-9));
    let k = abs(m) * c.w;
    let t1 = -m * o - k;
    let t2 = -m * o + k;
    let tn = max(max(t1.x, t1.y), t1.z);
    let tf = min(min(t2.x, t2.y), t2.z);
    if (tn > tf || tn < EPS) {
        return miss();
    }
    let n = -sign(d) * step(t1.yzx, t1.xyz) * step(t1.zxy, t1.xyz);
    return Hit(tn, from_crate(n), CRATE);
}

fn nearer(a: Hit, b: Hit) -> Hit {
    if (b.t < a.t) {
        return b;
    }
    return a;
}

fn objects(ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    var h = sphere(ro, rd, buoy(), BUOY);
    h = nearer(h, sphere(ro, rd, ball(), BALL));
    h = nearer(h, pipe(ro, rd));
    return nearer(h, crate_hit(ro, rd));
}

// Where a ray leaves the embrasure: a WALL, the back opening (EXIT) or the
// viewer's room (ROOM_HIT). A ray from in front of the glass enters through
// the glass or stays in the room.
fn enclosure(ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    let tk = tank();
    var t0 = 0.0;
    if (ro.z > 0.0) {
        if (rd.z >= 0.0) {
            return room(ro, rd);
        }
        t0 = -ro.z / rd.z;
        let g = ro + rd * t0;
        if (abs(g.x) > tk.x || abs(g.y) > tk.y) {
            return room(ro, rd);
        }
    }
    let p = ro + rd * t0;
    let tx = select(BIG, (select(-tk.x, tk.x, rd.x > 0.0) - p.x) / rd.x, abs(rd.x) > 1e-9);
    let ty = select(BIG, (select(-tk.y, tk.y, rd.y > 0.0) - p.y) / rd.y, abs(rd.y) > 1e-9);
    let tz = select((0.0 - p.z) / max(rd.z, 1e-9), (-tk.z - p.z) / rd.z, rd.z < 0.0);
    let ts = min(tx, ty);
    if (ts < tz) {
        let n = select(vec3<f32>(0.0, -sign(rd.y), 0.0), vec3<f32>(-sign(rd.x), 0.0, 0.0), tx < ty);
        return Hit(t0 + ts, n, WALL);
    }
    if (rd.z < 0.0) {
        return Hit(t0 + tz, -rd, EXIT);
    }
    let r = room(p + rd * tz, rd);
    return Hit(t0 + tz + r.t, r.n, ROOM_HIT);
}

// The inside of the room box, from a point inside it.
fn room(ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    let d = select(rd, vec3<f32>(1e-9), abs(rd) < vec3<f32>(1e-9));
    let t = (select(ROOM_MIN, ROOM_MAX, d > vec3<f32>(0.0)) - ro) / d;
    let tm = min(t.x, min(t.y, t.z));
    let n = -sign(d) * step(t, t.yzx) * step(t, t.zxy);
    return Hit(tm, n, ROOM_HIT);
}

// What the room looks like at p: the same riveted, rusted steel plate as
// the embrasure, a lamp above the viewer and a window behind them. Lit by
// a constant (it is only ever seen in reflections), so a texture fetch or
// two is all it costs.
fn room_radiance(p: vec3<f32>, n: vec3<f32>, lod: f32) -> vec3<f32> {
    if (n.y < -0.5 && all(abs(p.xz - vec2<f32>(0.0, 1.1)) < vec2<f32>(0.45, 0.2))) {
        return vec3<f32>(6.0, 5.4, 4.6);
    }
    if (n.z < -0.5 && all(abs(p.xy - vec2<f32>(0.2, 0.45)) < vec2<f32>(0.6, 0.4))) {
        // daylight, split by a mullion
        let bar = abs(p.x - 0.2) < 0.02 || abs(p.y - 0.45) < 0.02;
        return select(mix(HORIZON, ZENITH, clamp(p.y, 0.0, 1.0)) * 2.5, vec3<f32>(0.05), bar);
    }
    let m = material(WALL, p, n, lod, 1.0);
    return m.albedo * mix(ROOM_LIGHT, ROOM_LIGHT * 0.5, m.metal);
}

fn ridge_height(x: f32, k: i32, lod: f32) -> f32 {
    let s = textureSampleLevel(mat_tex, wrap, vec2<f32>(x / RIDGE_W[k] + 0.37 * f32(k), 0.21 + 0.29 * f32(k)), lod);
    return WATER_Y + RIDGE_H[k] * pow(clamp(0.6 * s.g + 0.4 * s.r, 0.0, 1.0), 2.2) * 1.6;
}

// Beyond the back opening: water, three ridges, sky.
fn far(ro: vec3<f32>, rd: vec3<f32>, fp: f32) -> Hit {
    var h = Hit(BIG, -rd, SKY);
    if (rd.y < 0.0) {
        h = Hit((WATER_Y - ro.y) / rd.y, vec3<f32>(0.0, 1.0, 0.0), WATER);
    }
    if (rd.z < 0.0) {
        for (var k = 0; k < 3; k = k + 1) {
            let t = (RIDGE_Z[k] - ro.z) / rd.z;
            let p = ro + rd * t;
            let lod = log2(max(t * fp / RIDGE_W[k] * TEXELS, 1.0));
            let y = ridge_height(p.x, k, lod);
            if (t < h.t && p.y < y) {
                let slope = (ridge_height(p.x + 0.002 * RIDGE_W[k], k, lod) - y) / (0.002 * RIDGE_W[k]);
                h = Hit(t, normalize(vec3<f32>(-slope, 0.35, 1.0)), RIDGE);
                break;
            }
        }
    }
    return h;
}

fn trace(ro: vec3<f32>, rd: vec3<f32>, fp: f32) -> Hit {
    if (ro.z < -tank().z - EPS) {
        return far(ro, rd, fp);
    }
    let o = objects(ro, rd);
    let e = enclosure(ro, rd);
    if (o.t < e.t) {
        return o;
    }
    if (e.id == EXIT) {
        return far(ro, rd, fp);
    }
    return e;
}

// ---- light -----------------------------------------------------------------

fn sky(rd: vec3<f32>, disc: bool) -> vec3<f32> {
    let mu = max(dot(rd, SUN), 0.0);
    var c = mix(HORIZON, ZENITH, 1.0 - exp(-max(rd.y, 0.0) * 9.0));
    c = c + vec3<f32>(1.0, 0.6, 0.3) * (0.5 * pow(mu, 8.0) + 2.5 * pow(mu, 120.0));
    if (disc && mu > SUN_COS) {
        c = c + vec3<f32>(900.0, 700.0, 450.0);
    }
    // clouds: one textured layer at 1.8 km
    if (rd.y > 0.01) {
        let p = rd * (1800.0 / rd.y);
        let s = textureSampleLevel(mat_tex, wrap, p.xz / 9000.0 + vec2<f32>(frame.eye.w * 2e-4, 0.0), 0.0);
        let dens = smoothstep(0.55, 0.85, s.g + 0.25 * s.r) * smoothstep(0.01, 0.12, rd.y);
        c = mix(c, mix(vec3<f32>(0.55, 0.45, 0.45), vec3<f32>(2.4, 1.6, 1.0), 0.3 + 0.7 * pow(mu, 4.0)), dens * 0.85);
    }
    return c;
}

fn haze(c: vec3<f32>, t: f32, rd: vec3<f32>) -> vec3<f32> {
    return mix(sky(rd * vec3<f32>(1.0, 0.0, 1.0) + vec3<f32>(0.0, 0.02, 0.0), false), c, exp(-t / HAZE_M));
}

// The Lambertian form factor of the opening rectangle at depth z (|x| <=
// hw, |y| <= hh) seen from point p with normal n: the fraction of p's
// irradiance a uniform radiance over that rectangle contributes.
fn form_factor(p: vec3<f32>, n: vec3<f32>, z: f32) -> f32 {
    let tk = tank();
    var v = array<vec3<f32>, 4>(
        normalize(vec3<f32>(-tk.x, -tk.y, z) - p),
        normalize(vec3<f32>(tk.x, -tk.y, z) - p),
        normalize(vec3<f32>(tk.x, tk.y, z) - p),
        normalize(vec3<f32>(-tk.x, tk.y, z) - p),
    );
    var s = 0.0;
    for (var i = 0; i < 4; i = i + 1) {
        let a = v[i];
        let b = v[(i + 1) % 4];
        let c = cross(a, b);
        s = s + acos(clamp(dot(a, b), -1.0, 1.0)) * dot(n, c) / max(length(c), 1e-6);
    }
    return clamp(abs(s) / (2.0 * PI), 0.0, 1.0);
}

// The sun reaches p only through the back opening (and, from in front,
// through the glass), past every object. The penumbra is the sun's angular
// radius times the distance to the edge that casts it.
fn sun_visibility(p: vec3<f32>) -> f32 {
    if (objects(p, SUN).t < BIG) {
        return 0.0;
    }
    let tk = tank();
    let tb = (-tk.z - p.z) / SUN.z;
    let b = p + SUN * tb;
    var vis = smoothstep(-1.0, 1.0, min(tk.x - abs(b.x), tk.y - abs(b.y)) / (0.0047 * tb + 1e-4));
    if (p.z > 0.0) {
        let tg = -p.z / SUN.z;
        let g = p + SUN * tg;
        vis = vis * smoothstep(-1.0, 1.0, min(tk.x - abs(g.x), tk.y - abs(g.y)) / (0.0047 * tg + 1e-4));
    }
    return vis;
}

// The scene as a distance bound, only for ambient occlusion.
fn sd_scene(p: vec3<f32>) -> f32 {
    let tk = tank();
    var d = select(1.0, min(tk.x - abs(p.x), tk.y - abs(p.y)), p.z < 0.0 && p.z > -tk.z);
    d = min(d, length(p - buoy().xyz) - buoy().w);
    d = min(d, length(p - ball().xyz) - ball().w);
    let a = pipe_a();
    let ba = pipe_b() - a;
    let h = clamp(dot(p - a, ba) / dot(ba, ba), 0.0, 1.0);
    d = min(d, length(p - a - ba * h) - PIPE_R);
    let c = crate_box();
    let q = abs(to_crate(p - c.xyz)) - vec3<f32>(c.w);
    return min(d, length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0));
}

fn occlusion(p: vec3<f32>, n: vec3<f32>) -> f32 {
    var o = 0.0;
    var w = 1.0;
    for (var i = 1; i <= 3; i = i + 1) {
        let h = 0.01 * f32(i * i);
        o = o + w * max(h - sd_scene(p + n * h), 0.0) / h;
        w = w * 0.6;
    }
    return clamp(1.0 - 0.6 * o, 0.0, 1.0);
}

// GGX / Smith / Schlick, times n.l.
fn ggx(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, rough: f32, f0: vec3<f32>) -> vec3<f32> {
    let h = normalize(v + l);
    let a2 = pow(rough, 4.0);
    let nh = max(dot(n, h), 0.0);
    let nv = max(dot(n, v), 1e-3);
    let nl = max(dot(n, l), 0.0);
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let k = rough * rough * 0.5;
    let g = nv / (nv * (1.0 - k) + k) * nl / (nl * (1.0 - k) + k);
    let f = f0 + (1.0 - f0) * pow(1.0 - max(dot(h, v), 0.0), 5.0);
    return a2 / (PI * dd * dd) * g * f / (4.0 * nv * max(nl, 1e-3)) * nl;
}

// ---- materials -------------------------------------------------------------

struct Material {
    albedo: vec3<f32>,
    rough: f32,
    metal: f32,
    n: vec3<f32>,
    emit: vec3<f32>,
};

// A plane surface's own 2D coordinates, and a 2D vector in them back in
// 3D: the two axes across the normal's dominant one.
fn plate_uv(p: vec3<f32>, n: vec3<f32>) -> vec2<f32> {
    let a = abs(n);
    if (a.x > a.y && a.x > a.z) {
        return p.zy;
    }
    return select(p.xy, p.zx, a.y > a.z);
}
fn plate_vec(v: vec2<f32>, n: vec3<f32>) -> vec3<f32> {
    let a = abs(n);
    if (a.x > a.y && a.x > a.z) {
        return vec3<f32>(0.0, v.y, v.x);
    }
    return select(vec3<f32>(v.x, v.y, 0.0), vec3<f32>(v.y, 0.0, v.x), a.y > a.z);
}

// Steel under rust. The rust mask is blotch noise, pushed up in crevices
// (occlusion), along streaks that run down vertical faces, in the seams of
// the wall plates, and pushed down where edges are worn to bright metal.
// Rust is a rough dielectric with relief; steel is a pitted metal.
fn material(id: u32, p: vec3<f32>, n: vec3<f32>, lod: f32, ao: f32) -> Material {
    let q = p * TILES_PER_M + vec3<f32>(f32(id) * 0.37);
    let w = pow(abs(n), vec3<f32>(4.0)) / dot(pow(abs(n), vec3<f32>(4.0)), vec3<f32>(1.0));
    let sx = textureSampleLevel(mat_tex, wrap, q.yz, lod);
    let sy = textureSampleLevel(mat_tex, wrap, q.zx, lod);
    let sz = textureSampleLevel(mat_tex, wrap, q.xy, lod);
    let s = sx * w.x + sy * w.y + sz * w.z;
    let gx = (sx.ba - 0.5) * 2.0 * SLOPE_RANGE;
    let gy = (sy.ba - 0.5) * 2.0 * SLOPE_RANGE;
    let gz = (sz.ba - 0.5) * 2.0 * SLOPE_RANGE;
    var grad = (vec3<f32>(0.0, gx.x, gx.y) * w.x + vec3<f32>(gy.y, 0.0, gy.x) * w.y + vec3<f32>(gz.x, gz.y, 0.0) * w.z) * TILES_PER_M;

    let streak = textureSampleLevel(mat_tex, wrap, vec2<f32>(q.x + q.z, q.y * 0.08), lod).g;
    var amount = 0.75;
    var wear = 0.0;
    var groove = 0.0;
    switch id {
        case BUOY: { amount = 0.72; }
        case BALL: { amount = 0.3; }
        case BORE: { amount = 0.85; }
        case PIPE: {
            amount = 0.7;
            // the cut end is machined steel
            wear = smoothstep(0.85, 0.95, abs(dot(n, normalize(pipe_b() - pipe_a()))));
        }
        case CRATE: {
            let c = crate_box();
            let e = c.w - abs(to_crate(p - c.xyz));
            let mid = e.x + e.y + e.z - min(e.x, min(e.y, e.z)) - max(e.x, max(e.y, e.z));
            wear = 1.0 - smoothstep(0.0, 0.004, mid);
        }
        case WALL: {
            amount = 0.5;
            // plates 80 x 60 mm, riveted at 20 mm along their seams
            let uv = plate_uv(p, n) / vec2<f32>(0.08, 0.06);
            let f = abs(fract(uv) - 0.5);
            let seam = min((0.5 - f.x) * 0.08, (0.5 - f.y) * 0.06);
            groove = 1.0 - smoothstep(0.0006, 0.0014, seam);
            let r = fract(uv * vec2<f32>(4.0, 3.0)) - 0.5;
            let rv = select(vec2<f32>(r.x * 0.02, (0.5 - f.y) * 0.06), vec2<f32>((0.5 - f.x) * 0.08, r.y * 0.02), (0.5 - f.x) * 0.08 < (0.5 - f.y) * 0.06);
            let rd = length(rv - vec2<f32>(0.0, 0.004));
            if (rd < 0.0028) {
                grad = grad - plate_vec(rv, n) * 900.0;
            }
        }
        default: {}
    }
    let m = s.g + 0.3 * (s.r - 0.5) + amount - 0.55 + 0.3 * (streak - 0.5) * (1.0 - abs(n.y)) + 0.6 * (1.0 - ao) + 0.5 * groove - wear;
    // rust grain: the height at four times the scale, on the dominant face
    let big = select(select(q.xy, q.zx, w.y > w.z), q.yz, w.x > max(w.y, w.z)) * 4.0;
    let gs = textureSampleLevel(mat_tex, wrap, big, lod + 2.0);
    let grain = gs.r;
    let rust = smoothstep(0.44, 0.56, m + 0.6 * (grain - 0.5) + 0.3 * (s.r - 0.5));
    // relief: pitted steel is nearly flat, rust scale stands up to a
    // millimetre or so, its grain on top (at four times the frequency)
    let grain_grad = plate_vec((gs.ba - 0.5) * 2.0 * SLOPE_RANGE, n) * TILES_PER_M * 4.0;
    grad = grad * mix(0.0004, 0.0016, rust) + grain_grad * rust * 0.0004;
    let nb = normalize(n - (grad - n * dot(grad, n)));
    let rust_col = mix(vec3<f32>(0.10, 0.035, 0.014), vec3<f32>(0.46, 0.19, 0.065), clamp(0.6 * s.r + 0.6 * grain - 0.1, 0.0, 1.0)) * mix(0.75, 1.1, s.g);
    // bare steel: a bright metal; the wall plates and the ball are
    // polished nearly to a mirror, the rest is smooth but pitted
    let polished = id == BALL || id == WALL || wear > 0.5;
    let steel = vec3<f32>(0.62, 0.63, 0.64) * (0.85 + 0.15 * s.r);
    var emit = vec3<f32>(0.0);
    if (id == WALL && p.z < EPS && frame.tank.w > 0.5) {
        let g = abs(fract(plate_uv(p, n) * 20.0) - 0.5);
        emit = vec3<f32>(0.2, 0.9, 1.0) * 1.5 * smoothstep(0.46, 0.49, max(g.x, g.y));
    }
    return Material(mix(steel, rust_col, rust) * (1.0 - 0.6 * groove), mix(select(0.18 + 0.15 * (1.0 - s.r), 0.04 + 0.08 * (1.0 - s.r), polished), 0.88, rust), 1.0 - rust, nb, emit);
}

// What the embrasure reflects in direction r, for a rough surface: no
// trace, just where r leaves.
fn ambient_specular(p: vec3<f32>, r: vec3<f32>) -> vec3<f32> {
    let e = enclosure(p, r);
    if (e.id == ROOM_HIT) {
        return room_radiance(p + r * e.t, e.n, 5.0);
    }
    return select(BOUNCE, sky(r, false) * 0.8, e.id == EXIT);
}

struct Shaded {
    c: vec3<f32>,
    // what a mirror bounce from here passes on (zero: no bounce)
    f: vec3<f32>,
    n: vec3<f32>,
};

fn shade(h: Hit, ro: vec3<f32>, rd: vec3<f32>, path: f32, fp: f32, last: bool) -> Shaded {
    let none = vec3<f32>(0.0);
    let p = ro + rd * h.t;
    switch h.id {
        // the sun disc only where it is seen directly: in a reflection one
        // ray per pixel would catch it as aliased glints, so reflecting
        // surfaces take the sun as a specular highlight instead
        case SKY: { return Shaded(sky(rd, path == 0.0), none, h.n); }
        case ROOM_HIT: { return Shaded(room_radiance(p, h.n, log2(max((path + h.t) * fp * TILES_PER_M * TEXELS, 1.0))), none, h.n); }
        case RIDGE: {
            // backlit rock and forest: sky light from above, the sun only on
            // slopes that turn toward it
            let lit = ZENITH * (0.6 + 0.4 * h.n.y) + SUN_E * max(dot(h.n, SUN), 0.0);
            return Shaded(haze(vec3<f32>(0.05, 0.06, 0.055) * lit, h.t, rd), none, h.n);
        }
        case WATER: {
            let lod = log2(max((path + h.t) * fp / max(-rd.y, 0.02) * TEXELS / 9.0, 1.0));
            let s1 = textureSampleLevel(mat_tex, wrap, p.xz / 25.0 + vec2<f32>(frame.eye.w * 0.004, 0.0), lod);
            let s2 = textureSampleLevel(mat_tex, wrap, p.xz / 9.0 - vec2<f32>(0.0, frame.eye.w * 0.007), lod);
            let g = (s1.ba - 0.5) * 2.0 * SLOPE_RANGE / 25.0 + (s2.ba - 0.5) * 2.0 * SLOPE_RANGE / 9.0;
            let n = normalize(vec3<f32>(-g.x * 0.025, 1.0, -g.y * 0.025));
            let f = 0.02 + 0.98 * pow(1.0 - max(dot(n, -rd), 0.0), 5.0);
            let glitter = ggx(n, -rd, SUN, 0.12, vec3<f32>(0.02)) * SUN_E * 4.0;
            var c = haze(vec3<f32>(0.01, 0.025, 0.03) * (1.0 - f) + glitter, h.t, rd);
            if (last) {
                c = c + f * sky(reflect(rd, n), false) * exp(-h.t / HAZE_M);
            }
            return Shaded(c, select(vec3<f32>(f * exp(-h.t / HAZE_M)), none, last), n);
        }
        default: {}
    }
    // ambient occlusion only where it is seen directly: in a reflection it
    // is lost in the blur and would double the cost of every mirror pixel
    let ao = select(occlusion(p, h.n), 1.0, last);
    let lod = log2(max((path + h.t) * fp / max(abs(dot(h.n, rd)), 0.2) * TILES_PER_M * TEXELS, 1.0));
    let m = material(h.id, p, h.n, lod, ao);
    let n = m.n;
    let v = -rd;
    let f0 = mix(vec3<f32>(0.04), m.albedo, m.metal);
    let diffuse = m.albedo * (1.0 - m.metal);

    // the sun
    let vis = sun_visibility(p + h.n * EPS);
    var c = (diffuse / PI * max(dot(n, SUN), 0.0) + ggx(n, v, SUN, m.rough, f0)) * SUN_E * vis;

    // sky, room and walls by the share of the hemisphere each covers
    // in a reflection, a fixed share of sky stands in for the two exact
    // form factors: the reflection is blurred and dimmed by Fresnel anyway
    let tk = tank();
    var fb = 0.15;
    var ff = 0.2;
    if (!last) {
        fb = form_factor(p, n, -tk.z);
        ff = form_factor(p, n, 0.0);
    }
    let through_back = sky(vec3<f32>(0.0, 0.2, -0.98), false) * fb;
    var amb = select(through_back + ROOM * ff + BOUNCE * max(1.0 - fb - ff, 0.0), through_back + BOUNCE * max(ff - fb, 0.0) + ROOM * (1.0 - ff), p.z > 0.0);
    if (h.id == BORE) {
        amb = amb * 0.15;
    }
    c = c + diffuse * amb * ao;

    // Reflections. One ray sees one direction, but a rough surface reflects
    // a cone of them: the reflection fades toward the light averaged over
    // the hemisphere (`amb`) as roughness grows. Steel up to fairly rough is
    // traced, so it shows the scene around it; dielectric rust only takes
    // the cheap lookup.
    let nv = max(dot(n, v), 0.0);
    let fr = f0 + (max(vec3<f32>(1.0 - m.rough), f0) - f0) * pow(1.0 - nv, 5.0);
    let blur = smoothstep(0.1, 0.45, m.rough);
    var bounce = none;
    if (m.metal > 0.5 && m.rough < 0.45 && !last) {
        bounce = fr * ao * (1.0 - blur);
        c = c + fr * amb * ao * blur;
    } else {
        c = c + fr * mix(ambient_specular(p, reflect(rd, n)), amb, blur) * ao;
    }
    return Shaded(c + m.emit, bounce, n);
}

// ---- the pass --------------------------------------------------------------

struct FullOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_scene(@builtin(vertex_index) i: u32) -> FullOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    let ndc = uv * 2.0 - 1.0;
    return FullOut(vec4<f32>(ndc, 0.0, 1.0), ndc);
}

@fragment
fn fs_scene(v: FullOut) -> @location(0) vec4<f32> {
    let a = frame.inv_view_proj * vec4<f32>(v.ndc, 0.0, 1.0);
    let b = frame.inv_view_proj * vec4<f32>(v.ndc, 1.0, 1.0);
    let rd0 = normalize(b.xyz / b.w - a.xyz / a.w);
    // the angle one pixel subtends, for texture filtering (here, while
    // control flow is still uniform)
    let fp = length(fwidth(rd0));

    var ro = frame.eye.xyz;
    var rd = rd0;
    var c = vec3<f32>(0.0);
    var through = vec3<f32>(1.0);
    var path = 0.0;
    for (var bounce = 0; bounce < 2; bounce = bounce + 1) {
        let h = trace(ro, rd, fp);
        let s = shade(h, ro, rd, path, fp, bounce == 1);
        c = c + through * s.c;
        if (all(s.f == vec3<f32>(0.0))) {
            break;
        }
        through = through * s.f;
        path = path + h.t;
        ro = ro + rd * h.t + s.n * EPS * 10.0;
        rd = reflect(rd, s.n);
    }
    // a camera: exposure, the ACES filmic curve (Narkowicz's fit), and the
    // lens's vignette; the target is sRGB, so this is display-referred
    let x = c * EXPOSURE;
    let mapped = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(mapped * (1.0 - 0.18 * dot(v.ndc, v.ndc)), 1.0);
}
