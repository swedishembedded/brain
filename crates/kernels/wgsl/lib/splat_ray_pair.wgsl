// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// One (pixel, gaussian) pair of the ray-evaluated splat renderer. Imported by
// the rasterizer, by both backward kernels (through `splat_ray_window`, which
// holds the walks over a pixel's pairs, forward and backward) and by the tile
// sort (`splat_ray_emit`), so that what a pixel composites and what its
// gradient differentiates are one definition: a gaussian that one of them
// counts and the other skips would make every gradient through that pixel
// wrong.
//
// A gaussian arrives as its ray record (`splat_ray_project.wgsl`, 16 words):
// mean m, the six entries of its inverse covariance A (a00 a11 a22 a01 a02
// a12), unit normal n, compensated opacity op', colour. Along a ray o + t d it
// responds exp(-q/2) at the closest point: t* = d.A(m - o) / d.A d, e = m - o
// - t* d, q = e.A e.
//
// Compositing a pixel front to back in order of t* (`splat_ray_window`), a
// pair with alpha < 1/255 is skipped and the walk stops BEFORE the pair that
// would take transmittance to 1e-4 or below (gsplat semantics). Per pixel the
// walk accumulates colour sum w c, normal sum w n, range sum w t, weight sum
// w and the 2DGS distortion sum_{j<k} 2 w_j w_k (t_k - t_j), each pair adding
// 2 w (t sum w - sum w t).
//
// Backward, per pair (e the closest-point offset; the envelope theorem makes
// t* stationary in q):
//   dq/dm = 2 A e       dq/dA = e e^T      dq/dd = -2 t* A e    dq/do = -dq/dm
//   dt/dm = A d / v     dt/dA = sym(d e^T) / v
//   dt/dd = (A e - t* A d) / v                                  dt/do = -dt/dm
// The distortion is homogeneous of degree 2 in the weights, so its suffix over
// everything behind a pair is 2 L - (what the walk has passed), known from the
// totals alone. A pair's gradient is 17 channels: {dL/dm (3), dL/dA (6, a00
// a11 a22 a01 a02 a12), dL/dopacity', dL/dn (3), dL/dcolour (3), |dL/dm|}.
// The last is the per-pixel magnitude density control ranks by (AbsGS),
// which a sum over pixels cannot recover.

struct RayRec {
    m: vec3<f32>,
    a_diag: vec3<f32>, // a00 a11 a22
    a_off: vec3<f32>,  // a01 a02 a12
    n: vec3<f32>,
    op: f32,
    col: vec3<f32>,
};

struct RayPair {
    hit: bool,
    alpha: f32,
    gv: f32, // exp(-q/2)
    t: f32,
    vv: f32,
    e: vec3<f32>,
    ae: vec3<f32>,
    ad: vec3<f32>,
};

// A record from its four aligned vectors: {m, a00}, {a11, a22, a01, a02},
// {a12, n}, {op', colour} - read as vectors, a record is four loads, not 16.
fn ray_rec4(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>) -> RayRec {
    return RayRec(v0.xyz, vec3<f32>(v0.w, v1.x, v1.y), vec3<f32>(v1.z, v1.w, v2.x), v2.yzw, v3.x, v3.yzw);
}

fn ray_rec_sym(r: RayRec, x: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(r.a_diag.x * x.x + r.a_off.x * x.y + r.a_off.y * x.z,
                     r.a_off.x * x.x + r.a_diag.y * x.y + r.a_off.z * x.z,
                     r.a_off.y * x.x + r.a_off.z * x.y + r.a_diag.z * x.z);
}

fn ray_pair(r: RayRec, o: vec3<f32>, d: vec3<f32>) -> RayPair {
    var pr: RayPair;
    pr.hit = false;
    pr.ad = ray_rec_sym(r, d);
    pr.vv = dot(d, pr.ad);
    if (pr.vv <= 0.0) { return pr; }
    pr.t = dot(r.m - o, pr.ad) / pr.vv;
    if (pr.t <= 0.0) { return pr; }
    pr.e = r.m - o - pr.t * d;
    pr.ae = ray_rec_sym(r, pr.e);
    pr.gv = exp(-0.5 * dot(pr.e, pr.ae));
    pr.alpha = min(0.99, r.op * pr.gv);
    pr.hit = pr.alpha >= 1.0 / 255.0;
    return pr;
}
