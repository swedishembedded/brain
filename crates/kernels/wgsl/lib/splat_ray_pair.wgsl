// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// One (pixel, gaussian) pair of the ray-evaluated splat renderer, forward and
// backward. Imported by the rasterizer and by both backward kernels, so that
// what a pixel composites and what its gradient differentiates are one
// definition: a gaussian that one of them counts and the other skips would
// make every gradient through that pixel wrong.
//
// A gaussian arrives as its ray record (`splat_ray_project.wgsl`, 16 words):
// mean m, the six entries of its inverse covariance A (a00 a11 a22 a01 a02
// a12), unit normal n, compensated opacity op', colour. Along a ray o + t d it
// responds exp(-q/2) at the closest point: t* = d.A(m - o) / d.A d, e = m - o
// - t* d, q = e.A e.
//
// Compositing a pixel front to back, a pair with alpha < 1/255 is skipped and
// the walk stops BEFORE the pair that would take transmittance to 1e-4 or
// below (gsplat semantics). Per pixel the walk accumulates colour sum w c,
// normal sum w n, range sum w t, weight sum w and the 2DGS distortion
// sum_{j<k} 2 w_j w_k (t_k - t_j), each pair adding 2 w (t sum w - sum w t).
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

// A pixel's walk: its ray, and the sums so far. `live` turns false when the
// pixel has no ray or its walk has stopped.
struct RayWalk {
    o: vec3<f32>,
    d: vec3<f32>,
    live: bool,
    tr: f32,
    col: vec3<f32>,
    nsum: vec3<f32>,
    dsum: f32, // sum w t
    wsum: f32, // sum w
    dist: f32,
};

fn ray_walk(o: vec3<f32>, d: vec3<f32>, live: bool) -> RayWalk {
    var w: RayWalk;
    w.o = o;
    w.d = d;
    w.live = live;
    w.tr = 1.0;
    w.col = vec3<f32>(0.0, 0.0, 0.0);
    w.nsum = vec3<f32>(0.0, 0.0, 0.0);
    w.dsum = 0.0;
    w.wsum = 0.0;
    w.dist = 0.0;
    return w;
}

// Composite one gaussian into a walk.
fn ray_walk_step(s: RayWalk, r: RayRec) -> RayWalk {
    var w = s;
    if (!w.live) { return w; }
    let q = ray_pair(r, w.o, w.d);
    if (!q.hit) { return w; }
    let next_t = w.tr * (1.0 - q.alpha);
    if (next_t <= 1e-4) {
        w.live = false;
        return w;
    }
    let wt = q.alpha * w.tr;
    w.col = w.col + wt * r.col;
    w.nsum = w.nsum + wt * r.n;
    w.dist = w.dist + 2.0 * wt * (q.t * w.wsum - w.dsum);
    w.dsum = w.dsum + wt * q.t;
    w.wsum = w.wsum + wt;
    w.tr = next_t;
    return w;
}

// The backward's per-pixel state: the upstream gradient, the totals of the
// first walk (as suffixes, decreasing as the second walk passes pairs), the
// prefixes of the second walk, and the gradient of the pixel's ray.
struct RayBack {
    o: vec3<f32>,
    d: vec3<f32>,
    live: bool,
    tr: f32,
    // upstream: dL/d(rgb), dL/d(sum w t), dL/d(sum w n), dL/d(distortion),
    // and the alpha output's share t_final (dL/dalpha - bg . dL/drgb)
    gc: vec3<f32>,
    gdep: f32,
    gn: vec3<f32>,
    gl: f32,
    galpha: f32,
    // totals of the first walk
    w_tot: f32,
    d_tot: f32,
    sdist_all: f32,
    // suffixes (strictly behind the pairs passed so far)
    sc: vec3<f32>,
    sn: vec3<f32>,
    sd: f32,
    // prefixes (strictly before)
    pw: f32,
    pq: f32,
    rdist: f32, // prefix sum of w * dL/dw from the distortion, inclusive
    g_o: vec3<f32>,
    g_d: vec3<f32>,
};

// Start the second walk from the first. `dimg` = dL/d(rgb, alpha out),
// `daux` = dL/d(sum w t, sum w n (3), distortion), `bg` the background.
fn ray_back(w: RayWalk, live0: bool, dimg: vec4<f32>, daux0: vec4<f32>, gl: f32, bg: vec3<f32>) -> RayBack {
    var b: RayBack;
    b.o = w.o;
    b.d = w.d;
    b.live = live0;
    b.tr = 1.0;
    b.gc = dimg.xyz;
    b.gdep = daux0.x;
    b.gn = daux0.yzw;
    b.gl = gl;
    b.galpha = w.tr * (dimg.w - dot(bg, dimg.xyz));
    b.w_tot = w.wsum;
    b.d_tot = w.dsum;
    b.sdist_all = 2.0 * gl * w.dist;
    b.sc = w.col;
    b.sn = w.nsum;
    b.sd = w.dsum;
    b.pw = 0.0;
    b.pq = 0.0;
    b.rdist = 0.0;
    b.g_o = vec3<f32>(0.0, 0.0, 0.0);
    b.g_d = vec3<f32>(0.0, 0.0, 0.0);
    return b;
}

struct RayBackStep {
    b: RayBack,
    part: array<f32, 17>,
};

// Differentiate one pair of the second walk; `part` is the pair's 17
// channels, zero when the pair contributes nothing.
fn ray_back_step(s: RayBack, r: RayRec) -> RayBackStep {
    var out: RayBackStep;
    out.b = s;
    for (var c = 0u; c < 17u; c = c + 1u) { out.part[c] = 0.0; }
    if (!s.live) { return out; }
    let q = ray_pair(r, s.o, s.d);
    if (!q.hit) { return out; }
    let next_t = s.tr * (1.0 - q.alpha);
    if (next_t <= 1e-4) {
        out.b.live = false;
        return out;
    }
    var b = s;
    let d = b.d;
    let w = q.alpha * b.tr;
    b.sc = b.sc - w * r.col;
    b.sn = b.sn - w * r.n;
    b.sd = b.sd - w * q.t;
    let ps = b.w_tot - b.pw - w;         // strict suffix of w
    let qs = b.d_tot - b.pq - w * q.t;   // strict suffix of w t
    let gw_dist = b.gl * 2.0 * (q.t * b.pw - b.pq + qs - q.t * ps);
    b.rdist = b.rdist + w * gw_dist;
    let sdist = b.sdist_all - b.rdist;
    let gw = dot(r.col, b.gc) + q.t * b.gdep + dot(r.n, b.gn) + gw_dist;
    let om = 1.0 - q.alpha;
    let behind = dot(b.sc, b.gc) + b.sd * b.gdep + dot(b.sn, b.gn) + sdist;
    let g_alpha = gw * b.tr - behind / om + b.galpha / om;
    let g_t = w * b.gdep + b.gl * 2.0 * w * (b.pw - ps);
    var g_q = 0.0;
    var g_op = 0.0;
    if (r.op * q.gv < 0.99) {
        g_op = q.gv * g_alpha;
        g_q = -0.5 * q.alpha * g_alpha;
    }
    let de = g_t / q.vv;
    let gm = g_q * 2.0 * q.ae + de * q.ad;
    let e = q.e;
    out.part[0] = gm.x;
    out.part[1] = gm.y;
    out.part[2] = gm.z;
    out.part[3] = g_q * e.x * e.x + de * d.x * e.x;
    out.part[4] = g_q * e.y * e.y + de * d.y * e.y;
    out.part[5] = g_q * e.z * e.z + de * d.z * e.z;
    out.part[6] = g_q * 2.0 * e.x * e.y + de * (d.x * e.y + d.y * e.x);
    out.part[7] = g_q * 2.0 * e.x * e.z + de * (d.x * e.z + d.z * e.x);
    out.part[8] = g_q * 2.0 * e.y * e.z + de * (d.y * e.z + d.z * e.y);
    out.part[9] = g_op;
    out.part[10] = w * b.gn.x;
    out.part[11] = w * b.gn.y;
    out.part[12] = w * b.gn.z;
    out.part[13] = w * b.gc.x;
    out.part[14] = w * b.gc.y;
    out.part[15] = w * b.gc.z;
    out.part[16] = length(gm);
    b.g_o = b.g_o - gm;
    b.g_d = b.g_d - (2.0 * g_q * q.t) * q.ae + de * (q.ae - q.t * q.ad);
    b.pw = b.pw + w;
    b.pq = b.pq + w * q.t;
    b.tr = next_t;
    out.b = b;
    return out;
}
