// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pop-free differentiable splat renderers for
// its clients. If your team needs expertise in gaussian splatting then you
// can procure our services by sending an email to info@swedishembedded.com.

// The order a pixel of the ray renderer composites in, and the walks forward
// and backward over it. Imported after `splat_ray_pair` by the rasterizer and
// by the backward kernels: the order is defined here once, so the backward
// replays exactly the order the forward composited.
//
// The light a pixel sees meets the gaussians in order of t*, the range of
// each one's maximum response along the pixel's own ray - an order of its
// own per pixel wherever gaussians overlap or intersect. A tile's list is
// sorted once for all of its pixels (`splat_ray_emit.wgsl`), and each pixel
// refines that order through a window of the RAY_WINDOW nearest pairs it has
// met but not composited, in the manner of StopThePop's hierarchical
// resorting (Radl et al., SIGGRAPH 2024):
//
//   walking the list, a pair that hits (alpha >= 1/255) joins the window;
//   when that makes RAY_WINDOW + 1, the nearest of them (the lowest t*) is
//   composited. At the end of the list the window is composited nearest
//   first.
//
// The pixel's order is therefore exactly its t* order whenever the list
// never brings a pair later than RAY_WINDOW of the pixel's own hits that are
// farther along its ray. Equal t* resolve by the fixed compare-exchange order
// below, identically in every walk. The window counts the pixel's own hits,
// not list positions: a pixel hits a small fraction of its tile's list, so a
// window of positions would span only a few of its pairs.
//
// Importers define `fn ray_record(g: u32) -> RayRec`, the record of
// gaussian g.

const RAY_WINDOW: u32 = 6u; // the slots s0..s5 of RayWindow

// Six, from measurement. A host model of this order over a trained
// 500k-gaussian scene at 816x612 left 483 of 3072 sampled pixels more than
// 0.02 in colour from their exact order under the list order of
// `splat_ray_emit` alone, 183 with a window of 4, 129 with 6, 93 with 8 and
// 24 with 16 (a list sorted by the range to the mean alone: 557). The window
// is worked for every list entry any lane of a warp hits, and its registers
// decide occupancy: on a P40 the rasterizer took 1.6x the time of plain list
// order at 6 and up to 2x at 8, and the backward's walk 1.3x more at 8 than
// at 6.

// A pair in or leaving a window: its range and alpha along the pixel's ray,
// its gaussian, and a tag the walk gives it (the backward's log position, or
// the pair's instance in the list).
struct RayHeld {
    t: f32,
    alpha: f32,
    g: u32,
    tag: u32,
};

// What compositing a pair reads of its gaussian.
struct RayShade {
    col: vec3<f32>,
    n: vec3<f32>,
};

fn ray_shade_of(r: RayRec) -> RayShade {
    return RayShade(r.col, r.n);
}

// An empty slot sorts below every pair (whose t* is positive), a drained one
// above; neither is a gaussian.
const RAY_EMPTY: f32 = -1.0;
const RAY_DRAINED: f32 = 3.0e38;
const RAY_NO_GAUSSIAN: u32 = 0xffffffffu;

fn ray_held_real(h: RayHeld) -> bool {
    return h.g != RAY_NO_GAUSSIAN;
}

// A pixel's window, sorted: s0 holds the nearest pair, empty slots first.
// Every slot is a named field and every step below is a fixed
// compare-exchange, so the window lives in registers and costs the same
// branch-free work in every lane of a warp.
//
// `shade0` is s0's shade, read as soon as a pair becomes s0 rather than when
// it is composited: s0 is the next pair to leave, the lanes of a warp read
// different gaussians' shades at different times, and a read issued at the
// composite that needs it would stall there.
struct RayWindow {
    s0: RayHeld,
    s1: RayHeld,
    s2: RayHeld,
    s3: RayHeld,
    s4: RayHeld,
    s5: RayHeld,
    n: u32,      // pairs held
    joined: u32, // pairs joined so far
    shade0: RayShade,
};

fn ray_window() -> RayWindow {
    let e = RayHeld(RAY_EMPTY, 0.0, RAY_NO_GAUSSIAN, 0u);
    let z = vec3<f32>(0.0, 0.0, 0.0);
    return RayWindow(e, e, e, e, e, e, 0u, 0u, RayShade(z, z));
}

// One compare-exchange: the slot keeps the farther of itself and x, and x
// moves on as the nearer.
struct RayCx {
    keep: RayHeld,
    x: RayHeld,
};

fn ray_cx(s: RayHeld, x: RayHeld) -> RayCx {
    if (x.t > s.t) { return RayCx(x, s); }
    return RayCx(s, x);
}

// A window after a join or a take, and the pair that left it with its shade,
// if a pair did.
struct RayStep {
    win: RayWindow,
    out: RayHeld,
    shade: RayShade,
    leaves: bool,
};

// Push h (shade hs) through the window from its far end: the window keeps the
// farthest RAY_WINDOW of itself and h, sorted, and the nearest comes out.
fn ray_window_push(s: RayWindow, h: RayHeld, hs: RayShade) -> RayStep {
    var st: RayStep;
    st.win = s;
    var c = ray_cx(s.s5, h);
    st.win.s5 = c.keep;
    c = ray_cx(s.s4, c.x);
    st.win.s4 = c.keep;
    c = ray_cx(s.s3, c.x);
    st.win.s3 = c.keep;
    c = ray_cx(s.s2, c.x);
    st.win.s2 = c.keep;
    c = ray_cx(s.s1, c.x);
    st.win.s1 = c.keep;
    c = ray_cx(s.s0, c.x);
    st.win.s0 = c.keep;
    st.out = c.x;
    st.leaves = ray_held_real(c.x);
    // A gaussian is in a tile's list once, so its id tells pairs apart. What
    // leaves is h or the old s0 but for equal ranges; the new s0 is h, the
    // old s0 or a pair whose shade has not been read.
    if (st.out.g == h.g) {
        st.shade = hs;
    } else if (st.out.g == s.s0.g) {
        st.shade = s.shade0;
    } else if (st.leaves) {
        st.shade = ray_shade_of(ray_record(st.out.g));
    }
    let g0 = st.win.s0.g;
    if (g0 == h.g) {
        st.win.shade0 = hs;
    } else if (g0 != s.s0.g && ray_held_real(st.win.s0)) {
        st.win.shade0 = ray_shade_of(ray_record(g0));
    }
    return st;
}

// Offer gaussian g's pair (record r) to the window.
fn ray_window_join(s: RayWindow, q: RayPair, r: RayRec, g: u32, tag: u32) -> RayStep {
    if (!q.hit) { return RayStep(s, RayHeld(RAY_EMPTY, 0.0, RAY_NO_GAUSSIAN, 0u), s.shade0, false); }
    var st = ray_window_push(s, RayHeld(q.t, q.alpha, g, tag), ray_shade_of(r));
    st.win.joined = s.joined + 1u;
    st.win.n = s.n + 1u - u32(st.leaves);
    return st;
}

// Take the nearest pair out of the window, if it holds one.
fn ray_window_take(s: RayWindow) -> RayStep {
    var st: RayStep;
    st.win = s;
    st.leaves = false;
    // empty slots come out first, as drained ones take their place
    loop {
        if (st.win.n == 0u || st.leaves) { break; }
        st = ray_window_push(st.win, RayHeld(RAY_DRAINED, 0.0, RAY_NO_GAUSSIAN, 0u), s.shade0);
    }
    st.win.n = s.n - u32(st.leaves);
    return st;
}

// ---- forward ----

// A pixel's walk: its ray, its window and the sums so far. `live` turns false
// when the pixel has no ray or its walk has stopped.
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
    win: RayWindow,
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
    w.win = ray_window();
    return w;
}

// Composite a pair that left the window.
fn ray_walk_composite(s: RayWalk, h: RayHeld, sh: RayShade) -> RayWalk {
    var w = s;
    let next_t = w.tr * (1.0 - h.alpha);
    if (next_t <= 1e-4) {
        w.live = false;
        return w;
    }
    let wt = h.alpha * w.tr;
    w.col = w.col + wt * sh.col;
    w.nsum = w.nsum + wt * sh.n;
    w.dist = w.dist + 2.0 * wt * (h.t * w.wsum - w.dsum);
    w.dsum = w.dsum + wt * h.t;
    w.wsum = w.wsum + wt;
    w.tr = next_t;
    return w;
}

// Walk the list's next gaussian, g (record r).
fn ray_walk_join(s: RayWalk, g: u32, r: RayRec) -> RayWalk {
    var w = s;
    if (!w.live) { return w; }
    let st = ray_window_join(w.win, ray_pair(r, w.o, w.d), r, g, 0u);
    w.win = st.win;
    if (st.leaves) { w = ray_walk_composite(w, st.out, st.shade); }
    return w;
}

// At the end of the list: composite what the window holds.
fn ray_walk_drain(s: RayWalk) -> RayWalk {
    var w = s;
    loop {
        if (!w.live) { break; }
        let st = ray_window_take(w.win);
        if (!st.leaves) { break; }
        w.win = st.win;
        w = ray_walk_composite(w, st.out, st.shade);
    }
    return w;
}

// ---- backward ----

// The backward's per-pixel state: the upstream gradient, the totals of the
// first walk (as suffixes, decreasing as the second walk passes pairs), the
// prefixes of the second walk, its window, and the gradient of the pixel's
// ray.
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
    win: RayWindow,
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
    b.win = ray_window();
    return b;
}

// What a pair's composite leaves for its gradient: dL/dalpha, dL/dt* and its
// weight w = alpha T - all zero for a pair the walk stopped at or never
// composited. The rest is the pair's geometry (`ray_pair_grad`).
struct RayBackPair {
    g: u32,
    tag: u32,
    g_alpha: f32,
    g_t: f32,
    w: f32,
};

// A second walk after one pair left its window, and that pair's gradient.
struct RayBackStep {
    b: RayBack,
    out: RayBackPair,
    leaves: bool,
};

// Composite a pair that left the window the second time: advance the
// prefixes and suffixes past it and return its gradient.
fn ray_back_composite(s: RayBack, h: RayHeld, sh: RayShade) -> RayBackStep {
    var st: RayBackStep;
    st.b = s;
    st.leaves = true;
    st.out = RayBackPair(h.g, h.tag, 0.0, 0.0, 0.0);
    let next_t = s.tr * (1.0 - h.alpha);
    if (next_t <= 1e-4) {
        st.b.live = false;
        return st;
    }
    var b = s;
    let t = h.t;
    let w = h.alpha * b.tr;
    b.sc = b.sc - w * sh.col;
    b.sn = b.sn - w * sh.n;
    b.sd = b.sd - w * t;
    let ps = b.w_tot - b.pw - w;         // strict suffix of w
    let qs = b.d_tot - b.pq - w * t;     // strict suffix of w t
    let gw_dist = b.gl * 2.0 * (t * b.pw - b.pq + qs - t * ps);
    b.rdist = b.rdist + w * gw_dist;
    let sdist = b.sdist_all - b.rdist;
    let gw = dot(sh.col, b.gc) + t * b.gdep + dot(sh.n, b.gn) + gw_dist;
    let behind = dot(b.sc, b.gc) + b.sd * b.gdep + dot(b.sn, b.gn) + sdist;
    st.out.g_alpha = gw * b.tr - (behind - b.galpha) / (1.0 - h.alpha);
    st.out.g_t = w * b.gdep + b.gl * 2.0 * w * (b.pw - ps);
    st.out.w = w;
    b.pw = b.pw + w;
    b.pq = b.pq + w * t;
    b.tr = next_t;
    st.b = b;
    return st;
}

// Walk the list's next gaussian, g (record r), the second time, tagging its
// pair `tag`; `leaves` when a pair left the window, whose gradient is `out`.
fn ray_back_join(s: RayBack, g: u32, tag: u32, r: RayRec) -> RayBackStep {
    var st: RayBackStep;
    st.b = s;
    st.leaves = false;
    if (!s.live) { return st; }
    let j = ray_window_join(s.win, ray_pair(r, s.o, s.d), r, g, tag);
    st.b.win = j.win;
    if (j.leaves) { st = ray_back_composite(st.b, j.out, j.shade); }
    return st;
}

// At the end of the list, or once the walk has stopped: the next pair the
// window lets go of, composited while the walk is live and with a zero
// gradient after. `leaves` is false once the window is empty.
fn ray_back_drain(s: RayBack) -> RayBackStep {
    var st: RayBackStep;
    st.b = s;
    st.leaves = false;
    let j = ray_window_take(s.win);
    if (!j.leaves) { return st; }
    st.b.win = j.win;
    if (st.b.live) {
        st = ray_back_composite(st.b, j.out, j.shade);
    } else {
        st.leaves = true;
        st.out = RayBackPair(j.out.g, j.out.tag, 0.0, 0.0, 0.0);
    }
    return st;
}

// A pair's gradient in the gaussian's 17 channels (`splat_ray_pair`), and its
// share of the gradient of the pixel's ray: from dL/dalpha, dL/dt* and w of
// its composite and the pair's geometry, evaluated again from record r along
// the pixel's ray (o, d) with upstream dL/d(rgb) `gc` and dL/d(sum w n) `gn`.
struct RayPairGrad {
    part: array<f32, 17>,
    g_o: vec3<f32>,
    g_d: vec3<f32>,
};

fn ray_pair_grad(r: RayRec, o: vec3<f32>, d: vec3<f32>, gc: vec3<f32>, gn: vec3<f32>, g_alpha: f32, g_t: f32, w: f32) -> RayPairGrad {
    var out: RayPairGrad;
    let q = ray_pair(r, o, d);
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
    out.part[10] = w * gn.x;
    out.part[11] = w * gn.y;
    out.part[12] = w * gn.z;
    out.part[13] = w * gc.x;
    out.part[14] = w * gc.y;
    out.part[15] = w * gc.z;
    out.part[16] = length(gm);
    out.g_o = -gm;
    out.g_d = -(2.0 * g_q * q.t) * q.ae + de * (q.ae - q.t * q.ad);
    return out;
}
