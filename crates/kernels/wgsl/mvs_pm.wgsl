// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  PatchMatch MVS half-iteration: checkerboard propagation, joint view selection, refinement
// @how   one thread per pixel of one checkerboard colour; serial patch x view x hypothesis loops
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import camera
// @import mvs
//
// One colour of one red-black PatchMatch iteration over a reference view
// (Schönberger et al., "Pixelwise View Selection for Unstructured Multi-View
// Stereo", ECCV 2016, for the per-pixel plane hypotheses and the bilaterally
// weighted NCC; Xu & Tao, "Multi-Scale Geometric Consistency Guided Multi-View
// Stereo", CVPR 2019, for the adaptive checkerboard propagation and the
// multi-hypothesis joint view selection).
//
// Hypotheses. Pixel p holds (r, n): the plane through r d_p with normal n,
// d_p the unit ray through p's centre (`rays`), in the reference frame. A
// pixel only ever reads pixels of the OTHER colour, which this dispatch does
// not write, so updating `state` in place is race-free.
//
// Photometric cost (Schönberger §4.1). The patch is sampled every `step`
// pixels within `radius` of p; sample q's ray meets the plane at
// (n.(r d_p) / n.d_q) d_q, which is carried into source view k by its rigid
// transform and imaged through its OWN lens (`lens_project`) - the exact
// plane-induced warp for any lens, no rectification and no homography. Gray
// values are compared by NCC with bilateral weights
//   w_q = exp(-(g_q - g_p)^2 / 2 sigma_c^2 - |q - p|^2 / 2 sigma_s^2)
// from the reference alone; cost = 1 - NCC in [0, 2]. A patch whose weighted
// variance is below `min_var` in either view carries no evidence and costs 2.
//
// Adaptive propagation (Xu & Tao §3.1). Around p, eight regions of the other
// colour: along each image axis a near V-shaped region (the pixel one step
// out, then pairs fanning out to either side, 7 pixels) and a far strip
// (every second pixel from 3 to 23 out, 11 pixels). Each region contributes
// the plane of its lowest-cost pixel, re-intersected with p's own ray.
//
// Joint view selection (Xu & Tao §3.2). The current hypothesis and the eight
// candidates are scored against every source view: a view is trusted when at
// least two hypotheses match it well (cost < tau_good) and at most three
// badly (cost > tau_bad) - one lucky match is not evidence of visibility, an
// occluded view matches nothing - and weighted by exp(-c^2 / 2 sigma^2) of
// its best good cost. With no view trusted, every view is weighted by its
// best cost alone. A hypothesis's cost is the weighted mean over views.
//
// Geometric consistency (flag bit 0; Xu & Tao §3.3, after Schönberger
// §4.3). Each view's term adds geo_weight * min(e, geo_max): the hypothesis's
// point is imaged into source k, k's current range at that pixel is taken
// along k's ray to it, and that point is imaged back into the reference - e
// is its distance, in pixels, from p's centre. Zero exactly when the two
// maps agree, whatever the pixel quantization.
//
// Refinement. The winner is perturbed (range by a relative factor, normal by
// a random rotation of up to `perturb_normal` radians) and re-drawn at random,
// in six combinations, each scored with the same view weights; the best of
// all is kept. Flag bit 1 instead only re-scores the current hypothesis
// (after initialisation or upsampling), writing its cost.
//
// state: five planes - range, normal x, y, z (reference frame), cost.
// gray:  the reference's gray plane.
// quad:  source k's gray image in slot k, packed for bilinear sampling by
//        `mvs_quad` (one 8-byte load per sample).
// depth: source k's current range plane in slot k (geometric term only).

struct Params {
    w: u32,
    h: u32,
    plane: u32,
    nsrc: u32,
    color: u32,
    pass_id: u32,
    seed: u32,
    flags: u32,
    rmin: f32,
    rmax: f32,
    sigma_color: f32,
    sigma_spatial: f32,
    radius: i32,
    step: i32,
    pad0: u32,
    pad1: u32,
    tau_good: f32,
    tau_bad: f32,
    sel_sigma: f32,
    min_var: f32,
    geo_weight: f32,
    geo_max: f32,
    perturb_range: f32,
    perturb_normal: f32,
    lens: Lens,
    src: array<MvsCam, 16>,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       gray:  array<f32>;
@group(0) @binding(2) var<storage, read>       rays:  array<f32>;
@group(0) @binding(3) var<storage, read>       quad:  array<vec2<u32>>;
@group(0) @binding(4) var<storage, read>       depth: array<f32>;
@group(0) @binding(5) var<storage, read_write> state: array<f32>;

const MAXS: u32 = 16u;
const NH: u32 = 9u;
const BAD: f32 = 2.0;

fn ray_at(i: u32) -> vec3<f32> {
    return vec3<f32>(rays[i], rays[p.plane + i], rays[2u * p.plane + i]);
}

// Source k's gray value at continuous pixel `uv`, bilinear, clamped to the
// border.
fn src_gray(k: u32, uv: vec2<f32>) -> f32 {
    let dims = p.src[k].dims;
    let fx = clamp(uv.x - 0.5, 0.0, f32(dims.x - 1u));
    let fy = clamp(uv.y - 0.5, 0.0, f32(dims.y - 1u));
    let x0 = u32(fx);
    let y0 = u32(fy);
    let q = quad[k * p.plane + y0 * dims.x + x0];
    let s = 1.0 / 65535.0;
    let top = mix(f32(q.x & 65535u) * s, f32(q.x >> 16u) * s, fx - f32(x0));
    let bot = mix(f32(q.y & 65535u) * s, f32(q.y >> 16u) * s, fx - f32(x0));
    return mix(top, bot, fy - f32(y0));
}

fn inside(k: u32, uv: vec2<f32>) -> bool {
    let dims = p.src[k].dims;
    return uv.x >= 0.0 && uv.y >= 0.0 && uv.x < f32(dims.x) && uv.y < f32(dims.y);
}

// 1 - bilaterally weighted NCC of the plane (x0, n) between the reference
// patch around pixel (px, py) and source view k.
fn photo_cost(k: u32, px: i32, py: i32, x0: vec3<f32>, n: vec3<f32>) -> f32 {
    let cam = p.src[k];
    let centre = lens_project(cam.lens, mvs_to_cam(cam, x0));
    if (centre.ok == 0.0 || !inside(k, centre.uv)) { return BAD; }
    let w = i32(p.w);
    let h = i32(p.h);
    let gc = gray[u32(py * w + px)];
    let nd = dot(n, x0);
    let ic = 1.0 / (2.0 * p.sigma_color * p.sigma_color);
    let is = 1.0 / (2.0 * p.sigma_spatial * p.sigma_spatial);
    var sw = 0.0;
    var sr = 0.0;
    var ss = 0.0;
    var srr = 0.0;
    var sss = 0.0;
    var srs = 0.0;
    for (var dy = -p.radius; dy <= p.radius; dy = dy + p.step) {
        let qy = py + dy;
        if (qy < 0 || qy >= h) { continue; }
        for (var dx = -p.radius; dx <= p.radius; dx = dx + p.step) {
            let qx = px + dx;
            if (qx < 0 || qx >= w) { continue; }
            let qi = u32(qy * w + qx);
            let d = ray_at(qi);
            let den = dot(n, d);
            // also rejects the zero ray of a pixel the lens has no ray for
            if (den > -1e-4) { continue; }
            let pr = lens_project(cam.lens, mvs_to_cam(cam, d * (nd / den)));
            if (pr.ok == 0.0) { return BAD; }
            let gs = src_gray(k, pr.uv);
            let gr = gray[qi];
            let dg = gr - gc;
            let wt = exp(-dg * dg * ic - f32(dx * dx + dy * dy) * is);
            sw = sw + wt;
            sr = sr + wt * gr;
            ss = ss + wt * gs;
            srr = srr + wt * gr * gr;
            sss = sss + wt * gs * gs;
            srs = srs + wt * gr * gs;
        }
    }
    if (sw <= 0.0) { return BAD; }
    let inv = 1.0 / sw;
    let mr = sr * inv;
    let ms = ss * inv;
    let vr = srr * inv - mr * mr;
    let vs = sss * inv - ms * ms;
    if (vr < p.min_var || vs < p.min_var) { return BAD; }
    let ncc = (srs * inv - mr * ms) / sqrt(vr * vs);
    return clamp(1.0 - ncc, 0.0, BAD);
}

// Forward-backward reprojection error of reference point x0 (pixel centre
// pc) through source k's current range map, truncated at geo_max.
fn geo_cost(k: u32, x0: vec3<f32>, pc: vec2<f32>) -> f32 {
    let cam = p.src[k];
    let y = mvs_to_cam(cam, x0);
    let pr = lens_project(cam.lens, y);
    if (pr.ok == 0.0 || !inside(k, pr.uv)) { return p.geo_max; }
    let rk = depth[k * p.plane + u32(pr.uv.y) * cam.dims.x + u32(pr.uv.x)];
    if (rk <= 0.0) { return p.geo_max; }
    let back = lens_project(p.lens, mvs_from_cam(cam, normalize(y) * rk));
    if (back.ok == 0.0) { return p.geo_max; }
    return min(length(back.uv - pc), p.geo_max);
}

// A random range, uniform in inverse range over [rmin, rmax].
fn rand_range(rng: ptr<function, u32>) -> f32 {
    let a = 1.0 / p.rmax;
    let b = 1.0 / p.rmin;
    return 1.0 / (a + (b - a) * mvs_rand(rng));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = gid.y * (nwg.x * 64u) + gid.x;
    let hw = (p.w + 1u) / 2u;
    if (t >= hw * p.h) { return; }
    let y = t / hw;
    let x = 2u * (t % hw) + ((y + p.color) & 1u);
    if (x >= p.w) { return; }
    let i = y * p.w + x;
    let pl = p.plane;
    let d = ray_at(i);
    if (dot(d, d) == 0.0) {
        state[i] = 0.0;
        state[4u * pl + i] = BAD;
        return;
    }
    let px = i32(x);
    let py = i32(y);
    let pc = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);
    var rng = mvs_hash(i ^ mvs_hash(p.seed ^ mvs_hash(p.pass_id)));
    let geo = (p.flags & 1u) != 0u;
    let only_score = (p.flags & 2u) != 0u;

    var hr: array<f32, 9>;
    var hn: array<vec3<f32>, 9>;
    hr[0] = state[i];
    hn[0] = vec3<f32>(state[pl + i], state[2u * pl + i], state[3u * pl + i]);
    if (hr[0] <= 0.0 || dot(hn[0], d) >= 0.0) {
        hr[0] = rand_range(&rng);
        hn[0] = mvs_face(mvs_rand_dir(&rng), d);
    }
    var nh = 1u;
    if (!only_score) {
        let w = i32(p.w);
        let h = i32(p.h);
        for (var dir = 0u; dir < 4u; dir = dir + 1u) {
            // axis a and its perpendicular b
            var a = vec2<i32>(0, -1);
            if (dir == 1u) { a = vec2<i32>(0, 1); }
            if (dir == 2u) { a = vec2<i32>(-1, 0); }
            if (dir == 3u) { a = vec2<i32>(1, 0); }
            let b = vec2<i32>(-a.y, a.x);
            for (var region = 0u; region < 2u; region = region + 1u) {
                var best = 1e30;
                var bj = -1;
                let count = select(11, 7, region == 0u);
                for (var s = 0; s < count; s = s + 1) {
                    var o = a * (3 + 2 * s);
                    if (region == 0u) {
                        // V: a, then a (m + 1) +- b m for m = 1..3
                        let m = (s + 1) / 2;
                        let side = select(-1, 1, (s & 1) == 1);
                        o = a * (m + 1) + b * (m * side);
                        if (s == 0) { o = a; }
                    }
                    let q = vec2<i32>(px, py) + o;
                    if (q.x < 0 || q.y < 0 || q.x >= w || q.y >= h) { continue; }
                    let j = q.y * w + q.x;
                    let c = state[4u * pl + u32(j)];
                    if (state[u32(j)] > 0.0 && c < best) {
                        best = c;
                        bj = j;
                    }
                }
                if (bj < 0) { continue; }
                let j = u32(bj);
                let nj = vec3<f32>(state[pl + j], state[2u * pl + j], state[3u * pl + j]);
                let r = mvs_plane_range(nj, state[j] * ray_at(j), d);
                if (r > 0.0 && r < 4.0 * p.rmax) {
                    hr[nh] = r;
                    hn[nh] = nj;
                    nh = nh + 1u;
                }
            }
        }
    }

    // cost of every hypothesis against every source view - views outermost,
    // so the warps resident at any moment gather from ONE source image
    // instead of cycling all of them through the cache per pixel. Measured on
    // the finest level of a 1632x1224 capture: 1.11x from this order alone,
    // and it is what lets `mvs_quad`'s packed samples pay (a further 1.23x;
    // with hypotheses outermost the packing gained nothing)
    var cm: array<f32, 144>;
    for (var k = 0u; k < p.nsrc; k = k + 1u) {
        for (var hi = 0u; hi < nh; hi = hi + 1u) {
            cm[hi * MAXS + k] = photo_cost(k, px, py, hr[hi] * d, hn[hi]);
        }
    }

    // joint view selection
    var wk: array<f32, 16>;
    var trusted = false;
    let isel = 1.0 / (2.0 * p.sel_sigma * p.sel_sigma);
    for (var k = 0u; k < p.nsrc; k = k + 1u) {
        var good = 0u;
        var bad = 0u;
        var best_good = BAD;
        for (var hi = 0u; hi < nh; hi = hi + 1u) {
            let c = cm[hi * MAXS + k];
            if (c < p.tau_good) {
                good = good + 1u;
                best_good = min(best_good, c);
            }
            if (c > p.tau_bad) { bad = bad + 1u; }
        }
        wk[k] = 0.0;
        if (good >= 2u && bad <= 3u) {
            wk[k] = exp(-best_good * best_good * isel);
            trusted = true;
        }
    }
    if (!trusted) {
        for (var k = 0u; k < p.nsrc; k = k + 1u) {
            var best = BAD;
            for (var hi = 0u; hi < nh; hi = hi + 1u) { best = min(best, cm[hi * MAXS + k]); }
            wk[k] = exp(-best * best * isel);
        }
    }
    var wsum = 0.0;
    var wmax = 0.0;
    for (var k = 0u; k < p.nsrc; k = k + 1u) {
        wsum = wsum + wk[k];
        wmax = max(wmax, wk[k]);
    }
    // views weighing under 1% of the heaviest change nothing but the time taken
    for (var k = 0u; k < p.nsrc; k = k + 1u) {
        if (wk[k] < 0.01 * wmax) {
            wsum = wsum - wk[k];
            wk[k] = 0.0;
        }
    }
    let iw = 1.0 / max(wsum, 1e-30);

    var best_r = hr[0];
    var best_n = hn[0];
    var best_c = 1e30;
    for (var hi = 0u; hi < nh; hi = hi + 1u) {
        var s = 0.0;
        for (var k = 0u; k < p.nsrc; k = k + 1u) {
            if (wk[k] == 0.0) { continue; }
            var c = cm[hi * MAXS + k];
            if (geo) { c = c + p.geo_weight * geo_cost(k, hr[hi] * d, pc); }
            s = s + wk[k] * c;
        }
        s = s * iw;
        if (s < best_c) {
            best_c = s;
            best_r = hr[hi];
            best_n = hn[hi];
        }
    }

    if (!only_score) {
        let r0 = best_r;
        let n0 = best_n;
        let r_pert = r0 * exp(p.perturb_range * (2.0 * mvs_rand(&rng) - 1.0));
        let n_pert = normalize(n0 + p.perturb_normal * mvs_rand_dir(&rng));
        let r_rand = rand_range(&rng);
        let n_rand = mvs_face(mvs_rand_dir(&rng), d);
        var rr: array<f32, 6>;
        var rn: array<vec3<f32>, 6>;
        var rs: array<f32, 6>;
        for (var c = 0u; c < 6u; c = c + 1u) {
            var r = r0;
            var n = n0;
            if (c == 0u) { r = r_pert; }
            if (c == 1u) { n = n_pert; }
            if (c == 2u) { r = r_pert; n = n_pert; }
            if (c == 3u) { r = r_rand; }
            if (c == 4u) { n = n_rand; }
            if (c == 5u) { r = r_rand; n = n_rand; }
            rr[c] = r;
            rn[c] = n;
            // a plane seen this obliquely is not scored at all
            rs[c] = select(0.0, -1.0, dot(n, d) > -0.05);
        }
        // views outermost, as for the cost matrix above
        for (var k = 0u; k < p.nsrc; k = k + 1u) {
            if (wk[k] == 0.0) { continue; }
            for (var c = 0u; c < 6u; c = c + 1u) {
                if (rs[c] < 0.0) { continue; }
                var ck = photo_cost(k, px, py, rr[c] * d, rn[c]);
                if (geo) { ck = ck + p.geo_weight * geo_cost(k, rr[c] * d, pc); }
                rs[c] = rs[c] + wk[k] * ck;
            }
        }
        for (var c = 0u; c < 6u; c = c + 1u) {
            if (rs[c] < 0.0) { continue; }
            let s = rs[c] * iw;
            if (s < best_c) {
                best_c = s;
                best_r = rr[c];
                best_n = rn[c];
            }
        }
    }

    state[i] = best_r;
    state[pl + i] = best_n.x;
    state[2u * pl + i] = best_n.y;
    state[3u * pl + i] = best_n.z;
    state[4u * pl + i] = best_c;
}
