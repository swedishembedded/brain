// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  ray-evaluated splat backward: per-pixel partials into tile slots
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import camera
// @import splat_view
//
// The ray renderer's backward, first stage, over a BAND of tiles
// [tile0, tile0 + n/256) and instances [k0, k1) - the slot layout and banding
// of `splat_bwd_slots.wgsl`, so `splat_bwd_tile_reduce` and everything after
// it are shared. One thread per pixel replays the forward walk of
// `splat_ray_rasterize.wgsl` twice: once for the totals, once subtracting
// each contribution to get the strict suffixes the alpha gradient needs.
//
// Upstream (all per pixel): dimg = dL/d(rgb, alpha out); daux = dL/d(the
// ACCUMULATED range sum w t, the accumulated normal sum w n (3), the
// distortion). A caller supervising EXPECTED range splits its gradient
// between daux[0] and dimg's alpha (`splat::renderer::add_expected_depth_vjp`).
//
// Per (pair) derivatives, e the closest-point offset and t* the range of the
// maximum (the envelope theorem makes t* stationary in q):
//   dq/dm = 2 A e       dq/dA = e e^T      dq/dd = -2 t* A e    dq/do = -dq/dm
//   dt/dm = A d / v     dt/dA = sym(d e^T) / v
//   dt/dd = (A e - t* A d) / v                                  dt/do = -dt/dm
// The distortion sum_{j<k} 2 w_j w_k (t_k - t_j) is homogeneous of degree 2
// in the weights, so its suffix over everything behind a pair is
// 2 L - (what the walk has passed), known from the totals alone.
//
// Channels (17 per instance, per pixel, in slots[((k - k0) * 17 + c) * 256 +
// local]): {dL/dm (3), dL/dA (6, a00 a11 a22 a01 a02 a12), dL/dopacity',
// dL/dn (3), dL/dcolour (3), |dL/dm|}. The last is the per-pixel magnitude
// density control ranks by (AbsGS), which a per-instance sum cannot recover.
//
// With `want_ray` set, every pixel also writes dL/d(its ray origin) and
// dL/d(its ray direction) to dray[pix*6] - what `splat_ray_camera_grad`
// turns into pose, rolling-shutter and lens gradients. A tile split across
// bands writes them from the band that reaches its list's end, the only one
// that walks all of it.

struct Band {
    tile0: u32,
    k0: u32,
    k1: u32,
    n: u32,        // threads in the band: tiles * 256
    want_ray: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Params {
    v: View,
    b: Band,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       ray:    array<f32>; // N*16
@group(0) @binding(2) var<storage, read>       vals:   array<u32>; // sorted instance -> gaussian id
@group(0) @binding(3) var<storage, read>       ranges: array<u32>; // n_tiles*2
@group(0) @binding(4) var<storage, read>       dimg:   array<f32>; // W*H*4
@group(0) @binding(5) var<storage, read>       daux:   array<f32>; // W*H*5
@group(0) @binding(6) var<storage, read_write> slots:  array<f32>; // band_instances*17*256
@group(0) @binding(7) var<storage, read_write> dray:   array<f32>; // W*H*6

const CH: u32 = 17u;

// Everything one pair of (pixel, gaussian) needs, recomputed identically in
// both passes.
struct Pair {
    hit: bool,
    alpha: f32,
    gv: f32,     // exp(-q/2)
    op: f32,
    t: f32,
    vv: f32,
    e: vec3<f32>,
    ae: vec3<f32>,
    ad: vec3<f32>,
};

fn eval_pair(g: u32, o: vec3<f32>, d: vec3<f32>) -> Pair {
    var pr: Pair;
    pr.hit = false;
    let r = g * 16u;
    let dm = vec3<f32>(ray[r], ray[r + 1u], ray[r + 2u]) - o;
    let a00 = ray[r + 3u];
    let a11 = ray[r + 4u];
    let a22 = ray[r + 5u];
    let a01 = ray[r + 6u];
    let a02 = ray[r + 7u];
    let a12 = ray[r + 8u];
    pr.ad = vec3<f32>(a00 * d.x + a01 * d.y + a02 * d.z,
                      a01 * d.x + a11 * d.y + a12 * d.z,
                      a02 * d.x + a12 * d.y + a22 * d.z);
    pr.vv = dot(d, pr.ad);
    if (pr.vv <= 0.0) { return pr; }
    pr.t = dot(dm, pr.ad) / pr.vv;
    if (pr.t <= 0.0) { return pr; }
    pr.e = dm - pr.t * d;
    pr.ae = vec3<f32>(a00 * pr.e.x + a01 * pr.e.y + a02 * pr.e.z,
                      a01 * pr.e.x + a11 * pr.e.y + a12 * pr.e.z,
                      a02 * pr.e.x + a12 * pr.e.y + a22 * pr.e.z);
    let q = dot(pr.e, pr.ae);
    pr.gv = exp(-0.5 * q);
    pr.op = ray[r + 12u];
    pr.alpha = min(0.99, pr.op * pr.gv);
    pr.hit = pr.alpha >= 1.0 / 255.0;
    return pr;
}

fn col_of(g: u32) -> vec3<f32> {
    let r = g * 16u;
    return vec3<f32>(ray[r + 13u], ray[r + 14u], ray[r + 15u]);
}

fn nrm_of(g: u32) -> vec3<f32> {
    let r = g * 16u;
    return vec3<f32>(ray[r + 9u], ray[r + 10u], ray[r + 11u]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.b.n) { return; }
    let tiles_x = (p.v.width + 15u) / 16u;
    let tile = p.b.tile0 + idx / 256u;
    let local = idx % 256u;
    let px = (tile % tiles_x) * 16u + local % 16u;
    let py = (tile / tiles_x) * 16u + local / 16u;
    let inside = px < p.v.width && py < p.v.height;
    let start = ranges[tile * 2u];
    let end = ranges[tile * 2u + 1u];
    let pr = pixel_ray(p.v, vec2<f32>(f32(px) + 0.5, f32(py) + 0.5));
    let live0 = inside && pr.ok != 0.0;
    let o = pr.o;
    let d = pr.d;

    var gc = vec3<f32>(0.0, 0.0, 0.0);
    var ga = 0.0;
    var gdep = 0.0;
    var gn = vec3<f32>(0.0, 0.0, 0.0);
    var gl = 0.0;
    if (inside) {
        let pix = py * p.v.width + px;
        gc = vec3<f32>(dimg[pix * 4u], dimg[pix * 4u + 1u], dimg[pix * 4u + 2u]);
        ga = dimg[pix * 4u + 3u];
        gdep = daux[pix * 5u];
        gn = vec3<f32>(daux[pix * 5u + 1u], daux[pix * 5u + 2u], daux[pix * 5u + 3u]);
        gl = daux[pix * 5u + 4u];
    }

    // ---- pass 1: totals ----
    var tr = 1.0;
    var sc = vec3<f32>(0.0, 0.0, 0.0);
    var sn = vec3<f32>(0.0, 0.0, 0.0);
    var sd = 0.0;   // sum w t
    var sw = 0.0;   // sum w
    var ltot = 0.0; // distortion
    if (live0) {
        for (var j = start; j < end; j = j + 1u) {
            let g = vals[j];
            let q = eval_pair(g, o, d);
            if (!q.hit) { continue; }
            let next_t = tr * (1.0 - q.alpha);
            if (next_t <= 1e-4) { break; }
            let w = q.alpha * tr;
            sc = sc + w * col_of(g);
            sn = sn + w * nrm_of(g);
            ltot = ltot + 2.0 * w * (q.t * sw - sd);
            sd = sd + w * q.t;
            sw = sw + w;
            tr = next_t;
        }
    }
    let t_final = tr;
    let d_tot = sd;
    let w_tot = sw;
    let sdist_all = 2.0 * gl * ltot;
    let bgdot = dot(p.v.bg.xyz, gc);

    // ---- pass 2: the same walk, gradients per instance ----
    tr = 1.0;
    var live = live0;
    var pw = 0.0;    // prefix sum w (strictly before)
    var pq = 0.0;    // prefix sum w t
    var rdist = 0.0; // prefix sum of w * dL/dw from the distortion, inclusive
    var g_o = vec3<f32>(0.0, 0.0, 0.0);
    var g_d = vec3<f32>(0.0, 0.0, 0.0);
    let stop = min(end, p.b.k1);
    for (var j = start; j < stop; j = j + 1u) {
        var part: array<f32, 17>;
        for (var c = 0u; c < CH; c = c + 1u) { part[c] = 0.0; }
        if (live) {
            let g = vals[j];
            let q = eval_pair(g, o, d);
            if (q.hit) {
                let next_t = tr * (1.0 - q.alpha);
                if (next_t <= 1e-4) {
                    live = false;
                } else {
                    let w = q.alpha * tr;
                    let cg = col_of(g);
                    let ng = nrm_of(g);
                    sc = sc - w * cg;
                    sn = sn - w * ng;
                    sd = sd - w * q.t;
                    let ps = w_tot - pw - w;         // strict suffix of w
                    let qs = d_tot - pq - w * q.t;   // strict suffix of w t
                    let gw_dist = gl * 2.0 * (q.t * pw - pq + qs - q.t * ps);
                    rdist = rdist + w * gw_dist;
                    let sdist = sdist_all - rdist;
                    let gw = dot(cg, gc) + q.t * gdep + dot(ng, gn) + gw_dist;
                    let om = 1.0 - q.alpha;
                    let behind = dot(sc, gc) + sd * gdep + dot(sn, gn) + sdist;
                    let g_alpha = gw * tr - behind / om + (t_final / om) * (ga - bgdot);
                    let g_t = w * gdep + gl * 2.0 * w * (pw - ps);
                    var g_q = 0.0;
                    var g_op = 0.0;
                    if (q.op * q.gv < 0.99) {
                        g_op = q.gv * g_alpha;
                        g_q = -0.5 * q.alpha * g_alpha;
                    }
                    let ivv = 1.0 / q.vv;
                    let gm = g_q * 2.0 * q.ae + (g_t * ivv) * q.ad;
                    let e = q.e;
                    let de = g_t * ivv;
                    part[0] = gm.x;
                    part[1] = gm.y;
                    part[2] = gm.z;
                    part[3] = g_q * e.x * e.x + de * d.x * e.x;
                    part[4] = g_q * e.y * e.y + de * d.y * e.y;
                    part[5] = g_q * e.z * e.z + de * d.z * e.z;
                    part[6] = g_q * 2.0 * e.x * e.y + de * (d.x * e.y + d.y * e.x);
                    part[7] = g_q * 2.0 * e.x * e.z + de * (d.x * e.z + d.z * e.x);
                    part[8] = g_q * 2.0 * e.y * e.z + de * (d.y * e.z + d.z * e.y);
                    part[9] = g_op;
                    part[10] = w * gn.x;
                    part[11] = w * gn.y;
                    part[12] = w * gn.z;
                    part[13] = w * gc.x;
                    part[14] = w * gc.y;
                    part[15] = w * gc.z;
                    part[16] = length(gm);
                    g_o = g_o - gm;
                    g_d = g_d - (2.0 * g_q * q.t) * q.ae + de * (q.ae - q.t * q.ad);
                    pw = pw + w;
                    pq = pq + w * q.t;
                    tr = next_t;
                }
            }
        }
        if (j >= p.b.k0) {
            let base = (j - p.b.k0) * CH * 256u + local;
            for (var c = 0u; c < CH; c = c + 1u) {
                slots[base + c * 256u] = part[c];
            }
        }
    }
    if (p.b.want_ray != 0u && p.b.k1 >= end && inside) {
        let pix = (py * p.v.width + px) * 6u;
        dray[pix] = g_o.x;
        dray[pix + 1u] = g_o.y;
        dray[pix + 2u] = g_o.z;
        dray[pix + 3u] = g_d.x;
        dray[pix + 4u] = g_d.y;
        dray[pix + 5u] = g_d.z;
    }
}
