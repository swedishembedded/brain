// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  splat geometry objective per pixel: robust log-depth, normal prior / self-consistency, distortion
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import camera
// @import splat_view
//
// The geometry half of a fit's objective, one invocation per pixel, from the
// ray renderer's `aux` = {expected range D, composited normal N = sum w n,
// distortion} and alpha. Writes the upstream gradients the ray backward takes
// (`daux`: dL/d accumulated range, dL/dN, dL/d distortion) and adds the
// expected-range normalizer's share to dimg's alpha, and one partial sum per
// workgroup of each term.
//
// Every term is a weighted MEAN with the same normalizer as the photometric
// loss (the supervised pixels' weight sum), so each weight reads as a ratio
// against RGB whatever the image size and mask.
//
// * Depth, against a prior `target.depth` (RANGE along the pixel's ray; 0 =
//   none) with per-pixel confidence: the pseudo-Huber of the LOG ratio,
//   delta^2 (sqrt(1 + (r / delta)^2) - 1), r = log D - log D*. A log is what
//   makes the error scale-free along a ray - a centimetre at arm's length and
//   a metre across a valley are the same mistake - and the pseudo-Huber keeps
//   a prior's heavy tail (a wrong match, a reflection, a sky pixel) from
//   dominating: past delta it pulls with bounded force.
// * Normals, against a prior `target.n` (camera frame, facing the camera;
//   zero = none): w (alpha - N . n*), which is sum_i w_i (1 - n_i . n*) - the
//   2DGS normal consistency, linear in what the renderer composites. Where
//   there is NO prior, the target is the normal of the rendered range map
//   itself (held fixed, from its neighbours' 3D points), the 2DGS
//   self-consistency - weighted separately, because a surface that agrees
//   with itself can still be the wrong surface, and an external prior, where
//   there is one, is the evidence that says which.
// * Distortion: the rendered 2DGS depth distortion itself.

struct Weights {
    depth: f32,
    normal: f32,
    normal_self: f32,
    distortion: f32,
    delta: f32,      // pseudo-Huber knee, in log range
    inv_wsum: f32,   // 1 / (sum of supervision weights)
    min_alpha: f32,  // below this a pixel's range is not a surface
    has_weights: u32,
};

struct Params {
    v: View,
    w: Weights,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       aux:     array<f32>; // W*H*5
@group(0) @binding(2) var<storage, read>       img:     array<f32>; // W*H*4
@group(0) @binding(3) var<storage, read>       tgt:     array<f32>; // W*H*5 {depth, conf, n}
@group(0) @binding(4) var<storage, read>       wts:     array<f32>; // W*H supervision weight
@group(0) @binding(5) var<storage, read_write> dimg:    array<f32>; // W*H*4 (+= alpha share)
@group(0) @binding(6) var<storage, read_write> daux:    array<f32>; // W*H*5 (written)
@group(0) @binding(7) var<storage, read_write> partial: array<f32>; // n_wg*4

var<workgroup> acc: array<f32, 256>; // 4 terms x 64

// The 3D point a pixel's expected range puts on its ray, or w = 0.
fn surface(x: i32, y: i32) -> vec4<f32> {
    if (x < 0 || y < 0 || x >= i32(p.v.width) || y >= i32(p.v.height)) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
    let pix = u32(y) * p.v.width + u32(x);
    if (img[pix * 4u + 3u] < p.w.min_alpha) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
    let r = pixel_ray(p.v, vec2<f32>(f32(x) + 0.5, f32(y) + 0.5));
    if (r.ok == 0.0) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
    return vec4<f32>(r.o + aux[pix * 5u] * r.d, 1.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let pix = gid.y * (nwg.x * 64u) + gid.x;
    let t = lid.x;
    let n_pix = p.v.width * p.v.height;
    var l_depth = 0.0;
    var l_normal = 0.0;
    var l_self = 0.0;
    var l_dist = 0.0;
    if (pix < n_pix) {
        var wgt = 1.0;
        if (p.w.has_weights != 0u) { wgt = wts[pix]; }
        let alpha = img[pix * 4u + 3u];
        let d = aux[pix * 5u];
        let nsum = vec3<f32>(aux[pix * 5u + 1u], aux[pix * 5u + 2u], aux[pix * 5u + 3u]);
        var g_dacc = 0.0;
        var g_n = vec3<f32>(0.0, 0.0, 0.0);
        var g_alpha = 0.0;
        let s = wgt * p.w.inv_wsum;
        if (wgt > 0.0) {
            // ---- depth ----
            let dt = tgt[pix * 5u];
            let conf = tgt[pix * 5u + 1u];
            if (p.w.depth > 0.0 && dt > 0.0 && conf > 0.0 && alpha >= p.w.min_alpha && d > 0.0) {
                let r = log(d) - log(dt);
                let q = r / p.w.delta;
                let root = sqrt(1.0 + q * q);
                l_depth = p.w.depth * s * conf * p.w.delta * p.w.delta * (root - 1.0);
                // dL/dD, then split between the accumulated range and alpha:
                // D = Dacc / alpha
                let g_d = p.w.depth * s * conf * (r / root) / d;
                g_dacc = g_d / alpha;
                g_alpha = g_alpha - g_d * d / alpha;
            }
            // ---- normals ----
            let nt = vec3<f32>(tgt[pix * 5u + 2u], tgt[pix * 5u + 3u], tgt[pix * 5u + 4u]);
            if (dot(nt, nt) > 0.0) {
                if (p.w.normal > 0.0) {
                    let k = p.w.normal * s;
                    l_normal = k * (alpha - dot(nsum, nt));
                    g_n = g_n - k * nt;
                    g_alpha = g_alpha + k;
                }
            } else if (p.w.normal_self > 0.0 && alpha >= p.w.min_alpha) {
                let x = i32(pix % p.v.width);
                let y = i32(pix / p.v.width);
                let xl = surface(x - 1, y);
                let xr = surface(x + 1, y);
                let yu = surface(x, y - 1);
                let yd = surface(x, y + 1);
                if (xl.w * xr.w * yu.w * yd.w > 0.0) {
                    var ns = cross(xr.xyz - xl.xyz, yd.xyz - yu.xyz);
                    let len = length(ns);
                    if (len > 0.0) {
                        ns = ns / len;
                        // face the camera: against the pixel's own ray
                        let me = pixel_ray(p.v, vec2<f32>(f32(x) + 0.5, f32(y) + 0.5));
                        if (dot(ns, me.d) > 0.0) { ns = -ns; }
                        let k = p.w.normal_self * s;
                        l_self = k * (alpha - dot(nsum, ns));
                        g_n = g_n - k * ns;
                        g_alpha = g_alpha + k;
                    }
                }
            }
            // ---- distortion ----
            if (p.w.distortion > 0.0) {
                let k = p.w.distortion * s;
                l_dist = k * aux[pix * 5u + 4u];
                daux[pix * 5u + 4u] = k;
            } else {
                daux[pix * 5u + 4u] = 0.0;
            }
        } else {
            daux[pix * 5u + 4u] = 0.0;
        }
        daux[pix * 5u] = g_dacc;
        daux[pix * 5u + 1u] = g_n.x;
        daux[pix * 5u + 2u] = g_n.y;
        daux[pix * 5u + 3u] = g_n.z;
        dimg[pix * 4u + 3u] = dimg[pix * 4u + 3u] + g_alpha;
    }
    acc[t] = l_depth;
    acc[64u + t] = l_normal;
    acc[128u + t] = l_self;
    acc[192u + t] = l_dist;
    workgroupBarrier();
    if (t < 4u) {
        var sum = 0.0;
        for (var j = 0u; j < 64u; j = j + 1u) { sum = sum + acc[t * 64u + j]; }
        partial[(wid.y * nwg.x + wid.x) * 4u + t] = sum;
    }
}
