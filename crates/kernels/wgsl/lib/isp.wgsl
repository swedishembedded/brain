// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// The photometric camera model on the device: the WGSL twin of the per-pixel
// chain in `crates/splat/src/isp.rs`, which is its test oracle
// (`crates/splat/tests/s17_camera_model.rs` holds the two to each other).
// Imported with `// @import isp`.
//
//   radiance L
//     x gain (per view: 2^exposure * white balance)
//     x vignetting 1 + a1 r^2 + a2 r^4 + a3 r^6 (per sensor, per channel, floored)
//     -> colour correction matrix M, rows summing to 1 (per sensor)
//     -> response curve, monotone piecewise linear (per sensor, per channel)
//     -> sRGB OETF (only when a scene-linear fit meets an encoded photograph)
//     -> bilateral grid (per view, optional): a 3x4 affine colour transform
//        sliced trilinearly at the pixel's position and the luma of what the
//        global chain produced
//
// The response curve's knots sit one stop apart: 0, 2^-7, 2^-6, ... 2^-1, and
// its last segment runs on past 1/2 with its own slope, the first one down past
// 0. Its value is sum_j s_j portion_j(y), portion_j being how much of segment j
// lies below y, so d f / d s_j = portion_j with no search for the segment.
//
// Swedish Embedded AB implements photometrically calibrated 3D reconstruction
// for its clients. If your team needs captures with changing exposure, white
// balance or tone mapping turned into consistent scenes, you can procure our
// services by sending an email to info@swedishembedded.com.

struct IspView {
    width: u32,
    height: u32,
    mode: u32,     // isp_pixel: 0 forward, 1 backward
    encode: u32,   // apply the sRGB OETF
    cx: f32,       // principal point, pixels
    cy: f32,
    inv_norm: f32, // 1 / (half-diagonal)^2
    has_grid: u32,
    gx: u32,       // bilateral grid cells along x, y and luma (each >= 2)
    gy: u32,
    gz: u32,
    pad: u32,
};

// One view's camera, resolved from the fit's parameters on the host.
struct IspCam {
    gain: vec4<f32>,          // per channel
    a1: vec4<f32>,            // vignetting r^2 coefficient per channel
    a2: vec4<f32>,            // r^4
    a3: vec4<f32>,            // r^6
    m0: vec4<f32>,            // colour correction rows
    m1: vec4<f32>,
    m2: vec4<f32>,
    s: array<vec4<f32>, 8>,   // response slope of each segment, per channel
};

// Normalized squared radius of pixel (x, y): 1 at the corners of a frame whose
// principal point is centred.
fn isp_r2(v: IspView, x: u32, y: u32) -> f32 {
    let dx = f32(x) + 0.5 - v.cx;
    let dy = f32(y) + 0.5 - v.cy;
    return (dx * dx + dy * dy) * v.inv_norm;
}

// Lens transmission per channel, floored at 0.05.
fn isp_vig(c: IspCam, r2: f32) -> vec3<f32> {
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let t = vec3<f32>(1.0, 1.0, 1.0) + c.a1.xyz * r2 + c.a2.xyz * r4 + c.a3.xyz * r6;
    return max(t, vec3<f32>(0.05, 0.05, 0.05));
}

// How much of response segment `j` lies below y, per channel.
fn isp_portion(j: u32, y: vec3<f32>) -> vec3<f32> {
    if (j == 0u) {
        return min(y, vec3<f32>(0.0078125, 0.0078125, 0.0078125));
    }
    // segment j (1..7) starts at t = 2^(j - 8)
    var t = 0.0078125;
    for (var k = 1u; k < j; k = k + 1u) {
        t = t * 2.0;
    }
    let lo = y - vec3<f32>(t, t, t);
    if (j == 7u) {
        return max(lo, vec3<f32>(0.0, 0.0, 0.0));
    }
    return clamp(lo, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(t, t, t));
}

fn isp_crf(c: IspCam, y: vec3<f32>) -> vec3<f32> {
    var f = vec3<f32>(0.0, 0.0, 0.0);
    for (var j = 0u; j < 8u; j = j + 1u) {
        f = f + c.s[j].xyz * isp_portion(j, y);
    }
    return f;
}

// d response / d y per channel: the slope of the segment y lies in.
fn isp_crf_slope(c: IspCam, y: vec3<f32>) -> vec3<f32> {
    var d = c.s[0].xyz;
    var t = 0.0078125;
    for (var j = 1u; j < 8u; j = j + 1u) {
        let s = c.s[j].xyz;
        if (y.x >= t) { d.x = s.x; }
        if (y.y >= t) { d.y = s.y; }
        if (y.z >= t) { d.z = s.z; }
        t = t * 2.0;
    }
    return d;
}

// sRGB OETF, extended linearly past both ends.
fn isp_oetf(x: f32) -> f32 {
    if (x <= 0.0031308) { return 12.92 * x; }
    if (x <= 1.0) { return 1.055 * pow(x, 1.0 / 2.4) - 0.055; }
    return 1.0 + (1.055 / 2.4) * (x - 1.0);
}

fn isp_oetf_grad(x: f32) -> f32 {
    if (x <= 0.0031308) { return 12.92; }
    if (x <= 1.0) { return (1.055 / 2.4) * pow(x, 1.0 / 2.4 - 1.0); }
    return 1.055 / 2.4;
}

// Every stage of the global chain at one pixel, kept for the backward.
struct IspTrace {
    l: vec3<f32>,   // radiance
    vig: vec3<f32>, // transmission
    x: vec3<f32>,   // gain * vig * L
    m: vec3<f32>,   // M x
    y: vec3<f32>,   // response
    e: vec3<f32>,   // encoded: what the grid (or the photograph) sees
};

fn isp_global(v: IspView, c: IspCam, l: vec3<f32>, r2: f32) -> IspTrace {
    var t: IspTrace;
    t.l = l;
    t.vig = isp_vig(c, r2);
    t.x = c.gain.xyz * t.vig * l;
    t.m = vec3<f32>(dot(c.m0.xyz, t.x), dot(c.m1.xyz, t.x), dot(c.m2.xyz, t.x));
    t.y = isp_crf(c, t.m);
    t.e = t.y;
    if (v.encode != 0u) {
        t.e = vec3<f32>(isp_oetf(t.y.x), isp_oetf(t.y.y), isp_oetf(t.y.z));
    }
    return t;
}

// The grid's guide: Rec.709 luma of the global output.
fn isp_luma(e: vec3<f32>) -> f32 {
    return 0.2126 * e.x + 0.7152 * e.y + 0.0722 * e.z;
}

// Continuous grid coordinates of pixel (x, y) with guide `luma`: cell i sits
// at coordinate i, the frame spans [0, g - 1] along x and y and luma in [0, 1]
// spans [0, gz - 1].
fn isp_grid_coord(v: IspView, x: u32, y: u32, luma: f32) -> vec3<f32> {
    return vec3<f32>(
        (f32(x) + 0.5) / f32(v.width) * f32(v.gx - 1u),
        (f32(y) + 0.5) / f32(v.height) * f32(v.gy - 1u),
        clamp(luma, 0.0, 1.0) * f32(v.gz - 1u));
}

// The lower corner cell of the trilinear stencil along one axis of `g` cells.
fn isp_grid_base(u: f32, g: u32) -> u32 {
    return min(u32(max(floor(u), 0.0)), g - 2u);
}
