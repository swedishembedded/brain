// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  splat optimizer step: Adam on log-scale / logit / quaternion geometry, per-gaussian rates, SGLD, shape bounds
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// One Adam step (Kingma & Ba 2015) of a splat scene's geometry and opacity,
// one invocation per gaussian, entirely on the device.
//
// Parameters, packed 11 per gaussian in `geo`: {mean (3), log scale (3),
// quaternion (4, raw), opacity logit}. The renderer consumes ACTIVATED values
// (`splat_activate`), so `d_gauss` / `d_opac` arrive with respect to linear
// scale and opacity in [0,1] and are carried through the activations here:
//   d log s = d s * s,     d logit = d o * o (1 - o).
// A log scale and a logit are why nothing is clamped by projection any more:
// a step in log scale is a RELATIVE change of size, the same for a
// millimetre-thin disc as for a metre-wide blob, and a logit cannot leave
// (0, 1).
//
// Why this is not `adamw.wgsl`: the step each gaussian's POSITION takes can
// be in units of its OWN size (`rel_pos`: the step is `lr_pos` times its
// largest axis): Adam moves a parameter by about the
// learning rate whatever its gradient is, so one rate in world units is
// simultaneously a rounding error for a blob covering an unseeded object and
// a jump of several radii for a gaussian detail density control just
// subdivided - the second turns fine geometry into fog. That rate is per
// ELEMENT, which the tensor-generic optimizer has no notion of; fusing the
// activation chain, the rate, the SGLD noise and the shape bounds into the
// same per-gaussian pass is what keeps a fit's inner loop off the host.
//
// SGLD (3DGS-MCMC, Kheradmand et al. 2024, Eq. 8): with `noise > 0` each
// position is perturbed by `noise * step * sigma(-100 (o - 0.005)) * S ε / r`,
// `step` its position step and `r` its largest axis,
// S the gaussian's own covariance square root - transparent gaussians explore,
// opaque ones hold still. ε is a counter-based hash of (gaussian, step), so
// a fit is the same twice.
//
// Regularizers (3DGS-MCMC, Eq. 7): `opacity_reg * sum o` and
// `scale_reg * sum s`, added to the gradient - they are what manufactures the
// transparent, small gaussians relocation recycles.
//
// Shape bounds, in log space after the step: every log scale in
// [log_min, log smax[i]] and, sorted a >= b >= c, a - b <= log max_needle
// and b - c <= log max_flat, by RAISING the smaller axes (the constraint
// declines a degenerate shape without discarding what the gradient grew).

struct Params {
    n: u32,
    step: u32,       // for the noise stream
    rel_pos: u32,    // 1 = lr_pos is in units of each gaussian's largest axis
    opacity_reg: f32,
    lr_pos: f32,
    lr_scale: f32,
    lr_rot: f32,
    lr_op: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
    pad1: f32,
    noise: f32,
    log_min: f32,
    log_needle: f32, // log max_needle, <= 0 = unbounded
    log_flat: f32,   // log max_flat, <= 0 = unbounded
    logit_min: f32,
    logit_max: f32,
    scale_reg: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> geo:     array<f32>; // N*11
@group(0) @binding(2) var<storage, read>       d_gauss: array<f32>; // N*10 (mean, linear scale, quat)
@group(0) @binding(3) var<storage, read>       d_opac:  array<f32>; // N (activated opacity)
@group(0) @binding(4) var<storage, read_write> m:       array<f32>; // N*11
@group(0) @binding(5) var<storage, read_write> v:       array<f32>; // N*11
@group(0) @binding(6) var<storage, read>       smax:    array<f32>; // N log scale ceiling

// Counter-based hash to a uniform in (0, 1).
fn hash_unit(x: u32) -> f32 {
    var h = x * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    return (f32(h >> 8u) + 0.5) / 16777216.0;
}

fn gauss(a: u32, b: u32) -> f32 {
    let u = hash_unit(a);
    let w = hash_unit(b);
    return sqrt(-2.0 * log(u)) * cos(6.28318530718 * w);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    let o = i * 11u;
    let s0 = exp(geo[o + 3u]);
    let s1 = exp(geo[o + 4u]);
    let s2 = exp(geo[o + 5u]);
    let op = 1.0 / (1.0 + exp(-geo[o + 10u]));
    var own = 1.0;
    if (p.rel_pos != 0u) {
        own = max(s0, max(s1, s2));
    }

    // gradients with respect to the raw parameters
    var g: array<f32, 11>;
    for (var k = 0u; k < 3u; k = k + 1u) { g[k] = d_gauss[i * 10u + k]; }
    g[3] = (d_gauss[i * 10u + 3u] + p.scale_reg) * s0;
    g[4] = (d_gauss[i * 10u + 4u] + p.scale_reg) * s1;
    g[5] = (d_gauss[i * 10u + 5u] + p.scale_reg) * s2;
    for (var k = 6u; k < 10u; k = k + 1u) { g[k] = d_gauss[i * 10u + k]; }
    g[10] = (d_opac[i] + p.opacity_reg) * op * (1.0 - op);

    for (var k = 0u; k < 11u; k = k + 1u) {
        var lr = p.lr_op;
        if (k < 3u) {
            lr = p.lr_pos * own;
        } else if (k < 6u) {
            lr = p.lr_scale;
        } else if (k < 10u) {
            lr = p.lr_rot;
        }
        let mi = p.beta1 * m[o + k] + (1.0 - p.beta1) * g[k];
        let vi = p.beta2 * v[o + k] + (1.0 - p.beta2) * g[k] * g[k];
        m[o + k] = mi;
        v[o + k] = vi;
        geo[o + k] = geo[o + k] - lr * (mi / p.bc1) / (sqrt(vi / p.bc2) + p.eps);
    }

    // SGLD exploration, weighted to the transparent
    if (p.noise > 0.0) {
        let gate = 1.0 / (1.0 + exp(100.0 * (op - 0.005)));
        if (gate > 1e-6) {
            // the displacement is S e, already in units of the gaussian's
            // own size, times the position step measured in those units
            let amp = p.noise * p.lr_pos * own / max(max(s0, max(s1, s2)), 1e-30) * gate;
            let seed = i * 6u + p.step * 2654435761u;
            let e0 = gauss(seed, seed ^ 0x68e31da4u);
            let e1 = gauss(seed + 1u, (seed + 1u) ^ 0x68e31da4u);
            let e2 = gauss(seed + 2u, (seed + 2u) ^ 0x68e31da4u);
            // S e = sum_k axis_k * s_k * e_k, from the updated rotation
            let qn = sqrt(geo[o + 6u] * geo[o + 6u] + geo[o + 7u] * geo[o + 7u]
                + geo[o + 8u] * geo[o + 8u] + geo[o + 9u] * geo[o + 9u]) + 1e-8;
            let qw = geo[o + 6u] / qn;
            let qx = geo[o + 7u] / qn;
            let qy = geo[o + 8u] / qn;
            let qz = geo[o + 9u] / qn;
            let a = e0 * s0;
            let b = e1 * s1;
            let c = e2 * s2;
            geo[o] = geo[o] + amp * ((1.0 - 2.0 * (qy * qy + qz * qz)) * a + 2.0 * (qx * qy - qw * qz) * b + 2.0 * (qx * qz + qw * qy) * c);
            geo[o + 1u] = geo[o + 1u] + amp * (2.0 * (qx * qy + qw * qz) * a + (1.0 - 2.0 * (qx * qx + qz * qz)) * b + 2.0 * (qy * qz - qw * qx) * c);
            geo[o + 2u] = geo[o + 2u] + amp * (2.0 * (qx * qz - qw * qy) * a + 2.0 * (qy * qz + qw * qx) * b + (1.0 - 2.0 * (qx * qx + qy * qy)) * c);
        }
    }

    // shape bounds, in log space
    var l0 = clamp(geo[o + 3u], p.log_min, max(p.log_min, smax[i]));
    var l1 = clamp(geo[o + 4u], p.log_min, max(p.log_min, smax[i]));
    var l2 = clamp(geo[o + 5u], p.log_min, max(p.log_min, smax[i]));
    // sort descending into (a, b, c) by index
    var ia = 0u;
    var ib = 1u;
    var ic = 2u;
    var la = l0;
    var lb = l1;
    var lc = l2;
    if (lb > la) { let t = la; la = lb; lb = t; let u = ia; ia = ib; ib = u; }
    if (lc > la) { let t = la; la = lc; lc = t; let u = ia; ia = ic; ic = u; }
    if (lc > lb) { let t = lb; lb = lc; lc = t; let u = ib; ib = ic; ic = u; }
    if (p.log_needle > 0.0) { lb = max(lb, la - p.log_needle); }
    if (p.log_flat > 0.0) { lc = max(lc, lb - p.log_flat); }
    geo[o + 3u + ia] = la;
    geo[o + 3u + ib] = lb;
    geo[o + 3u + ic] = lc;
    geo[o + 10u] = clamp(geo[o + 10u], p.logit_min, p.logit_max);
}
