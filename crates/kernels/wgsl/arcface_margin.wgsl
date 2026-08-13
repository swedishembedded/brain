// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  ArcFace additive ANGULAR margin, applied to a cosine-similarity logit table
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// ArcFace additive ANGULAR margin, applied to a cosine-similarity logit table.
//
//   cos    : [rows, classes]  read        cos(theta_{i,j}) = <e_hat_i, w_hat_j>
//   labels : [rows]           read (u32)  the ground-truth class of row i
//   out    : [rows, classes]  read_write  the scaled logits fed to cross-entropy
//
//   out[i, j] = s * cos(theta_ij)                     for j != labels[i]
//   out[i, y] = s * cos(theta_iy + m)                 for y == labels[i]
//             = s * (cos_iy * cos(m) - sin_iy * sin(m)),  sin_iy = sqrt(1 - cos^2)
//
// One invocation per OUTPUT element (total = rows*classes) — a pure gather, no
// reduction, no atomics. `cos_m`/`sin_m`/`scale` ride as bit-cast f32 in the
// uniform, so ONE kernel serves every (s, m) pair without specialisation.
//
// WHY THE MARGIN LIVES IN A KERNEL AND NOT IN THE HOST.
// `s` and `m` are constants but `cos` is the whole [rows, classes] table and it
// is on the differentiation path: the host would have to round-trip it every
// forward AND own its adjoint. The matching derivative is
// arcface_margin_bwd.wgsl; the pair is finite-difference gated by
// `gradcheck::check_arcface`.
//
// THE PLAIN FORM, DELIBERATELY. insightface's release additionally clamps the
// target logit when `cos <= cos(pi - m)` (`cos - m*sin(m)`, or the "easy
// margin" `cos > 0` test). Both are piecewise switches that put a KINK in the
// objective, which a central finite difference straddles. This kernel
// implements the unclamped formula, which is what the paper states and what the
// gradient check can validate; a clamped variant would be a separate kernel
// with its own gate, not a flag here.
//
// The only non-smooth point left is |cos| = 1 (theta = 0 or pi), where
// d(sin)/d(cos) is unbounded. `sin` is floored at SIN_FLOOR so the forward and
// the backward degrade to a finite slope together rather than producing inf/nan
// in one of them.

const SIN_FLOOR: f32 = 1e-6;

struct Params {
    rows: u32,
    classes: u32,
    cos_m: u32,   // f32 bits: cos(margin)
    sin_m: u32,   // f32 bits: sin(margin)
    scale: u32,   // f32 bits: s
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       cosv:   array<f32>;
@group(0) @binding(2) var<storage, read>       labels: array<u32>;
@group(0) @binding(3) var<storage, read_write> out:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.rows * p.classes;
    if (gidx >= total) { return; }

    let row = gidx / p.classes;
    let col = gidx % p.classes;
    let s = bitcast<f32>(p.scale);
    let c = cosv[gidx];

    if (col != labels[row]) {
        out[gidx] = s * c;
        return;
    }
    // fp32 round-off can push |c| a hair past 1, so clamp before the sqrt.
    let sn = max(sqrt(max(1.0 - c * c, 0.0)), SIN_FLOOR);
    out[gidx] = s * (c * bitcast<f32>(p.cos_m) - sn * bitcast<f32>(p.sin_m));
}
