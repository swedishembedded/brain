// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward of arcface_margin.wgsl w.r.t
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Backward of arcface_margin.wgsl w.r.t. the cosine table.
//
//   cos    : [rows, classes]  read        the SAME input the forward saw
//   labels : [rows]           read (u32)
//   dy     : [rows, classes]  read        grad wrt the scaled logits
//   dx     : [rows, classes]  read_write  grad wrt cos            (OVERWRITES)
//
// With sn = sqrt(1 - c^2):
//   j != y :  d(out)/d(cos) = s
//   j == y :  d(out)/d(cos) = s * (cos_m + sin_m * c / sn)
// because d/dc [ c*cos_m - sqrt(1-c^2)*sin_m ] = cos_m + sin_m * c / sqrt(1-c^2).
//
// One invocation per element of the OUTPUT `dx` (total = rows*classes): each
// writes exactly one location and gathers the one input that touches it, so the
// kernel ASSIGNS, needs no pre-zeroing and uses no atomics. There is no
// reduction here, hence no cooperative twin — the caps-gated PAIR rule
// (prelu_bwd / prelu_bwd_wg) applies to per-channel reductions, and this is
// elementwise.
//
// `SIN_FLOOR` matches arcface_margin.wgsl's: the forward and the backward must
// leave the |cos| = 1 singularity at the SAME place, or the finite-difference
// check sees two different functions.
//
// AND WHEN THE FLOOR IS ACTIVE, THE SIN TERM'S DERIVATIVE IS ZERO, NOT HUGE.
// Once the forward has clamped, it computes `s*(c*cos_m - SIN_FLOOR*sin_m)`:
// the second term no longer depends on `c` at all, so d(out)/d(cos) there is
// exactly `s*cos_m`. Carrying `sin_m * c / SIN_FLOOR` into the clamped branch
// instead — which is what the naive "same formula, floored denominator" spelling
// does — returns ~1e6 * sin_m * s where the truth is O(s): with the paper's
// s = 64 that is a ~3e7 gradient spike on one logit, from a single sample whose
// embedding has collapsed onto its class centre. The finite-difference gate
// cannot see it, because `1 - c*c` in fp32 is either exactly 0 (c == +/-1.0 to
// the bit) or >= ~1.2e-7 — there is no c in between for a central difference to
// land on. So the branch is pinned by `the_margin_adjoint_is_finite_at_cos_one`
// in crates/facenet/src/train.rs instead.

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
@group(0) @binding(3) var<storage, read>       dy:     array<f32>;
@group(0) @binding(4) var<storage, read_write> dx:     array<f32>;

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

    if (col != labels[row]) {
        dx[gidx] = dy[gidx] * s;
        return;
    }
    let c = cosv[gidx];
    let sn = max(sqrt(max(1.0 - c * c, 0.0)), SIN_FLOOR);
    dx[gidx] = dy[gidx] * s * (bitcast<f32>(p.cos_m) + bitcast<f32>(p.sin_m) * c / sn);
}
