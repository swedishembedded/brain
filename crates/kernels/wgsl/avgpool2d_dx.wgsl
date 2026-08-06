// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Adaptive/box average-pool INPUT gradient, NCHW
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Adaptive/box average-pool INPUT gradient, NCHW.
//   dy : [N, C, Ho, Wo]
//   dx : [N, C, H,  W ]   read_write (one invocation per INPUT element)
//
// The adjoint of avgpool2d.wgsl, as a GATHER (brain's kernels are atomic-free):
// each input pixel finds the output windows containing it and sums dy/|window|.
//
// Inverting `h0 = floor(ho*H/Ho)`, `h1 = ceil((ho+1)*H/Ho)` exactly is fiddly and
// easy to get subtly wrong at the boundaries — and, with overlapping adaptive
// windows, a wrong bound loses gradient silently. So the candidate range is
// computed loosely and each candidate re-tests the FORWARD's own window
// predicate. The adjoint then holds by construction: there is one definition of
// the window, evaluated twice, not two definitions that must agree.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    Ho: u32,
    Wo: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.C * p.H * p.W;
    if (idx >= total) { return; }

    let wi = idx % p.W;
    let t1 = idx / p.W;
    let hi = t1 % p.H;
    let t2 = t1 / p.H;
    let c  = t2 % p.C;
    let n  = t2 / p.C;

    // Loose candidate window: ho ~ hi*Ho/H, widened by 1 each side and clamped.
    // The exact predicate below filters; a superset only costs iterations.
    let ho_c = (hi * p.Ho) / p.H;
    let ho_lo = select(ho_c - 1u, 0u, ho_c == 0u);
    let ho_hi = min(ho_c + 1u, p.Ho - 1u);
    let wo_c = (wi * p.Wo) / p.W;
    let wo_lo = select(wo_c - 1u, 0u, wo_c == 0u);
    let wo_hi = min(wo_c + 1u, p.Wo - 1u);

    var acc = 0.0;
    for (var ho: u32 = ho_lo; ho <= ho_hi; ho = ho + 1u) {
        let h0 = (ho * p.H) / p.Ho;
        let h1 = ((ho + 1u) * p.H + p.Ho - 1u) / p.Ho;
        if (hi >= h0 && hi < h1) {
            for (var wo: u32 = wo_lo; wo <= wo_hi; wo = wo + 1u) {
                let w0 = (wo * p.W) / p.Wo;
                let w1 = ((wo + 1u) * p.W + p.Wo - 1u) / p.Wo;
                if (wi >= w0 && wi < w1) {
                    let dy_idx = ((n * p.C + c) * p.Ho + ho) * p.Wo + wo;
                    acc = acc + dy[dy_idx] / f32((h1 - h0) * (w1 - w0));
                }
            }
        }
    }
    dx[idx] = acc;
}
