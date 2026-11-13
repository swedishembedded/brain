// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Nearest-neighbour resize, NCHW, ARBITRARY output size
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Nearest-neighbour resize, NCHW, ARBITRARY output size.
//   x : [N, C, H,  W ]
//   y : [N, C, Ho, Wo]   one invocation per OUTPUT element
//
// The generic form of upsample2.wgsl, whose 2x factor is hardcoded. ZipDepth's
// MinimalCrossScale interpolates to an arbitrary target size (x_high's spatial
// dims), which 2x cannot express.
//
// Two tap rules, chosen by the `mode` word of Params (the 7th):
//
//   mode 0  torch's `nearest` (== ONNX "asymmetric" + nearest_mode "floor"):
//             src = floor(o * in / out)
//   mode 1  half-pixel nearest (the tap whose centre is nearest the output
//           pixel's centre, ties going up):
//             src = round_half_up((o + 0.5) * in / out - 0.5)
//                 = floor((o + 0.5) * in / out)             (a tie rounds up)
//                 = floor((2*o + 1) * in / (2 * out))       (integers only)
//           clamped to [0, in - 1]. The last line is exact in u32, so the rule
//           has no float rounding to disagree about between devices.
//
// The two select different source pixels for most ratios (3 -> 2 picks taps
// 0, 1 under mode 0 and 0, 2 under mode 1), so a model imported against one
// does not run under the other. Both clamp to the last source row/column.
//
// The adjoint kernel `resize_nearest_dx` is the adjoint of mode 0 ONLY; mode 1
// is an inference-time preprocessing rule and has no backward.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    Ho: u32,
    Wo: u32,
    mode: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.C * p.Ho * p.Wo;
    if (idx >= total) { return; }
    let wo = idx % p.Wo;
    let t1 = idx / p.Wo;
    let ho = t1 % p.Ho;
    let nc = t1 / p.Ho;
    var hi = (ho * p.H) / p.Ho;
    var wi = (wo * p.W) / p.Wo;
    if (p.mode == 1u) {
        hi = ((2u * ho + 1u) * p.H) / (2u * p.Ho);
        wi = ((2u * wo + 1u) * p.W) / (2u * p.Wo);
    }
    hi = min(hi, p.H - 1u);
    wi = min(wi, p.W - 1u);
    y[idx] = x[(nc * p.H + hi) * p.W + wi];
}
