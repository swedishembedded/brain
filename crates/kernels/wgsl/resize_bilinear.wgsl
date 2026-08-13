// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Bilinear resize forward, NCHW, arbitrary output size
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Bilinear resize forward, NCHW, arbitrary output size.
//   x : [N, C, H,  W ]
//   y : [N, C, Ho, Wo]   one invocation per OUTPUT element
//
// `align_corners` (0 = false, 1 = true) selects the source-coordinate mapping.
// This is NOT a detail: the two conventions differ by half a pixel, both look
// plausible, and a gradient check CANNOT catch the wrong one — the kernel stays
// perfectly self-consistent while resampling the wrong grid. The only defence is
// a numeric parity test against the reference, which is why both modes exist here
// rather than whichever one the first caller happened to need.
//
//   align_corners = 1:  src = o * (in - 1) / (out - 1)        (out==1 -> 0)
//                       corner samples land exactly on corner pixels.
//   align_corners = 0:  src = (o + 0.5) * (in / out) - 0.5    (then clamped >= 0)
//                       pixel centres are half-integer; PyTorch's default.
//
// Correspondence to the ONNX ops brain exports, so the engine and the exported
// graph compute the SAME function:
//   align_corners = 1  <->  coordinate_transformation_mode = "align_corners"
//   align_corners = 0  <->  coordinate_transformation_mode = "half_pixel"
// (NOT "asymmetric", which is what the existing nearest Resize export emits.)
//
// Both target models need both modes: ZipDepth's UltraLightFusion and its NPU
// upsampler use align_corners=false, while the predictor's final upsample back to
// the source resolution uses align_corners=true — and DPT's fusion blocks use
// true.
//
// The source-coordinate mapping is written out TWICE (once per axis) rather than
// factored into a helper: the wgsl-cpu JIT inlines a single entry point and
// rejects user-defined function calls outright, so a helper here would compile on
// wgpu and hard-fail on the CPU backend, breaking the "same WGSL, both backends"
// invariant. Same reason gelu_erf.wgsl inlines its erf.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    Ho: u32,
    Wo: u32,
    align_corners: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.C * p.Ho * p.Wo;
    if (idx >= total) { return; }

    // Decode output coordinate (n, c, ho, wo).
    let wo = idx % p.Wo;
    let t1 = idx / p.Wo;
    let ho = t1 % p.Ho;
    let t2 = t1 / p.Ho;
    let c  = t2 % p.C;
    let n  = t2 / p.C;

    // --- source coordinate, y axis (inlined; see header) ---
    var sy = 0.0;
    if (p.align_corners == 1u) {
        if (p.Ho > 1u) { sy = f32(ho) * f32(p.H - 1u) / f32(p.Ho - 1u); }
    } else {
        sy = max((f32(ho) + 0.5) * (f32(p.H) / f32(p.Ho)) - 0.5, 0.0);
    }
    // --- source coordinate, x axis ---
    var sx = 0.0;
    if (p.align_corners == 1u) {
        if (p.Wo > 1u) { sx = f32(wo) * f32(p.W - 1u) / f32(p.Wo - 1u); }
    } else {
        sx = max((f32(wo) + 0.5) * (f32(p.W) / f32(p.Wo)) - 0.5, 0.0);
    }

    let fy0 = floor(sy);
    let fx0 = floor(sx);
    let y0 = u32(fy0);
    let x0 = u32(fx0);
    let y1 = min(y0 + 1u, p.H - 1u);   // clamp-to-edge on the high side
    let x1 = min(x0 + 1u, p.W - 1u);
    let fy = sy - fy0;
    let fx = sx - fx0;

    let base = (n * p.C + c) * p.H;
    let v00 = x[(base + y0) * p.W + x0];
    let v01 = x[(base + y0) * p.W + x1];
    let v10 = x[(base + y1) * p.W + x0];
    let v11 = x[(base + y1) * p.W + x1];

    let top = v00 + (v01 - v00) * fx;
    let bot = v10 + (v11 - v10) * fx;
    y[idx] = top + (bot - top) * fy;
}
