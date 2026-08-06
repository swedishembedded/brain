// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  2D convolution forward (bias-free), NCHW, square KxK, WITH grouping + dilation
// @how   one thread per output element, 3 nested serial reductions
// @opt   1
// @cpu   native
// @gpu   yes
// @npu   yes
// @quant none
//
// 2D convolution forward (bias-free), NCHW, square KxK, WITH grouping + dilation.
//   x : [N, Cin,        H,  W]   idx = ((n*Cin + ci)*H + hi)*W + wi
//   w : [Cout, Cin/G,   K,  K]   idx = ((co*(Cin/G) + cl)*K + kh)*K + kw
//   y : [N, Cout,       Ho, Wo]  idx = ((n*Cout + co)*Ho + ho)*Wo + wo
//
// The generalization of conv2d.wgsl; the index math is conv1d.wgsl's, lifted to
// 2D. Depthwise is the G == Cin == Cout case (w is [C,1,K,K]).
//
// One invocation per OUTPUT element. Generic stride & zero-pad (implicit): taps
// whose input coordinate falls outside [0,H)/[0,W) are skipped, which is exactly
// the contribution of a zero-padded border.
//   Ho = (H + 2*pad - (dilation*(K-1)+1))/stride + 1   (likewise Wo)
//
// NOTE the deliberately distinct name. `backend-cpu` binds its AVX2/winograd fast
// paths BY KERNEL NAME (`find("conv2d")`), and those paths are dense — they
// ignore `groups` entirely. Naming this `conv2d` would silently inherit them and
// compute wrong results with no error whatsoever.

struct Params {
    N: u32,
    Cin: u32,
    H: u32,
    W: u32,
    Cout: u32,
    K: u32,
    stride: u32,
    pad: u32,
    dilation: u32,
    groups: u32,
    Ho: u32,
    Wo: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.Cout * p.Ho * p.Wo;
    if (idx >= total) { return; }

    // Decode output coordinate (n, co, ho, wo) from the linear index.
    let wo = idx % p.Wo;
    let t1 = idx / p.Wo;
    let ho = t1 % p.Ho;
    let t2 = t1 / p.Ho;
    let co = t2 % p.Cout;
    let n  = t2 / p.Cout;

    let cin_g  = p.Cin / p.groups;
    let cout_g = p.Cout / p.groups;
    let g   = co / cout_g;   // group this output channel belongs to
    let ci0 = g * cin_g;     // first input channel of that group

    var acc = 0.0;
    for (var cl: u32 = 0u; cl < cin_g; cl = cl + 1u) {
        let ci = ci0 + cl;
        for (var kh: u32 = 0u; kh < p.K; kh = kh + 1u) {
            let hi_b = ho * p.stride + kh * p.dilation;
            if (hi_b >= p.pad) {
                let hi = hi_b - p.pad;
                if (hi < p.H) {
                    for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
                        let wi_b = wo * p.stride + kw * p.dilation;
                        if (wi_b >= p.pad) {
                            let wi = wi_b - p.pad;
                            if (wi < p.W) {
                                let x_idx = ((n * p.Cin + ci) * p.H + hi) * p.W + wi;
                                let w_idx = ((co * cin_g + cl) * p.K + kh) * p.K + kw;
                                acc = acc + x[x_idx] * w[w_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    y[idx] = acc;
}
