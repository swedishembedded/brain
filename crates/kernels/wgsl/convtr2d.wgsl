// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Transposed 2D convolution forward (ConvTranspose2d, bias-free), NCHW, square KxK, WITH grouping + dilation
// @how   one thread per output element, 3 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Transposed 2D convolution forward (ConvTranspose2d, bias-free), NCHW, square
// KxK, WITH grouping + dilation. Used for decoder upsampling (SAM 2's mask
// decoder does 2x twice; VQGAN/CodeFormer decoders).
//   x : [N, Cin,        H,  W]   idx = ((n*Cin + ci)*H + hi)*W + wi
//   w : [Cin, Cout/G,   K,  K]   idx = ((ci*(Cout/G) + co_local)*K + kh)*K + kw
//   y : [N, Cout,       Ho, Wo]  idx = ((n*Cout + co)*Ho + ho)*Wo + wo
//
// This is convtr1d.wgsl's index math lifted to 2D exactly as conv2d_gd.wgsl
// lifted conv1d.wgsl's. The weight layout is PyTorch's ConvTranspose2d
// convention `[in_channels, out_channels/groups, kH, kW]` — note that the
// INPUT channel is the outer axis, the transpose of conv2d_gd's
// `[Cout, Cin/G, K, K]`. Both layouts hold exactly Cin*Cout/G elements, so a
// weight in conv2d_gd's layout ALWAYS binds, at every Cin/Cout/G — it never
// fails a size check, it just computes a different operator.
//
// One invocation per OUTPUT element. The forward maps an input (hi,wi) to
//   ho = hi*stride - pad + kh*dilation,  wo = wi*stride - pad + kw*dilation,
// so each output gathers the inputs that land on it by inverting that:
//   hi = (ho + pad - kh*dilation)/stride  (exact division, in [0,H)); ditto wi.
//   Ho = (H-1)*stride - 2*pad + dilation*(K-1) + out_pad + 1  (caller-computed;
//   likewise Wo).
//
// out_pad needs no Params field: it only widens Ho/Wo, which the caller passes
// in, and the gather then covers whatever taps land in the widened range. Do NOT
// assume the extra bottom/right band is zero-fill — verified against PyTorch, it
// is not. When stride > 1 the far-side `pad` crop hides output positions that
// genuinely receive input (e.g. H=4,K=3,stride=2,pad=1 drops the position fed by
// hi=H-1,kh=K-1), and output_padding is exactly what un-crops them. That is why
// PyTorch documents it as resolving a strided conv's output-shape ambiguity
// rather than as padding.
//
// GROUPING: the group is determined by the OUTPUT channel co here as well
// (g = co/(Cout/G)), but the weight's SECOND axis is co_local in [0, Cout/G) and
// its FIRST axis is the absolute input channel ci in [g*Cin/G, (g+1)*Cin/G).

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
    let g = co / cout_g;         // group this output channel belongs to
    let co_local = co - g * cout_g;
    let ci0 = g * cin_g;         // first input channel of that group

    var acc = 0.0;
    for (var kh: u32 = 0u; kh < p.K; kh = kh + 1u) {
        let numh = ho + p.pad;
        let subh = kh * p.dilation;
        if (numh >= subh) {
            let numh2 = numh - subh;
            if ((numh2 % p.stride) == 0u) {
                let hi = numh2 / p.stride;
                if (hi < p.H) {
                    for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
                        let numw = wo + p.pad;
                        let subw = kw * p.dilation;
                        if (numw >= subw) {
                            let numw2 = numw - subw;
                            if ((numw2 % p.stride) == 0u) {
                                let wi = numw2 / p.stride;
                                if (wi < p.W) {
                                    for (var cl: u32 = 0u; cl < cin_g; cl = cl + 1u) {
                                        let ci = ci0 + cl;
                                        let x_idx = ((n * p.Cin + ci) * p.H + hi) * p.W + wi;
                                        let w_idx = ((ci * cout_g + co_local) * p.K + kh) * p.K + kw;
                                        acc = acc + x[x_idx] * w[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    y[idx] = acc;
}
