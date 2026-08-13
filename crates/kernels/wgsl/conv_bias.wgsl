// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Fused conv2d + per-output-channel bias
// @how   one thread per output element, 3 nested serial reductions
// @opt   1
// @cpu   native
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32|bf16|f16
// @tpl   w -> bf16/f16 storage variant (kernels::template::dtype_variant, B8;
//        `w_idx` was already a bare-identifier `let`, no hoist needed)
//
// Fused conv2d + per-output-channel bias. Identical convolution to conv2d.wgsl
// (bias-free, NCHW, square KxK, generic stride & implicit zero-pad), then adds
// the per-channel bias in the SAME pass:  y[co,...] = conv(x,w)[co,...] + bias[co].
//
// Used by the detection head's final 1x1 conv, fusing away the separate
// bias_add pass (and the host-built [C*HW] broadcast buffer it needed) — the
// bias param [Cout] is bound directly. Single source of truth: runs on wgpu and
// the wgsl-cpu JIT; the CPU backend routes it to the native AVX2 fast path.

struct Params {
    N: u32, Cin: u32, H: u32, W: u32, Cout: u32,
    K: u32, stride: u32, pad: u32, Ho: u32, Wo: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read>       w:    array<f32>;
@group(0) @binding(3) var<storage, read>       bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.Cout * p.Ho * p.Wo;
    if (idx >= total) { return; }

    let wo = idx % p.Wo;
    let t1 = idx / p.Wo;
    let ho = t1 % p.Ho;
    let t2 = t1 / p.Ho;
    let co = t2 % p.Cout;
    let n  = t2 / p.Cout;

    var acc = 0.0;
    for (var ci: u32 = 0u; ci < p.Cin; ci = ci + 1u) {
        for (var kh: u32 = 0u; kh < p.K; kh = kh + 1u) {
            let hi_b = ho * p.stride + kh;
            if (hi_b >= p.pad) {
                let hi = hi_b - p.pad;
                if (hi < p.H) {
                    for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
                        let wi_b = wo * p.stride + kw;
                        if (wi_b >= p.pad) {
                            let wi = wi_b - p.pad;
                            if (wi < p.W) {
                                let x_idx = ((n * p.Cin + ci) * p.H + hi) * p.W + wi;
                                let w_idx = ((co * p.Cin + ci) * p.K + kh) * p.K + kw;
                                acc = acc + x[x_idx] * w[w_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    y[idx] = acc + bias[co];
}
