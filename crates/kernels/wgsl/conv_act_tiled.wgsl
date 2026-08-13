// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Weight-staged fused conv -> per-channel affine -> activation
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   native
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Weight-staged fused conv -> per-channel affine -> activation. Same result as
// conv_act.wgsl (including its `p.act` selector: 0 = identity, 1 = ReLU,
// 2 = SiLU, 3 = sigmoid), but one workgroup stages its output channel's weights
// in WORKGROUP (on-chip) memory and reuses them across a 64-position block,
// instead of re-reading every weight from global memory once per output pixel.
//
// Single source of truth for both backends: wgpu runs it directly; the wgsl-cpu
// JIT compiles it with its work-group execution model (the CPU backend then
// routes it to the native AVX2 fused-conv fast path for speed — same math).
//
// Layout: one workgroup = one (n, output-channel, 64-position block); the 64
// invocations load w[co,:] into `wsh`, barrier, then each computes one output:
//   z   = conv(x,wsh) * sb[2co] + sb[2co+1]      // BatchNorm-eval affine collapsed
//   y   = act(z)
// Dispatch: total invocations = N * Cout * ceil(Ho*Wo / 64) * 64.

struct Params {
    N: u32,
    Cin: u32,
    H: u32,
    W: u32,
    Cout: u32,
    K: u32,
    stride: u32,
    pad: u32,
    Ho: u32,
    Wo: u32,
    act: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       w:  array<f32>;
@group(0) @binding(3) var<storage, read>       sb: array<f32>;
@group(0) @binding(4) var<storage, read_write> y:  array<f32>;

var<workgroup> wsh: array<f32, 8192>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let kg = p.Cin * p.K * p.K;
    let psz = p.Ho * p.Wo;
    let blocks = (psz + 63u) / 64u;
    let wg = wgid.y * nwg.x + wgid.x;
    let per_n = p.Cout * blocks;
    let n = wg / per_n;
    let cob = wg % per_n;
    let co = cob / blocks;
    let sblock = cob % blocks;

    var i: u32 = lid.x;
    loop {
        if (i >= kg) { break; }
        wsh[i] = w[co * kg + i];
        i = i + 64u;
    }
    workgroupBarrier();

    let pidx = sblock * 64u + lid.x;
    if (pidx < psz && n < p.N) {
        let wo = pidx % p.Wo;
        let ho = pidx / p.Wo;
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
                                    let w_off = (ci * p.K + kh) * p.K + kw;
                                    acc = acc + x[x_idx] * wsh[w_off];
                                }
                            }
                        }
                    }
                }
            }
        }
        var z = acc * sb[2u * co] + sb[2u * co + 1u];
        if (p.act == 1u) { z = max(z, 0.0); }
        else if (p.act == 2u) { z = z / (1.0 + exp(-z)); }
        else if (p.act == 3u) { z = 1.0 / (1.0 + exp(-z)); }
        y[(n * p.Cout + co) * psz + pidx] = z;
    }
}
