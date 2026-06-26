// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Register-tiled fused conv -> per-channel affine -> SiLU. Same result as
// conv_act.wgsl, but each invocation computes ONE output position for FOUR
// output channels at once, loading each input value a single time and reusing it
// across the 4 channels in registers. That cuts the dominant input-read traffic
// ~4x (the naive kernel re-reads the whole input once per output channel), with
// NO workgroup memory — so GPU occupancy stays full (unlike a weight-staged
// kernel). Plain per-invocation, so the wgsl-cpu JIT compiles it unchanged.
//
// Dispatch: one invocation per (n, channel-quad, output-position):
//   total = N * ceil(Cout/4) * Ho * Wo.

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
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       w:  array<f32>;
@group(0) @binding(3) var<storage, read>       sb: array<f32>;
@group(0) @binding(4) var<storage, read_write> y:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let kg = p.Cin * p.K * p.K;
    let psz = p.Ho * p.Wo;
    let ntc = (p.Cout + 3u) / 4u;            // channel quads
    let total = p.N * ntc * psz;
    if (idx >= total) { return; }

    let pidx = idx % psz;
    let t = idx / psz;
    let cq = t % ntc;
    let n = t / ntc;
    let co0 = cq * 4u;

    let wo = pidx % p.Wo;
    let ho = pidx / p.Wo;

    var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
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
                                let xv = x[((n * p.Cin + ci) * p.H + hi) * p.W + wi];
                                let woff = (ci * p.K + kh) * p.K + kw;
                                // One input load, reused across the 4 channels.
                                a0 = a0 + xv * w[(co0) * kg + woff];
                                if (co0 + 1u < p.Cout) { a1 = a1 + xv * w[(co0 + 1u) * kg + woff]; }
                                if (co0 + 2u < p.Cout) { a2 = a2 + xv * w[(co0 + 2u) * kg + woff]; }
                                if (co0 + 3u < p.Cout) { a3 = a3 + xv * w[(co0 + 3u) * kg + woff]; }
                            }
                        }
                    }
                }
            }
        }
    }

    // Per-channel affine (BN-eval collapsed) + SiLU (inlined), then store.
    let base = (n * p.Cout + co0) * psz + pidx;
    let z0 = a0 * sb[2u * co0] + sb[2u * co0 + 1u];
    y[base] = z0 / (1.0 + exp(-z0));
    if (co0 + 1u < p.Cout) {
        let c = co0 + 1u;
        let z = a1 * sb[2u * c] + sb[2u * c + 1u];
        y[base + psz] = z / (1.0 + exp(-z));
    }
    if (co0 + 2u < p.Cout) {
        let c = co0 + 2u;
        let z = a2 * sb[2u * c] + sb[2u * c + 1u];
        y[base + 2u * psz] = z / (1.0 + exp(-z));
    }
    if (co0 + 3u < p.Cout) {
        let c = co0 + 3u;
        let z = a3 * sb[2u * c] + sb[2u * c + 1u];
        y[base + 3u * psz] = z / (1.0 + exp(-z));
    }
}
