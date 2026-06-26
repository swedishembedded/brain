// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Register-tiled fused conv -> per-channel affine -> SiLU. Each invocation
// computes a 4x4 output tile = 4 output channels x 4 spatial positions, holding
// the 16 partial sums in registers. Per kernel tap it loads 4 weights (one per
// channel, reused across the 4 positions) and 4 input values (one per position,
// reused across the 4 channels) — so BOTH the weight and the input global-read
// traffic drop ~4x vs the naive one-output-per-thread kernel (which re-reads the
// whole input once per output channel and the whole weight once per position).
// No workgroup memory -> full GPU occupancy; plain per-invocation -> the
// wgsl-cpu JIT compiles it unchanged. Same result as conv_act.wgsl.
//
// Dispatch: total = N * ceil(Cout/4) * ceil(Ho*Wo/4).

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
    let ntc = (p.Cout + 3u) / 4u;      // channel quads
    let npq = (psz + 3u) / 4u;         // position quads
    let total = p.N * ntc * npq;
    if (idx >= total) { return; }

    let pq = idx % npq;
    let t = idx / npq;
    let cq = t % ntc;
    let n = t / ntc;
    let co0 = cq * 4u;
    let p0 = pq * 4u;

    // The 4 output positions (n is fixed); ho/wo per position.
    var ho: array<u32, 4>;
    var wo: array<u32, 4>;
    var valid_p: array<bool, 4>;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let pj = p0 + j;
        valid_p[j] = pj < psz;
        ho[j] = pj / p.Wo;
        wo[j] = pj % p.Wo;
    }
    let nc = min(4u, p.Cout - co0);   // valid channels in this quad

    var acc: array<f32, 16>;          // [pos*4 + ch]
    for (var i: u32 = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

    for (var ci: u32 = 0u; ci < p.Cin; ci = ci + 1u) {
        for (var kh: u32 = 0u; kh < p.K; kh = kh + 1u) {
            for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
                let woff = (ci * p.K + kh) * p.K + kw;
                // 4 weights (one per output channel), reused across positions.
                var wt: array<f32, 4>;
                for (var c: u32 = 0u; c < 4u; c = c + 1u) {
                    wt[c] = select(0.0, w[(co0 + c) * kg + woff], c < nc);
                }
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    if (!valid_p[j]) { continue; }
                    let hi_b = ho[j] * p.stride + kh;
                    let wi_b = wo[j] * p.stride + kw;
                    if (hi_b >= p.pad && wi_b >= p.pad) {
                        let hi = hi_b - p.pad;
                        let wi = wi_b - p.pad;
                        if (hi < p.H && wi < p.W) {
                            let xv = x[((n * p.Cin + ci) * p.H + hi) * p.W + wi];
                            acc[j * 4u + 0u] = acc[j * 4u + 0u] + xv * wt[0];
                            acc[j * 4u + 1u] = acc[j * 4u + 1u] + xv * wt[1];
                            acc[j * 4u + 2u] = acc[j * 4u + 2u] + xv * wt[2];
                            acc[j * 4u + 3u] = acc[j * 4u + 3u] + xv * wt[3];
                        }
                    }
                }
            }
        }
    }

    // Affine (BN-eval collapsed) + SiLU + store, for valid channels/positions.
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        if (!valid_p[j]) { continue; }
        for (var c: u32 = 0u; c < 4u; c = c + 1u) {
            if (c >= nc) { continue; }
            let co = co0 + c;
            let z = acc[j * 4u + c] * sb[2u * co] + sb[2u * co + 1u];
            y[(n * p.Cout + co) * psz + (p0 + j)] = z / (1.0 + exp(-z));
        }
    }
}
