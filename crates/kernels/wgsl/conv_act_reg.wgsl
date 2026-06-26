// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Register-tiled fused conv -> per-channel affine -> SiLU. Each invocation
// computes a 4x4 output tile = 4 output channels x 4 spatial positions, holding
// the 16 partial sums in SCALAR registers (fully unrolled, so the GPU keeps them
// in registers, not local memory). Per tap it loads 4 weights (reused across the
// 4 positions) and 4 inputs (reused across the 4 channels) -> both global-read
// traffics drop ~4x vs the naive one-output-per-thread kernel. The 4 positions
// are STRIDED by npq, so adjacent threads access adjacent addresses (coalesced).
//
// Loop order is (kh, kw) OUTER, ci INNER: the per-position boundary checks +
// input offsets depend only on (kh,kw,position), not on the input channel, so
// they are computed ONCE per tap instead of once per (ci,tap) — for a
// 256-channel layer that's ~256x fewer bound evaluations / index ops, the
// per-tap overhead that limited the earlier ci-outer version.
//
// No workgroup memory -> full GPU occupancy; plain per-invocation -> the
// wgsl-cpu JIT compiles it unchanged. Same result as conv_act.wgsl.
// Dispatch: total = N * ceil(Cout/4) * ceil(Ho*Wo/4).

struct Params {
    N: u32, Cin: u32, H: u32, W: u32, Cout: u32,
    K: u32, stride: u32, pad: u32, Ho: u32, Wo: u32,
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
    let kk = p.K * p.K;
    let kg = p.Cin * kk;
    let hw = p.H * p.W;
    let psz = p.Ho * p.Wo;
    let ntc = (p.Cout + 3u) / 4u;
    let npq = (psz + 3u) / 4u;
    if (idx >= p.N * ntc * npq) { return; }

    let pq = idx % npq;
    let tt = idx / npq;
    let cq = tt % ntc;
    let n = tt / ntc;
    let co0 = cq * 4u;
    let nc = min(4u, p.Cout - co0);

    // Coalesced: 4 positions strided by npq (adjacent threads -> adjacent addrs).
    let q0 = pq; let q1 = pq + npq; let q2 = pq + 2u * npq; let q3 = pq + 3u * npq;
    let ho0 = q0 / p.Wo; let wo0 = q0 % p.Wo;
    let ho1 = q1 / p.Wo; let wo1 = q1 % p.Wo;
    let ho2 = q2 / p.Wo; let wo2 = q2 % p.Wo;
    let ho3 = q3 / p.Wo; let wo3 = q3 % p.Wo;
    let v1 = q1 < psz; let v2 = q2 < psz; let v3 = q3 < psz;

    var a00 = 0.0; var a01 = 0.0; var a02 = 0.0; var a03 = 0.0;
    var a10 = 0.0; var a11 = 0.0; var a12 = 0.0; var a13 = 0.0;
    var a20 = 0.0; var a21 = 0.0; var a22 = 0.0; var a23 = 0.0;
    var a30 = 0.0; var a31 = 0.0; var a32 = 0.0; var a33 = 0.0;

    let nci = n * p.Cin;
    for (var kh: u32 = 0u; kh < p.K; kh = kh + 1u) {
        for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
            let wtap = kh * p.K + kw;
            // Per-position validity + spatial offset (channel part added below).
            let hb0 = ho0 * p.stride + kh; let xb0 = wo0 * p.stride + kw;
            let ok0 = hb0 >= p.pad && xb0 >= p.pad && hb0 - p.pad < p.H && xb0 - p.pad < p.W;
            let off0 = (hb0 - p.pad) * p.W + (xb0 - p.pad);
            let hb1 = ho1 * p.stride + kh; let xb1 = wo1 * p.stride + kw;
            let ok1 = v1 && hb1 >= p.pad && xb1 >= p.pad && hb1 - p.pad < p.H && xb1 - p.pad < p.W;
            let off1 = (hb1 - p.pad) * p.W + (xb1 - p.pad);
            let hb2 = ho2 * p.stride + kh; let xb2 = wo2 * p.stride + kw;
            let ok2 = v2 && hb2 >= p.pad && xb2 >= p.pad && hb2 - p.pad < p.H && xb2 - p.pad < p.W;
            let off2 = (hb2 - p.pad) * p.W + (xb2 - p.pad);
            let hb3 = ho3 * p.stride + kh; let xb3 = wo3 * p.stride + kw;
            let ok3 = v3 && hb3 >= p.pad && xb3 >= p.pad && hb3 - p.pad < p.H && xb3 - p.pad < p.W;
            let off3 = (hb3 - p.pad) * p.W + (xb3 - p.pad);

            for (var ci: u32 = 0u; ci < p.Cin; ci = ci + 1u) {
                let wbase = ci * kk + wtap;
                let wt0 = w[co0 * kg + wbase];
                let wt1 = select(0.0, w[(co0 + 1u) * kg + wbase], 1u < nc);
                let wt2 = select(0.0, w[(co0 + 2u) * kg + wbase], 2u < nc);
                let wt3 = select(0.0, w[(co0 + 3u) * kg + wbase], 3u < nc);
                let xc = (nci + ci) * hw;
                if (ok0) { let xv = x[xc + off0]; a00 = a00 + xv * wt0; a01 = a01 + xv * wt1; a02 = a02 + xv * wt2; a03 = a03 + xv * wt3; }
                if (ok1) { let xv = x[xc + off1]; a10 = a10 + xv * wt0; a11 = a11 + xv * wt1; a12 = a12 + xv * wt2; a13 = a13 + xv * wt3; }
                if (ok2) { let xv = x[xc + off2]; a20 = a20 + xv * wt0; a21 = a21 + xv * wt1; a22 = a22 + xv * wt2; a23 = a23 + xv * wt3; }
                if (ok3) { let xv = x[xc + off3]; a30 = a30 + xv * wt0; a31 = a31 + xv * wt1; a32 = a32 + xv * wt2; a33 = a33 + xv * wt3; }
            }
        }
    }

    let nco = n * p.Cout;
    let s0 = sb[2u * co0]; let b0 = sb[2u * co0 + 1u];
    { let z = a00 * s0 + b0; y[(nco + co0) * psz + q0] = z / (1.0 + exp(-z)); }
    if (v1) { let z = a10 * s0 + b0; y[(nco + co0) * psz + q1] = z / (1.0 + exp(-z)); }
    if (v2) { let z = a20 * s0 + b0; y[(nco + co0) * psz + q2] = z / (1.0 + exp(-z)); }
    if (v3) { let z = a30 * s0 + b0; y[(nco + co0) * psz + q3] = z / (1.0 + exp(-z)); }
    if (1u < nc) {
        let c = co0 + 1u; let s = sb[2u * c]; let b = sb[2u * c + 1u];
        { let z = a01 * s + b; y[(nco + c) * psz + q0] = z / (1.0 + exp(-z)); }
        if (v1) { let z = a11 * s + b; y[(nco + c) * psz + q1] = z / (1.0 + exp(-z)); }
        if (v2) { let z = a21 * s + b; y[(nco + c) * psz + q2] = z / (1.0 + exp(-z)); }
        if (v3) { let z = a31 * s + b; y[(nco + c) * psz + q3] = z / (1.0 + exp(-z)); }
    }
    if (2u < nc) {
        let c = co0 + 2u; let s = sb[2u * c]; let b = sb[2u * c + 1u];
        { let z = a02 * s + b; y[(nco + c) * psz + q0] = z / (1.0 + exp(-z)); }
        if (v1) { let z = a12 * s + b; y[(nco + c) * psz + q1] = z / (1.0 + exp(-z)); }
        if (v2) { let z = a22 * s + b; y[(nco + c) * psz + q2] = z / (1.0 + exp(-z)); }
        if (v3) { let z = a32 * s + b; y[(nco + c) * psz + q3] = z / (1.0 + exp(-z)); }
    }
    if (3u < nc) {
        let c = co0 + 3u; let s = sb[2u * c]; let b = sb[2u * c + 1u];
        { let z = a03 * s + b; y[(nco + c) * psz + q0] = z / (1.0 + exp(-z)); }
        if (v1) { let z = a13 * s + b; y[(nco + c) * psz + q1] = z / (1.0 + exp(-z)); }
        if (v2) { let z = a23 * s + b; y[(nco + c) * psz + q2] = z / (1.0 + exp(-z)); }
        if (v3) { let z = a33 * s + b; y[(nco + c) * psz + q3] = z / (1.0 + exp(-z)); }
    }
}
