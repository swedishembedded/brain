// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Register-tiled fused conv -> per-channel affine -> activation. Each invocation
// computes an 8x4 output tile = 8 output channels x 4 spatial positions, holding
// the 32 partial sums in SCALAR registers (fully unrolled, so the GPU keeps them
// in registers, not local memory). Per tap it loads 8 weights (one per output
// channel, reused across the 4 positions) and up to 4 inputs (one per position,
// reused across the 8 channels): each strided NCHW input load now feeds 8 outputs
// instead of 4, so input-read traffic drops ~8x vs naive (input reads, strided by
// H*W across channels, are the integrated-GPU bottleneck). The 4 positions are
// STRIDED by npq so adjacent threads access adjacent addresses (coalesced).
//
// Loop order is (kh,kw) OUTER, ci INNER so the per-position boundary checks +
// input offsets are computed once per tap, not once per (ci,tap).
//
// No workgroup memory -> full GPU occupancy; plain per-invocation -> the JIT
// compiles it unchanged. Same result as conv_act.wgsl, including its `p.act`
// selector (0 = identity, 1 = ReLU, 2 = SiLU, 3 = sigmoid) — the uniform branch
// is coherent, so a ReLU model (ZipDepth) fuses as cheaply as a SiLU one (yolo).
// Dispatch: total = N * ceil(Cout/8) * ceil(Ho*Wo/4).

struct Params {
    N: u32, Cin: u32, H: u32, W: u32, Cout: u32,
    K: u32, stride: u32, pad: u32, Ho: u32, Wo: u32,
    act: u32,
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
    let ntc = (p.Cout + 7u) / 8u;       // channel octets
    let npq = (psz + 3u) / 4u;          // position quads
    if (idx >= p.N * ntc * npq) { return; }

    let pq = idx % npq;
    let tt = idx / npq;
    let cq = tt % ntc;
    let n = tt / ntc;
    let co0 = cq * 8u;
    let nc = min(8u, p.Cout - co0);

    // 4 coalesced positions strided by npq.
    let q0 = pq; let q1 = pq + npq; let q2 = pq + 2u * npq; let q3 = pq + 3u * npq;
    let ho0 = q0 / p.Wo; let wo0 = q0 % p.Wo;
    let ho1 = q1 / p.Wo; let wo1 = q1 % p.Wo;
    let ho2 = q2 / p.Wo; let wo2 = q2 % p.Wo;
    let ho3 = q3 / p.Wo; let wo3 = q3 % p.Wo;
    let v1 = q1 < psz; let v2 = q2 < psz; let v3 = q3 < psz;

    // 32 partial sums: a{pos}{ch}, pos 0..3, ch 0..7.
    var a00 = 0.0; var a01 = 0.0; var a02 = 0.0; var a03 = 0.0; var a04 = 0.0; var a05 = 0.0; var a06 = 0.0; var a07 = 0.0;
    var a10 = 0.0; var a11 = 0.0; var a12 = 0.0; var a13 = 0.0; var a14 = 0.0; var a15 = 0.0; var a16 = 0.0; var a17 = 0.0;
    var a20 = 0.0; var a21 = 0.0; var a22 = 0.0; var a23 = 0.0; var a24 = 0.0; var a25 = 0.0; var a26 = 0.0; var a27 = 0.0;
    var a30 = 0.0; var a31 = 0.0; var a32 = 0.0; var a33 = 0.0; var a34 = 0.0; var a35 = 0.0; var a36 = 0.0; var a37 = 0.0;

    let nci = n * p.Cin;
    for (var kh: u32 = 0u; kh < p.K; kh = kh + 1u) {
        for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
            let wtap = kh * p.K + kw;
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
                let wt4 = select(0.0, w[(co0 + 4u) * kg + wbase], 4u < nc);
                let wt5 = select(0.0, w[(co0 + 5u) * kg + wbase], 5u < nc);
                let wt6 = select(0.0, w[(co0 + 6u) * kg + wbase], 6u < nc);
                let wt7 = select(0.0, w[(co0 + 7u) * kg + wbase], 7u < nc);
                let xc = (nci + ci) * hw;
                if (ok0) { let v = x[xc + off0]; a00 += v*wt0; a01 += v*wt1; a02 += v*wt2; a03 += v*wt3; a04 += v*wt4; a05 += v*wt5; a06 += v*wt6; a07 += v*wt7; }
                if (ok1) { let v = x[xc + off1]; a10 += v*wt0; a11 += v*wt1; a12 += v*wt2; a13 += v*wt3; a14 += v*wt4; a15 += v*wt5; a16 += v*wt6; a17 += v*wt7; }
                if (ok2) { let v = x[xc + off2]; a20 += v*wt0; a21 += v*wt1; a22 += v*wt2; a23 += v*wt3; a24 += v*wt4; a25 += v*wt5; a26 += v*wt6; a27 += v*wt7; }
                if (ok3) { let v = x[xc + off3]; a30 += v*wt0; a31 += v*wt1; a32 += v*wt2; a33 += v*wt3; a34 += v*wt4; a35 += v*wt5; a36 += v*wt6; a37 += v*wt7; }
            }
        }
    }

    // Affine (BN-eval collapsed) + selected activation, then store. Inlined per
    // channel (the wgsl-cpu JIT has no user-function-call support). a{pos}{ch}.
    // `p.act` is uniform across the dispatch, so the branches are coherent.
    let is_r = p.act == 1u;
    let is_si = p.act == 2u;
    let is_sg = p.act == 3u;
    let nco = n * p.Cout;
    {
        let co = co0; let s = sb[2u*co]; let b = sb[2u*co+1u]; let row = (nco+co)*psz;
        { var z = a00*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q0] = z; }
        if (v1) { var z = a10*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q1] = z; }
        if (v2) { var z = a20*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q2] = z; }
        if (v3) { var z = a30*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q3] = z; }
    }
    if (1u < nc) {
        let co = co0+1u; let s = sb[2u*co]; let b = sb[2u*co+1u]; let row = (nco+co)*psz;
        { var z = a01*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q0] = z; }
        if (v1) { var z = a11*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q1] = z; }
        if (v2) { var z = a21*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q2] = z; }
        if (v3) { var z = a31*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q3] = z; }
    }
    if (2u < nc) {
        let co = co0+2u; let s = sb[2u*co]; let b = sb[2u*co+1u]; let row = (nco+co)*psz;
        { var z = a02*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q0] = z; }
        if (v1) { var z = a12*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q1] = z; }
        if (v2) { var z = a22*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q2] = z; }
        if (v3) { var z = a32*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q3] = z; }
    }
    if (3u < nc) {
        let co = co0+3u; let s = sb[2u*co]; let b = sb[2u*co+1u]; let row = (nco+co)*psz;
        { var z = a03*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q0] = z; }
        if (v1) { var z = a13*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q1] = z; }
        if (v2) { var z = a23*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q2] = z; }
        if (v3) { var z = a33*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q3] = z; }
    }
    if (4u < nc) {
        let co = co0+4u; let s = sb[2u*co]; let b = sb[2u*co+1u]; let row = (nco+co)*psz;
        { var z = a04*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q0] = z; }
        if (v1) { var z = a14*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q1] = z; }
        if (v2) { var z = a24*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q2] = z; }
        if (v3) { var z = a34*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q3] = z; }
    }
    if (5u < nc) {
        let co = co0+5u; let s = sb[2u*co]; let b = sb[2u*co+1u]; let row = (nco+co)*psz;
        { var z = a05*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q0] = z; }
        if (v1) { var z = a15*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q1] = z; }
        if (v2) { var z = a25*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q2] = z; }
        if (v3) { var z = a35*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q3] = z; }
    }
    if (6u < nc) {
        let co = co0+6u; let s = sb[2u*co]; let b = sb[2u*co+1u]; let row = (nco+co)*psz;
        { var z = a06*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q0] = z; }
        if (v1) { var z = a16*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q1] = z; }
        if (v2) { var z = a26*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q2] = z; }
        if (v3) { var z = a36*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q3] = z; }
    }
    if (7u < nc) {
        let co = co0+7u; let s = sb[2u*co]; let b = sb[2u*co+1u]; let row = (nco+co)*psz;
        { var z = a07*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q0] = z; }
        if (v1) { var z = a17*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q1] = z; }
        if (v2) { var z = a27*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q2] = z; }
        if (v3) { var z = a37*s+b; if (is_r) { z = max(z, 0.0); } else if (is_si) { z = z/(1.0+exp(-z)); } else if (is_sg) { z = 1.0/(1.0+exp(-z)); } y[row+q3] = z; }
    }
}
