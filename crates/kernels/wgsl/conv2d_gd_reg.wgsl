// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Register-tiled GROUPED/DILATED conv2d forward (bias-free) - conv2d_gd's math with conv_act_reg's 8x4 register tile
// @how   register block per thread
// @opt   5
// @cpu   native
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Register-tiled GROUPED/DILATED conv2d forward (bias-free) — conv2d_gd's math
// with conv_act_reg's 8x4 register tile. Each invocation computes 8 output
// channels x 4 spatial positions, with the octet GROUP-ALIGNED: all 8 channels
// belong to the same group, so every strided NCHW input load feeds all 8
// (grouped 1x1 projections at high resolution are ZipDepth's hottest remaining
// kernel — 8x less input traffic is the whole win). Depthwise (cout_g == 1)
// degenerates to 1 channel x 4 positions, keeping the weight-reuse across the
// positions.
//
//   x : [N, Cin,        H,  W]
//   w : [Cout, Cin/G,   K,  K]
//   y : [N, Cout,       Ho, Wo]
//   Ho = (H + 2*pad - (dilation*(K-1)+1))/stride + 1   (likewise Wo)
//
// Octet layout: opg = ceil(cout_g/8) octets per group, ntc = G * opg. An octet
// never spans a group boundary (its tail lanes are masked by `nc`), which is
// what keeps the input load shared — channels of different groups read
// DIFFERENT input channels.
//
// Same (kh,kw)-outer / cl-inner loop order as conv_act_reg: the 4 positions'
// boundary checks + offsets are computed once per tap.
// Dispatch: total = N * G * ceil(Cout/G/8) * ceil(Ho*Wo/4).
//
// The name deliberately does NOT collide with the `conv2d`/`conv_act*` exact
// names `backend-cpu` binds its DENSE fast paths to; the CPU backend routes
// this (and `conv2d_gd`) to its grouped fast path instead.

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
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let kk = p.K * p.K;
    let cin_g = p.Cin / p.groups;
    let cout_g = p.Cout / p.groups;
    let kg = cin_g * kk;
    let hw = p.H * p.W;
    let psz = p.Ho * p.Wo;
    let opg = (cout_g + 7u) / 8u;       // octets per group
    let ntc = p.groups * opg;
    let npq = (psz + 3u) / 4u;          // position quads
    if (idx >= p.N * ntc * npq) { return; }

    let pq = idx % npq;
    let tt = idx / npq;
    let cq = tt % ntc;
    let n = tt / ntc;
    let g = cq / opg;                   // group of this octet
    let oc = cq % opg;                  // octet index within the group
    let co0 = g * cout_g + oc * 8u;
    let nc = min(8u, cout_g - oc * 8u); // lanes live in THIS group only
    let ci0 = g * cin_g;                // first input channel of the group

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

    let nci = n * p.Cin + ci0;
    for (var kh: u32 = 0u; kh < p.K; kh = kh + 1u) {
        for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
            let wtap = kh * p.K + kw;
            let hb0 = ho0 * p.stride + kh * p.dilation; let xb0 = wo0 * p.stride + kw * p.dilation;
            let ok0 = hb0 >= p.pad && xb0 >= p.pad && hb0 - p.pad < p.H && xb0 - p.pad < p.W;
            let off0 = (hb0 - p.pad) * p.W + (xb0 - p.pad);
            let hb1 = ho1 * p.stride + kh * p.dilation; let xb1 = wo1 * p.stride + kw * p.dilation;
            let ok1 = v1 && hb1 >= p.pad && xb1 >= p.pad && hb1 - p.pad < p.H && xb1 - p.pad < p.W;
            let off1 = (hb1 - p.pad) * p.W + (xb1 - p.pad);
            let hb2 = ho2 * p.stride + kh * p.dilation; let xb2 = wo2 * p.stride + kw * p.dilation;
            let ok2 = v2 && hb2 >= p.pad && xb2 >= p.pad && hb2 - p.pad < p.H && xb2 - p.pad < p.W;
            let off2 = (hb2 - p.pad) * p.W + (xb2 - p.pad);
            let hb3 = ho3 * p.stride + kh * p.dilation; let xb3 = wo3 * p.stride + kw * p.dilation;
            let ok3 = v3 && hb3 >= p.pad && xb3 >= p.pad && hb3 - p.pad < p.H && xb3 - p.pad < p.W;
            let off3 = (hb3 - p.pad) * p.W + (xb3 - p.pad);

            for (var cl: u32 = 0u; cl < cin_g; cl = cl + 1u) {
                let wbase = cl * kk + wtap;
                let wt0 = w[co0 * kg + wbase];
                let wt1 = select(0.0, w[(co0 + 1u) * kg + wbase], 1u < nc);
                let wt2 = select(0.0, w[(co0 + 2u) * kg + wbase], 2u < nc);
                let wt3 = select(0.0, w[(co0 + 3u) * kg + wbase], 3u < nc);
                let wt4 = select(0.0, w[(co0 + 4u) * kg + wbase], 4u < nc);
                let wt5 = select(0.0, w[(co0 + 5u) * kg + wbase], 5u < nc);
                let wt6 = select(0.0, w[(co0 + 6u) * kg + wbase], 6u < nc);
                let wt7 = select(0.0, w[(co0 + 7u) * kg + wbase], 7u < nc);
                let xc = (nci + cl) * hw;
                if (ok0) { let v = x[xc + off0]; a00 += v*wt0; a01 += v*wt1; a02 += v*wt2; a03 += v*wt3; a04 += v*wt4; a05 += v*wt5; a06 += v*wt6; a07 += v*wt7; }
                if (ok1) { let v = x[xc + off1]; a10 += v*wt0; a11 += v*wt1; a12 += v*wt2; a13 += v*wt3; a14 += v*wt4; a15 += v*wt5; a16 += v*wt6; a17 += v*wt7; }
                if (ok2) { let v = x[xc + off2]; a20 += v*wt0; a21 += v*wt1; a22 += v*wt2; a23 += v*wt3; a24 += v*wt4; a25 += v*wt5; a26 += v*wt6; a27 += v*wt7; }
                if (ok3) { let v = x[xc + off3]; a30 += v*wt0; a31 += v*wt1; a32 += v*wt2; a33 += v*wt3; a34 += v*wt4; a35 += v*wt5; a36 += v*wt6; a37 += v*wt7; }
            }
        }
    }

    // Raw store — no affine/act epilogue; grouped units apply BN/act separately
    // (their BN often sits over a SUM of branches, not over this conv alone).
    let nco = n * p.Cout;
    {
        let co = co0; let row = (nco+co)*psz;
        y[row+q0] = a00;
        if (v1) { y[row+q1] = a10; }
        if (v2) { y[row+q2] = a20; }
        if (v3) { y[row+q3] = a30; }
    }
    if (1u < nc) {
        let co = co0+1u; let row = (nco+co)*psz;
        y[row+q0] = a01;
        if (v1) { y[row+q1] = a11; }
        if (v2) { y[row+q2] = a21; }
        if (v3) { y[row+q3] = a31; }
    }
    if (2u < nc) {
        let co = co0+2u; let row = (nco+co)*psz;
        y[row+q0] = a02;
        if (v1) { y[row+q1] = a12; }
        if (v2) { y[row+q2] = a22; }
        if (v3) { y[row+q3] = a32; }
    }
    if (3u < nc) {
        let co = co0+3u; let row = (nco+co)*psz;
        y[row+q0] = a03;
        if (v1) { y[row+q1] = a13; }
        if (v2) { y[row+q2] = a23; }
        if (v3) { y[row+q3] = a33; }
    }
    if (4u < nc) {
        let co = co0+4u; let row = (nco+co)*psz;
        y[row+q0] = a04;
        if (v1) { y[row+q1] = a14; }
        if (v2) { y[row+q2] = a24; }
        if (v3) { y[row+q3] = a34; }
    }
    if (5u < nc) {
        let co = co0+5u; let row = (nco+co)*psz;
        y[row+q0] = a05;
        if (v1) { y[row+q1] = a15; }
        if (v2) { y[row+q2] = a25; }
        if (v3) { y[row+q3] = a35; }
    }
    if (6u < nc) {
        let co = co0+6u; let row = (nco+co)*psz;
        y[row+q0] = a06;
        if (v1) { y[row+q1] = a16; }
        if (v2) { y[row+q2] = a26; }
        if (v3) { y[row+q3] = a36; }
    }
    if (7u < nc) {
        let co = co0+7u; let row = (nco+co)*psz;
        y[row+q0] = a07;
        if (v1) { y[row+q1] = a17; }
        if (v2) { y[row+q2] = a27; }
        if (v3) { y[row+q3] = a37; }
    }
}
