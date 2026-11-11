// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  The whole structured Hadamard MLP of a row in one workgroup: three Kronecker rotations, the rank-r conditioning and SiLU, added to the residual
// @how   512-thread workgroup per row: the factors of all three rotations are staged once, the row stays in workgroup memory between the rotations
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// y += K3 (K2 (silu(d2 (1 + softmax(lg) . cond_u) * K1(x) + b2))) for one row, where K1, K2, K3 are the Kronecker row
// transforms of mirage_kron_rows (out[o] = post[o] * (a^T (pre * x) b)[perm[o]]). mirage_kron_hada, mirage_kron_rows and
// mirage_kron_rows (accumulating) run these as three dependent dispatches that store each rotated row to global memory
// and read it back; here the row never leaves the workgroup. Every thread owns one output element across all three
// rotations (its permutation, scales and conditioning terms are read once), the six factor matrices are staged into
// workgroup memory together up front so their loads overlap the first rotation, and the rotations are separated by
// the barriers the transposition needs.
//
// kp: the three rotations back to back, each [b (fb*fb) | a (fa*fa) | pre (n) | perm (n) | post (n)] as in Kron.
// lg: [rows, cr] conditioning logits   hp: [cond_u (cr * n) | d2 (n) | b2 (n)]   x: [rows, n]   out: [rows, n] (added to)
// Contract: n = fa * fb <= 512, fa <= 32, fb <= 32, cr <= 32. Params: rows, fa, fb, cr. Dispatch: one workgroup per row.

struct Params { rows: u32, fa: u32, fb: u32, cr: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       kp:  array<f32>;
@group(0) @binding(3) var<storage, read>       lg:  array<f32>;
@group(0) @binding(4) var<storage, read>       hp:  array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

var<workgroup> sa0: array<f32, 1024>;
var<workgroup> sb0: array<f32, 1024>;
var<workgroup> sa1: array<f32, 1024>;
var<workgroup> sb1: array<f32, 1024>;
var<workgroup> sa2: array<f32, 1024>;
var<workgroup> sb2: array<f32, 1024>;
var<workgroup> xp: array<f32, 512>;
var<workgroup> tt: array<f32, 512>;
var<workgroup> zs: array<f32, 512>;
var<workgroup> cw: array<f32, 32>;

@compute @workgroup_size(512)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    if (row >= p.rows) { return; }
    let e = li.x;
    let n = p.fa * p.fb;
    let fa2 = p.fa * p.fa;
    let fb2 = p.fb * p.fb;
    let rot = fb2 + fa2 + 3u * n;
    let live = e < n;
    // This thread's element: stage 1 reads row i, column l of the transposed layout, stage 2 reads column k, l of the permutation.
    let i = e / p.fb;
    let l = e % p.fb;
    for (var q = e; q < fb2; q = q + 512u) { sb0[q] = kp[q]; sb1[q] = kp[rot + q]; sb2[q] = kp[2u * rot + q]; }
    for (var q = e; q < fa2; q = q + 512u) { sa0[q] = kp[fb2 + q]; sa1[q] = kp[rot + fb2 + q]; sa2[q] = kp[2u * rot + fb2 + q]; }
    var src0 = 0u;
    var src1 = 0u;
    var pre1 = 0.0;
    var pre2 = 0.0;
    var post0 = 0.0;
    var post2 = 0.0;
    if (live) {
        xp[e] = kp[fb2 + fa2 + e] * x[row * n + e];
        src0 = u32(kp[fb2 + fa2 + n + e]);
        src1 = u32(kp[rot + fb2 + fa2 + n + e]);
        pre1 = kp[rot + fb2 + fa2 + e];
        pre2 = kp[2u * rot + fb2 + fa2 + e];
        post0 = kp[fb2 + fa2 + 2u * n + e];
        post2 = kp[2u * rot + fb2 + fa2 + 2u * n + e];
    }
    let src2 = select(0u, u32(kp[2u * rot + fb2 + fa2 + n + e]), live);
    if (e < p.cr) {
        var mx = -1.0e30;
        for (var c = 0u; c < p.cr; c = c + 1u) { mx = max(mx, lg[row * p.cr + c]); }
        cw[e] = exp(lg[row * p.cr + e] - mx);
    }
    workgroupBarrier();
    // Rotation 1 with the conditioning and SiLU.
    if (live) {
        var s = 0.0;
        for (var j = 0u; j < p.fb; j = j + 1u) { s = s + sb0[j * p.fb + l] * xp[i * p.fb + j]; }
        tt[e] = s;
    }
    workgroupBarrier();
    if (live) {
        let k = src0 / p.fb;
        let m = src0 % p.fb;
        var s = 0.0;
        for (var a = 0u; a < p.fa; a = a + 1u) { s = s + sa0[a * p.fa + k] * tt[a * p.fb + m]; }
        let v = s * post0;
        var den = 0.0;
        var mix = 0.0;
        for (var c = 0u; c < p.cr; c = c + 1u) {
            den = den + cw[c];
            mix = mix + cw[c] * hp[c * n + e];
        }
        let z = hp[p.cr * n + e] * (1.0 + mix / den) * v + hp[(p.cr + 1u) * n + e];
        zs[e] = z / (1.0 + exp(-z));
    }
    workgroupBarrier();
    // Rotation 2.
    if (live) { xp[e] = pre1 * zs[e]; }
    workgroupBarrier();
    if (live) {
        var s = 0.0;
        for (var j = 0u; j < p.fb; j = j + 1u) { s = s + sb1[j * p.fb + l] * xp[i * p.fb + j]; }
        tt[e] = s;
    }
    workgroupBarrier();
    if (live) {
        let k = src1 / p.fb;
        let m = src1 % p.fb;
        var s = 0.0;
        for (var a = 0u; a < p.fa; a = a + 1u) { s = s + sa1[a * p.fa + k] * tt[a * p.fb + m]; }
        zs[e] = s * kp[rot + fb2 + fa2 + 2u * n + e];
    }
    workgroupBarrier();
    // Rotation 3, accumulated into the residual.
    if (live) { xp[e] = pre2 * zs[e]; }
    workgroupBarrier();
    if (live) {
        var s = 0.0;
        for (var j = 0u; j < p.fb; j = j + 1u) { s = s + sb2[j * p.fb + l] * xp[i * p.fb + j]; }
        tt[e] = s;
    }
    workgroupBarrier();
    if (live) {
        let k = src2 / p.fb;
        let m = src2 % p.fb;
        var s = 0.0;
        for (var a = 0u; a < p.fa; a = a + 1u) { s = s + sa2[a * p.fa + k] * tt[a * p.fb + m]; }
        out[row * n + e] = out[row * n + e] + s * post2;
    }
}
