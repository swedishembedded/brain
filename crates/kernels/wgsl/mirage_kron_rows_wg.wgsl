// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  mirage_kron_rows with both factor matrices and the scaled input staged in workgroup memory
// @how   512-thread workgroup per row: stage the factors and pre * x, barrier, fb-stage, barrier, fa-stage with the epilogue
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// The operator of mirage_kron_rows (out[r, o] (+)= post[o] * K[r, perm[o]], K = a^T (pre * x) b over the row viewed as
// [fa, fb]) for a device whose workgroups execute any number of barriers. The original reads every factor entry, the
// scale and the input from global memory inside two dependent loops of 16 to 32 steps per thread; here the factors
// (fa * fa and fb * fb values, at most 1024 each) and pre * x are copied to workgroup memory once, coalesced, and the
// loops walk workgroup memory. The input is scaled before the sum (pre * x, then times b) where the original forms
// (b * pre) * x, so a result differs from the original by rounding.
// Contract: fa * fb <= 1024, fa <= 32 and fb <= 32 (the caller falls back otherwise); the original's params and bindings.
// Params: rows, fa, fb, acc (nonzero: add to out instead of storing). Dispatch: one workgroup per row.

struct Params { rows: u32, fa: u32, fb: u32, acc: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read>       b:    array<f32>;
@group(0) @binding(3) var<storage, read>       a:    array<f32>;
@group(0) @binding(4) var<storage, read>       aux:  array<f32>;
@group(0) @binding(5) var<storage, read_write> out:  array<f32>;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;
var<workgroup> xp: array<f32, 1024>;
var<workgroup> t: array<f32, 1024>;

@compute @workgroup_size(512)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let tid = li.x;
    let n = p.fa * p.fb;
    for (var e = tid; e < p.fa * p.fa; e = e + 512u) { sa[e] = a[e]; }
    for (var e = tid; e < p.fb * p.fb; e = e + 512u) { sb[e] = b[e]; }
    for (var e = tid; e < n; e = e + 512u) { xp[e] = aux[e] * x[row * n + e]; }
    workgroupBarrier();
    for (var e = tid; e < n; e = e + 512u) {
        let i = e / p.fb;
        let l = e % p.fb;
        var s = 0.0;
        for (var j = 0u; j < p.fb; j = j + 1u) {
            s = s + sb[j * p.fb + l] * xp[i * p.fb + j];
        }
        t[e] = s;
    }
    workgroupBarrier();
    for (var e = tid; e < n; e = e + 512u) {
        let src = u32(aux[n + e]);
        let k = src / p.fb;
        let l = src % p.fb;
        var s = 0.0;
        for (var i = 0u; i < p.fa; i = i + 1u) {
            s = s + sa[i * p.fa + k] * t[i * p.fb + l];
        }
        let v = s * aux[2u * n + e];
        if (p.acc != 0u) {
            out[row * n + e] = out[row * n + e] + v;
        } else {
            out[row * n + e] = v;
        }
    }
}
