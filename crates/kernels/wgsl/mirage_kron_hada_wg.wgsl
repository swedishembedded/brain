// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  mirage_kron_hada with both factor matrices and the scaled input staged in workgroup memory
// @how   512-thread workgroup per row: stage the factors and pre * x and the conditioning weights, barrier, fb-stage, barrier, fa-stage with the epilogue
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// The operator of mirage_kron_hada (mirage_kron_rows followed by the Hadamard-MLP conditioning and SiLU) built like
// mirage_kron_rows_wg: factors and pre * x staged once, the two sums walking workgroup memory. The conditioning
// softmax is formed once per row (the original recomputes its exponentials in every thread's epilogue).
// Contract: fa * fb <= 1024, fa <= 32, fb <= 32, cr <= 32; the original's params and bindings.
// Params: rows, fa, fb, cr (the conditioning rank). Dispatch: one workgroup per row.

struct Params { rows: u32, fa: u32, fb: u32, cr: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read>       b:    array<f32>;
@group(0) @binding(3) var<storage, read>       a:    array<f32>;
@group(0) @binding(4) var<storage, read>       aux:  array<f32>;
@group(0) @binding(5) var<storage, read>       lg:   array<f32>;
@group(0) @binding(6) var<storage, read>       hp:   array<f32>;
@group(0) @binding(7) var<storage, read_write> out:  array<f32>;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;
var<workgroup> xp: array<f32, 1024>;
var<workgroup> t: array<f32, 1024>;
var<workgroup> cw: array<f32, 32>;

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
    if (tid < p.cr) {
        var mx = -1.0e30;
        for (var k = 0u; k < p.cr; k = k + 1u) { mx = max(mx, lg[row * p.cr + k]); }
        cw[tid] = exp(lg[row * p.cr + tid] - mx);
    }
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
    var den = 0.0;
    for (var k = 0u; k < p.cr; k = k + 1u) { den = den + cw[k]; }
    for (var e = tid; e < n; e = e + 512u) {
        let src = u32(aux[n + e]);
        let k = src / p.fb;
        let l = src % p.fb;
        var s = 0.0;
        for (var i = 0u; i < p.fa; i = i + 1u) {
            s = s + sa[i * p.fa + k] * t[i * p.fb + l];
        }
        let v = s * aux[2u * n + e];
        var mix = 0.0;
        for (var c = 0u; c < p.cr; c = c + 1u) { mix = mix + cw[c] * hp[c * n + e]; }
        let z = hp[p.cr * n + e] * (1.0 + mix / den) * v + hp[(p.cr + 1u) * n + e];
        out[row * n + e] = z / (1.0 + exp(-z));
    }
}
