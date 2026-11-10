// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Kronecker product of two small matrices applied to every row, with a diagonal scale before, a gather permutation and a scale after
// @how   512-thread workgroup per row (one element per thread at width 512): the fb-stage into workgroup memory, one barrier, then the fa-stage with the epilogue
// @opt   4
// @cpu   yes
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// out[r, o] (+)= post[o] * K[r, perm[o]] with K = a^T (pre * x) b for the row viewed as [fa, fb]:
//   K[k, l] = sum_i a[i, k] * (sum_j b[j, l] * pre[i, j] * x[r, i, j]).
// The sequency Walsh transform (a and b the +-1 Sylvester factors, perm the bit-reversed Gray code, post = 1/sqrt(n))
// and the Hadamard MLP rotations (learned factors, learned permutations and diagonals) are this one operator, so the
// banded convolutions and the rotations no longer need an n x n matrix per layer. fa * fb <= 1024.
// aux: [pre | perm | post], each n = fa * fb values (permutation entries ride in f32 words, exact below 2^24); one buffer
// keeps the binding count of the fused variants (mirage_kron_hada) inside the 8-storage-buffer budget.
// Params: rows, fa, fb, acc (nonzero: add to out instead of storing). Dispatch: one workgroup per row.

struct Params { rows: u32, fa: u32, fb: u32, acc: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read>       b:    array<f32>;
@group(0) @binding(3) var<storage, read>       a:    array<f32>;
@group(0) @binding(4) var<storage, read>       aux:  array<f32>;
@group(0) @binding(5) var<storage, read_write> out:  array<f32>;

var<workgroup> t: array<f32, 1024>;

@compute @workgroup_size(512)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let n = p.fa * p.fb;
    for (var e = li.x; e < n; e = e + 512u) {
        let i = e / p.fb;
        let l = e % p.fb;
        var s = 0.0;
        for (var j = 0u; j < p.fb; j = j + 1u) {
            s = s + b[j * p.fb + l] * aux[i * p.fb + j] * x[row * n + i * p.fb + j];
        }
        t[e] = s;
    }
    workgroupBarrier();
    for (var e = li.x; e < n; e = e + 512u) {
        let src = u32(aux[n + e]);
        let k = src / p.fb;
        let l = src % p.fb;
        var s = 0.0;
        for (var i = 0u; i < p.fa; i = i + 1u) {
            s = s + a[i * p.fa + k] * t[i * p.fb + l];
        }
        let v = s * aux[2u * n + e];
        if (p.acc != 0u) {
            out[row * n + e] = out[row * n + e] + v;
        } else {
            out[row * n + e] = v;
        }
    }
}
