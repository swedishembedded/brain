// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Kronecker row transform fused with the Hadamard-MLP middle stage: z = silu(d2 * (1 + softmax(logits) . cond_u) * K(x) + b2)
// @how   512-thread workgroup per row (one element per thread at width 512): the fb-stage into workgroup memory, one barrier, then the fa-stage with the conditioning and activation as its epilogue
// @opt   4
// @cpu   yes
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// mirage_kron_rows followed by mirage_hada_mid in one dispatch: the transform of a row is never stored, the thread that
// finishes output o applies the rank-r conditioning of its row to it. The structured Hadamard MLP keeps its memory
// saving (no dense n x n matrix) and loses a launch and an n-wide round trip per layer.
//   K[r, o] = post[o] * (a^T (pre * x) b)[perm[o]]          (as mirage_kron_rows, aux = [pre | perm | post])
//   out[r, o] = silu(d2[o] * (1 + sum_k softmax(lg[r, :])_k cond_u[k, o]) * K[r, o] + b2[o])
// lg: [rows, rank] conditioning logits; hp = [cond_u (rank * n) | d2 (n) | b2 (n)].
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
        var mx = -1.0e30;
        for (var k = 0u; k < p.cr; k = k + 1u) { mx = max(mx, lg[row * p.cr + k]); }
        var den = 0.0;
        var mix = 0.0;
        for (var k = 0u; k < p.cr; k = k + 1u) {
            let w = exp(lg[row * p.cr + k] - mx);
            den = den + w;
            mix = mix + w * hp[k * n + e];
        }
        let z = hp[p.cr * n + e] * (1.0 + mix / den) * v + hp[(p.cr + 1u) * n + e];
        out[row * n + e] = z / (1.0 + exp(-z));
    }
}
