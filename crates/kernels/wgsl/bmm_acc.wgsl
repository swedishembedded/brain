// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Batched matmul, accumulating: out[b,m,n] += alpha * sum_k A[b,·]·B[b,·]
// @how   one thread per output element, serial inner reduction over k
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `bmm.wgsl`'s accumulating twin: identical addressing and `Params` contract
// (see that file's header for the full contract, including `trans_a`/
// `trans_b`/`alpha`/`*_off`), differing only in the final store:
//
//   out[b,m,n] = out[b,m,n] + alpha * sum_k Â[b,m,k] * B̂[b,k,n]
//
// A separate file rather than a runtime `accumulate` branch on `bmm.wgsl`,
// matching this engine's existing precedent of paired overwrite/accumulate
// siblings (e.g. `add.wgsl`'s in-place `+=` vs `add2.wgsl`'s out-of-place
// `=`): the two kernels differ only in the last line's addressing semantics
// (overwrite vs read-modify-write), which is a small enough body to
// duplicate rather than thread a branch through every caller's mental model
// of whether `out` is being read. Needed by Gated
// DeltaNet for `core_out = attn_inter + intra_scores @ v_new` (accumulate
// onto a value `bmm.wgsl` already wrote) and the recurrent state update
// `state = state*decay + decayed_k^T @ v_new` (accumulate onto a value
// `gdn_state_decay.wgsl` already scaled in place).

struct Params {
    batch: u32,
    m: u32,
    k: u32,
    n: u32,
    trans_a: u32,
    trans_b: u32,
    alpha: f32,
    a_off: u32,
    b_off: u32,
    out_off: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a:   array<f32>;
@group(0) @binding(2) var<storage, read>       b:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.batch * p.m * p.n;
    if (idx >= total) { return; }
    let bi = idx / (p.m * p.n);
    let rest = idx % (p.m * p.n);
    let mm = rest / p.n;
    let nn = rest % p.n;

    let a_base = p.a_off + bi * (p.m * p.k);
    let b_base = p.b_off + bi * (p.k * p.n);

    var acc = 0.0;
    for (var kk: u32 = 0u; kk < p.k; kk = kk + 1u) {
        var av: f32;
        if (p.trans_a == 0u) {
            av = a[a_base + mm * p.k + kk];
        } else {
            av = a[a_base + kk * p.m + mm];
        }
        var bv: f32;
        if (p.trans_b == 0u) {
            bv = b[b_base + kk * p.n + nn];
        } else {
            bv = b[b_base + nn * p.k + kk];
        }
        acc = acc + av * bv;
    }
    let o = p.out_off + bi * (p.m * p.n) + mm * p.n + nn;
    out[o] = out[o] + p.alpha * acc;
}
