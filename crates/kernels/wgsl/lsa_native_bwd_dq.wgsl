// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Native sparse attention training backward, query half: dQ over each query's selected keys, and D = dO . O for the key half
// @how   64-thread workgroup per (query row, KV head) serving every head of the group: probabilities rebuilt from the forward's log-sum-exp a tile of (key, head) pairs at a time
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Swedish Embedded AB implements long-context models trained with sparse
// attention in time and memory linear in the context, for clients whose
// sequences outgrow quadratic attention. If your team needs expertise in
// sparse attention training kernels, you can procure our services by sending
// an email to info@swedishembedded.com.
//
// The query half of `lsa_native_fwd`'s backward. For query row `i`, head
// `h`, over its selected keys `j`:
//
//   p_j  = exp(scale * q . k_j - lse)        (the forward's probabilities)
//   dS_j = p_j * (dO . v_j - D),  D = dO . O
//   dQ   = scale * sum_j dS_j k_j
//
// `D` is written to the second half of `stats` (the forward's log-sum-exp is
// the first) for `lsa_native_bwd_dkv`, which runs after this kernel. As in
// the forward, a workgroup owns one query row and one KV head, so the
// group's heads share every key and value row it reads. Layouts and the
// index are the forward's; `d_q` is overwritten. Limits: head_dim <= 256,
// group <= 8.

const WG: u32 = 64u;
const HEADER: u32 = 3u;              // count, row, live keys
const MAX_G: u32 = 8u;

struct Params {
    b: u32,
    n_heads: u32,
    n_kv_heads: u32,
    t: u32,
    head_dim: u32,
    group: u32,
    sel_stride: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:     array<f32>;
@group(0) @binding(2) var<storage, read>       k:     array<f32>;
@group(0) @binding(3) var<storage, read>       v:     array<f32>;
@group(0) @binding(4) var<storage, read>       ctx:   array<f32>;
@group(0) @binding(5) var<storage, read>       d_ctx: array<f32>;
@group(0) @binding(6) var<storage, read_write> stats: array<f32>;
@group(0) @binding(7) var<storage, read>       index: array<u32>;
@group(0) @binding(8) var<storage, read_write> d_q:   array<f32>;

var<workgroup> qs:   array<f32, 2048>;   // MAX_G * MAX_HD, pre-scaled
var<workgroup> dos:  array<f32, 2048>;
var<workgroup> dss:  array<f32, WG>;
var<workgroup> keyj: array<u32, WG>;
var<workgroup> dd:   array<f32, MAX_G>;
var<workgroup> lse:  array<f32, MAX_G>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = wgid.y * nwg.x + wgid.x;
    if (w >= p.b * p.t * p.n_kv_heads) { return; }
    let lt = lid.x;
    let hd = p.head_dim;
    let g_n = p.group;
    let kvh = w % p.n_kv_heads;
    let row = w / p.n_kv_heads;
    let seq = row / p.t;
    let bt0 = seq * p.t;
    let i = row % p.t;
    let h0 = kvh * g_n;
    let qrow = (row * p.n_heads + h0) * hd;
    let scale = inverseSqrt(f32(hd));
    for (var e = lt; e < g_n * hd; e = e + WG) {
        qs[e] = q[qrow + e] * scale;
        dos[e] = d_ctx[qrow + e];
    }
    if (lt < g_n) {
        var s = 0.0;
        for (var c = 0u; c < hd; c = c + 1u) { s = s + d_ctx[qrow + lt * hd + c] * ctx[qrow + lt * hd + c]; }
        let li = (seq * p.n_heads + h0 + lt) * p.t + i;
        dd[lt] = s;
        lse[lt] = stats[li];
        stats[p.b * p.n_heads * p.t + li] = s;
    }
    let srow = row * p.sel_stride;
    let count = index[srow];
    let kt = WG / g_n;
    workgroupBarrier();

    var acc: array<f32, 32>;
    for (var a = 0u; a < 32u; a = a + 1u) { acc[a] = 0.0; }
    for (var base = 0u; base < count; base = base + kt) {
        let n = min(kt, count - base);
        if (lt < n) { keyj[lt] = index[srow + HEADER + base + lt]; }
        workgroupBarrier();
        if (lt < n * g_n) {
            let g = lt % g_n;
            let kv = ((bt0 + keyj[lt / g_n]) * p.n_kv_heads + kvh) * hd;
            var s = 0.0;
            var dp = 0.0;
            for (var c = 0u; c < hd; c = c + 1u) {
                s = s + qs[g * hd + c] * k[kv + c];
                dp = dp + dos[g * hd + c] * v[kv + c];
            }
            dss[lt] = exp(s - lse[g]) * (dp - dd[g]);
        }
        workgroupBarrier();
        for (var a = 0u; a * WG + lt < g_n * hd; a = a + 1u) {
            let idx = a * WG + lt;
            let g = idx / hd;
            let c = idx % hd;
            var s = acc[a];
            for (var e = 0u; e < n; e = e + 1u) {
                s = s + dss[e * g_n + g] * k[((bt0 + keyj[e]) * p.n_kv_heads + kvh) * hd + c];
            }
            acc[a] = s;
        }
        workgroupBarrier();
    }
    for (var a = 0u; a * WG + lt < g_n * hd; a = a + 1u) {
        d_q[qrow + a * WG + lt] = acc[a] * scale;
    }
}
