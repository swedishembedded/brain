// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Native sparse attention training forward: each query attends only its selected keys, writing the context and the row log-sum-exp
// @how   64-thread workgroup per (query row, KV head) serving every head of the group: (key, head) pairs scored a tile at a time, online softmax per head, (head, channel) accumulators split over threads
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
// The forward of native latent sparse attention (N-LSA): query row `i` of
// batch row `b` attends exactly the keys its selection lists, with the
// softmax over those keys only - dense causal attention with every other key
// masked out, without the `[t, t]` scores. Layouts follow the causal GQA
// flash trio: q and ctx `[b * t, n_heads * head_dim]`, k and v
// `[b * t, n_kv_heads * head_dim]`; query head `h` reads KV head `h / group`;
// the scale is `1 / sqrt(head_dim)`.
//
//   index : u32, the selection then its inverse (`lsa_native_bwd_dkv`):
//           first `[b * t, sel_stride]` rows as `lsa_read` writes them - the
//           key count, the row, its live keys, then the selected positions
//           within its own sequence (each causal, none repeated); the same
//           row serves every head
//   stats : f32 `[2, b, n_heads, t]` - the row log-sum-exp (written here),
//           then the backward's `D = dO . O`
//
// A workgroup owns one query row and one KV head, so the group's heads share
// every key and value row it reads. A tile is `64 / group` keys: thread `e`
// scores key `e / group` for head `e % group`; thread `g < group` folds its
// head's scores into that head's running max and sum; then thread `lt` owns
// the (head, channel) accumulators `lt`, `lt + 64`, ... over the tile. The
// log-sum-exp keeps `max + log(sum)`, all the backward needs to rebuild the
// probabilities. Limits: head_dim <= 256, group <= 8.

const WG: u32 = 64u;
const HEADER: u32 = 3u;              // count, row, live keys
const MAX_HD: u32 = 256u;
const MAX_G: u32 = 8u;
const NEG: f32 = -3.4e38;

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
@group(0) @binding(4) var<storage, read>       index: array<u32>;
@group(0) @binding(5) var<storage, read_write> ctx:   array<f32>;
@group(0) @binding(6) var<storage, read_write> stats: array<f32>;

var<workgroup> qs:   array<f32, 2048>;   // MAX_G * MAX_HD, pre-scaled
var<workgroup> ps:   array<f32, WG>;     // the tile's scores, then probabilities
var<workgroup> keyj: array<u32, WG>;
var<workgroup> ms:   array<f32, MAX_G>;
var<workgroup> ls:   array<f32, MAX_G>;
var<workgroup> corr: array<f32, MAX_G>;

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
    let row = w / p.n_kv_heads;        // b * t + i
    let bt0 = (row / p.t) * p.t;
    let i = row % p.t;
    let h0 = kvh * g_n;
    let scale = inverseSqrt(f32(hd));
    for (var e = lt; e < g_n * hd; e = e + WG) {
        qs[e] = q[(row * p.n_heads + h0) * hd + e] * scale;
    }
    if (lt < g_n) {
        ms[lt] = NEG;
        ls[lt] = 0.0;
    }
    let srow = row * p.sel_stride;
    let count = index[srow];
    let kt = WG / g_n;                 // keys per tile
    workgroupBarrier();

    var acc: array<f32, 32>;           // MAX_G * MAX_HD / WG
    for (var a = 0u; a < 32u; a = a + 1u) { acc[a] = 0.0; }
    for (var base = 0u; base < count; base = base + kt) {
        let n = min(kt, count - base);
        if (lt < n) { keyj[lt] = index[srow + HEADER + base + lt]; }
        workgroupBarrier();
        if (lt < n * g_n) {
            let g = lt % g_n;
            let krow = ((bt0 + keyj[lt / g_n]) * p.n_kv_heads + kvh) * hd;
            var d = 0.0;
            for (var c = 0u; c < hd; c = c + 1u) { d = d + qs[g * hd + c] * k[krow + c]; }
            ps[lt] = d;
        }
        workgroupBarrier();
        if (lt < g_n) {
            var mt = ms[lt];
            for (var e = 0u; e < n; e = e + 1u) { mt = max(mt, ps[e * g_n + lt]); }
            var sum = 0.0;
            for (var e = 0u; e < n; e = e + 1u) {
                let pe = exp(ps[e * g_n + lt] - mt);
                ps[e * g_n + lt] = pe;
                sum = sum + pe;
            }
            let cr = exp(ms[lt] - mt);
            corr[lt] = cr;
            ls[lt] = ls[lt] * cr + sum;
            ms[lt] = mt;
        }
        workgroupBarrier();
        for (var a = 0u; a * WG + lt < g_n * hd; a = a + 1u) {
            let idx = a * WG + lt;
            let g = idx / hd;
            let c = idx % hd;
            var s = acc[a] * corr[g];
            for (var e = 0u; e < n; e = e + 1u) {
                s = s + ps[e * g_n + g] * v[((bt0 + keyj[e]) * p.n_kv_heads + kvh) * hd + c];
            }
            acc[a] = s;
        }
        workgroupBarrier();
    }
    for (var a = 0u; a * WG + lt < g_n * hd; a = a + 1u) {
        let idx = a * WG + lt;
        let g = idx / hd;
        let l = ls[g];
        ctx[(row * p.n_heads + h0) * hd + idx] = select(0.0, acc[a] / l, l > 0.0);
    }
    if (lt < g_n) {
        stats[((row / p.t) * p.n_heads + h0 + lt) * p.t + i] = ms[lt] + log(ls[lt]);
    }
}
