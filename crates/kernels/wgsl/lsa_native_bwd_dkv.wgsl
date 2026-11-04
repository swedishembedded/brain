// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Native sparse attention training backward, key half: dK and dV over the queries that selected each key, no atomics
// @how   64-thread workgroup per (key row, KV head): (query, head) pairs in tiles of 64 (one per thread), channels split over threads
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
// The key half of `lsa_native_fwd`'s backward, run after
// `lsa_native_bwd_dq` (which leaves `D = dO . O` in `stats`). The selection
// inverted per key - which queries selected key `j` - lets one workgroup own
// `dK_j` and `dV_j` outright, so nothing is scattered:
//
//   over every query i that selected j and every head h of the KV group:
//     p   = exp(scale * q_ih . k_j - lse_ih)
//     dV_j += p * dO_ih
//     dK_j += scale * p * (dO_ih . v_j - D_ih) * q_ih
//
// The index continues past the forward's `[b * t, sel_stride]` selection:
//
//   inv_off : [b * t + 1] - key row `r`'s queries are
//             `inv[inv_off[r] .. inv_off[r + 1]]`
//   inv     : positions of those queries within the key's sequence
//
// Layouts are the forward's; `d_k` and `d_v` are overwritten. Limits:
// head_dim <= 256.

const WG: u32 = 64u;
const MAX_HD: u32 = 256u;

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
@group(0) @binding(1) var<storage, read>       q:       array<f32>;
@group(0) @binding(2) var<storage, read>       k:       array<f32>;
@group(0) @binding(3) var<storage, read>       v:       array<f32>;
@group(0) @binding(4) var<storage, read>       d_ctx:   array<f32>;
@group(0) @binding(5) var<storage, read>       stats:   array<f32>;
@group(0) @binding(6) var<storage, read>       index:   array<u32>;
@group(0) @binding(7) var<storage, read_write> d_k:     array<f32>;
@group(0) @binding(8) var<storage, read_write> d_v:     array<f32>;

var<workgroup> ks:  array<f32, MAX_HD>;
var<workgroup> vs:  array<f32, MAX_HD>;
var<workgroup> pp:  array<f32, WG>;
var<workgroup> dss: array<f32, WG>;
var<workgroup> qr:  array<u32, WG>;       // the pair's q / dO row offset

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = wgid.y * nwg.x + wgid.x;
    if (w >= p.b * p.t * p.n_kv_heads) { return; }
    let lt = lid.x;
    let hd = p.head_dim;
    let kvh = w % p.n_kv_heads;
    let row = w / p.n_kv_heads;        // b * t + j
    let seq = row / p.t;
    let bt0 = seq * p.t;
    let kvrow = (row * p.n_kv_heads + kvh) * hd;
    let scale = inverseSqrt(f32(hd));
    for (var c = lt; c < hd; c = c + WG) {
        ks[c] = k[kvrow + c];
        vs[c] = v[kvrow + c];
    }
    let rows = p.b * p.t;
    let inv_off = rows * p.sel_stride;
    let inv = inv_off + rows + 1u;
    let first = index[inv_off + row];
    let pairs = (index[inv_off + row + 1u] - first) * p.group;

    var ak: array<f32, 4>;
    var av: array<f32, 4>;
    for (var c = 0u; c < 4u; c = c + 1u) {
        ak[c] = 0.0;
        av[c] = 0.0;
    }
    for (var base = 0u; base < pairs; base = base + WG) {
        workgroupBarrier();
        if (base + lt < pairs) {
            let e = base + lt;
            let i = index[inv + first + e / p.group];
            let h = kvh * p.group + e % p.group;
            let qo = ((bt0 + i) * p.n_heads + h) * hd;
            qr[lt] = qo;
            var s = 0.0;
            var dp = 0.0;
            for (var c = 0u; c < hd; c = c + 1u) {
                s = s + q[qo + c] * ks[c];
                dp = dp + d_ctx[qo + c] * vs[c];
            }
            let li = (seq * p.n_heads + h) * p.t + i;
            let pr = exp(s * scale - stats[li]);
            pp[lt] = pr;
            dss[lt] = pr * (dp - stats[rows * p.n_heads + li]) * scale;
        }
        workgroupBarrier();
        let n = min(WG, pairs - base);
        for (var c = 0u; c * WG + lt < hd; c = c + 1u) {
            let ch = c * WG + lt;
            var a = ak[c];
            var b = av[c];
            for (var e = 0u; e < n; e = e + 1u) {
                a = a + dss[e] * q[qr[e] + ch];
                b = b + pp[e] * d_ctx[qr[e] + ch];
            }
            ak[c] = a;
            av[c] = b;
        }
    }
    for (var c = 0u; c * WG + lt < hd; c = c + 1u) {
        d_k[kvrow + c * WG + lt] = ak[c];
        d_v[kvrow + c * WG + lt] = av[c];
    }
}
