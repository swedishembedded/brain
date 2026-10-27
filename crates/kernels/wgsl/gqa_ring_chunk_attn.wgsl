// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Sliding-window GQA attention of a chunk of query rows against a run of keys: scores, softmax and apply in one kernel
// @how   one workgroup per (query row, head); the window's logits held in workgroup memory, softmax statistics reduced across the workgroup
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// The fused form of `gqa_scores_win`, `attn_softmax` and `gqa_apply` for ONE sequence's chunk of `n` query rows
// that sit at logical rows `m .. m + n` of a run of `len = m + n` keys (`m` earlier positions gathered from the
// ring, then the chunk's own): query row `r` attends keys `max(0, m + r - window + 1) ..= m + r` and nothing
// else. Unlike the full-sequence kernels it computes only the `n` rows that have a query and only their window.
//
//   q   : [n, n_heads * head_dim]      k, v : [len, n_kv_heads * head_dim]      out : [n, n_heads * head_dim]
//
// Requires `window <= 2048` and `head_dim <= 256` (the caller keeps the three-kernel path otherwise).
// Dispatch: n * n_heads workgroups.

struct Params {
    n: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    m: u32,
    window: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:   array<f32>;
@group(0) @binding(2) var<storage, read>       k:   array<f32>;
@group(0) @binding(3) var<storage, read>       v:   array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> qs: array<f32, 256>;
var<workgroup> logit: array<f32, 2048>;
var<workgroup> red: array<f32, 256>;
var<workgroup> kt: array<f32, 4352>;     // a tile of 16 key rows, each padded by 16 so the slices of a warp hit distinct banks
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let wg = wid.y * nwg.x + wid.x;
    if (wg >= p.n * p.n_heads) { return; }
    let t = lid.x;
    let hd = p.head_dim;
    let r = wg / p.n_heads;
    let h = wg % p.n_heads;
    let hkv = h / (p.n_heads / p.n_kv_heads);
    let i = p.m + r;
    var lo = 0u;
    if (i + 1u > p.window) { lo = i + 1u - p.window; }
    let cnt = i + 1u - lo;
    let kd = p.n_kv_heads * hd;

    if (t < hd) { qs[t] = q[(r * p.n_heads + h) * hd + t]; }
    workgroupBarrier();

    // Logits, a tile of 16 keys at a time: the tile is loaded coalesced, then thread (key t / 16, slice t % 16)
    // takes the dot product over its 1/16 of the channels and the 16 slices are summed.
    let scale = inverseSqrt(f32(hd));
    for (var base = 0u; base < cnt; base = base + 16u) {
        if (t < hd) {
            for (var kk = 0u; kk < 16u; kk = kk + 1u) {
                var val = 0.0;
                if (base + kk < cnt) { val = k[(lo + base + kk) * kd + hkv * hd + t]; }
                kt[kk * (hd + 16u) + t] = val;
            }
        }
        workgroupBarrier();
        let kk = t / 16u;
        let sl = t % 16u;
        var acc = 0.0;
        for (var d = sl; d < hd; d = d + 16u) { acc = acc + qs[d] * kt[kk * (hd + 16u) + d]; }
        part[t] = acc;
        workgroupBarrier();
        if (t < 16u && base + t < cnt) {
            var s = 0.0;
            for (var u = 0u; u < 16u; u = u + 1u) { s = s + part[t * 16u + u]; }
            logit[base + t] = s * scale;
        }
        workgroupBarrier();
    }
    var best = -3.4e38;
    for (var jj = t; jj < cnt; jj = jj + 256u) { best = max(best, logit[jj]); }
    red[t] = best;
    workgroupBarrier();
    var big = -3.4e38;
    for (var u = 0u; u < 256u; u = u + 1u) { big = max(big, red[u]); }
    workgroupBarrier();
    var sum = 0.0;
    for (var jj = t; jj < cnt; jj = jj + 256u) {
        let e = exp(logit[jj] - big);
        logit[jj] = e;
        sum = sum + e;
    }
    red[t] = sum;
    workgroupBarrier();
    var total = 0.0;
    for (var u = 0u; u < 256u; u = u + 1u) { total = total + red[u]; }

    if (t < hd) {
        // Four independent partial sums keep several value loads in flight.
        var a0 = 0.0;
        var a1 = 0.0;
        var a2 = 0.0;
        var a3 = 0.0;
        let vb = lo * kd + hkv * hd + t;
        var jj = 0u;
        for (; jj + 4u <= cnt; jj = jj + 4u) {
            a0 = a0 + logit[jj] * v[vb + jj * kd];
            a1 = a1 + logit[jj + 1u] * v[vb + (jj + 1u) * kd];
            a2 = a2 + logit[jj + 2u] * v[vb + (jj + 2u) * kd];
            a3 = a3 + logit[jj + 3u] * v[vb + (jj + 3u) * kd];
        }
        for (; jj < cnt; jj = jj + 1u) { a0 = a0 + logit[jj] * v[vb + jj * kd]; }
        let acc = (a0 + a1) + (a2 + a3);
        out[(r * p.n_heads + h) * hd + t] = acc / total;
    }
}
