// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Greedy token of banned-token-adjusted logits: argmax over rows of (logits + ban), lowest index among equal values
// @how   512-thread workgroup per row, four values in flight per thread, tree fold of (value, index) pairs down to sixteen then a serial pass
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// mirage_argmax_wg after an elementwise add of a ban row (0 or a large negative per token), as one dispatch that reads
// both and never writes the sum: the step only needs the picked token. A 64-thread workgroup walking an 8299-wide
// vocabulary row serially was the slowest kernel of the decoder step after the attention. The value compared is the
// f32 sum logits[i] + ban[i], exactly what the separate add stores; ties resolve to the lowest index as before.
//
//   x: [m, n] logits   ban: [m, n]   out: [m] picked index as f32. Params: m, n. Dispatch: one workgroup per row.

struct Params { m: u32, n: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       ban: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> pv: array<f32, 512>;
var<workgroup> pi: array<u32, 512>;

@compute @workgroup_size(512)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.m) { return; }
    let base = row * p.n;
    var best = -3.4e38;
    var at = 0xFFFFFFFFu;
    var c = t;
    for (; c + 1536u < p.n; c = c + 2048u) {
        let v0 = x[base + c] + ban[base + c];
        let v1 = x[base + c + 512u] + ban[base + c + 512u];
        let v2 = x[base + c + 1024u] + ban[base + c + 1024u];
        let v3 = x[base + c + 1536u] + ban[base + c + 1536u];
        if (v0 > best) { best = v0; at = c; }
        if (v1 > best) { best = v1; at = c + 512u; }
        if (v2 > best) { best = v2; at = c + 1024u; }
        if (v3 > best) { best = v3; at = c + 1536u; }
    }
    for (; c < p.n; c = c + 512u) {
        let v = x[base + c] + ban[base + c];
        if (v > best) { best = v; at = c; }
    }
    pv[t] = best;
    pi[t] = at;
    workgroupBarrier();
    for (var s = 256u; s > 8u; s = s >> 1u) {
        if (t < s) {
            let o = t + s;
            if (pv[o] > pv[t] || (pv[o] == pv[t] && pi[o] < pi[t])) { pv[t] = pv[o]; pi[t] = pi[o]; }
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        var gb = -3.4e38;
        var gi = 0xFFFFFFFFu;
        for (var i = 0u; i < 16u; i = i + 1u) {
            if (pv[i] > gb || (pv[i] == gb && pi[i] < gi)) { gb = pv[i]; gi = pi[i]; }
        }
        out[row] = f32(gi);
    }
}
