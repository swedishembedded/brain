// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Latent sparse attention index vectors from post-RoPE attention heads: un-rotate, sum each KV group, project, normalise
// @how   64-thread workgroup per row: pairs un-rotated into workgroup memory, one thread per index channel, tree-free norm by one thread
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Swedish Embedded AB implements long-context attention whose key index is
// derived from the KV cache it already holds, for clients that cannot afford
// to recompute cached tokens. If your team needs expertise in sparse
// attention indexing, you can procure our services by sending an email to
// info@swedishembedded.com.
//
// One index vector per row. A row is `heads` attention heads of `head_dim`
// (queries: every query head; keys: every KV head) starting at element
// `row_off[r]` of `x` - a chunk's query rows or a cached key's slot in the
// paged pool alike - rotated by half-split RoPE at position `pos[r]`:
//
//   1. un-rotate each pair (m, m + half) with the forward kernels' own angle
//      `pos * rope_base^(-2m / head_dim)` (`qknorm_rope_fused`, `rope_paged`),
//      so the inverse uses the very cos/sin that rotated it;
//   2. sum the `heads / n_kv_heads` heads of each KV group (one head each for
//      keys), giving `n_kv_heads * head_dim` channels - the head-summed logit
//      `sum_h q_h . k_(h/G)` is exactly the dot product of these;
//   3. project by `w` (`[n_kv_heads * head_dim, index_dim]`, row-major);
//   4. normalise to unit length (a zero row stays zero).
//
// `rope_base == 0` means the heads carry no rotation and step 1 is skipped.
// Rotation is linear and the same for every head of a row, so summing the
// group before un-rotating gives the same result with a fraction of the work.
//
// Limits: `n_kv_heads * head_dim <= MAX_IN`, `index_dim <= 64`.

const MAX_IN: u32 = 2048u;
const WG: u32 = 64u;

struct Params {
    rows: u32,
    heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    index_dim: u32,
    rope_base: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:       array<f32>;
@group(0) @binding(2) var<storage, read>       row_off: array<u32>;
@group(0) @binding(3) var<storage, read>       pos:     array<u32>;
@group(0) @binding(4) var<storage, read>       w:       array<f32>;
@group(0) @binding(5) var<storage, read_write> out:     array<f32>;

var<workgroup> u:  array<f32, MAX_IN>;
var<workgroup> sq: array<f32, WG>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let r = wgid.y * nwg.x + wgid.x;
    if (r >= p.rows) { return; }
    let lt = lid.x;
    let hd = p.head_dim;
    let half = hd / 2u;
    let group = p.heads / p.n_kv_heads;
    let base = row_off[r];
    let fpos = f32(pos[r]);

    for (var e = lt; e < p.n_kv_heads * half; e = e + WG) {
        let kvh = e / half;
        let m = e % half;
        var s0 = 0.0;
        var s1 = 0.0;
        for (var g = 0u; g < group; g = g + 1u) {
            let h = base + (kvh * group + g) * hd;
            s0 = s0 + x[h + m];
            s1 = s1 + x[h + m + half];
        }
        var cs = 1.0;
        var sn = 0.0;
        if (p.rope_base != 0.0) {
            let angle = fpos * pow(p.rope_base, -f32(2u * m) / f32(hd));
            cs = cos(angle);
            sn = sin(angle);
        }
        u[kvh * hd + m] = s0 * cs + s1 * sn;
        u[kvh * hd + m + half] = s1 * cs - s0 * sn;
    }
    workgroupBarrier();

    let in_dim = p.n_kv_heads * hd;
    var acc = 0.0;
    if (lt < p.index_dim) {
        for (var i = 0u; i < in_dim; i = i + 1u) {
            acc = acc + u[i] * w[i * p.index_dim + lt];
        }
    }
    sq[lt] = acc * acc;
    workgroupBarrier();
    if (lt == 0u) {
        var total = 0.0;
        for (var t = 0u; t < WG; t = t + 1u) { total = total + sq[t]; }
        sq[0] = total;
    }
    workgroupBarrier();
    let norm = sqrt(sq[0]);
    if (lt < p.index_dim) {
        out[r * p.index_dim + lt] = select(0.0, acc / norm, norm > 0.0);
    }
}
