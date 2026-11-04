// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Latent sparse attention index keys read from an int8 KV pool: dequantise, un-rotate, project, normalise
// @how   64-thread workgroup per row: pairs un-rotated into workgroup memory, one thread per index channel, tree-free norm by one thread
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant int8
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
// `rope_mode` names the rotation: 0 none (step 1 skipped), 1 half-split RoPE
// at `rope_base` (the angle computed exactly as the plain forward kernels
// compute it), 2 a per-channel inverse-frequency table `inv_freq[half]` (YaRN:
// `rope_paged_yarn`'s angle; its uniform attention factor scales the whole
// vector, which the normalisation removes).
// Rotation is linear and the same for every head of a row, so summing the
// group before un-rotating gives the same result with a fraction of the work.
//
// Limits: `n_kv_heads * head_dim <= MAX_IN`, `index_dim <= MAX_IN`; `identity`
// keeps the summed heads unprojected, as `lsa_index_vectors` does.
//
// The int8-pool sibling of `lsa_index_vectors` for KEYS (one head per KV
// head): row `r` is the key at token slot `row_off[r]` of the packed pool
// (`paged_kv_append_i8_clipped_batched`'s layout: four signed bytes to a
// word, one f32 scale per (slot, kv head)), dequantised as it is read.
// The byte decode between the kv8-decode markers becomes FP8 E4M3 through
// `kernels::template::fp8_decode_variant`.

const MAX_IN: u32 = 2048u;
const WG: u32 = 64u;

struct Params {
    rows: u32,
    heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    index_dim: u32,
    rope_base: f32,
    rope_mode: u32,
    out0: u32,         // first output row (a layer's slot in a shared buffer)
    identity: u32,     // 1: no projection, the summed heads are the index vector
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:       array<u32>;
@group(0) @binding(2) var<storage, read>       row_off: array<u32>;
@group(0) @binding(3) var<storage, read>       pos:     array<u32>;
@group(0) @binding(4) var<storage, read>       w:       array<f32>;
@group(0) @binding(5) var<storage, read_write> out:     array<f32>;
@group(0) @binding(6) var<storage, read>       scales:  array<f32>;
@group(0) @binding(7) var<storage, read>       inv_freq: array<f32>;

var<workgroup> u:  array<f32, MAX_IN>;

// @kv8-decode-begin
// Byte `lane` (0 = lowest) of a packed word as a signed int8, widened to f32.
fn kv8_byte(w: u32, lane: u32) -> f32 {
    return f32(bitcast<i32>(w << (24u - 8u * lane)) >> 24u);
}
// @kv8-decode-end

// Channel `i` of the key at token slot `slot`, dequantised.
fn key_at(slot: u32, i: u32) -> f32 {
    let e = slot * p.n_kv_heads * p.head_dim + i;
    return kv8_byte(x[e / 4u], e % 4u) * scales[slot * p.n_kv_heads + i / p.head_dim];
}
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
            let h = (kvh * group + g) * hd;
            s0 = s0 + key_at(base, h + m);
            s1 = s1 + key_at(base, h + m + half);
        }
        var cs = 1.0;
        var sn = 0.0;
        if (p.rope_mode == 1u) {
            let angle = fpos * pow(p.rope_base, -f32(2u * m) / f32(hd));
            cs = cos(angle);
            sn = sin(angle);
        } else if (p.rope_mode == 2u) {
            let angle = fpos * inv_freq[m];
            cs = cos(angle);
            sn = sin(angle);
        }
        u[kvh * hd + m] = s0 * cs + s1 * sn;
        u[kvh * hd + m + half] = s1 * cs - s0 * sn;
    }
    workgroupBarrier();

    // Each thread owns channels `lt`, `lt + 64`, ...: written unnormalised,
    // then scaled by the row's norm (its own writes, so no barrier between).
    let in_dim = p.n_kv_heads * hd;
    let orow = (p.out0 + r) * p.index_dim;
    var part = 0.0;
    for (var o = lt; o < p.index_dim; o = o + WG) {
        var acc = 0.0;
        if (p.identity == 1u) {
            acc = u[o];
        } else {
            for (var i = 0u; i < in_dim; i = i + 1u) {
                acc = acc + u[i] * w[i * p.index_dim + o];
            }
        }
        out[orow + o] = acc;
        part = part + acc * acc;
    }
    sq[lt] = part;
    workgroupBarrier();
    if (lt == 0u) {
        var total = 0.0;
        for (var t = 0u; t < WG; t = t + 1u) { total = total + sq[t]; }
        sq[0] = total;
    }
    workgroupBarrier();
    let norm = sqrt(sq[0]);
    for (var o = lt; o < p.index_dim; o = o + WG) {
        out[orow + o] = select(0.0, out[orow + o] / norm, norm > 0.0);
    }
}
