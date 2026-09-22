// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Query gradient of span-local attention over RAGGED spans in one dispatch
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `attn_bwd_dq_cross` for the ragged-span layout:
//   d_q[row0+i, h, d] = scale * sum_{j<len} d_scores[s,h,i,j] * k[row0+j, h, d]
// See `attn_scores_spans` for why this family exists, for the work table, and
// for the shared `Params`.
//
// The table entry here is counted in `heads * len * head_dim` output elements
// per span. A query row belongs to exactly one workgroup element, so this
// ASSIGNS - unlike the key and value gradients, which sum over queries.

struct Params {
    n_wg: u32,
    n_heads: u32,
    head_dim: u32,
    qkv_stride: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    d_model: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       work:     array<u32>;
@group(0) @binding(2) var<storage, read>       d_scores: array<f32>;
@group(0) @binding(3) var<storage, read>       qkv:      array<f32>;  // K region
@group(0) @binding(4) var<storage, read_write> d_qkv:    array<f32>;  // Q region

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch), split into
    // the workgroup this thread belongs to and its lane within it.
    //
    // Derived from the GLOBAL id rather than read from `workgroup_id` /
    // `local_invocation_id` on purpose: without workgroup memory or a barrier
    // this is a plain per-invocation kernel, and the CPU JIT's per-invocation
    // path supplies `global_invocation_id` and `num_workgroups` only. Reading
    // the workgroup builtins here would make the kernel GPU-only for no gain,
    // since the two are the same number.
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let w = gidx / 64u;
    let lane = gidx % 64u;
    if (w >= p.n_wg) { return; }
    let e4 = 4u * w;
    let row0 = work[e4];
    let len = work[e4 + 1u];
    let sbase = work[e4 + 2u];
    let hd = p.head_dim;
    let e = work[e4 + 3u] + lane;
    if (e >= p.n_heads * len * hd) { return; }

    let d = e % hd;
    let r1 = e / hd;
    let i = r1 % len;
    let h = r1 / len;

    let s_base = sbase + (h * len + i) * len;
    var acc = 0.0;
    for (var j: u32 = 0u; j < len; j = j + 1u) {
        let k = qkv[(row0 + j) * p.qkv_stride + p.k_off + h * hd + d];
        acc = acc + d_scores[s_base + j] * k;
    }
    d_qkv[(row0 + i) * p.qkv_stride + p.q_off + h * hd + d] = acc * inverseSqrt(f32(hd));
}
