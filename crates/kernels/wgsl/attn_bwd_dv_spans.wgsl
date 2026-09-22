// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Value gradient of span-local attention over RAGGED spans in one dispatch
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `attn_bwd_dv_cross_acc` for the ragged-span layout:
//   d_v[row0+j, h, d] = sum_{i<len} probs[s,h,i,j] * d_ctx[row0+i, h, d]
// See `attn_scores_spans` for why this family exists, for the work table, and
// for the shared `Params`, and `attn_bwd_dk_spans` for why there is no
// accumulate flag.

struct Params {
    n_wg: u32,
    n_heads: u32,
    head_dim: u32,
    qkv_stride: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    d_model: u32,      // d_ctx row stride
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       work:  array<u32>;
@group(0) @binding(2) var<storage, read>       probs: array<f32>;
@group(0) @binding(3) var<storage, read>       d_ctx: array<f32>;
@group(0) @binding(4) var<storage, read_write> d_qkv: array<f32>;  // V region

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
    let j = r1 % len;
    let h = r1 / len;

    var acc = 0.0;
    for (var i: u32 = 0u; i < len; i = i + 1u) {
        let prob = probs[sbase + (h * len + i) * len + j];
        acc = acc + prob * d_ctx[(row0 + i) * p.d_model + h * hd + d];
    }
    d_qkv[(row0 + j) * p.qkv_stride + p.v_off + h * hd + d] = acc;
}
