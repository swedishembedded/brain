// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Bidirectional attention scores over RAGGED spans in one dispatch
// @how   one thread per output element, serial inner reduction, a work table that names each workgroup's span
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `attn_scores_cross`'s arithmetic for SELF-attention inside each packed span,
// addressed from a WORK TABLE instead of from a uniform sequence length - so
// one dispatch covers spans that are all different lengths:
//   scores[s,h,i,j] = (q[row0_s+i, h, :] . k[row0_s+j, h, :]) / sqrt(head_dim)
//
// WHY THIS EXISTS. The reverse pass used to run this family once PER SPAN, with
// the span's first row supplied as a non-zero storage-binding offset. A packed
// decision request is a couple of state windows and one slot per option, so
// that is six dispatches per span per layer - several hundred for one training
// step, each doing microseconds of arithmetic. Worse, a non-zero binding offset
// is exactly the pattern `backend_wgpu`'s Intel ANV workaround has to serialise
// (see `WgpuBackend::flush_serialized`), so on an Intel GPU every one of them
// also became its own queue submit and fence. This kernel takes the span's
// first row as DATA, binds whole buffers, and covers every span in one
// dispatch - which removes the dispatch count and the serialisation together.
// It is the reverse-pass twin of `flash_attn_bidir_spans`.
//
// THE WORK TABLE is four `u32` per workgroup, built on the host when the
// spans change: `(row0, len, score_base, elem0)`. `row0` is the span's first
// row in the PACKED slab, so this kernel addresses rows absolutely and has no
// batch dimension. `score_base` is where the span's `[heads, len, len]` score
// block starts; `elem0` is the first of the 64 elements of that block this
// workgroup owns. One table serves every kernel of this family that is indexed
// by score element; `Params` is shared by all six for the same reason.
//
// Q, K and V are three regions of ONE fused `[rows, 3*d_model]` buffer, which
// is why `qkv` is bound once and read at three offsets.

struct Params {
    n_wg: u32,         // work-table entries this dispatch covers
    n_heads: u32,
    head_dim: u32,
    qkv_stride: u32,   // 3 * d_model
    q_off: u32,        // 0
    k_off: u32,        // d_model
    v_off: u32,        // 2 * d_model (unused here; shared layout)
    d_model: u32,      // unused here; shared layout
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       work:   array<u32>;
@group(0) @binding(2) var<storage, read>       qkv:    array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

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
    let e = work[e4 + 3u] + lane;
    if (e >= p.n_heads * len * len) { return; }

    let hd = p.head_dim;
    let j = e % len;
    let r1 = e / len;
    let i = r1 % len;
    let h = r1 / len;

    let q_base = (row0 + i) * p.qkv_stride + p.q_off + h * hd;
    let k_base = (row0 + j) * p.qkv_stride + p.k_off + h * hd;
    var s = 0.0;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        s = s + qkv[q_base + d] * qkv[k_base + d];
    }
    scores[sbase + e] = s * inverseSqrt(f32(hd));
}
