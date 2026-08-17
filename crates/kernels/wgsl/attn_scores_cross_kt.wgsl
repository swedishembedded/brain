// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Cross-attention scores against a key-minor K (`kv_k_headt` output) - same math as attn_scores_cross, coalesced
// @how   one thread per output element, serial inner reduction over head_dim
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Cross-attention scores:
//   scores[b,h,i,j] = (q[b,i,h,:] . k[b,j,h,:]) / sqrt(head_dim)
// computed exactly as `attn_scores_cross.wgsl` does - same output values, same
// `((b*H + h)*T_dec + i)*T_enc + j` layout - but reading K from the KEY-MINOR
// `[d_model, T_enc]` buffer `kv_k_headt.wgsl` produces instead of from the
// fused KV slab.
//
// That is the whole point. The thread index runs `j` fastest, because `j` is
// the axis the output is contiguous in. Against the natural `[T_enc, d_model]`
// layout that means every lane of a warp reads an address `kv_stride` floats
// from its neighbour, so each load is its own memory transaction; measured on a
// P40 at 91 GFLOP/s, 0.77% of fp32 peak and 3.5% of the bandwidth roofline -
// under a tenth of BOTH, which is the definition of a defect rather than a
// ceiling. With K transposed, `kt[(h*hd + d)*T_enc + j]` is contiguous in `j`,
// the warp's loads coalesce, and the whole 3 MB of K stays L2-resident across
// the T_dec sweep - which is why `attn_apply_cross`, moving the SAME bytes with
// a contiguous thread index, already ran 5.4x faster.
//
// Q still comes from the DECODER buffer in fused-QKV layout (stride q_stride,
// region at q_off), indexed by query position i; it is broadcast across the
// warp, so its layout does not matter.
//
// `kt` carries NO batch axis: it is one encoder memory shared by every sample,
// which is what a diffusion transformer's text context is. `bsz > 1` therefore
// indexes only `q` and `scores`, exactly as the batched flash path does.
//
// One invocation per (b,h,i,j).

struct Params {
    bsz: u32,
    n_heads: u32,
    t_dec: u32,        // query length
    t_enc: u32,        // key length
    head_dim: u32,
    q_stride: u32,     // 3*d_model (decoder fused QKV)
    q_off: u32,        // 0
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:      array<f32>;  // decoder buffer
@group(0) @binding(2) var<storage, read>       kt:     array<f32>;  // [d_model, T_enc]
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let Tq = p.t_dec;
    let Tk = p.t_enc;
    let total = p.bsz * p.n_heads * Tq * Tk;
    if (gidx >= total) { return; }

    let j = gidx % Tk;
    let r1 = gidx / Tk;
    let i = r1 % Tq;
    let r2 = r1 / Tq;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;

    let hd = p.head_dim;
    let q_base = (b * Tq + i) * p.q_stride + p.q_off + h * hd;
    // Row of `kt` this head's first component lives on; +1 row per d.
    var k_row = h * hd * Tk + j;
    var s = 0.0;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        s = s + q[q_base + d] * kt[k_row];
        k_row = k_row + Tk;
    }
    scores[gidx] = s * inverseSqrt(f32(hd));
}
