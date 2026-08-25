// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Transpose the K region of a fused KV slab to key-minor `[d_model, T_enc]`, the layout `attn_scores_cross_kt` reads coalesced
// @how   one thread per output element, key index fastest
// @opt   2
// @cpu   native
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Transpose the K region of an encoder-memory slab into KEY-MINOR order:
//
//   kv : [T_enc, kv_stride]   fused KV, K at k_off (kv_stride = 2*d_model)
//   kt : [d_model, T_enc]     kt[c*T_enc + j] = kv[j*kv_stride + k_off + c]
//
// Why this exists: cross-attention scores parallelise over the KEY index `j`
// and reduce over `d`. In the natural `[T_enc, d_model]` layout that makes
// consecutive threads read addresses `kv_stride` floats apart - one memory
// transaction per lane. `attn_apply_cross` reads the SAME number of bytes of
// the same slab and runs several times faster purely because its thread index is `d`,
// which is contiguous. Transposing K once per block (a 512x1536 shuffle, ~3 MB)
// buys that same coalescing for the scores, whose traffic is ~44 GB per block.
//
// One invocation per (c, j) element. `j` is the fastest thread index, so the
// WRITES are fully coalesced; the reads are strided, which is the right way
// round for a kernel that runs once against a consumer that runs T_dec times.

struct Params {
    t_enc: u32,
    d_model: u32,
    kv_stride: u32,   // 2*d_model
    k_off: u32,       // 0
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       kv: array<f32>;
@group(0) @binding(2) var<storage, read_write> kt: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.d_model * p.t_enc) { return; }
    let j = idx % p.t_enc;
    let c = idx / p.t_enc;
    kt[idx] = kv[j * p.kv_stride + p.k_off + c];
}
