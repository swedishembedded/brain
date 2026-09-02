// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gather whole paged-KV blocks out of a pool into one compact staging buffer
// @how   one thread per output word
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype n/a
//
// The read half of host-RAM KV offload (`model::kv_offload`): a demoted
// sequence's physical blocks are scattered all over the pool, and a readback
// per block would be one PCIe round trip per block. This gathers an arbitrary
// SET of blocks into one contiguous staging region so the whole swap-out is a
// single readback.
//
//   pool  : the engine's KV pool for one layer and one of K/V (or its int8
//           scales) - `[num_blocks * words_per_block]`
//   ids   : `[blocks]` physical block ids, in the order the caller wants them
//   out   : staging, `blocks` records of `dst_stride` words
//
// Deliberately typed `u32`, not `f32`: a swap must round-trip the pool's BYTES
// whatever they encode (fp32, packed int8, bf16 pairs, a dequant scale), and a
// float-typed copy is free to quiet a NaN payload or flush a denormal.
//
// `dst_off`/`dst_stride` let one submit gather every layer's K and V (and, on
// an int8 pool, their dequant scales) into ONE staging buffer laid out
// BLOCK-major: block `b`'s whole record is `out[b*dst_stride .. +dst_stride]`,
// each tensor at its own fixed `dst_off` within it. So a whole sequence costs
// one readback instead of `2 * n_layers` of them, AND the host bytes are laid
// out per block - independent of how the caller chunked the transfer, which is
// what makes a swap-out and a later swap-in agree without either having to
// remember the chunk size.
//
// One invocation per output word.

struct Params {
    /// Blocks being gathered.
    blocks: u32,
    /// Pool words one block occupies (`block_size * kv_stride`, or the packed
    /// int8 / scale equivalent - the caller's business, not this kernel's).
    words_per_block: u32,
    /// This tensor's word offset within one block's staging record.
    dst_off: u32,
    /// Words one block's staging record spans (every tensor's share summed).
    dst_stride: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       ids: array<u32>;
@group(0) @binding(2) var<storage, read>       pool: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.blocks * p.words_per_block;
    if (idx >= total) { return; }
    let b = idx / p.words_per_block;
    let w = idx - b * p.words_per_block;
    out[b * p.dst_stride + p.dst_off + w] = pool[ids[b] * p.words_per_block + w];
}
