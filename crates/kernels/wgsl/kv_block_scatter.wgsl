// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Scatter compactly staged KV blocks back into their physical pool slots
// @how   one thread per input word
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype n/a
//
// The write half of host-RAM KV offload (`model::kv_offload`), exactly
// inverse to `kv_block_gather.wgsl`: a promoted sequence's bytes arrive in one
// contiguous upload and are placed into whatever physical blocks the allocator
// handed out this time (which are NOT the ones it was demoted from - a paged
// cache is addressed through its block table).
//
//   src   : staging, `blocks` records of `src_stride` words
//   ids   : `[blocks]` destination physical block ids, in `src`'s block order
//   pool  : the engine's KV pool for one layer and one of K/V (or its scales)
//
// `src_off`/`src_stride` address the same BLOCK-major staging layout the
// gather writes (see its header): block `b`'s record is
// `src[b*src_stride .. +src_stride]`, this tensor at `src_off` within it.
//
// `u32`-typed for the same reason the gather is: the bytes are round-tripped,
// never interpreted.
//
// One invocation per input word.

struct Params {
    /// Blocks being scattered.
    blocks: u32,
    /// Pool words one block occupies.
    words_per_block: u32,
    /// This tensor's word offset within one block's staging record.
    src_off: u32,
    /// Words one block's staging record spans.
    src_stride: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       ids: array<u32>;
@group(0) @binding(2) var<storage, read>       src: array<u32>;
@group(0) @binding(3) var<storage, read_write> pool: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.blocks * p.words_per_block;
    if (idx >= total) { return; }
    let b = idx / p.words_per_block;
    let w = idx - b * p.words_per_block;
    pool[ids[b] * p.words_per_block + w] = src[b * p.src_stride + p.src_off + w];
}
