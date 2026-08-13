// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Append a batch of new tokens' K (or V) into the paged pool - one thread per TOKEN, not per element
// @how   one thread per sequence, serial inner loop over kv_stride
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// B9: the WRITE-direction sibling of `paged_kv_append_batched.wgsl` - same
// contract (`pool[(blocks[b]*block_size+offsets[b])*kv_stride+c] = src[b*kv_stride+c]`
// for every (b,c)), but dispatched ONE THREAD PER TOKEN with a serial inner
// loop over `c`, instead of one thread per (b,c) element.
//
// This is the source `kernels::template::dtype_variant_store` templates for
// the bf16-packed `pool` tier (`model::ops::Ops::kv_append_batched`'s `BF16`
// arm) - NOT the plain element-parallel `paged_kv_append_batched.wgsl`,
// which stays untouched at its own best parallelism (`@opt 3`, one thread per
// element) since the F32 tier never needs this restructuring.
//
// WHY a second physical kernel, not a rewrite of the existing one. A
// bf16-packed pool holds TWO elements per storage word. Dispatching one
// thread per ELEMENT (the existing kernel's own design) means two DIFFERENT
// concurrent threads can target the SAME packed `u32` word for every
// adjacent pair within one token, ALWAYS - not merely the odd-`kv_stride`
// cross-token edge case `kernels::template::rewrite_packed_stores`'s own doc
// comment describes, a plain consequence of ANY 2-per-word packing dispatched
// per-element. This is a genuine data race a real GPU exposes: this repo's
// own dual-backend KV-cache test (`crates/model/tests/kv_bf16_roundtrip.rs`)
// caught EXACTLY this - green on the CPU JIT's serial execution (which runs
// a dispatch's threads one after another, so whichever "wins" the race is
// deterministic there), red on real wgpu hardware (genuine parallel
// execution exposes the race immediately). Dispatching one thread PER TOKEN
// instead makes every packed word within that token's own write private to
// ONE thread - no two concurrent threads from the SAME dispatch ever target
// the same word for a realistic (even) `kv_stride`; only a genuinely SHARED
// word between TWO DIFFERENT tokens in the SAME batched call (only possible
// with an odd `kv_stride`) can still race, exactly as documented by
// `rewrite_packed_stores`'s own doc comment, and avoided by using separate
// sequential dispatches for that case (a real decode loop already appends
// one token per step, never multiple tokens of ONE sequence in one batched
// call).
//
// Same physical-kernel-per-dtype precedent B4 already established for
// `RegisterTiled` (`matmul_reg3` vs `matmul_reg2` - see `model::ops::kname::
// MATMUL_REG3_BF16`'s own doc comment): the same logical operation, two
// physically different kernel files, dispatched with different thread counts
// per tier by `Ops::kv_append_batched`.
struct Params { batch: u32, kv_stride: u32, block_size: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src:     array<f32>;
@group(0) @binding(2) var<storage, read>       blocks:  array<u32>;
@group(0) @binding(3) var<storage, read>       offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> pool:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let b = gid.y * (nwg.x * 64u) + gid.x;
    if (b >= p.batch) { return; }
    let base = (blocks[b] * p.block_size + offsets[b]) * p.kv_stride;
    let sbase = b * p.kv_stride;
    for (var c: u32 = 0u; c < p.kv_stride; c = c + 1u) {
        // Hoisted to a bare identifier (B9), same reason
        // `paged_kv_append_batched.wgsl`'s own `wi` hoist exists: the
        // write-direction rewrite reads this index twice.
        let wi = base + c;
        pool[wi] = src[sbase + c];
    }
}
