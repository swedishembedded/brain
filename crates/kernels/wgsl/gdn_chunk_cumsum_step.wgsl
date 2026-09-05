// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gated DeltaNet's per-chunk log-decay cumsum
// @how   one thread per row, serial loop over the chunk
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The per-chunk-row cumulative sum `g_cs[row,i] = sum_{j<=i} g[row,j]`
// (`torch_chunk_gated_delta_rule`'s `g = g.cumsum(-1)`, RESET at every chunk
// boundary - `row` ranges over `bhc = B*H*n_chunks`, one row per
// (batch,head,chunk) triple). `g_cs` must already hold a copy of the raw
// per-token decay `g` (e.g. via `region_copy.wgsl`) before this call.
//
// ONE DISPATCH does the WHOLE row: each of the `bhc` threads owns one row and
// walks it with a plain serial `for` loop, no barrier -
//   for i in 1..c_len: g_cs[row,i] += g_cs[row,i-1]
// This used to be `c_len - 1` separate dispatches, one host-issued step per
// row index `i`, because the CPU (Cranelift) JIT allows at most one top-level
// `workgroupBarrier()` per kernel - but that constraint only rules out a
// workgroup-COOPERATIVE scan; it says nothing about a single thread looping
// serially with zero barriers, the exact idiom `scan_block.wgsl` already
// uses. `c_len` is only tens up to 64 for this model family, so the whole
// scan is cheap per thread - the `bhc-1..1` extra dispatches were pure
// host-submission latency for zero extra parallelism.
//
// Flat layout: `g_cs` is `[bhc, c_len]` row-major.

struct Params { bhc: u32, c_len: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> g_cs: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = gid.y * (nwg.x * 64u) + gid.x;
    if (row >= p.bhc) { return; }
    let base = row * p.c_len;
    for (var i = 1u; i < p.c_len; i = i + 1u) {
        g_cs[base + i] = g_cs[base + i] + g_cs[base + i - 1u];
    }
}
