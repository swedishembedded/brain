// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  One sequential step of Gated DeltaNet's per-chunk cumsum BACKWARD (suffix sum)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Backward of `gdn_chunk_cumsum_step.wgsl`'s per-chunk-row cumulative sum
// `g_cs[row,i] = sum_{j<=i} g[row,j]`. The adjoint of a prefix sum is a
// SUFFIX sum: `d_raw_g[row,i] = sum_{k>=i} d_g_cs[row,k]`. `d_raw_g` must
// already hold a copy of the complete `d_g_cs` (e.g. via `region_copy.wgsl`,
// mirroring forward's own `g_cs = copy(raw_g)` priming step) before the first
// call.
//
// The host issues ONE DISPATCH PER ROW INDEX `i`, `i` from `c_len - 2` DOWN TO
// `0` (the reverse of forward's `1` UP TO `c_len - 1`), each updating EVERY
// row in parallel:
//   d_raw_g[row,i] += d_raw_g[row,i+1]
// Same JIT constraint as the forward step (the CPU backend allows exactly one
// top-level `workgroupBarrier()` per kernel, so a
// true parallel reverse-scan cannot fit in one kernel here) — `c_len` is only
// tens for this model family, so the same O(c_len) sequential-dispatch idiom
// applies.
//
// Flat layout: `d_raw_g` is `[bhc, c_len]` row-major, matching `g_cs`'s own.

struct Params { bhc: u32, c_len: u32, i: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> d_raw_g: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = gid.y * (nwg.x * 64u) + gid.x;
    if (row >= p.bhc) { return; }
    let base = row * p.c_len;
    d_raw_g[base + p.i] = d_raw_g[base + p.i] + d_raw_g[base + p.i + 1u];
}
