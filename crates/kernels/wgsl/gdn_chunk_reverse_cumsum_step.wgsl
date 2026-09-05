// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gated DeltaNet's per-chunk cumsum BACKWARD (suffix sum)
// @how   one thread per row, serial loop over the chunk
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Backward of `gdn_chunk_cumsum_step.wgsl`'s per-chunk-row cumulative sum
// `g_cs[row,i] = sum_{j<=i} g[row,j]`. The adjoint of a prefix sum is a
// SUFFIX sum: `d_raw_g[row,i] = sum_{k>=i} d_g_cs[row,k]`. `d_raw_g` must
// already hold a copy of the complete `d_g_cs` (e.g. via `region_copy.wgsl`,
// mirroring forward's own `g_cs = copy(raw_g)` priming step) before this
// call.
//
// ONE DISPATCH does the WHOLE row: each of the `bhc` threads owns one row and
// walks it, DOWNWARD from `c_len - 2` to `0`, with a plain serial `for` loop,
// no barrier -
//   for i in (c_len-2)..=0 (descending): d_raw_g[row,i] += d_raw_g[row,i+1]
// This used to be `c_len - 1` separate dispatches, one host-issued step per
// row index `i` counting down - same JIT constraint as the forward step (the
// CPU backend allows exactly one top-level `workgroupBarrier()` per kernel,
// which only rules out a workgroup-COOPERATIVE scan, not a single thread
// looping serially with zero barriers) - `c_len` is only tens up to 64 for
// this model family, so the whole reverse scan is cheap per thread.
//
// Flat layout: `d_raw_g` is `[bhc, c_len]` row-major, matching `g_cs`'s own.

struct Params { bhc: u32, c_len: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> d_raw_g: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = gid.y * (nwg.x * 64u) + gid.x;
    if (row >= p.bhc) { return; }
    let base = row * p.c_len;
    for (var step = 1u; step < p.c_len; step = step + 1u) {
        let i = p.c_len - 1u - step;
        d_raw_g[base + i] = d_raw_g[base + i] + d_raw_g[base + i + 1u];
    }
}
