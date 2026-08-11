// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  One sequential step of Gated DeltaNet's per-chunk log-decay cumsum
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// One step of the per-chunk-row cumulative sum
// `g_cs[row,i] = sum_{j<=i} g[row,j]` (`torch_chunk_gated_delta_rule`'s
// `g = g.cumsum(-1)`, RESET at every chunk boundary — `row` ranges over
// `bhc = B*H*n_chunks`, one row per (batch,head,chunk) triple). `g_cs` must
// already hold a copy of the raw per-token decay `g` (e.g. via
// `region_copy.wgsl`) before the first call.
//
// The host issues ONE DISPATCH PER ROW INDEX `i`, `i` from 1 to `c_len - 1`,
// each updating EVERY row in parallel:
//   g_cs[row,i] += g_cs[row,i-1]
// The CPU (Cranelift) JIT allows exactly one top-level `workgroupBarrier()`
// per kernel, so a true parallel-scan
// reduction cannot fit in one kernel here; `c_len` is only tens up to ~64 for
// this model family, so a plain O(c_len) sequential-DISPATCH scan — the same
// "host-orchestrated multi-pass" idiom as `scan_block.wgsl`/`scan_add.wgsl` —
// is the correct choice, not a missed optimisation.
//
// Flat layout: `g_cs` is `[bhc, c_len]` row-major.

struct Params { bhc: u32, c_len: u32, i: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> g_cs: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = gid.y * (nwg.x * 64u) + gid.x;
    if (row >= p.bhc) { return; }
    let base = row * p.c_len;
    g_cs[base + p.i] = g_cs[base + p.i] + g_cs[base + p.i - 1u];
}
