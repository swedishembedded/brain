// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-ROW (per-token) max/x/ -> int8 scale, one WORKGROUP per row — the cooperative form of `max_abs_row.wgsl`
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
//
// Per-ROW (per-token) max|x| -> int8 scale, one WORKGROUP per row — the
// cooperative form of `max_abs_row.wgsl`.
//
//   x : [m, k]    sx: [m]    params: m, k
//   sx[row] = max(max|x[row,:]|, 1e-8) / 127
//
// `max_abs_row` gives thread `t` row `t` and walks the whole row from that one
// invocation: a warp's 32 loads are `k` floats apart, so each 32-byte sector
// fetched serves ONE useful float, and the row itself is a serial chain of
// `k` dependent loads. That is trap C2 of `.agents/rules/kernels.md`, the same
// shape as `gn_stats` (159x), `rmsnorm` (19.4x) and the `layernorm` family
// (2.8-10x) — and it sits on the int8 dynamic-activation-quant path, so EVERY
// int8 linear in `qwen::q8`, `zimage`, and the FLUX.2 DiT pays it once per
// quantized activation (measured 43.6 ms of a 668 ms FLUX.2 int8 text-encoder
// forward, 6.5%).
//
// Here 64 threads cooperate on one row: each takes a stride-64 slice (so the
// 64 lanes read 64 consecutive words every step — every fetched sector is
// fully used), writes its partial into workgroup memory, ONE barrier, then
// lane 0 folds the 64 partials (64 `max`es — cheaper than a second barrier,
// which the CPU JIT could not compile anyway; see checklist D).
//
// Dispatch: m * 64 invocations (one workgroup per row) — 64x the reference
// kernel's `m`. Callers do not compute that themselves: `gpu_core::Gpu` swaps
// this kernel in for `max_abs_row` and scales the thread count (see
// `gpu_core`'s kernel-upgrade table + `backend_api::select::Op::MaxAbsRow`),
// so a model inherits it without touching its dispatch sites.
//
// NUMERICS: bit-identical to `max_abs_row`, not merely close. `max` is
// associative *and* exact on floats, so splitting the row across 64 lanes and
// re-folding cannot change the result — unlike a sum reduction. The int8
// activation scales, and therefore every quantized activation downstream, are
// unchanged by construction.
//
// `max_abs_part`/`max_abs_final` are NOT this kernel: they reduce a whole
// buffer to ONE scale (per-tensor quant). Per-token scales are what keep a
// deep int8 activation path accurate — a single outlier token no longer
// crushes every other token's resolution — so the two cannot be substituted.

struct Params { m: u32, k: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read_write> sx: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear workgroup index (identity for a 1D dispatch).
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    // Uniform across the workgroup, so returning before the barrier is legal.
    if (row >= p.m) { return; }
    let base = row * p.k;
    var a = 0.0;
    for (var c = t; c < p.k; c = c + 64u) {
        a = max(a, abs(x[base + c]));
    }
    partial[t] = a;
    workgroupBarrier();
    if (t == 0u) {
        var mx = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            mx = max(mx, partial[i]);
        }
        // Guard the all-zero row so the scale is finite (dequant multiplies by it).
        sx[row] = max(mx, 1e-8) / 127.0;
    }
}
