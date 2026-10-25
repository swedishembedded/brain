// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-(row, 32-group) max/x/ -> int8 activation scale, one WORKGROUP per row - the cooperative form of `max_abs_row.wgsl`
// @how   64-thread workgroup, no barrier
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// `max_abs_row.wgsl`'s scales, one WORKGROUP per row:
//
//   x : [m, k]    sx: [m, k/32]    params: m, k (k a multiple of 32)
//   sx[row, g] = max(max|x[row, 32g .. 32g+32]|, 1e-8) / 127
//
// `max_abs_row` gives thread `t` row `t` and walks the whole row from that one
// invocation, a serial chain of `k` dependent loads per thread. Here lane `t`
// of a row's workgroup takes groups `t, t + 64, ...`, each a contiguous 128
// bytes, so a row's groups reduce in parallel and need no exchange between
// lanes. Dispatch: m * 64 invocations; `gpu_core::Gpu` swaps this kernel in
// for `max_abs_row` and scales the thread count (the kernel-upgrade table and
// `backend_api::select::Op::MaxAbsRow`), so a model inherits it without
// touching its dispatch sites.
//
// NUMERICS: bit-identical to `max_abs_row`: `max` is exact, and each group is
// folded in the same order.

struct Params { m: u32, k: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read_write> sx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear workgroup index (identity for a 1D dispatch).
    let row = wg.y * nwg.x + wg.x;
    if (row >= p.m) { return; }
    let groups = p.k / 32u;
    for (var g = li.x; g < groups; g = g + 64u) {
        let base = row * p.k + g * 32u;
        var a = 0.0;
        for (var c = 0u; c < 32u; c = c + 1u) {
            a = max(a, abs(x[base + c]));
        }
        // Guard the all-zero group so the scale is finite (dequant multiplies by it).
        sx[row * groups + g] = max(a, 1e-8) / 127.0;
    }
}
