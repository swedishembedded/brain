// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Fold `matmul_dw_reg_splitk`'s per-slice partials into the weight gradient:
//   dW[i] += sum_{s} partial[s * rc + i]
//
// ACCUMULATES, because that is the contract every parameter gradient in
// `blocks::grad` follows (a weight used twice gets two contributions, and the
// caller clears once per step). The split-K GEMM itself ASSIGNS into its
// slices, so the two together reproduce `matmul_dw_reg`'s accumulate exactly.
//
// One invocation per output element, walking the slice axis with stride `rc`.
// Barrier-free, so `backend-cpu` compiles it (docs/lessons.md #26).
//
// params: [rc, slices]  where rc = n * k
//
// @workgroup_size(64).

struct Params { rc: u32, slices: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       partial: array<f32>;
@group(0) @binding(2) var<storage, read_write> dw:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.rc) {
        return;
    }
    var acc = 0.0;
    for (var s = 0u; s < p.slices; s = s + 1u) {
        acc = acc + partial[s * p.rc + i];
    }
    dw[i] = dw[i] + acc;
}
