// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Fold `matmul_dw_reg_splitk`'s per-slice partials into the weight gradient
// @how   one thread per output element, serial inner reduction
// @opt   5
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Fold `matmul_dw_reg_splitk`'s per-slice partials into the weight gradient:
//   dW[i] += sum_{s} partial[s * rc + i]
//
// ACCUMULATES, because that is the contract every parameter gradient in
// `blocks::grad` follows (a weight used twice gets two contributions, and the
// caller clears once per step). The split-K GEMM itself ASSIGNS into its
// slices, so the two together reproduce `matmul_dw_reg`'s accumulate exactly.
//
// One invocation per output element, walking the slice axis with stride `rc`.
// Barrier-free, so `backend-cpu` compiles it — but only since it started
// taking `num_workgroups`. Without that arg the CPU JIT
// refuses the kernel outright (`kernel missing num_workgroups arg`), so the
// claim in this comment was false for as long as it was written, and
// `crates/wgsl-cpu/tests/compile_all.rs` was red on exactly this one kernel.
//
// The same omission was a silent CORRECTNESS bug on the GPU. `gid.x` alone is
// only the flat invocation id while the grid is 1D; past
// `backend_api::MAX_GROUPS_PER_DIM` (65535) the backends tile the dispatch into
// 2D, and this kernel then folded slices for the first 65535*64 = 4194240
// outputs and left the rest of `dw` untouched — a wrong weight gradient with no
// crash. `rc = Cout*Cin*K*K` passes that at, e.g., Cout 1024 with a 3x3 over
// 1024 channels (9.4M).
//
// params: [rc, slices]  where rc = n * k
//
// @workgroup_size(64).

// `acc`: 1 accumulates into `dw` (the parameter-gradient contract every
// adjoint in `blocks::grad` follows), 0 ASSIGNS (a forward split-K GEMM owns
// its output outright and has nothing to accumulate into). Same flag and same
// reason as `matmul_dx_reg`'s.
struct Params { rc: u32, slices: u32, acc: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       partial: array<f32>;
@group(0) @binding(2) var<storage, read_write> dw:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.rc) {
        return;
    }
    var acc_v = 0.0;
    for (var s = 0u; s < p.slices; s = s + 1u) {
        acc_v = acc_v + partial[s * p.rc + i];
    }
    dw[i] = select(acc_v, dw[i] + acc_v, p.acc == 1u);
}
