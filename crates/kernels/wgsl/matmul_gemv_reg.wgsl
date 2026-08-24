// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Skinny-M matmul (out = x @ W^T), one WORKGROUP per output COLUMN, REGISTER accumulators - the GPU decode-regime GEMM
// @how   register block per thread, 64-thread workgroup tile, 1 barrier
// @opt   5
// @cpu   no
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// The GPU-only sibling of `matmul_gemv.wgsl`. **Edit the two together**: the
// contract, the k-stride order and the 64-partial fold below are copied from
// that kernel deliberately, and `gpu_core::upgrade` substitutes this one for
// it at dispatch, so a change to either that is not mirrored in the other
// silently changes results on exactly one backend.
//
//   x  : [M, K]   W: [N, K]   out: [M, N]   (all row-major, identical
//   `Params{m, k, n}` and binding order to `matmul_gemv.wgsl`.)
//   REQUIRES m <= MREG. Dispatch: n * 64 invocations - the SAME thread count
//   `matmul_gemv` takes, so a caller needs no change at all.
//
// ## What differs, and why it is a separate file rather than a template knob
//
// `matmul_gemv` accumulates in WORKGROUP memory (`partial[m*64 + t]`), which
// costs it twice on a GPU:
//
//  1. its `partial` array is sized for the worst case (`m = 32`), so every
//     workgroup reserves 8 KB of shared memory whatever `m` is. On a GP102
//     (96 KB shared/SM, 32 blocks/SM, 2048 threads/SM) that caps residency at
//     12 workgroups = 768 of 2048 threads, ~37.5% occupancy;
//  2. its inner loop is a read-modify-WRITE per `(k, m)` into that array, so
//     each accumulator carries a serial dependency chain through shared-memory
//     latency, once per k-step.
//
// Here the accumulators are a function-local `array<f32, MREG>` whose every
// index is a COMPILE-TIME-bounded loop, so it lands in registers (checklist
// §C1) - the chain becomes register-latency, and `partial` is written ONCE per
// row at the end and sized `MREG * 64`, so the occupancy cost scales with the
// `m` actually asked for instead of the worst case. Both fixes come from the
// one `MREG` constant.
//
// That is why this cannot be a `kernels::template` variant of `matmul_gemv`:
// the templater rewrites literals, and a function-local accumulator array is a
// different BODY. And it is why this kernel is `@cpu no` while its sibling is
// `@cpu yes` - the CPU JIT rejects a function-local array in a work-group
// kernel outright ("array local in a work-group kernel is unsupported",
// `crates/wgsl-cpu/src/lib.rs`), which is exactly the constraint
// `matmul_gemv`'s header records and the reason that kernel must keep its
// workgroup accumulators. `backend-cpu` reports `workgroup_reductions: false`
// and therefore never selects either of them; `gpu_core::upgrade` only
// appends this one where `backend_api::select` heads the decode regime with
// `WorkgroupPerOutput`.
//
// ## Bit-identity with `matmul_gemv`
//
// Deliberate, and gated (`crates/gpu-core/tests/gemv_reg_upgrade.rs`): the
// k-stride is the same (`k = t; k += 64`), each output keeps its own
// accumulator, and the final fold sums the same 64 partials in the same
// ascending order. Nothing is reassociated, so the results are byte-identical
// and the gate is `assert_eq!` on the raw bits, not a tolerance.
//
// ## `MREG` and rows past `p.m`
//
// `MREG` is the `kernels::template` knob (`const MREG: u32`): one specialised
// variant per bucket, so the accumulator count is a compile-time constant.
// `m` values between two buckets use the next bucket up and simply compute
// rows they discard - those rows are pointed at row 0 of `x` (`xoff` below)
// rather than read out of bounds, because bounds behaviour past the end of a
// storage binding is a per-backend policy, not something to rely on.
//
// NOT covered: the `#w=bf16` / `#w=f16` storage tiers of `matmul_gemv`
// (`kernels::template::dtype_variant`) register under their own names and so
// are not upgraded - they still carry the 8 KB `partial`. Adding their reg
// siblings is additive, not a change to anything here.

struct Params {
    m: u32,
    k: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       w:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

/// Rows this specialisation carries in registers. `kernels::template`
/// rewrites this declaration; `gpu_core::upgrade`'s bucket ladder is what
/// picks which rewrite runs for a given `p.m`.
const MREG: u32 = 32u;

// Only the cross-thread FOLD needs shared memory now, so this is `MREG * 64`
// floats rather than the worst case - 512 B at MREG = 2, against 8 KB.
var<workgroup> partial: array<f32, MREG * 64u>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let col = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (col >= p.n || p.m == 0u) { return; }

    var acc: array<f32, MREG>;
    var xoff: array<u32, MREG>;
    for (var m = 0u; m < MREG; m = m + 1u) {
        acc[m] = 0.0;
        // Rows past `p.m` are computed and thrown away (see the header); point
        // them at row 0, which is already hot, instead of past the binding.
        xoff[m] = select(0u, m * p.k, m < p.m);
    }

    let wbase = col * p.k;
    for (var k = t; k < p.k; k = k + 64u) {
        // Hoisted to a bare identifier for `kernels::template`'s sake, exactly
        // as `matmul_gemv.wgsl` does - keep the two spellings the same.
        let wi = wbase + k;
        let wv = w[wi];
        for (var m = 0u; m < MREG; m = m + 1u) {
            acc[m] = acc[m] + x[xoff[m] + k] * wv;
        }
    }

    for (var m = 0u; m < MREG; m = m + 1u) {
        partial[m * 64u + t] = acc[m];
    }
    workgroupBarrier();
    // Threads 0..m each fold one row's 64 partials - the same ascending order
    // `matmul_gemv` folds in, which is what makes this bit-identical.
    //
    // `min(p.m, MREG)`, not `p.m`, is a memory-safety floor, not a feature:
    // `partial` is only `MREG * 64` long, so a caller (or a selector bug)
    // handing this variant more rows than it was specialised for would fold
    // PAST the workgroup array. wgpu clamps such an index; a raw SPIR-V
    // backend need not. Under the contract the two expressions are equal, so
    // this costs nothing and cannot change a correct result - it only turns a
    // would-be out-of-bounds read into missing rows.
    if (t < min(p.m, MREG)) {
        var s = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            s = s + partial[t * 64u + i];
        }
        out[t * p.n + col] = s;
    }
}
