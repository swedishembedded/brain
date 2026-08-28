// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Skinny-M INT8 matmul (out = dequant(x_q @ W_qᵀ)), one WORKGROUP per output COLUMN, REGISTER accumulators - the GPU decode-regime int8 GEMM
// @how   DP4A packed int8, register block per thread, 64-thread workgroup tile, 1 barrier
// @opt   5
// @cpu   no
// @gpu   yes
// @npu   yes
// @quant int8
// @dtype f32
// @tpl   MREG -> rows carried in registers (kernels::template::interned)
//
// The GPU-only sibling of `matmul_i8_gemv.wgsl`, standing to it exactly as
// `matmul_gemv_reg` stands to `matmul_gemv`. **Edit the two together**: the
// contract, the k-stride order and the 64-partial fold below are copied from
// that kernel deliberately, and `gpu_core::upgrade` substitutes this one for
// it at dispatch, so a change to either that is not mirrored in the other
// silently changes results on exactly one backend.
//
//   x_q : [M, K/4] u32   w_q : [N, K/4] u32   sx: [M]   sw: [N]
//   out : [M, N] f32     params: m, kg (=K/4), n. REQUIRES m <= MREG.
//   Dispatch: n * 64 invocations - the SAME thread count `matmul_i8_gemv`
//   takes, so a caller needs no change at all.
//
// ## Why this exists
//
// The int8 decode GEMV was left carrying the very cost its fp32 twin had
// already been fixed for. `matmul_i8_gemv` accumulates in WORKGROUP memory
// (`partial[m * 64 + t]`, `array<i32, 2048>`), which costs it twice on a GPU:
//
//  1. the array is sized for the worst case (`m = 32`), so every workgroup
//     reserves 8 KB of shared memory whatever `m` is. On a GP102 (96 KB
//     shared/SM, 32 blocks/SM, 2048 threads/SM) that caps residency at 12
//     workgroups = 768 of 2048 threads, barely a third of them;
//  2. its inner loop is a read-modify-WRITE per `(k, m)` into that array, so
//     each accumulator carries a serial dependency chain through
//     shared-memory latency, once per k-step.
//
// Both fixes come from the one `MREG` constant, exactly as in the fp32 twin:
// the accumulators become a function-local `array<i32, MREG>` whose every
// index is a compile-time-bounded loop, so they land in registers, and
// `partial` is written ONCE per row at the end and sized `MREG * 64`.
//
// Measured on a Tesla P40 at the Qwen3-VL-4B decode shape, the workgroup
// version streamed its weight bytes at about half the card's DRAM roof while
// the fp32 register version reached essentially all of it - so int8's four-fold
// smaller weights were buying a little over two-fold in time. This closes that.
//
// `@cpu no`, like the fp32 twin and for the same reason: the CPU JIT rejects a
// function-local array in a work-group kernel outright, which is precisely the
// constraint `matmul_i8_gemv`'s header records and why THAT kernel must keep
// its workgroup accumulators. `backend-cpu` reports
// `workgroup_reductions: false` and never selects either.
//
// ## Bit-identity with `matmul_i8_gemv`
//
// Stronger here than in the fp32 pair, and by construction rather than by
// care: the accumulator is `i32` and integer addition is exact and
// associative, so no grouping of the same terms can differ. On top of that the
// k-stride is the same (`g = t; g += 64`), each output keeps its own
// accumulator, and the final fold sums the same 64 partials in the same
// ascending order. The single f32 operation is the dequantise multiply at the
// very end, applied to an identical integer. Gated on the raw bits.
//
// ## `MREG` and rows past `p.m`
//
// `MREG` is the `kernels::template` knob: one specialised variant per bucket,
// so the accumulator count is a compile-time constant. `m` values between two
// buckets use the next bucket up and simply compute rows they discard - those
// rows are pointed at row 0 of `x_q` (`xoff` below) rather than read out of
// bounds, because bounds behaviour past the end of a storage binding is a
// per-backend policy, not something to rely on.

struct Params {
    m: u32,
    kg: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       xq:  array<u32>;  // [M, kg]
@group(0) @binding(2) var<storage, read>       wq:  array<u32>;  // [N, kg]
@group(0) @binding(3) var<storage, read>       sx:  array<f32>;  // [M]
@group(0) @binding(4) var<storage, read>       sw:  array<f32>;  // [N]
@group(0) @binding(5) var<storage, read_write> out: array<f32>;  // [M, N]

/// Rows this specialisation carries in registers. `kernels::template`
/// rewrites this declaration; `gpu_core::upgrade`'s bucket ladder is what
/// picks which rewrite runs for a given `p.m`.
const MREG: u32 = 32u;

// Only the cross-thread FOLD needs shared memory now, so this is `MREG * 64`
// i32s rather than the worst case - 512 B at MREG = 2, against 8 KB.
var<workgroup> partial: array<i32, MREG * 64u>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let col = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (col >= p.n || p.m == 0u) { return; }

    var acc: array<i32, MREG>;
    var xoff: array<u32, MREG>;
    for (var m = 0u; m < MREG; m = m + 1u) {
        acc[m] = 0;
        // Rows past `p.m` are computed and thrown away (see the header); point
        // them at row 0, which is already hot, instead of past the binding.
        xoff[m] = select(0u, m * p.kg, m < p.m);
    }

    let wbase = col * p.kg;
    for (var g = t; g < p.kg; g = g + 64u) {
        let wv = wq[wbase + g];
        for (var m = 0u; m < MREG; m = m + 1u) {
            acc[m] = acc[m] + dot4I8Packed(xq[xoff[m] + g], wv);
        }
    }

    for (var m = 0u; m < MREG; m = m + 1u) {
        partial[m * 64u + t] = acc[m];
    }
    workgroupBarrier();
    // Threads 0..m each fold one row's 64 partials and dequantize - the same
    // ascending order `matmul_i8_gemv` folds in.
    //
    // `min(p.m, MREG)`, not `p.m`, is a memory-safety floor, not a feature:
    // `partial` is only `MREG * 64` long, so a caller (or a selector bug)
    // handing this variant more rows than it was specialised for would fold
    // PAST the workgroup array. Under the contract the two expressions are
    // equal, so this costs nothing and cannot change a correct result.
    if (t < min(p.m, MREG)) {
        var s = 0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            s = s + partial[t * 64u + i];
        }
        out[t * p.n + col] = f32(s) * sx[t] * sw[col];
    }
}
