// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse-MoE expert linear at skinny M: moe_linear_gated.wgsl, one WORKGROUP per output COLUMN - the decode-regime expert GEMM
// @how   64-thread workgroup tile, 1 barrier, skips the K loop on a dead expert
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `moe_linear_gated.wgsl` is `matmul.wgsl` plus a row gate; this is
// `matmul_gemv.wgsl` plus the SAME row gate, and it exists for the same reason
// `matmul_gemv` exists next to `matmul`. **Edit it and `moe_linear_gated.wgsl`
// together**: the gate semantics below (a row whose gate weight for this
// expert is <= 0 writes 0 and is never reduced, and down_proj re-reads the
// SAME row's gate rather than inspecting its input) are that kernel's
// contract, copied deliberately, and a change to one that is not mirrored in
// the other silently changes results at exactly one M.
//
// ## Why the element-per-thread kernel is the wrong shape at decode
//
// `moe_linear_gated` gives one thread each output ELEMENT, so a single decode
// row dispatches `m * n` = `n` threads - 896 for this MoE's gate/up
// projections. A Tesla P40 has 30 SMs and 2048 resident thread slots each:
// 896 threads is 14 warps against a capacity of 960, so nothing hides the
// DRAM latency of the `k`-long serial reduction each of them runs.
// perf-number: the shortfall this kernel exists to close, on the one card it
// was measured on - not a throughput claim for any other device or model.
// perf-number: on the DeepSeek-OCR decoder at one row, 0.91 GB of routed expert weights read at ~10 GB/s against that card's ~346 GB/s roof, 51% of the whole decode step's GPU kernel time.
//
// Here the 64 threads of a workgroup split K, each reading its slice of weight
// row `n` ONCE and applying it to all M rows of x from workgroup memory; one
// barrier; then threads 0..m fold the 64 partials for their row. The dispatch
// becomes `n * 64` invocations for the same work, and the reads are
// K-contiguous (coalesced) instead of K-strided.
//
// REQUIRES m <= 32 (the `partial` bound below), which is what
// `backend_api::select`'s `DECODE_REGIME_MAX_ROWS` already gates every
// `WorkgroupPerOutput` GEMM selection on - a prefill round wider than that
// takes `moe_linear_gated` and its element-per-thread grid, which at those M
// is the right shape.
//
// Single top-level barrier + no atomics, so the CPU JIT can compile it; it is
// never selected there, because `backend-cpu` reports
// `workgroup_reductions: false` and the selection is keyed on that.
//
//   x    : [m, k]           row-major
//   w    : [n, k]           row-major (w[j, l] is weight row j)
//   gate : [m, n_experts]   dense per-token-per-expert weight (0 = not routed)
//   out  : [m, n]           row-major; out[row, :] = 0 for a non-routed row
//
// Same bindings, same `Params` and the same `[m, k, n, n_experts, e_idx]`
// parameter order as `moe_linear_gated.wgsl`, so a caller swaps the kernel
// index and the thread count and nothing else.

struct Params {
    m: u32,
    k: u32,
    n: u32,
    n_experts: u32,
    e_idx: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read>       w:    array<f32>;
@group(0) @binding(3) var<storage, read>       gate: array<f32>;
@group(0) @binding(4) var<storage, read_write> out:  array<f32>;

// Workgroup rather than function-local, for the reason `matmul_gemv.wgsl`'s
// own comment records: `wgsl_cpu::Jit` rejects a function-local array in a
// work-group kernel outright, and this kernel is `@cpu yes`. Sized for the
// worst legal `m` (32 rows x 64 threads).
var<workgroup> partial: array<f32, 2048>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let col = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (col >= p.n) { return; }

    // Whether ANY row routed to this expert, decided BEFORE the k-loop.
    //
    // This is the whole sparsity of the sparse MoE and it has to be here, not
    // inside the loop. `w` is read once per k-step and shared by every row, so
    // a k-loop entered on a dead expert streams that expert's entire weight
    // matrix to discard it - at decode 58 of this model's 64 experts are dead
    // per token, which turns a 0.91 GB read into a 9.7 GB one.
    // perf-number: this ordering is load-bearing and a future edit that hoists
    // the load back out of the test would look harmless, so the measurement
    // that establishes it stays. A version of this kernel that loaded `w`
    // before testing the gate cost 41.9 us per call against
    // `moe_linear_gated`'s 16.8 us - it LOST to the kernel it replaces,
    // entirely on that traffic.
    //
    // `p.m <= 32` reads at most 32 dwords here, all of which the whole
    // workgroup hits in cache after the first.
    var any_routed = false;
    for (var m = 0u; m < p.m; m = m + 1u) {
        if (gate[m * p.n_experts + p.e_idx] > 0.0) {
            any_routed = true;
        }
    }

    // A dead expert still has to WRITE its zeros: `moe_linear_gated.wgsl`'s
    // contract is that a non-routed row's output slot is `0`, and `scale_add`
    // reads it unconditionally. So it zeroes `partial` and folds 64 zeros to
    // exactly `0.0` below like everyone else - it just never touches `w`.
    //
    // The k-loop is SKIPPED rather than returned from, and the barrier stays
    // at the top level. `any_routed` is dynamically uniform across the
    // workgroup (every thread reads the same gate entries), but WGSL's
    // uniformity analysis is static: a `workgroupBarrier()` reached under a
    // branch on a storage-derived value is a compile error, not a fast path.
    for (var m = 0u; m < p.m; m = m + 1u) {
        partial[m * 64u + t] = 0.0;
    }

    if (any_routed) {
        let wbase = col * p.k;
        for (var k = t; k < p.k; k = k + 64u) {
            let wv = w[wbase + k];
            for (var m = 0u; m < p.m; m = m + 1u) {
                // Still per ROW inside the loop, because a live expert can
                // have a mix of routed and non-routed rows and they share this
                // column's one `w` read - which is exactly the traffic the
                // outer test above protects, and which is already paid once
                // the expert is live.
                if (gate[m * p.n_experts + p.e_idx] > 0.0) {
                    partial[m * 64u + t] = partial[m * 64u + t] + x[m * p.k + k] * wv;
                }
            }
        }
    }
    workgroupBarrier();

    // Threads 0..m each fold one row's 64 partials, in the SAME ascending
    // order matmul_gemv.wgsl folds them. A non-routed row folded 64 zeros and
    // therefore writes exactly 0.0, which is what moe_linear_gated.wgsl's
    // early-exit path writes.
    if (t < p.m) {
        var s = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            s = s + partial[t * 64u + i];
        }
        out[t * p.n + col] = s;
    }
}
