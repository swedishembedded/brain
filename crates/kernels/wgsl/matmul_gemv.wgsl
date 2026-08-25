// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Skinny-M matmul (out = x @ W^T), one WORKGROUP per output COLUMN - the decode-regime GEMM
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32|bf16|f16
// @tpl   w -> bf16 storage variant (kernels::template::dtype_variant, B4;
//        header field, parsing deferred to B6)
//
// Skinny-M matmul (out = x @ W^T), one WORKGROUP per output COLUMN — the
// decode-regime GEMM.
//
//   x  : [M, K]   W: [N, K]   out: [M, N]   (all row-major, same contract as
//   matmul.wgsl); params: m, k, n. REQUIRES m <= 32.
//
// At decode, M is the number of concurrent sequences (1-32). The naive kernel
// (one thread per output element) then re-streams weight row `n` once per m —
// M-fold redundant traffic on a kernel that is memory-bound on W (measured as
// two thirds of decode time, at a low single-digit percent of the card's
// bandwidth roof). Here the 64 threads of
// a workgroup split K, each reading its slice of W row `n` ONCE and applying
// it to all M rows of x from registers; one barrier; then threads 0..m fold
// the 64 partials for their row. W traffic drops M-fold and the reads are
// K-contiguous (coalesced).
//
// Single top-level barrier + no atomics: runs on the CPU JIT and every GPU
// backend unchanged. Dispatch: n * 64 invocations (one workgroup per column).

struct Params {
    m: u32,
    k: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       w:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

// Accumulators live in workgroup memory (indexed [m*64 + t]) rather than a
// function-local array: the CPU JIT's work-group execution model does not
// support local arrays (`wgsl_cpu::Jit` returns "array local in a work-group
// kernel is unsupported" - re-verified, not inherited), and workgroup memory
// is fast enough that the extra stores are negligible next to the M-fold cut
// in W traffic.
//
// **On a GPU that is no longer true, and this kernel is no longer the one that
// runs.** Sizing `partial` for the worst case costs 8 KB of shared memory per
// workgroup at every `m` (on a GP102: 12 resident workgroups of a possible 32),
// and the read-modify-write below is a shared-memory
// dependency chain per `(k, m)`. `matmul_gemv_reg.wgsl` is the register-
// accumulator sibling that fixes both - bit-identical, same `Params`, same
// bindings, same `n * 64` thread count - and `gpu_core::upgrade` substitutes
// it at dispatch wherever `backend_api::select` heads the decode regime with
// `WorkgroupPerOutput`. **Edit the two together**; `BRAIN_NO_KERNEL_UPGRADE=1`
// is what pins a GPU back onto this one.
//
// So THIS kernel keeps the 2048-float array on purpose: it is what the CPU JIT
// and the `@npu` path can execute, and neither pays a shared-memory occupancy
// cost. Shrinking it here (`kernels::template`) was measured and is NOT what
// ships - it recovers most, but not all, of what the register sibling gets,
// and only on the backend that cannot use it.
var<workgroup> partial: array<f32, 2048>; // up to 32 rows x 64 threads

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let col = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (col >= p.n) { return; }
    for (var m = 0u; m < p.m; m = m + 1u) {
        partial[m * 64u + t] = 0.0;
    }
    let wbase = col * p.k;
    for (var k = t; k < p.k; k = k + 64u) {
        // Hoisted to a bare identifier (B4) -- see matmul.wgsl's comment on
        // the same pattern: `dtype_variant`'s bf16 decode reads `wi` twice.
        let wi = wbase + k;
        let wv = w[wi];
        for (var m = 0u; m < p.m; m = m + 1u) {
            partial[m * 64u + t] = partial[m * 64u + t] + x[m * p.k + k] * wv;
        }
    }
    workgroupBarrier();
    // Threads 0..m each fold one row's 64 partials.
    if (t < p.m) {
        var s = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            s = s + partial[t * 64u + i];
        }
        out[t * p.n + col] = s;
    }
}
