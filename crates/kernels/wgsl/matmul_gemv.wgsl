// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Skinny-M matmul (out = x @ W^T), one WORKGROUP per output COLUMN — the
// decode-regime GEMM.
//
//   x  : [M, K]   W: [N, K]   out: [M, N]   (all row-major, same contract as
//   matmul.wgsl); params: m, k, n. REQUIRES m <= 32.
//
// At decode, M is the number of concurrent sequences (1-32). The naive kernel
// (one thread per output element) then re-streams weight row `n` once per m —
// M-fold redundant traffic on a kernel that is memory-bound on W (measured:
// 67.5% of decode time at ~7 GB/s of a 346 GB/s card). Here the 64 threads of
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
// support local arrays, and workgroup memory is fast enough that the extra
// stores are negligible next to the M-fold cut in W traffic.
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
        let wv = w[wbase + k];
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
