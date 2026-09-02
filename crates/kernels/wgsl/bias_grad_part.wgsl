// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Bias gradient, STAGE 1 of 2 - partial column sums over row chunks
// @how   one thread per partial, strided serial reduction (no barrier)
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Bias gradient, STAGE 1 of 2 - partial column sums over row chunks.
//
// `bias_grad` is one invocation per output feature `n`, walking all `m` rows
// serially: at `m` in the tens of thousands (a real conv layer's spatial
// extent x batch) that dispatches only `n` threads total - a couple of
// workgroups against a card with dozens of SMs - and was measured at 1.3% of
// the memory roof on a VQGAN training step (kernel-performance.md M5.7),
// squarely the occupancy pathology C.3 names for grad-norm.
//
// Splits the `m`-row reduction across `P` chunks per column instead:
//   part[chunk * n + col] = sum over { mm : mm ≡ chunk (mod P), mm < m } of dy[mm,n+col]
// `P * ceil(n/64)` workgroups instead of `ceil(n/64)` - `P` more parallelism at
// the SAME total serial work, folded by `bias_grad_final`.
//
// Coalescing note: `col = gidx % n`, not `gidx / P` - `dy` is ROW-major
// (`dy[m,n]`, column fastest), so adjacent COLUMNS are adjacent addresses at a
// fixed row. Adjacent THREADS therefore read adjacent addresses at every step
// (the same property the naive `bias_grad` already had); only the number of
// independent chunks is new. Mirroring `gn_dsum_part`'s own `gidx / P` split
// verbatim would be wrong here - GroupNorm's data is channel-major, the
// opposite convention, so that indexing coalesces THERE for a different
// reason than the one this kernel needs.
//
// BARRIER-FREE by construction (no `workgroupBarrier`), so `backend-cpu` JITs
// it with no capability branch needed - the same shape `gn_dsum_part` and
// `gn_dgb_part` already established for this class of fix.
//
// Determinism: each partial sums a FIXED strided subset in ascending row
// order, and `bias_grad_final` folds them in ascending chunk order, so the
// result is reproducible run to run. It is NOT bit-identical to the naive
// kernel's single ascending-`m` pass (a different association order is a
// different fp32 rounding) - the same trade `gn_dsum_part`/`gn_part` already
// make.

struct Params {
    m: u32, // rows
    n: u32, // columns (the bias width)
    P: u32, // chunks per column
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:   array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.n * p.P;
    if (gidx >= total) { return; }

    let col = gidx % p.n;
    let chunk = gidx / p.n;

    var acc = 0.0;
    for (var mm = chunk; mm < p.m; mm = mm + p.P) {
        acc = acc + dy[mm * p.n + col];
    }
    part[chunk * p.n + col] = acc;
}
