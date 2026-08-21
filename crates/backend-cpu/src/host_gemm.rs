// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A **blocked** host GEMM for `out[m,n] = Σ_k x[m,k]·w[n,k] (+ bias[n])`,
//! bit-identical to the naive row-parallel loop it replaces.
//!
//! Swedish Embedded AB implements cache-blocked numerical kernels for
//! embedded and edge inference. If your team needs a host matmul that is both
//! memory-efficient and bit-reproducible against the reference it replaces,
//! you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # Why this exists, measured
//!
//! Several models in this workspace share one host `linear` whose shape is
//! `[M,K] x [N,K]ᵀ`, written as: for each output row `m`, for each output
//! column `n`, walk `k`. That loop nest reads the ENTIRE `[N,K]` weight
//! matrix once per row of `M`. An earlier pass parallelized it over `m`,
//! which hid the cost behind aggregate DRAM bandwidth but removed none of the
//! redundant reads, and recorded that as the remaining defect.
//!
//! At LTX-2.5's real 720p adaLN-single shape (`M=3520`, `K=4096`,
//! `N=36864` - a 604 MB weight matrix) that is ~2.1 TB of DRAM traffic for
//! 5.3e11 multiply-adds, and it measured 76.3 s per forward call on a
//! 48-thread Xeon E5-2690 v3 - ~81% of a real denoise step's wall clock,
//! at ~14 GB/s, i.e. squarely memory-bound rather than compute-bound.
//!
//! # What the blocking changes, and what it deliberately does not
//!
//! Two things, neither of which touches the arithmetic:
//!
//! 1. **A register block over `M`.** [`MR`] output rows are computed against
//!    ONE weight row in one pass over `k`, so that weight row is read once
//!    per `MR` output rows instead of once per output row, and it stays in L1
//!    while all `MR` dot products consume it.
//! 2. **A cache block over `M`.** Threads take tiles of `m` rows rather than
//!    single rows, sized so the tile's slice of `x` stays in L2 while the
//!    whole weight matrix streams past it once per tile.
//!
//! The register block also removes a second, quieter defect: the naive loop's
//! inner statement is `acc += x·w` on ONE accumulator, a loop-carried
//! dependency on a ~4-cycle f32 add, so a core retires roughly one MAC every
//! four cycles no matter how much bandwidth it has. [`MR`] independent
//! accumulators give the out-of-order engine [`MR`] independent chains to
//! interleave.
//!
//! **Bit-identity is structural, not a tolerance.** Every output element is
//! still `bias` then `+= x[m,k]·w[n,k]` for `k` ascending, one f32 add at a
//! time, in that exact order. Nothing is reassociated, no accumulator is
//! split and recombined, no FMA replaces a separate multiply and add (which
//! would round once instead of twice), and no SIMD lane sums a partial range
//! of `k`. The blocking changes only WHICH element is computed WHEN, which
//! IEEE-754 does not observe. That claim is gated rather than asserted:
//! `tests/host_gemm.rs` compares against a transcription of the naive loop
//! with `assert_eq!` on the bit patterns, over shapes including the real one.
//!
//! What is NOT done here, and why: the obvious next step is to vectorize
//! ACROSS `M` (eight output rows in one AVX2 lane group, each lane still
//! summing its own `k` sequentially - which would also be bit-identical,
//! unlike vectorizing across `k`). That needs a transposed pack of `x` and is
//! a larger change; this module leaves the arithmetic scalar and takes the
//! memory win first. [`crate::fast_ops::matmul_abt`] is the AVX2 kernel for
//! this same shape and is several times faster still, but it is NOT
//! bit-identical to the naive loop (it splits `k` across eight lanes and uses
//! FMA), so it cannot be dropped into a path whose gate is exact agreement.

/// Output rows computed against one weight row per pass over `k`.
///
/// Eight, from the sweep in `tests/host_gemm.rs`: it is enough independent
/// accumulator chains to cover the f32 add latency, and the eight `x` rows it
/// keeps live (8·K·4 bytes = 128 KB at K=4096) still sit inside one core's
/// 256 KB L2 alongside the weight row streaming past.
pub const MR: usize = 8;

/// The default `m`-tile: how many output rows one thread's tile covers.
///
/// **One register block**, from the sweep in `tests/host_gemm.rs` at the real
/// `[3520,4096]x[36864,4096]ᵀ` shape on a 48-thread Xeon E5-2690 v3 - measured,
/// not guessed, and the measurement pointed the opposite way from the obvious
/// intuition. Bigger tiles do stream the weight matrix fewer times in total,
/// but they lose more than they gain here:
///
/// | tile | secs | GFLOP/s | vs the naive loop |
/// |---|---:|---:|---:|
/// | naive (row-parallel) | 14.39 | 73.9 | 1.00x |
/// | **8** | **8.32** | **127.8** | **1.73x** |
/// | 16 | 8.83 | 120.4 | 1.63x |
/// | 32 | 9.32 | 114.1 | 1.54x |
/// | 64 | 11.44 | 92.9 | 1.26x |
/// | 128 | 11.79 | 90.2 | 1.22x |
/// | 256 | 16.22 | 65.5 | 0.89x |
///
/// The reason the curve is monotone rather than U-shaped: at tile 8 the
/// kernel is already ARITHMETIC-bound, not bandwidth-bound (127.8 GFLOP/s is
/// ~1 scalar MAC per core-cycle, which is the ceiling for a non-reassociating
/// f32 multiply-then-add), so buying more weight reuse buys nothing, while
/// the tile's slice of `x` (`tile · K · 4` bytes - 512 KB already at tile 8
/// with K=4096) grows past this core's 256 KB L2 and starts costing. Tile 256
/// is slower than the loop it replaces.
pub fn default_m_tile() -> usize {
    MR
}

/// `out[m,n] = Σ_k x[m,k]·w[n,k] (+ b[n])`, with `w` `[out_dim, in_dim]`
/// row-major - the blocked replacement for a naive row-parallel `linear`,
/// bit-identical to it. See this module's doc.
pub fn blocked_linear(x: &[f32], rows: usize, in_dim: usize, w: &[f32], b: Option<&[f32]>, out_dim: usize) -> Vec<f32> {
    blocked_linear_tiled(x, rows, in_dim, w, b, out_dim, default_m_tile())
}

/// [`blocked_linear`] with an explicit `m`-tile - the seam the tile sweep and
/// the bit-identity gate drive, so both can cover tile sizes that do not
/// divide `rows` and tails shorter than [`MR`].
pub fn blocked_linear_tiled(x: &[f32], rows: usize, in_dim: usize, w: &[f32], b: Option<&[f32]>, out_dim: usize, m_tile: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * in_dim, "blocked_linear: x is {} floats, expected rows*in_dim = {}", x.len(), rows * in_dim);
    assert_eq!(w.len(), out_dim * in_dim, "blocked_linear: w is {} floats, expected out_dim*in_dim = {}", w.len(), out_dim * in_dim);
    let mut out = vec![0f32; rows * out_dim];
    if rows == 0 || out_dim == 0 || in_dim == 0 {
        // `in_dim == 0` is a real (degenerate) case: every output is just the
        // bias, and the loop below would never write it.
        if in_dim == 0 {
            if let Some(b) = b {
                for r in 0..rows {
                    out[r * out_dim..(r + 1) * out_dim].copy_from_slice(&b[..out_dim]);
                }
            }
        }
        return out;
    }
    let m_tile = m_tile.max(1);
    crate::par::chunks_mut(&mut out, m_tile * out_dim, |ti, tile| {
        let r0 = ti * m_tile;
        let tile_rows = tile.len() / out_dim;
        for j in 0..out_dim {
            let wr = &w[j * in_dim..j * in_dim + in_dim];
            let bias = b.map_or(0.0, |b| b[j]);
            let mut mi = 0;
            // The register-blocked body: MR output rows share one pass over
            // this weight row, each on its own accumulator chain.
            while mi + MR <= tile_rows {
                let base = (r0 + mi) * in_dim;
                // Hoisted as MR fixed-length subslices rather than indexed
                // out of `x` inside the loop: the inner statement then
                // indexes a slice whose length the optimizer can match
                // against `wr`'s, instead of re-proving a computed offset
                // against the whole 57 MB activation buffer MR times per `k`.
                let xs: [&[f32]; MR] = std::array::from_fn(|u| &x[base + u * in_dim..base + (u + 1) * in_dim]);
                let mut acc = [bias; MR];
                for (kk, &wv) in wr.iter().enumerate() {
                    for (u, a) in acc.iter_mut().enumerate() {
                        *a += xs[u][kk] * wv;
                    }
                }
                for (u, &a) in acc.iter().enumerate() {
                    tile[(mi + u) * out_dim + j] = a;
                }
                mi += MR;
            }
            // The tail: fewer than MR rows left in this tile. Same
            // accumulation, one row at a time.
            while mi < tile_rows {
                let xr = &x[(r0 + mi) * in_dim..(r0 + mi) * in_dim + in_dim];
                let mut a = bias;
                for (xi, wi) in xr.iter().zip(wr) {
                    a += xi * wi;
                }
                tile[mi * out_dim + j] = a;
                mi += 1;
            }
        }
    });
    out
}

/// The naive row-parallel loop this module replaces, kept as the REFERENCE
/// definition of the arithmetic: `blocked_linear` must agree with it bit for
/// bit, and a gate that transcribed the formula independently could drift
/// from what callers used to run.
///
/// Public because the gate lives in `tests/`, and because a caller migrating
/// to [`blocked_linear`] can A/B against exactly the code it had.
pub fn naive_linear(x: &[f32], rows: usize, in_dim: usize, w: &[f32], b: Option<&[f32]>, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * out_dim];
    crate::par::rows_mut(&mut out, out_dim, |r, orow| {
        let xr = &x[r * in_dim..r * in_dim + in_dim];
        for (o, slot) in orow.iter_mut().enumerate() {
            let wr = &w[o * in_dim..o * in_dim + in_dim];
            let mut acc = b.map_or(0.0, |b| b[o]);
            for (xi, wi) in xr.iter().zip(wr) {
                acc += xi * wi;
            }
            *slot = acc;
        }
    });
    out
}
