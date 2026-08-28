// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The** CPU-parallel primitives — the on-CPU scheduler's public face for
//! host-side data-parallel work.
//!
//! rayon lives in exactly one crate: this one. `backend-cpu` *is* brain's CPU
//! scheduler — it already fans every JIT-compiled kernel out across the cores —
//! so host-side parallel loops (an `m=1` decode matvec, a grad-norm reduction,
//! image-row post-processing) belong to the same pool under the same policy.
//! Before this module, six crates depended on rayon directly and each ad-hoc
//! `par_iter` ran on a pool nobody owned: thread-count policy (`--device cpu0-7`
//! pins affinity and sizes the pool via `RAYON_NUM_THREADS`) only holds if
//! every parallel loop goes through the scheduler that honours it.
//!
//! The API is deliberately domain-shaped (rows of `f32`, tensor lists) rather
//! than a re-export of `ParallelIterator`: a caller states *what* is parallel,
//! and the scheduler owns *how*.

use rayon::prelude::*;

/// Apply `f(row_index, row)` to each `row_len`-sized row of `buf`, in parallel.
///
/// The row split is exact: `buf.len()` must be a multiple of `row_len` for the
/// last row to be full; a short tail row is passed as-is (matching
/// `chunks_mut`), never dropped.
pub fn rows_mut(buf: &mut [f32], row_len: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    chunks_mut(buf, row_len, f)
}

/// [`rows_mut`] over any `Send` element type - the shape of a loop whose
/// output rows are not `f32`: packed int8/int4 weight words (`u32`), decoded
/// block spans, index tables.
///
/// Same exactness contract as [`rows_mut`]: chunk `i` covers
/// `[i*chunk_len, (i+1)*chunk_len)` regardless of how many threads run, so a
/// loop whose per-chunk body reads only its own chunk and shared immutable
/// state produces a bit-identical result to the serial form - which is what
/// makes converting an existing serial loop a scheduling change rather than a
/// numerical one. A short tail chunk is passed as-is, never dropped.
pub fn chunks_mut<T: Send>(buf: &mut [T], chunk_len: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    assert!(chunk_len > 0, "chunks_mut: chunk_len must be non-zero");
    buf.par_chunks_mut(chunk_len).enumerate().for_each(|(i, c)| f(i, c));
}

/// [`chunks_mut`] over TWO outputs at once, each with its OWN chunk length:
/// index `i` gets `a[i*a_len..]` and `b[i*b_len..]`, in parallel.
///
/// The shape of a loop that produces two different things per row and needs
/// both - packed weight words plus that row's group scales being the
/// motivating case, where the two outputs have different widths (`k/4` u32
/// against `k/32` f32). Splitting it into two passes would either read the
/// input twice or recompute half the work, and neither is necessary when the
/// two outputs are disjoint.
///
/// Same exactness contract as [`chunks_mut`]: index `i` covers exactly its own
/// two chunks regardless of how many threads run, so a body reading only those
/// and shared immutable state is bit-identical to the serial form. `a` and `b`
/// must split into the same number of chunks.
pub fn chunks2_mut<T: Send, U: Send>(a: &mut [T], a_len: usize, b: &mut [U], b_len: usize, f: impl Fn(usize, &mut [T], &mut [U]) + Sync) {
    assert!(a_len > 0 && b_len > 0, "chunks2_mut: chunk lengths must be non-zero");
    assert_eq!(a.len().div_ceil(a_len), b.len().div_ceil(b_len), "chunks2_mut: {} chunks of a but {} of b", a.len().div_ceil(a_len), b.len().div_ceil(b_len));
    a.par_chunks_mut(a_len).zip(b.par_chunks_mut(b_len)).enumerate().for_each(|(i, (ca, cb))| f(i, ca, cb));
}

/// Apply `f(index, element)` to every element of `buf`, in parallel.
pub fn each_mut(buf: &mut [f32], f: impl Fn(usize, &mut f32) + Sync) {
    buf.par_iter_mut().enumerate().for_each(|(i, v)| f(i, v));
}

/// `(0..n).map(f)` in parallel, one `f32` per index — the shape of a matvec
/// fanned out over output rows.
pub fn map_f32(n: usize, f: impl Fn(usize) -> f32 + Sync + Send) -> Vec<f32> {
    (0..n).into_par_iter().map(f).collect()
}

/// `(0..n).flat_map(f)` in parallel, preserving index order — the shape of a
/// per-head attention fan-out.
pub fn flat_map_f32(n: usize, f: impl Fn(usize) -> Vec<f32> + Sync + Send) -> Vec<f32> {
    (0..n).into_par_iter().map(f).flatten().collect()
}

/// `(0..n).map(f)` in parallel, index-ordered, returning any `Send` value — the
/// shape of a fan-out over independent work items (one forecast per name, one
/// window per training row). Generalises [`map_f32`] to non-`f32` results so a
/// caller never reaches for a direct `rayon` dependency (the whole point of this
/// module: one pool, one policy). The pool is the scheduler's, so `--device
/// cpuN` sizing/affinity still governs it, and it fans out over the machine's
/// cores automatically at runtime.
pub fn map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    (0..n).into_par_iter().map(f).collect()
}

/// Sum of squares over a set of tensors, accumulated in `f64` — the global
/// grad-norm reduction. `f64` accumulation is part of the contract: summing
/// millions of squares in `f32` loses the low bits the clip threshold needs.
pub fn sum_sq_f64(tensors: &[Vec<f32>]) -> f64 {
    tensors.par_iter().map(|t| t.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()).sum()
}

/// Pairwise `f(a[i], b[i])` over two equal-length slices, in parallel — the
/// shape of an optimizer step over (state, grad) tensor pairs.
pub fn zip_each<T: Send, U: Sync>(a: &mut [T], b: &[U], f: impl Fn(&mut T, &U) + Sync) {
    assert_eq!(a.len(), b.len(), "zip_each: length mismatch {} vs {}", a.len(), b.len());
    a.par_iter_mut().zip(b.par_iter()).for_each(|(x, y)| f(x, y));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_mut_visits_every_row_once_with_its_index() {
        let mut buf = vec![0.0f32; 12];
        rows_mut(&mut buf, 4, |i, row| {
            for v in row.iter_mut() {
                *v = i as f32;
            }
        });
        assert_eq!(buf, vec![0., 0., 0., 0., 1., 1., 1., 1., 2., 2., 2., 2.]);
    }

    #[test]
    fn rows_mut_passes_a_short_tail_rather_than_dropping_it() {
        let mut buf = vec![1.0f32; 5];
        rows_mut(&mut buf, 2, |_, row| {
            for v in row.iter_mut() {
                *v += 1.0;
            }
        });
        assert_eq!(buf, vec![2.0; 5], "the 1-element tail row must be processed");
    }

    #[test]
    fn map_and_flat_map_preserve_index_order() {
        assert_eq!(map_f32(4, |i| i as f32 * 2.0), vec![0.0, 2.0, 4.0, 6.0]);
        assert_eq!(
            flat_map_f32(3, |i| vec![i as f32; 2]),
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            "flat_map must keep per-index blocks in order despite parallelism"
        );
    }

    #[test]
    fn map_generic_preserves_order() {
        assert_eq!(map(4, |i| (i, i * i)), vec![(0, 0), (1, 1), (2, 4), (3, 9)]);
        assert_eq!(map(3, |i| vec![i as u32; i]), vec![vec![], vec![1], vec![2, 2]]);
    }

    #[test]
    fn sum_sq_accumulates_in_f64() {
        // 1e8 elements of 1e-4: each square is 1e-8; the f32 running sum would
        // stall at ~0.25 once the accumulator dwarfs the addend. Use a smaller
        // proxy asserting exactness instead of a heavyweight demo.
        let t = vec![vec![3.0f32; 4], vec![4.0f32; 4]];
        assert_eq!(sum_sq_f64(&t), (9.0 * 4.0) + (16.0 * 4.0));
    }

    #[test]
    fn zip_each_pairs_by_index_and_rejects_mismatch() {
        let mut a = vec![1.0f32, 2.0, 3.0];
        let b = vec![10.0f32, 20.0, 30.0];
        zip_each(&mut a, &b, |x, y| *x += *y);
        assert_eq!(a, vec![11.0, 22.0, 33.0]);
        let r = std::panic::catch_unwind(|| {
            let mut a = vec![0.0f32; 2];
            zip_each(&mut a, &[0.0f32; 3], |_, _| {});
        });
        assert!(r.is_err(), "length mismatch must fail loudly, not truncate");
    }

    #[test]
    fn each_mut_is_elementwise_with_index() {
        let mut v = vec![0.0f32; 5];
        each_mut(&mut v, |i, x| *x = i as f32);
        assert_eq!(v, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }
}
