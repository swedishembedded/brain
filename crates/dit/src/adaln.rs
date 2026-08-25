// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! adaLN modulation-table combination - PixArt-alpha's/Wan's parameter-saving
//! "shared table" trick: instead of every block projecting the conditioning
//! vector through its own Linear, ONE shared MLP produces a timestep-derived
//! vector and each block adds its own small learned table to it before that
//! sum is split into the usual `(shift, scale, gate)` triples (see `wan::
//! block::ModBufs`, whose `modulation` field IS that per-block table and
//! whose `upload` is [`add_table`]'s only caller today).
//!
//! Z-Image/S³-DiT does **not** use this trick - its `adaLN_modulation.0` is a
//! genuine per-block `Linear` over the shared conditioning vector (classic
//! DiT/PixArt-alpha adaLN-zero, folded straight into the RMSNorm weights by
//! `s3dit::block::fold_adaln`), so there is no separate table being added to
//! a shared vector and [`add_table`] has no Z-Image caller. Routing Z-Image
//! through this helper anyway would mean changing what its checkpoint
//! actually computes, not just where the code lives, which is exactly the
//! kind of "unify something that is deliberately different" this crate's
//! other modules (see `patchify`'s doc) also decline to do.

use std::collections::HashMap;

/// A per-token table stored as its DISTINCT rows plus a per-token index into
/// them - `[u, width]` and `[t]` in place of `[t, width]`.
///
/// # Why a per-token adaLN table is almost never `t` different rows
///
/// A per-token-timestep DiT (ltxv) computes one modulation row per token from
/// that token's own timestep, and the timestep is `denoise_mask * sigma`.
/// Plain text-to-video leaves `denoise_mask` all ones, so every token of a
/// step shares ONE timestep and the `[t, width]` table is one row copied `t`
/// times. Keyframe/anchor conditioning and a long-form window's carried
/// context each add exactly one more value (the frozen tokens' `0 * sigma`,
/// or `(1 - strength) * sigma`), so the real distinct-row count is 1 or 2 at
/// t in the thousands.
///
/// Rows are keyed by [`distinct_rows`] on the RAW BITS of the scalar that
/// produced them, so two rows share an entry only when they are the result of
/// the identical computation on the identical input. That makes the compact
/// form BIT-IDENTICAL to the expanded one, not merely close: it is the same
/// roundings in the same order, computed once instead of `t` times.
///
/// The width is the row stride, so one type serves both a `[t, 9*dim]` adaLN
/// table and a `[t, dim]` timestep embedding.
#[derive(Clone, Debug, PartialEq)]
pub struct RowTable {
    distinct: Vec<f32>,
    row_of: Vec<u32>,
    width: usize,
}

impl RowTable {
    /// `distinct` is `[u, width]` row-major; `row_of` is `[t]`, each entry a
    /// row index below `u`.
    pub fn new(distinct: Vec<f32>, row_of: Vec<u32>, width: usize) -> RowTable {
        assert!(width > 0, "adaln::RowTable: width must be positive");
        assert_eq!(distinct.len() % width, 0, "adaln::RowTable: {} floats is not a whole number of {width}-wide rows", distinct.len());
        let u = distinct.len() / width;
        assert!(row_of.iter().all(|&r| (r as usize) < u), "adaln::RowTable: a row index is past the {u} distinct rows");
        RowTable { distinct, row_of, width }
    }

    /// One row per token, no sharing - the identity encoding of a `[t, width]`
    /// table, for a caller that has no key to dedup on.
    pub fn dense(rows: Vec<f32>, width: usize) -> RowTable {
        let t = rows.len() / width.max(1);
        RowTable::new(rows, (0..t as u32).collect(), width)
    }

    /// Token count (`t`).
    pub fn len(&self) -> usize {
        self.row_of.len()
    }
    pub fn is_empty(&self) -> bool {
        self.row_of.is_empty()
    }
    /// Distinct-row count (`u`).
    pub fn distinct_len(&self) -> usize {
        self.distinct.len() / self.width
    }
    /// `[u, width]` row-major.
    pub fn distinct(&self) -> &[f32] {
        &self.distinct
    }
    /// `[t]`, each entry a row of [`Self::distinct`].
    pub fn row_of(&self) -> &[u32] {
        &self.row_of
    }
    pub fn width(&self) -> usize {
        self.width
    }
    /// Token `i`'s own `width` values.
    pub fn row(&self, i: usize) -> &[f32] {
        let r = self.row_of[i] as usize * self.width;
        &self.distinct[r..r + self.width]
    }
    /// The full `[t, width]` table this compacts - what a host consumer that
    /// indexes it densely (a parity tap, a reference block forward) needs.
    pub fn expand(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.len() * self.width];
        for (i, chunk) in out.chunks_mut(self.width).enumerate() {
            chunk.copy_from_slice(self.row(i));
        }
        out
    }
}

/// Split `keys` into its distinct values (in first-appearance order) and a
/// per-entry index into them.
///
/// Keyed on `f32::to_bits`, deliberately: bit equality is the only relation
/// that guarantees two keys drive the identical computation to the identical
/// result, which is what lets a caller compute one row per distinct key and
/// still be bit-identical to computing one row per entry. It is conservative
/// in exactly one direction that costs nothing here - `0.0` and `-0.0` get
/// separate rows - and it is total, so a NaN timestep dedups by payload
/// instead of collapsing every row or none.
pub fn distinct_rows(keys: &[f32]) -> (Vec<f32>, Vec<u32>) {
    let mut seen: HashMap<u32, u32> = HashMap::new();
    let mut distinct: Vec<f32> = Vec::new();
    let mut row_of: Vec<u32> = Vec::with_capacity(keys.len());
    for &k in keys {
        let next = distinct.len() as u32;
        let idx = *seen.entry(k.to_bits()).or_insert_with(|| {
            distinct.push(k);
            next
        });
        row_of.push(idx);
    }
    (distinct, row_of)
}

/// `out[r,i] = table[i] + v[r,i]`.
///
/// `v` is `[rows, width]` row-major: `rows == 1` is today's shape, where ONE
/// modulation vector (a function of the timestep alone) is shared by every
/// token in the sequence - Wan's `e0`. A future PER-TOKEN caller (ltxv, whose
/// modulation varies per token) is simply `rows == n_tokens`: `table` always
/// stays `[width]` and is added to every row unchanged, so going from
/// token-independent to token-dependent is a one-line change to the `rows`
/// argument at the call site, not a new function or a reshaped table.
pub fn add_table(v: &[f32], table: &[f32], rows: usize, width: usize) -> Vec<f32> {
    assert_eq!(v.len(), rows * width, "adaln::add_table: v is {}, need {}", v.len(), rows * width);
    assert_eq!(table.len(), width, "adaln::add_table: table is {}, need {width}", table.len());
    let mut out = vec![0f32; rows * width];
    for r in 0..rows {
        for i in 0..width {
            out[r * width + i] = table[i] + v[r * width + i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcasts_the_table_across_every_row() {
        let table = vec![1.0, 2.0, 3.0];
        let v = vec![0.0, 0.0, 0.0, 10.0, 20.0, 30.0];
        let out = add_table(&v, &table, 2, 3);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 11.0, 22.0, 33.0]);
    }

    #[test]
    fn rows_1_is_the_token_independent_case() {
        let table = vec![1.0, -1.0];
        let v = vec![0.5, 0.5];
        assert_eq!(add_table(&v, &table, 1, 2), vec![1.5, -0.5]);
    }

    #[test]
    fn a_uniform_key_vector_collapses_to_one_row() {
        let (distinct, row_of) = distinct_rows(&[0.7; 5]);
        assert_eq!(distinct, vec![0.7]);
        assert_eq!(row_of, vec![0, 0, 0, 0, 0]);
    }

    /// The anchored/long-form shape: frozen tokens at 0, generated tokens at
    /// sigma, interleaved rather than in one contiguous block - a scatter that
    /// only happens to be right for a contiguous split would pass a
    /// block-shaped case and fail this one.
    #[test]
    fn two_interleaved_keys_dedup_to_two_rows_in_first_appearance_order() {
        let (distinct, row_of) = distinct_rows(&[0.9, 0.0, 0.9, 0.9, 0.0]);
        assert_eq!(distinct, vec![0.9, 0.0]);
        assert_eq!(row_of, vec![0, 1, 0, 0, 1]);
    }

    #[test]
    fn every_key_distinct_is_the_dense_encoding() {
        let (distinct, row_of) = distinct_rows(&[1.0, 2.0, 3.0]);
        assert_eq!(distinct, vec![1.0, 2.0, 3.0]);
        assert_eq!(row_of, vec![0, 1, 2]);
    }

    /// `0.0` and `-0.0` compare EQUAL under `==` and differ in bits. Keying on
    /// bits keeps them apart, which is the conservative direction: they would
    /// drive the same rows here, but nothing in this module is entitled to
    /// assume that of an arbitrary caller's row function.
    #[test]
    fn signed_zero_is_two_keys_not_one() {
        let (distinct, row_of) = distinct_rows(&[0.0, -0.0]);
        assert_eq!(distinct.len(), 2);
        assert_eq!(row_of, vec![0, 1]);
    }

    #[test]
    fn expand_reproduces_the_table_the_compact_form_stands_for() {
        let t = RowTable::new(vec![1.0, 2.0, 30.0, 40.0], vec![1, 0, 1], 2);
        assert_eq!(t.len(), 3);
        assert_eq!(t.distinct_len(), 2);
        assert_eq!(t.row(0), &[30.0, 40.0]);
        assert_eq!(t.expand(), vec![30.0, 40.0, 1.0, 2.0, 30.0, 40.0]);
    }

    #[test]
    fn a_dense_table_round_trips_unchanged() {
        let rows = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = RowTable::dense(rows.clone(), 3);
        assert_eq!(t.distinct_len(), 2);
        assert_eq!(t.expand(), rows);
    }
}
