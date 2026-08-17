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
}
