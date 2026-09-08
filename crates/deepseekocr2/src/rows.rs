// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The image-row layout: which decoder row is a projector token, and where
//! the single learned `vision.view_separator` sits.
//!
//! Pure host index math, unit-tested in isolation, no GPU and no weights -
//! the same shape `crates/deepseek2ocr/src/rows.rs` uses for v1, but a much
//! smaller formula: v2 has no per-row `image_newline` at all (the real
//! mmproj checkpoint carries exactly one `view_separator` tensor and no
//! newline tensor of any kind), so a view's tokens sit in the sequence as
//! one uninterrupted run rather than being interleaved with a newline after
//! every token row the way v1's global/local blocks are.
//!
//! ## What is, and is not, settled here
//!
//! **Settled from real tensors and the merged mtmd graph builder**: one
//! view's own row order is image tokens then queries, and a view's
//! projected output is one contiguous run with no interleaving (only one
//! `view_separator` tensor exists in the real checkpoint, and no
//! `image_newline` at all - unlike v1, where both exist).
//! **Still a design assumption, not yet an empirical fact**: the order
//! MULTIPLE local tiles take relative to each other and to the global view.
//! This module assumes row-major-over-the-tile-grid (width fastest), then
//! the global view, then the separator - the same assumption
//! `crates/deepseekocr2/src/encoder.rs`'s `gather_rows` already codified for
//! M3/M4's fixture. If a later milestone's real forward or a real llama.cpp
//! dump finds a different order, only [`row_plan`] changes - nothing that
//! consumes a [`RowPlan`] needs to know the formula.

/// Which vector occupies one row of the decoder's spliced image block.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Src {
    /// Row `i` of the flat projector-row space: every local tile's rows
    /// first (tile 0's `n_query_local` rows, then tile 1's, ...), then the
    /// global view's `n_query_global` rows.
    Projector(u32),
    /// The one learned `vision.view_separator` row, terminating the block.
    Separator,
}

/// A local-tile grid. `0x0` means "no local tiles, global view only".
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct TileGrid {
    pub tiles_w: u32,
    pub tiles_h: u32,
}

impl TileGrid {
    pub fn none() -> TileGrid {
        TileGrid { tiles_w: 0, tiles_h: 0 }
    }
    pub fn new(tiles_w: u32, tiles_h: u32) -> TileGrid {
        TileGrid { tiles_w, tiles_h }
    }
    pub fn tiles(&self) -> u32 {
        self.tiles_w * self.tiles_h
    }
}

/// One image's decoder-row layout: the tile grid plus the two per-view query
/// counts fully determine it, so a [`RowPlan`] carries both rather than
/// forcing every caller to re-derive [`RowPlan::runs`] from raw integers.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RowPlan {
    pub rows: Vec<Src>,
    pub grid: TileGrid,
    pub n_query_local: u32,
    pub n_query_global: u32,
}

impl RowPlan {
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
    /// Projector rows every view together must produce to fill this plan.
    pub fn projector_rows(&self) -> u32 {
        self.grid.tiles() * self.n_query_local + self.n_query_global
    }
    /// The `(row0, n_rows)` run each view occupies, in the SAME order
    /// [`crate::encoder::gather_rows`] concatenates them: one run per local
    /// tile, then one for the global view. Unlike v1's `RowPlan::runs` (which
    /// must stop at every `Newline`), nothing here ever splits a view's rows,
    /// because there is nothing between them to split on.
    pub fn runs(&self) -> Vec<(u32, u32)> {
        let mut out = Vec::with_capacity(self.grid.tiles() as usize + 1);
        let mut row0 = 0u32;
        for _ in 0..self.grid.tiles() {
            out.push((row0, self.n_query_local));
            row0 += self.n_query_local;
        }
        out.push((row0, self.n_query_global));
        out
    }
    /// The one row that carries `vision.view_separator` - always last.
    pub fn separator_row(&self) -> u32 {
        self.projector_rows()
    }
}

/// Build the layout for one image: `grid.tiles()` local-tile runs, then the
/// global view's run, then one separator row.
pub fn row_plan(grid: TileGrid, n_query_local: u32, n_query_global: u32) -> RowPlan {
    assert!(n_query_global > 0, "the global view must produce at least one row");
    assert!(grid.tiles() == 0 || n_query_local > 0, "a non-empty tile grid needs a positive per-tile query count");
    let mut rows = Vec::with_capacity((grid.tiles() * n_query_local + n_query_global + 1) as usize);
    let mut base = 0u32;
    for _ in 0..grid.tiles() {
        rows.extend((0..n_query_local).map(|i| Src::Projector(base + i)));
        base += n_query_local;
    }
    rows.extend((0..n_query_global).map(|i| Src::Projector(base + i)));
    rows.push(Src::Separator);
    RowPlan { rows, grid, n_query_local, n_query_global }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The closed-form row count (`runs` summed, plus the separator) must
    /// match the built plan's length, over grid shapes that are NOT square
    /// (so a transposed `tiles_w`/`tiles_h` would be caught).
    #[test]
    fn closed_form_matches_the_built_layout() {
        for grid in [TileGrid::none(), TileGrid::new(1, 1), TileGrid::new(3, 2), TileGrid::new(2, 5)] {
            for (nl, ng) in [(5u32, 8u32), (3, 11)] {
                let p = row_plan(grid, nl, ng);
                let run_total: u32 = p.runs().iter().map(|(_, n)| n).sum();
                assert_eq!(run_total, p.projector_rows(), "grid={grid:?} nl={nl} ng={ng}");
                assert_eq!(p.len() as u32, p.projector_rows() + 1, "grid={grid:?} nl={nl} ng={ng}");
            }
        }
    }

    /// Every projector row appears exactly once, in ascending order, and the
    /// separator is the very last row - the two properties a caller's
    /// exact-equality splice test actually depends on.
    #[test]
    fn projector_rows_are_used_exactly_once_in_order_and_the_separator_is_last() {
        let p = row_plan(TileGrid::new(3, 2), 5, 8);
        let mut seen = Vec::new();
        for (i, s) in p.rows.iter().enumerate() {
            match *s {
                Src::Projector(k) => {
                    assert_eq!(k, i as u32, "row {i} should carry projector row {i}, got {k}");
                    seen.push(k);
                }
                Src::Separator => assert_eq!(i, p.rows.len() - 1, "the separator must be the last row"),
            }
        }
        assert_eq!(seen.len() as u32, p.projector_rows());
    }

    /// Tiles occupy contiguous, non-overlapping runs in row-major
    /// (width-first) order, each exactly `n_query_local` rows wide - the
    /// property a wrong `tiles_w`/`tiles_h` order or an off-by-one tile
    /// boundary would break.
    #[test]
    fn tile_runs_are_contiguous_width_first_and_the_right_width() {
        let (grid, nl, ng) = (TileGrid::new(3, 2), 4u32, 6u32);
        let p = row_plan(grid, nl, ng);
        let runs = p.runs();
        assert_eq!(runs.len(), (grid.tiles() + 1) as usize);
        for (i, (row0, n)) in runs.iter().enumerate().take(grid.tiles() as usize) {
            assert_eq!(*row0, i as u32 * nl, "tile {i} does not start where the previous one ended");
            assert_eq!(*n, nl, "tile {i} is not {nl} rows wide");
        }
        let (grow0, gn) = *runs.last().unwrap();
        assert_eq!(grow0, grid.tiles() * nl, "the global run does not start right after the last tile");
        assert_eq!(gn, ng);
        assert_eq!(p.separator_row(), grow0 + gn);
    }

    /// A global-only image (no local tiles) is one run plus the separator.
    #[test]
    fn a_global_only_grid_has_no_local_runs() {
        let p = row_plan(TileGrid::none(), 5, 8);
        assert_eq!(p.runs(), vec![(0, 8)]);
        assert_eq!(p.len(), 9);
    }
}
