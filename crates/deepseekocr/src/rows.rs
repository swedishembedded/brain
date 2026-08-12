// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The multi-view **image-row layout**: which decoder row is a projector token,
//! which is the learned `image_newline` vector and which is the learned
//! `view_separator`.
//!
//! Pure host index math. No GPU, no weights, no buffers -- it is a function from
//! a resolution mode's tile grid to a sequence of [`Src`] entries, and it is
//! unit-tested in isolation. **Nothing consumes it yet**: the parity fixture is
//! single-view, and a single view needs no layout at all (its rows are the
//! projector's rows in order). It is written now because it is load-bearing for
//! the real-preprocessing phase and because it is the piece most likely to be
//! got wrong by assumption, so it is worth pinning while the reference is in
//! front of us.
//!
//! ## Where the formulas come from
//!
//! Not from a sketch: from llama.cpp's own consumer of this GGUF
//! (`tools/mtmd/models/deepseekocr.cpp` + `clip_n_output_tokens`), which is the
//! authority on what the shipped `mmproj-DeepSeek-OCR-Q8_0.gguf`'s
//! `v.image_newline` / `v.view_seperator` [sic] tensors are for.
//!
//! * **Global (overview) view** -- `h * (w + 1) + 1` rows: the `h x w` projector
//!   grid with ONE newline appended to each of the `h` rows, then ONE view
//!   separator terminating the view. (`n_patches = h * (h + 1) + 1`, and the
//!   overview is always square, so `w == h`.)
//! * **Local (tile) views** -- `(g * tiles_w + 1) * g` rows per tile ROW: the
//!   tiles of one row are **interleaved by grid row**, so a row of the layout is
//!   grid-row `y` of tile 0, then grid-row `y` of tile 1, …, and only then ONE
//!   newline. It is NOT "each tile, whole, followed by a newline". There is
//!   **no separator** in the local branch.
//!
//! ## The one thing that is NOT verified here
//!
//! The **order of the two blocks** -- global-then-local, as implemented -- is not
//! settled by the graph builder, because llama.cpp builds one graph per view and
//! the preprocessor that orders them is outside the file above. It is the order
//! this model's roadmap sketches, and it is the reading consistent with the
//! separator being attached to the *global* view and to nothing else (a
//! separator is a boundary; a trailing separator on the last block would be
//! pointless). Re-confirm it against the reference preprocessor when the real
//! image pipeline lands -- a swap changes no row COUNT, so only a real forward
//! can catch it. `RowPlan::rows` is a plain `Vec`, so reversing the two blocks
//! is a local change here and nowhere else.

/// Which vector occupies one row of the decoder's image-embedding block.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Src {
    /// Row `i` of the projector's output, over the concatenated views: view 0 is
    /// the global view and occupies `[0, g*g)`, tile `(ty, tx)` is view
    /// `1 + ty*tiles_w + tx` and occupies the next `g*g` rows.
    Projector(u32),
    /// The learned `image_newline` vector (`vision.image_newline`).
    Newline,
    /// The learned `view_separator` vector (`vision.view_separator`).
    Separator,
}

/// The local tile grid of a resolution mode. `0 x 0` means "global view only",
/// which is what every non-Gundam mode uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct ViewGrid {
    pub tiles_w: u32,
    pub tiles_h: u32,
}

impl ViewGrid {
    /// Global view only.
    pub fn global_only() -> ViewGrid {
        ViewGrid { tiles_w: 0, tiles_h: 0 }
    }
    pub fn new(tiles_w: u32, tiles_h: u32) -> ViewGrid {
        assert_eq!(
            tiles_w == 0,
            tiles_h == 0,
            "a tile grid is either empty in both axes or non-empty in both, got {tiles_w}x{tiles_h}"
        );
        ViewGrid { tiles_w, tiles_h }
    }
    pub fn tiles(&self) -> u32 {
        self.tiles_w * self.tiles_h
    }
}

/// One image's decoder-row layout.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RowPlan {
    /// One entry per decoder row, in order.
    pub rows: Vec<Src>,
    /// Projector tokens per view side (`grid_h/4`, and `grid_w/4`, which are
    /// equal for every real view -- both the overview and a tile are square).
    pub tokens_per_side: u32,
    pub grid: ViewGrid,
}

impl RowPlan {
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
    /// Projector rows the encoder must produce to fill this plan -- one view's
    /// worth per view, global included.
    pub fn projector_rows(&self) -> u32 {
        (1 + self.grid.tiles()) * self.tokens_per_side * self.tokens_per_side
    }
    /// How many rows are `Newline` / `Separator` (i.e. not projector output).
    pub fn special_rows(&self) -> usize {
        self.rows.iter().filter(|s| !matches!(s, Src::Projector(_))).count()
    }
    /// The contiguous runs of projector rows, as
    /// `(decoder_row0, projector_row0, n_rows)` -- exactly the argument shape the
    /// decoder's splice seam takes, one call per run.
    pub fn runs(&self) -> Vec<(u32, u32, u32)> {
        let mut out: Vec<(u32, u32, u32)> = Vec::new();
        for (i, s) in self.rows.iter().enumerate() {
            let Src::Projector(p) = *s else { continue };
            match out.last_mut() {
                // Extend only when BOTH sides stay contiguous; a newline breaks
                // the decoder side and a view change breaks the projector side.
                Some((d0, p0, n)) if *d0 + *n == i as u32 && *p0 + *n == p => *n += 1,
                _ => out.push((i as u32, p, 1)),
            }
        }
        out
    }
}

/// Rows in the global (overview) block, separator included: `g*(g+1) + 1`.
pub fn global_rows(tokens_per_side: u32) -> u32 {
    tokens_per_side * (tokens_per_side + 1) + 1
}

/// Rows in ONE tile row of the local block: `(g*tiles_w + 1) * g`.
pub fn local_row_rows(tokens_per_side: u32, tiles_w: u32) -> u32 {
    (tokens_per_side * tiles_w + 1) * tokens_per_side
}

/// Total decoder rows one image occupies, in closed form.
pub fn n_rows(tokens_per_side: u32, grid: ViewGrid) -> u32 {
    global_rows(tokens_per_side) + grid.tiles_h * local_row_rows(tokens_per_side, grid.tiles_w)
}

/// Build the full row layout for one image.
///
/// `tokens_per_side` is the projector tokens along one side of a view -- the SAM
/// compressor quarters the patch grid, so it is `grid_side / 4` (16 for the real
/// 1024² overview).
pub fn row_plan(tokens_per_side: u32, grid: ViewGrid) -> RowPlan {
    let g = tokens_per_side;
    assert!(g > 0, "a view must have at least one token per side");
    let view_len = g * g;
    let mut rows: Vec<Src> = Vec::with_capacity(n_rows(g, grid) as usize);

    // ---- global view: one newline per token row, then the view separator ----
    for y in 0..g {
        rows.extend((0..g).map(|x| Src::Projector(y * g + x)));
        rows.push(Src::Newline);
    }
    rows.push(Src::Separator);

    // ---- local tiles: tiles of one tile-row interleaved BY TOKEN ROW ----
    for ty in 0..grid.tiles_h {
        for y in 0..g {
            for tx in 0..grid.tiles_w {
                let base = (1 + ty * grid.tiles_w + tx) * view_len;
                rows.extend((0..g).map(|x| Src::Projector(base + y * g + x)));
            }
            rows.push(Src::Newline);
        }
    }

    debug_assert_eq!(rows.len() as u32, n_rows(g, grid));
    RowPlan { rows, tokens_per_side: g, grid }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projector_indices(p: &RowPlan) -> Vec<u32> {
        p.rows
            .iter()
            .filter_map(|s| match s {
                Src::Projector(i) => Some(*i),
                _ => None,
            })
            .collect()
    }

    /// The built layout and the closed form must agree, for every mode shape --
    /// the closed form is what a preprocessor sizes its prompt with, the vector
    /// is what actually fills it, and they are computed independently.
    #[test]
    fn closed_form_matches_the_built_layout() {
        for g in [1u32, 2, 4, 16] {
            for grid in [ViewGrid::global_only(), ViewGrid::new(1, 1), ViewGrid::new(3, 2), ViewGrid::new(2, 3)] {
                let p = row_plan(g, grid);
                assert_eq!(p.len() as u32, n_rows(g, grid), "g={g} grid={grid:?}");
            }
        }
    }

    /// The two per-view counts, against llama.cpp's own `clip_n_output_tokens`
    /// arithmetic for `PROJECTOR_TYPE_DEEPSEEKOCR`, at the REAL geometry: a
    /// 1024² overview compresses to 16x16 tokens.
    #[test]
    fn per_view_counts_match_the_reference_consumer() {
        // overview: n_patches = h*(h+1) + 1 with h = 16
        assert_eq!(global_rows(16), 16 * 17 + 1);
        assert_eq!(global_rows(16), 273);
        // tile row: (tile_patches*grid_w + 1) * h, tile_patches = h = 16
        assert_eq!(local_row_rows(16, 3), (16 * 3 + 1) * 16);
        assert_eq!(local_row_rows(16, 3), 784);
        // Gundam: one 3x2 tile grid plus the overview.
        assert_eq!(n_rows(16, ViewGrid::new(3, 2)), 273 + 2 * 784);
    }

    /// Every projector row is used exactly once and in ascending order WITHIN a
    /// view -- a layout that dropped, duplicated or transposed a token row would
    /// break one of the two.
    #[test]
    fn every_projector_row_is_used_exactly_once() {
        let g = 3;
        let grid = ViewGrid::new(2, 2);
        let p = row_plan(g, grid);
        let mut idx = projector_indices(&p);
        assert_eq!(idx.len() as u32, p.projector_rows());
        idx.sort_unstable();
        assert_eq!(idx, (0..p.projector_rows()).collect::<Vec<u32>>());
    }

    /// The global block comes first, is terminated by exactly one separator, and
    /// carries exactly one newline per token row. Newlines total one per token
    /// row of every view; separators total exactly one for the whole image.
    #[test]
    fn newlines_and_the_single_separator_land_where_the_reference_puts_them() {
        let g = 4;
        let grid = ViewGrid::new(3, 2);
        let p = row_plan(g, grid);
        let seps: Vec<usize> = p.rows.iter().enumerate().filter(|(_, s)| **s == Src::Separator).map(|(i, _)| i).collect();
        assert_eq!(seps, vec![(global_rows(g) - 1) as usize], "exactly one separator, ending the global block");
        let newlines = p.rows.iter().filter(|s| **s == Src::Newline).count();
        // g rows in the global view + g rows per tile-row.
        assert_eq!(newlines as u32, g + grid.tiles_h * g);
        // Every newline is preceded by a full row of projector tokens.
        for (i, s) in p.rows.iter().enumerate() {
            if *s != Src::Newline {
                continue;
            }
            let width = if i < global_rows(g) as usize { g } else { g * grid.tiles_w };
            for k in 1..=width as usize {
                assert!(matches!(p.rows[i - k], Src::Projector(_)), "row {} before a newline is not a token", i - k);
            }
        }
        assert_eq!(p.special_rows(), newlines + 1);
    }

    /// The tiles of one tile-row are interleaved BY TOKEN ROW, not concatenated
    /// whole. This is the property most likely to be got wrong, so it is spelled
    /// out on a 2-tile row at `g = 2`: the first layout row must be tile 0's row
    /// 0 followed by tile 1's row 0.
    #[test]
    fn tiles_in_a_row_are_interleaved_by_token_row() {
        let g = 2;
        let p = row_plan(g, ViewGrid::new(2, 1));
        let local = &p.rows[global_rows(g) as usize..];
        // view 1 = tile (0,0) at base 4, view 2 = tile (0,1) at base 8.
        assert_eq!(
            local,
            &[
                Src::Projector(4), Src::Projector(5), Src::Projector(8), Src::Projector(9), Src::Newline,
                Src::Projector(6), Src::Projector(7), Src::Projector(10), Src::Projector(11), Src::Newline,
            ]
        );
    }

    /// `runs()` is what a caller hands the decoder's splice seam. It must break
    /// at every newline AND at every view boundary, and cover every projector row.
    #[test]
    fn runs_break_at_newlines_and_view_boundaries() {
        let g = 2;
        let p = row_plan(g, ViewGrid::new(2, 1));
        let runs = p.runs();
        assert_eq!(runs, vec![(0, 0, 2), (3, 2, 2), (7, 4, 2), (9, 8, 2), (12, 6, 2), (14, 10, 2)]);
        let covered: u32 = runs.iter().map(|(_, _, n)| n).sum();
        assert_eq!(covered, p.projector_rows());
        for (d0, p0, n) in runs {
            for k in 0..n {
                assert_eq!(p.rows[(d0 + k) as usize], Src::Projector(p0 + k));
            }
        }
    }

    /// A global-only mode is one uninterrupted view: `g` runs of `g` tokens.
    #[test]
    fn global_only_mode_has_no_local_block() {
        let p = row_plan(16, ViewGrid::global_only());
        assert_eq!(p.len(), 273);
        assert_eq!(p.projector_rows(), 256);
        assert_eq!(p.runs().len(), 16);
        assert!(p.runs().iter().all(|(_, _, n)| *n == 16));
    }

    #[test]
    #[should_panic(expected = "empty in both axes")]
    fn a_half_empty_tile_grid_is_refused() {
        ViewGrid::new(3, 0);
    }
}
