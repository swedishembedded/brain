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
//! ## Block order
//!
//! The reference model (`modeling_deepseekocr.py`) concatenates its image
//! embeddings as `[local tiles, global view, view separator]`
//! (`torch.cat([local_features, global_features, self.view_seperator[None, :]])`)
//! and scatters them over the image placeholder tokens in that order. The
//! placeholder ids are all the same, so only the embedding order matters, and it
//! is **local first, then global**: the local block's tiles, one newline per
//! token row of the whole tile mosaic; the global view, one newline per token
//! row; one separator, last. llama.cpp's graph builders say nothing about it
//! (one graph per view); the reference's own code does.

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
    /// Projector tokens per side of the global view (`grid_h/4`; both views are square).
    pub tokens_per_side: u32,
    /// Projector tokens per side of a local tile (10 for a 640 px tile, against the
    /// global view's 16).
    pub tile_tokens_per_side: u32,
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
        self.tokens_per_side * self.tokens_per_side + self.grid.tiles() * self.tile_tokens_per_side * self.tile_tokens_per_side
    }
    /// The `(view, token)` projector row `i` reads: view 0 is the global view, view
    /// `1 + t` the `t`-th tile (row-major), and `token` the row within that view's own output.
    pub fn locate(&self, i: u32) -> (usize, usize) {
        let global = self.tokens_per_side * self.tokens_per_side;
        if i < global {
            return (0, i as usize);
        }
        let tile = self.tile_tokens_per_side * self.tile_tokens_per_side;
        (1 + ((i - global) / tile) as usize, ((i - global) % tile) as usize)
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
pub fn n_rows(tokens_per_side: u32, tile_tokens_per_side: u32, grid: ViewGrid) -> u32 {
    global_rows(tokens_per_side) + grid.tiles_h * local_row_rows(tile_tokens_per_side, grid.tiles_w)
}

/// Build the full row layout for one image.
///
/// `tokens_per_side` is the projector tokens along one side of the global view --
/// the SAM compressor quarters the patch grid, so it is `grid_side / 4` (16 for
/// the real 1024² overview); `tile_tokens_per_side` is the same for a local tile
/// (10 for a 640² tile). Projector row `i` belongs to the global view when
/// `i < g*g`, and to tile `(i - g*g) / (t*t)` after it.
pub fn row_plan(tokens_per_side: u32, tile_tokens_per_side: u32, grid: ViewGrid) -> RowPlan {
    let (g, t) = (tokens_per_side, tile_tokens_per_side);
    assert!(g > 0 && t > 0, "a view must have at least one token per side");
    let (view_len, tile_len) = (g * g, t * t);
    let mut rows: Vec<Src> = Vec::with_capacity(n_rows(g, t, grid) as usize);

    // ---- local tiles first: tiles of one tile-row interleaved BY TOKEN ROW ----
    for ty in 0..grid.tiles_h {
        for y in 0..t {
            for tx in 0..grid.tiles_w {
                let base = view_len + (ty * grid.tiles_w + tx) * tile_len;
                rows.extend((0..t).map(|x| Src::Projector(base + y * t + x)));
            }
            rows.push(Src::Newline);
        }
    }

    // ---- then the global view: one newline per token row, then the view separator ----
    for y in 0..g {
        rows.extend((0..g).map(|x| Src::Projector(y * g + x)));
        rows.push(Src::Newline);
    }
    rows.push(Src::Separator);

    debug_assert_eq!(rows.len() as u32, n_rows(g, t, grid));
    RowPlan { rows, tokens_per_side: g, tile_tokens_per_side: t, grid }
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
        for (g, t) in [(1u32, 1u32), (2, 3), (4, 2), (16, 10)] {
            for grid in [ViewGrid::global_only(), ViewGrid::new(1, 1), ViewGrid::new(3, 2), ViewGrid::new(2, 3)] {
                let p = row_plan(g, t, grid);
                assert_eq!(p.len() as u32, n_rows(g, t, grid), "g={g} t={t} grid={grid:?}");
            }
        }
    }

    /// The per-view counts, against the reference's own arithmetic at the REAL
    /// geometry: a 1024² overview compresses to 16x16 tokens, a 640² tile to 10x10.
    #[test]
    fn per_view_counts_match_the_reference() {
        // overview: h*(w+1) + 1 with h = w = 16
        assert_eq!(global_rows(16), 273);
        // the tile mosaic: (10 * tiles_w + 1) * (10 * tiles_h)
        assert_eq!(local_row_rows(10, 3), 31 * 10);
        // Gundam: a 3x2 mosaic of 640 px tiles plus the overview.
        assert_eq!(n_rows(16, 10, ViewGrid::new(3, 2)), 273 + 2 * 310);
    }

    /// Every projector row is used exactly once, in ascending order within a
    /// view -- a layout that dropped, duplicated or transposed a token row would
    /// break one of the two.
    #[test]
    fn every_projector_row_is_used_exactly_once() {
        let p = row_plan(4, 3, ViewGrid::new(2, 2));
        let mut idx = projector_indices(&p);
        assert_eq!(idx.len() as u32, p.projector_rows());
        idx.sort_unstable();
        assert_eq!(idx, (0..p.projector_rows()).collect::<Vec<u32>>());
    }

    /// The reference concatenates `[local tiles, global view, separator]`: the
    /// image block ends with the global view's rows and exactly one separator,
    /// and has one newline per token row of the tile mosaic and of the global view.
    #[test]
    fn the_local_block_comes_first_and_the_global_block_ends_with_the_one_separator() {
        let (g, t) = (4, 3);
        let grid = ViewGrid::new(3, 2);
        let p = row_plan(g, t, grid);
        let seps: Vec<usize> = p.rows.iter().enumerate().filter(|(_, s)| **s == Src::Separator).map(|(i, _)| i).collect();
        assert_eq!(seps, vec![p.len() - 1], "exactly one separator, ending the image block");
        let newlines = p.rows.iter().filter(|s| **s == Src::Newline).count();
        assert_eq!(newlines as u32, g + grid.tiles_h * t);
        let local_len = (grid.tiles_h * local_row_rows(t, grid.tiles_w)) as usize;
        // Every newline is preceded by a full row of projector tokens: the mosaic's width, then the global view's.
        for (i, s) in p.rows.iter().enumerate() {
            if *s != Src::Newline {
                continue;
            }
            let width = if i < local_len { t * grid.tiles_w } else { g };
            for k in 1..=width as usize {
                assert!(matches!(p.rows[i - k], Src::Projector(_)), "row {} before a newline is not a token", i - k);
            }
        }
        // The first projector row of the block is a TILE token (view 1 starts at g*g), the global view's first follows the local block.
        assert_eq!(p.rows[0], Src::Projector(g * g));
        assert_eq!(p.rows[local_len], Src::Projector(0));
        assert_eq!(p.special_rows(), newlines + 1);
    }

    /// The tiles of one tile-row are interleaved BY TOKEN ROW, not concatenated
    /// whole: the first layout row is tile 0's row 0 followed by tile 1's row 0.
    #[test]
    fn tiles_in_a_row_are_interleaved_by_token_row() {
        let (g, t) = (3, 2);
        let p = row_plan(g, t, ViewGrid::new(2, 1));
        // view 0 = global (9 tokens); tile (0,0) = [9, 13), tile (0,1) = [13, 17).
        assert_eq!(
            &p.rows[..local_row_rows(t, 2) as usize],
            &[
                Src::Projector(9), Src::Projector(10), Src::Projector(13), Src::Projector(14), Src::Newline,
                Src::Projector(11), Src::Projector(12), Src::Projector(15), Src::Projector(16), Src::Newline,
            ]
        );
    }

    /// `runs()` is what a caller hands a splice seam. It must break at every
    /// newline AND at every view boundary, and cover every projector row.
    #[test]
    fn runs_break_at_newlines_and_view_boundaries() {
        let p = row_plan(2, 2, ViewGrid::new(2, 1));
        let runs = p.runs();
        assert_eq!(runs, vec![(0, 4, 2), (2, 8, 2), (5, 6, 2), (7, 10, 2), (10, 0, 2), (13, 2, 2)]);
        let covered: u32 = runs.iter().map(|(_, _, n)| n).sum();
        assert_eq!(covered, p.projector_rows());
        for (d0, p0, n) in runs {
            for k in 0..n {
                assert_eq!(p.rows[(d0 + k) as usize], Src::Projector(p0 + k));
            }
        }
    }

    #[test]
    fn locate_names_the_view_and_the_token_of_a_projector_row() {
        let p = row_plan(4, 3, ViewGrid::new(2, 1));
        // 16 global tokens, then two 9-token tiles.
        assert_eq!([0, 15, 16, 24, 25, 33].map(|i| p.locate(i)), [(0, 0), (0, 15), (1, 0), (1, 8), (2, 0), (2, 8)]);
    }

    /// A global-only mode is one uninterrupted view: `g` runs of `g` tokens.
    #[test]
    fn global_only_mode_has_no_local_block() {
        let p = row_plan(16, 10, ViewGrid::global_only());
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
