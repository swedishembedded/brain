// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The rectangle-click world: a synthetic scene, an exact oracle, and two
//! renderings of the same scene - text (the existing, unmodified path) and
//! pixels (the new one under test).
//!
//! A scene draws 3-5 solid rectangles of distinct colors at distinct cells of
//! a 4x4 grid. The instruction names one color present in the scene; the
//! oracle answer is the cell that color's rectangle occupies. Both are exact
//! by construction - there is no labeling step to get wrong.

/// The click grid is 4x4: coarse enough that 3-5 non-overlapping rectangles
/// almost never contest a cell, fine enough that sixteen options is a real
/// discrimination task (chance = 6.25%).
pub const GRID: usize = 4;
pub const CELLS: usize = GRID * GRID;
pub const CANVAS: u32 = 256;
pub const CELL_PX: u32 = CANVAS / GRID as u32;

/// (name, RGB). More colors than the max rectangle count, so a scene never
/// needs to reuse one and "the red rectangle" is always unambiguous.
pub const COLORS: &[(&str, [u8; 3])] = &[
    ("red", [220, 40, 40]),
    ("blue", [40, 80, 220]),
    ("green", [40, 170, 70]),
    ("yellow", [225, 195, 30]),
    ("purple", [150, 60, 200]),
    ("orange", [230, 130, 30]),
];

/// The background the canvas is cleared to, and what an empty cell's patch
/// feature reads as - so "no rectangle here" is a real, distinct color
/// rather than a hole in the data.
pub const BACKGROUND: [u8; 3] = [245, 245, 245];

const MIN_RECTS: usize = 3;
const MAX_RECTS: usize = 5;

/// A seedable SplitMix64 generator, kept local rather than depending on
/// `brain::Rng` for it - the same choice `samples/decision/arena` made: an
/// environment this small does not need the SDK facade widened for its own
/// randomness. See `samples/decision/arena/src/arena.rs`.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform index in `[0, n)`.
    pub fn index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// `k` distinct indices in `[0, n)`, via partial Fisher-Yates.
    pub fn distinct(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = (0..n).collect();
        let mut out = Vec::with_capacity(k);
        for _ in 0..k.min(n) {
            let i = self.index(pool.len());
            out.push(pool.swap_remove(i));
        }
        out
    }
}

pub struct Scene {
    /// `(color index into COLORS, cell index 0..CELLS)`, one per rectangle.
    pub rects: Vec<(usize, usize)>,
    pub target_color: usize,
}

impl Scene {
    /// One scene: 3-5 rectangles at distinct cells and distinct colors, with
    /// the instruction naming one of the colors actually present.
    pub fn generate(rng: &mut Rng) -> Scene {
        let n = MIN_RECTS + rng.index(MAX_RECTS - MIN_RECTS + 1);
        let cells = rng.distinct(CELLS, n);
        let colors = rng.distinct(COLORS.len(), n);
        let rects: Vec<(usize, usize)> = colors.into_iter().zip(cells).collect();
        let target_color = rects[rng.index(rects.len())].0;
        Scene { rects, target_color }
    }

    /// The cell the instruction's color occupies - exact, by construction:
    /// `target_color` is always chosen from a color actually placed.
    pub fn oracle_cell(&self) -> usize {
        self.rects
            .iter()
            .find(|(c, _)| *c == self.target_color)
            .map(|(_, cell)| *cell)
            .expect("target_color is always one of the placed colors")
    }

    pub fn instruction(&self) -> String {
        format!("click the {} rectangle", COLORS[self.target_color].0)
    }

    /// Every rectangle's color and cell, in text, followed by the
    /// instruction - the `text` arm's state. This is the EXISTING,
    /// unmodified `Decide` path: plain tokenized text, no splice.
    pub fn text_state(&self) -> String {
        let mut s = String::new();
        for (c, cell) in &self.rects {
            s.push_str(&format!("{} rectangle at cell {cell}. ", COLORS[*c].0));
        }
        s.push_str(&self.instruction());
        s
    }

    /// Mean RGB per grid cell, normalized to `[0, 1]`, background for an
    /// empty cell. What the `pixels` arm's projector actually sees - no
    /// position, no rectangle count, just sixteen colors in a fixed order.
    pub fn patch_colors(&self) -> [[f32; 3]; CELLS] {
        let mut out = [[BACKGROUND[0] as f32 / 255.0, BACKGROUND[1] as f32 / 255.0, BACKGROUND[2] as f32 / 255.0]; CELLS];
        for (c, cell) in &self.rects {
            let [r, g, b] = COLORS[*c].1;
            out[*cell] = [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0];
        }
        out
    }

    /// Render onto a headless, GPU-free canvas - `Viewport::headless`'s own
    /// pixel-buffer path, the same one a `--shot` mode reads back as PNG.
    pub fn render(&self) -> brain::viewport::Canvas {
        let mut canvas = brain::viewport::Canvas::new(CANVAS, CANVAS);
        canvas.fill(0, 0, CANVAS, CANVAS, BACKGROUND);
        let margin = 8;
        let side = CELL_PX - 2 * margin as u32;
        for (c, cell) in &self.rects {
            let (row, col) = (cell / GRID, cell % GRID);
            let x = col as i32 * CELL_PX as i32 + margin;
            let y = row as i32 * CELL_PX as i32 + margin;
            canvas.fill(x, y, side, side, COLORS[*c].1);
        }
        canvas
    }
}

/// The sixteen click targets, in cell order - the same `Question::Choice`
/// option set for every scene and every arm, so no arm can leak the answer
/// through which options exist.
pub fn option_names() -> Vec<String> {
    (0..CELLS).map(|i| format!("cell {i}")).collect()
}
