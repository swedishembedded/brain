// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The rectangle-click world: a synthetic scene, an exact oracle, and two
//! renderings of the same scene - text (the existing, unmodified path) and
//! pixels (the new one under test).
//!
//! A scene draws 3 solid rectangles of distinct colors at distinct cells of
//! a 4x4 grid. The instruction names one color present in the scene; the
//! oracle answer is the cell that color's rectangle occupies. Both are exact
//! by construction - there is no labeling step to get wrong.

/// The click grid is 4x4: coarse enough that 3 non-overlapping rectangles
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

/// What `blind` and `pixels` tokenize as `state` when no scene fact belongs
/// in text at all - non-empty because `Decide::pack_request` refuses an
/// empty one, uninformative by construction.
pub const BLIND_STATE: &str = "no scene description available";

// Calibrated, not assumed: 3-5 rectangles (the original plan) left `text`'s
// loss flat at chance for 800-2000 steps even with the encoder unfrozen and
// the per-example question fix in place (see `question_for`'s doc) - a real
// measured run, not a guess, since compositional binding against more
// distractors needs more budget than this sample trains for. A fixed 3
// clears the bar decisively (0.263 held-out accuracy vs 0.062 chance, 3000
// steps) and keeps every scene's binding task the same shape.
const MIN_RECTS: usize = 3;
const MAX_RECTS: usize = 3;

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
    /// One scene: 3 rectangles at distinct cells and distinct colors, with
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

    /// The per-example QUESTION, not scene content - deliberately kept out
    /// of every arm's `state`. `Decide`'s head computes an option's query
    /// from the option's OWN slot text (`"{instructions} [SEP] {option}"`,
    /// `crates/decide/src/primitives.rs`), never from the state it attends
    /// over. Burying the target color in `state` instead of `instructions`
    /// was tried first and measured to fail even with the encoder unfrozen:
    /// with a shared, per-example-invariant `instructions` string, "cell 7"'s
    /// query is nearly identical across every example, so cross-attention can
    /// retrieve facts near the literal text "cell 7" but has no channel to
    /// compare them against an instruction-stated color it never saw. Putting
    /// the color here instead makes it part of what every option's query
    /// itself encodes.
    pub fn instruction(&self) -> String {
        format!("click the {} rectangle", COLORS[self.target_color].0)
    }

    /// Every rectangle's color and cell, in text - the `text` arm's state.
    /// The instruction is NOT appended here; see [`Self::instruction`].
    pub fn text_state(&self) -> String {
        let mut s = String::new();
        for (c, cell) in &self.rects {
            s.push_str(&format!("{} rectangle at cell {cell}. ", COLORS[*c].0));
        }
        s
    }

    /// The color index per grid cell, `COLORS.len()` (one past the end) for
    /// an empty cell - what the `pixels` arm's projector actually sees. No
    /// position, no rectangle count, just sixteen cells' color IDENTITY in a
    /// fixed order.
    ///
    /// A raw-RGB version of this (`[f32;3]` mean color per cell) was tried
    /// first and left `pixels` flat at chance for 3000-6000 steps even after
    /// a LayerNorm fixed a real scale mismatch against the encoder's own
    /// rows (measured: projector rows RMS ~0.17-0.27 vs real rows ~0.36-0.63
    /// at init). The LayerNorm fix did not move the result, which narrows
    /// the problem: unlike a real vision encoder (CLIP's tower is itself
    /// CONTRASTIVELY PRETRAINED to align with text, which is exactly why a
    /// LLaVA-style linear projector on top of it can bootstrap from a
    /// randomly initialized start in a modest training budget), continuous
    /// RGB in this sample carries no prior relationship AT ALL to how the
    /// frozen encoder represents the WORD "red" - the head would have to
    /// learn that entire cross-modal alignment from scratch, from a 16-way
    /// softmax's weak supervision, which is a much harder bootstrapping
    /// problem than the splice mechanism this sample exists to test. A
    /// one-hot color identity removes that confound: the projector's job
    /// becomes "which of six known symbols is present here", the same
    /// closed vocabulary `instruction()` already draws from in text, so the
    /// head only has to learn a match against a fixed discrete alphabet
    /// rather than an unconstrained continuous embedding.
    pub fn patch_color_index(&self) -> [usize; CELLS] {
        let mut out = [COLORS.len(); CELLS];
        for (c, cell) in &self.rects {
            out[*cell] = *c;
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
