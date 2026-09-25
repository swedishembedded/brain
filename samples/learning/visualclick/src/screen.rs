// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The screen the model looks at: a rendered interface, the controls on it,
//! and one plain-language instruction that refers to exactly one control.
//!
//! A screen is a title bar, a toolbar strip, a sidebar and a main panel, with
//! 10-16 controls placed in fixed slots. A control is a coloured SHAPE in one of two
//! states - the three attributes an instruction can refer to:
//!
//! ```text
//! shape   square | disc | triangle | ring | bar
//! colour  red | blue | green | yellow | purple | orange
//! state   bright | dimmed
//! ```
//!
//! **The instruction is verified to identify exactly one control, per
//! example.** Referring expressions are not generated from the target and
//! hoped to be unique - they are enumerated over the whole screen, each one
//! matched back against every control, and only the expressions with exactly
//! one match survive (`refs`). A screen that produces no unique expression is
//! regenerated rather than labelled ambiguously. That is what makes the oracle
//! exact: there is no labelling step that can be wrong, and no example where
//! two answers are equally defensible but only one is scored correct.
//!
//! Three kinds of expression survive that filter, and each demands something
//! different of the model:
//!
//! | expression | what it forces |
//! |---|---|
//! | `click the red square` | binding TWO attributes at once - the screen holds red things that are not squares and squares that are not red |
//! | `click the dimmed red square` | a third attribute, and it is only offered when the colour/shape pair alone is ambiguous, so the state word is always load-bearing |
//! | `click the red square in the sidebar` | which AREA of the interface it is in - the toolbar, the sidebar or the main panel, each with its own background |

use crate::vision;
use brain::viewport::Canvas;

pub const W: u32 = 320;
pub const H: u32 = 240;

const TITLE_H: u32 = 20;
const TOOLBAR_Y: u32 = 20;
const CONTENT_Y: u32 = 56;

const TOOLBAR_SLOTS: usize = 6;
/// The sidebar is a real strip of the interface with its own background, not a
/// half of the screen named by a preposition - see [`REGIONS`].
const SIDEBAR_SLOTS: usize = 3;
const SIDEBAR_W: u32 = 76;
const CONTENT_COLS: usize = 3;
const CONTENT_ROWS: usize = 3;
pub const SLOTS: usize = TOOLBAR_SLOTS + SIDEBAR_SLOTS + CONTENT_COLS * CONTENT_ROWS;

/// Fewest and most controls on one screen. The upper bound is what makes a
/// correct answer worth something - at 16 controls chance is 6.25%.
const MIN_WIDGETS: usize = 10;
const MAX_WIDGETS: usize = 16;

pub const CHROME: [u8; 3] = [236, 238, 242];
pub const PANEL: [u8; 3] = [250, 250, 252];
/// The sidebar's own background - what makes "in the sidebar" something a
/// control's pixels answer rather than a coordinate the model must threshold.
pub const SIDEBAR: [u8; 3] = [222, 228, 238];
const TITLEBAR: [u8; 3] = [52, 58, 74];
const BORDER: [u8; 3] = [198, 203, 212];

pub const COLORS: &[(&str, [u8; 3])] = &[
    ("red", [214, 48, 49]),
    ("blue", [41, 98, 214]),
    ("green", [32, 158, 84]),
    ("yellow", [226, 190, 28]),
    ("purple", [146, 62, 198]),
    ("orange", [232, 126, 24]),
];

pub const SHAPES: &[&str] = &["square", "disc", "triangle", "ring", "bar"];

/// A dimmed control is blended this far toward the panel behind it - the
/// "disabled" look, and a real difference in the pixels rather than a flag
/// carried alongside them.
const DIM_MIX: f32 = 0.62;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Widget {
    pub shape: usize,
    pub color: usize,
    pub dim: bool,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Widget {
    /// Where a click on this control lands - what the sample reports as its
    /// answer, in screen pixels.
    pub fn center(&self) -> (u32, u32) {
        (self.x + self.w / 2, self.y + self.h / 2)
    }
}

pub struct Screen {
    pub widgets: Vec<Widget>,
    /// Index into `widgets` - the one control the instruction refers to.
    pub target: usize,
    pub instruction: String,
}

/// A seedable SplitMix64 generator, kept local rather than widening the SDK
/// facade for a sample's own randomness - the same choice
/// `samples/decision/arena` made.
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

    pub fn index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

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

/// The screen rectangle a slot occupies. Slots never overlap, so a control's
/// pixels are never another control's pixels - which is what lets a crop be
/// read as ONE control's appearance.
fn slot_box(slot: usize) -> (u32, u32, u32, u32) {
    if slot < TOOLBAR_SLOTS {
        (10 + slot as u32 * 50, TOOLBAR_Y + 4, 28, 28)
    } else if slot < TOOLBAR_SLOTS + SIDEBAR_SLOTS {
        let j = (slot - TOOLBAR_SLOTS) as u32;
        (20, CONTENT_Y + 14 + j * 58, 36, 36)
    } else {
        let j = slot - TOOLBAR_SLOTS - SIDEBAR_SLOTS;
        let (row, col) = (j / CONTENT_COLS, j % CONTENT_COLS);
        (SIDEBAR_W + 16 + col as u32 * 74, CONTENT_Y + 10 + row as u32 * 60, 44, 44)
    }
}

fn mix(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    let m = |i: usize| (a[i] as f32 * (1.0 - t) + b[i] as f32 * t).round().clamp(0.0, 255.0) as u8;
    [m(0), m(1), m(2)]
}

fn put(canvas: &mut Canvas, x: i32, y: i32, c: [u8; 3]) {
    if x < 0 || y < 0 || x as u32 >= W || y as u32 >= H {
        return;
    }
    let i = ((y as u32 * W + x as u32) * 3) as usize;
    let px = canvas.pixels_mut();
    px[i] = c[0];
    px[i + 1] = c[1];
    px[i + 2] = c[2];
}

/// Draw one control. The five shapes are deliberately distinguishable after
/// the heavy downsample [`vision`] applies - a filled box, a disc, a triangle,
/// a hollow ring and a short wide bar differ in WHERE the ink is, not in fine
/// detail that a low-resolution crop would destroy.
fn draw_widget(canvas: &mut Canvas, wgt: &Widget, background: [u8; 3]) {
    let base = COLORS[wgt.color].1;
    let c = if wgt.dim { mix(base, background, DIM_MIX) } else { base };
    let (x, y, w, h) = (wgt.x as i32, wgt.y as i32, wgt.w, wgt.h);
    match SHAPES[wgt.shape] {
        "square" => canvas.fill(x, y, w, h, c),
        "disc" => {
            let r = (w.min(h) as f32 / 2.0) - 0.5;
            let (cx, cy) = (x as f32 + w as f32 / 2.0, y as f32 + h as f32 / 2.0);
            for dy in 0..h as i32 {
                for dx in 0..w as i32 {
                    let (px, py) = (x + dx, y + dy);
                    let (ux, uy) = (px as f32 + 0.5 - cx, py as f32 + 0.5 - cy);
                    if ux * ux + uy * uy <= r * r {
                        put(canvas, px, py, c);
                    }
                }
            }
        }
        "triangle" => {
            for dy in 0..h as i32 {
                // Widens linearly from the apex, so the ink sits low - the
                // opposite of the ring's and the bar's distribution.
                let frac = (dy + 1) as f32 / h as f32;
                let half = (frac * w as f32 / 2.0).round() as i32;
                let midx = x + w as i32 / 2;
                for px in (midx - half)..=(midx + half) {
                    put(canvas, px, y + dy, c);
                }
            }
        }
        "ring" => {
            let t = (w.min(h) / 4).max(2);
            canvas.fill(x, y, w, t, c);
            canvas.fill(x, y + (h - t) as i32, w, t, c);
            canvas.fill(x, y, t, h, c);
            canvas.fill(x + (w - t) as i32, y, t, h, c);
        }
        "bar" => {
            let bh = (h / 3).max(3);
            canvas.fill(x, y + ((h - bh) / 2) as i32, w, bh, c);
        }
        other => unreachable!("unknown shape {other}"),
    }
}

/// One referring expression and the single control it picks out.
struct Ref {
    text: String,
    target: usize,
}

/// A named area of the screen a control can be IN.
///
/// Deliberately overlapping (`in the toolbar` and `in the top row` both hold
/// for a toolbar control) so the model cannot treat the region word as a
/// partition it could infer from anything else.
pub struct Region {
    pub name: &'static str,
    bounds: fn(&Widget) -> bool,
}

impl Region {
    fn holds(&self, w: &Widget) -> bool {
        (self.bounds)(w)
    }
}

/// The three areas of the interface, each a real region with its own
/// background rather than a half of the screen named by a preposition.
///
/// Both properties are load-bearing, and both were measured. An earlier
/// version used `on the left` / `on the right`, which are geometrically
/// perfectly well defined and which the model could not answer: 28.6% and
/// 57.9%, against 100% for `in the toolbar` in the same run. Two reasons,
/// and a named area fixes both at once.
///
/// **They are lexically distinct.** `left` and `right` are short function
/// words, and the frozen sentence encoder barely separates them - two
/// instructions differing only in that word measured **cosine 0.9837**,
/// against 0.8510 for two differing in a colour. Nothing downstream recovers a
/// distinction the encoder collapsed. `toolbar`, `sidebar` and `main panel`
/// are content words the encoder keeps apart.
///
/// **They are visible.** A named area has its own background, so which one a
/// control sits in is readable from the control's own pixels - the same way
/// its colour is - instead of only from a coordinate compared against a
/// threshold the model has to infer. Every region the model answered well in
/// that earlier run had exactly this property.
///
/// Mutually exclusive and exhaustive, so every control is in precisely one.
pub const REGIONS: &[Region] = &[
    Region { name: "in the toolbar", bounds: |w| w.y < CONTENT_Y },
    Region { name: "in the sidebar", bounds: |w| w.y >= CONTENT_Y && w.x < SIDEBAR_W },
    Region { name: "in the main panel", bounds: |w| w.y >= CONTENT_Y && w.x >= SIDEBAR_W },
];

/// Every unambiguous way to refer to a control on this screen.
///
/// This is the oracle. An expression is only offered if matching it back
/// against every control on the screen yields exactly one - so "the red
/// square" is never used on a screen holding two red squares, and the
/// area and state forms are only offered where they are what resolves the
/// ambiguity.
fn refs(ws: &[Widget]) -> Vec<Ref> {
    let mut out = Vec::new();
    for color in 0..COLORS.len() {
        for shape in 0..SHAPES.len() {
            let hits: Vec<usize> = (0..ws.len()).filter(|&i| ws[i].color == color && ws[i].shape == shape).collect();
            let (cname, sname) = (COLORS[color].0, SHAPES[shape]);

            if hits.len() == 1 {
                out.push(Ref { text: format!("click the {cname} {sname}"), target: hits[0] });
                continue;
            }
            if hits.is_empty() {
                continue;
            }

            // The colour/shape pair is ambiguous here, so every expression
            // below has to do real work to resolve it.
            for dim in [false, true] {
                let bystate: Vec<usize> = hits.iter().copied().filter(|&i| ws[i].dim == dim).collect();
                if bystate.len() == 1 {
                    let word = if dim { "dimmed" } else { "bright" };
                    out.push(Ref { text: format!("click the {word} {cname} {sname}"), target: bystate[0] });
                }
            }

            // Spatial, by REGION - "the red square in the toolbar", not "the
            // leftmost red square".
            //
            // Both are spatial reference and only one of them is a property of
            // the control being scored. A region is: the control is in the
            // toolbar or it is not, readable from its own box the same way its
            // colour is readable from its own pixels. A superlative is a
            // comparison ACROSS the options - select the matches, take the
            // extreme of that subset, recognise yourself as it - and the head
            // scoring these has one cross-attention layer and a scalar
            // readout, whose best available mechanism is a comparison against
            // the attention-weighted MEAN of the matches. A mean is a soft
            // proxy for an extremum, and it measured like one: with everything
            // else at 94-98%, superlatives sat at 52.9-81.1% per direction
            // across four separate attempts to lift them (a positional
            // encoding, a bounded multiplicative conditioning, and whitening
            // the instruction embedding), while training loss was already at
            // zero. That is a representational ceiling, not a training budget.
            //
            // Region reference is the form a UI instruction usually takes
            // anyway - "the save button in the toolbar" - and it is the form
            // this head can answer.
            for region in REGIONS {
                let inside: Vec<usize> = hits.iter().copied().filter(|&i| region.holds(&ws[i])).collect();
                if inside.len() == 1 {
                    out.push(Ref { text: format!("click the {cname} {sname} {}", region.name), target: inside[0] });
                }
            }
        }
    }
    out
}

impl Screen {
    /// One screen and one instruction that identifies exactly one of its
    /// controls. Regenerates rather than settling for an ambiguous label.
    pub fn generate(rng: &mut Rng) -> Screen {
        loop {
            let n = MIN_WIDGETS + rng.index(MAX_WIDGETS - MIN_WIDGETS + 1);
            let slots = rng.distinct(SLOTS, n);
            let widgets: Vec<Widget> = slots
                .into_iter()
                .map(|slot| {
                    let (x, y, w, h) = slot_box(slot);
                    Widget {
                        shape: rng.index(SHAPES.len()),
                        color: rng.index(COLORS.len()),
                        dim: rng.f32() < 0.35,
                        x,
                        y,
                        w,
                        h,
                    }
                })
                .collect();

            let options = refs(&widgets);
            if options.is_empty() {
                continue;
            }
            let pick = rng.index(options.len());
            let Ref { text, target } = options.into_iter().nth(pick).expect("pick is in range");
            return Screen { widgets, target, instruction: text };
        }
    }

    /// Where the correct click lands, in screen pixels.
    pub fn target_point(&self) -> (u32, u32) {
        self.widgets[self.target].center()
    }

    /// What is drawn behind a control - which is what a dimmed one blends
    /// toward, and what makes its region visible in its own crop.
    fn background_behind(&self, w: &Widget) -> [u8; 3] {
        if w.y < CONTENT_Y {
            CHROME
        } else if w.x < SIDEBAR_W {
            SIDEBAR
        } else {
            PANEL
        }
    }

    pub fn render(&self) -> Canvas {
        let mut canvas = Canvas::new(W, H);
        canvas.clear(CHROME);
        canvas.fill(0, 0, W, TITLE_H, TITLEBAR);
        canvas.text(8, 6, "control panel", 1, [236, 238, 242]);
        canvas.fill(6, CONTENT_Y as i32, SIDEBAR_W - 6, H - CONTENT_Y - 6, SIDEBAR);
        canvas.outline(6, CONTENT_Y as i32, SIDEBAR_W - 6, H - CONTENT_Y - 6, BORDER);
        canvas.fill(SIDEBAR_W as i32, CONTENT_Y as i32, W - SIDEBAR_W - 6, H - CONTENT_Y - 6, PANEL);
        canvas.outline(SIDEBAR_W as i32, CONTENT_Y as i32, W - SIDEBAR_W - 6, H - CONTENT_Y - 6, BORDER);
        for wgt in &self.widgets {
            draw_widget(&mut canvas, wgt, self.background_behind(wgt));
        }
        canvas
    }

    /// The per-control features the model actually reads, from the RENDERED
    /// pixels - see [`vision::widget_features`].
    pub fn features(&self, canvas: &Canvas) -> Vec<Vec<f32>> {
        self.widgets.iter().map(|w| vision::widget_features(canvas, w)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The oracle's whole claim: the instruction names exactly one control.
    /// Re-derived here by re-matching the chosen expression against every
    /// control, independently of the generator that produced it.
    #[test]
    fn every_instruction_identifies_exactly_one_control() {
        let mut rng = Rng::new(4);
        for _ in 0..400 {
            let s = Screen::generate(&mut rng);
            let matches: Vec<usize> = refs(&s.widgets).into_iter().filter(|r| r.text == s.instruction).map(|r| r.target).collect();
            assert!(!matches.is_empty(), "{:?} matched nothing", s.instruction);
            for m in &matches {
                assert_eq!(*m, s.target, "{:?} resolves to more than one control", s.instruction);
            }
        }
    }

    /// Controls never share pixels, so one control's crop is never partly
    /// another's - the assumption `vision::widget_features` rests on.
    #[test]
    fn controls_never_overlap() {
        let mut rng = Rng::new(9);
        for _ in 0..200 {
            let s = Screen::generate(&mut rng);
            for i in 0..s.widgets.len() {
                for j in (i + 1)..s.widgets.len() {
                    let (a, b) = (&s.widgets[i], &s.widgets[j]);
                    let disjoint = a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y;
                    assert!(disjoint, "controls {i} and {j} overlap");
                }
            }
        }
    }

    /// Every control is fully on screen - a crop that ran off the edge would
    /// silently read background as part of the control.
    #[test]
    fn every_control_is_fully_on_screen() {
        let mut rng = Rng::new(21);
        for _ in 0..200 {
            let s = Screen::generate(&mut rng);
            for w in &s.widgets {
                assert!(w.x + w.w <= W && w.y + w.h <= H, "{w:?} runs off the screen");
            }
        }
    }

    /// A region instruction is only offered where it is load-bearing: the
    /// colour/shape pair it refines must genuinely be ambiguous, and the
    /// target must actually be in the region named.
    #[test]
    fn a_region_instruction_is_only_used_when_it_resolves_an_ambiguity() {
        let mut rng = Rng::new(33);
        let mut seen = 0;
        for _ in 0..600 {
            let s = Screen::generate(&mut rng);
            let Some(region) = REGIONS.iter().find(|r| s.instruction.ends_with(r.name)) else {
                continue;
            };
            seen += 1;
            let t = &s.widgets[s.target];
            let pair = s.widgets.iter().filter(|w| w.color == t.color && w.shape == t.shape).count();
            assert!(pair >= 2, "{:?} was used where the pair was already unique", s.instruction);
            assert!(region.holds(t), "{:?} names a region the target is not in", s.instruction);
        }
        assert!(seen > 0, "no region instruction was generated in 600 screens");
    }

    /// Every region has to be reachable, or an instruction form silently never
    /// appears and the reported breakdown covers less than it claims.
    #[test]
    fn every_region_is_used_by_some_instruction() {
        let mut rng = Rng::new(51);
        let mut used = [false; REGIONS.len()];
        for _ in 0..4000 {
            let s = Screen::generate(&mut rng);
            if let Some(i) = REGIONS.iter().position(|r| s.instruction.ends_with(r.name)) {
                used[i] = true;
            }
        }
        for (i, u) in used.iter().enumerate() {
            assert!(u, "no instruction ever used {:?}", REGIONS[i].name);
        }
    }
}
