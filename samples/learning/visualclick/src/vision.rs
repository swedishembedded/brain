// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the model sees: one control's appearance, read out of the RENDERED
//! PIXELS.
//!
//! Every number this module produces comes from [`Canvas::pixels`] - the same
//! bytes that get written to the PNG a reader can open. Nothing here consults
//! the [`crate::screen::Widget`] struct for what colour or shape it is; the
//! box is used to say WHERE to look, and the answer to WHAT IS THERE comes
//! from the image. `mean_crop_color_matches_what_was_drawn` pins that down, so
//! a future edit cannot quietly start feeding the model the scene description
//! it is supposed to be reading off the screen.
//!
//! A control becomes `GRID x GRID` mean-pooled RGB cells plus its normalised
//! box. The pooling is deliberately coarse: it keeps the three things an
//! instruction can ask about - which colour (the cells' hue), which shape
//! (WHERE the ink sits inside the box) and whether it is dimmed (how far the
//! ink is blended toward the panel behind it) - and keeps the projector small
//! enough to run in well under a millisecond for a whole screen.

use crate::screen::{Widget, H, W};
use brain::viewport::Canvas;

pub const GRID: usize = 8;
pub const CROP_DIMS: usize = GRID * GRID * 3;
const BBOX_DIMS: usize = 6;

/// Frequencies for the positional encoding, in half-cycles across the screen.
///
/// `0.5` is the important one and the reason this is not just the raw box: over
/// `[0, 1]` a half-cycle sine is monotone, so it is a smooth position channel
/// the head can compare against a threshold or against another control's. The
/// higher frequencies give the same information at finer resolution, which is
/// what separates two controls sitting close together.
const FREQS: [f32; 5] = [0.5, 1.0, 2.0, 4.0, 8.0];
const POS_DIMS: usize = BBOX_DIMS + 4 * FREQS.len();

/// One control's feature vector: `GRID x GRID` RGB cells, then its position.
///
/// The position block is deliberately WIDE - 26 dimensions for what is
/// geometrically two numbers - and `project::Grounder` gives it its own
/// projection rather than letting it share one with the 192 dimensions of
/// appearance. Both exist for the same reason: a projection spreads its output
/// across the dimensions it is given, so position competing head-on with
/// appearance in one map arrives as a whisper. See `Grounder::state_pos` for
/// what that measured.
///
/// The box itself is what a detector or an accessibility tree supplies in a
/// real deployment - it says where to look, never what is there.
pub const FEAT_DIM: usize = CROP_DIMS + POS_DIMS;

/// Mean-pool the control's rectangle into `GRID x GRID` RGB cells, read
/// straight out of the canvas.
pub fn widget_features(canvas: &Canvas, wgt: &Widget) -> Vec<f32> {
    assert!(wgt.x + wgt.w <= W && wgt.y + wgt.h <= H, "control {wgt:?} is not fully on screen");
    let px = canvas.pixels();
    let mut out = Vec::with_capacity(FEAT_DIM);

    for gy in 0..GRID {
        for gx in 0..GRID {
            // Half-open cell bounds derived from the box, so every pixel of
            // the control belongs to exactly one cell and none is counted
            // twice or dropped.
            let x0 = wgt.x + (wgt.w as usize * gx / GRID) as u32;
            let x1 = wgt.x + (wgt.w as usize * (gx + 1) / GRID) as u32;
            let y0 = wgt.y + (wgt.h as usize * gy / GRID) as u32;
            let y1 = wgt.y + (wgt.h as usize * (gy + 1) / GRID) as u32;
            let (mut r, mut g, mut b, mut n) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for y in y0..y1.max(y0 + 1) {
                for x in x0..x1.max(x0 + 1) {
                    let i = ((y * W + x) * 3) as usize;
                    r += px[i] as f32;
                    g += px[i + 1] as f32;
                    b += px[i + 2] as f32;
                    n += 1.0;
                }
            }
            out.push(r / n / 255.0);
            out.push(g / n / 255.0);
            out.push(b / n / 255.0);
        }
    }

    let (cx, cy) = ((wgt.x as f32 + wgt.w as f32 / 2.0) / W as f32, (wgt.y as f32 + wgt.h as f32 / 2.0) / H as f32);
    out.push(wgt.x as f32 / W as f32);
    out.push(wgt.y as f32 / H as f32);
    out.push(wgt.w as f32 / W as f32);
    out.push(wgt.h as f32 / H as f32);
    out.push(cx);
    out.push(cy);
    for f in FREQS {
        out.push((std::f32::consts::PI * f * cx).sin());
        out.push((std::f32::consts::PI * f * cx).cos());
        out.push((std::f32::consts::PI * f * cy).sin());
        out.push((std::f32::consts::PI * f * cy).cos());
    }
    debug_assert_eq!(out.len(), FEAT_DIM);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screen::{Rng, Screen, COLORS, SHAPES};

    /// The features carry what was DRAWN, not what the struct says - this is
    /// the test that fails the moment someone starts sourcing the model's
    /// input from the scene description instead of the screen.
    ///
    /// Checked on a filled square, where every pooled cell is the control's
    /// own colour, so the comparison needs no tolerance for shape coverage.
    #[test]
    fn mean_crop_color_matches_what_was_drawn() {
        let mut rng = Rng::new(2);
        let square = SHAPES.iter().position(|s| *s == "square").expect("square exists");
        let mut checked = 0;
        for _ in 0..200 {
            let s = Screen::generate(&mut rng);
            let canvas = s.render();
            for wgt in s.widgets.iter().filter(|w| w.shape == square && !w.dim) {
                let f = widget_features(&canvas, wgt);
                let want = COLORS[wgt.color].1;
                for cell in 0..GRID * GRID {
                    for ch in 0..3 {
                        let got = f[cell * 3 + ch] * 255.0;
                        assert!(
                            (got - want[ch] as f32).abs() < 2.0,
                            "cell {cell} channel {ch}: crop says {got:.1}, {} was drawn as {}",
                            COLORS[wgt.color].0,
                            want[ch]
                        );
                    }
                }
                checked += 1;
            }
        }
        assert!(checked > 0, "no bright square appeared in 200 screens");
    }

    /// A dimmed control really is dimmer in the pixels - the state an
    /// instruction can refer to has to be visible, not just recorded.
    #[test]
    fn a_dimmed_control_is_closer_to_the_background_than_a_bright_one() {
        let mut rng = Rng::new(6);
        let square = SHAPES.iter().position(|s| *s == "square").expect("square exists");
        let mut pairs = 0;
        for _ in 0..400 {
            let s = Screen::generate(&mut rng);
            let canvas = s.render();
            // Compare like with like: same colour, same shape, same region.
            for color in 0..COLORS.len() {
                let of = |dim: bool| {
                    s.widgets
                        .iter()
                        .find(|w| w.shape == square && w.color == color && w.dim == dim && w.y >= crate::screen::H / 4)
                        .map(|w| widget_features(&canvas, w))
                };
                if let (Some(bright), Some(dim)) = (of(false), of(true)) {
                    let spread = |f: &[f32]| -> f32 {
                        let mean: f32 = f[..GRID * GRID * 3].iter().sum::<f32>() / (GRID * GRID * 3) as f32;
                        f[..GRID * GRID * 3].iter().map(|v| (v - mean).abs()).sum::<f32>()
                    };
                    assert!(spread(&dim) < spread(&bright), "a dimmed control should sit closer to its background");
                    pairs += 1;
                }
            }
        }
        assert!(pairs > 0, "no bright/dimmed pair of the same colour appeared");
    }

    /// The position block carries a MONOTONE channel in each axis.
    ///
    /// Which area of the interface a control is in is a question about where
    /// it sits relative to a boundary, so position has to be readable as a
    /// magnitude rather than only as an identity. A purely high-frequency
    /// encoding would tell every control apart perfectly and still leave
    /// "above this line" unrepresentable.
    #[test]
    fn the_position_block_is_monotone_in_each_axis() {
        let canvas = Canvas::new(W, H);
        let at = |x: u32, y: u32| widget_features(&canvas, &Widget { shape: 0, color: 0, dim: false, x, y, w: 30, h: 30 });

        // The half-cycle sine, first of FREQS, for x and for y.
        let sin_x = CROP_DIMS + BBOX_DIMS;
        let sin_y = sin_x + 2;
        for k in 1..8u32 {
            let (lo, hi) = (at((k - 1) * 30, 40), at(k * 30, 40));
            assert!(hi[sin_x] > lo[sin_x], "the x position channel did not rise from column {} to {k}", k - 1);
            let (lo, hi) = (at(40, (k - 1) * 25), at(40, k * 25));
            assert!(hi[sin_y] > lo[sin_y], "the y position channel did not rise from row {} to {k}", k - 1);
        }
    }

    /// Two controls that differ only in WHERE they are must differ mostly in
    /// the position block, or the only thing distinguishing them is buried
    /// under the appearance they share.
    #[test]
    fn position_dominates_the_difference_between_two_identically_drawn_controls() {
        let canvas = Canvas::new(W, H);
        let a = widget_features(&canvas, &Widget { shape: 0, color: 0, dim: false, x: 20, y: 40, w: 30, h: 30 });
        let b = widget_features(&canvas, &Widget { shape: 0, color: 0, dim: false, x: 200, y: 40, w: 30, h: 30 });
        let appearance: f32 = (0..CROP_DIMS).map(|i| (a[i] - b[i]).abs()).sum();
        let position: f32 = (CROP_DIMS..FEAT_DIM).map(|i| (a[i] - b[i]).abs()).sum();
        assert!(position > appearance, "position moved by {position:.3} but appearance by {appearance:.3}");
    }

    /// The five shapes are genuinely different after pooling. If two shapes
    /// pooled to the same thing, an instruction naming one of them would be
    /// unanswerable from the pixels no matter how good the model was.
    #[test]
    fn every_shape_pools_to_a_distinguishable_pattern() {
        let mut canvas = Canvas::new(W, H);
        let ink = |f: &[f32]| -> Vec<f32> {
            // How dark each cell is - the shape's footprint, independent of hue.
            (0..GRID * GRID).map(|c| 1.0 - (f[c * 3] + f[c * 3 + 1] + f[c * 3 + 2]) / 3.0).collect()
        };
        let mut prints = Vec::new();
        for shape in 0..SHAPES.len() {
            canvas.clear(crate::screen::PANEL);
            let wgt = Widget { shape, color: 0, dim: false, x: 100, y: 100, w: 44, h: 44 };
            let s = Screen { widgets: vec![wgt], target: 0, instruction: String::new() };
            let c = s.render();
            prints.push(ink(&widget_features(&c, &wgt)));
        }
        for i in 0..prints.len() {
            for j in (i + 1)..prints.len() {
                let d: f32 = prints[i].iter().zip(&prints[j]).map(|(a, b)| (a - b).abs()).sum();
                assert!(d > 1.0, "{} and {} pool to nearly the same footprint (L1 {d:.3})", SHAPES[i], SHAPES[j]);
            }
        }
    }
}
