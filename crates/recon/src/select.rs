// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which frames are worth a slot in a pass.
//!
//! Two things decide it, and neither is the frame index:
//!
//! * SHARPNESS. A frame smeared by camera motion contributes smeared
//!   geometry, and no later stage takes it back out - it is baked into the
//!   depth the model predicts.
//! * VIEWPOINT. Two frames from nearly the same place cost two slots and say
//!   one thing. On a model whose cost grows faster than linearly in frames,
//!   that is the most expensive kind of nothing.
//!
//! Uniform spread over the capture - every k-th frame - gets both wrong: it
//! keeps a blurred frame because of where it sat in the sequence, and it keeps
//! both halves of a pause while the photographer stood still.

use imaging::Rgb8;

use crate::Frame;

/// Downsampled grid the frame-comparison statistic works on. Small on
/// purpose: what decides "same viewpoint" is the coarse layout of the
/// picture, and the detail is exactly what sensor noise and compression move
/// around between two shots of one wall.
const CHANGE_GRID: usize = 32;

fn luma(img: &Rgb8) -> Vec<f32> {
    (0..(img.w as usize * img.h as usize))
        .map(|i| {
            let p = &img.px[i * 3..i * 3 + 3];
            (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0
        })
        .collect()
}

/// Variance of the Laplacian of the luma - the standard cheap sharpness
/// measure. Camera motion during the exposure averages the high frequencies
/// away, and this is what responds to that.
///
/// It is NOT comparable between captures: the value scales with the picture's
/// own contrast, so a flat, perfectly focused wall scores below a contrasty
/// blurred street. [`select`] therefore only ever compares it WITHIN one
/// capture, against that capture's own median.
pub fn sharpness(img: &Rgb8) -> f32 {
    let (w, h) = (img.w as usize, img.h as usize);
    if w < 3 || h < 3 {
        return 0.0;
    }
    let y = luma(img);
    let mut vals = Vec::with_capacity((w - 2) * (h - 2));
    for j in 1..h - 1 {
        for i in 1..w - 1 {
            let k = j * w + i;
            vals.push(4.0 * y[k] - y[k - 1] - y[k + 1] - y[k - w] - y[k + w]);
        }
    }
    let n = vals.len() as f32;
    let mean = vals.iter().sum::<f32>() / n;
    vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n
}

/// Coarse, contrast-normalised luma. Fixed grid, so two frames of different
/// sizes - a still and a video frame - are still comparable.
fn thumbnail(img: &Rgb8) -> Vec<f32> {
    let (w, h) = (img.w as usize, img.h as usize);
    let y = luma(img);
    let mut t = vec![0.0f32; CHANGE_GRID * CHANGE_GRID];
    for (c, cell) in t.iter_mut().enumerate() {
        let (gx, gy) = (c % CHANGE_GRID, c / CHANGE_GRID);
        let (x0, y0) = (gx * w / CHANGE_GRID, gy * h / CHANGE_GRID);
        let x1 = (((gx + 1) * w) / CHANGE_GRID).clamp(x0 + 1, w);
        let y1 = (((gy + 1) * h) / CHANGE_GRID).clamp(y0 + 1, h);
        let mut sum = 0.0;
        let mut n = 0.0;
        for yy in y0..y1 {
            for xx in x0..x1 {
                sum += y[yy * w + xx];
                n += 1.0;
            }
        }
        *cell = if n > 0.0 { sum / n } else { 0.0 };
    }
    // Contrast-normalise, so a change of exposure between two shots of the
    // same wall does not read as a change of viewpoint.
    let n = t.len() as f32;
    let mean = t.iter().sum::<f32>() / n;
    let var = t.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let sd = var.sqrt().max(1e-6);
    t.iter().map(|v| (v - mean) / sd).collect()
}

/// How much the PICTURE changed between two frames: the RMS difference of
/// their contrast-normalised coarse luma. `0` is the same picture; two
/// unrelated views land near `sqrt(2)`.
///
/// **This is not parallax, and the difference matters.** Parallax is
/// triangulation baseline - how far the camera translated relative to how far
/// away the scene is - and no image-space statistic recovers it without
/// correspondences. What this measures is apparent image change, which a pure
/// pan (large change, no baseline at all) and a dolly straight down a corridor
/// (small change, real baseline) get wrong in opposite directions.
///
/// It is therefore used for exactly the one case where it IS reliable: a value
/// near zero means the two frames are very nearly the same picture, and two
/// nearly identical pictures cannot have a useful baseline between them
/// whatever the camera did. It rejects near-duplicates, and it is never used
/// to rank one pair as having "more parallax" than another.
pub fn viewpoint_change(a: &Rgb8, b: &Rgb8) -> f32 {
    let (ta, tb) = (thumbnail(a), thumbnail(b));
    let n = ta.len() as f32;
    (ta.iter().zip(&tb).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / n).sqrt()
}

/// Why a frame never reached the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dropped {
    /// Below `blur_ratio` of the capture's median sharpness.
    Blurred,
    /// Too nearly the same picture as the frame kept before it.
    NearDuplicate,
    /// Thinned out to fit the frame budget.
    OverBudget,
}

/// What the caller is willing to spend, and what it refuses to spend it on.
#[derive(Clone, Copy, Debug)]
pub struct SelectOpts {
    /// Keep a frame only if its sharpness is at least this fraction of the
    /// capture's median. `0` disables the blur gate.
    pub blur_ratio: f32,
    /// Drop a frame whose [`viewpoint_change`] against the last KEPT frame is
    /// below this. `0` disables near-duplicate rejection.
    pub min_change: f32,
    /// Hard ceiling on how many frames are reconstructed at all. `0` is
    /// unbounded - a long capture is handled by chunking, not by throwing
    /// frames away, so this is a cost choice and not a correctness one.
    pub budget: usize,
}

impl Default for SelectOpts {
    fn default() -> SelectOpts {
        SelectOpts { blur_ratio: 0.5, min_change: 0.05, budget: 0 }
    }
}

/// The frames that will be reconstructed, and the fate of the ones that will
/// not.
#[derive(Clone, Debug)]
pub struct Selection {
    /// Indices into the input frames, ascending.
    pub keep: Vec<usize>,
    pub sharpness: Vec<f32>,
    /// Change against the frame before it; `0` for the first.
    pub change: Vec<f32>,
    pub dropped: Vec<(usize, Dropped)>,
}

impl Selection {
    /// The kept frames themselves.
    pub fn frames(&self, all: &[Frame]) -> Vec<Frame> {
        self.keep.iter().map(|&i| all[i].clone()).collect()
    }
}

/// Choose the frames worth reconstructing.
///
/// Drop what is blurred relative to this capture, then walk the sequence
/// keeping a frame only once the picture has actually changed since the last
/// one kept - and when two frames are the same viewpoint, keep the SHARPER of
/// them rather than the earlier one.
///
/// The near-duplicate test is against the last frame kept, not against the
/// previous input frame: a slow pan compared pairwise would creep past the
/// threshold one frame at a time and keep every frame of it.
///
/// Never returns an empty selection for a non-empty capture.
pub fn select(frames: &[Frame], opts: &SelectOpts) -> Selection {
    let sharp: Vec<f32> = frames.iter().map(|f| sharpness(&f.image)).collect();
    let change: Vec<f32> = frames
        .iter()
        .enumerate()
        .map(|(i, f)| if i == 0 { 0.0 } else { viewpoint_change(&frames[i - 1].image, &f.image) })
        .collect();
    let mut dropped: Vec<(usize, Dropped)> = Vec::new();
    if frames.is_empty() {
        return Selection { keep: Vec::new(), sharpness: sharp, change, dropped };
    }

    // 1. the blur gate, relative to this capture's own median.
    let mut sorted = sharp.clone();
    sorted.sort_by(f32::total_cmp);
    let floor = opts.blur_ratio * sorted[sorted.len() / 2];
    let mut keep: Vec<usize> = (0..frames.len())
        .filter(|&i| {
            let ok = opts.blur_ratio <= 0.0 || sharp[i] >= floor;
            if !ok {
                dropped.push((i, Dropped::Blurred));
            }
            ok
        })
        .collect();
    if keep.is_empty() {
        // Everything failed the gate, which means the gate learned nothing.
        // The sharpest frame is still the best available one.
        let best = (0..frames.len()).max_by(|&a, &b| sharp[a].total_cmp(&sharp[b])).unwrap_or(0);
        keep.push(best);
        dropped.retain(|(i, _)| *i != best);
    }

    // 2. near-duplicate suppression.
    if opts.min_change > 0.0 {
        let mut kept: Vec<usize> = vec![keep[0]];
        for &i in &keep[1..] {
            let last = *kept.last().expect("non-empty");
            if viewpoint_change(&frames[last].image, &frames[i].image) >= opts.min_change {
                kept.push(i);
            } else if sharp[i] > sharp[last] {
                kept.pop();
                kept.push(i);
                dropped.push((last, Dropped::NearDuplicate));
            } else {
                dropped.push((i, Dropped::NearDuplicate));
            }
        }
        keep = kept;
    }

    // 3. the budget, SPREAD over the whole capture. A cap that truncated would
    // turn a full orbit into an arc, which reconstructs far worse than the
    // same number of frames spanning all of it.
    if opts.budget > 0 && keep.len() > opts.budget {
        let n = keep.len();
        let thinned: Vec<usize> =
            (0..opts.budget).map(|i| keep[i * (n - 1) / (opts.budget - 1).max(1)]).collect();
        for &i in &keep {
            if !thinned.contains(&i) {
                dropped.push((i, Dropped::OverBudget));
            }
        }
        keep = thinned;
    }

    dropped.sort_by_key(|(i, _)| *i);
    Selection { keep, sharpness: sharp, change, dropped }
}
