// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-object masks: the probability field on the 128x128 prototype grid and
//! its rendering to a run-length bitmap at the original image size.
//!
//! An object's coefficient vector is read at the cell holding its box centre
//! on the stride-8 map. Its field is `sigmoid(prototype . coefficient)` inside
//! the box and zero outside, plus the cell holding the box centre (always set,
//! even when it falls outside). A bitmap samples that field bilinearly at each
//! original pixel's position and thresholds it, restricted to the box.

use crate::config::MASK_RANK;

/// Side of the prototype grid.
pub const FIELD_SIDE: usize = 128;

/// Side of the coefficient map.
pub const COEF_SIDE: usize = 64;

/// A box rounded to integer thousandths of the canvas, as `[x0, y0, x1, y1]`
/// in canvas-normalised units.
pub fn round_box(b: [f32; 4]) -> [f32; 4] {
    b.map(|v| (v.clamp(0.0, 1.0) * 1000.0).round() * 0.001)
}

/// The stride-8 cell `(row, col)` whose coefficients describe a box.
pub fn coefficient_cell(b: [f32; 4]) -> (usize, usize) {
    let cell = |lo: f32, hi: f32| ((COEF_SIDE as f32 * (lo + hi) / 2.0) as usize).min(COEF_SIDE - 1);
    (cell(b[1], b[3]), cell(b[0], b[2]))
}

/// The mask probability field `[128, 128]` of one object.
pub fn object_field(protos: &[f32], coef: &[f32], b: [f32; 4]) -> Vec<f32> {
    assert_eq!(protos.len(), FIELD_SIDE * FIELD_SIDE * MASK_RANK);
    assert_eq!(coef.len(), MASK_RANK);
    let side = FIELD_SIDE as f32;
    let centre = |lo: f32, hi: f32| ((side * (lo + hi) / 2.0) as usize).min(FIELD_SIDE - 1);
    let (cr, cc) = (centre(b[1], b[3]), centre(b[0], b[2]));
    let mut field = vec![0f32; FIELD_SIDE * FIELD_SIDE];
    for r in 0..FIELD_SIDE {
        let y = (r as f32 + 0.5) / side;
        for c in 0..FIELD_SIDE {
            let x = (c as f32 + 0.5) / side;
            let inside = x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3];
            if inside || (r == cr && c == cc) {
                let p = &protos[(r * FIELD_SIDE + c) * MASK_RANK..][..MASK_RANK];
                let logit: f32 = p.iter().zip(coef).map(|(a, b)| a * b).sum();
                field[r * FIELD_SIDE + c] = 1.0 / (1.0 + (-logit).exp());
            }
        }
    }
    field
}

/// A column-major run-length mask: `counts` alternate background / foreground
/// runs starting with background (possibly of length zero).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rle {
    /// `[height, width]` of the original image.
    pub size: [usize; 2],
    /// Run lengths.
    pub counts: Vec<u32>,
}

/// Geometry of the resized image inside the canvas and of the original image.
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    /// Resized width on the canvas.
    pub resized_w: usize,
    /// Resized height on the canvas.
    pub resized_h: usize,
    /// Original width.
    pub width: usize,
    /// Original height.
    pub height: usize,
}

/// Render a field to the original pixel grid and run-length encode it.
pub fn render_bitmap(field: &[f32], b: [f32; 4], frame: Frame, threshold: f32) -> Rle {
    assert_eq!(field.len(), FIELD_SIDE * FIELD_SIDE);
    let (h, w) = (frame.height, frame.width);
    let canvas = crate::vision::preprocess::CANVAS as f32;
    let to_x = |v: f32| v * canvas / frame.resized_w as f32 * w as f32;
    let to_y = |v: f32| v * canvas / frame.resized_h as f32 * h as f32;
    let (x0, x1, y0, y1) = (to_x(b[0]), to_x(b[2]), to_y(b[1]), to_y(b[3]));
    let last = (FIELD_SIDE - 1) as f32;
    let grid = |pos: f32, extent_resized: usize, extent: usize| (((pos + 0.5) * extent_resized as f32 / extent as f32) / 4.0 - 0.5).clamp(0.0, last);
    // Sampling coordinates along each axis do not depend on the other axis.
    let gx: Vec<f32> = (0..w).map(|j| grid(j as f32, frame.resized_w, w)).collect();
    let gy: Vec<f32> = (0..h).map(|i| grid(i as f32, frame.resized_h, h)).collect();
    let at = |r: usize, c: usize| field[r * FIELD_SIDE + c];
    let mut counts = Vec::new();
    let (mut run, mut state) = (0u32, false);
    for (j, &fx) in gx.iter().enumerate() {
        let (c0, tx) = (fx.floor() as usize, fx - fx.floor());
        let c1 = (c0 + 1).min(FIELD_SIDE - 1);
        for (i, &fy) in gy.iter().enumerate() {
            let (r0, ty) = (fy.floor() as usize, fy - fy.floor());
            let r1 = (r0 + 1).min(FIELD_SIDE - 1);
            let v = (at(r0, c0) * (1.0 - tx) + at(r0, c1) * tx) * (1.0 - ty) + (at(r1, c0) * (1.0 - tx) + at(r1, c1) * tx) * ty;
            let set = v > threshold && (j as f32) >= x0 && (j as f32) <= x1 && (i as f32) >= y0 && (i as f32) <= y1;
            if set != state {
                counts.push(run);
                run = 0;
                state = set;
            }
            run += 1;
        }
    }
    counts.push(run);
    Rle { size: [h, w], counts }
}
