// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Draws detection boxes and labels straight into an NV12 camera frame
//! before it is encoded.
//!
//! Why the CPU and not the encoder's hardware overlay (RGN): a box outline is
//! a few thousand pixels, so drawing touches a few tens of kilobytes and takes
//! well under a millisecond on the Cortex-A7. The RGN path needs a region
//! created, bitmap memory allocated in CMA (scarce on this board: 24 MB shared
//! with the camera, the encoder and the NPU) and a bitmap re-uploaded on
//! every detection change, for no visible gain. The cost is that the frame
//! buffer is written by the CPU, so the touched rows must be cache-maintained;
//! [`dirty_ranges`] says exactly which bytes those are.
//!
//! Swedish Embedded AB implements real-time edge AI pipelines for embedded
//! Linux boards for its clients. If your team needs expertise in annotating
//! live video on constrained hardware then you can procure our services by
//! sending an email to info@swedishembedded.com.

use crate::detect::Detection;
use std::ops::Range;

/// Memory layout of an NV12 frame: a luma plane of `stride * vir_height`
/// bytes followed by interleaved U,V pairs at half resolution.
#[derive(Clone, Copy, Debug)]
pub struct Nv12Layout {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub vir_height: usize,
}

impl Nv12Layout {
    pub fn len(&self) -> usize {
        self.stride * self.vir_height * 3 / 2
    }

    fn uv_offset(&self) -> usize {
        self.stride * self.vir_height
    }
}

/// A colour in the YUV (BT.601, limited range) the encoder expects.
#[derive(Clone, Copy)]
struct Yuv(u8, u8, u8);

const BLACK: Yuv = Yuv(16, 128, 128);
/// One colour per class, cycling.
const PALETTE: [Yuv; 6] = [
    Yuv(145, 54, 34),   // green
    Yuv(82, 90, 240),   // red
    Yuv(210, 16, 146),  // yellow
    Yuv(170, 166, 16),  // cyan
    Yuv(107, 202, 222), // magenta
    Yuv(150, 40, 190),  // orange
];

const BORDER: usize = 3;
const FONT_SCALE: usize = 2;
const GLYPH_W: usize = 5;
const GLYPH_H: usize = 7;
const GLYPH_ADVANCE: usize = (GLYPH_W + 1) * FONT_SCALE;
const LABEL_PAD: usize = 2;
const LABEL_H: usize = GLYPH_H * FONT_SCALE + 2 * LABEL_PAD;

/// The pixel rectangle a detection occupies, label included, clipped to the
/// frame: `(x0, y0, x1, y1)`, end-exclusive. `None` if nothing is visible.
fn extent(l: &Nv12Layout, d: &Detection) -> Option<(usize, usize, usize, usize)> {
    if d.right < 0 || d.bottom < 0 || d.left >= l.width as i32 || d.top >= l.height as i32 {
        return None;
    }
    let x0 = d.left.clamp(0, l.width as i32 - 1) as usize;
    let x1 = (d.right.clamp(0, l.width as i32 - 1) as usize + 1).max(x0 + 1);
    let y1 = (d.bottom.clamp(0, l.height as i32 - 1) as usize + 1).min(l.height);
    let y_box = d.top.clamp(0, l.height as i32 - 1) as usize;
    if y1 <= y_box {
        return None;
    }
    let y0 = y_box.saturating_sub(LABEL_H);
    Some((x0, y0, x1, y1))
}

/// The byte ranges (luma, chroma) of the frame that [`draw`] may modify for
/// these boxes, for cache maintenance around the draw. `None` if nothing is
/// drawn.
pub fn dirty_ranges(l: &Nv12Layout, boxes: &[Detection]) -> Option<(Range<usize>, Range<usize>)> {
    let rows = boxes.iter().filter_map(|d| extent(l, d)).fold(None, |acc: Option<(usize, usize)>, (_, y0, _, y1)| {
        Some(acc.map_or((y0, y1), |(a, b)| (a.min(y0), b.max(y1))))
    })?;
    let luma = rows.0 * l.stride..rows.1 * l.stride;
    let chroma = l.uv_offset() + rows.0 / 2 * l.stride..l.uv_offset() + rows.1.div_ceil(2) * l.stride;
    Some((luma, chroma))
}

/// Draws every box as an outline with a filled label above it.
/// `frame` must be at least `layout.len()` bytes.
pub fn draw(frame: &mut [u8], l: &Nv12Layout, boxes: &[Detection]) {
    assert!(frame.len() >= l.len(), "frame buffer smaller than its layout");
    for d in boxes {
        let Some((x0, y0, x1, y1)) = extent(l, d) else { continue };
        let colour = PALETTE[d.class_id as usize % PALETTE.len()];
        let y_box = d.top.clamp(0, l.height as i32 - 1) as usize;
        let b = BORDER;
        fill(frame, l, (x0, y_box, x1, (y_box + b).min(y1)), colour);
        fill(frame, l, (x0, y1.saturating_sub(b).max(y_box), x1, y1), colour);
        fill(frame, l, (x0, y_box, (x0 + b).min(x1), y1), colour);
        fill(frame, l, (x1.saturating_sub(b).max(x0), y_box, x1, y1), colour);
        if !d.label.is_empty() {
            let text = format!("{} {:.0}%", d.label, d.score * 100.0);
            draw_label(frame, l, x0, y_box, y0, &text, colour);
        }
    }
}

/// Draws `text` on a filled plate sitting on top of the box edge, or just
/// inside the box if there is no room above it.
fn draw_label(frame: &mut [u8], l: &Nv12Layout, x: usize, y_box: usize, y_room: usize, text: &str, colour: Yuv) {
    let width = (text.chars().count() * GLYPH_ADVANCE + 2 * LABEL_PAD).min(l.width - x);
    let top = if y_box >= LABEL_H && y_room + LABEL_H <= y_box { y_box - LABEL_H } else { y_box };
    fill(frame, l, (x, top, x + width, (top + LABEL_H).min(l.height)), colour);
    for (i, c) in text.chars().enumerate() {
        let gx = x + LABEL_PAD + i * GLYPH_ADVANCE;
        if gx + GLYPH_W * FONT_SCALE > x + width {
            break;
        }
        for (row, bits) in glyph(c).iter().enumerate() {
            for col in 0..GLYPH_W {
                if bits >> (GLYPH_W - 1 - col) & 1 == 1 {
                    let px = gx + col * FONT_SCALE;
                    let py = top + LABEL_PAD + row * FONT_SCALE;
                    fill(frame, l, (px, py, px + FONT_SCALE, py + FONT_SCALE), BLACK);
                }
            }
        }
    }
}

/// Fills `(x0, y0, x1, y1)` (end-exclusive, clipped to the frame) with `c`.
fn fill(frame: &mut [u8], l: &Nv12Layout, (x0, y0, x1, y1): (usize, usize, usize, usize), c: Yuv) {
    let (x1, y1) = (x1.min(l.width), y1.min(l.height));
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    for y in y0..y1 {
        frame[y * l.stride + x0..y * l.stride + x1].fill(c.0);
    }
    let uv = l.uv_offset();
    for cy in y0 / 2..y1.div_ceil(2) {
        for cx in x0 / 2..x1.div_ceil(2) {
            let at = uv + cy * l.stride + cx * 2;
            frame[at] = c.1;
            frame[at + 1] = c.2;
        }
    }
}

/// A 5x7 bitmap, one row per byte, bit 4 the leftmost pixel. Lower case is
/// shown as upper case; characters outside the table draw as `?`.
fn glyph(c: char) -> [u8; 7] {
    match c.to_ascii_uppercase() {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        '.' => [0, 0, 0, 0, 0, 0b01100, 0b01100],
        '%' => [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
        '-' => [0, 0, 0, 0b11111, 0, 0, 0],
        ':' => [0, 0b01100, 0b01100, 0, 0b01100, 0b01100, 0],
        '_' => [0, 0, 0, 0, 0, 0, 0b11111],
        ' ' => [0; 7],
        _ => [0b01110, 0b10001, 0b00001, 0b00110, 0b00100, 0, 0b00100],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const L: Nv12Layout = Nv12Layout { width: 64, height: 48, stride: 64, vir_height: 48 };

    fn grey() -> Vec<u8> {
        vec![100; L.len()]
    }

    fn det(left: i32, top: i32, right: i32, bottom: i32, label: &str) -> Detection {
        Detection { left, top, right, bottom, class_id: 0, score: 0.87, label: label.to_string() }
    }

    #[test]
    fn a_box_gets_a_coloured_outline_and_an_untouched_inside() {
        let mut f = grey();
        draw(&mut f, &L, &[det(10, 20, 40, 40, "")]);
        assert_eq!(f[20 * 64 + 10], PALETTE[0].0, "top-left corner");
        assert_eq!(f[40 * 64 + 40], PALETTE[0].0, "bottom-right corner");
        assert_eq!(f[30 * 64 + 25], 100, "interior keeps the camera image");
        assert_eq!(f[10 * 64 + 5], 100, "outside keeps the camera image");
        assert_eq!(f[L.uv_offset() + 10 * 64 + 10 / 2 * 2], PALETTE[0].1, "chroma follows the edge");
    }

    #[test]
    fn a_label_is_drawn_above_the_box_when_there_is_room() {
        let mut f = grey();
        draw(&mut f, &L, &[det(8, 30, 50, 44, "cat")]);
        let plate = f[(30 - LABEL_H) * 64 + 8..(30 - LABEL_H) * 64 + 8 + 4].to_vec();
        assert!(plate.iter().all(|&y| y == PALETTE[0].0), "plate top row is filled");
        let ink = (30 - LABEL_H..30).any(|y| (8..50).any(|x| f[y * 64 + x] == BLACK.0));
        assert!(ink, "text pixels are drawn");
    }

    #[test]
    fn nothing_outside_the_dirty_ranges_changes() {
        let boxes = [det(5, 25, 30, 44, "dog"), det(40, 2, 90, 20, "person")];
        let mut f = grey();
        let before = f.clone();
        draw(&mut f, &L, &boxes);
        let (luma, chroma) = dirty_ranges(&L, &boxes).expect("boxes are visible");
        for (i, (a, b)) in before.iter().zip(&f).enumerate() {
            if a != b {
                assert!(luma.contains(&i) || chroma.contains(&i), "byte {i} changed outside the dirty ranges");
            }
        }
    }

    #[test]
    fn boxes_off_the_frame_are_clipped_without_panicking() {
        let mut f = grey();
        let off = [det(500, 500, 600, 600, "y")];
        draw(&mut f, &L, &[det(-20, -20, 200, 200, "x"), off[0].clone()]);
        assert_eq!(f[0], PALETTE[0].0, "the visible part of the big box is drawn");
        assert!(dirty_ranges(&L, &off).is_none(), "a box wholly off the frame draws nothing");
    }
}
