// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The text reader: a strip of a text line is cut into 48x320 windows, each
//! window runs through a small convolutional stack and a gated transformer
//! to 80 frames of 307 CTC logits (label 0 is the blank), and the windows'
//! central frames are joined and greedily collapsed to text.

use backend_cpu::par;

use super::ops::{self, Linear, Map};
use super::preprocess::{resize_bilinear, ImageError, Resized};
use super::weights::Weights;
use crate::config::{MirageConfig, READER_CLASSES, READER_MIX_KERNEL, READER_STAGES};

/// Window height in pixels.
pub const WINDOW_H: usize = 48;
/// Window width in pixels.
pub const WINDOW_W: usize = 320;
/// Logit frames per window.
pub const FRAMES: usize = 80;
/// Horizontal pixels between consecutive windows.
pub const WINDOW_STEP: usize = 224;
/// Pixel value of the strip beyond its right edge.
pub const PAD_VALUE: u8 = 127;
/// Pixels per logit frame.
const FRAME_PX: usize = WINDOW_W / FRAMES;
/// Frames dropped at the left of every window after the first.
const LEFT_TRIM: usize = 12;
/// First frame excluded at the right of a window that is not the last.
const RIGHT_END: usize = 68;

struct Block {
    norm: Vec<f32>,
    mix: Vec<f32>,
    up: Linear,
    down: Linear,
}

/// The reader network.
pub struct Reader {
    dim: usize,
    w0: Vec<f32>,
    stage_norm: Vec<Vec<f32>>,
    dw: Vec<Vec<f32>>,
    pw: Vec<Linear>,
    blocks: Vec<Block>,
    final_norm: Vec<f32>,
    out: Linear,
    bias: Vec<f32>,
}

impl Reader {
    /// Prepare from the checkpoint tensors.
    pub fn new(cfg: &MirageConfig, w: &Weights) -> Reader {
        Reader {
            dim: cfg.reader_dim,
            w0: w.vec("reader/w0"),
            stage_norm: (0..=READER_STAGES.len()).map(|k| w.vec(&format!("reader/stage_norm_{k}/scale"))).collect(),
            dw: (0..READER_STAGES.len()).map(|k| w.vec(&format!("reader/dw_{k}"))).collect(),
            pw: (0..READER_STAGES.len()).map(|k| w.linear(&format!("reader/pw_{k}/kernel"))).collect(),
            blocks: (0..cfg.reader_blocks)
                .map(|i| Block {
                    norm: w.vec(&format!("reader/norm_{i}/scale")),
                    mix: w.vec(&format!("reader/mix_{i}")),
                    up: w.linear(&format!("reader/up_{i}/kernel")),
                    down: w.linear(&format!("reader/down_{i}/kernel")),
                })
                .collect(),
            final_norm: w.vec("reader/final_norm/scale"),
            out: w.linear("reader/out/kernel"),
            bias: w.vec("reader/out/bias"),
        }
    }

    /// Logits `[80, 307]` for one `48 x 320` RGB window.
    pub fn forward(&self, window: &[u8]) -> Vec<f32> {
        assert_eq!(window.len(), WINDOW_H * WINDOW_W * 3, "a reader window is {WINDOW_W}x{WINDOW_H} RGB");
        let x: Vec<f32> = window.iter().map(|&p| f32::from(p) / 127.5 - 1.0).collect();
        let (mut x, mut m) = ops::conv_dense(&x, Map { h: WINDOW_H, w: WINDOW_W, c: 3 }, &self.w0, 3, 3, READER_STAGES[0], 2, 2);
        x = ops::zc_norm(&x, m.c, &self.stage_norm[0]);
        ops::silu_inplace(&mut x);
        let strides = [(2, 2), (2, 1), (2, 1)];
        for (i, &(sy, sx)) in strides.iter().enumerate() {
            let (d, dm) = ops::depthwise(&x, m, &self.dw[i], 3, 3, sy, sx);
            let out_c = self.pw[i].out;
            x = self.pw[i].apply(&d, dm.h * dm.w);
            m = Map { h: dm.h, w: dm.w, c: out_c };
            x = ops::zc_norm(&x, m.c, &self.stage_norm[i + 1]);
            ops::silu_inplace(&mut x);
        }
        // Mean over the remaining rows leaves one feature vector per frame.
        let mut seq = vec![0f32; m.w * m.c];
        for row in x.chunks_exact(m.w * m.c) {
            seq.iter_mut().zip(row).for_each(|(a, b)| *a += b / m.h as f32);
        }
        let (frames, d) = (m.w, self.dim);
        for b in &self.blocks {
            let n = ops::zc_norm(&seq, d, &b.norm);
            let (h, _) = ops::depthwise(&n, Map { h: 1, w: frames, c: d }, &b.mix, 1, READER_MIX_KERNEL, 1, 1);
            let mut uv = b.up.apply(&h, frames);
            let mut gated = vec![0f32; frames * d];
            par::rows_mut(&mut gated, d, |f, row| {
                let r = &uv[f * 2 * d..(f + 1) * 2 * d];
                for c in 0..d {
                    row[c] = r[c] * model::hostmath::silu(r[d + c]);
                }
            });
            uv.clear();
            ops::add_inplace(&mut seq, &b.down.apply(&gated, frames));
        }
        let n = ops::zc_norm(&seq, d, &self.final_norm);
        self.out.apply_bias(&n, frames, &self.bias)
    }
}

/// Pixel crop `(x0, y0, x1, y1)` of a text box given in rounded thousandths of
/// the original image: floor for the near edges, ceil for the far ones.
pub fn crop_rect(bbox: [f32; 4], width: usize, height: usize) -> (usize, usize, usize, usize) {
    let (w, h) = (width as f64, height as f64);
    let lo = |v: f32, extent: f64| ((f64::from(v) * extent / 1000.0).floor().max(0.0) as usize).min(extent as usize);
    let hi = |v: f32, extent: f64| ((f64::from(v) * extent / 1000.0).ceil().max(0.0) as usize).min(extent as usize);
    (lo(bbox[0], w), lo(bbox[1], h), hi(bbox[2], w), hi(bbox[3], h))
}

/// Resize a crop to the reader's height, keeping its aspect ratio.
pub fn strip_from_crop(rgb: &[u8], h: usize, w: usize) -> Result<Resized, ImageError> {
    let width = ((WINDOW_H as f64 * w as f64 / h as f64).round_ties_even() as usize).max(1);
    resize_bilinear(rgb, h, w, WINDOW_H, width)
}

/// The windows of a strip: 320 wide, every 224 pixels, padded with 127.
pub fn windows(strip: &Resized) -> Vec<Vec<u8>> {
    let count = if strip.w <= WINDOW_W { 1 } else { (strip.w - WINDOW_W).div_ceil(WINDOW_STEP) + 1 };
    (0..count)
        .map(|j| {
            let x0 = j * WINDOW_STEP;
            let mut win = vec![PAD_VALUE; WINDOW_H * WINDOW_W * 3];
            for y in 0..WINDOW_H {
                let n = strip.w.saturating_sub(x0).min(WINDOW_W);
                win[y * WINDOW_W * 3..][..n * 3].copy_from_slice(&strip.rgb[(y * strip.w + x0) * 3..][..n * 3]);
            }
            win
        })
        .collect()
}

fn argmax(row: &[f32]) -> usize {
    row.iter().enumerate().fold((0, f32::NEG_INFINITY), |best, (i, &v)| if v > best.1 { (i, v) } else { best }).0
}

/// Join the windows' frames and collapse the greedy label path to text.
///
/// Window `j` starts at pixel `224 j`; it contributes frames from 0 (first
/// window) or 12, up to the strip's end when the window reaches it, else 68.
/// A label is emitted when it is not blank and differs from the previous
/// frame's label (a blank frame resets the previous label).
pub fn decode_windows(logits: &[Vec<f32>], strip_w: usize, charset: &[char]) -> String {
    let mut text = String::new();
    let mut prev = 0usize;
    for (j, lg) in logits.iter().enumerate() {
        let x0 = j * WINDOW_STEP;
        let lo = if x0 == 0 { 0 } else { LEFT_TRIM };
        let hi = if x0 + WINDOW_W >= strip_w { (strip_w - x0).div_ceil(FRAME_PX) } else { RIGHT_END };
        for f in lo..hi.min(FRAMES) {
            let label = argmax(&lg[f * READER_CLASSES..(f + 1) * READER_CLASSES]);
            if label != 0 && label != prev {
                if let Some(&c) = charset.get(label - 1) {
                    text.push(c);
                }
            }
            prev = label;
        }
    }
    text
}

impl Reader {
    /// Read the text inside `rect` (`(x0, y0, x1, y1)` pixels, see [`crop_rect`])
    /// of an `h x w` RGB image.
    pub fn read_line(&self, rgb: &[u8], h: usize, w: usize, rect: (usize, usize, usize, usize), charset: &[char]) -> Result<String, ImageError> {
        let (x0, y0, x1, y1) = rect;
        if x1 <= x0 || y1 <= y0 || x1 > w || y1 > h || rgb.len() != h * w * 3 {
            return Err(ImageError(format!("crop {rect:?} is empty or outside the {w}x{h} image")));
        }
        let (cw, ch) = (x1 - x0, y1 - y0);
        let mut crop = Vec::with_capacity(cw * ch * 3);
        for y in y0..y1 {
            crop.extend_from_slice(&rgb[(y * w + x0) * 3..(y * w + x1) * 3]);
        }
        let strip = strip_from_crop(&crop, ch, cw)?;
        let logits: Vec<Vec<f32>> = windows(&strip).iter().map(|win| self.forward(win)).collect();
        Ok(decode_windows(&logits, strip.w, charset))
    }
}
