// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Procedural synthetic video clips for LoRA finetune gates: two small,
//! deterministic, clearly-distinguishable moving shapes, rendered with the
//! same `data::rng::Rng` convention `gen_pong`/`gen_detect` use.
//!
//! * **Concept**: a magenta triangle orbiting a fixed white dot on a black
//!   background - something a pretrained text-to-video prior does not already
//!   produce, so a LoRA trained on it has an unambiguous signature to prove.
//! * **Distractor**: a cyan square bouncing between two fixed walls - a
//!   different, easily distinguished motion/shape/colour, used as the
//!   held-out control a concept-only adapter should NOT improve on.
//!
//! Frames are HWC `f32` in `[0,1]`, `[w,h] = size,size]`, three channels -
//! [`crate::videoset::write_clipset`]'s expected input.

use crate::rng::Rng;

/// A handful of paraphrases per concept, so training and held-out-eval
/// prompts are never byte-identical strings.
pub const CONCEPT_CAPTIONS: &[&str] = &[
    "a magenta triangle orbiting a white dot on a black background",
    "a magenta triangular shape circling a small white dot on a black background",
    "a small magenta triangle moving in a circle around a white point",
];

pub const DISTRACTOR_CAPTIONS: &[&str] = &[
    "a cyan square bouncing between two fixed walls",
    "a cyan block bouncing back and forth between two vertical walls",
    "a small cyan square moving side to side between two barriers",
];

fn pick<'a>(rng: &mut Rng, items: &'a [&'a str]) -> &'a str {
    items[rng.gen_range_inclusive(0, items.len() as i64 - 1) as usize]
}

fn new_frame(w: u32, h: u32) -> Vec<f32> {
    vec![0.0f32; (w * h * 3) as usize]
}

fn put_px(buf: &mut [f32], w: u32, x: i32, y: i32, color: [f32; 3]) {
    if x < 0 || y < 0 {
        return;
    }
    let (x, y) = (x as u32, y as u32);
    let idx = ((y * w + x) * 3) as usize;
    if idx + 2 < buf.len() {
        buf[idx] = color[0];
        buf[idx + 1] = color[1];
        buf[idx + 2] = color[2];
    }
}

/// Fill an axis-aligned rectangle `rw x rh`, top-left at `(x0,y0)`.
fn fill_rect(buf: &mut [f32], w: u32, h: u32, x0: i32, y0: i32, rw: i32, rh: i32, color: [f32; 3]) {
    for y in y0.max(0)..(y0 + rh).min(h as i32) {
        for x in x0.max(0)..(x0 + rw).min(w as i32) {
            put_px(buf, w, x, y, color);
        }
    }
}

/// Fill an axis-aligned square of side `size`, top-left at `(x0,y0)`.
fn fill_square(buf: &mut [f32], w: u32, h: u32, x0: i32, y0: i32, size: i32, color: [f32; 3]) {
    fill_rect(buf, w, h, x0, y0, size, size, color);
}

/// Fill a small filled circle of radius `r` centered at `(cx,cy)`.
fn fill_circle(buf: &mut [f32], w: u32, h: u32, cx: f32, cy: f32, r: f32, color: [f32; 3]) {
    let (x0, x1) = ((cx - r).floor().max(0.0) as i32, (cx + r).ceil().min(w as f32) as i32);
    let (y0, y1) = ((cy - r).floor().max(0.0) as i32, (cy + r).ceil().min(h as f32) as i32);
    for y in y0..y1 {
        for x in x0..x1 {
            let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
            if dx * dx + dy * dy <= r * r {
                put_px(buf, w, x, y, color);
            }
        }
    }
}

/// Fill a triangle centered at `(cx,cy)`, circumradius `r`, rotated `rot`
/// radians - a standard sign-of-cross-product rasterizer over the bounding box.
fn fill_triangle(buf: &mut [f32], w: u32, h: u32, cx: f32, cy: f32, r: f32, rot: f32, color: [f32; 3]) {
    let tau3 = std::f32::consts::TAU / 3.0;
    let verts: [(f32, f32); 3] =
        std::array::from_fn(|k| (cx + r * (rot + tau3 * k as f32).cos(), cy + r * (rot + tau3 * k as f32).sin()));
    let sign = |p: (f32, f32), a: (f32, f32), b: (f32, f32)| (p.0 - b.0) * (a.1 - b.1) - (a.0 - b.0) * (p.1 - b.1);
    let minx = verts.iter().map(|p| p.0).fold(f32::MAX, f32::min).floor().max(0.0) as i32;
    let maxx = verts.iter().map(|p| p.0).fold(f32::MIN, f32::max).ceil().min(w as f32) as i32;
    let miny = verts.iter().map(|p| p.1).fold(f32::MAX, f32::min).floor().max(0.0) as i32;
    let maxy = verts.iter().map(|p| p.1).fold(f32::MIN, f32::max).ceil().min(h as f32) as i32;
    for y in miny..maxy {
        for x in minx..maxx {
            let p = (x as f32 + 0.5, y as f32 + 0.5);
            let (d1, d2, d3) = (sign(p, verts[0], verts[1]), sign(p, verts[1], verts[2]), sign(p, verts[2], verts[0]));
            let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
            let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
            if !(has_neg && has_pos) {
                put_px(buf, w, x, y, color);
            }
        }
    }
}

/// One concept clip: `frames` HWC `f32` `[0,1]` frames of a magenta triangle
/// orbiting a fixed white dot, with a per-clip random phase/spin/direction so
/// clips are visibly distinct while the concept stays fixed.
pub fn concept_clip(rng: &mut Rng, frames: usize, w: u32, h: u32) -> Vec<Vec<f32>> {
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let orbit_r = w.min(h) as f64 * 0.28;
    let tri_r = w.min(h) as f32 * 0.10;
    let dot_r = w.min(h) as f32 * 0.02;
    let phase = rng.uniform(0.0, std::f64::consts::TAU);
    let dir = if rng.gen_range_inclusive(0, 1) == 0 { -1.0 } else { 1.0 };
    let turns = rng.uniform(0.6, 1.4); // how much of a full orbit this clip covers
    let spin = dir * rng.uniform(0.5, 2.0);
    (0..frames)
        .map(|t| {
            let mut buf = new_frame(w, h);
            let frac = if frames <= 1 { 0.0 } else { t as f64 / (frames - 1) as f64 };
            let ang = phase + dir * std::f64::consts::TAU * turns * frac;
            let (px, py) = (cx + orbit_r * ang.cos(), cy + orbit_r * ang.sin());
            fill_triangle(&mut buf, w, h, px as f32, py as f32, tri_r, (spin * ang) as f32, [1.0, 0.0, 1.0]);
            fill_circle(&mut buf, w, h, cx as f32, cy as f32, dot_r, [1.0, 1.0, 1.0]);
            buf
        })
        .collect()
}

/// One distractor clip: `frames` HWC `f32` `[0,1]` frames of a cyan square
/// bouncing between two fixed vertical walls, per-clip random speed/phase.
pub fn distractor_clip(rng: &mut Rng, frames: usize, w: u32, h: u32) -> Vec<Vec<f32>> {
    let size = (w.min(h) as f32 * 0.14).round();
    let wall_l = (w as f32 * 0.14).round();
    let wall_r = (w as f32 * 0.86).round();
    let span = (wall_r - wall_l - size).max(1.0);
    let y0 = ((h as f32 - size) / 2.0).round() as i32;
    let start = rng.uniform(0.0, 2.0) as f32;
    let speed = rng.uniform(0.12, 0.30) as f32;
    (0..frames)
        .map(|t| {
            let mut buf = new_frame(w, h);
            fill_rect(&mut buf, w, h, wall_l as i32 - 3, 0, 3, h as i32, [0.5, 0.5, 0.5]);
            fill_rect(&mut buf, w, h, wall_r as i32, 0, 3, h as i32, [0.5, 0.5, 0.5]);
            // Triangle-wave bounce: reflect the linear ramp at each wall.
            let raw = (start + speed * t as f32).rem_euclid(2.0);
            let frac = if raw <= 1.0 { raw } else { 2.0 - raw };
            let x0 = (wall_l + frac * span).round() as i32;
            fill_square(&mut buf, w, h, x0, y0, size as i32, [0.0, 1.0, 1.0]);
            buf
        })
        .collect()
}

/// `n_clips` concept clips, each `frames` frames of `w x h`, deterministic
/// from `seed`. Returns `(caption, frames)` pairs in generation order -
/// [`crate::videoset::write_clipset`]'s expected input.
pub fn generate_concept_set(n_clips: usize, frames: usize, w: u32, h: u32, seed: u64) -> Vec<(String, Vec<Vec<f32>>)> {
    let mut rng = Rng::new(seed);
    (0..n_clips).map(|_| (pick(&mut rng, CONCEPT_CAPTIONS).to_string(), concept_clip(&mut rng, frames, w, h))).collect()
}

/// `n_clips` distractor clips, each `frames` frames of `w x h`, deterministic
/// from `seed`.
pub fn generate_distractor_set(n_clips: usize, frames: usize, w: u32, h: u32, seed: u64) -> Vec<(String, Vec<Vec<f32>>)> {
    let mut rng = Rng::new(seed);
    (0..n_clips).map(|_| (pick(&mut rng, DISTRACTOR_CAPTIONS).to_string(), distractor_clip(&mut rng, frames, w, h))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concept_and_distractor_clips_are_deterministic_and_in_range() {
        let a = generate_concept_set(3, 9, 64, 64, 7);
        let b = generate_concept_set(3, 9, 64, 64, 7);
        assert_eq!(a.len(), b.len());
        for ((ca, fa), (cb, fb)) in a.iter().zip(&b) {
            assert_eq!(ca, cb, "same seed must give the same caption");
            assert_eq!(fa, fb, "same seed must give the same frames");
        }
        for (cap, frames) in &a {
            assert!(CONCEPT_CAPTIONS.contains(&cap.as_str()));
            assert_eq!(frames.len(), 9);
            for f in frames {
                assert_eq!(f.len(), 64 * 64 * 3);
                assert!(f.iter().all(|&v| (0.0..=1.0).contains(&v)));
            }
        }
    }

    #[test]
    fn concept_and_distractor_are_visibly_different_sets() {
        let concept = generate_concept_set(4, 9, 64, 64, 11);
        let distractor = generate_distractor_set(4, 9, 64, 64, 11);
        for (cap, _) in &concept {
            assert!(CONCEPT_CAPTIONS.contains(&cap.as_str()));
        }
        for (cap, _) in &distractor {
            assert!(DISTRACTOR_CAPTIONS.contains(&cap.as_str()));
        }
        // Different seeds within a set still vary (position/rotation).
        let c2 = generate_concept_set(4, 9, 64, 64, 12);
        assert_ne!(concept[0].1, c2[0].1);
        // The two shapes never share a colour: concept is magenta+white on
        // black, distractor is cyan+grey on black.
        let has_color = |frames: &[Vec<f32>], color: [f32; 3]| {
            frames.iter().any(|f| f.chunks(3).any(|px| (px[0] - color[0]).abs() < 1e-6 && (px[1] - color[1]).abs() < 1e-6 && (px[2] - color[2]).abs() < 1e-6))
        };
        assert!(has_color(&concept[0].1, [1.0, 0.0, 1.0]), "concept clips must show magenta");
        assert!(!has_color(&concept[0].1, [0.0, 1.0, 1.0]), "concept clips must never show cyan");
        assert!(has_color(&distractor[0].1, [0.0, 1.0, 1.0]), "distractor clips must show cyan");
        assert!(!has_color(&distractor[0].1, [1.0, 0.0, 1.0]), "distractor clips must never show magenta");
    }
}
