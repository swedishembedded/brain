// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `imageops` — deterministic, no-weights image utilities exposed through the
//! generalized capability interface. They prove the whole image-output path of
//! `brain do` end-to-end (typed params → an actual viewable image blob), and
//! provide the pieces an inpainting workflow needs (a rectangle mask) without any
//! neural model. Everything here is pure procedural math, so results are
//! bit-deterministic for a given set of params.

use std::sync::Arc;

use capability::{Action, ActionSpec, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use serde_json::json;

pub const MODEL: &str = "brain/imageops";

pub fn manifest() -> Manifest {
    let size = |s: ActionSpec| {
        s.param(ParamSpec::new("width", ParamType::Int, "image width, px").default(json!(512)))
            .param(ParamSpec::new("height", ParamType::Int, "image height, px").default(json!(512)))
    };
    let mask_rect = size(ActionSpec::new("mask_rect", "make a binary inpainting mask: white rectangle on black (white = regenerate)"))
        .param(ParamSpec::new("x", ParamType::Int, "rectangle left, px").default(json!(0)))
        .param(ParamSpec::new("y", ParamType::Int, "rectangle top, px").default(json!(0)))
        .param(ParamSpec::new("w", ParamType::Int, "rectangle width, px").required())
        .param(ParamSpec::new("h", ParamType::Int, "rectangle height, px").required())
        .output(BlobSpec::new("mask", Media::Mask, "the mask (white rect on black)"));

    let gradient = size(ActionSpec::new("gradient", "render a smooth procedural gradient image — a deterministic, viewable test image"))
        .param(ParamSpec::new("style", ParamType::Enum(vec!["sunset".into(), "ocean".into(), "aurora".into()]), "palette").default(json!("sunset")))
        .output(BlobSpec::new("image", Media::Image, "the rendered image"));

    Manifest::new(MODEL, "deterministic image utilities (mask + procedural renders) — no weights", vec![mask_rect, gradient])
}

pub struct ImageOps;

impl Provider for ImageOps {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match name {
            "mask_rect" => Some(Arc::new(MaskRect)),
            "gradient" => Some(Arc::new(Gradient)),
            _ => None,
        }
    }
}

struct MaskRect;
impl Action for MaskRect {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == "mask_rect").unwrap()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> capability::ActionResult {
        let (wd, ht) = (inv.get_i64("width").unwrap_or(512) as usize, inv.get_i64("height").unwrap_or(512) as usize);
        let (x, y) = (inv.get_i64("x").unwrap_or(0).max(0) as usize, inv.get_i64("y").unwrap_or(0).max(0) as usize);
        let (rw, rh) = (inv.get_i64("w").unwrap_or(0).max(0) as usize, inv.get_i64("h").unwrap_or(0).max(0) as usize);
        progress(Progress::step(1, 1, "drawing mask"));
        let mut m = vec![0f32; wd * ht]; // 1-channel
        for row in y..(y + rh).min(ht) {
            for col in x..(x + rw).min(wd) {
                m[row * wd + col] = 1.0;
            }
        }
        Ok(Outcome::new().set("white_px", json!(rw.min(wd) * rh.min(ht))).blob("mask", capability::blob::image_blob(&m, wd as u32, ht as u32, 1).with_media(Media::Mask)))
    }
}

struct Gradient;
impl Action for Gradient {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == "gradient").unwrap()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> capability::ActionResult {
        let (wd, ht) = (inv.get_i64("width").unwrap_or(512) as usize, inv.get_i64("height").unwrap_or(512) as usize);
        let style = inv.get_str("style").unwrap_or_else(|| "sunset".into());
        // Two-stop palette per style, blended diagonally with a soft radial vignette.
        let (a, b): ([f32; 3], [f32; 3]) = match style.as_str() {
            "ocean" => ([0.02, 0.10, 0.25], [0.15, 0.75, 0.85]),
            "aurora" => ([0.03, 0.05, 0.15], [0.20, 0.95, 0.55]),
            _ => ([0.98, 0.55, 0.20], [0.35, 0.10, 0.45]), // sunset
        };
        let mut img = vec![0f32; wd * ht * 3];
        let (cx, cy) = (wd as f32 * 0.5, ht as f32 * 0.55);
        let maxr = ((cx * cx + cy * cy) as f32).sqrt();
        for row in 0..ht {
            if row % 64 == 0 {
                progress(Progress::step((row / 64) as u32 + 1, (ht / 64) as u32 + 1, "rendering"));
            }
            for col in 0..wd {
                let t = (col as f32 / wd as f32 * 0.55 + row as f32 / ht as f32 * 0.45).clamp(0.0, 1.0);
                let dx = col as f32 - cx;
                let dy = row as f32 - cy;
                let vig = 1.0 - 0.35 * ((dx * dx + dy * dy).sqrt() / maxr);
                for ch in 0..3 {
                    let base = a[ch] * (1.0 - t) + b[ch] * t;
                    img[(row * wd + col) * 3 + ch] = (base * vig).clamp(0.0, 1.0);
                }
            }
        }
        Ok(Outcome::new().set("style", json!(style)).blob("image", capability::blob::image_blob(&img, wd as u32, ht as u32, 3)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::Registry;

    #[test]
    fn imageops_are_deterministic_and_shaped() {
        let mut reg = Registry::new();
        reg.register(Arc::new(ImageOps));

        // mask_rect: exact white-pixel count, mask media, right byte length.
        let out = reg
            .run(MODEL, "mask_rect", Invocation::new().set("width", json!(64)).set("height", json!(48)).set("x", json!(10)).set("y", json!(8)).set("w", json!(20)).set("h", json!(12)), &mut |_| {})
            .unwrap();
        assert_eq!(out.outputs["white_px"], 20 * 12);
        let m = &out.blobs["mask"];
        assert_eq!(m.media, Media::Mask);
        assert_eq!(m.bytes.len(), 64 * 48 * 4); // 1 channel f32
        assert_eq!(m.meta, json!({"w":64,"h":48,"c":1}));

        // gradient: deterministic (two runs bit-identical), 3-channel, in range.
        let g = |_: &()| reg.run(MODEL, "gradient", Invocation::new().set("width", json!(32)).set("height", json!(32)).set("style", json!("ocean")), &mut |_| {}).unwrap();
        let (a, b) = (g(&()), g(&()));
        assert_eq!(a.blobs["image"].bytes, b.blobs["image"].bytes, "gradient must be deterministic");
        let px: Vec<f32> = a.blobs["image"].bytes.chunks_exact(4).map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();
        assert_eq!(px.len(), 32 * 32 * 3);
        assert!(px.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }
}
