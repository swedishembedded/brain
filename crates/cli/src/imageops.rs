// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `imageops` - deterministic, no-weights image utilities exposed through the
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

    let gradient = size(ActionSpec::new("gradient", "render a smooth procedural gradient image - a deterministic, viewable test image"))
        .param(ParamSpec::new("style", ParamType::Enum(vec!["sunset".into(), "ocean".into(), "aurora".into()]), "palette").default(json!("sunset")))
        .output(BlobSpec::new("image", Media::Image, "the rendered image"));

    let draw_boxes = ActionSpec::new("draw_boxes", "draw labeled detection boxes onto an image (e.g. yolov8 detect's own output)")
        .param(ParamSpec::new("boxes", ParamType::Str, "JSON array of {bbox:[x1,y1,x2,y2], conf, class} in image coords - yolov8 detect's exact output shape").required())
        .param(ParamSpec::new("thickness", ParamType::Int, "box edge thickness, px").default(json!(2)))
        .input(BlobSpec::new("image", Media::Image, "the image the boxes were detected on"))
        .output(BlobSpec::new("image", Media::Image, "the image with boxes and class/confidence labels drawn"));

    let colorize = ActionSpec::new("colorize", "false-color a single-channel field (a depth map, a segmentation mask) for viewing")
        .param(ParamSpec::new("colormap", ParamType::Enum(vec!["turbo".into(), "gray".into(), "grayinv".into()]), "palette").default(json!("turbo")))
        .input(BlobSpec::new("image", Media::Image, "a 1-channel field, replicated to grey RGB (what save_blob writes for a Mask output)"))
        .output(BlobSpec::new("image", Media::Image, "the false-colored image"));

    Manifest::new(MODEL, "deterministic image utilities (mask + procedural renders + detection/mask visualization) - no weights", vec![mask_rect, gradient, draw_boxes, colorize])
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
            "draw_boxes" => Some(Arc::new(DrawBoxes)),
            "colorize" => Some(Arc::new(Colorize)),
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
        let maxr = (cx * cx + cy * cy).sqrt();
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

/// One detection box, as `brain yolov8 detect`'s own `detections` JSON emits
/// it (`{"bbox":[x1,y1,x2,y2],"conf":...,"class":N}`) - parsed loosely (a
/// missing/malformed entry is skipped, not a hard error) so a caller piping
/// live model output straight through never has to pre-validate it.
struct Box_ {
    bbox: [f32; 4],
    conf: f32,
    class: u32,
}

impl Box_ {
    fn from_json(v: &serde_json::Value) -> Option<Box_> {
        let b = v.get("bbox")?.as_array()?;
        if b.len() != 4 {
            return None;
        }
        let mut bbox = [0f32; 4];
        for (i, slot) in bbox.iter_mut().enumerate() {
            *slot = b[i].as_f64()? as f32;
        }
        Some(Box_ { bbox, conf: v.get("conf").and_then(|c| c.as_f64()).unwrap_or(0.0) as f32, class: v.get("class").and_then(|c| c.as_u64()).unwrap_or(0) as u32 })
    }
}

/// A small, high-contrast, cycling palette - enough to tell classes apart at
/// a glance without a model-specific color table.
const BOX_PALETTE: [[u8; 3]; 8] =
    [[255, 64, 64], [64, 200, 255], [255, 210, 40], [80, 220, 100], [220, 100, 255], [255, 150, 60], [90, 255, 200], [255, 255, 255]];

struct DrawBoxes;
impl Action for DrawBoxes {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == "draw_boxes").unwrap()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> capability::ActionResult {
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
        let mut img = imaging::pixels::hwc_to_rgb8(&hwc, w, h, 3, imaging::ChannelPolicy::RequireRgb)?;
        let thickness = inv.get_i64("thickness").unwrap_or(2).clamp(1, 32) as u32;
        let raw = inv.get_str("boxes").ok_or("draw_boxes: missing required param 'boxes'")?;
        let boxes: Vec<Box_> = serde_json::from_str::<Vec<serde_json::Value>>(&raw)
            .map_err(|e| format!("draw_boxes: 'boxes' is not a JSON array: {e}"))?
            .iter()
            .filter_map(Box_::from_json)
            .collect();
        progress(Progress::step(1, 1, format!("drawing {} box(es)", boxes.len())));
        for b in &boxes {
            let color = BOX_PALETTE[b.class as usize % BOX_PALETTE.len()];
            let (x0, y0, x1, y1) = (b.bbox[0].round() as i64, b.bbox[1].round() as i64, b.bbox[2].round() as i64, b.bbox[3].round() as i64);
            draw_rect(&mut img.px, w, h, x0, y0, x1, y1, thickness, color);
            let label = format!("{}:{:.2}", b.class, b.conf);
            let (lx, ly) = (x0.max(0) as u32, y0.saturating_sub(9).max(0) as u32);
            imaging::viz::draw_text(&mut img.px, w, h, lx, ly, &label, 1, color);
        }
        Ok(Outcome::new().set("boxes_drawn", json!(boxes.len())).blob("image", capability::blob::image_blob(&img.to_hwc_unit(), w, h, 3)))
    }
}

/// Draw a `thickness`-px rectangle outline, clamped to the frame - a demo
/// annotation overlay, not a training-path op, so this stays host-side next
/// to `imaging::viz::draw_text` rather than becoming a kernel dispatch.
#[allow(clippy::too_many_arguments)]
fn draw_rect(rgb: &mut [u8], w: u32, h: u32, x0: i64, y0: i64, x1: i64, y1: i64, thickness: u32, color: [u8; 3]) {
    let (x0, y0) = (x0.clamp(0, w as i64 - 1) as u32, y0.clamp(0, h as i64 - 1) as u32);
    let (x1, y1) = (x1.clamp(0, w as i64 - 1) as u32, y1.clamp(0, h as i64 - 1) as u32);
    let mut set = |x: u32, y: u32| {
        if x < w && y < h {
            let o = ((y * w + x) * 3) as usize;
            rgb[o] = color[0];
            rgb[o + 1] = color[1];
            rgb[o + 2] = color[2];
        }
    };
    for t in 0..thickness {
        for x in x0..=x1 {
            set(x, y0.saturating_add(t));
            set(x, y1.saturating_sub(t));
        }
        for y in y0..=y1 {
            set(x0.saturating_add(t), y);
            set(x1.saturating_sub(t), y);
        }
    }
}

struct Colorize;
impl Action for Colorize {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == "colorize").unwrap()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> capability::ActionResult {
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
        let map = imaging::viz::parse_colormap(&inv.get_str("colormap").unwrap_or_else(|| "turbo".into()))?;
        // A saved Mask/depth blob is replicated to grey RGB (`save_blob`'s
        // `ChannelPolicy::ReplicateFirst`), so channel 0 already carries the
        // scalar field regardless of whether `image` came in as 1 or 3 chan.
        let c = hwc.len() / (w as usize * h as usize).max(1);
        let field: Vec<f32> = hwc.chunks_exact(c).map(|px| px[0]).collect();
        progress(Progress::step(1, 1, "colorizing"));
        let bounds = imaging::viz::Bounds::from_percentiles(&field, 0.02, 0.98);
        let rgb = imaging::viz::colorize(&field, bounds, map);
        let hwc_out = imaging::pixels::u8_to_unit(&rgb);
        Ok(Outcome::new().blob("image", capability::blob::image_blob(&hwc_out, w, h, 3)))
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
