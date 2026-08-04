// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Synthetic object-detection dataset generator.
//!
//! Produces tiny RGB scenes of simple shapes (filled rectangles + circles) on a
//! constant background, with EXACT axis-aligned ground-truth boxes. It is the
//! detection analogue of the char-level synthetic generators: a fully
//! deterministic-for-a-fixed-seed task a correct detector *must* be able to
//! overfit, used by the P5 overfit integration tests.
//!
//! ## Image-crate choice
//! Per the approved plan we render with EXISTING crates rather than hand-rolling
//! rasterization: [`image::RgbImage`] as the canvas and [`imageproc::drawing`]
//! for the primitives ([`draw_filled_rect_mut`] / [`draw_filled_circle_mut`]).
//! They are plain (non-feature-gated) dependencies of `brain-data` so the
//! generator + its tests build and run by default in CI. All randomness is
//! driven by [`crate::rng::Rng`] (SplitMix64), so a fixed seed yields a
//! byte-identical corpus; imageproc's drawing is deterministic given integer
//! coordinates.
//!
//! ## Class <-> color
//! Each object class maps to a distinct, saturated fill color (drawn from a
//! fixed palette indexed by class id) on a constant dark-grey background. A
//! shape's class is therefore recoverable from its color alone — the property
//! the `classification` preset isolates.
//!
//! ## Ground-truth boxes
//! For every drawn shape we record the EXACT painted pixel extent:
//!   * rectangle `Rect::at(x,y).of_size(w,h)` paints `[x, x+w-1] × [y, y+h-1]`.
//!   * circle center `(cx,cy)` radius `r` paints `[cx-r, cx+r] × [cy-r, cy+r]`
//!     (imageproc fills the inclusive disc, so the bounding box spans `2r+1`px).
//!
//! The extent is clipped to the image; zero/negative-area boxes are dropped. The
//! box is then stored normalized center-xywh in `[0,1]` (see [`DetectBox`]).
//!
//! ## On-disk layout (under `<dir>/`)
//!   * `images.f32` — raw LE f32, `N×3×H×W`, CHW contiguous, normalized `[0,1]`.
//!   * `boxes.bin`  — see [`crate::binio::write_detect_boxes`].
//!   * `meta.json`  — `{n, c:3, h, w, nc}`.

use std::fs;
use std::io;
use std::path::Path;

use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_filled_rect_mut};
use imageproc::rect::Rect;

use crate::binio::{self, DetectBox};
use crate::rng::Rng;

/// Constant scene background (dark grey, distinct from every class color).
const BG: Rgb<u8> = Rgb([32, 32, 32]);

/// Per-class fill palette. A scene's `nc` is clamped to this length. Colors are
/// saturated and mutually distinct so class is recoverable from color.
const PALETTE: [[u8; 3]; 6] = [
    [220, 40, 40],   // 0: red
    [40, 200, 40],   // 1: green
    [60, 80, 230],   // 2: blue
    [230, 200, 40],  // 3: yellow
    [210, 60, 210],  // 4: magenta
    [40, 210, 210],  // 5: cyan
];

/// Which shape primitive a sampled object uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShapeKind {
    Rect,
    Circle,
}

/// A concrete drawn shape (integer pixel geometry) plus its class.
#[derive(Clone, Copy, Debug)]
struct Shape {
    kind: ShapeKind,
    class: u32,
    /// rect: top-left; circle: center.
    x: i32,
    y: i32,
    /// rect: width/height; circle: `w == h == radius` (radius stored in `w`).
    w: i32,
    h: i32,
}

impl Shape {
    /// Exact painted pixel extent as inclusive `[x0,y0,x1,y1]` BEFORE clipping.
    fn extent(&self) -> [i32; 4] {
        match self.kind {
            ShapeKind::Rect => [self.x, self.y, self.x + self.w - 1, self.y + self.h - 1],
            ShapeKind::Circle => {
                let r = self.w;
                [self.x - r, self.y - r, self.x + r, self.y + r]
            }
        }
    }

    fn draw(&self, img: &mut RgbImage) {
        let color = Rgb(PALETTE[self.class as usize]);
        match self.kind {
            ShapeKind::Rect => {
                let rect = Rect::at(self.x, self.y).of_size(self.w as u32, self.h as u32);
                draw_filled_rect_mut(img, rect, color);
            }
            ShapeKind::Circle => {
                draw_filled_circle_mut(img, (self.x, self.y), self.w, color);
            }
        }
    }
}

/// The generator presets, mirroring the user's test plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    /// 1 shape, fixed class, varied position/size — isolates box regression.
    Localization,
    /// Fixed position/size, varied class/color — isolates classification.
    Classification,
    /// 1 shape, sizes spanning small/medium/large — isolates scale handling.
    Scale,
    /// 2–6 non-overlapping-ish shapes of mixed classes — the full task.
    MultiObject,
    /// A mix that INCLUDES some images with zero shapes (background only).
    Background,
}

impl Preset {
    pub fn from_name(name: &str) -> Option<Preset> {
        Some(match name {
            "localization" => Preset::Localization,
            "classification" => Preset::Classification,
            "scale" => Preset::Scale,
            "multi_object" => Preset::MultiObject,
            "background" => Preset::Background,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Preset::Localization => "localization",
            Preset::Classification => "classification",
            Preset::Scale => "scale",
            Preset::MultiObject => "multi_object",
            Preset::Background => "background",
        }
    }
}

/// One generated scene: the canvas plus its normalized ground-truth boxes.
pub struct Scene {
    pub image: RgbImage,
    pub boxes: Vec<DetectBox>,
}

/// Sample ONE scene for `preset` at resolution `w×h` with `nc` classes.
/// Deterministic for a fixed `rng` state.
pub fn gen_scene(preset: Preset, w: u32, h: u32, nc: u32, rng: &mut Rng) -> Scene {
    let nc = nc.min(PALETTE.len() as u32).max(1);
    let mut img = RgbImage::from_pixel(w, h, BG);

    // How many shapes does this preset draw?
    let count: usize = match preset {
        Preset::Localization | Preset::Classification | Preset::Scale => 1,
        Preset::MultiObject => rng.gen_range_inclusive(2, 6) as usize,
        // ~30% of background images are truly empty; the rest carry 1–3 shapes
        // so the preset still exercises the foreground/background contrast.
        Preset::Background => {
            if rng.next_f64() < 0.30 {
                0
            } else {
                rng.gen_range_inclusive(1, 3) as usize
            }
        }
    };

    let mut shapes: Vec<Shape> = Vec::with_capacity(count);
    let mut attempts = 0;
    while shapes.len() < count && attempts < count * 40 + 40 {
        attempts += 1;
        if let Some(s) = sample_shape(preset, w, h, nc, shapes.len(), rng) {
            // For multi-object / background, reject heavy overlaps so the GT
            // boxes stay clean & separable ("non-overlapping-ish").
            let reject_overlap = matches!(preset, Preset::MultiObject | Preset::Background);
            if reject_overlap && shapes.iter().any(|t| overlaps(&clip(s.extent(), w, h), &clip(t.extent(), w, h))) {
                continue;
            }
            shapes.push(s);
        }
    }

    // Draw + emit boxes. Drawing order is shape order (deterministic).
    let mut boxes = Vec::with_capacity(shapes.len());
    for s in &shapes {
        s.draw(&mut img);
    }
    for s in &shapes {
        if let Some(b) = to_box(s, w, h) {
            boxes.push(b);
        }
    }

    Scene { image: img, boxes }
}

/// Sample one shape's geometry for the given preset. Returns `None` if the
/// sample is degenerate (caller retries).
fn sample_shape(preset: Preset, w: u32, h: u32, nc: u32, idx: usize, rng: &mut Rng) -> Option<Shape> {
    let (wi, hi) = (w as i32, h as i32);
    // A circle vs rectangle, biased so both appear.
    let kind = if rng.next_f64() < 0.5 { ShapeKind::Rect } else { ShapeKind::Circle };

    match preset {
        Preset::Localization => {
            // Fixed class 0, varied position + size.
            place_random(ShapeKind::Rect, 0, wi, hi, 0.12, 0.45, rng)
                .or_else(|| place_random(kind, 0, wi, hi, 0.12, 0.45, rng))
        }
        Preset::Classification => {
            // Fixed position/size (centered, medium), varied class/color.
            let class = rng.gen_range_inclusive(0, nc as i64 - 1) as u32;
            let side = (wi.min(hi) as f64 * 0.4) as i32;
            Some(centered(ShapeKind::Rect, class, wi, hi, side))
        }
        Preset::Scale => {
            // One shape; size drawn from small/medium/large buckets.
            let frac = *rng.choice(&[0.10f64, 0.25, 0.50]);
            place_random(kind, 0, wi, hi, frac, frac + 0.02, rng)
        }
        Preset::MultiObject | Preset::Background => {
            let class = rng.gen_range_inclusive(0, nc as i64 - 1) as u32;
            // Smaller shapes so several fit without overlap.
            let _ = idx;
            place_random(kind, class, wi, hi, 0.10, 0.28, rng)
        }
    }
}

/// Place a shape of fractional size in `[lo,hi]` of the min image side at a
/// random in-bounds position.
fn place_random(kind: ShapeKind, class: u32, wi: i32, hi: i32, lo: f64, hi_frac: f64, rng: &mut Rng) -> Option<Shape> {
    let minside = wi.min(hi) as f64;
    let size = (rng.uniform(lo, hi_frac) * minside).round() as i32;
    if size < 4 {
        return None;
    }
    match kind {
        ShapeKind::Rect => {
            // independent w/h within [size*0.7, size]
            let ww = ((size as f64) * rng.uniform(0.7, 1.0)).round().max(4.0) as i32;
            let hh = ((size as f64) * rng.uniform(0.7, 1.0)).round().max(4.0) as i32;
            if ww >= wi || hh >= hi {
                return None;
            }
            let x = rng.gen_range_inclusive(0, (wi - ww) as i64) as i32;
            let y = rng.gen_range_inclusive(0, (hi - hh) as i64) as i32;
            Some(Shape { kind, class, x, y, w: ww, h: hh })
        }
        ShapeKind::Circle => {
            let r = (size / 2).max(2);
            if 2 * r + 1 >= wi || 2 * r + 1 >= hi {
                return None;
            }
            // center kept r..side-r so the full disc is in-bounds.
            let cx = rng.gen_range_inclusive(r as i64, (wi - r - 1) as i64) as i32;
            let cy = rng.gen_range_inclusive(r as i64, (hi - r - 1) as i64) as i32;
            Some(Shape { kind, class, x: cx, y: cy, w: r, h: r })
        }
    }
}

/// A centered shape of the given pixel `side` (square rect / circle of that
/// diameter). Used by the `classification` preset (fixed geometry).
fn centered(kind: ShapeKind, class: u32, wi: i32, hi: i32, side: i32) -> Shape {
    match kind {
        ShapeKind::Rect => Shape { kind, class, x: (wi - side) / 2, y: (hi - side) / 2, w: side, h: side },
        ShapeKind::Circle => {
            let r = (side / 2).max(2);
            Shape { kind, class, x: wi / 2, y: hi / 2, w: r, h: r }
        }
    }
}

/// Clip an inclusive `[x0,y0,x1,y1]` extent to the image bounds.
fn clip(e: [i32; 4], w: u32, h: u32) -> [i32; 4] {
    [
        e[0].clamp(0, w as i32 - 1),
        e[1].clamp(0, h as i32 - 1),
        e[2].clamp(0, w as i32 - 1),
        e[3].clamp(0, h as i32 - 1),
    ]
}

/// Do two clipped inclusive extents overlap at all?
fn overlaps(a: &[i32; 4], b: &[i32; 4]) -> bool {
    a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
}

/// Convert a shape to a clipped normalized center-xywh box. `None` if the
/// clipped box has zero/negative area.
fn to_box(s: &Shape, w: u32, h: u32) -> Option<DetectBox> {
    let e = clip(s.extent(), w, h);
    let (x0, y0, x1, y1) = (e[0], e[1], e[2], e[3]);
    // inclusive pixel extent => pixel width = (x1 - x0 + 1).
    let bw = (x1 - x0 + 1) as f32;
    let bh = (y1 - y0 + 1) as f32;
    if bw <= 0.0 || bh <= 0.0 {
        return None;
    }
    let cx = (x0 as f32 + x1 as f32 + 1.0) * 0.5; // center in pixels (edges at x0, x1+1)
    let cy = (y0 as f32 + y1 as f32 + 1.0) * 0.5;
    Some(DetectBox {
        class: s.class,
        cx: cx / w as f32,
        cy: cy / h as f32,
        w: bw / w as f32,
        h: bh / h as f32,
    })
}

/// Convert an [`RgbImage`] to a normalized `f32` CHW tensor (`3×H×W`, channel
/// order R,G,B), values in `[0,1]`.
pub fn image_to_chw(img: &RgbImage) -> Vec<f32> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut out = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x as u32, y as u32).0;
            let idx = y * w + x;
            out[idx] = p[0] as f32 / 255.0;
            out[h * w + idx] = p[1] as f32 / 255.0;
            out[2 * h * w + idx] = p[2] as f32 / 255.0;
        }
    }
    out
}

/// Generate `n` scenes for a single preset, returning the concatenated CHW image
/// blob (`N×3×H×W`) and the per-image boxes.
pub fn generate(
    preset: Preset,
    n: usize,
    w: u32,
    h: u32,
    nc: u32,
    rng: &mut Rng,
) -> (Vec<f32>, Vec<Vec<DetectBox>>) {
    let mut images = Vec::with_capacity(n * 3 * h as usize * w as usize);
    let mut boxes = Vec::with_capacity(n);
    for _ in 0..n {
        let scene = gen_scene(preset, w, h, nc, rng);
        images.extend_from_slice(&image_to_chw(&scene.image));
        boxes.push(scene.boxes);
    }
    (images, boxes)
}

/// `meta.json` for a detection dataset.
fn meta_json(n: usize, h: u32, w: u32, nc: u32) -> String {
    serde_json::json!({ "n": n, "c": 3, "h": h, "w": w, "nc": nc }).to_string()
}

/// Write a full detection dataset (`images.f32` + `boxes.bin` + `meta.json`)
/// for a single preset into `dir`.
pub fn write_dataset(
    dir: &Path,
    preset: Preset,
    n: usize,
    w: u32,
    h: u32,
    nc: u32,
    seed: u64,
) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let mut rng = Rng::new(seed);
    let (images, boxes) = generate(preset, n, w, h, nc, &mut rng);
    binio::write_f32_bin(&dir.join("images.f32"), &images)?;
    binio::write_detect_boxes(&dir.join("boxes.bin"), &boxes)?;
    fs::write(dir.join("meta.json"), meta_json(n, h, w, nc))?;
    Ok(())
}

/// Loaded detection dataset: the flat CHW image blob, per-image boxes, and
/// geometry read back from `meta.json`.
pub struct DetectData {
    pub images: Vec<f32>,
    pub boxes: Vec<Vec<DetectBox>>,
    pub n: usize,
    pub h: u32,
    pub w: u32,
    pub nc: u32,
}

impl DetectData {
    /// Number of f32 elements in one image (`3*H*W`).
    pub fn image_stride(&self) -> usize {
        3 * self.h as usize * self.w as usize
    }
}

/// Read a detection dataset written by [`write_dataset`] back into memory.
pub fn load_dataset(dir: &Path) -> io::Result<DetectData> {
    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("meta.json"))?).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    let n = meta["n"].as_u64().unwrap_or(0) as usize;
    let h = meta["h"].as_u64().unwrap_or(0) as u32;
    let w = meta["w"].as_u64().unwrap_or(0) as u32;
    let nc = meta["nc"].as_u64().unwrap_or(0) as u32;
    let images = binio::read_detect_images(&dir.join("images.f32"))?;
    let boxes = binio::read_detect_boxes(&dir.join("boxes.bin"), n)?;
    Ok(DetectData { images, boxes, n, h, w, nc })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("brain_gen_detect_{name}_{}", std::process::id()))
    }

    #[test]
    fn deterministic_same_seed_byte_equal() {
        let mut a = Rng::new(99);
        let mut b = Rng::new(99);
        let (ia, ba) = generate(Preset::MultiObject, 8, 64, 64, 4, &mut a);
        let (ib, bb) = generate(Preset::MultiObject, 8, 64, 64, 4, &mut b);
        assert_eq!(ia, ib, "images differ for same seed");
        assert_eq!(ba, bb, "boxes differ for same seed");
    }

    #[test]
    fn different_seed_differs() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let (ia, _) = generate(Preset::MultiObject, 8, 64, 64, 4, &mut a);
        let (ib, _) = generate(Preset::MultiObject, 8, 64, 64, 4, &mut b);
        assert_ne!(ia, ib, "different seeds produced identical images");
    }

    /// Label-geometry test: the emitted GT box must match the shape's true
    /// painted pixel extent within <=1px, for both a rectangle and a circle.
    #[test]
    fn box_matches_drawn_pixels_within_1px() {
        let (w, h) = (128u32, 128u32);
        for (kind, shape) in [
            (ShapeKind::Rect, Shape { kind: ShapeKind::Rect, class: 0, x: 20, y: 30, w: 40, h: 24 }),
            (ShapeKind::Circle, Shape { kind: ShapeKind::Circle, class: 1, x: 70, y: 60, w: 18, h: 18 }),
        ] {
            assert_eq!(shape.kind, kind);
            let mut img = RgbImage::from_pixel(w, h, BG);
            shape.draw(&mut img);
            // Measure the actual painted (non-background) pixel extent.
            let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
            for y in 0..h {
                for x in 0..w {
                    if *img.get_pixel(x, y) != BG {
                        x0 = x0.min(x as i32);
                        y0 = y0.min(y as i32);
                        x1 = x1.max(x as i32);
                        y1 = y1.max(y as i32);
                    }
                }
            }
            // Emitted box -> pixel xyxy.
            let b = to_box(&shape, w, h).unwrap();
            let px_cx = b.cx * w as f32;
            let px_cy = b.cy * h as f32;
            let px_w = b.w * w as f32;
            let px_h = b.h * h as f32;
            let bx0 = px_cx - px_w * 0.5;
            let by0 = px_cy - px_h * 0.5;
            let bx1 = px_cx + px_w * 0.5;
            let by1 = px_cy + px_h * 0.5;
            // The label edges should bound the painted pixels within 1px. The
            // painted region is [x0,x1] inclusive => true edges at x0 and x1+1.
            assert!((bx0 - x0 as f32).abs() <= 1.0, "{kind:?} x0: label {bx0} vs painted {x0}");
            assert!((by0 - y0 as f32).abs() <= 1.0, "{kind:?} y0: label {by0} vs painted {y0}");
            assert!((bx1 - (x1 + 1) as f32).abs() <= 1.0, "{kind:?} x1: label {bx1} vs painted {}", x1 + 1);
            assert!((by1 - (y1 + 1) as f32).abs() <= 1.0, "{kind:?} y1: label {by1} vs painted {}", y1 + 1);
        }
    }

    #[test]
    fn presets_produce_expected_shape_counts() {
        let mut rng = Rng::new(7);
        // localization/classification/scale: exactly 1 box.
        for p in [Preset::Localization, Preset::Classification, Preset::Scale] {
            for _ in 0..20 {
                let s = gen_scene(p, 96, 96, 3, &mut rng);
                assert_eq!(s.boxes.len(), 1, "{:?} should have 1 box", p);
            }
        }
        // background: at least one empty image appears across many samples.
        let mut empty_seen = false;
        for _ in 0..50 {
            let s = gen_scene(Preset::Background, 96, 96, 3, &mut rng);
            if s.boxes.is_empty() {
                empty_seen = true;
            }
        }
        assert!(empty_seen, "background preset never produced an empty image");
    }

    #[test]
    fn classification_recovers_class_from_color() {
        // Fixed geometry, varied class: the box class must equal the palette
        // color drawn at the image center.
        let mut rng = Rng::new(3);
        for _ in 0..10 {
            let s = gen_scene(Preset::Classification, 96, 96, 6, &mut rng);
            assert_eq!(s.boxes.len(), 1);
            let cls = s.boxes[0].class as usize;
            let center = *s.image.get_pixel(48, 48);
            assert_eq!(center, Rgb(PALETTE[cls]), "center color must match class palette");
        }
    }

    #[test]
    fn write_load_roundtrip() {
        let dir = tmp("io");
        let _ = fs::remove_dir_all(&dir);
        write_dataset(&dir, Preset::MultiObject, 5, 64, 64, 4, 123).unwrap();
        let d = load_dataset(&dir).unwrap();
        assert_eq!(d.n, 5);
        assert_eq!((d.h, d.w, d.nc), (64, 64, 4));
        assert_eq!(d.images.len(), 5 * d.image_stride());
        assert_eq!(d.boxes.len(), 5);
        // re-generate in memory and compare (determinism through disk).
        let mut rng = Rng::new(123);
        let (img2, box2) = generate(Preset::MultiObject, 5, 64, 64, 4, &mut rng);
        assert_eq!(d.images, img2);
        assert_eq!(d.boxes, box2);
        let _ = fs::remove_dir_all(&dir);
    }
}
