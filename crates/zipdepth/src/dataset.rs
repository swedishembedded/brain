// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verified depth-estimation training for its clients. If your team needs expertise in
// training monocular depth models on your own camera and sensor data, you can procure our services by sending an
// email to info@swedishembedded.com.

//! A paired RGB / depth directory as a ZipDepth training source.
//!
//! ## Format
//!
//! ```text
//! <dir>/manifest.json      {"depth_per_count": 0.001, "invalid_count": 0}
//! <dir>/<stem>.png         the RGB image (or <stem>.jpg / <stem>.jpeg)
//! <dir>/<stem>.depth.png   the aligned depth, a 16-bit grayscale PNG, the same size as the RGB image
//! ```
//!
//! A stored count `c` is a depth of `c * depth_per_count` (the manifest's unit, in whatever length unit the sensor
//! reports; the loss is scale-invariant, so the unit never matters to training, but it must be stated). The count
//! `invalid_count` means "no measurement" (0 for most sensors, 65535 for some). The target is inverse depth
//! `1 / (c * depth_per_count)`, the quantity the network predicts, with a validity mask that is 0 at invalid counts.
//!
//! ## Where this lives
//!
//! `data` cannot depend on `imaging` (`imaging -> vision -> model -> data`, see `imaging`'s crate docs), and a
//! depth-specific manifest does not belong in `data` anyway, so the whole loader is here: the 16-bit decode is
//! `imaging::decode_gray16`, the RGB decode `imaging::decode_bounded`, the RGB resize `imaging::resize_bilinear_hwc`
//! and the nearest tap `imaging::nearest_tap`; this module owns the manifest, the pairing and the validation.
//!
//! ## Hostile input
//!
//! A dataset directory is outside the program's control, so every defect is refused by name instead of coerced or
//! dropped: an unknown or missing manifest field, a non-positive unit, an unrecognised file, a stem with only one of
//! its two files, two RGB files for one stem, an RGB and depth map of different sizes, a depth map that is not a
//! 16-bit grayscale PNG, a zero count that is not the invalid marker (a depth of 0 has no inverse), and an image with
//! under two valid pixels. [`DepthDataset::open`] checks everything that needs only file names and PNG headers, so a
//! bad layout fails before the first step; pixel-level defects are found when a sample is first decoded, which is
//! within the first epoch because epochs visit every sample, and abort training naming the file.
//!
//! ## Geometry
//!
//! A sample is centre-cropped to the model input's aspect ratio (never letterboxed: grey borders are not depth and
//! would be supervised), then resized: the RGB bilinearly, the inverse depth and the mask by nearest tap. A bilinear
//! tap across an invalid pixel would invent a depth no sensor measured.

use std::path::{Path, PathBuf};

use data::rng::{random_seed, Rng};
use serde::Deserialize;

use crate::train::{Batch, BatchSource};

/// Largest image the loader decodes, in pixels (64 MP): a PNG header can declare far more than its file holds.
const MAX_PIXELS: u64 = 1 << 26;

/// `manifest.json`: how to read the depth maps.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    /// Depth represented by one stored count, in the sensor's length unit.
    depth_per_count: f64,
    /// The stored count that means "no measurement".
    invalid_count: u16,
}

/// One decoded, cropped and resized sample at the model's input size.
#[derive(Clone, Debug)]
pub struct Sample {
    /// CHW RGB in `[0, 1]`, `[3 * h * w]`.
    pub rgb: Vec<f32>,
    /// Inverse depth, `[h * w]`; 0 where the mask is 0.
    pub inv: Vec<f32>,
    /// 1.0 where a depth was measured, else 0.0; `[h * w]`.
    pub mask: Vec<f32>,
}

struct Entry {
    stem: String,
    rgb: PathBuf,
    depth: PathBuf,
}

/// A validated paired dataset, drawn in seeded shuffled epochs.
pub struct DepthDataset {
    manifest: Manifest,
    entries: Vec<Entry>,
    rng: Rng,
    /// The current epoch's order and how much of it has been drawn.
    order: Vec<usize>,
    cursor: usize,
}

fn name_of(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| p.display().to_string())
}

impl DepthDataset {
    /// Open and validate `dir` (format in the module docs). `seed` fixes the epoch shuffles; a caller with no
    /// seed to give draws one with [`data::rng::random_seed`] and reports it.
    pub fn open(dir: &Path, seed: u64) -> Result<DepthDataset, String> {
        let manifest_path = dir.join("manifest.json");
        let text = std::fs::read_to_string(&manifest_path).map_err(|e| format!("manifest.json: reading {}: {e}", manifest_path.display()))?;
        let manifest: Manifest = serde_json::from_str(&text).map_err(|e| format!("manifest.json: {e}"))?;
        if !(manifest.depth_per_count.is_finite() && manifest.depth_per_count > 0.0) {
            return Err(format!("manifest.json: depth_per_count must be a positive finite number, got {}", manifest.depth_per_count));
        }

        // Pair the files by stem. Every name must be one of the three kinds; nothing is skipped silently.
        let mut rgb: std::collections::BTreeMap<String, PathBuf> = Default::default();
        let mut depth: std::collections::BTreeMap<String, PathBuf> = Default::default();
        let listing = std::fs::read_dir(dir).map_err(|e| format!("reading {}: {e}", dir.display()))?;
        for entry in listing {
            let path = entry.map_err(|e| format!("listing {}: {e}", dir.display()))?.path();
            let name = name_of(&path);
            if name == "manifest.json" {
                continue;
            }
            if path.is_dir() {
                return Err(format!("{name}: a directory; a depth dataset is flat"));
            }
            let lower = name.to_ascii_lowercase();
            if let Some(stem) = lower.strip_suffix(".depth.png").map(|_| &name[..name.len() - ".depth.png".len()]) {
                depth.insert(stem.to_string(), path);
            } else if let Some(stem) = [".png", ".jpg", ".jpeg"].iter().find_map(|ext| lower.strip_suffix(ext).map(|_| &name[..name.len() - ext.len()])) {
                if let Some(other) = rgb.insert(stem.to_string(), path) {
                    return Err(format!("{name}: stem `{stem}` already has an RGB image, {}", name_of(&other)));
                }
            } else {
                return Err(format!("{name}: not a manifest, an RGB image (.png/.jpg/.jpeg) or a depth map (.depth.png)"));
            }
        }
        if let Some((stem, p)) = rgb.iter().find(|(s, _)| !depth.contains_key(*s)) {
            return Err(format!("{}: RGB image has no depth map {stem}.depth.png", name_of(p)));
        }
        if let Some((stem, p)) = depth.iter().find(|(s, _)| !rgb.contains_key(*s)) {
            return Err(format!("{}: depth map has no RGB image {stem}.png or {stem}.jpg", name_of(p)));
        }
        if rgb.is_empty() {
            return Err(format!("{}: no samples (expected <stem>.png beside <stem>.depth.png)", dir.display()));
        }

        // BTreeMap iteration is name order: the sample index is a stable, reproducible identity.
        let mut entries = Vec::with_capacity(rgb.len());
        for (stem, rgb_path) in rgb {
            let depth_path = depth.remove(&stem).expect("paired above");
            let (rw, rh) = imaging::dimensions(&rgb_path)?;
            let (dw, dh) = imaging::dimensions(&depth_path)?;
            if (rw, rh) != (dw, dh) {
                return Err(format!("{} is {rw}x{rh} but {} is {dw}x{dh}", name_of(&rgb_path), name_of(&depth_path)));
            }
            entries.push(Entry { stem, rgb: rgb_path, depth: depth_path });
        }
        let mut ds = DepthDataset { manifest, entries, rng: Rng::new(seed), order: Vec::new(), cursor: 0 };
        ds.reshuffle();
        Ok(ds)
    }

    /// [`DepthDataset::open`] on a freshly drawn seed, returned so the run can be reproduced.
    pub fn open_random(dir: &Path) -> Result<(DepthDataset, u64), String> {
        let seed = random_seed();
        Ok((DepthDataset::open(dir, seed)?, seed))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The sample stems in index order (name order).
    pub fn stems(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.stem.as_str()).collect()
    }

    /// Fisher-Yates over the sample indices.
    fn reshuffle(&mut self) {
        self.order = (0..self.entries.len()).collect();
        for i in (1..self.order.len()).rev() {
            let j = self.rng.gen_range_inclusive(0, i as i64) as usize;
            self.order.swap(i, j);
        }
        self.cursor = 0;
    }

    /// The next `k` sample indices: whole shuffled epochs, one after another, each visiting every sample once.
    pub fn draw(&mut self, k: usize) -> Vec<usize> {
        (0..k)
            .map(|_| {
                if self.cursor == self.order.len() {
                    self.reshuffle();
                }
                self.cursor += 1;
                self.order[self.cursor - 1]
            })
            .collect()
    }

    /// Decode sample `index`, convert its depth to inverse depth and a mask, centre-crop to `w:h` and resize to
    /// `h x w`. Every refusal names the file.
    pub fn sample(&self, index: usize, h: u32, w: u32) -> Result<Sample, String> {
        let e = &self.entries[index];
        let at = |file: &Path, msg: String| format!("{}: {msg}", name_of(file));
        let read = |file: &Path| std::fs::read(file).map_err(|err| at(file, format!("reading: {err}")));
        let img = imaging::codec::decode_bounded(&read(&e.rgb)?, MAX_PIXELS).map_err(|m| at(&e.rgb, m))?;
        let depth = imaging::decode_gray16(&read(&e.depth)?, MAX_PIXELS).map_err(|m| at(&e.depth, m))?;
        if (img.w, img.h) != (depth.w, depth.h) {
            return Err(format!("{} is {}x{} but {} is {}x{}", name_of(&e.rgb), img.w, img.h, name_of(&e.depth), depth.w, depth.h));
        }

        // Counts -> (inverse depth, valid) on the source grid.
        let (mut inv, mut mask) = (Vec::with_capacity(depth.px.len()), Vec::with_capacity(depth.px.len()));
        for (i, &c) in depth.px.iter().enumerate() {
            if c == self.manifest.invalid_count {
                inv.push(0.0f32);
                mask.push(0.0f32);
            } else if c == 0 {
                let (x, y) = (i as u32 % depth.w, i as u32 / depth.w);
                return Err(at(&e.depth, format!("count 0 at ({x}, {y}) is not the manifest's invalid_count ({}) and has no inverse depth", self.manifest.invalid_count)));
            } else {
                inv.push((1.0 / (f64::from(c) * self.manifest.depth_per_count)) as f32);
                mask.push(1.0);
            }
        }

        // Centre crop to the target aspect, the same rectangle for all three planes.
        let (sw, sh) = (img.w, img.h);
        let (cw, ch) = if u64::from(sw) * u64::from(h) >= u64::from(sh) * u64::from(w) {
            (((u64::from(sh) * u64::from(w) + u64::from(h) / 2) / u64::from(h)).clamp(1, u64::from(sw)) as u32, sh)
        } else {
            (sw, ((u64::from(sw) * u64::from(h) + u64::from(w) / 2) / u64::from(w)).clamp(1, u64::from(sh)) as u32)
        };
        let (x0, y0) = ((sw - cw) / 2, (sh - ch) / 2);
        let rgb_hwc = img.to_hwc_unit();
        let mut crop = Vec::with_capacity(3 * (cw * ch) as usize);
        for y in y0..y0 + ch {
            let row = (y * sw + x0) as usize * 3;
            crop.extend_from_slice(&rgb_hwc[row..row + 3 * cw as usize]);
        }
        let rgb = imaging::pixels::hwc_to_chw(&imaging::resize_bilinear_hwc(&crop, 3, cw, ch, w, h), 3, h as usize, w as usize);

        let (mut inv_out, mut mask_out) = (Vec::with_capacity((h * w) as usize), Vec::with_capacity((h * w) as usize));
        for oy in 0..h {
            let sy = y0 + imaging::nearest_tap(oy, ch, h);
            for ox in 0..w {
                let i = (sy * sw + x0 + imaging::nearest_tap(ox, cw, w)) as usize;
                inv_out.push(inv[i]);
                mask_out.push(mask[i]);
            }
        }
        let valid = mask_out.iter().filter(|&&m| m == 1.0).count();
        if valid < 2 {
            return Err(at(&e.depth, format!("{valid} valid pixel(s) after cropping and resizing to {w}x{h}; the loss needs at least 2")));
        }
        Ok(Sample { rgb, inv: inv_out, mask: mask_out })
    }
}

impl BatchSource for DepthDataset {
    fn next_batch(&mut self, batch: u32, h: u32, w: u32) -> Result<Batch, String> {
        let mut out = Batch::default();
        for i in self.draw(batch as usize) {
            let s = self.sample(i, h, w)?;
            out.rgb.extend_from_slice(&s.rgb);
            out.inv.extend_from_slice(&s.inv);
            out.mask.extend_from_slice(&s.mask);
        }
        Ok(out)
    }
}
