// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA training dataset: a folder of images each paired with a text prompt.
//!
//! Prompts come from a caption file in the folder. Two formats, in priority order:
//!  1. **`captions.yaml`** (primary — easy to hand-edit): a flat mapping of
//!     `filename: prompt`, one per line. Values may be quoted or bare; `#` starts a
//!     comment; blank lines are ignored.
//!  2. **`captions.jsonl`** (override): one JSON object per line,
//!     `{"file": "...", "prompt": "..."}`. Entries here **override / add to** the
//!     YAML ones — so you can keep a readable YAML base and script exceptions.
//!
//! Example `captions.yaml`:
//! ```yaml
//! # a subject token like "sks" helps the adapter bind to your concept
//! cat01.png: a photo of sks cat sitting on a chair
//! cat02.jpg: "a photo of sks cat, closeup, studio light"
//! ```
//!
//! Images are decoded (JPEG/PNG/PPM), center-cropped to square, and resized to the
//! training size, yielding interleaved-RGB HWC f32 in `[0,1]` — exactly what the VAE
//! encoder consumes. An image with no caption entry is skipped (with a warning).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One training example: the pre-processed image (HWC f32 `[0,1]`, `size×size×3`)
/// and its prompt.
pub struct Sample {
    pub path: PathBuf,
    pub prompt: String,
    pub hwc: Vec<f32>,
    pub size: u32,
}

/// Load and pre-process every captioned image in `dir` to `size×size`. Errors only
/// on a missing/unreadable directory or a total lack of usable samples; individual
/// unreadable images are skipped with a `warn` line so one bad file can't abort a run.
pub fn load_dir(dir: &Path, size: u32, mut warn: impl FnMut(&str)) -> Result<Vec<Sample>, String> {
    if !dir.is_dir() {
        return Err(format!("dataset dir {} does not exist", dir.display()));
    }
    let mut caps = parse_captions_yaml(&dir.join("captions.yaml"), &mut warn);
    apply_jsonl_overrides(&dir.join("captions.jsonl"), &mut caps, &mut warn);
    if caps.is_empty() {
        return Err(format!(
            "no captions in {}: add a captions.yaml (`filename: prompt` per line) or captions.jsonl",
            dir.display()
        ));
    }

    let mut samples = Vec::new();
    for (file, prompt) in &caps {
        let path = dir.join(file);
        if !path.exists() {
            warn(&format!("caption references missing image {file} — skipping"));
            continue;
        }
        match load_image_square(&path, size) {
            Ok(hwc) => samples.push(Sample { path, prompt: prompt.clone(), hwc, size }),
            Err(e) => warn(&format!("skipping {file}: {e}")),
        }
    }
    if samples.is_empty() {
        return Err(format!("no usable images decoded from {}", dir.display()));
    }
    Ok(samples)
}

/// Parse the flat `filename: prompt` YAML subset. Tolerant by design (this is a
/// hand-edited file): `#` comments, blank lines, and optional quotes are handled;
/// anything else is warned about and skipped rather than aborting the load.
fn parse_captions_yaml(path: &Path, warn: &mut impl FnMut(&str)) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let Ok(text) = std::fs::read_to_string(path) else { return out };
    for (i, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        let Some((key, val)) = line.split_once(':') else {
            warn(&format!("captions.yaml:{}: no ':' — skipping `{raw}`", i + 1));
            continue;
        };
        let file = key.trim().trim_matches(|c| c == '"' || c == '\'').trim();
        let prompt = unquote(val.trim());
        if file.is_empty() || prompt.is_empty() {
            warn(&format!("captions.yaml:{}: empty filename or prompt — skipping", i + 1));
            continue;
        }
        out.insert(file.to_string(), prompt);
    }
    out
}

/// Overlay `captions.jsonl` `{"file","prompt"}` entries onto `caps` (override/add).
fn apply_jsonl_overrides(path: &Path, caps: &mut BTreeMap<String, String>, warn: &mut impl FnMut(&str)) {
    let Ok(text) = std::fs::read_to_string(path) else { return };
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(v) => {
                let file = v.get("file").and_then(|x| x.as_str());
                let prompt = v.get("prompt").and_then(|x| x.as_str());
                match (file, prompt) {
                    (Some(f), Some(p)) if !f.is_empty() && !p.is_empty() => {
                        caps.insert(f.to_string(), p.to_string());
                    }
                    _ => warn(&format!("captions.jsonl:{}: need non-empty \"file\" and \"prompt\"", i + 1)),
                }
            }
            Err(e) => warn(&format!("captions.jsonl:{}: {e}", i + 1)),
        }
    }
}

/// Drop a trailing `#` comment (outside of quotes — captions rarely quote, but a
/// prompt like `"a # sign"` should keep its hash).
fn strip_comment(line: &str) -> &str {
    let mut in_q = false;
    let bytes = line.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'"' | b'\'' => in_q = !in_q,
            b'#' if !in_q => return &line[..i],
            _ => {}
        }
    }
    line
}

/// Strip one layer of matching surrounding quotes.
fn unquote(s: &str) -> String {
    let b = s.as_bytes();
    if b.len() >= 2 && (b[0] == b'"' || b[0] == b'\'') && b[b.len() - 1] == b[0] {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

/// Decode an image (JPEG/PNG via the `image` crate; PPM P6 handled directly),
/// center-crop to square, resize to `size×size`, and return HWC f32 in `[0,1]`.
fn load_image_square(path: &Path, size: u32) -> Result<Vec<f32>, String> {
    let img = decode_rgb(path)?;
    let (w, h) = (img.width(), img.height());
    // Center square crop → resize (Lanczos3, matching diffusers' default resample).
    let side = w.min(h);
    let (x0, y0) = ((w - side) / 2, (h - side) / 2);
    let cropped = image::imageops::crop_imm(&img, x0, y0, side, side).to_image();
    let resized = image::imageops::resize(&cropped, size, size, image::imageops::FilterType::Lanczos3);
    Ok(resized.iter().map(|&b| b as f32 / 255.0).collect())
}

/// Decode `path` to an RGB8 image (JPEG/PNG/PPM — the enabled `image` codecs).
fn decode_rgb(path: &Path) -> Result<image::RgbImage, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read: {e}"))?;
    image::load_from_memory(&bytes).map(|i| i.to_rgb8()).map_err(|e| format!("decode: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yaml_parse_quotes_comments_and_jsonl_override() {
        let mut warn = |_: &str| {};
        let dir = std::env::temp_dir().join(format!("zimage_ds_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("captions.yaml"),
            "# subject cat\n\
             a.png: a photo of sks cat\n\
             b.jpg: \"a photo of sks cat, closeup\"  # trailing comment\n\
             c.png: prompt with a \"#hashtag\" inside\n",
        )
        .unwrap();
        let mut caps = parse_captions_yaml(&dir.join("captions.yaml"), &mut warn);
        assert_eq!(caps["a.png"], "a photo of sks cat");
        assert_eq!(caps["b.jpg"], "a photo of sks cat, closeup");
        assert_eq!(caps["c.png"], "prompt with a \"#hashtag\" inside");
        assert_eq!(caps.len(), 3);

        // jsonl overrides b.jpg and adds d.png
        std::fs::write(
            dir.join("captions.jsonl"),
            "{\"file\":\"b.jpg\",\"prompt\":\"OVERRIDDEN\"}\n{\"file\":\"d.png\",\"prompt\":\"added\"}\n",
        )
        .unwrap();
        apply_jsonl_overrides(&dir.join("captions.jsonl"), &mut caps, &mut warn);
        assert_eq!(caps["b.jpg"], "OVERRIDDEN");
        assert_eq!(caps["d.png"], "added");
        assert_eq!(caps["a.png"], "a photo of sks cat"); // untouched
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn decode_and_square_resize_png() {
        // Build a 6×4 RGB PNG in memory, load_dir it, expect a 32×32×3 sample.
        let dir = std::env::temp_dir().join(format!("zimage_ds_img_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut img = image::RgbImage::new(6, 4);
        for (x, _y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb([(x * 40) as u8, 128, 200]);
        }
        img.save(dir.join("x.png")).unwrap();
        std::fs::write(dir.join("captions.yaml"), "x.png: a test swatch\n").unwrap();
        let s = load_dir(&dir, 32, |_| {}).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].prompt, "a test swatch");
        assert_eq!(s[0].hwc.len(), 32 * 32 * 3);
        assert!(s[0].hwc.iter().all(|&v| (0.0..=1.0).contains(&v)));
        std::fs::remove_dir_all(&dir).ok();
    }
}
