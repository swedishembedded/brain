// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Captioned-image folder dataset (LoRA/finetune training): a folder of images
//! each paired with a text prompt. Hoisted from `s3dit::dataset` so every
//! image-conditioned trainer (Z-Image, FLUX.2, …) shares the ONE loader;
//! `s3dit::dataset` re-exports this module unchanged.
//!
//! Prompts come from a caption file in the folder. Two formats, in priority order:
//!  1. **`captions.yaml`** (primary - easy to hand-edit): a YAML mapping of
//!     `filename: prompt`, deserialized into [`CaptionFile`].
//!  2. **`captions.jsonl`** (override): one JSON object per line, deserialized
//!     into [`CaptionLine`]. Entries here **override / add to** the YAML ones -
//!     so you can keep a readable YAML base and script exceptions.
//!
//! Example `captions.yaml`:
//! ```yaml
//! # a subject token like "sks" helps the adapter bind to your concept
//! cat01.png: a photo of sks cat sitting on a chair
//! cat02.jpg: "a photo of sks cat, closeup, studio light"
//! cat03.jpg: |-
//!   A photo of sks cat curled on a wicker chair by a south-facing window.
//!   Warm afternoon light rakes across its fur from the left; the background
//!   falls off into a soft, unlit hallway.
//! ```
//!
//! **Both files are parsed by real parsers into typed schemas** - `serde_norway`
//! for the YAML, `serde_json` for the JSONL - never by a line scanner of our
//! own. This file is hand-edited, which means it eventually meets every corner
//! of the YAML grammar: anchors, aliases, multiple documents, quoted colons,
//! tabs, CRLF, and the block scalars below. A hand-rolled subset silently
//! mis-parses the first construct it does not implement, which is how `key: |`
//! once produced the literal one-character prompt `"|"` instead of a multi-line
//! caption. A real parser either understands the input or says where it failed.
//!
//! **Block scalars** are what make a *detailed* caption editable, and a detailed
//! caption is the training signal - a folder of one-line prompts caps whatever
//! is trained on it. `key: |` keeps the line breaks (`|-`/`|+` decide the
//! trailing ones) and `key: >` folds wrapped lines into spaces. Inside either,
//! `#` is caption text, not a comment.
//!
//! [`write_captions_yaml`] emits that form for any caption containing a
//! newline, and [`read_captions_yaml`] reads it back byte-for-byte, so a labeler
//! can write a caption set, a human can edit it in place, and the trainer sees
//! exactly what is on disk.
//!
//! Images are decoded (JPEG/PNG/PPM), center-cropped to square, and resized to the
//! training size, yielding interleaved-RGB HWC f32 in `[0,1]` - exactly what the VAE
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
    let mut caps = read_captions_yaml(&dir.join("captions.yaml"), &mut warn);
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
            warn(&format!("caption references missing image {file} - skipping"));
            continue;
        }
        // An entry with no caption text is an inventory line, not a sample:
        // the labeler lists an image it could not caption so the file covers
        // the whole folder, and re-running the labeler fills it in. Training
        // on an empty prompt would quietly teach the empty string.
        if prompt.trim().is_empty() {
            warn(&format!("{file} has no caption yet - skipping (re-run `brain label` to fill it in)"));
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

/// The `captions.yaml` schema: a mapping of image file name to caption.
///
/// This is the whole of what a caption file may contain, stated as a type
/// rather than as parser behaviour. A newtype over `BTreeMap` rather than a
/// bare alias so the schema has a name to document and to point errors at, and
/// so the iteration order is the file-name order `load_dir` depends on for
/// deterministic sample ordering.
#[derive(Debug, Default, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct CaptionFile(pub BTreeMap<String, String>);

impl CaptionFile {
    pub fn into_inner(self) -> BTreeMap<String, String> {
        self.0
    }
}

/// One `captions.jsonl` line: an override for a single image's caption.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CaptionLine {
    pub file: String,
    pub prompt: String,
}

/// Read `captions.yaml` into the caption map.
///
/// A missing file is an empty map rather than an error - the folder may be
/// captioned entirely by `captions.jsonl`. A file that is present but does not
/// parse is reported through `warn` and treated as empty, so one bad edit
/// cannot abort a long run; the message carries the parser's own line/column.
pub fn read_captions_yaml(path: &Path, warn: &mut impl FnMut(&str)) -> BTreeMap<String, String> {
    let Ok(text) = std::fs::read_to_string(path) else { return BTreeMap::new() };
    match serde_norway::from_str::<CaptionFile>(&text) {
        Ok(f) => f.into_inner(),
        Err(e) => {
            warn(&format!("{}: {e}", path.display()));
            BTreeMap::new()
        }
    }
}

/// Write `caps` as a `captions.yaml` that [`read_captions_yaml`] reads back
/// **byte-for-byte**, including every embedded newline.
///
/// A caption containing a newline is emitted as a literal block scalar, which
/// is the form a human can edit: the text sits on its own lines at a fixed
/// indent with no escaping, no quoting, and no significance to `#` or `:`. That
/// choice is the serializer's, not ours - `serde_norway` emits
/// `ScalarStyle::Literal` for any string containing a newline, and falls back
/// to a quoted form for the strings a block scalar cannot represent exactly
/// (a line with trailing whitespace, for instance). Preferring exactness over
/// prettiness in those cases is what keeps the round trip total.
/// An empty caption is allowed and meaningful: it records an image the labeler
/// saw but could not caption, so the file is a complete inventory of the folder
/// rather than a list of the ones that worked. [`load_dir`] skips such an entry
/// with a warning and `brain label` treats it as outstanding work, so nothing
/// downstream mistakes it for a real caption.
pub fn write_captions_yaml(path: &Path, caps: &BTreeMap<String, String>) -> Result<(), String> {
    let text = serde_norway::to_string(&CaptionFile(caps.clone())).map_err(|e| format!("captions.yaml: {e}"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("captions.yaml: {}: {e}", parent.display()))?;
    }
    std::fs::write(path, text).map_err(|e| format!("captions.yaml: {}: {e}", path.display()))
}

/// Overlay `captions.jsonl` entries onto `caps` (override/add), one
/// [`CaptionLine`] per line.
fn apply_jsonl_overrides(path: &Path, caps: &mut BTreeMap<String, String>, warn: &mut impl FnMut(&str)) {
    let Ok(text) = std::fs::read_to_string(path) else { return };
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<CaptionLine>(line) {
            Ok(e) if !e.file.is_empty() && !e.prompt.is_empty() => {
                caps.insert(e.file, e.prompt);
            }
            Ok(_) => warn(&format!("captions.jsonl:{}: need non-empty \"file\" and \"prompt\"", i + 1)),
            Err(e) => warn(&format!("captions.jsonl:{}: {e}", i + 1)),
        }
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

/// Decode `path` to an RGB8 image (JPEG/PNG/PPM - the enabled `image` codecs).
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
        let dir = std::env::temp_dir().join(format!("imageset_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("captions.yaml"),
            "# subject cat\n\
             a.png: a photo of sks cat\n\
             b.jpg: \"a photo of sks cat, closeup\"  # trailing comment\n\
             c.png: prompt with a \"#hashtag\" inside\n",
        )
        .unwrap();
        let mut caps = read_captions_yaml(&dir.join("captions.yaml"), &mut warn);
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
        let dir = std::env::temp_dir().join(format!("imageset_img_{}", std::process::id()));
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
