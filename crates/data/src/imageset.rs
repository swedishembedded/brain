// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Captioned-image folder dataset (LoRA/finetune training): a folder of images
//! each paired with a text prompt. Hoisted from `s3dit::dataset` so every
//! image-conditioned trainer (Z-Image, FLUX.2, …) shares the ONE loader;
//! `s3dit::dataset` re-exports this module unchanged.
//!
//! Prompts come from a caption file in the folder. Two formats, in priority order:
//!  1. **`captions.yaml`** (primary - easy to hand-edit): a flat mapping of
//!     `filename: prompt`. A value is either a single line (quoted or bare; `#`
//!     starts a comment) or a **block scalar** spanning as many lines as it
//!     needs; blank lines between entries are ignored.
//!  2. **`captions.jsonl`** (override): one JSON object per line,
//!     `{"file": "...", "prompt": "..."}`. Entries here **override / add to** the
//!     YAML ones - so you can keep a readable YAML base and script exceptions.
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
//! **Block scalars** are what make a *detailed* caption editable, and a detailed
//! caption is the training signal - a folder of one-line prompts caps whatever
//! is trained on it. Both YAML forms are read:
//!
//! * `key: |` keeps the line breaks. `|-` drops the trailing newline, bare `|`
//!   keeps exactly one, `|+` keeps every one.
//! * `key: >` folds wrapped lines into spaces, with a blank line becoming a real
//!   line break - the form to reach for when the caption is one long paragraph
//!   that should soft-wrap in the editor.
//! * The body is every following line indented further than the key; the first
//!   non-empty one sets the indent, or an explicit indicator (`|2-`) states it.
//!   Inside a block, `#` is caption text, not a comment.
//!
//! [`write_captions_yaml`] emits that form, and [`read_captions_yaml`] reads it
//! back byte-for-byte - a labeler can write a caption set, a human can edit it in
//! place, and the trainer sees exactly what is on disk. Every single-line
//! spelling that parsed before block scalars existed still parses identically;
//! the one changed input is a value that is *only* a bare `|` or `>`, which used
//! to yield that character as the prompt and now opens a block.
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

/// Read the flat `filename: prompt` YAML subset from `path`. Tolerant by design
/// (this is a hand-edited file): `#` comments, blank lines, optional quotes and
/// [block scalars](self) are handled; anything else is warned about and skipped
/// rather than aborting the load. A missing file is an empty map, not an error.
pub fn read_captions_yaml(path: &Path, warn: &mut impl FnMut(&str)) -> BTreeMap<String, String> {
    let Ok(text) = std::fs::read_to_string(path) else { return BTreeMap::new() };
    parse_captions_yaml(&text, warn)
}

/// [`read_captions_yaml`] over text already in memory.
fn parse_captions_yaml(text: &str, warn: &mut impl FnMut(&str)) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let raw = lines[i];
        let indent = raw.len() - raw.trim_start().len();
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            i += 1;
            continue;
        }
        let Some((key, val)) = line.split_once(':') else {
            warn(&format!("captions.yaml:{}: no ':' - skipping `{raw}`", i + 1));
            i += 1;
            continue;
        };
        let file = key.trim().trim_matches(|c| c == '"' || c == '\'').trim();
        // A value that is nothing but a block header opens a block scalar; the
        // body is the following lines indented past THIS line's own indent.
        let (prompt, consumed) = match BlockHeader::parse(val.trim()) {
            Some(h) => {
                let (v, n) = read_block_body(&lines[i + 1..], indent, &h);
                (v, 1 + n)
            }
            None => (unquote(val.trim()), 1),
        };
        let lineno = i + 1;
        i += consumed;
        if file.is_empty() || prompt.is_empty() {
            warn(&format!("captions.yaml:{lineno}: empty filename or prompt - skipping"));
            continue;
        }
        out.insert(file.to_string(), prompt);
    }
    out
}

/// How a block scalar folds its line breaks and what it does with the trailing
/// ones - the `|`/`>` and `-`/`+` indicators, plus an optional explicit indent.
struct BlockHeader {
    /// `true` for `|` (keep line breaks), `false` for `>` (fold them to spaces).
    literal: bool,
    chomp: Chomp,
    /// An explicit indentation indicator (`|2-`), if the author stated one.
    indent: Option<usize>,
}

/// What happens to the newlines at the very end of a block scalar.
#[derive(PartialEq, Eq)]
enum Chomp {
    /// `-`: drop them all.
    Strip,
    /// (none): keep exactly one.
    Clip,
    /// `+`: keep every one.
    Keep,
}

impl BlockHeader {
    /// Parse the text after `key:`, or `None` if it is an ordinary scalar.
    /// Accepts `|`, `>`, each with an optional indentation digit and an
    /// optional `-`/`+`, and an optional trailing `#` comment.
    fn parse(val: &str) -> Option<BlockHeader> {
        let mut c = val.chars();
        let literal = match c.next()? {
            '|' => true,
            '>' => false,
            _ => return None,
        };
        let rest: String = c.collect();
        let rest = rest.split('#').next().unwrap_or("").trim().to_string();
        let mut indent = None;
        let mut chomp = Chomp::Clip;
        for ch in rest.chars() {
            match ch {
                '-' => chomp = Chomp::Strip,
                '+' => chomp = Chomp::Keep,
                d if d.is_ascii_digit() => indent = Some(d.to_digit(10)? as usize),
                _ => return None, // not a block header after all - leave it alone
            }
        }
        Some(BlockHeader { literal, chomp, indent })
    }
}

/// Collect a block scalar's body from the lines after its header. `key_indent`
/// is the header line's indentation: the block ends at the first non-empty line
/// indented no further than that. Returns the value and how many lines it ate.
fn read_block_body(rest: &[&str], key_indent: usize, h: &BlockHeader) -> (String, usize) {
    // The content indent is either stated by the header or taken from the first
    // non-empty line. Taking it from the first line is what YAML does, and it is
    // why a value whose first line is itself indented needs the explicit form.
    let mut n = 0;
    let mut body: Vec<String> = Vec::new();
    let content_indent = h.indent.map(|d| key_indent + d).or_else(|| {
        rest.iter()
            .take_while(|l| l.trim().is_empty() || indent_of(l) > key_indent)
            .find(|l| !l.trim().is_empty())
            .map(|l| indent_of(l))
    });
    let Some(content_indent) = content_indent else { return (String::new(), 0) };
    for line in rest {
        // Indentation is checked BEFORE emptiness: a line of nothing but spaces
        // that still reaches the content indent contributes its extra spaces,
        // and treating it as blank would silently eat them.
        if line.len() >= content_indent && indent_of(line) >= content_indent {
            body.push(line[content_indent..].to_string());
        } else if line.trim().is_empty() {
            body.push(String::new());
        } else {
            break;
        }
        n += 1;
    }
    // Trailing blank lines are the chomping indicator's business, not content.
    let mut trailing = 0;
    while body.last().is_some_and(|l| l.is_empty()) {
        body.pop();
        trailing += 1;
    }
    let mut value = if h.literal { body.join("\n") } else { fold(&body) };
    match h.chomp {
        Chomp::Strip => {}
        // "Clip" keeps one newline as long as the block had any content line
        // that was actually terminated - which, read from `lines()`, is always.
        Chomp::Clip => value.push('\n'),
        Chomp::Keep => {
            value.push('\n');
            for _ in 0..trailing {
                value.push('\n');
            }
        }
    }
    (value, n)
}

/// Leading-space count of a line.
fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// Fold a `>` block: consecutive non-empty lines join with a space, and a blank
/// line between them becomes one real line break.
fn fold(body: &[String]) -> String {
    let mut out = String::new();
    let mut pending_break = false;
    for line in body {
        if line.is_empty() {
            pending_break = true;
            continue;
        }
        if !out.is_empty() {
            out.push(if pending_break { '\n' } else { ' ' });
        }
        pending_break = false;
        out.push_str(line);
    }
    out
}

/// Write `caps` as a `captions.yaml` that [`read_captions_yaml`] reads back
/// **byte-for-byte**, including every embedded newline.
///
/// Every caption is emitted as a literal block scalar, because that is the form
/// a human can edit: the text sits on its own lines at a fixed indent with no
/// escaping, no quoting, and no significance to `#` or `:`. The chomping
/// indicator records how many trailing newlines the caption had, and a caption
/// whose first line is itself indented gets the explicit indentation indicator
/// so those leading spaces survive the round trip.
pub fn write_captions_yaml(path: &Path, caps: &BTreeMap<String, String>) -> Result<(), String> {
    let mut text = String::new();
    for (file, prompt) in caps {
        if prompt.is_empty() {
            return Err(format!("captions.yaml: empty caption for {file} (the loader would skip it)"));
        }
        let trailing = prompt.len() - prompt.trim_end_matches('\n').len();
        let chomp = match trailing {
            0 => "-",
            1 => "",
            _ => "+",
        };
        // State the indent whenever inferring it would be wrong, i.e. whenever
        // the caption's first line starts with whitespace of its own.
        let body = prompt.trim_end_matches('\n');
        let explicit = if body.starts_with([' ', '\t']) { "2" } else { "" };
        text.push_str(&format!("{file}: |{explicit}{chomp}\n"));
        for line in body.split('\n') {
            if line.is_empty() {
                text.push('\n');
            } else {
                text.push_str(&format!("  {line}\n"));
            }
        }
        // `|+` keeps every trailing newline, so write them out as blank lines.
        for _ in 1..trailing {
            text.push('\n');
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("captions.yaml: {}: {e}", parent.display()))?;
    }
    std::fs::write(path, text).map_err(|e| format!("captions.yaml: {}: {e}", path.display()))
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

/// Drop a trailing `#` comment (outside of quotes - captions rarely quote, but a
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
