// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements dataset pipelines for training perception models for its clients. If your team
// needs expertise in turning annotated images into masked, task-aware training targets then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The training record: a versioned JSONL schema with task-presence semantics, and its loader.
//!
//! One line is one image. A label family is *present* when its key is present (even as an empty list): an empty
//! `objects` list says "there are no objects", a missing `objects` key says "nobody labelled objects". Every
//! loss treats an absent family as unknown, never as a negative: no heatmap negatives, no unsupervised stop
//! token, no first-token supervision after a gap (see `objective`).
//!
//! ```json
//! {"version":1,"image":"images/a.jpg","language":"en",
//!  "scene":{"caption":"Ivy on a wall."},
//!  "objects":[{"id":1,"name":"clock","bbox":[726,135,784,211],"mask_polygons":[[753.5,135.2,780.1,152.3,...]]}],
//!  "text":[{"value":"EXIT","bbox":[112,246,765,322]}],
//!  "relations":[{"subject":1,"predicate":"above","object":2}],
//!  "closed":false}
//! ```
//!
//! Coordinates are thousandths of the original image, `[x0, y0, x1, y1]`; polygons are flat `[x, y, ...]`
//! rings in the same units. `closed` asserts that nothing exists beyond the labelled families, which is what
//! allows the end-of-sequence token to be supervised.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;

/// The schema version this build reads and writes.
pub const SCHEMA_VERSION: u32 = 1;
/// Most objects one record may carry (the decoder has this many object reference tokens at release size).
pub const MAX_OBJECT_LABELS: usize = 64;
/// Most text lines one record may carry.
pub const MAX_TEXT_LABELS: usize = 32;
/// Most pixels of a decoded training image.
pub const MAX_IMAGE_PIXELS: u64 = 64_000_000;
/// Longest accepted text string, in characters.
pub const MAX_TEXT_CHARS: usize = 512;
/// Most polygon points of one object.
pub const MAX_POLYGON_POINTS: usize = 4096;

fn default_version() -> u32 {
    SCHEMA_VERSION
}

/// `scene` label.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SceneLabel {
    /// One-sentence description.
    pub caption: Option<String>,
}

/// One object.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ObjectLabel {
    /// Dataset-local id (relations refer to it).
    pub id: u32,
    /// Class or phrase.
    pub name: String,
    /// `[x0, y0, x1, y1]` in thousandths of the image.
    pub bbox: [f64; 4],
    /// Polygon rings of the mask, when labelled (an absent mask is unknown, not empty).
    pub mask_polygons: Option<Vec<Vec<f64>>>,
    /// Position phrase.
    pub position: Option<String>,
    /// Attributes, written in key order.
    pub attributes: Option<BTreeMap<String, String>>,
}

/// One text line.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TextLabel {
    /// The string on the line.
    pub value: String,
    /// `[x0, y0, x1, y1]` in thousandths of the image.
    pub bbox: [f64; 4],
}

/// One relation between objects.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RelationLabel {
    /// Subject object id.
    pub subject: u32,
    /// Predicate words.
    pub predicate: String,
    /// Object object id.
    pub object: u32,
}

/// One training record.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Record {
    /// Schema version (missing means the current one).
    #[serde(default = "default_version")]
    pub version: u32,
    /// Image path relative to the dataset root.
    pub image: String,
    /// Output language code (default `en`).
    pub language: Option<String>,
    /// Scene label family.
    pub scene: Option<SceneLabel>,
    /// Object label family.
    pub objects: Option<Vec<ObjectLabel>>,
    /// Text label family.
    pub text: Option<Vec<TextLabel>>,
    /// Relation label family.
    pub relations: Option<Vec<RelationLabel>>,
    /// Nothing exists beyond the labelled families.
    #[serde(default)]
    pub closed: bool,
    /// Provenance, carried through untouched.
    pub license: Option<String>,
    /// Provenance, carried through untouched.
    pub source: Option<String>,
}

/// Why a record or a file was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordError(pub String);

impl std::fmt::Display for RecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RecordError {}

fn check_box(what: &str, b: &[f64; 4]) -> Result<(), RecordError> {
    let finite = b.iter().all(|v| v.is_finite() && (0.0..=1000.0).contains(v));
    if !finite || b[2] <= b[0] || b[3] <= b[1] {
        return Err(RecordError(format!("{what}: bbox {b:?} is not an ordered box inside 0..1000")));
    }
    Ok(())
}

impl Record {
    /// Parse one JSON line and validate it.
    pub fn parse(line: &str) -> Result<Record, RecordError> {
        let rec: Record = serde_json::from_str(line).map_err(|e| RecordError(format!("record: {e}")))?;
        rec.validate()?;
        Ok(rec)
    }

    /// The language code, defaulting to English.
    pub fn language(&self) -> &str {
        self.language.as_deref().unwrap_or("en")
    }

    /// Check bounds and cross-references.
    pub fn validate(&self) -> Result<(), RecordError> {
        if self.version != SCHEMA_VERSION {
            return Err(RecordError(format!("record: schema version {} is not the supported {SCHEMA_VERSION}", self.version)));
        }
        let path = Path::new(&self.image);
        if self.image.is_empty() || path.is_absolute() || path.components().any(|c| matches!(c, Component::ParentDir | Component::Prefix(_))) {
            return Err(RecordError(format!("record: image path `{}` must be relative and stay inside the dataset", self.image)));
        }
        if let Some(objects) = &self.objects {
            if objects.len() > MAX_OBJECT_LABELS {
                return Err(RecordError(format!("record: {} objects exceed the {MAX_OBJECT_LABELS} the model can reference", objects.len())));
            }
            let mut ids = std::collections::BTreeSet::new();
            for o in objects {
                check_box(&format!("object {}", o.id), &o.bbox)?;
                if !ids.insert(o.id) {
                    return Err(RecordError(format!("record: object id {} appears twice", o.id)));
                }
                for ring in o.mask_polygons.iter().flatten() {
                    if ring.len() < 6 || ring.len() % 2 != 0 || ring.len() > MAX_POLYGON_POINTS * 2 || ring.iter().any(|v| !v.is_finite()) {
                        return Err(RecordError(format!("object {}: a polygon needs 3..={MAX_POLYGON_POINTS} finite points", o.id)));
                    }
                }
            }
            for rel in self.relations.iter().flatten() {
                if !ids.contains(&rel.subject) || !ids.contains(&rel.object) {
                    return Err(RecordError(format!("record: relation {} {} {} names an unlabelled object", rel.subject, rel.predicate, rel.object)));
                }
            }
        } else if self.relations.is_some() {
            return Err(RecordError("record: relations without objects".into()));
        }
        if let Some(text) = &self.text {
            if text.len() > MAX_TEXT_LABELS {
                return Err(RecordError(format!("record: {} text lines exceed the {MAX_TEXT_LABELS} the model can reference", text.len())));
            }
            for t in text {
                check_box("text", &t.bbox)?;
                if t.value.chars().count() > MAX_TEXT_CHARS {
                    return Err(RecordError(format!("record: a text line is longer than {MAX_TEXT_CHARS} characters")));
                }
            }
        }
        Ok(())
    }
}

/// Read a JSONL file of records (blank lines skipped), naming the line of the first bad record.
pub fn read_jsonl(path: &Path) -> Result<Vec<Record>, RecordError> {
    let text = std::fs::read_to_string(path).map_err(|e| RecordError(format!("{}: {e}", path.display())))?;
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| Record::parse(l).map_err(|e| RecordError(format!("{} line {}: {e}", path.display(), i + 1))))
        .collect()
}

/// [`read_jsonl`] for datasets whose boxes may overhang the image by a few pixels (text rendered near an edge):
/// every box is clamped to `0..1000` and a label whose box collapses is dropped. The strict reader is the default;
/// this is the explicit opt-in.
pub fn read_jsonl_clamped(path: &Path) -> Result<Vec<Record>, RecordError> {
    let text = std::fs::read_to_string(path).map_err(|e| RecordError(format!("{}: {e}", path.display())))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let fail = |e: String| RecordError(format!("{} line {}: {e}", path.display(), i + 1));
        let mut value: serde_json::Value = serde_json::from_str(line).map_err(|e| fail(e.to_string()))?;
        for family in ["objects", "text"] {
            let Some(list) = value.get_mut(family).and_then(serde_json::Value::as_array_mut) else { continue };
            list.retain_mut(|label| {
                let Some(b) = label.get_mut("bbox").and_then(serde_json::Value::as_array_mut) else { return true };
                let c: Vec<f64> = b.iter().filter_map(serde_json::Value::as_f64).map(|v| v.clamp(0.0, 1000.0)).collect();
                if c.len() != 4 || c[2] <= c[0] || c[3] <= c[1] {
                    return false;
                }
                *b = c.into_iter().map(serde_json::Value::from).collect();
                true
            });
        }
        out.push(Record::parse(&value.to_string()).map_err(|e| fail(e.to_string()))?);
    }
    Ok(out)
}

/// A decoded image: `h x w` RGB bytes.
pub struct Picture {
    /// Height.
    pub h: usize,
    /// Width.
    pub w: usize,
    /// RGB bytes.
    pub rgb: Vec<u8>,
}

/// Decode the record's image under `root` with a pixel bound.
pub fn load_picture(root: &Path, rec: &Record) -> Result<Picture, RecordError> {
    let path: PathBuf = root.join(&rec.image);
    let bytes = std::fs::read(&path).map_err(|e| RecordError(format!("{}: {e}", path.display())))?;
    let img = imaging::codec::decode_bounded(&bytes, MAX_IMAGE_PIXELS).map_err(|e| RecordError(format!("{}: {e}", path.display())))?;
    Ok(Picture { h: img.h as usize, w: img.w as usize, rgb: img.px })
}
