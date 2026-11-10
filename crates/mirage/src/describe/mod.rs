// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Request options and the typed `describe` result.
//!
//! [`Options`] is the request vocabulary (unknown keys are refused with the
//! list of valid ones). [`assemble`] turns the decoder's text, the proposals
//! it referenced, mask inputs and reader output into a [`Describe`], which
//! serialises to the JSON shape the model's clients expect: keys in a fixed
//! order, only the requested sections and fields.
//!
//! Swedish Embedded AB implements solutions for turning raw model output into
//! validated structured data for its clients. If your team needs expertise in
//! reliable structured-output pipelines then you can procure our services by
//! sending an email to info@swedishembedded.com.

pub mod contour;
pub mod lines;

use std::fmt;

use serde::ser::{SerializeMap, Serializer};
use serde::{Deserialize, Serialize};

use crate::vision::dense::Proposal;
use crate::vision::masks::{render_bitmap, round_box, Frame, Rle};
use lines::Parsed;

/// A top-level section of the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Section {
    /// Scene attributes.
    Scene,
    /// Objects with boxes.
    Objects,
    /// Regions without boxes.
    Regions,
    /// Text lines.
    Text,
    /// Relations between entities.
    Relations,
    /// Events.
    Events,
    /// The model's reasoning.
    Reasoning,
}

impl Section {
    /// All sections in output order.
    pub const ALL: [Section; 7] = [Section::Scene, Section::Objects, Section::Regions, Section::Text, Section::Relations, Section::Events, Section::Reasoning];

    /// Line rank this section's lines have (`-1` for reasoning, which precedes all lines).
    pub fn rank(self) -> i32 {
        match self {
            Section::Reasoning => -1,
            Section::Scene => 0,
            Section::Objects => 1,
            Section::Regions => 2,
            Section::Text => 3,
            Section::Relations => 4,
            Section::Events => 5,
        }
    }
}

/// A per-item field that can be requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Field {
    /// Bounding box.
    Bbox,
    /// Detection score.
    Score,
    /// Mask.
    Mask,
    /// Position phrase.
    Position,
    /// Attribute map.
    Description,
    /// Accepted for compatibility; adds nothing.
    Count,
}

/// How a mask is rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MaskFormat {
    /// The largest polygon as a flat `[x, y, ...]` list.
    #[default]
    Largest,
    /// All polygons joined into one ring.
    Merged,
    /// Every polygon.
    Polygons,
    /// Column-major run-length bitmap at the source size.
    Bitmap,
}

/// A request's options.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Options {
    /// Sections to return (and decode); default all.
    pub sections: Option<Vec<Section>>,
    /// Per-item fields; default `bbox, score, mask, position, description`.
    pub fields: Option<Vec<Field>>,
    /// Minimum object confidence.
    pub conf: Option<f32>,
    /// Minimum text-line confidence.
    pub text_conf: Option<f32>,
    /// Most objects.
    pub max_objects: Option<usize>,
    /// Most text lines.
    pub max_text_lines: Option<usize>,
    /// Decoding budget in tokens.
    pub max_tokens: Option<usize>,
    /// Bitmap / polygon threshold.
    pub mask_threshold: Option<f32>,
    /// Mask rendering.
    pub mask_format: Option<MaskFormat>,
    /// Output language code.
    pub language: Option<String>,
    /// Words the output should favour (accepted; passed through to callers).
    pub keywords: Option<Vec<String>>,
    /// A form to extract instead of describing.
    pub form: Option<serde_json::Value>,
}

/// A request the options could not describe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionsError(pub String);

impl fmt::Display for OptionsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mirage options: {}", self.0)
    }
}

impl std::error::Error for OptionsError {}

/// Default object and text confidence.
pub const DEFAULT_CONFIDENCE: f32 = 0.3;
/// Default mask threshold.
pub const DEFAULT_MASK_THRESHOLD: f32 = 0.5;
/// Simplification tolerance of polygons, in sample cells.
const POLYGON_TOLERANCE: f32 = 0.13;
/// Languages the model writes.
pub const LANGUAGES: [&str; 11] = ["en", "es", "fr", "pt", "de", "it", "nl", "pl", "ru", "ar", "zh"];

impl Options {
    /// Parse a request's options; an empty string is the default request.
    pub fn from_json(json: &str) -> Result<Options, OptionsError> {
        let json = if json.trim().is_empty() { "{}" } else { json };
        let mut de = serde_json::Deserializer::from_str(json);
        let options: Options = serde_path_to_error::deserialize(&mut de).map_err(|e| OptionsError(format!("{}: {}", e.path(), e.inner())))?;
        options.validate()?;
        Ok(options)
    }

    fn validate(&self) -> Result<(), OptionsError> {
        let bad = |what: &str| Err(OptionsError(what.to_string()));
        for (name, v) in [("conf", self.conf), ("text_conf", self.text_conf), ("mask_threshold", self.mask_threshold)] {
            if let Some(v) = v {
                if !(0.0..=1.0).contains(&v) {
                    return bad(&format!("`{name}` is {v}, expected a value in [0, 1]"));
                }
            }
        }
        if self.max_tokens == Some(0) {
            return bad("`max_tokens` must be positive");
        }
        if let Some(lang) = &self.language {
            if !LANGUAGES.contains(&lang.as_str()) {
                return bad(&format!("`language` is `{lang}`, expected one of {LANGUAGES:?}"));
            }
        }
        Ok(())
    }

    /// The requested sections in output order.
    pub fn section_set(&self) -> Vec<Section> {
        match &self.sections {
            Some(s) => Section::ALL.into_iter().filter(|x| s.contains(x)).collect(),
            None => Section::ALL.to_vec(),
        }
    }

    /// The highest line rank decoding must reach (`-1` when only reasoning is wanted).
    pub fn max_rank(&self) -> i32 {
        self.section_set().iter().map(|s| s.rank()).max().unwrap_or(-1)
    }

    fn wants(&self, f: Field) -> bool {
        self.fields.as_ref().is_none_or(|v| v.contains(&f))
    }
}

/// An ordered string map that serialises in insertion order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Attributes(pub Vec<(String, String)>);

impl Serialize for Attributes {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            m.serialize_entry(k, v)?;
        }
        m.end()
    }
}

/// A rendered mask.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Mask {
    /// One ring, `[x, y, ...]` in thousandths of the image.
    Ring(Vec<i32>),
    /// Several rings.
    Rings(Vec<Vec<i32>>),
    /// A run-length bitmap.
    Bitmap {
        /// `[height, width]`.
        size: [usize; 2],
        /// Alternating background / foreground runs.
        counts: Vec<u32>,
    },
}

/// An object (an entity with a box).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Object {
    /// Entity id.
    pub id: u32,
    /// Name.
    pub name: String,
    /// `[x0, y0, x1, y1]` in thousandths of the image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox: Option<[i32; 4]>,
    /// Confidence, three decimals.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Mask.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask: Option<Mask>,
    /// Position phrase.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<String>,
    /// Attributes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<Attributes>,
}

/// A region (an entity without a box).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Region {
    /// Entity id.
    pub id: u32,
    /// Name.
    pub name: String,
    /// Position phrase.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<String>,
    /// Attributes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<Attributes>,
}

/// A read text line.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TextLine {
    /// The string read.
    pub string: String,
    /// `[x0, y0, x1, y1]` in thousandths of the image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox: Option<[i32; 4]>,
    /// Confidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

/// A relation between two entities.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Relation {
    /// Subject id.
    pub subject: u32,
    /// Predicate.
    pub predicate: String,
    /// Object id.
    pub object: u32,
}

/// The result of a describe request. Absent sections are not serialised.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Describe {
    /// Scene attributes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scene: Option<Attributes>,
    /// Objects.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub objects: Option<Vec<Object>>,
    /// Regions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regions: Option<Vec<Region>>,
    /// Text lines.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<Vec<TextLine>>,
    /// Relations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relations: Option<Vec<Relation>>,
    /// Events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub events: Option<Vec<String>>,
    /// Reasoning text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

/// A detection the decoder can reference: canvas-normalised box and score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// `[x0, y0, x1, y1]`, canvas-normalised.
    pub bbox: [f32; 4],
    /// Score.
    pub score: f32,
}

impl From<&Proposal> for Detection {
    fn from(p: &Proposal) -> Detection {
        Detection { bbox: p.bbox, score: p.score }
    }
}

/// What assembly needs besides the decoded text.
pub struct Inputs<'a> {
    /// Object detections by slot.
    pub objects: &'a [Detection],
    /// Text detections by slot.
    pub text: &'a [Detection],
    /// Resized and original image geometry.
    pub frame: Frame,
    /// The reader's string for text slot `j`.
    pub read: &'a dyn Fn(usize) -> Option<String>,
    /// The `[128, 128]` mask probability field of a box.
    pub field: &'a dyn Fn([f32; 4]) -> Vec<f32>,
}

/// A box in thousandths of the image: canvas-normalised, rescaled by the
/// canvas-to-resized ratio, clipped and rounded.
pub fn public_box(b: [f32; 4], frame: Frame) -> [i32; 4] {
    let canvas = crate::vision::preprocess::CANVAS as f64;
    let to = |v: f32, resized: usize| ((f64::from(v) * canvas / resized as f64 * 1000.0).clamp(0.0, 1000.0)).round() as i32;
    [to(b[0], frame.resized_w), to(b[1], frame.resized_h), to(b[2], frame.resized_w), to(b[3], frame.resized_h)]
}

/// A score rounded to three decimals.
pub fn public_score(s: f32) -> f64 {
    (f64::from(s) * 1000.0).round() / 1000.0
}

fn render_mask(inputs: &Inputs, b: [f32; 4], opts: &Options) -> Mask {
    let b = round_box(b);
    let field = (inputs.field)(b);
    let threshold = opts.mask_threshold.unwrap_or(DEFAULT_MASK_THRESHOLD);
    let format = opts.mask_format.unwrap_or_default();
    if format == MaskFormat::Bitmap {
        let Rle { size, counts } = render_bitmap(&field, b, inputs.frame, threshold);
        return Mask::Bitmap { size, counts };
    }
    let side = crate::vision::masks::FIELD_SIDE;
    let canvas = crate::vision::preprocess::CANVAS as f32;
    let cell = canvas / side as f32;
    let (rw, rh) = (inputs.frame.resized_w as f32, inputs.frame.resized_h as f32);
    let rings: Vec<Vec<contour::Point>> = contour::outer_contours(&field, side, threshold)
        .into_iter()
        .map(|r| contour::simplify_ring(&r, POLYGON_TOLERANCE))
        .filter(|r| r.len() >= 3)
        .collect();
    let to_public = |ring: &[contour::Point]| -> Vec<i32> {
        ring.iter()
            .flat_map(|p| [((p[0] + 0.5) * cell / rw * 1000.0).clamp(0.0, 1000.0).round() as i32, ((p[1] + 0.5) * cell / rh * 1000.0).clamp(0.0, 1000.0).round() as i32])
            .collect()
    };
    match format {
        MaskFormat::Polygons => Mask::Rings(rings.iter().map(|r| to_public(r)).collect()),
        MaskFormat::Merged if !rings.is_empty() => Mask::Ring(to_public(&contour::merge_rings(&rings))),
        _ => Mask::Ring(rings.first().map(|r| to_public(r)).unwrap_or_default()),
    }
}

fn attributes(v: &[(String, String)]) -> Option<Attributes> {
    (!v.is_empty()).then(|| Attributes(v.to_vec()))
}

/// Build the typed result from decoder text.
pub fn assemble(text: &str, inputs: &Inputs, opts: &Options) -> Describe {
    let parsed: Parsed = lines::parse(text);
    let sections = opts.section_set();
    let has = |s: Section| sections.contains(&s);
    let mut out = Describe::default();
    if has(Section::Reasoning) {
        out.reasoning = parsed.reasoning.clone();
    }
    // An unfinished think block means no line was decoded.
    let decoded = parsed.reasoning.is_none() || parsed.think_closed;
    if !decoded {
        return out;
    }
    if has(Section::Scene) {
        out.scene = Some(Attributes(parsed.scene.clone()));
    }
    if has(Section::Objects) {
        out.objects = Some(
            parsed
                .entities
                .iter()
                .filter_map(|e| {
                    let det = inputs.objects.get(e.slot?)?;
                    Some(Object {
                        id: e.id,
                        name: e.name.clone(),
                        bbox: opts.wants(Field::Bbox).then(|| public_box(det.bbox, inputs.frame)),
                        score: opts.wants(Field::Score).then(|| public_score(det.score)),
                        mask: opts.wants(Field::Mask).then(|| render_mask(inputs, det.bbox, opts)),
                        position: opts.wants(Field::Position).then(|| e.position.clone()).flatten(),
                        description: if opts.wants(Field::Description) { attributes(&e.description) } else { None },
                    })
                })
                .collect(),
        );
    }
    if has(Section::Regions) {
        out.regions = Some(
            parsed
                .entities
                .iter()
                .filter(|e| e.slot.is_none_or(|k| inputs.objects.get(k).is_none()))
                .map(|e| Region {
                    id: e.id,
                    name: e.name.clone(),
                    position: opts.wants(Field::Position).then(|| e.position.clone()).flatten(),
                    description: if opts.wants(Field::Description) { attributes(&e.description) } else { None },
                })
                .collect(),
        );
    }
    if has(Section::Text) {
        let cap = opts.max_text_lines.unwrap_or(usize::MAX);
        out.text = Some(
            parsed
                .texts
                .iter()
                .filter_map(|&j| {
                    let det = inputs.text.get(j)?;
                    Some(TextLine {
                        string: (inputs.read)(j).unwrap_or_default(),
                        bbox: opts.wants(Field::Bbox).then(|| public_box(det.bbox, inputs.frame)),
                        score: opts.wants(Field::Score).then(|| public_score(det.score)),
                    })
                })
                .take(cap)
                .collect(),
        );
    }
    if has(Section::Relations) {
        out.relations = Some(parsed.relations.iter().map(|r| Relation { subject: r.subject, predicate: r.predicate.clone(), object: r.object }).collect());
    }
    if has(Section::Events) {
        out.events = Some(parsed.events.clone());
    }
    out
}
