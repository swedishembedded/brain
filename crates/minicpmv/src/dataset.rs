// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The multimodal fine-tuning dataset: `generic-messages-v2` records - one
//! conversation per JSONL line, `{"messages": [...], "tools": [...],
//! "metadata": {...}}`, every message carrying an explicit `train` flag -
//! whose message `content` may also be a list of typed parts:
//!
//! ```json
//! {"messages": [
//!   {"role": "user", "train": false, "content": [
//!     {"type": "image", "path": "images/0001.png"},
//!     {"type": "video", "path": "clips/0001.mp4"},
//!     {"type": "text", "text": "What happens after the light changes?"}]},
//!   {"role": "assistant", "train": true, "content": "The car starts moving."}]}
//! ```
//!
//! Media is referenced by path (relative to the JSONL file), never embedded,
//! and validated before training starts. Parsing is strict: an unknown field,
//! a missing `train`, a media part in a supervised turn or a part of unknown
//! type is an error naming the line.
//!
//! Each supervised assistant turn becomes one [`Sample`], rendered as the
//! model is asked for it at inference: the template's prompt for the
//! conversation before the turn (with the generation prompt), then the turn
//! itself as the supervised span. The prompt and the answer are tokenized
//! separately, exactly as inference tokenizes a prompt and then generates.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::preprocess::{process_image, ImageOptions, ProcessedImage};
use crate::prompt::{view_layout, Formatter, IMAGE_PAD};

/// A media file one record refers to.
#[derive(Clone, Debug, PartialEq)]
pub enum Media {
    Image(PathBuf),
    Video(PathBuf),
}

/// One record: the conversation in the chat template's own message form
/// (media parts as `{"type": "image"}` / `{"type": "video"}`), the media in
/// placeholder order, and which messages are supervised.
#[derive(Clone, Debug)]
pub struct Record {
    pub messages: Vec<Value>,
    pub tools: Vec<Value>,
    pub media: Vec<Media>,
    pub train: Vec<bool>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRecord {
    messages: Vec<WireMessage>,
    #[serde(default)]
    tools: Vec<Value>,
    #[allow(dead_code)]
    #[serde(default)]
    metadata: Value,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WireMessage {
    role: String,
    content: Value,
    train: bool,
    #[serde(default)]
    tool_calls: Vec<Value>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

/// Parse one record; paths resolve against `base`.
fn parse_record(line: &str, base: &Path) -> Result<Record, String> {
    let wire: WireRecord = serde_json::from_str(line).map_err(|e| e.to_string())?;
    if wire.messages.is_empty() {
        return Err("\"messages\" is empty".into());
    }
    let mut messages = Vec::new();
    let mut media = Vec::new();
    let mut train = Vec::new();
    for m in wire.messages {
        if !matches!(m.role.as_str(), "system" | "user" | "assistant" | "tool") {
            return Err(format!("role {:?} is not system, user, assistant or tool", m.role));
        }
        if m.train && m.role != "assistant" {
            return Err(format!("a {} message is marked train: only assistant turns are supervised", m.role));
        }
        let content = match m.content {
            Value::String(s) => Value::String(s),
            Value::Array(parts) => {
                let mut out = Vec::new();
                for p in parts {
                    let kind = p.get("type").and_then(Value::as_str).ok_or("a content part has no \"type\"")?;
                    match kind {
                        "text" => out.push(json!({"type": "text", "text": p.get("text").and_then(Value::as_str).ok_or("a text part has no \"text\"")?})),
                        "image" | "video" => {
                            if m.train {
                                return Err(format!("a supervised turn carries a {kind} part - the model is trained to write text, not media"));
                            }
                            let rel = p.get("path").and_then(Value::as_str).ok_or_else(|| format!("a {kind} part has no \"path\""))?;
                            let path = base.join(rel);
                            if !path.is_file() {
                                return Err(format!("{kind} {} does not exist", path.display()));
                            }
                            media.push(if kind == "image" { Media::Image(path) } else { Media::Video(path) });
                            out.push(json!({"type": kind}));
                        }
                        other => return Err(format!("content part type {other:?} is not text, image or video")),
                    }
                }
                Value::Array(out)
            }
            other => return Err(format!("content must be a string or a list of parts, got {other}")),
        };
        let mut msg = json!({"role": m.role, "content": content});
        if !m.tool_calls.is_empty() {
            msg["tool_calls"] = Value::Array(m.tool_calls);
        }
        if let Some(id) = m.tool_call_id {
            msg["tool_call_id"] = json!(id);
        }
        messages.push(msg);
        train.push(m.train);
    }
    if !train.iter().any(|&t| t) {
        return Err("no message is marked train: the record supervises nothing".into());
    }
    Ok(Record { messages, tools: wire.tools, media, train })
}

/// Read and validate every record of a JSONL file.
pub fn read_jsonl(path: &Path) -> Result<Vec<Record>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let base = path.parent().unwrap_or(Path::new("."));
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| parse_record(l, base).map_err(|e| format!("{}:{}: {e}", path.display(), i + 1)))
        .collect()
}

/// One supervised turn ready to train on.
pub struct Sample {
    /// The prompt's and the answer's tokens, concatenated.
    pub ids: Vec<u32>,
    /// Next-token targets per position: the answer's tokens, `IGNORE` before.
    pub targets: Vec<u32>,
    /// `(row0, n)` runs of image and of video placeholder rows, in order.
    pub image_runs: Vec<(u32, u32)>,
    pub video_runs: Vec<(u32, u32)>,
    pub images: Vec<ProcessedImage>,
    /// Each video's frames, processed.
    pub videos: Vec<Vec<ProcessedImage>>,
}

/// Load a record's media: images decoded, videos sampled and decoded as the
/// reference processor does.
pub fn load_media(record: &Record, opts: &ImageOptions, max_frames: u32) -> Result<(Vec<ProcessedImage>, Vec<Vec<ProcessedImage>>), String> {
    let (mut images, mut videos) = (Vec::new(), Vec::new());
    for m in &record.media {
        match m {
            Media::Image(p) => images.push(process_image(&imaging::codec::load(p)?, opts)),
            Media::Video(p) => videos.push(crate::video::load_clip(p, max_frames, 1)?.iter().map(|f| process_image(f, opts)).collect()),
        }
    }
    Ok((images, videos))
}

/// Every supervised turn of `record` as a [`Sample`] (one per assistant
/// message marked `train`), each rendered after its own context.
pub fn samples(fmt: &Formatter, record: &Record, images: &[ProcessedImage], videos: &[Vec<ProcessedImage>], opts: &ImageOptions) -> Result<Vec<Sample>, String> {
    let template = fmt.template();
    let tools = (!record.tools.is_empty())
        .then(|| data::chat_template::parse_json_ordered(&Value::Array(record.tools.clone()).to_string()))
        .transpose()
        .map_err(|e| format!("tools: {e:?}"))?;
    let layouts: Vec<String> = images.iter().map(|p| view_layout(p, opts.downsample, IMAGE_PAD)).collect();
    let video_layouts: Vec<String> = videos.iter().map(|c| crate::video::video_layout(c, opts.downsample)).collect();
    let mut out = Vec::new();
    for (k, _) in record.train.iter().enumerate().filter(|(_, t)| **t) {
        let msgs = data::chat_template::parse_json_ordered(&Value::Array(record.messages[..=k].to_vec()).to_string()).map_err(|e| format!("{e:?}"))?;
        let msgs: Vec<minijinja::Value> = msgs.try_iter().map_err(|e| format!("{e:?}"))?.collect();
        let (full, supervised) = template.render_with_supervised_suffix(&msgs, tools.clone(), false).map_err(|e| format!("turn {k}: {e:?}"))?;
        let prompt_text = &full[..supervised.start];
        // The media placed before this turn are the record's first ones.
        let n_img = prompt_text.matches(IMAGE_PAD).count();
        let n_vid = prompt_text.matches(crate::prompt::VIDEO_PAD).count();
        let prompt = fmt.expand(prompt_text, &layouts[..n_img], &video_layouts[..n_vid], opts.use_image_id)?;
        let answer = data::tokenizer::Tokenizer::encode(&*fmt.tokenizer, &full[supervised]);
        if answer.is_empty() {
            return Err(format!("turn {k} renders to no tokens"));
        }
        let mut ids = prompt.ids.clone();
        ids.extend(&answer);
        let first = prompt.ids.len();
        let targets: Vec<u32> = (0..ids.len()).map(|i| if i + 1 >= first && i + 1 < ids.len() { ids[i + 1] } else { model::IGNORE }).collect();
        let (mut image_runs, mut video_runs) = (Vec::new(), Vec::new());
        for &(r0, n) in &prompt.runs {
            if ids[r0 as usize] == fmt.image_pad {
                image_runs.push((r0, n));
            } else {
                video_runs.push((r0, n));
            }
        }
        out.push(Sample { ids, targets, image_runs, video_runs, images: images[..n_img].to_vec(), videos: videos[..n_vid].to_vec() });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(line: &str) -> Result<Record, String> {
        parse_record(line, Path::new("/"))
    }

    #[test]
    fn strict_parsing_names_what_is_wrong() {
        assert!(record(r#"{"messages":[{"role":"user","content":"hi","train":false},{"role":"assistant","content":"yo","train":true}]}"#).is_ok());
        assert!(record(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap_err().contains("train"));
        assert!(record(r#"{"messages":[{"role":"user","content":"hi","train":false}]}"#).unwrap_err().contains("supervises nothing"));
        assert!(record(r#"{"messages":[{"role":"user","content":"hi","train":true}]}"#).unwrap_err().contains("only assistant"));
        assert!(record(r#"{"messages":[{"role":"user","content":[{"type":"audio"}],"train":false},{"role":"assistant","content":"x","train":true}]}"#).unwrap_err().contains("audio"));
        assert!(record(r#"{"messages":[{"role":"user","content":[{"type":"image","path":"no/such.png"}],"train":false},{"role":"assistant","content":"x","train":true}]}"#).unwrap_err().contains("does not exist"));
        assert!(record(r#"{"messages":[],"extra":1}"#).is_err());
    }
}
