// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The prompt: a conversation rendered through the checkpoint's OWN chat
//! template (`data::chat_template`, the Jinja HuggingFace runs), then every
//! media placeholder the template emitted (`<|image_pad|>` per image,
//! `<|video_pad|>` per video) expanded into the reference processor's layout,
//! then tokenized. Per image:
//!
//! ```text
//! [<image_id>N</image_id>] <image> pad*src </image> [ (<slice> pad*s </slice>){cols} ]{rows, joined by \n}
//! ```
//!
//! where `pad` is the placeholder token, `src`/`s` the source view's and one
//! slice's visual token counts, `N` the image's index in the request. The
//! visual features later replace the pad rows in order, so the layout fixes
//! which features go where; [`Prompt::runs`] reports them.

use std::collections::BTreeMap;
use std::path::Path;

use data::chat_template::{parse_json_ordered, ChatTemplate};
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;

use crate::config::Downsample;
use crate::preprocess::ProcessedImage;
use crate::vision::View;

pub const IMAGE_PAD: &str = "<|image_pad|>";
pub const VIDEO_PAD: &str = "<|video_pad|>";

/// One part of a message's content.
#[derive(Clone, Debug, PartialEq)]
pub enum Part {
    Text(String),
    /// The next image of the request (in request order).
    Image,
    /// The next video of the request.
    Video,
}

/// One conversation turn.
#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub role: String,
    pub parts: Vec<Part>,
}

impl Message {
    pub fn user(parts: Vec<Part>) -> Message {
        Message { role: "user".into(), parts }
    }

    pub fn assistant(text: impl Into<String>) -> Message {
        Message { role: "assistant".into(), parts: vec![Part::Text(text.into())] }
    }

    pub fn system(text: impl Into<String>) -> Message {
        Message { role: "system".into(), parts: vec![Part::Text(text.into())] }
    }
}

/// A tokenized prompt and where its visual rows are.
#[derive(Clone, Debug, PartialEq)]
pub struct Prompt {
    pub ids: Vec<u32>,
    /// `(row0, n_rows)` of each contiguous run of image/video pad tokens, in
    /// order - the rows the visual features replace.
    pub runs: Vec<(u32, u32)>,
}

/// The chat template and tokenizer of a checkpoint.
pub struct Formatter {
    template: ChatTemplate,
    pub tokenizer: std::sync::Arc<QwenBpe>,
    pub image_pad: u32,
    pub video_pad: u32,
}

/// Visual tokens of one view at `mode`.
pub fn view_tokens(view: View, mode: Downsample) -> u32 {
    view.patches() / mode.divisor()
}

/// The placeholder text of one processed image (or one video frame, with
/// `pad = VIDEO_PAD`): the reference's `replace_image_token`.
pub fn view_layout(image: &ProcessedImage, mode: Downsample, pad: &str) -> String {
    let mut s = format!("<image>{}</image>", pad.repeat(view_tokens(image.views[0], mode) as usize));
    if let Some((rows, cols)) = image.grid {
        let per_slice = image.views.get(1).map_or(0, |&v| view_tokens(v, mode));
        let slice = format!("<slice>{}</slice>", pad.repeat(per_slice as usize));
        let row = slice.repeat(cols as usize);
        s.push_str(&vec![row; rows as usize].join("\n"));
    }
    s
}

impl Formatter {
    pub fn from_dir(dir: &Path) -> Result<Formatter, String> {
        let template = ChatTemplate::from_model_dir(dir).map_err(|e| format!("minicpmv: chat template: {e:?}"))?;
        let tok_path = dir.join("tokenizer.json");
        let tokenizer = std::sync::Arc::new(QwenBpe::from_file(tok_path.to_str().ok_or("non-UTF-8 tokenizer path")?)?);
        let id = |t: &str| tokenizer.special_id(t).ok_or_else(|| format!("minicpmv: the tokenizer has no {t}"));
        Ok(Formatter { image_pad: id(IMAGE_PAD)?, video_pad: id(VIDEO_PAD)?, template, tokenizer })
    }

    /// The conversation rendered by the chat template, media placeholders not
    /// yet expanded.
    pub fn render(&self, messages: &[Message], add_generation_prompt: bool, enable_thinking: bool) -> Result<String, String> {
        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                let content: Vec<serde_json::Value> = m
                    .parts
                    .iter()
                    .map(|p| match p {
                        Part::Text(t) => serde_json::json!({"type": "text", "text": t}),
                        Part::Image => serde_json::json!({"type": "image"}),
                        Part::Video => serde_json::json!({"type": "video"}),
                    })
                    .collect();
                serde_json::json!({"role": m.role, "content": content})
            })
            .collect();
        let value = parse_json_ordered(&serde_json::Value::Array(msgs).to_string()).map_err(|e| format!("minicpmv: {e:?}"))?;
        let mut extra = BTreeMap::new();
        extra.insert("enable_thinking".to_string(), minijinja::Value::from(enable_thinking));
        self.template.render(value, None, add_generation_prompt, &extra).map_err(|e| format!("minicpmv: render: {e:?}"))
    }

    /// Expand the rendered text's placeholders - the i-th `<|image_pad|>` into
    /// `images[i]`, the i-th `<|video_pad|>` into `videos[i]` - and tokenize.
    /// `images`/`videos` are the per-item layouts ([`view_layout`]);
    /// `use_image_id` prefixes each with `<image_id>N</image_id>`, `N`
    /// counting images and videos separately from 0.
    pub fn expand(&self, rendered: &str, images: &[String], videos: &[String], use_image_id: bool) -> Result<Prompt, String> {
        let mut text = String::with_capacity(rendered.len());
        let (mut ni, mut nv) = (0usize, 0usize);
        let mut rest = rendered;
        loop {
            let next = [(rest.find(IMAGE_PAD), true), (rest.find(VIDEO_PAD), false)].into_iter().filter_map(|(at, img)| at.map(|a| (a, img))).min();
            let Some((at, is_image)) = next else {
                text.push_str(rest);
                break;
            };
            text.push_str(&rest[..at]);
            let (layouts, n, pad) = if is_image { (images, &mut ni, IMAGE_PAD) } else { (videos, &mut nv, VIDEO_PAD) };
            let layout = layouts.get(*n).ok_or_else(|| format!("minicpmv: the prompt has more {pad} placeholders than the {} items given", layouts.len()))?;
            if use_image_id {
                text.push_str(&format!("<image_id>{}</image_id>", *n));
            }
            text.push_str(layout);
            *n += 1;
            rest = &rest[at + pad.len()..];
        }
        if ni != images.len() || nv != videos.len() {
            return Err(format!("minicpmv: {} images and {} videos given, the prompt places {ni} and {nv}", images.len(), videos.len()));
        }
        let ids = self.tokenizer.encode(&text);
        let mut runs = Vec::new();
        let mut i = 0;
        while i < ids.len() {
            if ids[i] == self.image_pad || ids[i] == self.video_pad {
                let start = i;
                while i < ids.len() && ids[i] == ids[start] {
                    i += 1;
                }
                runs.push((start as u32, (i - start) as u32));
            } else {
                i += 1;
            }
        }
        Ok(Prompt { ids, runs })
    }
}
