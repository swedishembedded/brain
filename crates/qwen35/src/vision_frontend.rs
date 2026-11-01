// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements image input for the LLMs its clients serve on their own GPUs. If your team needs
// expertise in turning chat requests with pictures into batched decoder rows then you can procure our services by
// sending an email to info@swedishembedded.com.

//! How a chat request with images becomes a scheduler request: the prompt the checkpoint's chat template renders with
//! each image expanded to its run of placeholder tokens, the vision tower's rows standing at those tokens
//! ([`model::serve::PromptEmbeds`]), and the three-axis rotary positions of the whole prompt (an image of `h x w`
//! merged rows spans `max(h, w)` positions, not `h * w`).
//!
//! The tower runs on the front end's own thread, on the decoder's card, while the scheduler decodes other requests;
//! a request's rows arrive on a channel the scheduler polls. Text-only requests never wait for it: they take the
//! ordinary text front end and its intake threads.
//!
//! Everything a client controls is checked here and answered as an invalid request: an image part without an image
//! (or the reverse), a content part this model does not read, the placeholder tokens typed by hand into text, an
//! image or a set of images the context cannot hold.

use std::sync::{mpsc, Arc};

use capability::{failure::tag, FailureClass, Invocation};
use data::qwen_tokenizer::QwenBpe;
use imaging::pixels::Rgb8;
use llmserve::{Encoding, Frontend, ParsedRequest, TextFrontend};
use model::serve::PromptEmbeds;
use serde_json::{json, Value};

use crate::vision_tower::{VisionTokens, VisionTower};

/// The literal placeholder strings the chat template writes for an image; the tokenizer turns each into one token.
const IMAGE_START: &str = "<|vision_start|>";
const IMAGE_PAD: &str = "<|image_pad|>";
const IMAGE_END: &str = "<|vision_end|>";
/// Every literal a client's text must not contain: typed by hand it would become a placeholder token that has no row.
const RESERVED: [&str; 4] = [IMAGE_START, IMAGE_PAD, IMAGE_END, "<|video_pad|>"];

fn invalid(message: impl AsRef<str>) -> String {
    tag(FailureClass::InvalidRequest, message)
}

/// The images of `inv`, in blob order (`image`, `image1`, ...).
pub fn request_images(inv: &Invocation) -> Result<Vec<Rgb8>, String> {
    let mut out = Vec::new();
    for i in 0..capability::blob::MAX_IMAGES {
        let key = capability::blob::image_key(i);
        if inv.get_blob(&key).is_none() {
            break;
        }
        let (hwc, w, h) = capability::blob::decode_image(inv, &key).map_err(invalid)?;
        out.push(Rgb8::from_hwc_unit(&hwc, w, h).map_err(invalid)?);
    }
    Ok(out)
}

/// Whether `inv` carries an image.
pub fn has_images(inv: &Invocation) -> bool {
    inv.get_blob("image").is_some()
}

/// Whether any string inside `v` holds a reserved placeholder literal.
fn holds_reserved(v: &Value) -> bool {
    match v {
        Value::String(s) => RESERVED.iter().any(|r| s.contains(r)),
        Value::Array(a) => a.iter().any(holds_reserved),
        Value::Object(o) => o.iter().any(|(k, v)| RESERVED.iter().any(|r| k.contains(r)) || holds_reserved(v)),
        _ => false,
    }
}

/// The request's `messages` with every `image_url` part replaced by its placeholder run (`rows[i]` image tokens for the
/// `i`th image part in message order), each message's parts joined to one string as the checkpoint's chat template
/// joins them. A message whose content is a string is left as it is.
///
/// `messages` is the contract's JSON array; `rows` has one entry per image of the request. Refuses a request whose
/// image parts and images do not match, a content part this model does not read, and reserved placeholder text.
pub fn expand_images(messages: &Value, rows: &[u32]) -> Result<Value, String> {
    let list = messages.as_array().ok_or_else(|| invalid("'messages' must be an array"))?;
    if holds_reserved(messages) {
        return Err(invalid(format!("message text must not contain the image placeholder tokens ({})", RESERVED.join(", "))));
    }
    let mut next = 0usize;
    let mut out = Vec::with_capacity(list.len());
    for message in list {
        let mut message = message.clone();
        if let Some(parts) = message.get("content").and_then(Value::as_array).cloned() {
            let mut text = String::new();
            for part in &parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") | None => text.push_str(part.get("text").and_then(Value::as_str).unwrap_or_default()),
                    Some("image_url") | Some("image") => {
                        let n = *rows.get(next).ok_or_else(|| invalid(format!("the request has more image parts than images ({})", rows.len())))?;
                        next += 1;
                        text.push_str(IMAGE_START);
                        text.push_str(&IMAGE_PAD.repeat(n as usize));
                        text.push_str(IMAGE_END);
                    }
                    Some(other) => return Err(invalid(format!("content part '{other}' is not read by this model (text and images only)"))),
                }
            }
            message["content"] = json!(text);
        }
        out.push(message);
    }
    if next != rows.len() {
        return Err(invalid(format!("the request carries {} images but its messages place {next}", rows.len())));
    }
    Ok(Value::Array(out))
}

/// The lengths of the runs of `image` tokens in `ids`, in order.
pub fn placeholder_runs(ids: &[u32], image: u32) -> Vec<usize> {
    let mut runs = Vec::new();
    let mut prev = false;
    for &t in ids {
        match (t == image, prev) {
            (true, true) => *runs.last_mut().expect("a run is open") += 1,
            (true, false) => runs.push(1),
            _ => {}
        }
        prev = t == image;
    }
    runs
}

/// What a request for the image worker carries and where its result goes.
type Job = (Invocation, mpsc::Sender<Result<(ParsedRequest, Option<PromptEmbeds>), String>>);

/// Everything building an image request needs.
struct ImageBuild {
    tok: QwenBpe,
    tower: Arc<VisionTower>,
    tokens: VisionTokens,
    /// The most prompt tokens the scheduler can hold for one sequence.
    context: usize,
    parse: fn(&QwenBpe, &Invocation) -> Result<ParsedRequest, String>,
}

impl ImageBuild {
    /// The scheduler request for `inv`, which carries at least one image: the tower runs here.
    fn build(&self, inv: &Invocation) -> Result<(ParsedRequest, Option<PromptEmbeds>), String> {
        if inv.cancel.is_cancelled() {
            return Err("cancelled".to_string());
        }
        let started = std::time::Instant::now();
        let images = request_images(inv)?;
        let sizes: Vec<(u32, (u32, u32))> = images.iter().map(|i| self.tower.tokens_for(i.h, i.w)).collect();
        let image_tokens: usize = sizes.iter().map(|&(n, _)| n as usize).sum();
        if image_tokens >= self.context {
            return Err(invalid(format!("the images need {image_tokens} tokens, more than the context of {} tokens holds; send fewer or smaller images", self.context)));
        }
        let messages = match inv.get_str("messages").filter(|m| !m.is_empty()) {
            Some(raw) => serde_json::from_str::<Value>(&raw).map_err(|e| invalid(format!("messages JSON: {e}")))?,
            // A bare prompt with images: one user turn, the images first.
            None => {
                let mut parts: Vec<Value> = images.iter().map(|_| json!({"type": "image_url", "image_url": {}})).collect();
                parts.push(json!({"type": "text", "text": inv.get_str("prompt").unwrap_or_default()}));
                json!([{"role": "user", "content": parts}])
            }
        };
        for extra in ["system", "tools"] {
            if let Some(text) = inv.get_str(extra) {
                let leaf = serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text));
                if holds_reserved(&leaf) {
                    return Err(invalid(format!("'{extra}' must not contain the image placeholder tokens")));
                }
            }
        }
        let rows: Vec<u32> = sizes.iter().map(|&(n, _)| n).collect();
        let expanded = expand_images(&messages, &rows)?;
        let rendered = inv.without_blobs().set("messages", json!(expanded.to_string()));
        let req = (self.parse)(&self.tok, &rendered)?;

        let runs = placeholder_runs(&req.ids, self.tokens.image);
        if runs.iter().map(|&r| r as u32).ne(rows.iter().copied()) {
            return Err(invalid(format!("the prompt holds image runs of {runs:?} tokens, the images need {rows:?}")));
        }
        let positions: Vec<u32> = req.ids.iter().enumerate().filter(|(_, &t)| t == self.tokens.image).map(|(i, _)| i as u32).collect();

        let prepared = started.elapsed();
        let width = self.tower.spec().config.out_hidden_size as usize;
        let mut rows_f32 = Vec::with_capacity(positions.len() * width);
        let mut grids = Vec::with_capacity(images.len());
        for (image, &(_, grid)) in images.iter().zip(&sizes) {
            if inv.cancel.is_cancelled() {
                return Err("cancelled".to_string());
            }
            let encoded = self.tower.encode(image);
            debug_assert_eq!(encoded.grid, grid, "the tower and the prompt agree on the grid");
            rows_f32.extend_from_slice(&encoded.rows);
            grids.push((1, grid.0, grid.1));
        }
        residency::log::info(&format!(
            "qwen35: {} image(s), {image_tokens} tokens: decode and render {:.0} ms, tower {:.0} ms",
            images.len(),
            prepared.as_secs_f64() * 1e3,
            (started.elapsed() - prepared).as_secs_f64() * 1e3
        ));
        let rotary = qwen3vl::mrope::get_rope_index(&req.ids, self.tokens.image, &grids);
        let embeds = PromptEmbeds::new(width, positions, rows_f32).map_err(|e| format!("vision rows: {e}"))?.with_rotary_positions(rotary);
        Ok((req, Some(embeds)))
    }
}

/// The front end of a Qwen3.8 checkpoint that reads images.
pub struct VisionFrontend {
    text: TextFrontend,
    build: Arc<ImageBuild>,
    queue: Option<mpsc::Sender<Job>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl VisionFrontend {
    /// `text` serves requests without images; `tok` (a second instance of the same tokenizer) builds those with.
    /// `context` is the longest prompt the scheduler holds.
    pub fn new(text: TextFrontend, tok: QwenBpe, tower: VisionTower, context: usize, parse: fn(&QwenBpe, &Invocation) -> Result<ParsedRequest, String>) -> Result<VisionFrontend, String> {
        let tokens = tower.spec().tokens;
        for (literal, id) in [(IMAGE_START, tokens.start), (IMAGE_PAD, tokens.image), (IMAGE_END, tokens.end)] {
            if tok.special_id(literal) != Some(id) {
                return Err(format!("the tokenizer's {literal} is not token {id}, which the checkpoint's config names"));
            }
        }
        let build = Arc::new(ImageBuild { tok, tower: Arc::new(tower), tokens, context, parse });
        let (queue, jobs) = mpsc::channel::<Job>();
        let worker_build = build.clone();
        let worker = std::thread::Builder::new()
            .name("qwen35-vision".to_string())
            .spawn(move || {
                // Requests are built in arrival order; the scheduler decodes other requests meanwhile.
                for (inv, reply) in jobs {
                    // The caller may have gone (the request was cancelled): nothing to deliver.
                    let _ = reply.send(worker_build.build(&inv));
                }
            })
            .map_err(|e| format!("the vision worker did not start: {e}"))?;
        Ok(VisionFrontend { text, build, queue: Some(queue), worker: Some(worker) })
    }

    /// The vision tower, for the callers that measure it.
    pub fn tower(&self) -> &VisionTower {
        &self.build.tower
    }
}

impl Drop for VisionFrontend {
    fn drop(&mut self) {
        drop(self.queue.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Frontend for VisionFrontend {
    fn tok(&self) -> &QwenBpe {
        self.text.tok()
    }
    fn stop(&self) -> &[u32] {
        self.text.stop()
    }
    fn label(&self) -> &str {
        self.text.label()
    }
    fn parse(&self, inv: &Invocation) -> Result<ParsedRequest, String> {
        if has_images(inv) {
            return self.parse_embedded(inv).map(|(req, _)| req);
        }
        self.text.parse(inv)
    }
    fn parse_embedded(&self, inv: &Invocation) -> Result<(ParsedRequest, Option<PromptEmbeds>), String> {
        if has_images(inv) {
            return self.build.build(inv);
        }
        self.text.parse_embedded(inv)
    }
    fn parse_in_background(&self, inv: &Invocation) -> Option<Encoding> {
        if !has_images(inv) {
            return self.text.parse_in_background(inv);
        }
        let (reply, rx) = mpsc::channel();
        self.queue.as_ref()?.send((inv.clone(), reply)).ok()?;
        Some(rx)
    }
    /// Building an image request runs the vision tower: it is not run ahead of the request's slot.
    fn encodes_inputs(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(s: &str) -> Value {
        json!({"type": "text", "text": s})
    }
    fn image() -> Value {
        json!({"type": "image_url", "image_url": {}})
    }

    #[test]
    fn each_image_part_becomes_its_run_of_placeholders_in_place() {
        let messages = json!([
            {"role": "user", "content": [text("a "), image(), text(" b "), image(), text(" c")]},
            {"role": "assistant", "content": "plain"},
        ]);
        let out = expand_images(&messages, &[2, 3]).unwrap();
        assert_eq!(out[0]["content"], "a <|vision_start|><|image_pad|><|image_pad|><|vision_end|> b <|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|> c");
        assert_eq!(out[1]["content"], "plain", "a string stays as it is");
    }

    #[test]
    fn images_and_image_parts_must_match() {
        let one = json!([{"role": "user", "content": [image(), text("x")]}]);
        assert!(expand_images(&one, &[1, 1]).unwrap_err().contains("carries 2 images but its messages place 1"));
        let two = json!([{"role": "user", "content": [image(), image()]}]);
        assert!(expand_images(&two, &[1]).unwrap_err().contains("more image parts than images"));
    }

    #[test]
    fn a_part_the_model_does_not_read_is_refused_not_dropped() {
        let audio = json!([{"role": "user", "content": [text("x"), {"type": "input_audio", "input_audio": {}}]}]);
        let err = expand_images(&audio, &[]).unwrap_err();
        assert!(err.contains("input_audio") && err.contains("[invalid-request]"), "{err}");
    }

    #[test]
    fn placeholder_text_typed_by_hand_is_refused_wherever_it_hides() {
        for hidden in ["<|image_pad|>", "x <|vision_start|> y", "<|video_pad|>", "<|vision_end|>"] {
            let in_text = json!([{"role": "user", "content": [text(hidden), image()]}]);
            assert!(expand_images(&in_text, &[1]).unwrap_err().contains("placeholder"), "{hidden}");
            let in_string = json!([{"role": "user", "content": hidden}]);
            assert!(expand_images(&in_string, &[]).is_err(), "{hidden}");
            let in_name = json!([{"role": "user", "name": hidden, "content": "x"}]);
            assert!(expand_images(&in_name, &[]).is_err(), "{hidden}");
        }
    }

    #[test]
    fn runs_of_placeholders_are_counted_apart() {
        assert_eq!(placeholder_runs(&[1, 9, 9, 9, 2, 9, 9, 3, 9], 9), vec![3, 2, 1]);
        assert!(placeholder_runs(&[1, 2, 3], 9).is_empty());
    }
}
