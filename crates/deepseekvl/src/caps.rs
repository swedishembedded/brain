// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The chat surface of the `MultiModalityCausalLM` understanding composite:
//! one streaming `generate` action (messages or a prompt, up to
//! [`MAX_IMAGES`] images, greedy text out), shared by DeepSeek-VL's provider
//! here and Janus-Pro's in `brain-januspro`.
//!
//! Images sit where the conversation puts them: each image part of a message
//! (an OpenAI `image_url` or Anthropic `image` part) becomes a placeholder at
//! its position, the images taken in order from the `image`, `image1`, ...
//! blobs. A conversation that carries images but no image parts (a bare
//! `prompt`, or text-only messages next to an attached image) gets one
//! placeholder per image at the start of its last user turn, the way the
//! reference processors' examples write them.

use std::sync::Mutex;

use capability::{Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use imaging::pixels::Rgb8;
use serde_json::{json, Value};

use crate::model::{GenRequest, Vlm};
use crate::prompt::{Role, Turn, IMAGE_TAG};

/// DeepSeek-VL's catalog id: the resolver accepts any directory holding a
/// DeepSeek-VL checkpoint, so the id names the family rather than a release.
pub const MODEL: &str = "brain/deepseekvl";

/// Images one request may carry: `image`, then `image1` ... `image7`. What a
/// request may actually fit is the served context's business: each image is
/// 576 rows of it.
pub const MAX_IMAGES: usize = 8;

/// The longest context either checkpoint's position table covers. A served
/// composite is built for whatever KV cache fits its decoder's card, up to
/// this ([`crate::model::place`]).
pub const MAX_CONTEXT: u32 = 16384;

pub const DEFAULT_MAX_NEW: i64 = 512;

fn image_key(i: usize) -> String {
    if i == 0 {
        "image".to_string()
    } else {
        format!("image{i}")
    }
}

/// The `generate` action for `model_desc`.
pub fn generate_spec(model_desc: &str) -> ActionSpec {
    let mut spec = ActionSpec::new("generate", model_desc)
        .streaming()
        .param(ParamSpec::new("messages", ParamType::Str, "flattened chat messages (JSON array string); image parts mark where each image goes"))
        .param(ParamSpec::new("prompt", ParamType::Str, "a single user message (alternative to messages)"))
        .param(ParamSpec::new("max_new", ParamType::Int, "max tokens to generate").default(json!(DEFAULT_MAX_NEW)).min(1.0).max(MAX_CONTEXT as f64))
        .param(ParamSpec::new("weights", ParamType::Str, "checkpoint DIRECTORY; overrides the model-store resolver's own pick when set").host_resolved())
        .input(BlobSpec::new("image", Media::Image, "an image: raw HWC f32 pixels in [0,1], meta {w,h} (capability::blob's wire convention)"));
    for i in 1..MAX_IMAGES {
        spec = spec.input(BlobSpec::new(&image_key(i), Media::Image, "a further image, read only when every earlier one is present"));
    }
    spec.output(BlobSpec::new("text", Media::Text, "the reply"))
}

pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "DeepSeek-VL-7B-chat -- images and text in, text out. SAM-B at 1024 px and SigLIP-L at 384 px joined by a split \
         aligner into a Llama decoder, at the checkpoint's own fp16. Greedy.",
        vec![generate_spec("DeepSeek-VL: up to 8 images + a conversation in, greedy text out (streamed per token)")],
    )
    .with_max_context_tokens(MAX_CONTEXT as u64)
}

/// The manifest for the scheduled service: the checkpoint is service-side
/// configuration, so `weights` is stripped.
pub fn manifest_resident() -> Manifest {
    manifest().for_serving()
}

/// The request's images, in blob order.
pub fn decode_images(inv: &Invocation) -> Result<Vec<Rgb8>, String> {
    let mut out = Vec::new();
    for i in 0..MAX_IMAGES {
        if inv.get_blob(&image_key(i)).is_none() {
            break;
        }
        let (hwc, w, h) = capability::blob::decode_image(inv, &image_key(i))?;
        if hwc.len() != (w * h * 3) as usize {
            return Err(format!("{}: {} values for a {w}x{h} RGB image", image_key(i), hwc.len()));
        }
        out.push(Rgb8 { w, h, px: hwc.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect() });
    }
    Ok(out)
}

/// A message's text with a placeholder at every image part.
fn content_with_tags(content: Option<&Value>, image_sep: &str) -> (String, usize) {
    match content {
        Some(Value::String(s)) => (s.clone(), 0),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            let mut images = 0;
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("text") => text.push_str(p.get("text").and_then(Value::as_str).unwrap_or("")),
                    Some("image_url") | Some("image") => {
                        text.push_str(IMAGE_TAG);
                        text.push_str(image_sep);
                        images += 1;
                    }
                    _ => {}
                }
            }
            (text, images)
        }
        _ => (String::new(), 0),
    }
}

/// The conversation of a request: an optional system prompt and the turns,
/// with exactly `images` placeholders. `image_sep` follows a placeholder
/// the model's processor places itself (`""` for DeepSeek-VL, `"\n"` for
/// Janus-Pro).
pub fn conversation(inv: &Invocation, images: usize, image_sep: &str) -> Result<(Option<String>, Vec<Turn>), String> {
    let mut system = None;
    let mut turns: Vec<Turn> = Vec::new();
    let mut tagged = 0;
    let messages = inv.get_str("messages").filter(|s| !s.trim().is_empty());
    if let Some(raw) = messages {
        let arr: Vec<Value> = serde_json::from_str(&raw).map_err(|e| format!("messages: {e}"))?;
        for m in &arr {
            let (text, n) = content_with_tags(m.get("content"), image_sep);
            tagged += n;
            let role = match m.get("role").and_then(Value::as_str) {
                Some("system") | Some("developer") => {
                    system = Some(text);
                    continue;
                }
                Some("user") => Role::User,
                Some("assistant") => Role::Assistant,
                other => return Err(format!("messages: role {other:?} is not system, user or assistant")),
            };
            // Consecutive turns of one role are one turn to the model.
            match turns.last_mut() {
                Some(t) if t.role == role => {
                    t.content.push('\n');
                    t.content.push_str(&text);
                }
                _ => turns.push(Turn { role, content: text }),
            }
        }
    } else {
        let prompt = inv.get_str("prompt").unwrap_or_default();
        turns.push(Turn { role: Role::User, content: prompt });
    }
    if tagged == 0 && images > 0 {
        let last = turns.iter_mut().rev().find(|t| t.role == Role::User).ok_or("a request with images needs a user turn to attach them to")?;
        last.content = format!("{}{}", format!("{IMAGE_TAG}{image_sep}").repeat(images), last.content);
    } else if tagged != images {
        return Err(format!("the messages place {tagged} images but the request carries {images}"));
    }
    if turns.iter().all(|t| t.content.trim().is_empty()) {
        return Err("empty conversation: pass 'messages' or 'prompt'".into());
    }
    Ok((system, turns))
}

/// A loaded composite behind the `generate` action.
pub struct Session {
    pub vlm: Vlm,
    image_sep: &'static str,
}

impl Session {
    pub fn new(vlm: Vlm, image_sep: &'static str) -> Session {
        Session { vlm, image_sep }
    }

    /// Run one `generate` invocation, streaming each token's text.
    pub fn generate(&mut self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.generate_batch(std::slice::from_ref(inv), &mut |_, p| progress(p)).pop().expect("one result per invocation")
    }

    /// Run `invs` as one batch: every request is prefilled and the sequences
    /// then decode together, each streaming its own tokens (`progress` gets
    /// the request's index). A request that cannot be prepared fails alone.
    pub fn generate_batch(&mut self, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        // The reference processors stop on the next user turn as well as on
        // the end-of-sentence token.
        let stops = vec![format!("{}:", self.vlm.style.user)];
        let mut prepared: Vec<Result<Prepared, String>> = invs.iter().map(|inv| self.prepare(inv)).collect();
        let (mut at, mut requests, mut streams) = (Vec::new(), Vec::new(), Vec::new());
        for (i, p) in prepared.iter_mut().enumerate() {
            if let Ok(p) = p {
                at.push(i);
                streams.push(Stream::default());
                requests.push(GenRequest { ids: &p.ids, embeds: &p.embeds, max_new: p.max_new });
            }
        }
        let tok = std::sync::Arc::clone(&self.vlm.tokenizer);
        // The reply's leading space (after `Assistant:`) is dropped from the
        // stream and the final text alike.
        let text_of = |ids: &[u32]| data::tokenizer::Tokenizer::decode(&*tok, ids).trim_start().to_string();
        let generated = self.vlm.generate_batch(&requests, &mut |r, id| {
            let (st, max_new) = (&mut streams[r], requests[r].max_new);
            st.out_ids.push(id);
            let (delta, now, stop) = qwen3::chat::visible_delta(&st.printed, &text_of(&st.out_ids), &stops);
            if !delta.is_empty() {
                progress(at[r], Progress::token(st.out_ids.len() as u32, max_new as u32, delta));
            }
            st.printed = now;
            st.stop_at = stop;
            st.stop_at.is_none()
        });
        let mut results: Vec<ActionResult> = prepared.iter().map(|p| p.as_ref().map(|_| Outcome::new()).map_err(|e| e.clone())).collect();
        for (r, generated) in generated.into_iter().enumerate() {
            let (i, st, p) = (at[r], &streams[r], prepared[at[r]].as_ref().expect("prepared requests only"));
            results[i] = generated.map(|ids_out| {
                let mut text = text_of(&st.out_ids);
                let finish = if let Some(cut) = st.stop_at.or_else(|| qwen3::chat::find_stop(&text, &stops)) {
                    text.truncate(cut);
                    "stop"
                } else if ids_out.len() < p.max_new {
                    "stop"
                } else {
                    "length"
                };
                if let Some(tail) = text.get(st.printed.len()..).filter(|t| !t.is_empty() && st.stop_at.is_none()) {
                    progress(i, Progress::token(st.out_ids.len() as u32, p.max_new as u32, tail.to_string()));
                }
                Outcome::new()
                    .set("text", json!(text.clone()))
                    .set("prompt_tokens", json!(p.prompt_tokens))
                    .set("completion_tokens", json!(st.out_ids.len()))
                    .set("finish_reason", json!(finish))
                    .blob("text", Blob::new(Media::Text, text.into_bytes()))
            });
        }
        results
    }

    /// One request's prompt ids, image rows and token budget.
    fn prepare(&self, inv: &Invocation) -> Result<Prepared, String> {
        let images = decode_images(inv)?;
        let (system, turns) = conversation(inv, images.len(), self.image_sep)?;
        let ids = self.vlm.prompt_ids_with(system.as_deref(), &turns)?;
        let embeds = self.vlm.image_embeds(&images)?;
        let prompt_tokens = self.vlm.splice.expand_ids(&ids).len();
        let context = self.vlm.decoder.max_seq_len();
        let room = context.saturating_sub(prompt_tokens);
        if room == 0 {
            return Err(format!("the prompt takes {prompt_tokens} tokens of the {context}-token context"));
        }
        let max_new = (inv.get_i64("max_new").unwrap_or(DEFAULT_MAX_NEW).max(1) as usize).min(room);
        Ok(Prepared { ids, embeds, max_new, prompt_tokens })
    }
}

/// A request ready to decode.
struct Prepared {
    ids: Vec<u32>,
    embeds: Vec<f32>,
    max_new: usize,
    prompt_tokens: usize,
}

/// The reply a request has streamed so far.
#[derive(Default)]
struct Stream {
    out_ids: Vec<u32>,
    printed: String,
    stop_at: Option<usize>,
}

/// Direct provider: builds (and caches) one [`Session`] per checkpoint
/// directory on first use.
#[derive(Default)]
pub struct DeepseekVlProvider {
    default_dir: Option<String>,
}

impl DeepseekVlProvider {
    pub fn new(default_dir: Option<String>) -> DeepseekVlProvider {
        DeepseekVlProvider { default_dir }
    }
}

impl Provider for DeepseekVlProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<std::sync::Arc<dyn Action>> {
        (name == "generate").then(|| std::sync::Arc::new(GenerateAction { default_dir: self.default_dir.clone() }) as std::sync::Arc<dyn Action>)
    }
}

struct GenerateAction {
    default_dir: Option<String>,
}

/// One process-wide composite, keyed by directory: a 16 GB build is not
/// something to repeat per call.
static RESIDENT: Mutex<Option<(String, Session)>> = Mutex::new(None);

/// Load DeepSeek-VL from `dir` for serving, at the checkpoint's own fp16, as
/// `placement` puts it.
pub fn load_session(dir: &str, placement: crate::model::Placement) -> Result<Session, String> {
    load_session_tuned(dir, placement, None)
}

/// [`load_session`] with the fine-tune in `tuned` (what `brain deepseekvl
/// finetune` wrote) applied.
pub fn load_session_tuned(dir: &str, placement: crate::model::Placement, tuned: Option<&std::path::Path>) -> Result<Session, String> {
    let adapter = tuned.map(crate::tuned::decoder_adapter).transpose()?.flatten();
    let vlm = crate::model::load_placed(std::path::Path::new(dir), placement.tier(qwen3::Dtype::F16), placement, adapter.as_deref())?;
    if let Some(t) = tuned {
        crate::tuned::apply_aligner(&vlm, t)?;
    }
    Ok(Session::new(vlm, ""))
}

/// Place DeepSeek-VL from `dir` over the cards' free memory now.
pub fn place_now(dir: &str) -> Result<crate::model::Placement, String> {
    crate::model::place(&crate::model::footprint(std::path::Path::new(dir))?, &gpu_core::capacity::available_gpus())
}

impl Action for GenerateAction {
    fn spec(&self) -> ActionSpec {
        let mut m = manifest();
        m.actions.remove(0)
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let dir = inv
            .get_str("weights")
            .filter(|s| !s.is_empty())
            .or_else(|| self.default_dir.clone())
            .ok_or("deepseekvl generate: no checkpoint (pass 'weights', or configure one through the models directory)")?;
        let mut guard = RESIDENT.lock().map_err(|_| "deepseekvl: resident lock poisoned")?;
        if !matches!(&*guard, Some((d, _)) if *d == dir) {
            *guard = None; // drop the old build before the new one allocates
            *guard = Some((dir.clone(), load_session(&dir, place_now(&dir)?)?));
        }
        guard.as_mut().expect("just loaded").1.generate(inv, progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inv_with(messages: Value) -> Invocation {
        Invocation::new().set("messages", json!(messages.to_string()))
    }

    #[test]
    fn the_action_has_the_chat_shape_the_api_serves() {
        let m = manifest();
        let a = &m.actions[0];
        assert_eq!((m.model.as_str(), a.name.as_str()), (MODEL, "generate"));
        assert!(a.streaming);
        assert!(a.params.iter().any(|p| p.name == "messages") && a.params.iter().any(|p| p.name == "prompt"));
        assert!(a.outputs.iter().any(|o| o.media == Media::Text));
        assert_eq!(a.inputs.iter().filter(|b| b.media == Media::Image).count(), MAX_IMAGES);
    }

    #[test]
    fn image_parts_become_placeholders_where_they_stand() {
        let inv = inv_with(json!([
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "Compare "}, {"type": "image_url"}, {"type": "text", "text": " and "}, {"type": "image_url"}]},
        ]));
        let (system, turns) = conversation(&inv, 2, "").unwrap();
        assert_eq!(system.as_deref(), Some("Be brief."));
        assert_eq!(turns, vec![Turn { role: Role::User, content: format!("Compare {IMAGE_TAG} and {IMAGE_TAG}") }]);
        assert!(conversation(&inv, 1, "").unwrap_err().contains("place 2 images"));
    }

    #[test]
    fn untagged_images_open_the_last_user_turn() {
        let inv = Invocation::new().set("prompt", json!("What is this?"));
        let (_, turns) = conversation(&inv, 2, "\n").unwrap();
        assert_eq!(turns[0].content, format!("{IMAGE_TAG}\n{IMAGE_TAG}\nWhat is this?"));
        let inv = inv_with(json!([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}, {"role": "user", "content": "Look"}]));
        let (_, turns) = conversation(&inv, 1, "").unwrap();
        assert_eq!((turns.len(), turns[0].content.as_str(), turns[2].content.as_str()), (3, "Hi", &*format!("{IMAGE_TAG}Look")));
    }

    #[test]
    fn a_foreign_role_or_an_empty_request_is_refused() {
        assert!(conversation(&inv_with(json!([{"role": "tool", "content": "x"}])), 0, "").unwrap_err().contains("tool"));
        assert!(conversation(&Invocation::new(), 0, "").unwrap_err().contains("empty"));
    }
}
