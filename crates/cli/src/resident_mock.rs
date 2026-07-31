// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A deterministic, weight-free **mock** resident model, gated on `BRAIN_MOCK`.
//!
//! It is a real [`ResidentModel`] so a `BRAIN_MOCK=1 brain serve` exercises the true
//! serving path — placement → claim → activate → `run_batch` — with no weights, no
//! GPU, and no external model. Registered under the id **`mock`**, it advertises the
//! exact three actions the apiserve handlers dispatch:
//!
//! * **`generate`** (streaming chat): reads the flattened `messages` (a JSON-array
//!   string) or `prompt`, echoes the last user turn back deterministically, streams
//!   it token-by-token via [`Progress::token`], and returns a `text` blob plus
//!   `{prompt_tokens, completion_tokens, finish_reason}` — matching
//!   [`apiserve`'s `bridge::read_outcome`].
//! * **`embed`**: returns `outputs.mean` (a fixed dim-8 `Vec<f32>` derived from the
//!   input) plus `outputs.tokens`.
//! * **`text2image`**: returns an `image` blob in brain's raw HWC-f32 wire format
//!   (`capability::blob::image_blob`) after a couple of denoise-`step` ticks.
//!
//! `estimate()` reports a small non-zero VRAM footprint so real placement runs (and,
//! on a GPU-less box, falls back to the CPU/RAM pool); `activate()` is instant.

use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::{json, Value};

/// The catalog id the mock registers under.
pub const MODEL: &str = "mock";

/// The mock model family (no state — every action is a pure function of its inputs).
pub struct MockResident;

impl MockResident {
    /// The mock resident when `BRAIN_MOCK` is set to a non-empty value, else `None`.
    pub fn from_env() -> Option<MockResident> {
        std::env::var("BRAIN_MOCK").ok().filter(|v| !v.is_empty()).map(|_| MockResident)
    }

    /// The streaming chat action. Declares every param the OpenAI/Anthropic/OpenRouter
    /// chat handlers set (`messages`, `max_new`, `temp`, `top_p`, `top_k`, `seed`,
    /// optional `system`/`stop`) plus a `prompt` alias — so `ActionSpec::validate`
    /// never rejects a handler-built invocation. `streaming` + a `messages`/`prompt`
    /// param + a `Text` output is what `catalog::api_caps` classifies as chat.
    fn generate_spec() -> ActionSpec {
        ActionSpec::new("generate", "deterministic mock chat generation")
            .streaming()
            .param(ParamSpec::new("messages", ParamType::Str, "flattened chat messages (JSON array string)"))
            .param(ParamSpec::new("prompt", ParamType::Str, "a raw prompt (alternative to messages)"))
            .param(ParamSpec::new("system", ParamType::Str, "system prompt"))
            .param(ParamSpec::new("max_new", ParamType::Int, "max new tokens").default(json!(1024)))
            .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature").default(json!(1.0)))
            .param(ParamSpec::new("top_p", ParamType::Float, "nucleus sampling p").default(json!(1.0)))
            .param(ParamSpec::new("top_k", ParamType::Int, "top-k").default(json!(0)))
            .param(ParamSpec::new("seed", ParamType::Int, "sampling seed").default(json!(0)))
            .param(ParamSpec::new("stop", ParamType::Str, "stop sequences (JSON array string)"))
            .output(BlobSpec::new("text", Media::Text, "the generated text"))
    }

    /// The embeddings action (`api_caps` classifies any action named `embed`).
    fn embed_spec() -> ActionSpec {
        ActionSpec::new("embed", "deterministic mock text embedding")
            .param(ParamSpec::new("text", ParamType::Str, "input text").required())
            .output(BlobSpec::new("embedding", Media::Bytes, "embedding vector bytes"))
    }

    /// The text-to-image action. An `Image` output + a `prompt` param + no required
    /// input blob is what `catalog::text2image_action` selects for `/images/generations`.
    fn text2image_spec() -> ActionSpec {
        ActionSpec::new("text2image", "deterministic mock image generation")
            .streaming()
            .param(ParamSpec::new("prompt", ParamType::Str, "the prompt").required())
            .param(ParamSpec::new("width", ParamType::Int, "image width").default(json!(1024)))
            .param(ParamSpec::new("height", ParamType::Int, "image height").default(json!(1024)))
            .param(ParamSpec::new("seed", ParamType::Int, "generation seed").default(json!(0)))
            .output(BlobSpec::new("image", Media::Image, "the generated image"))
    }
}

impl ResidentModel for MockResident {
    fn manifest(&self) -> Manifest {
        Manifest::new(MODEL, "deterministic mock model (chat + embeddings + image)", vec![Self::generate_spec(), Self::embed_spec(), Self::text2image_spec()])
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(MODEL, "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // A small non-zero footprint so real placement runs: on a box with a GPU it is
        // budgeted there, on a GPU-less box it spills to the CPU/RAM pool (place.rs).
        MemCost::new(64 << 20, 0)
    }
    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        // No weights — the instance is built instantly.
        Ok(Box::new(MockInstance))
    }
}

/// The built mock instance (holds nothing — actions are pure functions).
struct MockInstance;

impl Instance for MockInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match action {
            "generate" => Ok(generate(inv, progress)),
            "embed" => Ok(embed(inv)),
            "text2image" => Ok(text2image(inv, progress)),
            other => Err(format!("mock: unknown action '{other}'")),
        }
    }
}

/// The last `user`-role message's text (falling back to the last message's content,
/// then the `prompt` param) — what the deterministic reply echoes.
fn last_user_text(inv: &Invocation) -> String {
    if let Some(s) = inv.get_str("messages") {
        if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(&s) {
            for m in arr.iter().rev() {
                if m.get("role").and_then(|v| v.as_str()) == Some("user") {
                    if let Some(c) = m.get("content").and_then(|v| v.as_str()) {
                        return c.to_string();
                    }
                }
            }
            if let Some(c) = arr.last().and_then(|m| m.get("content")).and_then(|v| v.as_str()) {
                return c.to_string();
            }
        }
    }
    inv.get_str("prompt").unwrap_or_default()
}

/// The stop sequences (the handler passes them as a JSON-array string), or empty.
fn parse_stop(inv: &Invocation) -> Vec<String> {
    inv.get_str("stop")
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .and_then(|v| v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()))
        .unwrap_or_default()
}

/// Deterministic chat generation: echo the last user turn as `"You said: <text>"`
/// (or a canned line for an empty prompt), streamed token-by-token so the concatenated
/// deltas exactly reconstruct the returned `text` blob. Honors `max_new` (capping ⇒
/// `finish_reason: "length"`) and a `stop` sequence (an early stop ⇒ `"stop"`).
fn generate(inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> Outcome {
    let max_new = inv.get_i64("max_new").unwrap_or(1024).max(1) as usize;
    let user = last_user_text(inv);
    let reply = if user.trim().is_empty() { "mock reply ready".to_string() } else { format!("You said: {}", user.trim()) };
    let words: Vec<&str> = reply.split_whitespace().collect();
    let stops = parse_stop(inv);

    // Build the streamed pieces, each carrying its leading separator so the
    // concatenation of all deltas is byte-for-byte the full text.
    let mut pieces: Vec<String> = Vec::new();
    let mut hit_stop = false;
    for (i, w) in words.iter().enumerate() {
        if pieces.len() >= max_new {
            break;
        }
        if stops.iter().any(|s| !s.is_empty() && w.contains(s.as_str())) {
            hit_stop = true;
            break;
        }
        pieces.push(if i == 0 { (*w).to_string() } else { format!(" {w}") });
    }
    // Hit max_new before emitting every word (and no stop match) ⇒ length-limited.
    let finish = if !hit_stop && pieces.len() < words.len() { "length" } else { "stop" };

    let total = pieces.len() as u32;
    let mut text = String::new();
    for (i, piece) in pieces.iter().enumerate() {
        text.push_str(piece);
        progress(Progress::token(i as u32, total, piece.clone()));
    }
    let prompt_tokens = user.split_whitespace().count().max(1) as i64;
    Outcome::new()
        .set("prompt_tokens", json!(prompt_tokens))
        .set("completion_tokens", json!(pieces.len()))
        .set("finish_reason", json!(finish))
        .blob("text", Blob::new(Media::Text, text.into_bytes()))
}

/// The fixed embedding dimension the mock returns.
const EMBED_DIM: usize = 8;

/// A deterministic dim-8 embedding of `text`: byte sums folded into 8 buckets, mean-
/// normalized and squashed with `sin` to a bounded, reproducible vector. `outputs.mean`
/// + `outputs.tokens` are exactly what the embeddings handler reads.
fn embed(inv: &Invocation) -> Outcome {
    let text = inv.get_str("text").unwrap_or_default();
    let mut v = [0f32; EMBED_DIM];
    for (i, b) in text.bytes().enumerate() {
        v[i % EMBED_DIM] += b as f32;
    }
    let n = text.len().max(1) as f32;
    let mean: Vec<f32> = v.iter().map(|x| (x / n).sin()).collect();
    let tokens = text.split_whitespace().count().max(1);
    Outcome::new().set("mean", json!(mean)).set("tokens", json!(tokens)).set("dim", json!(EMBED_DIM)).blob("embedding", Blob::new(Media::Bytes, Vec::new()))
}

/// Deterministic image generation: a seed-shifted RGB gradient in brain's raw HWC-f32
/// `[0,1]` wire format, after a couple of `step` denoise ticks (mapped to
/// `image_generation.partial_image` events on the streaming surface).
fn text2image(inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> Outcome {
    let w = inv.get_i64("width").unwrap_or(1024).clamp(1, 4096) as u32;
    let h = inv.get_i64("height").unwrap_or(1024).clamp(1, 4096) as u32;
    let seed = inv.get_i64("seed").unwrap_or(0).rem_euclid(256) as u32;
    let steps = 4u32;
    for s in 0..steps {
        progress(Progress::step(s + 1, steps, "denoise"));
    }
    let mut hwc = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            hwc.push((x.wrapping_add(seed) % 256) as f32 / 255.0);
            hwc.push((y % 256) as f32 / 255.0);
            hwc.push((x.wrapping_add(y) % 256) as f32 / 255.0);
        }
    }
    Outcome::new().set("width", json!(w)).set("height", json!(h)).blob("image", capability::blob::image_blob(&hwc, w, h, 3))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_classifies_as_chat_embeddings_and_image() {
        let caps = apiserve::api_caps(&MockResident.manifest());
        assert!(caps.chat, "generate must classify as chat");
        assert!(caps.embeddings, "embed must classify as embeddings");
        assert!(caps.image, "text2image must classify as image");
    }

    #[test]
    fn generate_streams_deltas_that_concatenate_to_the_text_blob() {
        let model = MockResident;
        let mut inst = model.activate(&model.instance_key("generate", &Invocation::new()), Device::Cpu).unwrap();
        let messages = serde_json::to_string(&json!([{ "role": "user", "content": "hello there" }])).unwrap();
        let inv = Invocation::new().set("messages", json!(messages)).set("max_new", json!(1024));

        let mut streamed = String::new();
        let out = inst.run("generate", &inv, &mut |p| {
            if let Some(d) = p.delta {
                streamed.push_str(&d);
            }
        });
        let out = out.unwrap();
        let text = String::from_utf8(out.blobs["text"].bytes.clone()).unwrap();
        assert_eq!(streamed, text, "concatenated per-token deltas must equal the text blob");
        assert_eq!(text, "You said: hello there");
        assert_eq!(out.outputs["finish_reason"], "stop");
        assert_eq!(out.outputs["completion_tokens"], 4);
    }

    #[test]
    fn generate_caps_at_max_new_and_reports_length() {
        let mut inst = MockInstance;
        let inv = Invocation::new().set("prompt", json!("one two three four five")).set("max_new", json!(2));
        let out = inst.run("generate", &inv, &mut |_| {}).unwrap();
        assert_eq!(out.outputs["completion_tokens"], 2);
        assert_eq!(out.outputs["finish_reason"], "length");
    }

    #[test]
    fn embed_returns_fixed_dim_mean_vector() {
        let mut inst = MockInstance;
        let out = inst.run("embed", &Invocation::new().set("text", json!("hello world")), &mut |_| {}).unwrap();
        let mean = out.outputs["mean"].as_array().unwrap();
        assert_eq!(mean.len(), EMBED_DIM);
        assert!(out.outputs["tokens"].as_i64().unwrap() > 0);
    }

    #[test]
    fn text2image_emits_raw_hwc_image_blob() {
        let mut inst = MockInstance;
        let inv = Invocation::new().set("prompt", json!("a cat")).set("width", json!(4)).set("height", json!(4));
        let mut steps = 0u32;
        let out = inst.run("text2image", &inv, &mut |_| steps += 1).unwrap();
        assert!(steps >= 2, "must emit denoise step ticks");
        let blob = &out.blobs["image"];
        assert_eq!(blob.media, Media::Image);
        assert_eq!(blob.meta, json!({ "w": 4, "h": 4, "c": 3 }));
        assert_eq!(blob.bytes.len(), 4 * 4 * 3 * 4);
    }
}
