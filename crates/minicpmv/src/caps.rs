// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniCPM-V's capability surface: one streaming `generate` action - a
//! conversation (or a prompt) with up to [`MAX_IMAGES`] images and a video
//! in, text out - served over HTTP (OpenAI `image_url` parts) and D-Bus
//! (`image`, `image1`, ... and `video` blobs) alike. On the CLI a video file
//! (`video_path`) is decoded by frame index from its container metadata, as
//! the reference processor samples it.
//!
//! Requests ride the shared chat layer (`qwen3::chat`): the conversation is
//! rendered by the checkpoint's OWN template (`ChatFormat::Template`), so
//! system prompts, tools and `tool_choice`, `enable_thinking`, stop strings
//! and the sampling params mean what they mean for every other served chat
//! model, and the reply is scanned into content, reasoning and tool calls by
//! the same per-sequence state. Each image part of a message becomes a
//! placeholder where it stands; a request whose messages carry no image
//! parts gets its images at the start of its last user turn.
//!
//! Requests dispatched together decode as one batch ([`Session::generate_batch`]).

use std::path::Path;
use std::sync::Mutex;

use capability::{Action, ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, ParamSpec, ParamType, Progress, Provider};
use data::chat_template::ChatTemplate;
use imaging::Rgb8;
use model::kv_tier::KvTier;
use qwen3::chat::{ChatFormat, ParsedRequest, SeqState};
use serde_json::{json, Value};

use crate::config::Downsample;
use crate::model::{GenOptions, MiniCpmV, Prepared};

pub const MODEL: &str = "brain/minicpmv";

/// Images one request may carry: `image`, then `image1` ... `image7`.
pub const MAX_IMAGES: usize = 8;

/// The per-sequence context a served instance is built for (prompt plus
/// reply). A 36-slice image at 4x is about 9k placeholder rows.
pub const SERVE_CONTEXT: u32 = 16384;

/// Sequences a served instance decodes together.
pub const SERVE_CONCURRENT: u32 = 4;

pub const DEFAULT_MAX_NEW: i64 = 512;

fn image_key(i: usize) -> String {
    if i == 0 {
        "image".to_string()
    } else {
        format!("image{i}")
    }
}

pub fn generate_spec() -> ActionSpec {
    let mut spec = ActionSpec::new("generate", "MiniCPM-V 4.6: up to 8 images and a video + a conversation in, text out (streamed per token)")
        .streaming()
        .param(ParamSpec::new("messages", ParamType::Str, "flattened chat messages (JSON array string); image parts mark where each image goes"))
        .param(ParamSpec::new("prompt", ParamType::Str, "a single user message (alternative to messages)"))
        .param(ParamSpec::new("system", ParamType::Str, "optional system prompt"))
        .param(ParamSpec::new("tools", ParamType::Str, "JSON array of tool definitions (OpenAI function-calling schema)"))
        .param(ParamSpec::new("tool_choice", ParamType::Str, "auto | none | required | {\"type\":\"function\",\"function\":{\"name\":...}}"))
        .param(ParamSpec::new("stop", ParamType::Str, "JSON array of stop strings"))
        .param(ParamSpec::new("enable_thinking", ParamType::Bool, "let the model reason in a <think> block first").default(json!(false)))
        .param(ParamSpec::new("max_new", ParamType::Int, "number of new tokens to generate; omitted, the reply runs to its end of turn or to the end of the context").min(1.0).max(SERVE_CONTEXT as f64))
        .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature (<= 0: greedy)").default(json!(0.8)))
        .param(ParamSpec::new("top_k", ParamType::Int, "top-k cut (0: off)").default(json!(40)))
        .param(ParamSpec::new("top_p", ParamType::Float, "nucleus cut (1: off)").default(json!(1.0)))
        .param(ParamSpec::new("seed", ParamType::Int, "sampling seed (random when unset; echoed in the outcome)"))
        .param(ParamSpec::new("downsample", ParamType::Str, "visual tokens per patch: \"16x\" (default) or \"4x\" (four times the tokens, for fine detail)"))
        .param(ParamSpec::new("max_slices", ParamType::Int, "most slices one high-resolution image is cut into (the processor's 9 when unset)").min(1.0).max(64.0))
        .param(ParamSpec::new("use_image_id", ParamType::Bool, "prefix each image with its <image_id>N</image_id> tag (the processor's default when unset)"))
        .param(ParamSpec::new("max_frames", ParamType::Int, "most frames one video contributes (the processor's 128)").default(json!(crate::video::MAX_NUM_FRAMES)).min(1.0).max(512.0))
        .param(ParamSpec::new("stack_frames", ParamType::Int, "sub-second frames stacked per second; only 1 is supported (the reference's stacking path fails on its own output)").default(json!(1)).min(1.0).max(1.0))
        .param(ParamSpec::new("video_path", ParamType::Str, "a video FILE on this host, its frames picked by index from the container's metadata as the reference processor does").host_resolved())
        .param(ParamSpec::new("weights", ParamType::Str, "checkpoint DIRECTORY; overrides the model-store resolver's own pick when set").host_resolved())
        .input(BlobSpec::new("image", Media::Image, "an image: raw HWC f32 pixels in [0,1], meta {w,h} (capability::blob's wire convention)"));
    for i in 1..MAX_IMAGES {
        spec = spec.input(BlobSpec::new(&image_key(i), Media::Image, "a further image, read only when every earlier one is present"));
    }
    spec = spec.input(BlobSpec::new(
        "video",
        Media::Video,
        "a video as decoded frames (capability::blob::video_blob); with meta {fps} the frames are the clip at that rate and are sampled as the reference processor does, without it they are the frames to use",
    ));
    spec.output(BlobSpec::new("text", Media::Text, "the reply"))
}

pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "MiniCPM-V 4.6 -- images, a video and text in, text out. A NaViT SigLIP tower whose token count a window merger and a 2x2 \
         merger cut 16x (or 4x), spliced into a Qwen3.5-0.8B hybrid decoder; high-resolution images are sliced. fp32.",
        vec![generate_spec()],
    )
    .with_max_context_tokens(SERVE_CONTEXT as u64)
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

/// The request's video, as the frames the model sees: the `video_path` file
/// decoded by index, or the `video` blob - sampled when its meta says the
/// rate its frames are at, thinned evenly to `max_frames` otherwise.
pub fn decode_video(inv: &Invocation) -> Result<Option<Vec<Rgb8>>, String> {
    let max_frames = inv.get_i64("max_frames").unwrap_or(crate::video::MAX_NUM_FRAMES as i64).max(1) as u32;
    let stack = inv.get_i64("stack_frames").unwrap_or(1).max(1) as u32;
    if let Some(path) = inv.get_str("video_path").filter(|s| !s.is_empty()) {
        return crate::video::load_clip(Path::new(&path), max_frames, stack).map(Some);
    }
    if inv.get_blob("video").is_none() {
        return Ok(None);
    }
    let frames: Vec<Rgb8> = capability::blob::decode_video(inv, "video")?
        .into_iter()
        .map(|(hwc, w, h)| Rgb8 { w, h, px: hwc.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect() })
        .collect();
    let fps = inv.get_blob("video").and_then(|b| b.meta.get("fps").and_then(Value::as_f64));
    let picked: Vec<usize> = match fps {
        Some(fps) => {
            let meta = crate::video::ClipMeta { frames: frames.len() as u64, fps, duration: frames.len() as f64 / fps };
            crate::video::sample_frames(meta, max_frames, stack)?.into_iter().map(|i| i as usize).collect()
        }
        None if frames.len() > max_frames as usize => crate::video::linspace_int(frames.len(), max_frames as usize),
        None => (0..frames.len()).collect(),
    };
    Ok(Some(picked.into_iter().map(|i| frames[i].clone()).collect()))
}

fn is_image_part(p: &Value) -> bool {
    matches!(p.get("type").and_then(Value::as_str), Some("image") | Some("image_url"))
}

fn is_video_part(p: &Value) -> bool {
    matches!(p.get("type").and_then(Value::as_str), Some("video") | Some("video_url"))
}

/// `inv` with its conversation carrying exactly `images` image parts and
/// `videos` video parts, each in the template's own `{"type": "image"}` /
/// `{"type": "video"}` form: where its messages place them, else prepended
/// to the last user turn (images first; a bare `prompt` becomes that turn).
pub fn with_media_parts(inv: &Invocation, images: usize, videos: usize) -> Result<Invocation, String> {
    let mut turns: Vec<Value> = match inv.get_str("messages").filter(|s| !s.trim().is_empty()) {
        Some(raw) => serde_json::from_str::<Value>(&raw).map_err(|e| format!("messages: {e}"))?.as_array().cloned().ok_or("messages must be a JSON array")?,
        None => vec![json!({"role": "user", "content": inv.get_str("prompt").unwrap_or_default()})],
    };
    // Every media part becomes the template's own form (an OpenAI part
    // flattened to `{"type": "image_url"}` matches none of the template's
    // tests), its pixels travelling as blobs.
    let (mut placed_images, mut placed_videos) = (0, 0);
    for t in turns.iter_mut() {
        if let Some(parts) = t.get_mut("content").and_then(Value::as_array_mut) {
            for p in parts.iter_mut() {
                if is_image_part(p) {
                    *p = json!({"type": "image"});
                    placed_images += 1;
                } else if is_video_part(p) {
                    *p = json!({"type": "video"});
                    placed_videos += 1;
                }
            }
        }
    }
    let untagged = placed_images == 0 && placed_videos == 0;
    if untagged && images + videos > 0 {
        let last = turns.iter_mut().rev().find(|t| t.get("role").and_then(Value::as_str) == Some("user")).ok_or("a request with images or a video needs a user turn to attach them to")?;
        let mut parts: Vec<Value> = (0..images).map(|_| json!({"type": "image"})).chain((0..videos).map(|_| json!({"type": "video"}))).collect();
        match last.get("content") {
            Some(Value::String(s)) => parts.push(json!({"type": "text", "text": s})),
            Some(Value::Array(a)) => parts.extend(a.iter().cloned()),
            _ => {}
        }
        last["content"] = Value::Array(parts);
    } else if !untagged && (placed_images != images || placed_videos != videos) {
        return Err(format!("the messages place {placed_images} images and {placed_videos} videos but the request carries {images} and {videos}"));
    }
    Ok(inv.clone().set("messages", json!(Value::Array(turns).to_string())))
}

/// A loaded composite behind the `generate` action.
pub struct Session {
    pub model: MiniCpmV,
    format: ChatFormat,
}

/// A request ready to decode.
struct Request {
    parsed: ParsedRequest,
    prepared: Prepared,
    opts: GenOptions,
}

impl Session {
    pub fn new(model: MiniCpmV, template: ChatTemplate) -> Session {
        Session { model, format: ChatFormat::Template(Box::new(template)) }
    }

    /// Run one `generate` invocation, streaming each token's text.
    pub fn generate(&mut self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.generate_batch(std::slice::from_ref(inv), &mut |_, p| progress(p)).pop().expect("one result per invocation")
    }

    /// Run `invs` as one batch: every request is prepared and prefilled, and
    /// the sequences then decode together, each streaming its own reply
    /// (`progress` gets the request's index). A request that cannot be
    /// prepared fails alone; a cancelled one stops at its next token and
    /// reports `finish_reason: "cancelled"`.
    pub fn generate_batch(&mut self, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        let prepared: Vec<Result<Request, String>> = invs.iter().map(|inv| self.prepare(inv)).collect();
        let at: Vec<usize> = (0..invs.len()).filter(|&i| prepared[i].is_ok()).collect();
        let reqs: Vec<&Request> = at.iter().map(|&i| prepared[i].as_ref().expect("filtered")).collect();
        let mut states: Vec<(SeqState, Vec<u32>)> = at.iter().zip(&reqs).map(|(&i, r)| (SeqState::new(&r.parsed, invs[i].cancel.clone()), Vec::new())).collect();
        let batch: Vec<(&Prepared, &GenOptions)> = reqs.iter().map(|r| (&r.prepared, &r.opts)).collect();
        let tok = std::sync::Arc::clone(&self.model.fmt.tokenizer);
        let mut model_out = self.model.generate_batch(&batch, &mut |r, id| {
            let (seq, out) = &mut states[r];
            out.push(id);
            !seq.advance(&tok, out, &mut |p| progress(at[r], p))
        });
        let mut results: Vec<ActionResult> = prepared.iter().map(|p| p.as_ref().map(|_| capability::Outcome::new()).map_err(|e| e.clone())).collect();
        for (r, ((seq, out), generated)) in states.into_iter().zip(model_out.drain(..)).enumerate() {
            let i = at[r];
            results[i] = generated.map(|_| seq.finish(&tok, &out, &mut |p| progress(i, p)));
        }
        results
    }

    /// One request's prompt, embeddings and options.
    fn prepare(&self, inv: &Invocation) -> Result<Request, String> {
        if inv.cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        let images = decode_images(inv)?;
        let videos: Vec<Vec<Rgb8>> = decode_video(inv)?.into_iter().collect();
        // The checkpoint's template thinks only when asked; the shared
        // renderer reads an absent flag as thinking on, so state it.
        let thinking = inv.get_bool("enable_thinking").unwrap_or(false);
        let inv = with_media_parts(inv, images.len(), videos.len())?.set("enable_thinking", json!(thinking));
        let rendered = qwen3::chat::render_prompt_as(&self.format, &inv)?;
        let (max_new, temp, top_k, seed) = qwen3::chat::sampling_params(&inv)?;
        let top_p = inv.get_f64("top_p").unwrap_or(1.0) as f32;
        let opts = GenOptions {
            // No budget named: what the prompt leaves of the context (below).
            max_new: max_new.unwrap_or(usize::MAX).max(1),
            temperature: temp,
            top_k,
            top_p,
            seed,
            enable_thinking: thinking,
            downsample: inv.get_str("downsample").filter(|s| !s.is_empty()).map(|s| Downsample::parse(&s)).transpose()?,
            max_slices: inv.get_i64("max_slices").map(|n| n.max(1) as u32),
            use_image_id: inv.get_bool("use_image_id"),
        };
        let prepared = self.model.prepare_text(&rendered.text, &images, &videos, &opts)?;
        let n = prepared.prompt.ids.len();
        let room = self.model.engine_capacity().saturating_sub(n);
        if room == 0 {
            return Err(format!("the prompt takes {n} tokens of the {}-token context", self.model.engine_capacity()));
        }
        let opts = GenOptions { max_new: opts.max_new.min(room), ..opts };
        let parsed = ParsedRequest {
            ids: prepared.prompt.ids.clone(),
            max_new: Some(opts.max_new),
            temp,
            top_p,
            top_k,
            sampler: qwen3::chat::plain_sampler(temp, top_k, top_p),
            seed,
            stops: qwen3::chat::parse_stops(inv.get_str("stop").as_deref())?,
            tool_choice: rendered.tool_choice,
            flavor: rendered.flavor,
            thinking_open: rendered.thinking_open,
            raw: rendered.raw,
        };
        Ok(Request { parsed, prepared, opts })
    }
}

/// Load MiniCPM-V from `dir` onto `gpu` (a card index, or the ambient device
/// when `None`) for serving, with the fine-tune in `tune` applied when given
/// (a directory `brain minicpmv finetune` wrote).
pub fn load_session(dir: &str, tune: Option<&Path>, gpu: Option<u32>) -> Result<Session, String> {
    let pipelines = qwen35::model::pipelines();
    let device = match gpu {
        Some(i) => gpu_core::Gpu::new_on_index(i, pipelines)?,
        None => gpu_core::Gpu::new(pipelines),
    };
    let dir = Path::new(dir);
    let model = MiniCpmV::load_tuned(dir, tune, &device, SERVE_CONTEXT, SERVE_CONCURRENT, KvTier::F32)?;
    let template = ChatTemplate::from_model_dir(dir).map_err(|e| format!("minicpmv: chat template: {e:?}"))?;
    Ok(Session::new(model, template))
}

/// Direct provider: builds (and caches) one [`Session`] per checkpoint
/// directory on first use.
#[derive(Default)]
pub struct MiniCpmVProvider {
    default_dir: Option<String>,
}

impl MiniCpmVProvider {
    pub fn new(default_dir: Option<String>) -> MiniCpmVProvider {
        MiniCpmVProvider { default_dir }
    }
}

impl Provider for MiniCpmVProvider {
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

/// One process-wide composite, keyed by directory.
static RESIDENT: Mutex<Option<(String, Session)>> = Mutex::new(None);

impl Action for GenerateAction {
    fn spec(&self) -> ActionSpec {
        generate_spec()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let dir = inv
            .get_str("weights")
            .filter(|s| !s.is_empty())
            .or_else(|| self.default_dir.clone())
            .ok_or("minicpmv generate: no checkpoint (pass 'weights', or configure one through the models directory)")?;
        let mut guard = RESIDENT.lock().map_err(|_| "minicpmv: resident lock poisoned")?;
        if !matches!(&*guard, Some((d, _)) if *d == dir) {
            *guard = None; // drop the old build before the new one allocates
            *guard = Some((dir.clone(), load_session(&dir, None, None)?));
        }
        guard.as_mut().expect("just loaded").1.generate(inv, progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_action_has_the_chat_shape_the_api_serves() {
        let m = manifest();
        let a = &m.actions[0];
        assert_eq!((m.model.as_str(), a.name.as_str()), (MODEL, "generate"));
        assert!(a.streaming);
        for p in ["messages", "prompt", "tools", "temp", "top_p", "seed", "downsample"] {
            assert!(a.params.iter().any(|q| q.name == p), "missing param {p}");
        }
        assert_eq!(a.inputs.iter().filter(|b| b.media == Media::Image).count(), MAX_IMAGES);
        assert!(!manifest_resident().actions[0].params.iter().any(|p| p.name == "weights"));
    }

    #[test]
    fn untagged_images_open_the_last_user_turn_and_tagged_ones_must_match() {
        let inv = Invocation::new().set("prompt", json!("What is this?"));
        let out = with_media_parts(&inv, 2, 0).unwrap();
        let msgs: Value = serde_json::from_str(&out.get_str("messages").unwrap()).unwrap();
        assert_eq!(msgs[0]["content"], json!([{"type": "image"}, {"type": "image"}, {"type": "text", "text": "What is this?"}]));

        let tagged = Invocation::new().set("messages", json!(json!([{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url"}]}]).to_string()));
        let out = with_media_parts(&tagged, 1, 0).unwrap();
        let msgs: Value = serde_json::from_str(&out.get_str("messages").unwrap()).unwrap();
        assert_eq!(msgs[0]["content"][1], json!({"type": "image"}), "an OpenAI image part takes the template's form");
        assert!(with_media_parts(&tagged, 2, 0).unwrap_err().contains("place 1 images"));
        assert!(with_media_parts(&Invocation::new().set("messages", json!("[{\"role\":\"assistant\",\"content\":\"x\"}]")), 1, 0).is_err());
        let both = with_media_parts(&Invocation::new().set("prompt", json!("Compare.")), 1, 1).unwrap();
        let msgs: Value = serde_json::from_str(&both.get_str("messages").unwrap()).unwrap();
        assert_eq!(msgs[0]["content"], json!([{"type": "image"}, {"type": "video"}, {"type": "text", "text": "Compare."}]));
    }
}
