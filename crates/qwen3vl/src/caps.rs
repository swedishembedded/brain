// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `capability::Provider` for Qwen3-VL: image + text in, text out (greedy by
//! default; `temp`/`top_k`/`top_p`/`seed` request real sampling).
//!
//! One action, `generate`, in the SAME chat-capable shape
//! `crates/omni/src/caps.rs::generate_spec()` uses (`messages`/`prompt`,
//! `.streaming()`, `Media::Text` output) - required, not by convention:
//! `apiserve::catalog::api_caps` classifies a model chat-capable only on that
//! exact shape (see `qwen3omnimoe::caps`'s own doc for the full reasoning), and both
//! HTTP handlers always populate `messages`, never a bare `prompt`.
//!
//! Real, working, but validation-tier - the honest scope of what serving
//! wiring for `Qwen3Vl::generate()` turned out to need:
//!
//! - **Image placement is per-request, not baked into the resident model.**
//!   `Qwen3Vl::generate()`'s incremental KV-cache decode derives image
//!   placement dynamically from the token stream (`tok ==
//!   self.image_token_id`), NOT from the `image_row0`/`n_visual` this crate's
//!   `Qwen3Vl::new` takes at construction - those only gate the BATCHED
//!   `forward()` (training) graph, which `generate()` never calls. The
//!   resident model is therefore built ONCE with a generous CAPACITY
//!   (`MAX_VISUAL_TOKENS`, wired to `Qwen3Vl::new`'s `n_visual`), and each
//!   request's actual (smaller-or-equal) image writes only the front of that
//!   capacity - `checkpoint::upload_at`'s own `assert!(offset + len <=
//!   buf.size)` makes an oversized request a loud, immediate error, never a
//!   silent overflow.
//! - **Preprocessing does the real "smart resize"**, not the "caller
//!   supplies already-patch-aligned dimensions" minimum this crate's own
//!   follow-up doc once proposed: `preprocess::smart_resize_default`
//!   computes the patch-aligned target size for ANY input resolution, and
//!   this module bilinear-resizes to it (`resize_bilinear_chw`, the same
//!   shape as `fastvlm::caps::pad_resize_chw` but without the square-pad -
//!   Qwen3-VL's own preprocessor does not pad).
//! - **DeepStack is real** (this session's `qwen3::Qwen::decode_steps`
//!   `deepstack_row` fix, which threads each level's per-row residual add
//!   into the incremental decode path), so real Qwen3-VL-4B checkpoints
//!   (`deepstack_indexes: [5, 11, 17]`) work here, not just DeepStack-free
//!   configs.
//! - **Multi-image is real, via numbered blob keys.** `crates/capability`'s
//!   `Invocation`/`Outcome` blob API (`grep 'fn blob'` there) is keyed by ONE
//!   string name per call -- there is no array-blob wire convention anywhere
//!   in this repo, and adding one (a repeated/dynamic blob) would ripple into
//!   the D-Bus fd-map and every HTTP transport for one input kind (the same
//!   tradeoff `capability::blob::decode_video`'s own doc weighs, for a
//!   different reason). The convention here is instead the simplest thing
//!   consistent with the existing single-blob shape: numbered keys `image`,
//!   `image1`, `image2`, … contiguous from `image`, up to [`MAX_IMAGES`]. A
//!   request with only `image` set is unchanged from before multi-image
//!   support existed -- byte-for-byte, see `crate::model::tests::
//!   generate_is_deterministic_and_respects_eos`'s hardcoded pre-change-output
//!   assertion, and this module's own `decode_images_single_image_backward_
//!   compatible` test. Each image gets its own smart-resize + patch/
//!   token count (`Prepared::build`) and its own vision-start/`[IMG]*`/
//!   vision-end run in the assembled prompt, in key order (`image`, `image1`,
//!   …) -- N runs back-to-back, no text between them, ahead of the user's
//!   prompt text. The resident's `n_visual_capacity` (see [`Resident`]) bounds
//!   the SUM of every image's visual tokens in ONE request, not any single
//!   image's own count -- `qwen3::Qwen::enable_deepstack`'s per-level buffer
//!   is one flat `[n_rows, d_model]` block addressed by a `deepstack_row` that
//!   walks every image's rows in the request in order (see
//!   `crate::model::Qwen3Vl::generate_timed`'s doc), so a capacity sized for
//!   only the largest single image would silently corrupt (or panic on) the
//!   second image of any two-image request that individually fits.
//! - **Tool calling is the same request/response CONTRACT `qwen3::caps`'s own
//!   `generate` has** - `tools`/`tool_choice` params with identical names and
//!   semantics, prompt-level enforcement (`none` withholds the schemas,
//!   `named` must name an offered function) and post-generation enforcement
//!   (`required`/`named` unmet -> `finish_reason: "tool_choice_unmet"`) -
//!   reused directly from `qwen3::chat` (`parse_tools`, `parse_tool_choice`,
//!   `ToolChoice`, `tool_schema_names`, and `SeqState` itself for the
//!   streaming scan + post-hoc enforcement), not reimplemented. Only the
//!   tools *preamble text* is rendered locally (`data::qwen_chat::render`
//!   with no messages, just tools), because `qwen3::chat::parse_request`
//!   renders a full chat prompt as one string and has no seam to splice an
//!   image-token run into the middle of it the way this crate's manual
//!   token assembly needs. **Out of scope**: actually EXECUTING a tool and
//!   feeding its result back in another turn - that is a separate, larger
//!   piece of work (a real agent loop), not part of this contract.

use std::sync::Mutex;

use capability::{Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use data::qwen_chat::{self, TemplateFlavor, TemplateOpts};
use data::tokenizer::Tokenizer;
use qwen3::chat::{parse_tool_choice, parse_tools, tool_schema_names, ParsedRequest, SeqState, ToolChoice};
use serde_json::json;

use crate::config::Qwen3VlConfig;
use crate::model::Qwen3Vl;
use crate::preprocess::{normalize_unit, pack_patches, pack_patches_temporal, pad_frames_to_temporal_multiple, patch_grid, smart_resize};

pub const MODEL: &str = "brain/qwen3vl";

/// One decoded RGB frame/image: raw HWC f32 pixels in `[0,1]` + `(w,h)` -
/// [`capability::blob::decode_image`]/[`capability::blob::decode_video`]'s
/// own element shape, named here so [`decode_media`]'s and
/// [`Resident::generate`]'s signatures read as intent rather than a bare
/// nested tuple (clippy's `type_complexity`, and genuinely clearer either way).
type DecodedFrame = (Vec<f32>, u32, u32);
/// A video request's decoded frames + real fps, or `None` for an image
/// request - [`decode_media`]'s return type and [`Resident::generate`]'s
/// `video_frames` parameter.
pub type VideoFrames = Option<(Vec<DecodedFrame>, f32)>;

/// The largest frame count a video request may carry. A real, bounded scope
/// decision (AGENTS.md: no hours-long/streaming video for this change) -
/// not a verified upstream limit. `Qwen3VlCaptioner::capabilities` reports
/// the same number as `max_frames`, so the two cannot silently disagree
/// about what "a video this model accepts" means.
pub const MAX_VIDEO_FRAMES: u32 = 32;

/// The storage tier this checkpoint's DECODER is built at. The vision tower
/// is always fp32: it is a small fraction of the weights and none of the
/// per-token bandwidth, so narrowing it would trade accuracy for nothing.
///
/// `int8` is **LOSSY** and exists as a named request precisely so that it can
/// never arrive by defaulting. Everything downstream of a caption made this
/// way is downstream of a different model, so it is reported on load and it
/// is part of the resident key - an int8 request never silently reuses an
/// fp32 resident, or the reverse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Precision {
    #[default]
    F32,
    I8,
}

impl Precision {
    /// Parse the user-facing spelling. The ONE place these strings are
    /// recognised, so a CLI flag, an action parameter and an error message
    /// cannot drift about what `int8` is called.
    pub fn from_name(v: &str) -> Result<Precision, String> {
        match v {
            "fp32" | "f32" | "" => Ok(Precision::F32),
            "int8" | "i8" => Ok(Precision::I8),
            other => Err(format!("qwenvl: unknown precision {other:?} (fp32, int8)")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Precision::F32 => "fp32",
            Precision::I8 => "int8",
        }
    }

    fn dtype(self) -> gpu_core::select::Dtype {
        match self {
            Precision::F32 => gpu_core::select::Dtype::F32,
            Precision::I8 => gpu_core::select::Dtype::I8,
        }
    }
}

/// The env var naming the checkpoint directory (or a raw GGUF path) - read
/// directly by [`default_weights`], and (via
/// `crates/cli/src/resident_qwen3vl.rs::Qwen3VlResident::from_env`) the one
/// place the residency adapter learns where its weights live.
pub const DIR_VAR: &str = "BRAIN_QWEN3VL_WEIGHTS";

/// Default checkpoint directory - `$BRAIN_QWEN3VL_WEIGHTS`, never a baked-in
/// absolute path (AGENTS.md: no absolute paths in source).
fn default_weights() -> String {
    std::env::var(DIR_VAR).unwrap_or_default()
}

/// Pixel-area budget for the resident model's DeepStack/splice buffer
/// CAPACITY (see this module's doc) - a practical default (roughly a
/// 1024x1024 image, ~1024 visual tokens at the 4B config's patch/merge
/// granularity), not `preprocess::DEFAULT_MAX_PIXELS`'s own real-checkpoint
/// ceiling (3584² -- see that constant's doc), which would still allocate
/// multiple GB of DeepStack scratch per level for a capacity most requests
/// never approach. Override via the `max_pixels` param for a
/// checkpoint/workload that genuinely needs bigger images.
pub const DEFAULT_SERVE_MAX_PIXELS: u32 = 1024 * 1024;

/// The KV-cache capacity this resident's decoder is BUILT for, before
/// clamping to the checkpoint's own declared ceiling (see
/// [`load_hf_resident`]/[`load_gguf_resident`]).
///
/// `$BRAIN_QWEN3VL_CTX`, mirroring `qwen3`'s own `BRAIN_QWEN_CTX`
/// (`crates/cli/src/resident_llm.rs`) - an env-level operator knob, not a
/// per-request parameter, because it sizes a real device allocation the
/// resident is built with once, not something a caller picks per call.
/// 24576 matches that sibling's own default: real Qwen3-VL-4B-Instruct
/// checkpoints declare a 262144-token `max_position_embeddings`, but this
/// decode path allocates a PLAIN LINEAR fp32 KV cache (`Qwen::
/// new_shard_dt_decode`), not the paged/int8 cache `qwen3::serve::Engine`
/// uses to reach that native length affordably - at the 4B config's
/// `n_layers=36, n_kv_heads=8, head_dim=128`, one token costs
/// `36*8*128*2*4 = 294912` bytes, so the full 262144 would be ~77 GiB of KV
/// alone. 24576 (~7.1 GiB) is a real, working default for a single request;
/// raising it is an explicit operator choice, not a silent truncation - and
/// a request that overflows it is refused BY NAME (see [`Prepared::build`]),
/// never silently cropped.
fn default_ctx_len() -> u32 {
    std::env::var("BRAIN_QWEN3VL_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(24576u32).max(1)
}

/// Cap on the number of images one request may supply -- see this module's
/// doc for the numbered-blob-key convention this bounds. A cap exists so
/// `generate_spec()` can declare a fixed, self-describing set of optional
/// inputs (`image1`..`image{MAX_IMAGES-1}`) rather than an open-ended one;
/// 8 is a practical ceiling for a single chat turn, not a checkpoint limit.
pub const MAX_IMAGES: usize = 8;

/// The wire name of image `i` (0-based): `image`, `image1`, `image2`, … --
/// the ONE place this numbering is spelled, so `generate_spec()` and
/// [`decode_images`] cannot drift about what key holds image 2.
fn image_key(i: usize) -> String {
    if i == 0 { "image".to_string() } else { format!("image{i}") }
}

/// Decode every image blob present, contiguous from `image` (required) up to
/// [`MAX_IMAGES`] -- stops at the first missing numbered key, so `image`,
/// `image1` present but `image2` absent is 2 images, never a request to skip
/// index 2 and check `image3`. Mirrors [`capability::blob::decode_image`]'s
/// own per-blob error shape (names the key, not just "an image").
fn decode_images(inv: &Invocation) -> Result<Vec<(Vec<f32>, u32, u32)>, String> {
    let mut out = Vec::new();
    for i in 0..MAX_IMAGES {
        let key = image_key(i);
        if i > 0 && inv.get_blob(&key).is_none() {
            break;
        }
        out.push(capability::blob::decode_image(inv, &key)?);
    }
    Ok(out)
}

pub fn generate_spec() -> ActionSpec {
    let mut spec = ActionSpec::new("generate", "Qwen3-VL: 1-8 images + text in, greedy text completion (validation-tier -- see this module's doc)")
        .streaming()
        .param(ParamSpec::new("messages", ParamType::Str, "flattened chat messages (JSON array string)"))
        .param(ParamSpec::new("prompt", ParamType::Str, "a raw prompt (alternative to messages)"))
        .param(ParamSpec::new("max_new", ParamType::Int, "max tokens to generate").default(json!(64)))
        .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature (<= 0 = greedy)").default(json!(0.0)).min(0.0).max(2.0).step(0.01))
        .param(ParamSpec::new("top_k", ParamType::Int, "top-k filter (40 = standard; 1 = greedy; 0 or negative = disabled)").default(json!(40)).min(0.0).max(1000.0).step(1.0))
        .param(ParamSpec::new("top_p", ParamType::Float, "nucleus sampling threshold (>= 1 = disabled)").default(json!(1.0)).min(0.0).max(1.0).step(0.01))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed").default(json!(0)))
        .param(
            ParamSpec::new("weights", ParamType::Str, "Qwen3-VL checkpoint DIRECTORY (config.json + model.safetensors[.index.json] + tokenizer.json)")
                .default(json!(default_weights())),
        )
        .param(
            ParamSpec::new(
                "max_pixels",
                ParamType::Int,
                "resident capacity: max input image area in pixels, applied to EACH image independently (up to 8 images/request; a larger single image, or a request whose combined visual tokens exceed this resident's total capacity, errors -- never silently truncates)",
            )
            .default(json!(DEFAULT_SERVE_MAX_PIXELS)),
        )
        .param(
            ParamSpec::new("precision", ParamType::Str, "decoder storage tier: fp32 (default, exact) or int8 (LOSSY, ~4x less weight traffic per token)")
                .default(json!("fp32")),
        )
        .param(ParamSpec::new(
            "fps",
            ParamType::Float,
            "REQUIRED with 'video': the clip's real frames-per-second, used to place each frame in real time \
             (crate::mrope::get_rope_index_video's T axis) rather than by frame count -- see that function's doc \
             for what is verified vs assumed about the formula",
        ))
        .param(ParamSpec::new("tools", ParamType::Str, "JSON array of tool definitions (OpenAI function-calling schema; needs a tokenizer)"))
        .param(ParamSpec::new("tool_choice", ParamType::Str, "tool_choice directive, raw JSON text (\"auto\"|\"none\"|\"required\"|{\"type\":\"function\",...}); none withholds tool schemas, required/named are enforced post-generation (finish_reason \"tool_choice_unmet\" when unmet)"))
        .input(BlobSpec::new("image", Media::Image, "a still image: raw HWC f32 pixels in [0,1], meta {w,h} (capability::blob's wire convention) -- exactly one of 'image'/'video' is required"));
    for i in 1..MAX_IMAGES {
        spec = spec.input(BlobSpec::new(
            &image_key(i),
            Media::Image,
            "optional additional image (same wire shape as 'image'); present contiguous from 'image', i.e. 'image2' is only read if 'image1' is also set. Not used with 'video'.",
        ));
    }
    spec.input(BlobSpec::new(
        "video",
        Media::Video,
        "a short video clip (at most MAX_VIDEO_FRAMES frames): N concatenated interleaved-HWC f32 RGB frames in [0,1], \
         meta {frames,w,h,c=3} (capability::blob::decode_video's wire convention) -- exactly one of 'image'/'video' is required",
    ))
    .output(BlobSpec::new("text", Media::Text, "the generated continuation"))
}

/// `lora_train` - fine-tune a LoRA adapter for the decoder on a folder of
/// captioned images. The vision tower stays frozen (see
/// `crate::finetune`'s own module doc: this composite's `DecoderBuild::Batched`
/// only ever gives the DECODER gradient buffers) - only the decoder's
/// attention+MLP projections adapt.
pub fn lora_train_spec() -> ActionSpec {
    ActionSpec::new("lora_train", "fine-tune a LoRA adapter for the decoder on a folder of captioned images (data::imageset's captions.yaml/.jsonl - the format `brain label` writes)")
        .streaming()
        .param(ParamSpec::new("data", ParamType::Str, "folder with images + a captions.yaml (`filename: prompt`) and/or captions.jsonl").required())
        .param(ParamSpec::new("save", ParamType::Str, "output path for the trained adapter").required())
        .param(
            ParamSpec::new("weights", ParamType::Str, "base Qwen3-VL checkpoint DIRECTORY to adapt (config.json + model.safetensors[.index.json] + tokenizer.json)")
                .default(json!(default_weights())),
        )
        .param(ParamSpec::new("rank", ParamType::Int, "LoRA rank (capacity/size tradeoff)").default(json!(8)))
        .param(ParamSpec::new("alpha", ParamType::Float, "LoRA alpha (delta scale = alpha/rank)").default(json!(16.0)))
        .param(ParamSpec::new("steps", ParamType::Int, "training steps").default(json!(200)))
        .param(ParamSpec::new("lr", ParamType::Float, "peak learning rate (cosine schedule)").default(json!(1e-4)))
        .param(ParamSpec::new("size", ParamType::Int, "training image square size, px (must be a multiple of patch_size*spatial_merge_size)").default(json!(224)))
        .param(ParamSpec::new("seq_len", ParamType::Int, "fixed per-sample token budget (prompt + caption, padded); a caption that overflows it is skipped, not truncated").default(json!(256)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed").default(json!(0)))
        .output(BlobSpec::new("adapter", Media::Bytes, "the trained LoRA adapter checkpoint"))
}

pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "Qwen3-VL -- image + text in, text out. Validation-tier: fp32 weights by default \
         (int8 decoder opt-in), temperature/top-k/top-p sampling (greedy by default), \
         one request at a time (no batching), context capacity set by $BRAIN_QWEN3VL_CTX \
         (clamped to the checkpoint's own declared max_position_embeddings). \
         `lora_train` fine-tunes a decoder-only LoRA adapter on a captioned-image folder.",
        vec![generate_spec(), lora_train_spec()],
    )
}

/// The manifest for the RESIDENT/scheduled service (D-Bus, executor, HTTP):
/// the checkpoint directory is service-side configuration ([`DIR_VAR`]), so
/// the served action carries only real per-request parameters - see
/// `moondream3::caps::manifest_resident`'s doc for why a static, CLI-facing
/// manifest and a stripped resident one are two different things, not one
/// hidden behind deployment state. `crate::resident_qwen3vl::Qwen3VlResident::
/// manifest` (in `crates/cli`) calls this rather than [`manifest`].
pub fn manifest_resident() -> Manifest {
    let mut m = manifest();
    for a in &mut m.actions {
        a.params.retain(|p| p.name != "weights");
    }
    m
}

use capability::last_user_text;

/// A built Qwen3-VL checkpoint: the model, its tokenizer and the config it
/// was assembled from.
///
/// `pub` so `crates/cli/src/resident_qwen3vl.rs`'s residency adapter can own
/// one directly ([`Resident::load_on`]/[`Resident::generate`]) - the
/// residency adapter and this crate's own [`GenerateAction`] (behind the
/// process-wide [`RESIDENT`] static below) then run the SAME code and cannot
/// drift about preprocessing, prompt assembly or token accounting, matching
/// `moondream3::caps::Session`'s split.
pub struct Resident {
    weights: String,
    max_pixels: u32,
    /// The tier this resident was BUILT at, part of its key - see
    /// [`Precision`]. What the device actually landed on is
    /// `Qwen3Vl::linear_dtype`, which `load_resident` checks against this.
    precision: Precision,
    /// How many visual tokens this resident's DeepStack/splice buffers were
    /// allocated for (computed once at construction from `max_pixels`) - see
    /// this module's own doc on why construction-time capacity, not one
    /// request's exact size. Sized for up to [`MAX_IMAGES`] images each at
    /// `max_pixels` (`visual_capacity(cfg, max_pixels) * MAX_IMAGES`) -
    /// bounding the SUM of one request's images, not any single image's own
    /// count, because the decoder's DeepStack buffer is one flat block a
    /// request's images all share (see this module's own doc).
    n_visual_capacity: u32,
    /// The KV-cache capacity this resident's decoder was actually BUILT for -
    /// `min(default_ctx_len(), cfg.text.max_position_embeddings)`, see
    /// [`load_hf_resident`]. Read by [`Prepared::build`] instead of a
    /// compile-time constant, so a request's context error names the real
    /// number this checkpoint and this box actually support.
    seq_len: u32,
    cfg: Qwen3VlConfig,
    model: Qwen3Vl,
    tok: data::qwen_tokenizer::QwenBpe,
}

impl Resident {
    /// [`load_resident`] on a chosen physical card (`gpu_core::devices`'
    /// canonical index), or `None` for the CPU backend.
    ///
    /// Placement is a SCOPED registry selection (`gpu_core::devices::with_gpu`),
    /// never an env mutation - a server-lifetime resident must not change the
    /// backend every other model builds on afterwards. `Qwen::
    /// new_shard_dt_decode` (which `Qwen3Vl::new` calls) already documents
    /// that it lands on "the ambient selection (`--device` / scoped
    /// `with_gpu`)", so scoping the call here is sufficient - no device
    /// parameter needs to thread through `crate::model`.
    pub fn load_on(dir: &str, max_pixels: u32, precision: Precision, gpu: Option<u32>) -> Result<Resident, String> {
        match gpu {
            None => load_resident(dir, max_pixels, precision),
            Some(i) => gpu_core::devices::with_gpu(i, || load_resident(dir, max_pixels, precision))?,
        }
    }

    /// Run one `generate` invocation against this already-built resident:
    /// image(s)/video + prompt in, streamed text out. The body
    /// [`GenerateAction::run`] used to hold directly, extracted so a
    /// residency-scheduled instance and the direct provider execute
    /// byte-for-byte the same code.
    ///
    /// `video_frames` and `tool_choice`/`tools` are already validated by
    /// [`GenerateAction::run`] (`video_frames` is `None` for an image
    /// request) - see that function's doc for why the "exactly one of
    /// image/video", frame-cap/fps and named-tool-choice checks live THERE
    /// and not here: they must fail before this resident is even built, and
    /// re-deriving them here (after `with_resident` has already run) would
    /// be too late for that contract, not just a duplicate.
    pub fn generate(
        &self,
        inv: &Invocation,
        video_frames: VideoFrames,
        tool_choice: ToolChoice,
        tools: &[String],
        progress: &mut dyn FnMut(Progress),
    ) -> ActionResult {
        let prompt = last_user_text(inv);
        if prompt.trim().is_empty() {
            return Err("qwenvl generate: empty prompt (need 'messages' with a user turn, or 'prompt')".to_string());
        }
        let max_new = inv.get_i64("max_new").unwrap_or(64).clamp(1, 2048) as u32;

        let temperature = inv.get_f64("temp").unwrap_or(0.0).max(0.0) as f32;
        let top_k = inv.get_i64("top_k").unwrap_or(40).max(0) as usize;
        let top_p = inv.get_f64("top_p").unwrap_or(1.0) as f32;
        let seed = inv.get_i64("seed").unwrap_or(0).max(0) as u64;
        let sample = crate::model::SampleParams { temperature, top_k, top_p };
        let mut rng = data::rng::Rng::new(seed);

        // Shared chat-serving state (`qwen3::chat::SeqState`): scans the
        // decode stream for `<tool_call>` markup and resolves the SAME
        // `tool_choice` enforcement + `Outcome` shape
        // (`prompt_tokens`/`completion_tokens`/`finish_reason`/
        // `reasoning_content`[/`tool_calls`]) `qwen3::caps`'s own
        // `GenerateAction` returns on `finish` - shared by both the video and
        // image branches below, not a second copy per media kind.
        let build_request = |ids: Vec<u32>| ParsedRequest {
            ids,
            max_new: max_new as usize,
            temp: temperature,
            top_k,
            top_p,
            seed,
            stops: Vec::new(),
            tool_choice: tool_choice.clone(),
            flavor: TemplateFlavor::Qwen3,
            thinking_open: false,
        };
        if let Some((frames, fps)) = video_frames {
            let p = PreparedVideo::build(self, &frames, fps, &prompt, max_new, tools)?;
            let req = build_request(p.tokens.clone());
            let mut seq = SeqState::new(&req, inv.cancel.clone());
            let mut ids: Vec<u32> = Vec::new();
            progress(Progress::step(0, max_new, "generating"));
            let out_ids = self.model.generate_video_cb(
                &p.tokens,
                p.grid,
                p.n_frames,
                &p.pixels,
                self.cfg.video_token_id,
                &p.frame_timestamps_s,
                self.cfg.vision.tokens_per_second as f32,
                max_new,
                &p.eos,
                sample,
                &mut rng,
                |tok_id| {
                    ids.push(tok_id);
                    seq.advance(&self.tok, &ids, progress);
                },
            );
            Ok(seq.finish(&self.tok, &out_ids, progress))
        } else {
            let images = decode_images(inv)?;
            let p = Prepared::build_multi(self, &images, &prompt, max_new, tools)?;
            let req = build_request(p.tokens.clone());
            let mut seq = SeqState::new(&req, inv.cancel.clone());
            let mut ids: Vec<u32> = Vec::new();
            progress(Progress::step(0, max_new, "generating"));
            let out_ids = self.model.generate_cb(&p.tokens, &p.image_inputs(), max_new, &p.eos, sample, &mut rng, |tok_id| {
                ids.push(tok_id);
                seq.advance(&self.tok, &ids, progress);
            });
            Ok(seq.finish(&self.tok, &out_ids, progress))
        }
    }
}

pub struct QwenVlProvider;

impl QwenVlProvider {
    pub fn new() -> QwenVlProvider {
        QwenVlProvider
    }
}

impl Default for QwenVlProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for QwenVlProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<std::sync::Arc<dyn Action>> {
        match name {
            "generate" => Some(std::sync::Arc::new(GenerateAction) as std::sync::Arc<dyn Action>),
            "lora_train" => Some(std::sync::Arc::new(LoraTrainAction) as std::sync::Arc<dyn Action>),
            _ => None,
        }
    }
}

/// Which media a `generate` request carries -- exactly one of image(s)/video
/// -- and, for a video, its decoded frames + fps. Called by both
/// [`GenerateAction::run`] (BEFORE any checkpoint I/O, so a malformed request
/// fails without building or touching a resident) and
/// `crates/cli/src/resident_qwen3vl.rs`'s `Instance::run` (where the resident
/// is already built by residency's own `activate`, but the media shape still
/// needs validating and decoding exactly once, the same way, rather than a
/// second copy of this logic per call site).
pub fn decode_media(inv: &Invocation) -> Result<VideoFrames, String> {
    let has_video = inv.get_blob("video").is_some();
    let has_image = inv.get_blob("image").is_some();
    if has_video == has_image {
        return Err(format!("qwenvl generate: exactly one of 'image'/'video' is required (got image={has_image}, video={has_video})"));
    }
    if !has_video {
        return Ok(None);
    }
    let frames = capability::blob::decode_video(inv, "video")?;
    if frames.len() as u32 > MAX_VIDEO_FRAMES {
        return Err(format!("qwenvl generate: video has {} frames, this model accepts at most {MAX_VIDEO_FRAMES}", frames.len()));
    }
    let fps = inv
        .get_f64("fps")
        .filter(|f| *f > 0.0)
        .ok_or("qwenvl generate: a 'video' input needs a positive 'fps' to place its frames in real time")? as f32;
    Ok(Some((frames, fps)))
}

/// The tool-calling request contract - reused directly from `qwen3::chat`
/// (see this module's doc): `none` withholds the tool schemas from the
/// rendered prompt entirely, `named` must name a function the `tools` array
/// actually offers. Validated before any weights are touched, the same
/// precedence `qwen3::chat::parse_request` gives this check (a typo'd name
/// would otherwise degrade into a guaranteed-unmet post-hoc demand). Called
/// by both [`GenerateAction::run`] and `resident_qwen3vl.rs`'s
/// `Instance::run` for the same reason [`decode_media`] is - one
/// implementation, not a copy per call site.
pub fn parse_tool_request(inv: &Invocation) -> Result<(ToolChoice, Vec<String>), String> {
    let tool_choice = parse_tool_choice(inv.get_str("tool_choice").as_deref())?;
    let mut tools = parse_tools(inv.get_str("tools").as_deref())?;
    match &tool_choice {
        ToolChoice::None => tools.clear(),
        ToolChoice::Named(name) => {
            if !tool_schema_names(&tools).iter().any(|n| n == name) {
                return Err(format!("qwenvl generate: tool_choice names function '{name}' which is not present in tools"));
            }
        }
        _ => {}
    }
    Ok((tool_choice, tools))
}

// One process-wide resident, keyed by (checkpoint dir, max_pixels) so
// switching either swaps cleanly -- mirrors fastvlm::caps's DECODE static.
static RESIDENT: Mutex<Option<Resident>> = Mutex::new(None);

struct GenerateAction;

impl Action for GenerateAction {
    fn spec(&self) -> ActionSpec {
        generate_spec()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        // Checked before ANY checkpoint I/O, per
        // `empty_prompt_is_a_clean_error_before_touching_weights` below.
        if last_user_text(inv).trim().is_empty() {
            return Err("qwenvl generate: empty prompt (need 'messages' with a user turn, or 'prompt')".to_string());
        }
        let video_frames = decode_media(inv)?;
        let (tool_choice, tools) = parse_tool_request(inv)?;

        let dir = inv.get_str("weights").filter(|s| !s.is_empty()).unwrap_or_else(default_weights);
        if dir.is_empty() {
            return Err("qwenvl generate: no checkpoint directory (set 'weights' or $BRAIN_QWEN3VL_WEIGHTS)".to_string());
        }
        let max_pixels = inv.get_i64("max_pixels").unwrap_or(DEFAULT_SERVE_MAX_PIXELS as i64).max(1) as u32;
        let precision = Precision::from_name(inv.get_str("precision").unwrap_or_default().as_str())?;
        with_resident(&dir, max_pixels, precision, |hot| hot.generate(inv, video_frames, tool_choice, &tools, progress))
    }
}

struct LoraTrainAction;

impl Action for LoraTrainAction {
    fn spec(&self) -> ActionSpec {
        lora_train_spec()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let dir = inv.get_str("weights").filter(|s| !s.is_empty()).unwrap_or_else(default_weights);
        if dir.is_empty() {
            return Err("qwenvl lora_train: no base checkpoint directory (set 'weights' or $BRAIN_QWEN3VL_WEIGHTS)".to_string());
        }
        let data = inv.get_str("data").ok_or("qwenvl lora_train: 'data' folder is required")?;
        let save = inv.get_str("save").ok_or("qwenvl lora_train: 'save' path is required")?;
        let opts = crate::finetune::TrainOpts {
            rank: inv.get_i64("rank").unwrap_or(8).max(1) as u32,
            alpha: inv.get_f64("alpha").unwrap_or(16.0) as f32,
            steps: inv.get_i64("steps").unwrap_or(200).max(1) as u32,
            lr: inv.get_f64("lr").unwrap_or(1e-4) as f32,
            size: inv.get_i64("size").unwrap_or(224).max(1) as u32,
            seq_len: inv.get_i64("seq_len").unwrap_or(256).max(1) as u32,
            seed: inv.get_i64("seed").unwrap_or(0).max(0) as u64,
            ..crate::finetune::TrainOpts::default()
        };
        let mut prog = |step: u32, total: u32, message: String| progress(Progress::step(step, total, message));
        let (initial_loss, final_loss) = crate::finetune::run(&dir, std::path::Path::new(&data), &opts, &save, &inv.cancel, &mut prog)?;

        // Return the trained artifact itself, not just its server-side path -
        // a remote client has no filesystem access to `save` (serving-contract:
        // "training actions return their artifact as a blob").
        let bytes = std::fs::read(&save).map_err(|e| format!("qwenvl lora_train: read trained adapter '{save}': {e}"))?;
        Ok(Outcome::new()
            .set("adapter", json!(save))
            .set("steps", json!(opts.steps))
            .set("initial_loss", json!(initial_loss))
            .set("final_loss", json!(final_loss))
            .blob("adapter", Blob::new(Media::Bytes, bytes).with_meta(json!({"path": save}))))
    }
}

/// Run `f` against the process-wide resident for `(dir, max_pixels)`, building
/// it first when the key changed.
///
/// The ONE place the resident is built and the lock is held, so a second entry
/// point (the profiler below) cannot acquire it in a different order or forget
/// to swap on a key change. `f` runs under the lock, which is also the
/// concurrency contract this action already had ("one request at a time").
fn with_resident<T>(dir: &str, max_pixels: u32, precision: Precision, f: impl FnOnce(&Resident) -> Result<T, String>) -> Result<T, String> {
    let mut guard = RESIDENT.lock().map_err(|_| "qwenvl: resident lock poisoned")?;
    if !matches!(&*guard, Some(r) if r.weights == dir && r.max_pixels == max_pixels && r.precision == precision) {
        *guard = None;
        *guard = Some(load_resident(dir, max_pixels, precision)?);
    }
    f(guard.as_ref().unwrap())
}

/// One image's preprocessed input, ready for [`crate::model::ImageInput`] to
/// borrow from (pre-merge patch grid + packed `[N, patch_vec]` pixels).
struct PreparedImage {
    grid: (u32, u32),
    pixels: Vec<f32>,
}

/// Everything one request needs from its images and prompt, before any model
/// weight is touched: each image's packed patch tensor and grid, the token
/// stream with every image's vision run spliced in (in image order), and the
/// stop set.
///
/// Extracted so the served action and the profiler share it byte-for-byte -
/// two copies of the prompt assembly would be free to disagree about the chat
/// template, and a profile of a DIFFERENT prompt than the one that ships is
/// not a profile of anything.
struct Prepared {
    tokens: Vec<u32>,
    eos: Vec<u32>,
    images: Vec<PreparedImage>,
    n_visual: u32,
    /// Host-side preprocessing wall time (smart resize, resample, normalize,
    /// pack, over every image) - a stage in its own right at real image sizes.
    preprocess_s: f64,
}

impl Prepared {
    /// [`Self::build_multi`] for exactly one image -- every caller before
    /// multi-image support existed (`generate_profiled`, this module's own
    /// single-image tests) reaches this, so it must reproduce their output
    /// byte-for-byte, which it does by definition: it is a zero-length loop
    /// away from `build_multi` itself, not a second copy of the prompt/patch
    /// assembly.
    fn build(hot: &Resident, hwc: &[f32], w: u32, h: u32, prompt: &str, max_new: u32, tools: &[String]) -> Result<Prepared, String> {
        Self::build_multi(hot, std::slice::from_ref(&(hwc.to_vec(), w, h)), prompt, max_new, tools)
    }

    /// Assemble ONE prompt from N images (in request order): each image gets
    /// its own smart-resize + patch/token count, computed independently (a
    /// wide image and a tall one in the same request are each resized to
    /// their own patch-aligned target, not forced to share one), and its own
    /// vision-start/`[IMG]*`/vision-end run, back-to-back with no text
    /// between runs, ahead of the user's prompt text - see this module's doc
    /// for the request shape this backs.
    fn build_multi(hot: &Resident, images: &[(Vec<f32>, u32, u32)], prompt: &str, max_new: u32, tools: &[String]) -> Result<Prepared, String> {
        let t0 = std::time::Instant::now();
        if images.is_empty() {
            return Err("qwenvl generate: at least one image is required".to_string());
        }
        let factor = hot.cfg.vision.patch_size * hot.cfg.vision.spatial_merge_size;

        // Tools preamble (a system turn carrying the `<tools>` JSON block),
        // rendered from the SAME `data::qwen_chat::render` the shared
        // `qwen3::chat` path itself calls on an empty message list. This is
        // the one piece of `parse_request`'s rendering this crate cannot
        // call directly: `render` returns one whole messages+tools prompt as
        // a single string, with no seam to splice an image-token run into
        // the middle of it the way Qwen3-VL needs.
        //
        // NOT byte-for-byte identical to what `qwen3::caps generate` renders
        // for the same tools: `qwen3::chat::parse_request` resolves
        // `reasoning_effort` to `Some("xhigh")` whenever `enable_thinking` is
        // true (its own default), and `qwen_chat::render` then injects an
        // extra "Reasoning effort is set to..." directive paragraph into the
        // preamble for both template flavors. This action has no
        // `enable_thinking`/`reasoning_effort` param, so `TemplateOpts`
        // resolves that to `None` here and the directive paragraph is always
        // omitted. The `<tools>` JSON block and surrounding structure match;
        // this one paragraph does not - pinned by
        // `caps::tests::tools_preamble_matches_qwen3_except_the_reasoning_effort_directive_it_cannot_opt_into`
        // rather than left as an unverified comment.
        let mut tokens = if tools.is_empty() {
            Vec::new()
        } else {
            let preamble = qwen_chat::render(&[], tools, TemplateOpts { add_generation_prompt: false, ..Default::default() })?;
            hot.tok.encode(&preamble)
        };
        tokens.extend(hot.tok.encode("<|im_start|>user\n"));
        let mut prepared = Vec::with_capacity(images.len());
        let mut n_visual = 0u32;
        for (hwc, w, h) in images {
            // Smart-resize to a patch-aligned target, bilinear resample, pack
            // patches -- the same three real preprocessing steps the
            // checkpoint's HF processor runs, no "caller must pre-align"
            // shortcut. Independent per image: a request's images need not
            // share a resolution or aspect ratio.
            let (h_bar, w_bar) = smart_resize(*h, *w, factor, preprocess_min_pixels(), hot.max_pixels);
            let n_visual_i = crate::preprocess::image_token_count(h_bar, w_bar, hot.cfg.vision.patch_size, hot.cfg.vision.spatial_merge_size);
            n_visual += n_visual_i;
            if n_visual > hot.n_visual_capacity {
                return Err(format!(
                    "qwenvl generate: this request's {} image(s) need {n_visual} visual tokens combined, exceeding \
                     this resident's capacity {} (raise 'max_pixels', or send fewer/smaller images -- current cap {} px/image)",
                    images.len(),
                    hot.n_visual_capacity,
                    hot.max_pixels
                ));
            }
            let mut chw = hwc_to_chw_resized(hwc, *w, *h, w_bar, h_bar);
            normalize_unit(&mut chw);
            let pixels =
                pack_patches(&chw, hot.cfg.vision.in_channels, h_bar, w_bar, hot.cfg.vision.patch_size, hot.cfg.vision.spatial_merge_size, hot.cfg.vision.temporal_patch_size);
            let grid = patch_grid(h_bar, w_bar, hot.cfg.vision.patch_size);

            tokens.push(hot.cfg.vision_start_token_id);
            tokens.extend(std::iter::repeat_n(hot.cfg.image_token_id, n_visual_i as usize));
            tokens.push(hot.cfg.vision_end_token_id);
            prepared.push(PreparedImage { grid, pixels });
        }
        tokens.extend(hot.tok.encode(&format!("{prompt}<|im_end|>\n<|im_start|>assistant\n")));
        let eos = hot.tok.encode("<|im_end|>");

        if tokens.len() as u32 + max_new > hot.seq_len {
            return Err(format!(
                "qwenvl generate: prompt ({} tokens incl. {n_visual} image tokens across {} image(s)) + max_new ({max_new}) \
                 exceeds this resident's context {} (set $BRAIN_QWEN3VL_CTX to raise it, up to this checkpoint's own {} max_position_embeddings)",
                tokens.len(),
                images.len(),
                hot.seq_len,
                hot.cfg.text.max_position_embeddings
            ));
        }
        Ok(Prepared { tokens, eos, images: prepared, n_visual, preprocess_s: t0.elapsed().as_secs_f64() })
    }

    /// Borrowed [`crate::model::ImageInput`] view over [`Self::images`], the
    /// shape `Qwen3Vl::generate_cb`/`generate_timed` take.
    fn image_inputs(&self) -> Vec<crate::model::ImageInput<'_>> {
        self.images.iter().map(|im| crate::model::ImageInput { grid: im.grid, pixels: &im.pixels }).collect()
    }
}

/// [`Prepared`]'s video counterpart: the packed multi-frame patch tensor, one
/// REAL timestamp per merged temporal group, and the token stream with the
/// video run spliced in under `video_token_id`.
struct PreparedVideo {
    tokens: Vec<u32>,
    eos: Vec<u32>,
    grid: (u32, u32),
    n_frames: u32,
    pixels: Vec<f32>,
    /// One real timestamp (seconds) per merged temporal group -- see
    /// [`crate::mrope::get_rope_index_video`]'s doc for how this drives the
    /// T axis and exactly what is verified vs assumed about the formula.
    frame_timestamps_s: Vec<f32>,
}

impl PreparedVideo {
    /// `frames` is [`capability::blob::decode_video`]'s output (`(hwc, w,
    /// h)` per RAW frame, in order); `fps` is the clip's real, constant
    /// frame rate -- the ONLY per-clip timing this crate's `captioner::Clip`
    /// contract carries (`Clip::fps: Option<f32>`), so it is also all a
    /// served request can supply here: `frame[i]`'s real timestamp is
    /// `i / fps`. Every frame is resized to ONE shared smart-resize target
    /// (picked from the FIRST frame's dimensions, matching HF's own video
    /// preprocessor, which assumes one resolution per clip), padded up to a
    /// multiple of `temporal_patch_size` by repeating the last frame
    /// ([`pad_frames_to_temporal_multiple`]), then packed with
    /// [`pack_patches_temporal`].
    fn build(hot: &Resident, frames: &[(Vec<f32>, u32, u32)], fps: f32, prompt: &str, max_new: u32, tools: &[String]) -> Result<PreparedVideo, String> {
        let (_, w0, h0) = *frames.first().ok_or("qwenvl generate: an empty video has no frames to caption")?;
        let factor = hot.cfg.vision.patch_size * hot.cfg.vision.spatial_merge_size;
        let (h_bar, w_bar) = smart_resize(h0, w0, factor, preprocess_min_pixels(), hot.max_pixels);

        let mut chw_frames: Vec<Vec<f32>> = Vec::with_capacity(frames.len());
        for (hwc, w, h) in frames {
            let mut chw = hwc_to_chw_resized(hwc, *w, *h, w_bar, h_bar);
            normalize_unit(&mut chw);
            chw_frames.push(chw);
        }
        let temporal = hot.cfg.vision.temporal_patch_size;
        let refs: Vec<&[f32]> = chw_frames.iter().map(|f| f.as_slice()).collect();
        let padded = pad_frames_to_temporal_multiple(&refs, temporal);
        let n_frames = padded.len() as u32 / temporal;

        let pixels = pack_patches_temporal(&padded, hot.cfg.vision.in_channels, h_bar, w_bar, hot.cfg.vision.patch_size, hot.cfg.vision.spatial_merge_size, temporal);

        let n_visual_per_frame = crate::preprocess::image_token_count(h_bar, w_bar, hot.cfg.vision.patch_size, hot.cfg.vision.spatial_merge_size);
        let n_visual = n_frames * n_visual_per_frame;
        if n_visual > hot.n_visual_capacity {
            return Err(format!(
                "qwenvl generate: video needs {n_visual} visual tokens across {n_frames} frame group(s), exceeding \
                 this resident's capacity {} (raise 'max_pixels' -- current cap {} px)",
                hot.n_visual_capacity, hot.max_pixels
            ));
        }

        // One real timestamp per RAW frame, extended over the padding
        // exactly the way the padding itself was built (repeat the last
        // real timestamp for a repeated last frame -- it is the same
        // moment, not a new one), then one per merged GROUP, anchored at
        // that group's first raw frame.
        let mut padded_ts: Vec<f32> = (0..frames.len() as u32).map(|i| i as f32 / fps).collect();
        while (padded_ts.len() as u32) < padded.len() as u32 {
            padded_ts.push(*padded_ts.last().unwrap());
        }
        let frame_timestamps_s: Vec<f32> = (0..n_frames).map(|g| padded_ts[(g * temporal) as usize]).collect();

        // Tools preamble (a system turn carrying the `<tools>` JSON block),
        // rendered from the SAME `data::qwen_chat::render` the shared
        // `qwen3::chat` path itself calls on an empty message list. This is
        // the one piece of `parse_request`'s rendering this crate cannot
        // call directly: `render` returns one whole messages+tools prompt as
        // a single string, with no seam to splice an image-token run into
        // the middle of it the way Qwen3-VL needs.
        //
        // NOT byte-for-byte identical to what `qwen3::caps generate` renders
        // for the same tools: `qwen3::chat::parse_request` resolves
        // `reasoning_effort` to `Some("xhigh")` whenever `enable_thinking` is
        // true (its own default), and `qwen_chat::render` then injects an
        // extra "Reasoning effort is set to..." directive paragraph into the
        // preamble for both template flavors. This action has no
        // `enable_thinking`/`reasoning_effort` param, so `TemplateOpts`
        // resolves that to `None` here and the directive paragraph is always
        // omitted. The `<tools>` JSON block and surrounding structure match;
        // this one paragraph does not.
        let mut tokens = if tools.is_empty() {
            Vec::new()
        } else {
            let preamble = qwen_chat::render(&[], tools, TemplateOpts { add_generation_prompt: false, ..Default::default() })?;
            hot.tok.encode(&preamble)
        };
        // Prompt: <|im_start|>user\n <|vision_start|> [VIDEO]*n_visual <|vision_end|> {prompt}<|im_end|>\n<|im_start|>assistant\n
        tokens.extend(hot.tok.encode("<|im_start|>user\n"));
        tokens.push(hot.cfg.vision_start_token_id);
        tokens.extend(std::iter::repeat_n(hot.cfg.video_token_id, n_visual as usize));
        tokens.push(hot.cfg.vision_end_token_id);
        tokens.extend(hot.tok.encode(&format!("{prompt}<|im_end|>\n<|im_start|>assistant\n")));
        let eos = hot.tok.encode("<|im_end|>");

        if tokens.len() as u32 + max_new > hot.seq_len {
            return Err(format!(
                "qwenvl generate: prompt ({} tokens incl. {n_visual} video tokens) + max_new ({max_new}) \
                 exceeds this resident's context {} (set $BRAIN_QWEN3VL_CTX to raise it, up to this checkpoint's own {} max_position_embeddings)",
                tokens.len(),
                hot.seq_len,
                hot.cfg.text.max_position_embeddings
            ));
        }
        let grid = patch_grid(h_bar, w_bar, hot.cfg.vision.patch_size);
        Ok(PreparedVideo { tokens, eos, grid, n_frames, pixels, frame_timestamps_s })
    }
}

/// One caption through the SAME resident, preprocessing, prompt assembly and
/// decode the `generate` action runs, with per-stage wall-clock attribution.
///
/// The seam `qwen3vl_bench` measures from. It exists so a profile is a profile
/// of the shipped path: a bench that re-assembled the prompt or re-built the
/// model itself would be measuring its own copy, and every optimisation would
/// then be justified against a program nobody runs.
#[allow(clippy::too_many_arguments)]
pub fn generate_profiled(
    dir: &str,
    max_pixels: u32,
    precision: Precision,
    prompt: &str,
    hwc: &[f32],
    w: u32,
    h: u32,
    max_new: u32,
) -> Result<(String, crate::model::StageTimes, f64), String> {
    with_resident(dir, max_pixels, precision, |hot| {
        let p = Prepared::build(hot, hwc, w, h, prompt, max_new, &[])?;
        // Greedy: a bench measures cost, and a deterministic decode keeps
        // `qwen3vl_bench compare`'s tier-divergence numbers reproducible run
        // to run rather than confounded by sampling noise.
        let mut rng = data::rng::Rng::new(0);
        let (ids, st) = hot.model.generate_timed(&p.tokens, &p.image_inputs(), max_new, &p.eos, crate::model::SampleParams::greedy(), &mut rng, |_| {});
        debug_assert_eq!(st.visual_tokens, p.n_visual);
        Ok((hot.tok.decode(&ids), st, p.preprocess_s))
    })
}

/// The MEASURED roofline of the device this checkpoint's resident runs on,
/// measuring it once if this process has not already.
///
/// `gpu_core::roof::known` only ever answers from the in-process cache, so a
/// caller that has not itself called `ensure` on a real handle gets `None` and
/// silently reports every stage as "no measured roof". The handle lives behind
/// the resident, so this is where the two can meet.
pub fn device_roof(dir: &str, max_pixels: u32, precision: Precision) -> Result<Option<gpu_core::roof::Roofs>, String> {
    with_resident(dir, max_pixels, precision, |hot| Ok(gpu_core::roof::ensure(hot.model.gpu())))
}

/// The tier this checkpoint's decoder ACTUALLY landed on, as the same string
/// [`Precision::name`] uses.
///
/// `None` when the shard owns no layer at all. The distinction that matters is
/// between what was ASKED for and what runs: a device that cannot serve a
/// packed int8 dot has its request promoted back to fp32 by `Weight::upload`,
/// and a caller reporting its own request would claim a lossy run that never
/// happened - or, worse, miss one that did.
pub fn linear_dtype(dir: &str, max_pixels: u32, precision: Precision) -> Result<Option<String>, String> {
    use gpu_core::select::Dtype;
    // Exhaustive, and deliberately not a `Debug` format: this string is
    // compared against [`Precision::name`]'s vocabulary, and `{:?}` spells
    // `F32` as "f32" where that vocabulary says "fp32". Matching every variant
    // means a new tier is a compile error here rather than a name that
    // silently matches nothing.
    let name = |dt: Dtype| match dt {
        Dtype::F32 => Precision::F32.name(),
        Dtype::I8 => Precision::I8.name(),
        Dtype::F16 => "f16",
        Dtype::BF16 => "bf16",
        Dtype::Q4 => "q4",
        // M12's affine K-quant tiers - no `qwen3vl` resident ever builds a
        // `Weight::KQuant` today (this crate has no GGUF K-quant loader),
        // but the match must stay exhaustive per this function's own doc
        // comment above.
        Dtype::Q4K => "q4k",
        Dtype::Q8K => "q8k",
    };
    with_resident(dir, max_pixels, precision, |hot| Ok(hot.model.linear_dtype().map(|dt| name(dt).to_string())))
}

/// Print the per-kernel `BRAIN_PROFILE` table for this checkpoint's resident.
///
/// The accumulator normally prints when the device drops, and a resident
/// model's device never does - so on this path the table was simply never
/// emitted. `Gpu::dump_profile` is the escape hatch for exactly that; this is
/// where a caller with a checkpoint path can reach it. `BRAIN_PROFILE` must
/// already be set when the resident is BUILT, since the backend reads it once
/// at construction.
pub fn dump_profile(dir: &str, max_pixels: u32, precision: Precision) -> Result<(), String> {
    with_resident(dir, max_pixels, precision, |hot| {
        hot.model.gpu().dump_profile();
        Ok(())
    })
}

/// Build (or reuse) the resident for `(dir, max_pixels)` and report how long
/// that took, without generating anything - the one-off setup cost a caption
/// profile must not fold into its per-image numbers.
pub fn load_time(dir: &str, max_pixels: u32, precision: Precision) -> Result<f64, String> {
    let t0 = std::time::Instant::now();
    with_resident(dir, max_pixels, precision, |_| Ok(()))?;
    Ok(t0.elapsed().as_secs_f64())
}

/// Say on every load which tier the decoder ACTUALLY landed on, and refuse a
/// silent fallback.
///
/// A lossy tier is a different model, so a run that asked for it and did not
/// get it must not look like a run that did - and a run that got it must say
/// so where the operator can see it, because nothing downstream of a caption
/// can tell. `Weight::upload` promotes a request the device cannot serve back
/// to fp32; that is a legitimate outcome and an illegitimate silence.
fn report_tier(model: &Qwen3Vl, asked: Precision) {
    let landed = model.linear_dtype();
    let landed_i8 = landed == Some(gpu_core::select::Dtype::I8);
    match (asked, landed_i8) {
        (Precision::I8, false) => eprintln!(
            "qwenvl: WARNING: int8 was requested but this device promoted the decoder back to fp32 \
             (its capabilities cannot serve a packed int8 dot) -- this run is exact, and slow"
        ),
        (Precision::I8, true) => eprintln!("qwenvl: decoder linears at INT8 (lossy tier, explicitly requested); vision tower fp32"),
        (Precision::F32, _) => {}
    }
}

fn preprocess_min_pixels() -> u32 {
    crate::preprocess::DEFAULT_MIN_PIXELS
}

/// Which of the two checkpoint formats `weights` names, and everything read
/// out of it that does not depend on the resident's capacity.
///
/// The choice is made from the files themselves, never from a flag: what a
/// user has is a path to a checkpoint, and asking them to also declare its
/// format is an opportunity to be wrong about something the filesystem
/// already knows. A GGUF is recognized by its own leading magic (and, for a
/// directory, by a GGUF being what is in it), then routed by the
/// `general.architecture` in its metadata.
enum Source {
    /// A HuggingFace directory: `config.json` + `model.safetensors[.index.json]`
    /// + `tokenizer.json`.
    HfDir(String),
    /// A llama.cpp checkpoint: the language half plus its `mmproj-*.gguf`
    /// vision tower.
    Gguf(crate::gguf_import::GgufFiles),
}

fn classify_source(weights: &str) -> Result<Source, String> {
    let p = std::path::Path::new(weights);
    let is_gguf = if p.is_dir() {
        std::fs::read_dir(p)
            .map(|it| it.filter_map(|e| e.ok()).any(|e| gguf::route::is_gguf(&e.path())))
            .unwrap_or(false)
    } else {
        gguf::route::is_gguf(p)
    };
    if is_gguf {
        return Ok(Source::Gguf(crate::gguf_import::GgufFiles::locate(p)?));
    }
    if p.is_dir() {
        return Ok(Source::HfDir(weights.to_string()));
    }
    Err(format!("qwenvl: {weights} is neither a checkpoint directory nor a GGUF file"))
}

fn load_resident(dir: &str, max_pixels: u32, precision: Precision) -> Result<Resident, String> {
    match classify_source(dir)? {
        Source::HfDir(d) => load_hf_resident(dir, &d, max_pixels, precision),
        Source::Gguf(files) => load_gguf_resident(dir, files, max_pixels, precision),
    }
}

/// Build from a two-file llama.cpp checkpoint. Both halves are named on the
/// way in, because a run that silently used a different projector than the
/// operator expected has no visible symptom.
fn load_gguf_resident(weights: &str, files: crate::gguf_import::GgufFiles, max_pixels: u32, precision: Precision) -> Result<Resident, String> {
    eprintln!("qwenvl: gguf checkpoint: model {}, vision projector {}", files.lm.display(), files.mmproj.display());
    let tok = crate::gguf_import::tokenizer(&files)?;
    let lm = checkpoint::gguf::MmapGguf::open(files.lm.to_str().ok_or("qwenvl: non-UTF8 lm path")?)?;
    let mmproj = checkpoint::gguf::MmapGguf::open(files.mmproj.to_str().ok_or("qwenvl: non-UTF8 mmproj path")?)?;
    let cfg = crate::gguf_import::config(&lm, &mmproj, &tok)?;
    drop(lm);
    drop(mmproj);
    let n_visual_capacity = visual_capacity(&cfg, max_pixels);
    let seq_len = resolved_ctx_len(&cfg);
    let w = crate::gguf_import::weights(&files, &cfg)?;
    let model = Qwen3Vl::from_imported(
        w,
        cfg.vision.clone(),
        cfg.text.clone(),
        seq_len,
        cfg.image_token_id,
        0,
        n_visual_capacity,
        cfg.mrope_section,
        precision.dtype(),
    );
    report_tier(&model, precision);
    Ok(Resident { weights: weights.to_string(), max_pixels, precision, n_visual_capacity, seq_len, cfg, model, tok })
}

/// The KV-cache capacity to actually build a resident's decoder with:
/// [`default_ctx_len`] clamped DOWN to this checkpoint's own declared
/// `max_position_embeddings` - "derive context from the checkpoint config"
/// means the ceiling is real (read off `config.json`/GGUF metadata, never a
/// number this crate made up), while the allocated default stays the
/// documented, VRAM-bounded budget an operator can raise. Never clamps UP:
/// a checkpoint declaring less than the default (e.g. a smaller variant)
/// must not have this resident silently over-allocate past what training
/// ever taught it to use.
fn resolved_ctx_len(cfg: &Qwen3VlConfig) -> u32 {
    default_ctx_len().min(cfg.text.max_position_embeddings.max(1))
}

/// The largest visual-token count ONE image can produce at `max_pixels`.
///
/// A square at the pixel budget is the largest-area, most token-hungry shape
/// smart_resize can produce for that budget (any other aspect ratio at the same
/// area yields <= tokens after patch-grid rounding), so it is the right
/// per-image upper bound to allocate for.
fn per_image_visual_capacity(cfg: &Qwen3VlConfig, max_pixels: u32) -> u32 {
    let factor = cfg.vision.patch_size * cfg.vision.spatial_merge_size;
    let side = (max_pixels as f64).sqrt() as u32;
    let (h_cap, w_cap) = smart_resize(side, side, factor, preprocess_min_pixels(), max_pixels);
    crate::preprocess::image_token_count(h_cap, w_cap, cfg.vision.patch_size, cfg.vision.spatial_merge_size)
}

/// Capacity placement: image_row0 is arbitrary (Qwen3Vl::generate's
/// incremental decode derives real placement from the token stream -- see this
/// module's own doc); this is the CAPACITY this resident's DeepStack/splice
/// buffers are sized for, not any one request's actual visual-token count.
///
/// Bounds the SUM of one request's images, not one image's own count:
/// [`MAX_IMAGES`] images at [`per_image_visual_capacity`] each. This is the
/// right answer, not merely a safe one -- `qwen3::Qwen::enable_deepstack`
/// allocates ONE flat `[n_rows, d_model]` buffer per level, and
/// `decode_steps`'s `deepstack_row` addresses a row within it that walks
/// every image in the request in sequence (`generate_timed`'s single running
/// `visual_row` counter, see that function's doc), never a per-image
/// sub-range. A capacity sized for only the largest single image would let a
/// second, even tiny, image's rows land past the buffer's end -- exactly the
/// out-of-bounds write `checkpoint::upload_at`'s own `assert!` this module's
/// doc already relies on to make an oversized SINGLE image loud would, for a
/// multi-image overflow, either panic on a request that "looks" well within
/// budget per image, or (worse, on a backend without that assert) silently
/// corrupt neighbouring DeepStack rows.
fn visual_capacity(cfg: &Qwen3VlConfig, max_pixels: u32) -> u32 {
    per_image_visual_capacity(cfg, max_pixels) * MAX_IMAGES as u32
}

fn load_hf_resident(weights: &str, dir: &str, max_pixels: u32, precision: Precision) -> Result<Resident, String> {
    let cfg_path = format!("{dir}/config.json");
    let cfg_text = std::fs::read_to_string(&cfg_path).map_err(|e| format!("qwenvl: cannot read {cfg_path}: {e}"))?;
    let cfg_json: serde_json::Value = serde_json::from_str(&cfg_text).map_err(|e| format!("qwenvl: cannot parse {cfg_path}: {e}"))?;
    let cfg = Qwen3VlConfig::from_hf(&cfg_json);
    let tok = data::qwen_tokenizer::QwenBpe::from_dir(dir).map_err(|e| format!("qwenvl: tokenizer: {e}"))?;

    let n_visual_capacity = visual_capacity(&cfg, max_pixels);
    let seq_len = resolved_ctx_len(&cfg);
    let model =
        Qwen3Vl::from_hf(dir, cfg.vision.clone(), cfg.text.clone(), seq_len, cfg.image_token_id, 0, n_visual_capacity, cfg.mrope_section, precision.dtype())?;
    report_tier(&model, precision);

    Ok(Resident { weights: weights.to_string(), max_pixels, precision, n_visual_capacity, seq_len, cfg, model, tok })
}

/// Bilinear-resample interleaved-HWC `[0,1]` pixels from `(w,h)` to
/// `(w_bar,h_bar)`, returning CHW (`pack_patches`'s expected layout). Unlike
/// `fastvlm::caps::pad_resize_chw`, no square padding -- Qwen3-VL's own
/// preprocessor resizes directly to the smart-resize target, aspect ratio
/// already accounted for by `smart_resize` itself. The resample is the shared
/// `imaging::host::resize_bilinear_hwc` (same half-pixel formula this
/// function used to re-derive locally), followed by the shared HWC->CHW
/// permutation.
fn hwc_to_chw_resized(hwc: &[f32], w: u32, h: u32, w_bar: u32, h_bar: u32) -> Vec<f32> {
    let resized = imaging::host::resize_bilinear_hwc(hwc, 3, w, h, w_bar, h_bar);
    imaging::pixels::hwc_to_chw(&resized, 3, h_bar as usize, w_bar as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::Blob;

    #[test]
    fn manifest_validates_without_weights() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let a = &m.actions[0];
        assert_eq!(a.name, "generate");
        assert!(a.streaming, "streaming is required for api_caps' chat-capable classification");
        assert!(a.params.iter().any(|p| p.name == "messages"));
        assert!(a.params.iter().any(|p| p.name == "prompt"));
        assert!(a.params.iter().any(|p| p.name == "fps"));
        // Neither is spec-required: GenerateAction::run enforces "exactly
        // one of image/video" itself, at RUN time, so a request naming
        // neither fails with a message pointing at the real cause instead
        // of the framework's generic "missing required input".
        assert!(a.inputs.iter().any(|b| b.name == "image" && b.media == Media::Image && !b.required));
        assert!(a.inputs.iter().any(|b| b.name == "video" && b.media == Media::Video && !b.required));
        assert!(a.outputs.iter().any(|b| b.name == "text" && b.media == Media::Text));
        // Sampling is a real, first-class request shape now, not just an
        // internal decode-loop capability - the served surface must declare
        // it the same way `qwen3::caps::manifest` does.
        let p = |name: &str| a.params.iter().find(|p| p.name == name).unwrap_or_else(|| panic!("missing param {name}"));
        assert_eq!(p("temp").default, Some(json!(0.0)), "default must stay greedy for backward compatibility");
        assert!(a.params.iter().any(|p| p.name == "top_k"));
        assert!(a.params.iter().any(|p| p.name == "top_p"));
        assert!(a.params.iter().any(|p| p.name == "seed"));
        // Numbered multi-image keys: 'image1'..'image{MAX_IMAGES-1}' present and
        // optional -- see this module's doc for the convention. 'image' itself is
        // also optional now (checked above), since a video-only request has none.
        for i in 1..MAX_IMAGES {
            let key = image_key(i);
            let b = a.inputs.iter().find(|b| b.name == key).unwrap_or_else(|| panic!("generate_spec missing '{key}'"));
            assert!(!b.required, "'{key}' must be optional (only 'image' is required)");
        }
    }

    /// [`resolved_ctx_len`] must never allocate a KV cache bigger than what
    /// the checkpoint itself declares as trained/valid - "derive context from
    /// the checkpoint config" is a claim about the CEILING, not about what
    /// gets allocated by default.
    #[test]
    fn resolved_ctx_len_never_exceeds_the_checkpoints_own_ceiling() {
        let mut cfg = Qwen3VlConfig::qwen3_vl_4b();
        cfg.text.max_position_embeddings = 2048; // smaller than the 24576 default
        assert_eq!(resolved_ctx_len(&cfg), 2048, "must clamp DOWN to the checkpoint's declared max, never allocate past it");

        cfg.text.max_position_embeddings = 262144; // the real released config's declared ceiling
        assert_eq!(resolved_ctx_len(&cfg), default_ctx_len(), "a checkpoint with real headroom gets the operator's configured default, not a fixed 4096");
    }

    /// [`decode_images`] reads contiguous from `image`: `image`+`image1` present
    /// with `image2` ABSENT is 2 images, and a later `image3` (present but past
    /// the gap) is never read -- this is the "contiguous from image1" rule this
    /// module's doc promises, not "any numbered key present".
    #[test]
    fn decode_images_stops_at_the_first_gap() {
        let px = |v: f32| {
            let hwc = [v, v, v];
            let bytes: Vec<u8> = hwc.iter().flat_map(|f| f.to_le_bytes()).collect();
            Blob::new(Media::Image, bytes).with_meta(json!({"w": 1, "h": 1}))
        };
        let inv = Invocation::new().blob("image", px(0.0)).blob("image1", px(1.0)).blob("image3", px(3.0));
        let imgs = decode_images(&inv).expect("decode_images");
        assert_eq!(imgs.len(), 2, "must stop at the gap ('image2' absent), never skip to 'image3'");
        assert_eq!(imgs[0].0, vec![0.0, 0.0, 0.0]);
        assert_eq!(imgs[1].0, vec![1.0, 1.0, 1.0]);
    }

    /// A request with only `image` set (every caller before multi-image
    /// support existed) must decode to exactly one image -- backward
    /// compatibility for [`decode_images`] itself, independent of the
    /// `Prepared::build`-level regression test in `crate::model`.
    #[test]
    fn decode_images_single_image_backward_compatible() {
        let hwc = [0.25f32, 0.5, 0.75];
        let bytes: Vec<u8> = hwc.iter().flat_map(|f| f.to_le_bytes()).collect();
        let inv = Invocation::new().blob("image", Blob::new(Media::Image, bytes).with_meta(json!({"w": 1, "h": 1})));
        let imgs = decode_images(&inv).expect("decode_images");
        assert_eq!(imgs.len(), 1);
        assert_eq!(imgs[0].0, vec![0.25, 0.5, 0.75]);
    }

    /// The run-time "exactly one of image/video" rule, checked without
    /// touching a checkpoint (both branches fail before `weights` is even
    /// read) - this is the ONE place that rule lives, so a request with
    /// neither or both gets a message naming the real cause.
    #[test]
    fn generate_requires_exactly_one_of_image_or_video() {
        let base = || Invocation::new().set("prompt", json!("describe this"));
        let neither = base();
        let err = GenerateAction.run(&neither, &mut |_| {}).unwrap_err();
        assert!(err.contains("exactly one of 'image'/'video'"), "{err}");

        let both = base()
            .blob("image", Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w": 1, "h": 1})))
            .blob("video", Blob::new(Media::Video, vec![0u8; 12]).with_meta(json!({"frames": 1, "w": 1, "h": 1, "c": 3})));
        let err = GenerateAction.run(&both, &mut |_| {}).unwrap_err();
        assert!(err.contains("exactly one of 'image'/'video'"), "{err}");
    }

    /// A video request with no `fps` is refused BY NAME before any checkpoint
    /// is touched - real-timestamp M-RoPE has no meaning without a real
    /// frame rate, so this must never silently default to "1 fps" or "just
    /// count frames" (the exact bug this whole change fixes).
    #[test]
    fn generate_video_without_fps_is_a_clean_error_before_touching_weights() {
        let inv = Invocation::new()
            .set("prompt", json!("describe this clip"))
            .blob("video", Blob::new(Media::Video, vec![0u8; 24]).with_meta(json!({"frames": 2, "w": 1, "h": 1, "c": 3})));
        let err = GenerateAction.run(&inv, &mut |_| {}).unwrap_err();
        assert!(err.contains("'fps'"), "{err}");
    }

    /// Too many frames is refused BY NAME before any checkpoint is touched
    /// (`MAX_VIDEO_FRAMES` - see this crate's honestly-scoped bound).
    #[test]
    fn generate_video_over_the_frame_cap_is_a_clean_error() {
        let n = (MAX_VIDEO_FRAMES + 1) as usize;
        let bytes = vec![0u8; n * 3 * 4]; // n frames * (w=1 * h=1 * c=3) pixels * 4 bytes/f32
        let inv = Invocation::new()
            .set("prompt", json!("describe this clip"))
            .set("fps", json!(30.0))
            .blob("video", Blob::new(Media::Video, bytes).with_meta(json!({"frames": n, "w": 1, "h": 1, "c": 3})));
        let err = GenerateAction.run(&inv, &mut |_| {}).unwrap_err();
        assert!(err.contains(&format!("{n} frames")), "{err}");
        assert!(err.contains("at most"), "{err}");
    }

    /// `tools`/`tool_choice` match `qwen3::caps::generate_spec()`'s own
    /// params byte-for-byte (name + help text) so a client driving both
    /// models' `generate` never has to special-case VLM tool-calling vs
    /// text-only tool-calling.
    #[test]
    fn manifest_declares_tools_and_tool_choice_matching_qwen3() {
        let qwenvl = &manifest().actions[0];
        let qwen3 = &qwen3::caps::manifest().actions[0];
        for name in ["tools", "tool_choice"] {
            let a = qwenvl.params.iter().find(|p| p.name == name).unwrap_or_else(|| panic!("qwen3vl generate missing param {name:?}"));
            let b = qwen3.params.iter().find(|p| p.name == name).unwrap_or_else(|| panic!("qwen3 generate missing param {name:?}"));
            assert_eq!(a.help, b.help, "param {name:?} help text must match qwen3's");
            assert!(!a.required, "param {name:?} must be optional");
        }
    }

    /// The tools preamble this action renders (`Prepared::build`'s
    /// `qwen_chat::render(&[], tools, TemplateOpts { add_generation_prompt:
    /// false, .. })` call, with `reasoning_effort: None` since this action has
    /// no `enable_thinking`/`reasoning_effort` param) is NOT byte-for-byte what
    /// `qwen3::caps generate` renders under ITS OWN default
    /// (`enable_thinking: true` resolves `reasoning_effort` to `Some("xhigh")`,
    /// per `qwen3::chat::parse_request`) - the two differ by exactly one
    /// injected "reasoning effort" directive paragraph. This test pins that
    /// real, narrower relationship rather than leaving the comment's claim
    /// unverified.
    #[test]
    fn tools_preamble_matches_qwen3_except_the_reasoning_effort_directive_it_cannot_opt_into() {
        let tools = vec![r#"{"type":"function","function":{"name":"get_weather","parameters":{}}}"#.to_string()];
        let this_crate_render = qwen_chat::render(&[], &tools, TemplateOpts { add_generation_prompt: false, ..Default::default() }).unwrap();
        let qwen3_default_render =
            qwen_chat::render(&[], &tools, TemplateOpts { add_generation_prompt: false, reasoning_effort: Some("xhigh".into()), ..Default::default() })
                .unwrap();
        assert_ne!(this_crate_render, qwen3_default_render, "the two must actually differ, or the documented gap is imaginary");
        assert!(!this_crate_render.to_lowercase().contains("reasoning effort"), "this crate's preamble has no enable_thinking param, so it must omit the directive");
        assert!(qwen3_default_render.to_lowercase().contains("reasoning effort"), "qwen3's own thinking-enabled default must inject the directive");
    }

    /// Server-side named-function validation runs BEFORE any weights are
    /// touched - the same precedence `qwen3::chat::parse_request` gives it
    /// (a typo'd name would otherwise degrade into a guaranteed-unmet
    /// post-hoc demand). Proven with a nonexistent weights path: if this
    /// validation ran after resident load, the error would instead name the
    /// missing checkpoint.
    #[test]
    fn tool_choice_named_function_must_exist_in_tools_before_touching_weights() {
        let inv = Invocation::new()
            .set("weights", json!("/nonexistent/qwenvl"))
            .set("prompt", json!("weather?"))
            .set("tools", json!(r#"[{"type":"function","function":{"name":"get_weather"}}]"#))
            .set("tool_choice", json!(r#"{"type":"function","function":{"name":"no_such_tool"}}"#))
            .blob("image", Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w": 1, "h": 1})));
        let err = GenerateAction.run(&inv, &mut |_| {}).unwrap_err();
        assert!(err.contains("no_such_tool"), "got: {err}");
        assert!(!err.contains("nonexistent"), "must fail on tool_choice before ever touching weights: {err}");
    }

    /// Served-path smoke on the real checkpoint (skip-if-absent, like the
    /// fastvlm caption parity tests): exercises the FULL `GenerateAction`
    /// path - smart-resize preprocessing, image splice, and
    /// `Qwen3Vl::generate`'s M-RoPE/DeepStack KV-cache incremental decode -
    /// which `parity.rs` (batched 4-layer decoder only) never touches.
    /// Bar is "runs end-to-end and produces text", not token parity (no HF
    /// golden for the incremental path yet - see VALIDATION.md).
    #[test]
    fn served_generate_path_runs_on_real_weights() {
        let dir = default_weights();
        if dir.is_empty() || !std::path::Path::new(&dir).join("config.json").exists() {
            brain_testutil::skip("BRAIN_QWEN3VL_WEIGHTS not set / checkpoint absent");
            return;
        }
        let (w, h) = (64u32, 64u32);
        let hwc: Vec<f32> = (0..w * h * 3).map(|i| (i % 251) as f32 / 250.0).collect();
        let bytes: Vec<u8> = hwc.iter().flat_map(|v| v.to_le_bytes()).collect();
        let inv = Invocation::new()
            .set("prompt", json!("Describe this image."))
            .set("max_new", json!(4))
            .blob("image", Blob::new(Media::Image, bytes).with_meta(json!({"w": w, "h": h})));
        let out = GenerateAction.run(&inv, &mut |_| {}).expect("served generate path failed on real weights");
        assert!(out.blobs.contains_key("text"), "generate must emit its declared text blob");
    }

    /// The tools/tool_choice CONTRACT on the real checkpoint, mirroring
    /// `qwen3::caps`'s own `tokenizer_present_runs_the_shared_chat_parse_with_tools`
    /// shape: a `messages`+`tools` request must go through the SAME
    /// `qwen3::chat::{parse_tools, parse_tool_choice, SeqState}` machinery the
    /// text-only path runs, and resolve a real `finish_reason` (`tool_calls`,
    /// `tool_choice_unmet` or `length`/`stop` - which one is not asserted,
    /// since a real model's actual output on an arbitrary image is not this
    /// test's business; the CONTRACT is that `tool_choice: "required"` is
    /// enforced post-generation at all).
    #[test]
    fn served_generate_with_tools_enforces_tool_choice_on_real_weights() {
        let dir = default_weights();
        if dir.is_empty() || !std::path::Path::new(&dir).join("config.json").exists() {
            brain_testutil::skip("BRAIN_QWEN3VL_WEIGHTS not set / checkpoint absent");
            return;
        }
        let (w, h) = (64u32, 64u32);
        let hwc: Vec<f32> = (0..w * h * 3).map(|i| (i % 251) as f32 / 250.0).collect();
        let bytes: Vec<u8> = hwc.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tools = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]).to_string();
        let inv = Invocation::new()
            .set("prompt", json!("Describe this image."))
            .set("max_new", json!(4))
            .set("tools", json!(tools))
            .set("tool_choice", json!("required"))
            .blob("image", Blob::new(Media::Image, bytes.clone()).with_meta(json!({"w": w, "h": h})));
        let mut events = 0u32;
        let out = GenerateAction.run(&inv, &mut |_p| events += 1).expect("served generate path failed on real weights");
        assert!(out.outputs.get("finish_reason").is_some(), "shared SeqState::finish must report a finish_reason");
        assert!(out.outputs.get("prompt_tokens").is_some());
        assert!(out.outputs.get("completion_tokens").is_some());
        assert!(events > 0, "must stream at least the final 'done' Progress");

        // `none` withholds the tool schemas entirely - a request otherwise
        // identical must still succeed (the model simply cannot call tools).
        let inv_none = Invocation::new()
            .set("prompt", json!("Describe this image."))
            .set("max_new", json!(4))
            .set("tools", json!(tools))
            .set("tool_choice", json!("none"))
            .blob("image", Blob::new(Media::Image, bytes.clone()).with_meta(json!({"w": w, "h": h})));
        let out_none = GenerateAction.run(&inv_none, &mut |_| {}).expect("served generate path failed on real weights (tool_choice none)");
        assert_ne!(out_none.outputs.get("finish_reason"), Some(&json!("tool_choice_unmet")), "none must never demand a tool call");
    }

    #[test]
    fn missing_weights_is_a_clean_error() {
        let inv = Invocation::new()
            .set("weights", json!("/nonexistent/qwenvl"))
            .set("prompt", json!("describe this"))
            .blob("image", Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w": 1, "h": 1})));
        let r = GenerateAction.run(&inv, &mut |_| {});
        let err = r.err().unwrap_or_default();
        // The spec is that a weights path that is neither of the two supported
        // checkpoint shapes is refused BY NAME, before any tensor is touched.
        // It used to be phrased as "cannot read <dir>/config.json", which only
        // held while a directory of safetensors was the single thing this
        // could load.
        assert!(err.contains("/nonexistent/qwenvl"), "the error must name the path: {err}");
        assert!(err.contains("neither a checkpoint directory nor a GGUF"), "{err}");
    }

    #[test]
    fn empty_prompt_is_a_clean_error_before_touching_weights() {
        let inv = Invocation::new().blob("image", Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w": 1, "h": 1})));
        let r = GenerateAction.run(&inv, &mut |_| {});
        let err = r.err().unwrap_or_default();
        assert!(err.contains("empty prompt"), "{err}");
    }

    /// Bilinear resize keeps content position and value roughly where
    /// expected -- a wrong axis swap or off-by-one shows up as a shifted or
    /// mirrored quadrant, the same check `fastvlm::caps`'s
    /// `pad_resize_centres_and_fills` runs for its own resize helper.
    #[test]
    fn resize_keeps_content_orientation() {
        // 4x2: left half black, right half white.
        let mut hwc = Vec::new();
        for _row in 0..2 {
            for col in 0..4 {
                let v = if col < 2 { 0.0f32 } else { 1.0 };
                hwc.extend([v, v, v]);
            }
        }
        let out = hwc_to_chw_resized(&hwc, 4, 2, 8, 4);
        let px = |x: usize, y: usize| out[y * 8 + x]; // channel 0
        assert!(px(0, 0) < 0.3, "left stays dark");
        assert!(px(7, 0) > 0.7, "right stays light");
        assert!(px(0, 3) < 0.3 && px(7, 3) > 0.7, "orientation holds across rows");
    }
}
