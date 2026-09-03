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

use std::sync::Mutex;

use capability::{
    Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider,
};
use data::tokenizer::Tokenizer;
use serde_json::json;

use crate::config::Qwen3VlConfig;
use crate::model::Qwen3Vl;
use crate::preprocess::{normalize_unit, pack_patches, patch_grid, smart_resize};

pub const MODEL: &str = "brain/qwen3vl";

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
        .input(BlobSpec::new("image", Media::Image, "raw HWC f32 pixels in [0,1], meta {w,h} (capability::blob's wire convention)").required());
    for i in 1..MAX_IMAGES {
        spec = spec.input(BlobSpec::new(
            &image_key(i),
            Media::Image,
            "optional additional image (same wire shape as 'image'); present contiguous from 'image', i.e. 'image2' is only read if 'image1' is also set",
        ));
    }
    spec.output(BlobSpec::new("text", Media::Text, "the generated continuation"))
}

pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "Qwen3-VL -- image + text in, text out. Validation-tier: fp32 weights by default \
         (int8 decoder opt-in), temperature/top-k/top-p sampling (greedy by default), \
         one request at a time (no batching), context capacity set by $BRAIN_QWEN3VL_CTX \
         (clamped to the checkpoint's own declared max_position_embeddings).",
        vec![generate_spec()],
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
    /// image + prompt in, streamed text out. The body [`GenerateAction::run`]
    /// used to hold directly, extracted so a residency-scheduled instance and
    /// the direct provider execute byte-for-byte the same code.
    pub fn generate(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let prompt = last_user_text(inv);
        if prompt.trim().is_empty() {
            return Err("qwenvl generate: empty prompt (need 'messages' with a user turn, or 'prompt')".to_string());
        }
        let max_new = inv.get_i64("max_new").unwrap_or(64).clamp(1, 2048) as u32;
        let images = decode_images(inv)?;
        let p = Prepared::build_multi(self, &images, &prompt, max_new)?;
        progress(Progress::step(0, max_new, "generating"));

        let temperature = inv.get_f64("temp").unwrap_or(0.0).max(0.0) as f32;
        let top_k = inv.get_i64("top_k").unwrap_or(40).max(0) as usize;
        let top_p = inv.get_f64("top_p").unwrap_or(1.0) as f32;
        let seed = inv.get_i64("seed").unwrap_or(0).max(0) as u64;
        let sample = crate::model::SampleParams { temperature, top_k, top_p };
        let mut rng = data::rng::Rng::new(seed);

        // Real per-token streaming deltas (the spec declares `.streaming()`):
        // re-decode the running id list each token and emit the UTF-8-safe
        // suffix, exactly like qwen3::chat's streaming path.
        let mut ids: Vec<u32> = Vec::new();
        let mut printed = String::new();
        let mut step = 0u32;
        let out_ids = self.model.generate_cb(&p.tokens, &p.image_inputs(), max_new, &p.eos, sample, &mut rng, |tok_id| {
            ids.push(tok_id);
            step += 1;
            let full = self.tok.decode(&ids);
            let (delta, np) = qwen3::chat::stream_delta(&printed, &full);
            printed = np;
            if !delta.is_empty() {
                progress(Progress::token(step, max_new, delta));
            }
        });
        let text = self.tok.decode(&out_ids);
        let ntok = out_ids.len();
        progress(Progress::step(max_new, max_new, text.clone()));
        Ok(Outcome::new()
            .set("text", json!(text.clone()))
            .set("tokens", json!(ntok))
            .blob("text", Blob::new(Media::Text, text.into_bytes())))
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
        (name == "generate").then(|| std::sync::Arc::new(GenerateAction) as std::sync::Arc<dyn Action>)
    }
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
        let dir = inv.get_str("weights").filter(|s| !s.is_empty()).unwrap_or_else(default_weights);
        if dir.is_empty() {
            return Err("qwenvl generate: no checkpoint directory (set 'weights' or $BRAIN_QWEN3VL_WEIGHTS)".to_string());
        }
        let max_pixels = inv.get_i64("max_pixels").unwrap_or(DEFAULT_SERVE_MAX_PIXELS as i64).max(1) as u32;
        let precision = Precision::from_name(inv.get_str("precision").unwrap_or_default().as_str())?;
        with_resident(&dir, max_pixels, precision, |hot| hot.generate(inv, progress))
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
    fn build(hot: &Resident, hwc: &[f32], w: u32, h: u32, prompt: &str, max_new: u32) -> Result<Prepared, String> {
        Self::build_multi(hot, std::slice::from_ref(&(hwc.to_vec(), w, h)), prompt, max_new)
    }

    /// Assemble ONE prompt from N images (in request order): each image gets
    /// its own smart-resize + patch/token count, computed independently (a
    /// wide image and a tall one in the same request are each resized to
    /// their own patch-aligned target, not forced to share one), and its own
    /// vision-start/`[IMG]*`/vision-end run, back-to-back with no text
    /// between runs, ahead of the user's prompt text - see this module's doc
    /// for the request shape this backs.
    fn build_multi(hot: &Resident, images: &[(Vec<f32>, u32, u32)], prompt: &str, max_new: u32) -> Result<Prepared, String> {
        let t0 = std::time::Instant::now();
        if images.is_empty() {
            return Err("qwenvl generate: at least one image is required".to_string());
        }
        let factor = hot.cfg.vision.patch_size * hot.cfg.vision.spatial_merge_size;

        let mut tokens = hot.tok.encode("<|im_start|>user\n");
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
        let p = Prepared::build(hot, hwc, w, h, prompt, max_new)?;
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

    #[test]
    fn manifest_validates_without_weights() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let a = &m.actions[0];
        assert_eq!(a.name, "generate");
        assert!(a.streaming, "streaming is required for api_caps' chat-capable classification");
        assert!(a.params.iter().any(|p| p.name == "messages"));
        assert!(a.params.iter().any(|p| p.name == "prompt"));
        assert!(a.inputs.iter().any(|b| b.name == "image" && b.required));
        assert!(a.outputs.iter().any(|b| b.name == "text" && b.media == Media::Text));
        // Sampling is a real, first-class request shape now, not just an
        // internal decode-loop capability - the served surface must declare
        // it the same way `qwen3::caps::manifest` does.
        let p = |name: &str| a.params.iter().find(|p| p.name == name).unwrap_or_else(|| panic!("missing param {name}"));
        assert_eq!(p("temp").default, Some(json!(0.0)), "default must stay greedy for backward compatibility");
        assert!(a.params.iter().any(|p| p.name == "top_k"));
        assert!(a.params.iter().any(|p| p.name == "top_p"));
        assert!(a.params.iter().any(|p| p.name == "seed"));
        // Numbered multi-image keys: 'image' required, 'image1'..'image{MAX_IMAGES-1}'
        // present and optional -- see this module's doc for the convention.
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
