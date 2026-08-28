// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `capability::Provider` for Qwen3-VL: image + text in, greedy text out.
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

/// Default checkpoint directory - `$BRAIN_QWEN3VL_WEIGHTS`, never a baked-in
/// absolute path (AGENTS.md: no absolute paths in source).
fn default_weights() -> String {
    std::env::var("BRAIN_QWEN3VL_WEIGHTS").unwrap_or_default()
}

/// Pixel-area budget for the resident model's DeepStack/splice buffer
/// CAPACITY (see this module's doc) - a practical default (roughly a
/// 1024x1024 image, ~1024 visual tokens at the 4B config's patch/merge
/// granularity), not `preprocess::DEFAULT_MAX_PIXELS`'s own real-checkpoint
/// ceiling (3584² -- see that constant's doc), which would still allocate
/// multiple GB of DeepStack scratch per level for a capacity most requests
/// never approach. Override via the `max_pixels` param for a
/// checkpoint/workload that genuinely needs bigger images.
const DEFAULT_SERVE_MAX_PIXELS: u32 = 1024 * 1024;
/// Decoder context: enough for a real prompt + the max image + a real
/// response. Matches `fastvlm::caps`'s `t_max` sizing philosophy (a fixed,
/// documented budget, not derived from the checkpoint).
const SEQ_LEN: u32 = 4096;

pub fn generate_spec() -> ActionSpec {
    ActionSpec::new("generate", "Qwen3-VL: image + text in, greedy text completion (validation-tier -- see this module's doc)")
        .streaming()
        .param(ParamSpec::new("messages", ParamType::Str, "flattened chat messages (JSON array string)"))
        .param(ParamSpec::new("prompt", ParamType::Str, "a raw prompt (alternative to messages)"))
        .param(ParamSpec::new("max_new", ParamType::Int, "max tokens to generate").default(json!(64)))
        .param(
            ParamSpec::new("weights", ParamType::Str, "Qwen3-VL checkpoint DIRECTORY (config.json + model.safetensors[.index.json] + tokenizer.json)")
                .default(json!(default_weights())),
        )
        .param(
            ParamSpec::new("max_pixels", ParamType::Int, "resident capacity: max input image area in pixels (larger requests error, never silently truncate)")
                .default(json!(DEFAULT_SERVE_MAX_PIXELS)),
        )
        .input(BlobSpec::new("image", Media::Image, "raw HWC f32 pixels in [0,1], meta {w,h} (capability::blob's wire convention)").required())
        .output(BlobSpec::new("text", Media::Text, "the generated continuation"))
}

pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "Qwen3-VL -- image + text in, greedy text out. Validation-tier: fp32 weights, greedy \
         argmax only (no temperature/top-k/top-p), one request at a time (no batching).",
        vec![generate_spec()],
    )
}

use capability::last_user_text;

struct Resident {
    weights: String,
    max_pixels: u32,
    /// How many visual tokens this resident's DeepStack/splice buffers were
    /// allocated for (computed once at construction from `max_pixels`) - see
    /// this module's own doc on why construction-time capacity, not one
    /// request's exact size.
    n_visual_capacity: u32,
    cfg: Qwen3VlConfig,
    model: Qwen3Vl,
    tok: data::qwen_tokenizer::QwenBpe,
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
        let prompt = last_user_text(inv);
        if prompt.trim().is_empty() {
            return Err("qwenvl generate: empty prompt (need 'messages' with a user turn, or 'prompt')".to_string());
        }
        let dir = inv.get_str("weights").filter(|s| !s.is_empty()).unwrap_or_else(default_weights);
        if dir.is_empty() {
            return Err("qwenvl generate: no checkpoint directory (set 'weights' or $BRAIN_QWEN3VL_WEIGHTS)".to_string());
        }
        let max_new = inv.get_i64("max_new").unwrap_or(64).clamp(1, 2048) as u32;
        let max_pixels = inv.get_i64("max_pixels").unwrap_or(DEFAULT_SERVE_MAX_PIXELS as i64).max(1) as u32;
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;

        let mut guard = RESIDENT.lock().map_err(|_| "qwenvl: resident lock poisoned")?;
        if !matches!(&*guard, Some(r) if r.weights == dir && r.max_pixels == max_pixels) {
            *guard = None;
            *guard = Some(load_resident(&dir, max_pixels)?);
        }
        let hot = guard.as_ref().unwrap();

        // Smart-resize to a patch-aligned target, bilinear resample, pack
        // patches -- the same three real preprocessing steps the checkpoint's
        // HF processor runs, no "caller must pre-align" shortcut.
        let factor = hot.cfg.vision.patch_size * hot.cfg.vision.spatial_merge_size;
        let (h_bar, w_bar) = smart_resize(h, w, factor, preprocess_min_pixels(), hot.max_pixels);
        let n_visual = crate::preprocess::image_token_count(h_bar, w_bar, hot.cfg.vision.patch_size, hot.cfg.vision.spatial_merge_size);
        if n_visual > hot.n_visual_capacity {
            return Err(format!(
                "qwenvl generate: image needs {n_visual} visual tokens, exceeding this resident's capacity {} \
                 (raise 'max_pixels' -- current cap {} px)",
                hot.n_visual_capacity, hot.max_pixels
            ));
        }
        let chw = hwc_to_chw_resized(&hwc, w, h, w_bar, h_bar);
        let mut chw = chw;
        normalize_unit(&mut chw);
        let pixels = pack_patches(&chw, hot.cfg.vision.in_channels, h_bar, w_bar, hot.cfg.vision.patch_size, hot.cfg.vision.spatial_merge_size, hot.cfg.vision.temporal_patch_size);

        // Prompt: <|im_start|>user\n <|vision_start|> [IMG]*n_visual <|vision_end|> {prompt}<|im_end|>\n<|im_start|>assistant\n
        let mut tokens = hot.tok.encode("<|im_start|>user\n");
        tokens.push(hot.cfg.vision_start_token_id);
        tokens.extend(std::iter::repeat_n(hot.cfg.image_token_id, n_visual as usize));
        tokens.push(hot.cfg.vision_end_token_id);
        tokens.extend(hot.tok.encode(&format!("{prompt}<|im_end|>\n<|im_start|>assistant\n")));
        let eos = hot.tok.encode("<|im_end|>");

        if tokens.len() as u32 + max_new > SEQ_LEN {
            return Err(format!(
                "qwenvl generate: prompt ({} tokens incl. {n_visual} image tokens) + max_new ({max_new}) \
                 exceeds this resident's context {SEQ_LEN}",
                tokens.len()
            ));
        }

        progress(Progress::step(0, max_new, "generating"));
        let (gh, gw) = patch_grid(h_bar, w_bar, hot.cfg.vision.patch_size);
        // Real per-token streaming deltas (the spec declares `.streaming()`):
        // re-decode the running id list each token and emit the UTF-8-safe
        // suffix, exactly like qwen3::chat's streaming path.
        let mut ids: Vec<u32> = Vec::new();
        let mut printed = String::new();
        let mut step = 0u32;
        let out_ids = hot.model.generate_cb(&tokens, (gh, gw), &pixels, max_new, &eos, |tok_id| {
            ids.push(tok_id);
            step += 1;
            let full = hot.tok.decode(&ids);
            let (delta, np) = qwen3::chat::stream_delta(&printed, &full);
            printed = np;
            if !delta.is_empty() {
                progress(Progress::token(step, max_new, delta));
            }
        });
        let text = hot.tok.decode(&out_ids);
        progress(Progress::step(max_new, max_new, text.clone()));
        Ok(Outcome::new()
            .set("text", json!(text.clone()))
            .set("tokens", json!(out_ids.len()))
            .blob("text", Blob::new(Media::Text, text.into_bytes())))
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

fn load_resident(dir: &str, max_pixels: u32) -> Result<Resident, String> {
    match classify_source(dir)? {
        Source::HfDir(d) => load_hf_resident(dir, &d, max_pixels),
        Source::Gguf(files) => load_gguf_resident(dir, files, max_pixels),
    }
}

/// Build from a two-file llama.cpp checkpoint. Both halves are named on the
/// way in, because a run that silently used a different projector than the
/// operator expected has no visible symptom.
fn load_gguf_resident(weights: &str, files: crate::gguf_import::GgufFiles, max_pixels: u32) -> Result<Resident, String> {
    eprintln!("qwenvl: gguf checkpoint: model {}, vision projector {}", files.lm.display(), files.mmproj.display());
    let tok = crate::gguf_import::tokenizer(&files)?;
    let lm = checkpoint::gguf::MmapGguf::open(files.lm.to_str().ok_or("qwenvl: non-UTF8 lm path")?)?;
    let mmproj = checkpoint::gguf::MmapGguf::open(files.mmproj.to_str().ok_or("qwenvl: non-UTF8 mmproj path")?)?;
    let cfg = crate::gguf_import::config(&lm, &mmproj, &tok)?;
    drop(lm);
    drop(mmproj);
    let n_visual_capacity = visual_capacity(&cfg, max_pixels);
    let w = crate::gguf_import::weights(&files, &cfg)?;
    let model = Qwen3Vl::from_imported(
        w,
        cfg.vision.clone(),
        cfg.text.clone(),
        SEQ_LEN,
        cfg.image_token_id,
        0,
        n_visual_capacity,
        cfg.mrope_section,
    );
    Ok(Resident { weights: weights.to_string(), max_pixels, n_visual_capacity, cfg, model, tok })
}

/// Capacity placement: image_row0 is arbitrary (Qwen3Vl::generate's
/// incremental decode derives real placement from the token stream -- see this
/// module's own doc); this is the CAPACITY this resident's DeepStack/splice
/// buffers are sized for, not any one request's actual visual-token count.
///
/// A square at the pixel budget is the largest-area, most token-hungry shape
/// smart_resize can produce for that budget (any other aspect ratio at the same
/// area yields <= tokens after patch-grid rounding), so it is the right
/// capacity upper bound to allocate for.
fn visual_capacity(cfg: &Qwen3VlConfig, max_pixels: u32) -> u32 {
    let factor = cfg.vision.patch_size * cfg.vision.spatial_merge_size;
    let side = (max_pixels as f64).sqrt() as u32;
    let (h_cap, w_cap) = smart_resize(side, side, factor, preprocess_min_pixels(), max_pixels);
    crate::preprocess::image_token_count(h_cap, w_cap, cfg.vision.patch_size, cfg.vision.spatial_merge_size)
}

fn load_hf_resident(weights: &str, dir: &str, max_pixels: u32) -> Result<Resident, String> {
    let cfg_path = format!("{dir}/config.json");
    let cfg_text = std::fs::read_to_string(&cfg_path).map_err(|e| format!("qwenvl: cannot read {cfg_path}: {e}"))?;
    let cfg_json: serde_json::Value = serde_json::from_str(&cfg_text).map_err(|e| format!("qwenvl: cannot parse {cfg_path}: {e}"))?;
    let cfg = Qwen3VlConfig::from_hf(&cfg_json);
    let tok = data::qwen_tokenizer::QwenBpe::from_dir(dir).map_err(|e| format!("qwenvl: tokenizer: {e}"))?;

    let n_visual_capacity = visual_capacity(&cfg, max_pixels);
    let model = Qwen3Vl::from_hf(dir, cfg.vision.clone(), cfg.text.clone(), SEQ_LEN, cfg.image_token_id, 0, n_visual_capacity, cfg.mrope_section)?;

    Ok(Resident { weights: weights.to_string(), max_pixels, n_visual_capacity, cfg, model, tok })
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
