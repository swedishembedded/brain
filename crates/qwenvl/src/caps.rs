// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `capability::Provider` for Qwen3-VL: image + text in, greedy text out.
//!
//! One action, `generate`, in the SAME chat-capable shape
//! `crates/omni/src/caps.rs::generate_spec()` uses (`messages`/`prompt`,
//! `.streaming()`, `Media::Text` output) — required, not by convention:
//! `apiserve::catalog::api_caps` classifies a model chat-capable only on that
//! exact shape (see `omni::caps`'s own doc for the full reasoning), and both
//! HTTP handlers always populate `messages`, never a bare `prompt`.
//!
//! Real, working, but validation-tier — the honest scope of what serving
//! wiring for `Qwen3Vl::generate()` turned out to need:
//!
//! - **Image placement is per-request, not baked into the resident model.**
//!   `Qwen3Vl::generate()`'s incremental KV-cache decode derives image
//!   placement dynamically from the token stream (`tok ==
//!   self.image_token_id`), NOT from the `image_row0`/`n_visual` this crate's
//!   `Qwen3Vl::new` takes at construction — those only gate the BATCHED
//!   `forward()` (training) graph, which `generate()` never calls. The
//!   resident model is therefore built ONCE with a generous CAPACITY
//!   (`MAX_VISUAL_TOKENS`, wired to `Qwen3Vl::new`'s `n_visual`), and each
//!   request's actual (smaller-or-equal) image writes only the front of that
//!   capacity — `checkpoint::upload_at`'s own `assert!(offset + len <=
//!   buf.size)` makes an oversized request a loud, immediate error, never a
//!   silent overflow.
//! - **Preprocessing does the real "smart resize"**, not the "caller
//!   supplies already-patch-aligned dimensions" minimum this crate's own
//!   follow-up doc once proposed: `preprocess::smart_resize_default`
//!   computes the patch-aligned target size for ANY input resolution, and
//!   this module bilinear-resizes to it (`resize_bilinear_chw`, the same
//!   shape as `fastvlm::caps::pad_resize_chw` but without the square-pad —
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

pub const MODEL: &str = "brain/qwenvl";

/// Default checkpoint directory — `$BRAIN_QWENVL_WEIGHTS`, never a baked-in
/// absolute path (AGENTS.md: no absolute paths in source).
fn default_weights() -> String {
    std::env::var("BRAIN_QWENVL_WEIGHTS").unwrap_or_default()
}

/// Pixel-area budget for the resident model's DeepStack/splice buffer
/// CAPACITY (see this module's doc) — a practical default (roughly a
/// 1024x1024 image, ~1024 visual tokens at the 4B config's patch/merge
/// granularity), not `preprocess::DEFAULT_MAX_PIXELS`'s theoretical 4096²
/// ceiling, which would allocate multiple GB of DeepStack scratch per level
/// for a capacity most requests never approach. Override via the `max_pixels`
/// param for a checkpoint/workload that genuinely needs bigger images.
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

/// Same extraction `omni::caps::last_user_text`/`resident_mock::last_user_text`
/// use — kept in sync deliberately, all three exist because OpenAI/Anthropic
/// always send `messages`, never a bare `prompt`.
fn last_user_text(inv: &Invocation) -> String {
    if let Some(s) = inv.get_str("messages") {
        if let Ok(serde_json::Value::Array(arr)) = serde_json::from_str::<serde_json::Value>(&s) {
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

struct Resident {
    weights: String,
    max_pixels: u32,
    /// How many visual tokens this resident's DeepStack/splice buffers were
    /// allocated for (computed once at construction from `max_pixels`) — see
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
            return Err("qwenvl generate: no checkpoint directory (set 'weights' or $BRAIN_QWENVL_WEIGHTS)".to_string());
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
        let out_ids = hot.model.generate(&tokens, (gh, gw), &pixels, max_new, &eos);
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

fn load_resident(dir: &str, max_pixels: u32) -> Result<Resident, String> {
    let cfg_path = format!("{dir}/config.json");
    let cfg_text = std::fs::read_to_string(&cfg_path).map_err(|e| format!("qwenvl: cannot read {cfg_path}: {e}"))?;
    let cfg_json: serde_json::Value = serde_json::from_str(&cfg_text).map_err(|e| format!("qwenvl: cannot parse {cfg_path}: {e}"))?;
    let cfg = Qwen3VlConfig::from_hf(&cfg_json);
    let tok = data::qwen_tokenizer::QwenBpe::from_dir(dir).map_err(|e| format!("qwenvl: tokenizer: {e}"))?;

    // Capacity placement: image_row0 is arbitrary (Qwen3Vl::generate's
    // incremental decode derives real placement from the token stream, not
    // from this construction-time value -- see this module's own doc);
    // n_visual is the CAPACITY this resident's DeepStack/splice buffers are
    // sized for, not any one request's actual visual-token count.
    let factor = cfg.vision.patch_size * cfg.vision.spatial_merge_size;
    // A square at the pixel budget is the largest-area, most token-hungry
    // shape smart_resize can produce for that budget (any other aspect ratio
    // at the same area yields <= tokens after patch-grid rounding), so it is
    // the right capacity upper bound to allocate for.
    let side = (max_pixels as f64).sqrt() as u32;
    let (h_cap, w_cap) = smart_resize(side, side, factor, preprocess_min_pixels(), max_pixels);
    let n_visual_capacity = crate::preprocess::image_token_count(h_cap, w_cap, cfg.vision.patch_size, cfg.vision.spatial_merge_size);
    let model = Qwen3Vl::from_hf(dir, cfg.vision.clone(), cfg.text.clone(), SEQ_LEN, cfg.image_token_id, 0, n_visual_capacity, cfg.mrope_section)?;

    Ok(Resident { weights: dir.to_string(), max_pixels, n_visual_capacity, cfg, model, tok })
}

/// Bilinear-resample interleaved-HWC `[0,1]` pixels from `(w,h)` to
/// `(w_bar,h_bar)`, returning CHW (`pack_patches`'s expected layout). Unlike
/// `fastvlm::caps::pad_resize_chw`, no square padding -- Qwen3-VL's own
/// preprocessor resizes directly to the smart-resize target, aspect ratio
/// already accounted for by `smart_resize` itself.
fn hwc_to_chw_resized(hwc: &[f32], w: u32, h: u32, w_bar: u32, h_bar: u32) -> Vec<f32> {
    let sample = |x: f32, y: f32, c: usize| -> f32 {
        let sx = x.clamp(0.0, (w - 1) as f32);
        let sy = y.clamp(0.0, (h - 1) as f32);
        let (x0, y0) = (sx.floor() as u32, sy.floor() as u32);
        let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
        let (fx, fy) = (sx - x0 as f32, sy - y0 as f32);
        let at = |x: u32, y: u32| hwc[((y * w + x) * 3) as usize + c];
        let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
        let bot = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
        top * (1.0 - fy) + bot * fy
    };
    let (sx, sy) = (w as f32 / w_bar as f32, h as f32 / h_bar as f32);
    let mut out = vec![0f32; (3 * h_bar * w_bar) as usize];
    for c in 0..3usize {
        for y in 0..h_bar {
            for x in 0..w_bar {
                let v = sample((x as f32 + 0.5) * sx - 0.5, (y as f32 + 0.5) * sy - 0.5, c);
                out[c * (h_bar * w_bar) as usize + (y * w_bar + x) as usize] = v;
            }
        }
    }
    out
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

    #[test]
    fn missing_weights_is_a_clean_error() {
        let inv = Invocation::new()
            .set("weights", json!("/nonexistent/qwenvl"))
            .set("prompt", json!("describe this"))
            .blob("image", Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w": 1, "h": 1})));
        let r = GenerateAction.run(&inv, &mut |_| {});
        let err = r.err().unwrap_or_default();
        assert!(err.contains("cannot read"), "{err}");
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
