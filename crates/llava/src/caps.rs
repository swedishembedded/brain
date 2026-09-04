// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `capability::Provider` for LLaVA - the image-caption action.
//!
//! One action, `caption`: CLIP-L/14@336 vision tower -> `mlp2x_gelu`
//! projector -> Vicuna-1.5-13B (LLaMA-2) decoder with the image-embed splice
//! -> greedy decode. Mirrors `fastvlm::caps` closely (same two-stage
//! resident-lock split, same `Invocation`/`Outcome` shape) - the
//! `captioner::Captioner` contract exists so `crates/supir`'s pipeline (or
//! any other composer) can drive this action through a
//! [`capability::Registry`] with no direct dependency on this crate; see the
//! `crates/imgpipe` `PipelineProvider` for the composition precedent.
//!
//! **Not exercised against real weights this session** - no
//! `resources/llava/` checkpoint (LLaVA-1.5-13B is a multi-ten-GB download,
//! well past a tokenizer-sized fetch), an honest, stated gap rather than a
//! silently skipped one. The `manifest_validates_without_weights` and
//! `missing_weights_is_a_clean_error` tests are what run without one.
//!
//! Preprocessing (`clip_preprocess_chw`) has no HF-dumped golden either -
//! same honest gap `fastvlm::caps`'s own module doc names for its pad+resize.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use capability::{
    Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome,
    ParamSpec, ParamType, Progress, Provider,
};
use clip::model::{ClipVision, PatchSource, CLIP_VISION_PIPELINES};
use data::llama_bpe::LlamaBpe;
use gpu_core::Gpu;
use qwen3::model::Qwen;

use crate::config::LlavaConfig;

pub const MODEL: &str = "brain/llava";
/// Default LLaVA checkpoint directory - from `$BRAIN_LLAVA_WEIGHTS`, never a
/// baked-in absolute path. Empty when unset, so the `weights` param (or the
/// caller) must supply one.
fn default_weights() -> String {
    std::env::var("BRAIN_LLAVA_WEIGHTS").unwrap_or_default()
}

/// SUPIR's own default caption prompt for its optional LLaVA pre-pass.
const DEFAULT_PROMPT: &str = "Describe this image and its style in a very detailed manner.";
/// The decoder context this crate builds; 576 of it is the image, so a
/// caption cannot ask for more than what is left after the instruction.
const MAX_NEW_LIMIT: u32 = 512;
const T_MAX: u32 = 2048;

pub fn manifest() -> Manifest {
    let caption = ActionSpec::new("caption", "describe an image (CLIP-L/14@336 tower + Vicuna-1.5-13B decoder, greedy)")
        .param(
            ParamSpec::new("weights", ParamType::Str, "LLaVA checkpoint DIRECTORY (config.json + model.safetensors + tokenizer.json)")
                .host_env("BRAIN_LLAVA_WEIGHTS"),
        )
        .param(ParamSpec::new("prompt", ParamType::Str, "instruction for the model").default(serde_json::json!(DEFAULT_PROMPT)))
        .param(ParamSpec::new("max_new", ParamType::Int, "max caption tokens").default(serde_json::json!(128)))
        .param(
            ParamSpec::new("precision", ParamType::Str, "decoder precision: fp32, or int8 (group-wise 32-element weight scales + dynamic activation quant)")
                .default(serde_json::json!("fp32")),
        )
        .input(BlobSpec::new("image", Media::Image, "raw HWC f32 pixels in [0,1], meta {w,h}").required())
        .output(BlobSpec::new("text", Media::Text, "the caption"))
        .streaming();
    Manifest::new(MODEL, "LLaVA-1.5-13B image captioning - CLIP-L336 tower + Vicuna-1.5-13B decoder.", vec![caption])
}

/// The manifest for the RESIDENT/scheduled service (D-Bus, executor, HTTP):
/// the checkpoint directory is service-side configuration
/// (`BRAIN_LLAVA_WEIGHTS`), so the served action carries only real
/// per-request parameters - see `glmdsa::caps::manifest_resident`'s doc for
/// why a static, CLI-facing manifest and a stripped resident one are two
/// different things, not one hidden behind deployment state. Used by
/// `residency::bridge::ProviderResident::stateless_with_manifest`, which
/// validates every served invocation against exactly this spec - so a caller
/// crafting a raw `weights` param cannot reach `CaptionAction::run`'s own
/// per-request override either, not just "not see it in the UI".
pub fn manifest_resident() -> Manifest {
    manifest().for_serving()
}

struct VisionStage {
    weights: String,
    gpu: Gpu,
    vision_weights: HashMap<String, Vec<f32>>,
    proj: HashMap<String, Vec<f32>>,
}

struct DecodeStage {
    weights: String,
    precision: String,
    dec: Qwen,
    head: Vec<f32>,
    tok: LlamaBpe,
}

pub struct LlavaProvider;

impl LlavaProvider {
    pub fn new() -> LlavaProvider {
        LlavaProvider
    }
}

impl Default for LlavaProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for LlavaProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "caption").then(|| Arc::new(CaptionAction) as Arc<dyn Action>)
    }
}

struct CaptionAction;

// One process-wide resident per STAGE, same rationale as `fastvlm::caps`:
// the vision lock is only held while the tower runs, so request N+1's vision
// overlaps request N's decode.
static VISION: Mutex<Option<VisionStage>> = Mutex::new(None);
static DECODE: Mutex<Option<DecodeStage>> = Mutex::new(None);

impl Action for CaptionAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().next().unwrap()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let dir = inv.get_str("weights").filter(|s| !s.is_empty()).unwrap_or_else(default_weights);
        let prompt_text = inv.get_str("prompt").unwrap_or_else(|| DEFAULT_PROMPT.to_string());
        let max_new = inv.get_i64("max_new").unwrap_or(128).clamp(1, MAX_NEW_LIMIT as i64) as usize;
        let precision = inv.get_str("precision").unwrap_or_else(|| "fp32".to_string());
        if precision != "fp32" && precision != "int8" {
            return Err(format!("llava caption: precision must be fp32 or int8, got {precision:?}"));
        }
        let (px, w, h) = capability::blob::decode_image(inv, "image")?;

        let cfg = LlavaConfig::llava_1_5_13b();
        let side = cfg.vision.image_size();
        let chw = clip_preprocess_chw(&px, w, h, side);

        // ---- vision stage: lock held ONLY while the tower runs ----
        let embeds = {
            let mut vguard = VISION.lock().map_err(|_| "llava: vision lock poisoned")?;
            if !matches!(&*vguard, Some(v) if v.weights == dir) {
                *vguard = None;
                *vguard = Some(load_vision(&dir, &cfg)?);
            }
            let hot = vguard.as_ref().unwrap();
            run_vision(&hot.gpu, &hot.vision_weights, &cfg, &chw, &hot.proj)?
        };

        // ---- decode stage: its own lock; the vision lock is already free ----
        let mut dguard = DECODE.lock().map_err(|_| "llava: decode lock poisoned")?;
        if !matches!(&*dguard, Some(d) if d.weights == dir && d.precision == precision) {
            *dguard = None;
            *dguard = Some(load_decode(&dir, &precision, &cfg)?);
        }
        let hot = dguard.as_ref().unwrap();

        let full_prompt = crate::template::caption_prompt(&prompt_text);
        let ids = crate::prompt::tokenize_with_image_splice(&hot.tok, &full_prompt);
        let inputs = crate::prompt::splice_image_embeds(&ids, &embeds, cfg.n_visual_tokens(), cfg.projector_out())?;
        let vocab = hot.dec.cfg.vocab as usize;
        let d = cfg.projector_out() as usize;
        let eos = hot.tok.eos_id();
        hot.dec.reset_cache();
        let mut hidden = hot.dec.prefill(&inputs);
        let logits_of = |hidden: &[f32]| -> Vec<f32> { model::hostmath::matvec_par(&hot.head, hidden, vocab, d) };
        let mut out_ids = Vec::new();
        for i in 0..max_new {
            let lg = logits_of(&hidden);
            let next = lg
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .ok_or("llava: empty logits")?;
            if next == eos {
                break;
            }
            out_ids.push(next);
            progress(Progress::step(i as u32 + 1, max_new as u32, ""));
            hidden = hot.dec.step(next);
        }
        let text = hot.tok.decode(&out_ids);
        Ok(Outcome::new()
            .set("tokens", serde_json::json!(out_ids.len()))
            .set("text", serde_json::json!(text.clone()))
            .blob("text", Blob::new(Media::Text, text.into_bytes())))
    }
}

/// Run the vision tower + `mlp2x_gelu` projector, returning `[n_visual,
/// d_model]` image embeddings. `gpu.share()` gets a fresh handle over the
/// SAME resident device the cached `vision_weights` were uploaded from -
/// `ClipVision::new_on` takes a `Gpu` by value, and `Gpu` is not `Clone` (it
/// wraps a boxed backend), so a plain clone is not an option; `share` is the
/// handle this codebase's own device layer provides for exactly this
/// "borrow the same device for one more owner" need.
fn run_vision(gpu: &Gpu, vision_weights: &HashMap<String, Vec<f32>>, cfg: &LlavaConfig, chw: &[f32], proj: &HashMap<String, Vec<f32>>) -> Result<Vec<f32>, String> {
    let vision = ClipVision::new_on(gpu.share(), cfg.vision.clone(), 1, PatchSource::Pixels, vision_weights);
    vision.set_pixels(chw);
    vision.forward();
    let tapped = vision.read_block_out(cfg.vision_tap_layer() as usize);
    let seq = cfg.vision.n_positions() as usize;
    let d = cfg.vision.d_model() as usize;
    let drop_cls = matches!(cfg.select_feature, crate::config::SelectFeature::Patch);
    let feats = crate::model::select_patch_tokens(&tapped, seq, d, drop_cls);
    let n = cfg.n_visual_tokens() as usize;

    let f1w = proj.get("fc1.weight").ok_or("llava: mm_projector fc1.weight missing")?;
    let f1b = proj.get("fc1.bias").ok_or("llava: mm_projector fc1.bias missing")?;
    let f2w = proj.get("fc2.weight").ok_or("llava: mm_projector fc2.weight missing")?;
    let f2b = proj.get("fc2.bias").ok_or("llava: mm_projector fc2.bias missing")?;
    let projector = crate::model::Projector {
        fc1_w: f1w.clone(),
        fc1_b: f1b.clone(),
        fc2_w: f2w.clone(),
        fc2_b: f2b.clone(),
        mm_hidden: cfg.projector_in() as usize,
        hidden: cfg.projector_out() as usize,
    };
    Ok(projector.forward(&feats, n))
}

fn load_vision(dir: &str, _cfg: &LlavaConfig) -> Result<VisionStage, String> {
    let ckpt = format!("{dir}/model.safetensors");
    let tensors = checkpoint::safetensors::read(&ckpt).map_err(|e| format!("llava: cannot read {ckpt}: {e}"))?;
    let vision_weights = crate::import::build_vision_weights(&tensors);
    let mut proj = HashMap::new();
    for t in &tensors {
        if let Some(k) = crate::import::map_projector(&t.name) {
            proj.insert(k, t.data.clone());
        }
    }
    let gpu = Gpu::new_cpu(CLIP_VISION_PIPELINES);
    Ok(VisionStage { weights: dir.to_string(), gpu, vision_weights, proj })
}

fn load_decode(dir: &str, precision: &str, cfg: &LlavaConfig) -> Result<DecodeStage, String> {
    let ckpt = format!("{dir}/model.safetensors");
    let tensors = checkpoint::safetensors::read(&ckpt).map_err(|e| format!("llava: cannot read {ckpt}: {e}"))?;
    let tok = LlamaBpe::from_dir(std::path::Path::new(dir)).map_err(|e| format!("llava: tokenizer: {e}"))?;
    let mut dec = HashMap::new();
    for t in &tensors {
        if let Some(k) = crate::import::map_decoder(&t.name) {
            dec.insert(k, t.data.clone());
        }
    }
    let dcfg = cfg.decoder.clone();
    let head = dec.get(dcfg.head_weight()).or_else(|| dec.get("tok.weight")).cloned().ok_or("llava: head weight missing from checkpoint")?;
    let dec_model = match precision {
        "int8" => Qwen::new_shard_i8(dcfg.clone(), 1, T_MAX, &dec, model::shard::Shard::whole(dcfg.n_layers as usize)),
        _ => Qwen::new(dcfg, 1, T_MAX, &dec),
    };
    Ok(DecodeStage { weights: dir.to_string(), precision: precision.to_string(), dec: dec_model, head, tok })
}

/// `CLIPImageProcessor`'s default pipeline for a `clip-vit-large-patch14-336`
/// tower: resize the shortest edge to `side` (aspect-preserving), center-crop
/// to `side x side`, normalize by [`crate::model::CLIP_MEAN`]/[`crate::model::CLIP_STD`],
/// return CHW. No HF-dumped golden this session (see the module doc).
fn clip_preprocess_chw(hwc: &[f32], w: u32, h: u32, side: u32) -> Vec<f32> {
    let (rw, rh) = if w <= h {
        (side, ((h as u64 * side as u64) / w as u64).max(1) as u32)
    } else {
        (((w as u64 * side as u64) / h as u64).max(1) as u32, side)
    };
    let resized = imaging::host::resize_bilinear_hwc(hwc, 3, w, h, rw, rh);
    let (ox, oy) = ((rw - side) / 2, (rh - side) / 2);
    let mut cropped = vec![0f32; (side * side * 3) as usize];
    for y in 0..side {
        let src = (((y + oy) * rw + ox) * 3) as usize;
        let dst = (y * side * 3) as usize;
        cropped[dst..dst + (side * 3) as usize].copy_from_slice(&resized[src..src + (side * 3) as usize]);
    }
    for px in cropped.chunks_exact_mut(3) {
        for ((v, &mean), &std) in px.iter_mut().zip(crate::model::CLIP_MEAN.iter()).zip(crate::model::CLIP_STD.iter()) {
            *v = (*v - mean) / std;
        }
    }
    imaging::pixels::hwc_to_chw(&cropped, 3, side as usize, side as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_validates_without_weights() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let a = &m.actions[0];
        assert_eq!(a.name, "caption");
        assert!(a.streaming);
        assert!(a.inputs.iter().any(|b| b.name == "image"));
    }

    #[test]
    fn missing_weights_is_a_clean_error() {
        let inv = Invocation::new()
            .set("weights", serde_json::json!("/nonexistent/llava"))
            .blob("image", Blob::new(Media::Image, vec![0u8; 12]).with_meta(serde_json::json!({"w": 1, "h": 1})));
        let r = CaptionAction.run(&inv, &mut |_| {});
        let err = r.err().unwrap_or_default();
        assert!(err.contains("cannot read"), "{err}");
    }

    /// Resize + center-crop keeps content centred: a 2:1 image resized to
    /// shortest-edge `side` then cropped square must drop the long-axis
    /// margins, not the content.
    #[test]
    fn preprocess_centres_a_wide_image() {
        // 8x4 image, left half black / right half white; side=4 -> resize
        // shortest edge (h=4) unchanged, resize w 8->8 (already >= side),
        // then center-crop the middle 4 columns.
        let (w, h) = (8u32, 4u32);
        let mut hwc = vec![0f32; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let v = if x < 4 { 0.0 } else { 1.0 };
                let i = ((y * w + x) * 3) as usize;
                hwc[i..i + 3].copy_from_slice(&[v, v, v]);
            }
        }
        let out = clip_preprocess_chw(&hwc, w, h, 4);
        assert_eq!(out.len(), 3 * 4 * 4);
    }
}
