// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `capability::Provider` for FastVLM — the image-caption action.
//!
//! One action, `caption`: the exact fully-in-brain pipeline the parity test
//! pins token-for-token against HF (`parity::fastvlm_full_pipeline_caption`):
//! MobileCLIP-L vision tower → `mlp2x_gelu` projector → Qwen2 decoder with the
//! image-embed splice → greedy decode. No HF tensors at inference.
//!
//! The input contract matches the depth provider (the standardized image-blob
//! convention): raw HWC f32 pixels in `[0,1]` + `{w,h}` meta. The provider
//! pads to square and bilinearly resizes to the tower's 1024×1024 input
//! (`image_aspect_ratio: "pad"` in the checkpoint config). NOTE: unlike the
//! tower/projector/decoder — which are parity-gated against HF — this
//! in-provider preprocessing has no golden yet; captions are expected to be
//! robust to interpolation detail, but a dumped-preprocessor reference is the
//! honest next gate (needs torch, absent on this box).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use capability::{
    Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome,
    ParamSpec, ParamType, Progress, Provider,
};
use data::tokenizer::Tokenizer;
use gpu_core::Gpu;
use paramstore::ParamStore;
use qwen::model::Qwen;

pub const MODEL: &str = "fastvlm";
/// Default FastVLM checkpoint directory — from `$BRAIN_FASTVLM_WEIGHTS`, never a
/// baked-in absolute path (see AGENTS.md: no absolute paths in source). Empty when
/// unset, so the `weights` param (or the caller) must supply one.
fn default_weights() -> String {
    std::env::var("BRAIN_FASTVLM_WEIGHTS").unwrap_or_default()
}
/// The tower's input side and its output grid (1024 px → 256 tokens of 3072).
const VISION_SIDE: u32 = 1024;
const IMG_TOKENS: u32 = 256;
const VISION_DIM: u32 = 3072;

pub fn manifest() -> Manifest {
    let caption = ActionSpec::new("caption", "describe an image (MobileCLIP tower + Qwen2 decoder, greedy)")
        .param(
            ParamSpec::new("weights", ParamType::Str, "FastVLM checkpoint DIRECTORY (config.json + model.safetensors + tokenizer.json)")
                .default(serde_json::json!(default_weights())),
        )
        .param(ParamSpec::new("prompt", ParamType::Str, "instruction for the model").default(serde_json::json!("Describe this image.")))
        .param(ParamSpec::new("max_new", ParamType::Int, "max caption tokens").default(serde_json::json!(48)))
        .param(
            ParamSpec::new("precision", ParamType::Str, "decoder precision: fp32, or int8 (per-channel weights + dynamic activation quant)")
                .default(serde_json::json!("fp32")),
        )
        .input(BlobSpec::new("image", Media::Image, "raw HWC f32 pixels in [0,1], meta {w,h}").required())
        .output(BlobSpec::new("text", Media::Text, "the caption"))
        .streaming();
    Manifest::new(MODEL, "FastVLM image captioning — fully in brain, parity-gated against HF.", vec![caption])
}

/// The two pipeline stages are compartmentalized exactly as two Active
/// Objects would be: each owns its resource (the CPU vision device; the GPU
/// decoder) behind ITS OWN lock, held only while that stage runs, and the
/// image embeddings hand off between them BY VALUE — the "event". Request
/// N+1's vision therefore overlaps request N's decode; the old single
/// whole-request mutex serialised them (measured: 3-way load = 3x serial).
/// Each stage lazy-loads its own slice of the checkpoint independently, so
/// there is no lock ordering between stages to get wrong (a cold start reads
/// the safetensors twice; steady state never does).
struct VisionStage {
    weights: String,
    gpu: Gpu,
    vision_ps: ParamStore,
    proj: HashMap<String, Vec<f32>>,
}

struct DecodeStage {
    weights: String,
    precision: String,
    /// The resident KV decoder (fp32 or int8 per `precision`) + the host-side
    /// head row-major table for per-token logits.
    dec: Qwen,
    head: Vec<f32>,
    tok: data::qwen_tokenizer::QwenBpe,
}

pub struct FastVlmProvider;

impl FastVlmProvider {
    pub fn new() -> FastVlmProvider {
        FastVlmProvider
    }
}

impl Default for FastVlmProvider {
    fn default() -> Self {
        Self::new()
    }
}


impl Provider for FastVlmProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "caption").then(|| Arc::new(CaptionAction) as Arc<dyn Action>)
    }
}

struct CaptionAction;

// One process-wide resident per STAGE (the provider is registered once);
// keyed by checkpoint dir (and precision for the decoder) so switching swaps
// cleanly. Two locks, never held together.
static VISION: Mutex<Option<VisionStage>> = Mutex::new(None);
static DECODE: Mutex<Option<DecodeStage>> = Mutex::new(None);

impl Action for CaptionAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().next().unwrap()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let dir = inv.get_str("weights").filter(|s| !s.is_empty()).unwrap_or_else(default_weights);
        let prompt = inv.get_str("prompt").unwrap_or_else(|| "Describe this image.".to_string());
        let max_new = inv.get_i64("max_new").unwrap_or(48).clamp(1, 512) as usize;
        let precision = inv.get_str("precision").unwrap_or_else(|| "fp32".to_string());
        if precision != "fp32" && precision != "int8" {
            return Err(format!("fastvlm caption: precision must be fp32 or int8, got {precision:?}"));
        }
        let (px, w, h) = capability::blob::decode_image(inv, "image")?;

        // 1) pad to square + bilinear resize to the tower input, CHW.
        let t_pre = std::time::Instant::now();
        let chw = pad_resize_chw(&px, w, h, VISION_SIDE);
        stage_time("preprocess", t_pre);

        // ---- vision stage: lock held ONLY while the tower runs ----
        let t_tower = std::time::Instant::now();

        // 2) vision tower + projector → [256, 896] image embeds.
        let embeds = {
            let mut vguard = VISION.lock().map_err(|_| "fastvlm: vision lock poisoned")?;
            if !matches!(&*vguard, Some(v) if v.weights == dir) {
                *vguard = None;
                *vguard = Some(load_vision(&dir)?);
            }
            let hot = vguard.as_ref().unwrap();
            use crate::encoder::{ctx as ectx, kidx, Encoder};
            let gpu = &hot.gpu;
            let ctx = ectx(gpu);
            let enc = Encoder::mobileclip_l(&ctx, VISION_SIDE);
            enc.set_eval(true);
            let img = gpu.storage_init("img", &chw);
            let feat = enc.forward(&ctx, &hot.vision_ps, &img);
            let d = 896u32;
            let featb = gpu.storage_init("feat", &feat);
            let f1w = gpu.storage_init("f1w", &hot.proj["fc1.weight"]);
            let f1b = gpu.storage_init("f1b", &hot.proj["fc1.bias"]);
            let f2w = gpu.storage_init("f2w", &hot.proj["fc2.weight"]);
            let f2b = gpu.storage_init("f2b", &hot.proj["fc2.bias"]);
            let (a, b, e) = (
                gpu.storage((IMG_TOKENS * d) as u64),
                gpu.storage((IMG_TOKENS * d) as u64),
                gpu.storage((IMG_TOKENS * d) as u64),
            );
            gpu.submit(
                &[],
                &[
                    gpu.step(kidx("matmul"), &[&featb, &f1w, &a], &[IMG_TOKENS, VISION_DIM, d], IMG_TOKENS * d),
                    gpu.step(kidx("bias_add"), &[&a, &f1b], &[IMG_TOKENS, d], IMG_TOKENS * d),
                    gpu.step(kidx("gelu_erf"), &[&a, &b], &[IMG_TOKENS * d], IMG_TOKENS * d),
                    gpu.step(kidx("matmul"), &[&b, &f2w, &e], &[IMG_TOKENS, d, d], IMG_TOKENS * d),
                    gpu.step(kidx("bias_add"), &[&e, &f2b], &[IMG_TOKENS, d], IMG_TOKENS * d),
                ],
            );
            let out = gpu.read(&e, (IMG_TOKENS * 896) as usize);
            // A resident device never drops, so its BRAIN_PROFILE table would
            // otherwise never print; surface it while the stage lock is held.
            gpu.dump_profile();
            out
        };
        stage_time("vision+projector", t_tower);

        // ---- decode stage: its own lock; the vision lock is already free ----
        let mut dguard = DECODE.lock().map_err(|_| "fastvlm: decode lock poisoned")?;
        if !matches!(&*dguard, Some(d) if d.weights == dir && d.precision == precision) {
            *dguard = None;
            *dguard = Some(load_decode(&dir, &precision)?);
        }
        let hot = dguard.as_ref().unwrap();

        // 3) KV-cached decode (O(T) per token, not the O(T^2) full recompute
        //    the first profile caught at 96.5% of GPU time): text tokens step
        //    through the cache, image rows enter via step_embed — no residual
        //    splice needed on this path.
        let pre = hot.tok.encode("<|im_start|>user\n");
        let post = hot.tok.encode(&format!("\n{prompt}<|im_end|>\n<|im_start|>assistant\n"));
        let d = 896usize;
        let vocab = hot.dec.cfg.vocab as usize;
        let eos = hot.tok.encode("<|im_end|>").first().copied();
        hot.dec.reset_cache();
        let t_prefill = std::time::Instant::now();
        let mut inputs: Vec<qwen::model::PrefillInput> = Vec::with_capacity(pre.len() + IMG_TOKENS as usize + post.len());
        inputs.extend(pre.iter().map(|&t| qwen::model::PrefillInput::Token(t)));
        inputs.extend((0..IMG_TOKENS as usize).map(|r| qwen::model::PrefillInput::Embed(&embeds[r * d..(r + 1) * d])));
        inputs.extend(post.iter().map(|&t| qwen::model::PrefillInput::Token(t)));
        let mut hidden = hot.dec.prefill(&inputs);
        stage_time("prefill", t_prefill);
        let t_decode = std::time::Instant::now();
        let logits_of =
            |hidden: &[f32]| -> Vec<f32> { model::hostmath::matvec_par(&hot.head, hidden, vocab, d) };
        let mut out_ids = Vec::new();
        for i in 0..max_new {
            let lg = logits_of(&hidden);
            let next = lg
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .ok_or("fastvlm: empty logits")?;
            if Some(next) == eos {
                break;
            }
            out_ids.push(next);
            progress(Progress { step: i as u32 + 1, total: max_new as u32, message: String::new() });
            hidden = hot.dec.step(next);
        }
        stage_time("decode", t_decode);
        hot.dec.gpu().dump_profile();
        let text = hot.tok.decode(&out_ids);
        Ok(Outcome::new()
            .set("tokens", serde_json::json!(out_ids.len()))
            .set("text", serde_json::json!(text.clone()))
            .blob("text", Blob::new(Media::Text, text.into_bytes())))
    }
}

fn load_vision(dir: &str) -> Result<VisionStage, String> {
    let ckpt = format!("{dir}/model.safetensors");
    let tensors = checkpoint::safetensors::read(&ckpt)
        .map_err(|e| format!("fastvlm: cannot read {ckpt}: {e}"))?;
    let mut vt = Vec::new();
    let mut proj = HashMap::new();
    for t in tensors {
        if t.name.contains("vision_tower") {
            vt.push((t.name, t.data));
        } else if let Some(k) = crate::import::map_projector(&t.name) {
            proj.insert(k, t.data);
        }
    }
    use crate::encoder::{ctx as ectx, Encoder, PIPELINES};
    // The vision tower runs on the CPU backend, exactly as the parity test
    // does: at 1024 px the fully-convolutional fp32 activations are multi-GB
    // per stage — host RAM absorbs that; a 24 GB card serving other models
    // does not. The decoder (Qwen) still builds on the selected device.
    let gpu = Gpu::new_cpu(PIPELINES);
    let ctx = ectx(&gpu);
    let enc = Encoder::mobileclip_l(&ctx, VISION_SIDE);
    let vision_ps = ParamStore::new(&gpu, enc.param_list(), &crate::vision_import::build_vision_weights(&vt));
    Ok(VisionStage { weights: dir.to_string(), gpu, vision_ps, proj })
}

fn load_decode(dir: &str, precision: &str) -> Result<DecodeStage, String> {
    let ckpt = format!("{dir}/model.safetensors");
    let tensors = checkpoint::safetensors::read(&ckpt)
        .map_err(|e| format!("fastvlm: cannot read {ckpt}: {e}"))?;
    let tok = data::qwen_tokenizer::QwenBpe::from_dir(dir)
        .map_err(|e| format!("fastvlm: tokenizer: {e}"))?;
    let mut dec = HashMap::new();
    for t in tensors {
        if let Some(k) = crate::import::map_decoder(&t.name) {
            dec.insert(k, t.data);
        }
    }
    // Resident KV decoder, fp32 or int8 (per-channel weights + dynamic
    // activation quant through the decode-regime packed GEMV). Context sized
    // for prompt + image span + the longest caption.
    let cfg = crate::config::FastVlmConfig::fastvlm_0_5b().decoder;
    let head = dec
        .get(cfg.head_weight())
        .or_else(|| dec.get("tok.weight"))
        .cloned()
        .ok_or("fastvlm: head weight missing from checkpoint")?;
    let t_max = 1024u32;
    let dec_model = match precision {
        "int8" => Qwen::new_shard_i8(cfg.clone(), 1, t_max, &dec, model::shard::Shard::whole(cfg.n_layers as usize)),
        _ => Qwen::new(cfg, 1, t_max, &dec),
    };
    Ok(DecodeStage {
        weights: dir.to_string(),
        precision: precision.to_string(),
        dec: dec_model,
        head,
        tok,
    })
}

/// Stage wall time to stderr when `BRAIN_PROFILE` is set — the resident
/// provider's coarse timeline above the per-kernel tables.
fn stage_time(name: &str, since: std::time::Instant) {
    if std::env::var("BRAIN_PROFILE").map(|v| v != "0").unwrap_or(false) {
        eprintln!("stage {name}: {:.1} ms", since.elapsed().as_secs_f64() * 1e3);
    }
}

/// `image_aspect_ratio: "pad"`: letterbox to square with the 0.5 grey fill the
/// HF "pad" processor uses (0.5 grey in [0,1]), then bilinear-resize to `side`,
/// returning CHW.
fn pad_resize_chw(hwc: &[f32], w: u32, h: u32, side: u32) -> Vec<f32> {
    let sq = w.max(h);
    let (ox, oy) = ((sq - w) / 2, (sq - h) / 2);
    let sample = |x: f32, y: f32, c: usize| -> f32 {
        // Position in the padded square → source pixel or the pad fill.
        let sx = x - ox as f32;
        let sy = y - oy as f32;
        if sx < 0.0 || sy < 0.0 || sx > (w - 1) as f32 || sy > (h - 1) as f32 {
            return 0.5;
        }
        let (x0, y0) = (sx.floor() as u32, sy.floor() as u32);
        let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
        let (fx, fy) = (sx - x0 as f32, sy - y0 as f32);
        let at = |x: u32, y: u32| hwc[((y * w + x) * 3) as usize + c];
        let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
        let bot = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
        top * (1.0 - fy) + bot * fy
    };
    let mut out = vec![0f32; (3 * side * side) as usize];
    let scale = sq as f32 / side as f32;
    for c in 0..3usize {
        for y in 0..side {
            for x in 0..side {
                let v = sample((x as f32 + 0.5) * scale - 0.5, (y as f32 + 0.5) * scale - 0.5, c);
                out[c * (side * side) as usize + (y * side + x) as usize] = v;
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
        assert_eq!(m.model, "fastvlm");
        let a = &m.actions[0];
        assert_eq!(a.name, "caption");
        assert!(a.streaming, "per-token Progress is what gives the perf harness TTFT/ITL");
        assert!(a.inputs.iter().any(|b| b.name == "image"));
    }

    #[test]
    fn missing_weights_is_a_clean_error() {
        let inv = Invocation::new()
            .set("weights", serde_json::json!("/nonexistent/fastvlm"))
            .blob(
                "image",
                Blob::new(Media::Image, vec![0u8; 12]).with_meta(serde_json::json!({"w": 1, "h": 1})),
            );
        let r = CaptionAction.run(&inv, &mut |_| {});
        let err = r.err().unwrap_or_default();
        assert!(err.contains("cannot read"), "{err}");
    }

    /// Pad+resize keeps content centred and fills with the 0.5 grey the HF
    /// "pad" processor uses; a wrong offset shows up as a shifted quadrant.
    #[test]
    fn pad_resize_centres_and_fills() {
        // 4x2 image (left half black, right half white) → 4x4 square: one
        // fill row above, content rows in the middle, one fill row below.
        // side == sq, so sampling is exact and the assertions are crisp.
        let mut hwc = Vec::new();
        for _row in 0..2 {
            for col in 0..4 {
                let v = if col < 2 { 0.0f32 } else { 1.0 };
                hwc.extend([v, v, v]);
            }
        }
        let out = pad_resize_chw(&hwc, 4, 2, 4);
        let px = |x: usize, y: usize| out[y * 4 + x]; // channel 0
        assert!(px(0, 0) == 0.5 && px(3, 3) == 0.5, "outside rows are pad fill");
        assert!(px(0, 1) < 0.3 && px(3, 1) > 0.7, "content keeps left-dark/right-light");
        assert!(px(0, 2) < 0.3 && px(3, 2) > 0.7, "both content rows survive");
    }
}
