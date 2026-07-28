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
const DEFAULT_WEIGHTS: &str = "/data/workspace/resources/vl/fastvlm/hf/FastVLM-0.5B";
/// The tower's input side and its output grid (1024 px → 256 tokens of 3072).
const VISION_SIDE: u32 = 1024;
const IMG_TOKENS: u32 = 256;
const VISION_DIM: u32 = 3072;

pub fn manifest() -> Manifest {
    let caption = ActionSpec::new("caption", "describe an image (MobileCLIP tower + Qwen2 decoder, greedy)")
        .param(
            ParamSpec::new("weights", ParamType::Str, "FastVLM checkpoint DIRECTORY (config.json + model.safetensors + tokenizer.json)")
                .default(serde_json::json!(DEFAULT_WEIGHTS)),
        )
        .param(ParamSpec::new("prompt", ParamType::Str, "instruction for the model").default(serde_json::json!("Describe this image.")))
        .param(ParamSpec::new("max_new", ParamType::Int, "max caption tokens").default(serde_json::json!(48)))
        .input(BlobSpec::new("image", Media::Image, "raw HWC f32 pixels in [0,1], meta {w,h}").required())
        .output(BlobSpec::new("text", Media::Text, "the caption"))
        .streaming();
    Manifest::new(MODEL, "FastVLM image captioning — fully in brain, parity-gated against HF.", vec![caption])
}

/// Everything expensive, resident per checkpoint dir: the split weight maps,
/// the vision device + encoder params, and the tokenizer.
struct Hot {
    weights: String,
    gpu: Gpu,
    vision_ps: ParamStore,
    proj: HashMap<String, Vec<f32>>,
    dec: HashMap<String, Vec<f32>>,
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

// One process-wide resident (the provider is registered once); keyed by the
// checkpoint dir so switching weights swaps cleanly.
static HOT: Mutex<Option<Hot>> = Mutex::new(None);

impl Action for CaptionAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().next().unwrap()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let dir = inv.get_str("weights").unwrap_or_else(|| DEFAULT_WEIGHTS.to_string());
        let prompt = inv.get_str("prompt").unwrap_or_else(|| "Describe this image.".to_string());
        let max_new = inv.get_i64("max_new").unwrap_or(48).clamp(1, 512) as usize;
        let (px, w, h) = image_of(inv)?;

        let mut guard = HOT.lock().map_err(|_| "fastvlm: hot lock poisoned")?;
        if !matches!(&*guard, Some(hot) if hot.weights == dir) {
            *guard = None; // free the old resident weights first
            *guard = Some(load_hot(&dir)?);
        }
        let hot = guard.as_ref().unwrap();

        // 1) pad to square + bilinear resize to the tower input, CHW.
        let chw = pad_resize_chw(&px, w, h, VISION_SIDE);

        // 2) vision tower + projector → [256, 896] image embeds.
        let embeds = {
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
            gpu.read(&e, (IMG_TOKENS * 896) as usize)
        };

        // 3) prompt layout: chat-template text around the spliced image span,
        //    exactly the shape the parity harness verified.
        let pre = hot.tok.encode(&format!("<|im_start|>user\n"));
        let post = hot.tok.encode(&format!("\n{prompt}<|im_end|>\n<|im_start|>assistant\n"));
        let img_start = pre.len() as u32;
        let mut seq: Vec<u32> = pre.clone();
        seq.extend(std::iter::repeat(0u32).take(IMG_TOKENS as usize));
        seq.extend(&post);

        // 4) decoder with the image-embed splice; greedy, one Progress/token.
        let cfg = crate::config::FastVlmConfig::fastvlm_0_5b().decoder;
        let vocab = cfg.vocab as usize;
        let eos = hot.tok.encode("<|im_end|>").first().copied();
        let t_max = (seq.len() + max_new + 1) as u32;
        let mut dec = Qwen::new(cfg, 1, t_max, &hot.dec);
        dec.enable_mm_splice(img_start, IMG_TOKENS);
        dec.write_img_embeds(&embeds);
        let mut out_ids = Vec::new();
        for i in 0..max_new {
            let lg = dec.logits_all(&seq);
            let last = &lg[(seq.len() - 1) * vocab..seq.len() * vocab];
            let next = last
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .ok_or("fastvlm: empty logits")?;
            if Some(next) == eos {
                break;
            }
            out_ids.push(next);
            seq.push(next);
            progress(Progress { step: i as u32 + 1, total: max_new as u32, message: String::new() });
        }
        let text = hot.tok.decode(&out_ids);
        Ok(Outcome::new()
            .set("tokens", serde_json::json!(out_ids.len()))
            .set("text", serde_json::json!(text.clone()))
            .blob("text", Blob::new(Media::Text, text.into_bytes())))
    }
}

fn load_hot(dir: &str) -> Result<Hot, String> {
    let ckpt = format!("{dir}/model.safetensors");
    let tensors = checkpoint::safetensors::read(&ckpt)
        .map_err(|e| format!("fastvlm: cannot read {ckpt}: {e}"))?;
    let tok = data::qwen_tokenizer::QwenBpe::from_dir(dir)
        .map_err(|e| format!("fastvlm: tokenizer: {e}"))?;
    let mut vt = Vec::new();
    let mut proj = HashMap::new();
    let mut dec = HashMap::new();
    for t in tensors {
        if t.name.contains("vision_tower") {
            vt.push((t.name, t.data));
        } else if let Some(k) = crate::import::map_projector(&t.name) {
            proj.insert(k, t.data);
        } else if let Some(k) = crate::import::map_decoder(&t.name) {
            dec.insert(k, t.data);
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
    Ok(Hot { weights: dir.to_string(), gpu, vision_ps, proj, dec, tok })
}

/// Decode the standardized image blob: raw HWC f32 `[0,1]` + `{w,h}` meta.
fn image_of(inv: &Invocation) -> Result<(Vec<f32>, u32, u32), String> {
    let blob = inv.get_blob("image").ok_or("fastvlm caption: missing input 'image'")?;
    let w = blob.meta.get("w").and_then(|v| v.as_u64()).ok_or("fastvlm caption: image meta needs w")? as u32;
    let h = blob.meta.get("h").and_then(|v| v.as_u64()).ok_or("fastvlm caption: image meta needs h")? as u32;
    let px: &[f32] = bytemuck::cast_slice(&blob.bytes);
    if px.len() != (w * h * 3) as usize {
        return Err(format!("fastvlm caption: expected {}x{}x3 f32, got {} values", w, h, px.len()));
    }
    Ok((px.to_vec(), w, h))
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
