// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LLaVA composite: CLIP-L/14@336 vision tower -> `mlp2x_gelu` projector ->
//! spliced Vicuna-1.5-13B (LLaMA-2) decoder.
//!
//! Mirrors `fastvlm::model::FastVlm`: the vision tower runs on its own `Gpu`;
//! the projected image tokens cross to the decoder host-side. The projector
//! runs host-side here (a tiny 2-layer MLP - a few tens of MB of weights,
//! nowhere near the cost that would justify keeping it device-resident
//! between calls).
//!
//! Swedish Embedded AB implements solutions for vision-language model
//! integration for its clients. If your team needs expertise in composing
//! existing vision and decoder graphs into one captioning pipeline then you
//! can procure our services by sending an email to info@swedishembedded.com.

use std::collections::HashMap;

use clip::config::ClipVisionConfig;
use clip::model::{ClipVision, PatchSource};
use gpu_core::Gpu;
use qwen3::{Qwen, QwenConfig};

/// OpenAI CLIP's published normalization constants - the preprocessing every
/// CLIP-family tower in this workspace (`EvaVisionConfig::eva02_l336`,
/// FastVLM's tower) was trained against. `ClipVisionConfig` carries no
/// mean/std of its own (see that type's docs): normalization is the caller's
/// job, done once, here, for every consumer of the vanilla CLIP-L tower.
// Two of these carry one more decimal than f32 can represent (the same
// literals `clip::config::EvaVisionConfig::eva02_l336` transcribes, digit
// for digit, from OpenAI's published preprocessing config); truncating them
// would make them stop matching the source they were copied from for no
// numerical gain.
#[allow(clippy::excessive_precision)]
pub const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
#[allow(clippy::excessive_precision)]
pub const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// `mlp2x_gelu` projector weights (host): `Linear(mm_hidden -> hidden)`,
/// erf-GELU, `Linear(hidden -> hidden)`. Identical shape to
/// `fastvlm::model::Projector` - LLaVA-1.5 and FastVLM both use this
/// projector type, just at different `mm_hidden`/`hidden` widths.
pub struct Projector {
    pub fc1_w: Vec<f32>, // [hidden, mm_hidden] (row-major, torch layout)
    pub fc1_b: Vec<f32>, // [hidden]
    pub fc2_w: Vec<f32>, // [hidden, hidden]
    pub fc2_b: Vec<f32>, // [hidden]
    pub mm_hidden: usize,
    pub hidden: usize,
}

impl Projector {
    /// Project `[tokens, mm_hidden]` -> `[tokens, hidden]` on the host.
    pub fn forward(&self, feats: &[f32], tokens: usize) -> Vec<f32> {
        let gelu = |x: f32| 0.5 * x * (1.0 + libm_erf(x / std::f32::consts::SQRT_2));
        let mut out = vec![0f32; tokens * self.hidden];
        for t in 0..tokens {
            let mut h = vec![0f32; self.hidden];
            for (o, ho) in h.iter_mut().enumerate() {
                let mut acc = self.fc1_b[o];
                let wrow = &self.fc1_w[o * self.mm_hidden..(o + 1) * self.mm_hidden];
                let frow = &feats[t * self.mm_hidden..(t + 1) * self.mm_hidden];
                for i in 0..self.mm_hidden {
                    acc += wrow[i] * frow[i];
                }
                *ho = gelu(acc);
            }
            for o in 0..self.hidden {
                let mut acc = self.fc2_b[o];
                let wrow = &self.fc2_w[o * self.hidden..(o + 1) * self.hidden];
                for i in 0..self.hidden {
                    acc += wrow[i] * h[i];
                }
                out[t * self.hidden + o] = acc;
            }
        }
        out
    }
}

/// Abramowitz-Stegun erf (matches the `gelu_erf` kernel) for the host
/// projector - the same approximation `fastvlm::model::libm_erf` uses.
fn libm_erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0 - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t + 0.254_829_6) * t * (-x * x).exp();
    if x < 0.0 {
        -y
    } else {
        y
    }
}

/// Extract the `n_visual` patch-token rows the projector consumes from a
/// `ClipVision::read_block_out` tap: `[1 + n_visual, d]` (class token first,
/// patches after - `model::vit`'s stem order) -> `[n_visual, d]`, dropping
/// the class row. The `SelectFeature::ClsPatch` config keeps all `1+n_visual`
/// rows instead (identity slice).
pub fn select_patch_tokens(tapped: &[f32], seq: usize, d_model: usize, drop_cls: bool) -> Vec<f32> {
    assert_eq!(tapped.len(), seq * d_model, "tapped hidden state shape");
    if drop_cls {
        tapped[d_model..].to_vec()
    } else {
        tapped.to_vec()
    }
}

/// An assembled LLaVA (training-shaped forward path, mirroring
/// `fastvlm::model::FastVlm::forward`). Image tokens occupy a contiguous run
/// of `IMAGE_TOKEN_INDEX` (-200) in the text stream at `image_row0` - see
/// `crate::prompt` for the inference-time (KV-cached) equivalent this shares
/// no code with, since prefill/decode is a different Qwen entry point than
/// the batched loss forward.
pub struct Llava {
    vision_pipelines: &'static [(&'static str, &'static str)],
    vision_cfg: ClipVisionConfig,
    vision_weights: HashMap<String, Vec<f32>>,
    projector: Projector,
    decoder: Qwen,
    n_visual: u32,
    drop_cls: bool,
}

impl Llava {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vision_cfg: ClipVisionConfig,
        vision_weights: HashMap<String, Vec<f32>>,
        projector: Projector,
        dcfg: QwenConfig,
        dweights: &HashMap<String, Vec<f32>>,
        seq_len: u32,
        image_row0: u32,
        n_visual: u32,
        drop_cls: bool,
        vision_pipelines: &'static [(&'static str, &'static str)],
    ) -> Llava {
        let mut decoder = Qwen::new(dcfg, 1, seq_len, dweights);
        decoder.enable_mm_splice(image_row0, n_visual);
        Llava { vision_pipelines, vision_cfg, vision_weights, projector, decoder, n_visual, drop_cls }
    }

    /// End-to-end forward: encode the image, project the tokens, splice them
    /// into the decoder, and return the decoder's scalar loss. `img` is
    /// `[1,3,image_size,image_size]`, CLIP-normalized; `tokens`/`targets` the
    /// text stream (image placeholders -> `qwen3::IGNORE`).
    ///
    /// A fresh vision `Gpu` is built per call (`ClipVision::new_on` takes one
    /// by value): the CPU backend this crate uses for the tower is cheap to
    /// stand up and this is not a hot serving path (see `caps.rs`'s own
    /// resident-stage split for the version that IS).
    pub fn forward(&self, tokens: &[u32], targets: &[u32], img: &[f32], select_layer: u32) -> f32 {
        let vgpu = Gpu::new_cpu(self.vision_pipelines);
        let vision = ClipVision::new_on(vgpu, self.vision_cfg.clone(), 1, PatchSource::Pixels, &self.vision_weights);
        vision.set_pixels(img);
        vision.forward();
        let tapped = vision.read_block_out(select_layer as usize);
        let seq = (1 + self.vision_cfg.native_patches()) as usize;
        let feats = select_patch_tokens(&tapped, seq, self.vision_cfg.d_model() as usize, self.drop_cls);
        let t = feats.len() / self.vision_cfg.d_model() as usize;
        assert_eq!(t as u32, self.n_visual, "vision token count must match the splice");

        let img_embeds = self.projector.forward(&feats, t);
        self.decoder.write_img_embeds(&img_embeds);
        self.decoder.set_batch(tokens, targets);
        self.decoder.forward()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clip::model::CLIP_VISION_PIPELINES;
    use data::rng::Rng;

    fn rvec(n: usize, rng: &mut Rng) -> Vec<f32> {
        (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect()
    }

    /// Weight-free end-to-end smoke test at a tiny reduced config (a real
    /// 24x1024 CLIP-L336 tower + 40x5120 LLaMA-2-13B decoder has no place in
    /// a fast unit test regardless of checkpoint access) - proves the vision
    /// -> projector -> decoder wiring produces a finite loss: the
    /// weight-free "mapping-units" rung of this crate's parity ladder.
    /// Single-forward/composed-loop rungs against real reference activations
    /// need a real checkpoint, which was not obtained this session (a stated
    /// gap, not a silent one).
    #[test]
    fn end_to_end_forward_is_finite() {
        let vision_cfg = ClipVisionConfig {
            shape: gguf::deepseek_ocr_vision::ClipConfig {
                d_model: 16,
                n_layers: 2,
                n_heads: 2,
                ffn_hidden: 32,
                patch_size: 4,
                image_size: 8,
                n_positions: 5, // 1 cls + 4 patches
                layer_norm_eps: 1e-5,
            },
            act: clip::config::TextAct::QuickGelu,
        };
        let hidden = 24usize;
        let n_visual = vision_cfg.native_patches(); // 4

        let mut rng = Rng::new(1);
        let vision_weights: HashMap<String, Vec<f32>> = vision_cfg
            .tensor_manifest()
            .iter()
            .map(|(n, s)| (n.clone(), rvec(s.iter().product(), &mut rng)))
            .collect();

        let projector = Projector {
            fc1_w: rvec(hidden * vision_cfg.d_model() as usize, &mut rng),
            fc1_b: rvec(hidden, &mut rng),
            fc2_w: rvec(hidden * hidden, &mut rng),
            fc2_b: rvec(hidden, &mut rng),
            mm_hidden: vision_cfg.d_model() as usize,
            hidden,
        };

        let dcfg = QwenConfig {
            vocab: 23,
            block_size: 12,
            n_layers: 2,
            d_model: hidden as u32,
            n_heads: 4,
            n_kv_heads: 4, // MHA, matching LLaMA-2
            head_dim: hidden as u32 / 4,
            d_ff: 32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_position_embeddings: 12,
            tie_embeddings: false,
            qk_norm: false,
            attn_bias: false,
            lora: None,
        };
        let dweights = qwen3::init_weights(&dcfg, 3);

        // Stream: 1 text, 4 image (-200 placeholders), 1 text; IGNORE at image rows.
        let tokens = vec![1u32, 7, 7, 7, 7, 3];
        let mut targets = vec![7u32, 0, 0, 0, 0, 5];
        for t in targets.iter_mut().take(5).skip(1) {
            *t = qwen3::IGNORE;
        }

        let model = Llava::new(vision_cfg.clone(), vision_weights, projector, dcfg, &dweights, tokens.len() as u32, 1, n_visual, true, CLIP_VISION_PIPELINES);
        let img: Vec<f32> = (0..(3 * 8 * 8) as usize).map(|_| rng.next_f32() - 0.5).collect();
        let loss = model.forward(&tokens, &targets, &img, vision_cfg.penultimate_layer());
        assert!(loss.is_finite() && loss > 0.0, "end-to-end LLaVA loss must be finite+positive, got {loss}");
    }

    #[test]
    fn select_patch_tokens_drops_only_the_class_row() {
        let tapped: Vec<f32> = (0..(3 * 2)).map(|v| v as f32).collect(); // seq=3, d=2
        let patch = select_patch_tokens(&tapped, 3, 2, true);
        assert_eq!(patch, vec![2.0, 3.0, 4.0, 5.0]);
        let cls_patch = select_patch_tokens(&tapped, 3, 2, false);
        assert_eq!(cls_patch, tapped);
    }
}
