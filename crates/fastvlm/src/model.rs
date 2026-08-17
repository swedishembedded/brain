// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FastVLM composite: FastViTHD encoder → mlp2x_gelu projector → spliced Qwen2
//! decoder (LLaVA-style, image token id -200 expanded to `n_visual` slots).
//!
//! Mirrors `qwen3vl::Qwen3Vl` but simpler: the Qwen2 decoder uses plain RoPE (no
//! M-RoPE) and there's no DeepStack. The vision tower runs on its own `Gpu`; the
//! projected image tokens cross to the decoder host-side via `write_img_embeds`.
//! The projector runs host-side here (a 2-layer MLP); an on-device/trainable
//! projector is the finetune path (as with qwenvl's host pos-embed).

use std::collections::HashMap;

use gpu_core::Gpu;
use qwen3::{Qwen, QwenConfig};

use crate::encoder::{ctx, Encoder, PIPELINES};

/// mlp2x_gelu projector weights (host): `Linear(mm_hidden→hidden)`, erf-GELU,
/// `Linear(hidden→hidden)`.
pub struct Projector {
    pub fc1_w: Vec<f32>, // [hidden, mm_hidden] (row-major, torch layout)
    pub fc1_b: Vec<f32>, // [hidden]
    pub fc2_w: Vec<f32>, // [hidden, hidden]
    pub fc2_b: Vec<f32>, // [hidden]
    pub mm_hidden: usize,
    pub hidden: usize,
}

impl Projector {
    /// Project `[tokens, mm_hidden]` → `[tokens, hidden]` on the host.
    fn forward(&self, feats: &[f32], tokens: usize) -> Vec<f32> {
        let gelu = |x: f32| 0.5 * x * (1.0 + libm_erf(x / std::f32::consts::SQRT_2));
        let mut out = vec![0f32; tokens * self.hidden];
        for t in 0..tokens {
            // h = gelu(fc1 · feat + b1)
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
            // out = fc2 · h + b2
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

/// Abramowitz-Stegun erf (matches the gelu_erf kernel) for the host projector.
fn libm_erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0 - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t + 0.254_829_6) * t * (-x * x).exp();
    if x < 0.0 {
        -y
    } else {
        y
    }
}

/// An assembled FastVLM (forward path). Image tokens occupy a contiguous run of
/// `image_token_index` (-200) in the text stream at `image_row0`.
pub struct FastVlm {
    egpu: Gpu,
    enc_layers: [u32; 5],
    enc_dims: [u32; 5],
    mlp_ratio: u32,
    cls_ratio: u32,
    input: u32,
    enc_weights: HashMap<String, Vec<f32>>,
    projector: Projector,
    decoder: Qwen,
    n_visual: u32,
}

impl FastVlm {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        enc_layers: [u32; 5],
        enc_dims: [u32; 5],
        mlp_ratio: u32,
        cls_ratio: u32,
        input: u32,
        enc_weights: HashMap<String, Vec<f32>>,
        projector: Projector,
        dcfg: QwenConfig,
        dweights: &HashMap<String, Vec<f32>>,
        seq_len: u32,
        image_row0: u32,
        n_visual: u32,
    ) -> FastVlm {
        let mut decoder = Qwen::new(dcfg, 1, seq_len, dweights);
        decoder.enable_mm_splice(image_row0, n_visual);
        FastVlm {
            egpu: Gpu::new_cpu(PIPELINES),
            enc_layers,
            enc_dims,
            mlp_ratio,
            cls_ratio,
            input,
            enc_weights,
            projector,
            decoder,
            n_visual,
        }
    }

    /// End-to-end forward: encode the image, project the tokens, splice them into
    /// the decoder, and return the decoder's scalar loss. `img` is `[1,3,input,
    /// input]`; `tokens`/`targets` the text stream (image placeholders → IGNORE).
    pub fn forward(&self, tokens: &[u32], targets: &[u32], img: &[f32]) -> f32 {
        let ctx = ctx(&self.egpu);
        let enc = Encoder::new(&ctx, self.enc_layers, self.enc_dims, self.mlp_ratio, self.cls_ratio, self.input);
        let ps = paramstore::ParamStore::new(&self.egpu, enc.param_list(), &self.enc_weights);
        let feats = enc.forward(&ctx, &ps, &self.egpu.storage_init("img", img));
        let t = enc.tokens() as usize;
        assert_eq!(t as u32, self.n_visual, "encoder token count must match the splice");
        let img_embeds = self.projector.forward(&feats, t);

        self.decoder.write_img_embeds(&img_embeds);
        self.decoder.set_batch(tokens, targets);
        self.decoder.forward()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Rng;

    fn rvec(n: usize, rng: &mut Rng) -> Vec<f32> {
        (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect()
    }

    #[test]
    fn end_to_end_forward_is_finite() {
        // Tiny tower (4 tokens × feature_dim 64) → projector 64→40 → Qwen2 d_model 40.
        let (layers, dims) = ([1u32, 1, 1, 1, 1], [8u32, 16, 16, 32, 32]);
        let (mlp_ratio, cls_ratio, input) = (2u32, 2u32, 128u32);
        let feature_dim = dims[4] * cls_ratio; // 64
        let hidden = 40usize;

        // Encoder weights: build a throwaway encoder to enumerate its param list.
        let egpu = Gpu::new_cpu(PIPELINES);
        let c = ctx(&egpu);
        let enc = Encoder::new(&c, layers, dims, mlp_ratio, cls_ratio, input);
        let mut rng = Rng::new(1);
        let enc_weights: HashMap<String, Vec<f32>> = enc
            .param_list()
            .iter()
            .map(|(n, sz)| {
                let v = if n.ends_with("running_var") { vec![1.0; *sz] } else if n.contains("running_mean") { vec![0.0; *sz] } else { rvec(*sz, &mut rng) };
                (n.clone(), v)
            })
            .collect();
        let n_visual = enc.tokens(); // 4

        let projector = Projector {
            fc1_w: rvec(hidden * feature_dim as usize, &mut rng),
            fc1_b: rvec(hidden, &mut rng),
            fc2_w: rvec(hidden * hidden, &mut rng),
            fc2_b: rvec(hidden, &mut rng),
            mm_hidden: feature_dim as usize,
            hidden,
        };

        let dcfg = QwenConfig::qwen2(23, 2, hidden as u32, 4, 2, 64, true);
        let dweights = qwen3::init_weights(&dcfg, 3);

        // Stream: 1 text, 4 image (-200 placeholders), 1 text; IGNORE at image rows.
        let tokens = vec![1u32, 7, 7, 7, 7, 3];
        let mut targets = vec![7u32, 0, 0, 0, 0, 5];
        for t in targets.iter_mut().take(5).skip(1) {
            *t = qwen3::IGNORE;
        }

        let model = FastVlm::new(layers, dims, mlp_ratio, cls_ratio, input, enc_weights, projector, dcfg, &dweights, tokens.len() as u32, 1, n_visual);
        let img: Vec<f32> = (0..(3 * input * input) as usize).map(|_| rng.next_f32() - 0.5).collect();
        let loss = model.forward(&tokens, &targets, &img);
        assert!(loss.is_finite() && loss > 0.0, "end-to-end FastVLM loss must be finite+positive, got {loss}");
    }
}
