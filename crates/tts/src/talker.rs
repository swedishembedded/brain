// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS **Talker**: a Qwen3 dense decoder (RMSNorm, GQA with per-head
//! QK-norm, half-split RoPE base 1e6, SwiGLU) over an *untied* codec embedding /
//! `codec_head`, predicting codebook-0 acoustic-token logits.
//!
//! ## Reuse
//! The decoder backbone is byte-for-byte a [`qwen::Qwen`] with `tie_embeddings =
//! false`, so the parity-exact Qwen3 forward/backward, LoRA, vocab-tiling and
//! safetensors loader are reused wholesale. `TalkerModel` wraps an inner `Qwen`
//! and adds the multi-modal embedding front-end (text projection) that the
//! Qwen decoder does not model.
//!
//! ## Resolved RoPE / M-RoPE convention (verified against the reference)
//! The Talker config declares M-RoPE (`rope_scaling = {interleaved: true,
//! mrope_section: [24,20,20]}`). In `modeling_qwen3_tts.py`, `get_rope_index`
//! builds `position_ids = cumsum(attention_mask) - 1` and **`expand(3, …)`** —
//! all three (temporal/height/width) mrope sections receive the *same* index for
//! a pure audio/text token stream. `apply_multimodal_rotary_pos_emb` then
//! interleaves the three sections, but since they are identical the result is
//! exactly the single-section `cat(freqs, freqs)` rotation with `rotate_half`
//! (split-in-half). That is **Qwen's half-split RoPE-base** (`kernels::ROPE_BASE`,
//! θ = 1e6) — identical to `qwen`. No new kernel and no interleaving is required
//! for the non-padded audio stream. (Padding/offset only shifts the shared index
//! by a constant `rope_delta`, which the per-position RoPE handles automatically.)

use qwen::Qwen;

use crate::config::TalkerConfig;

/// CPU-resident text-conditioning weights: `text_projection` is a 2-layer MLP
/// (`fc2(model::hostmath::silu(fc1(x)))`, both with bias) mapping a `text_hidden`-dim text
/// embedding to the Talker `d_model`. `text_embedding` (the `[text_vocab,
/// text_hidden]` lookup table) is optional — it is large (≈1.2 GB f32 for the
/// real model) so callers may pre-embed text and pass hidden states to
/// [`TextProjection::project`] directly.
pub struct TextProjection {
    pub text_embedding: Option<Vec<f32>>,
    pub fc1_w: Vec<f32>, // [inter, in]
    pub fc1_b: Vec<f32>, // [inter]
    pub fc2_w: Vec<f32>, // [out, inter]
    pub fc2_b: Vec<f32>, // [out]
    pub in_dim: usize,
    pub inter: usize,
    pub out: usize,
    pub text_vocab: usize,
}


impl TextProjection {
    /// Look up `ids` in the text-embedding table (panics if it was not loaded).
    pub fn embed_text(&self, ids: &[u32]) -> Vec<f32> {
        let emb = self
            .text_embedding
            .as_ref()
            .expect("text_embedding table not loaded");
        let mut out = vec![0.0f32; ids.len() * self.in_dim];
        for (r, &id) in ids.iter().enumerate() {
            let src = id as usize * self.in_dim;
            out[r * self.in_dim..(r + 1) * self.in_dim]
                .copy_from_slice(&emb[src..src + self.in_dim]);
        }
        out
    }

    /// Project `[n, in_dim]` text hidden states to `[n, out]` (the Talker
    /// `d_model`) via `fc2(model::hostmath::silu(fc1(x)))`. Matches `Qwen3TTSTalkerResizeMLP`.
    pub fn project(&self, hidden: &[f32]) -> Vec<f32> {
        let n = hidden.len() / self.in_dim;
        let mut out = vec![0.0f32; n * self.out];
        let mut mid = vec![0.0f32; self.inter];
        for r in 0..n {
            let x = &hidden[r * self.in_dim..(r + 1) * self.in_dim];
            for j in 0..self.inter {
                let w = &self.fc1_w[j * self.in_dim..(j + 1) * self.in_dim];
                let mut acc = self.fc1_b[j];
                for k in 0..self.in_dim {
                    acc += w[k] * x[k];
                }
                mid[j] = model::hostmath::silu(acc);
            }
            for o in 0..self.out {
                let w = &self.fc2_w[o * self.inter..(o + 1) * self.inter];
                let mut acc = self.fc2_b[o];
                for k in 0..self.inter {
                    acc += w[k] * mid[k];
                }
                out[r * self.out + o] = acc;
            }
        }
        out
    }
}

/// The Talker model: an inner Qwen3 decoder (untied codec head) plus the optional
/// text-projection front-end.
pub struct TalkerModel {
    pub inner: Qwen,
    pub cfg: TalkerConfig,
    pub text: Option<TextProjection>,
}

impl TalkerModel {
    /// Build a randomly-initialised **trainable** Talker (for tests / gradient
    /// checks). The text-projection front-end is omitted.
    pub fn new_trainable(cfg: TalkerConfig, b: u32, t: u32, seed: u64) -> TalkerModel {
        let qcfg = cfg.to_qwen(t);
        let init = qwen::init_weights(&qcfg, seed);
        let inner = Qwen::new(qcfg, b, t, &init);
        TalkerModel {
            inner,
            cfg,
            text: None,
        }
    }

    /// Load an inference-only Talker from a brain checkpoint written by
    /// [`crate::import::import_talker`]: the Qwen decoder (frozen weights) plus
    /// the text-projection tensors (and, if present, the text-embedding table).
    pub fn load_inference(path: &str, b: u32, t: u32) -> TalkerModel {
        let inner = Qwen::load_inference(path, b, t);
        let mut cfg = TalkerConfig::from_qwen(&inner.cfg);
        let c = checkpoint::load(path);
        let take = |name: &str| c.find(name, "").cloned();
        let text = match (
            take("text_projection.fc1.weight"),
            take("text_projection.fc1.bias"),
            take("text_projection.fc2.weight"),
            take("text_projection.fc2.bias"),
        ) {
            (Some(fc1_w), Some(fc1_b), Some(fc2_w), Some(fc2_b)) => {
                let inter = fc1_b.len();
                let in_dim = fc1_w.len() / inter;
                let out = fc2_b.len();
                let text_embedding = take("text_embedding.weight");
                let text_vocab = text_embedding
                    .as_ref()
                    .map(|e| e.len() / in_dim)
                    .unwrap_or(0);
                cfg.text_hidden_size = in_dim as u32;
                if text_vocab > 0 {
                    cfg.text_vocab_size = text_vocab as u32;
                }
                Some(TextProjection {
                    text_embedding,
                    fc1_w,
                    fc1_b,
                    fc2_w,
                    fc2_b,
                    in_dim,
                    inter,
                    out,
                    text_vocab,
                })
            }
            _ => None,
        };
        TalkerModel { inner, cfg, text }
    }

    /// Set a codec `(input, target)` batch on the inner decoder (for training /
    /// gradient checks). Targets use `qwen::IGNORE` to mask.
    pub fn set_codec_batch(&self, x: &[u32], y: &[u32]) {
        self.inner.set_batch(x, y);
    }

    /// Codebook-0 logits for every position of a single codec-token sequence,
    /// shape `[T, vocab]` (= `[T, 3072]` for the real model). This is the Talker
    /// decoder + `codec_head`; the multi-modal text/codec-sum embedding is built
    /// by the caller (see [`TextProjection`]) and is additive on the input side.
    pub fn logits_all(&self, codec_tokens: &[u32]) -> Vec<f32> {
        self.inner.logits_all(codec_tokens)
    }

    /// Codebook-0 vocab size (`cfg.vocab`).
    pub fn vocab(&self) -> u32 {
        self.cfg.vocab
    }
}

impl TalkerModel {
    /// Trainable parameter names of the inner decoder (for the gradient checker).
    pub fn param_names(&self) -> Vec<String> {
        model::Model::param_names(&self.inner)
    }
    /// Run the inner decoder forward and return the scalar masked-CE loss.
    pub fn forward(&self) -> f32 {
        self.inner.forward()
    }
    pub fn backward(&self) {
        self.inner.backward();
    }
    pub fn zero_grads(&self) {
        self.inner.zero_grads();
    }
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.inner.read_weight(name)
    }
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.inner.write_weight(name, data);
    }
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.inner.read_grad(name)
    }
    pub fn poll_wait(&self) {
        self.inner.poll_wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_projection_shapes_and_silu() {
        // in=3, inter=4, out=2. fc1 = identity-ish, biases zero.
        let tp = TextProjection {
            text_embedding: Some(vec![1.0; 2 * 3]),
            fc1_w: vec![
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, //
                1.0, 1.0, 1.0,
            ],
            fc1_b: vec![0.0; 4],
            fc2_w: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            fc2_b: vec![0.0; 2],
            in_dim: 3,
            inter: 4,
            out: 2,
            text_vocab: 2,
        };
        let h = vec![1.0f32, 2.0, 3.0];
        let y = tp.project(&h);
        assert_eq!(y.len(), 2);
        // y[0] = model::hostmath::silu(1), y[1] = model::hostmath::silu(2)
        assert!((y[0] - model::hostmath::silu(1.0)).abs() < 1e-6);
        assert!((y[1] - model::hostmath::silu(2.0)).abs() < 1e-6);
        let e = tp.embed_text(&[0, 1]);
        assert_eq!(e.len(), 6);
    }
}
