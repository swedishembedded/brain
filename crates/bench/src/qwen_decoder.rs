// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Qwen3 dense decoder as a [`DecoderLm`] — a third architecture the
//! benchmark battery can score, alongside [`GptDecoder`](crate::GptDecoder) and
//! [`MoeDecoder`](crate::MoeDecoder).
//!
//! `qwen::Qwen` implements the architecture-agnostic [`model::Model`] seam *and*
//! exposes `logits_all` on the trainer itself, so training routes through the
//! same generic `model::train::fit`, and scoring loads an inference-only
//! (frozen, no optimizer state) instance from the saved checkpoint.

use std::path::Path;

use model::FitOpts;
use qwen::config::QwenConfig;
use qwen::model::Qwen;

use crate::model::{DecoderLm, Scorer, TrainConfig};

/// The Qwen3 decoder as a [`DecoderLm`].
#[derive(Clone, Debug, Default)]
pub struct QwenDecoder;

impl QwenDecoder {
    /// Build a [`QwenConfig`] from the architecture-neutral [`TrainConfig`].
    /// GQA: `n_kv_heads = n_heads/2` when even (else MHA). `head_dim` and `d_ff`
    /// follow GPT-comparable conventions. `vocab` is `0` (inferred from the
    /// dataset's `meta.json`).
    pub fn qwen_config(&self, block_size: u32, cfg: &TrainConfig) -> QwenConfig {
        let n_kv = if cfg.n_heads % 2 == 0 { cfg.n_heads / 2 } else { cfg.n_heads };
        QwenConfig {
            vocab: 0,
            block_size,
            n_layers: cfg.n_layers,
            d_model: cfg.d_model,
            n_heads: cfg.n_heads,
            n_kv_heads: n_kv,
            head_dim: cfg.d_model / cfg.n_heads,
            d_ff: cfg.d_model * 4,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            tie_embeddings: true,
            lora: None,
        }
    }
}

impl DecoderLm for QwenDecoder {
    fn arch_name(&self) -> &'static str {
        "qwen"
    }

    fn train_decoder(
        &self,
        dir: &Path,
        block_size: u32,
        cfg: &TrainConfig,
        weights_out: &Path,
    ) -> std::io::Result<(f32, f32)> {
        let qcfg = self.qwen_config(block_size, cfg);
        let opts = FitOpts {
            steps: cfg.steps,
            batch_size: cfg.batch_size,
            block_size,
            lr: cfg.lr,
            warmup: 20,
            decay_iters: cfg.steps * 2,
            eval_interval: 0,
            seed: cfg.seed,
            mask_before: cfg.mask_before,
            mask_per_line: cfg.mask_per_line,
            align_to_lines: cfg.align_to_lines,
            ..Default::default()
        };
        model::train::fit::<Qwen>(dir, qcfg, &opts, Some(weights_out))
    }

    fn load_scorer(&self, weights: &Path, block_size: u32) -> Box<dyn Scorer> {
        // Inference-only (frozen) load — no optimizer-state allocation.
        Box::new(QwenScorer {
            model: Qwen::load_inference(weights.to_str().expect("utf-8 path"), 1, block_size),
        })
    }
}

struct QwenScorer {
    model: Qwen,
}

impl Scorer for QwenScorer {
    fn vocab(&self) -> usize {
        self.model.cfg.vocab as usize
    }
    fn logits_all(&self, tokens: &[u32]) -> Vec<f32> {
        Qwen::logits_all(&self.model, tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_decoder_arch_name_and_config() {
        assert_eq!(QwenDecoder.arch_name(), "qwen");
        let tc = TrainConfig { n_layers: 2, d_model: 64, n_heads: 4, ..Default::default() };
        let c = QwenDecoder.qwen_config(32, &tc);
        assert_eq!(c.n_layers, 2);
        assert_eq!(c.n_heads, 4);
        assert_eq!(c.n_kv_heads, 2); // GQA
        assert_eq!(c.head_dim, 16);
        assert_eq!(c.d_ff, 256);
        assert_eq!(c.block_size, 32);
        assert_eq!(c.vocab, 0);
    }
}
