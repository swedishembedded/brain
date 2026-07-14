// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The GLM-5.2 (`glm_moe_dsa`) decoder as a [`DecoderLm`] — MLA attention + a
//! sigmoid `noaux_tc` MoE (shared expert + dense→MoE schedule), scored by the
//! same benchmark battery as the dense/GQA/MoE baselines.
//!
//! `glm::Glm` implements the architecture-agnostic [`model::Model`] seam and
//! exposes `logits_all` on the trainer itself, so training routes through the
//! generic `model::train::fit` and scoring loads an inference-only (frozen)
//! instance from the saved checkpoint — exactly like the Qwen decoder.
//!
//! Benchmark configs are small; the MLA head split and MoE shape are derived
//! from the requested depth/width/heads (with `index_topk >= block_size`, so the
//! DSA indexer is a no-op and attention is exact dense MLA).

use std::path::Path;

use glm::config::GlmConfig;
use glm::model::Glm;
use model::FitOpts;

use crate::model::{DecoderLm, Scorer, TrainConfig};

/// GLM-specific layout the architecture-neutral [`TrainConfig`] doesn't carry.
const N_EXPERTS: u32 = 4;
const TOP_K: u32 = 2;
const N_SHARED: u32 = 1;
const FIRST_K_DENSE: u32 = 1;

/// The GLM-5.2 decoder as a [`DecoderLm`].
#[derive(Clone, Debug, Default)]
pub struct GlmDecoder;

impl GlmDecoder {
    /// Build a [`GlmConfig`] from the neutral [`TrainConfig`]. The MLA head split
    /// (`qk_nope`/`qk_rope`/`v_head_dim`) and low-rank ranks are derived from
    /// `d_model`/`n_heads`; the MoE uses fixed small defaults. `vocab = 0`
    /// (inferred from the dataset's `meta.json`). `index_topk` is set past
    /// `block_size` so the indexer is a no-op (dense MLA).
    pub fn glm_config(&self, block_size: u32, cfg: &TrainConfig) -> GlmConfig {
        let hd = (cfg.d_model / cfg.n_heads).max(2);
        let qk_rope = ((hd / 2).max(1) & !1).max(2); // even, >= 2
        let moe_ff = cfg.d_model * 2;
        GlmConfig {
            vocab: 0,
            block_size,
            n_layers: cfg.n_layers,
            d_model: cfg.d_model,
            n_heads: cfg.n_heads,
            q_lora_rank: cfg.d_model,
            kv_lora_rank: (cfg.d_model / 2).max(2),
            qk_nope_head_dim: hd,
            qk_rope_head_dim: qk_rope,
            v_head_dim: hd,
            n_routed_experts: N_EXPERTS,
            n_shared_experts: N_SHARED,
            num_experts_per_tok: TOP_K,
            moe_intermediate_size: moe_ff,
            intermediate_size: moe_ff,
            first_k_dense_replace: FIRST_K_DENSE.min(cfg.n_layers),
            n_group: 1,
            topk_group: 1,
            routed_scaling_factor: 2.5,
            norm_topk_prob: true,
            rope_theta: 1.0e4,
            rms_eps: 1e-5,
            tie_embeddings: false,
            index_topk: block_size + 1, // >= seq ⇒ dense (no-op indexer)
            index_n_heads: 2,
            index_head_dim: hd,
            indexer_full: Vec::new(),
        }
    }
}

impl DecoderLm for GlmDecoder {
    fn arch_name(&self) -> &'static str {
        "glm"
    }

    fn train_decoder(
        &self,
        dir: &Path,
        block_size: u32,
        cfg: &TrainConfig,
        weights_out: &Path,
    ) -> std::io::Result<(f32, f32)> {
        let gcfg = self.glm_config(block_size, cfg);
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
        model::train::fit::<Glm>(dir, gcfg, &opts, Some(weights_out))
    }

    fn load_scorer(&self, weights: &Path, block_size: u32) -> Box<dyn Scorer> {
        Box::new(GlmScorer {
            model: Glm::load_inference(weights.to_str().expect("utf-8 path"), 1, block_size),
        })
    }
}

struct GlmScorer {
    model: Glm,
}

impl Scorer for GlmScorer {
    fn vocab(&self) -> usize {
        self.model.cfg.vocab as usize
    }
    fn logits_all(&self, tokens: &[u32]) -> Vec<f32> {
        Glm::logits_all(&self.model, tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glm_decoder_arch_name_and_config() {
        assert_eq!(GlmDecoder.arch_name(), "glm");
        let tc = TrainConfig { n_layers: 2, d_model: 64, n_heads: 4, ..Default::default() };
        let c = GlmDecoder.glm_config(32, &tc);
        assert_eq!(c.n_layers, 2);
        assert_eq!(c.n_heads, 4);
        assert_eq!(c.qk_nope_head_dim, 16);
        assert_eq!(c.v_head_dim, 16);
        assert_eq!(c.qk_rope_head_dim % 2, 0);
        assert_eq!(c.n_routed_experts, N_EXPERTS);
        assert_eq!(c.first_k_dense_replace, 1);
        assert!(c.index_topk > 32); // dense (no-op indexer)
        assert_eq!(c.vocab, 0);
    }
}
