// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The sparse Mixture-of-Experts decoder as a [`DecoderLm`] — the second
//! architecture the benchmark battery can score, alongside the dense
//! [`GptDecoder`](crate::GptDecoder).
//!
//! MoE training already implements the architecture-agnostic [`model::Model`]
//! seam (`moe::Trainer`), so [`train_decoder`](MoeDecoder::train_decoder) routes
//! through the *same* generic trainer (`model::train::fit`) the GPT baseline
//! uses — identical LR schedule / grad-accum / checkpoint semantics, only the
//! architecture differs. Inference (per-position logits) lives in the standalone
//! `moe::Engine` (the `Trainer` has no in-trainer token head), so the
//! [`Scorer`] wraps an `Engine` loaded from the `fit`-saved checkpoint; the two
//! share the checkpoint container format (tied `lm_head.weight` included), so a
//! `fit`-saved MoE checkpoint loads in the `Engine` directly.

use std::path::Path;

use model::FitOpts;
use moe::train::{Config as MoeConfig, Trainer};
use moe::Engine;

use crate::model::{DecoderLm, Scorer, TrainConfig};

/// MoE-specific layout choices the benchmark's depth/width/heads don't carry.
/// Kept here (not on `TrainConfig`, which stays architecture-neutral) so every
/// benchmark scores MoE at the same sparse shape. 4 experts / top-2 is the
/// established MoE default (matches `moe::train::train` and the validate ref);
/// `d_ff = 2 * d_model` keeps a single expert's FFN comparable to GPT's `4*d`
/// dense MLP split across the routed experts.
const N_EXPERTS: u32 = 4;
const TOP_K: u32 = 2;
/// MoE's aux/z router-regularization coefficients (the validated defaults).
const AUX_COEF: f32 = 0.01;
const Z_COEF: f32 = 1e-4;

/// The sparse-MoE Transformer as a [`DecoderLm`].
#[derive(Clone, Debug, Default)]
pub struct MoeDecoder;

impl MoeDecoder {
    /// Build a MoE [`Config`](MoeConfig) from the architecture-neutral
    /// [`TrainConfig`]. `n_layers`/`d_model`/`n_heads` map directly to the MoE
    /// depth/width/heads; `n_experts`/`top_k`/`d_ff`/`aux`/`z` use MoE defaults.
    /// `vocab` is `0` (the trainer infers it from the dataset's `meta.json`).
    fn moe_config(&self, block_size: u32, cfg: &TrainConfig) -> MoeConfig {
        MoeConfig {
            vocab: 0, // inferred from the dataset
            block_size,
            n_layers: cfg.n_layers,
            d_model: cfg.d_model,
            n_heads: cfg.n_heads,
            n_experts: N_EXPERTS,
            top_k: TOP_K,
            d_ff: cfg.d_model * 2,
            aux_coef: AUX_COEF,
            z_coef: Z_COEF,
        }
    }
}

impl DecoderLm for MoeDecoder {
    fn arch_name(&self) -> &'static str {
        "moe"
    }

    fn train_decoder(
        &self,
        dir: &Path,
        block_size: u32,
        cfg: &TrainConfig,
        weights_out: &Path,
    ) -> std::io::Result<(f32, f32)> {
        let mcfg = self.moe_config(block_size, cfg);
        // Same TrainConfig -> FitOpts mapping the GPT baseline uses, so the only
        // difference between a GPT and a MoE eval run is the architecture.
        let opts = FitOpts {
            steps: cfg.steps,
            batch_size: cfg.batch_size,
            block_size,
            lr: cfg.lr,
            warmup: 20,
            decay_iters: cfg.steps * 2, // stop mid-cosine, don't crater LR early
            eval_interval: 0,
            seed: cfg.seed,
            mask_before: cfg.mask_before,
            mask_per_line: cfg.mask_per_line,
            align_to_lines: cfg.align_to_lines,
            ..Default::default()
        };
        // `Trainer` implements `model::Model`, so the generic trainer trains MoE.
        model::train::fit::<Trainer>(dir, mcfg, &opts, Some(weights_out))
    }

    fn load_scorer(&self, weights: &Path, _block_size: u32) -> Box<dyn Scorer> {
        // The `fit`-saved checkpoint is the inference engine's own container
        // (tied `lm_head.weight` included); the block_size is read from it.
        Box::new(MoeScorer {
            engine: Engine::load(weights.to_str().expect("utf-8 path")),
        })
    }
}

/// A trained MoE [`Engine`] exposed as a [`Scorer`]. Unlike the GPT baseline's
/// `ModelScorer` (which wraps the trainer's in-trainer token head), MoE's
/// inference is the standalone `Engine`, so scoring goes through it.
struct MoeScorer {
    engine: Engine,
}

impl Scorer for MoeScorer {
    fn vocab(&self) -> usize {
        self.engine.vocab_size() as usize
    }

    fn logits_all(&self, tokens: &[u32]) -> Vec<f32> {
        self.engine.logits_all(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_decoder_arch_name() {
        assert_eq!(MoeDecoder.arch_name(), "moe");
    }

    #[test]
    fn config_maps_size_and_uses_moe_defaults() {
        let tc = TrainConfig { n_layers: 3, d_model: 48, n_heads: 6, ..Default::default() };
        let c = MoeDecoder.moe_config(32, &tc);
        assert_eq!(c.n_layers, 3);
        assert_eq!(c.d_model, 48);
        assert_eq!(c.n_heads, 6);
        assert_eq!(c.n_experts, N_EXPERTS);
        assert_eq!(c.top_k, TOP_K);
        assert_eq!(c.d_ff, 96);
        assert_eq!(c.block_size, 32);
        assert_eq!(c.vocab, 0);
    }
}
