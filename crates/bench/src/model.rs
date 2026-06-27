// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The model-agnostic seam: a [`DecoderLm`] trait that abstracts the *one*
//! capability every registered benchmark needs from an architecture — train a
//! causal next-token decoder on a token dataset, then read per-position logits
//! to score it.
//!
//! Before this trait, each benchmark's `evaluate` hard-coded the GPT baseline:
//! build a `GptConfig` + `TrainOpts`, call [`gpt::train`], `Gpt::load`, then
//! `logits_all`. That boilerplate was duplicated across MQAR and every MAD
//! benchmark, and named "GPT" inside otherwise model-agnostic scoring code. Now
//! a benchmark depends only on [`DecoderLm`]: it hands over a [`TrainConfig`] and
//! a dataset directory, gets back a [`Scorer`], and asks it for logits. Swapping
//! in a different architecture (a MoE or PID decoder) is a new `DecoderLm` impl —
//! no benchmark changes.
//!
//! Scope, honestly: this abstracts a **causal next-token decoder LM** — the
//! common shape of every *registered* benchmark. It deliberately does **not** try
//! to model non-next-token objectives (e.g. the autoencoder bottleneck the
//! [`mad_compress`](crate::mad_compress) task needs); that requires a different
//! training path and is out of scope here.

use std::path::Path;

use data::tokenizer::{CharTokenizer, Tokenizer};
use gpt::model::Gpt;
use gpt::GptConfig;
use model::{FitOpts, Model, ModelConfig};

/// Architecture-independent training spec a benchmark hands to a [`DecoderLm`].
///
/// `block_size` is supplied separately to [`DecoderLm::train_decoder`] because it
/// is a property of the dataset (the benchmark's sequence length), not a tunable.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub steps: u32,
    pub batch_size: u32,
    pub lr: f32,
    /// Depth / width / heads requested of the architecture (an architecture may
    /// interpret these in its own terms; the GPT baseline maps them directly).
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    /// Mask loss up to & including this char (the answer-masking recipe). `None`
    /// trains on every position.
    pub mask_before: Option<char>,
    pub mask_per_line: bool,
    pub align_to_lines: bool,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            steps: 600,
            batch_size: 32,
            lr: 3e-3,
            n_layers: 2,
            d_model: 64,
            n_heads: 4,
            mask_before: None,
            mask_per_line: false,
            align_to_lines: false,
            seed: 1337,
        }
    }
}

/// A trained decoder, queried by benchmarks for per-position logits during
/// scoring. One instance corresponds to one trained checkpoint.
pub trait Scorer {
    /// Vocabulary size (logits row width).
    fn vocab(&self) -> usize;

    /// Logits for every position of a single sequence, flattened row-major as
    /// `[seq_len * vocab]`. `logits[t*vocab + i]` is the score for token `i` at
    /// position `t`; the next-token distribution at `t` predicts position `t+1`.
    fn logits_all(&self, tokens: &[u32]) -> Vec<f32>;

    /// Greedily decode the predicted token at the position **following** the
    /// prefix `tokens` (argmax of the last logits row). A convenience for the
    /// common "what does the model put here?" scoring step.
    fn predict_next(&self, tokens: &[u32]) -> u32 {
        let v = self.vocab();
        let logits = self.logits_all(tokens);
        let last = &logits[logits.len() - v..];
        argmax(last) as u32
    }
}

/// An architecture that can be trained as a causal next-token decoder and then
/// scored. The GPT baseline is [`GptDecoder`]; a MoE/PID decoder would be a new
/// impl, immediately usable by every benchmark.
pub trait DecoderLm {
    /// Short architecture identifier (e.g. `"gpt"`), for table labels / logs.
    fn arch_name(&self) -> &'static str;

    /// Train a fresh decoder on the token dataset in `dir` (brain's
    /// `train.bin`/`val.bin`/`meta.json` layout) with sequence length
    /// `block_size`, writing a checkpoint to `weights_out`. Returns the
    /// `(initial_ce, final_ce)` training cross-entropy (nats).
    fn train_decoder(
        &self,
        dir: &Path,
        block_size: u32,
        cfg: &TrainConfig,
        weights_out: &Path,
    ) -> std::io::Result<(f32, f32)>;

    /// Load a [`Scorer`] from a checkpoint written by
    /// [`train_decoder`](DecoderLm::train_decoder), sized for single-sequence
    /// scoring at sequence length `block_size`.
    fn load_scorer(&self, weights: &Path, block_size: u32) -> Box<dyn Scorer>;
}

/// The dense GPT baseline as a [`DecoderLm`] — wraps [`gpt::train`] / [`Gpt`].
/// This is the default architecture every benchmark uses today.
#[derive(Clone, Debug, Default)]
pub struct GptDecoder;

impl DecoderLm for GptDecoder {
    fn arch_name(&self) -> &'static str {
        "gpt"
    }

    fn train_decoder(
        &self,
        dir: &Path,
        block_size: u32,
        cfg: &TrainConfig,
        weights_out: &Path,
    ) -> std::io::Result<(f32, f32)> {
        let gcfg = GptConfig {
            vocab: 0, // inferred from the dataset's meta.json
            block_size,
            n_layers: cfg.n_layers,
            d_model: cfg.d_model,
            n_heads: cfg.n_heads,
            d_ff: cfg.d_model * 4,
        };
        // Route through the architecture-agnostic generic trainer (ADR §2.4): the
        // `TrainConfig` -> `FitOpts` mapping below is the same hyperparameters the
        // GPT-specific `gpt::train` used, so the benchmark behavior is unchanged.
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
        model::train::fit::<Gpt>(dir, gcfg, &opts, Some(weights_out))
    }

    fn load_scorer(&self, weights: &Path, block_size: u32) -> Box<dyn Scorer> {
        Box::new(ModelScorer {
            model: Gpt::load(weights.to_str().expect("utf-8 path"), 1, block_size),
        })
    }
}

/// A trained [`Model`] with a token head, exposed as a [`Scorer`] (ADR §2.4).
/// One blanket adapter replaces the former per-model `GptScorer`: swapping the
/// architecture a benchmark scores is choosing a different `M: Model` here.
struct ModelScorer<M: Model> {
    model: M,
}

impl<M: Model> Scorer for ModelScorer<M> {
    fn vocab(&self) -> usize {
        self.model.config().vocab() as usize
    }

    fn logits_all(&self, tokens: &[u32]) -> Vec<f32> {
        Model::logits_all(&self.model, tokens).expect("token head")
    }
}

/// Greedily decode characters until a newline (exclusive) or `max_new` reached,
/// starting from `prompt`. A shared helper for exact-match-style scoring across
/// benchmarks (kept here so it works against any [`Scorer`]).
pub fn greedy_until_newline(
    scorer: &dyn Scorer,
    prompt: &[u32],
    max_new: usize,
    tok: &CharTokenizer,
    block: usize,
) -> String {
    let mut ctx = prompt.to_vec();
    let mut out = String::new();
    for _ in 0..max_new {
        let window: Vec<u32> =
            if ctx.len() > block { ctx[ctx.len() - block..].to_vec() } else { ctx.clone() };
        let next = scorer.predict_next(&window);
        let ch = tok.decode(&[next]);
        if ch == "\n" {
            break;
        }
        out.push_str(&ch);
        ctx.push(next);
    }
    out
}

pub(crate) fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) })
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_config_defaults() {
        let c = TrainConfig::default();
        assert_eq!(c.n_layers, 2);
        assert_eq!(c.d_model, 64);
        assert!(c.mask_before.is_none());
    }

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[5.0, -1.0, 2.0]), 0);
    }

    #[test]
    fn gpt_decoder_arch_name() {
        assert_eq!(GptDecoder.arch_name(), "gpt");
    }
}
