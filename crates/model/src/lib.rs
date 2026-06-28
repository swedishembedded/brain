// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Architecture-agnostic model seam (ADR 0001).
//!
//! This crate defines the [`Model`]/[`ModelConfig`] traits — the union of the
//! surface `gpt::Gpt`, `moe::Trainer`, and `pid::Pid` already expose ad hoc —
//! plus the [`Batch`] input enum and [`Head`] objective marker, and one generic
//! trainer ([`train::fit`]) / sampler ([`train::generate`]) written over `Model`.
//!
//! Models (gpt/moe/pid/new seq2seq) implement [`Model`]; the generic trainer,
//! eval, and benches are written once against the trait. Per ADR §2, this crate
//! depends only on `gpu_core`, `paramstore`, `optim`, `kernels`, `checkpoint`,
//! and `data`.

use std::collections::HashMap;

pub mod block;
pub mod train;

pub use train::{cosine_lr, generate, FitOpts, IGNORE};
#[cfg(not(target_arch = "wasm32"))]
pub use train::fit;

/// What a batch looks like for a given model. Decoder-LM and seq2seq differ in
/// whether there is a separate source sequence; this enum keeps `set_batch`
/// uniform without forcing every model to accept encoder inputs it ignores.
pub enum Batch<'a> {
    /// Causal LM / single-stream: `targets[t]` predicts position `t+1`; masked
    /// (IGNORE) positions are dropped from the loss.
    Lm { tokens: &'a [u32], targets: &'a [u32] },
    /// Encoder-decoder: source feeds the encoder, target feeds the decoder,
    /// labels are the decoder's next-token targets (IGNORE masks padding).
    Seq2Seq { src: &'a [u32], tgt: &'a [u32], labels: &'a [u32] },
    /// Non-LM: float inputs + float targets (autoencoder reconstruction,
    /// regression). `tokens` is optional (e.g. token-id inputs reconstructed
    /// against themselves).
    Tensor { tokens: Option<&'a [u32]>, inputs: &'a [f32], targets: &'a [f32] },
}

/// The objective head + loss a model's final stage realizes (ADR §2.3). Selected
/// by config; chooses the loss kernel (CE for [`Head::TokenClassifier`], MSE for
/// [`Head::Regression`]).
#[derive(Clone, Copy, Debug)]
pub enum Head {
    /// Untied (or tied) projection to `vocab` + masked cross-entropy.
    /// Used by GPT, MoE, PID (u_bins), and the seq2seq decoder.
    TokenClassifier { vocab: u32, tied: bool },
    /// Project to `out_dim` floats + MSE. Used by the MAD compression
    /// autoencoder and future regression heads.
    Regression { out_dim: u32 },
}

/// Config behaviour shared by all models: param layout + (de)serialization.
pub trait ModelConfig: Clone {
    fn param_list(&self) -> Vec<(String, usize)>;
    fn to_json(&self) -> serde_json::Value;
    fn from_json(v: &serde_json::Value) -> Self
    where
        Self: Sized;
    fn vocab(&self) -> u32;
    fn block_size(&self) -> u32;

    /// Override `vocab`/`block_size` from the dataset + run options and apply any
    /// derived defaults (e.g. GPT's `4*d_model` feed-forward width). The generic
    /// trainer calls this on a fresh (non-resume) start, replacing the per-model
    /// `cfg.vocab = …; cfg.block_size = …; cfg.with_ff_default()` lines.
    fn finalize_for_dataset(self, vocab: u32, block_size: u32) -> Self
    where
        Self: Sized;
}

/// The primary model seam (ADR §2.2): the union of the forward/backward/param/
/// save surface every model exposes, normalized to one signature set and
/// extended with `gradcheck::CheckModel`'s requirements (so every `Model` is
/// gradient-checkable by construction).
pub trait Model {
    type Config: ModelConfig;

    /// Build from a config + initial weights, sized for batch `b` × seq `t`
    /// (and, for seq2seq, `t_kv` = encoder length via the config).
    fn new(cfg: Self::Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self
    where
        Self: Sized;

    /// Architecture-specific fresh weight initialization, deterministic for a
    /// fixed `seed`. The generic trainer needs this to construct a model from a
    /// bare config (this is the model's own `init_weights`, e.g. `gpt::init`).
    fn init_weights(cfg: &Self::Config, seed: u64) -> HashMap<String, Vec<f32>>
    where
        Self: Sized;

    /// Access the (typed) config this model was built from.
    fn config(&self) -> &Self::Config;

    /// Upload one batch (shape must match how the model was constructed).
    fn set_batch(&self, batch: Batch);

    /// Run forward; return the scalar objective loss that `backward` differentiates.
    fn forward(&self) -> f32;
    /// Accumulate analytic gradients for the current batch into the ParamStore.
    fn backward(&self);
    fn zero_grads(&self);

    /// One AdamW step (with optional global-norm clip and a grad-accum scale).
    fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32);

    /// Block until submitted device work completes (memory-aperture hygiene).
    fn poll_wait(&self);

    // ---- parameter access (also satisfies gradcheck::CheckModel) ----
    fn param_names(&self) -> Vec<String>;
    fn read_weight(&self, name: &str) -> Vec<f32>;
    fn write_weight(&self, name: &str, data: &[f32]);
    fn read_grad(&self, name: &str) -> Vec<f32>;

    // ---- inference / scoring ----
    /// Per-position logits for one sequence, row-major `[len * vocab]` (decoder
    /// models). Returns `None` for models without a token-classification head
    /// (e.g. a pure regression autoencoder).
    fn logits_all(&self, tokens: &[u32]) -> Option<Vec<f32>>;

    // ---- persistence ----
    fn save(&self, path: &str);

    /// Save the checkpoint, optionally embedding a char-tokenizer vocab (`itos`)
    /// in the manifest so inference needs no dataset reference. The generic
    /// trainer calls this so char-dataset checkpoints stay self-describing.
    /// Default ignores `itos` (token-id models with no char vocab).
    fn save_with_itos(&self, path: &str, _itos: Option<&[char]>) {
        self.save(path);
    }

    fn config_json(&self) -> serde_json::Value;
}
