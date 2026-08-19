// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Architecture-agnostic model seam (ADR 0001).
//!
//! This crate defines the [`Model`]/[`ModelConfig`] traits — the union of the
//! surface `gpt2::Gpt`, `toymoe::Trainer`, and `toypid::Pid` already expose ad hoc -
//! plus the [`Batch`] input enum and [`Head`] objective marker, and one generic
//! trainer ([`train::fit`]) / sampler ([`train::generate`]) written over `Model`.
//!
//! Models (gpt/moe/pid/new seq2seq) implement [`Model`]; the generic trainer,
//! eval, and benches are written once against the trait. Per ADR §2, this crate
//! depends only on `gpu_core`, `paramstore`, `optim`, `kernels`, `checkpoint`,
//! and `data`.

use std::collections::HashMap;

pub mod actstats;
pub mod kvcalib;
pub mod attninject;
pub mod block;
pub mod dispatch;
pub mod collective;
pub mod paged;
pub mod rowemit;
pub mod serve;
#[cfg(not(target_arch = "wasm32"))]
pub mod distributed;
pub mod fp8;
pub mod gdn;
pub mod gdn_mixer;
pub mod gqa_mixer;
pub mod grid;
/// bf16 pack/unpack (B4's storage tier) - see this module's own doc comment.
pub mod half;
pub mod hostmath;
pub mod int4;
pub mod int8;
pub mod lora;
pub mod moe;
pub mod ops;
// wasm-gated like `distributed`/`parallel`/`shard`: a TCP transport has no
// business compiling into the browser build, and its `pub use` below broke
// the wasm build the crate declares support for (the re-exports referenced
// modules whose declarations WERE gated).
#[cfg(not(target_arch = "wasm32"))]
pub mod netcollective;
#[cfg(not(target_arch = "wasm32"))]
pub mod parallel;
pub mod plan;
#[cfg(not(target_arch = "wasm32"))]
pub mod shard;
pub mod train;
pub mod vit;
pub mod vlm;

pub use collective::{Collective, HostCollective};
#[cfg(not(target_arch = "wasm32"))]
pub use distributed::{federated_average, DdpOptimizer};
#[cfg(not(target_arch = "wasm32"))]
pub use netcollective::NetworkCollective;
pub use grid::{Coord, Grid, LocalGroups};
#[cfg(not(target_arch = "wasm32"))]
pub use parallel::DataParallel;
pub use plan::{plan_tp, Hardware, ModelShape, TpPlan};
#[cfg(not(target_arch = "wasm32"))]
pub use shard::{plan_balanced, Pipeline, Shard, ShardCost, Shardable};

pub use train::{cosine_lr, generate, FitOpts, IGNORE};
#[cfg(not(target_arch = "wasm32"))]
pub use train::{fit, load_dataset};

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
    /// Vision-language: a causal-LM text stream with pre-projected image-token
    /// embeddings spliced into the residual stream. `tokens`/`targets` are the
    /// full text stream (image-placeholder positions carry `IGNORE` targets so
    /// they never enter the loss); `image_embeds` is the row-major
    /// `[image_rows.len(), d_model]` block of vision tokens already projected to
    /// decoder width, written over the residual rows named by `image_rows` (one
    /// row index per spliced token). The vision encoder + connector produce
    /// `image_embeds`; the decoder's embedding stage overwrites those rows after
    /// the text gather, and its backward routes those rows' gradient to the
    /// connector instead of `tok.weight`. Richer per-model side channels
    /// (DeepStack levels, M-RoPE position ids, prefix length) travel through
    /// model-specific `set_*` methods, keeping this shared variant minimal.
    Multimodal { tokens: &'a [u32], targets: &'a [u32], image_embeds: &'a [f32], image_rows: &'a [u32] },
    /// Causal LM with a per-POSITION scalar weight on the cross-entropy
    /// gradient - the seam continuous/reward-driven training (STaR-style
    /// rejection sampling, GRPO-lite advantage weighting) composes on top of
    /// ordinary supervised [`Batch::Lm`] training, generic over any
    /// [`Head::TokenClassifier`] model. `weights.len()` must equal
    /// `tokens.len()`/`targets.len()`; a caller with only a per-EXAMPLE
    /// (whole-completion) reward broadcasts that single scalar across the
    /// completion's own token positions before calling `set_batch` - this
    /// variant itself has no notion of example boundaries, matching DAPO's
    /// finding that token-level (not sample-level) loss aggregation is the
    /// more robust default.
    ///
    /// A weight of `0.0` must produce exactly zero gradient contribution
    /// from that position (masked-out and reward=0 collapse to the same
    /// thing); a weight of `1.0` everywhere must reproduce [`Batch::Lm`]'s
    /// gradient bit-for-bit - both are asserted by each adopting model's
    /// gradcheck (e.g. `qwen3`'s `check_qwen3_weighted`).
    ///
    /// A model opts into this variant by constructing itself with weighted-
    /// loss support enabled (each model's own opt-in method, e.g.
    /// `qwen3::Qwen::enable_weighted_loss`, following the same
    /// allocate-buffer/rebuild-steps pattern as `enable_mrope`/
    /// `enable_mm_splice`) - NOT by every model handling this variant by
    /// default, so ordinary (unweighted) training pays zero extra kernel
    /// dispatches. A model that has not opted in may treat this like any
    /// other unsupported `Batch` variant (panic, matching the existing
    /// `Batch::Lm`-only models' own wildcard arm).
    LmWeighted { tokens: &'a [u32], targets: &'a [u32], weights: &'a [f32] },
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
    /// bare config (this is the model's own `init_weights`, e.g. `gpt2::init`).
    fn init_weights(cfg: &Self::Config, seed: u64) -> HashMap<String, Vec<f32>>
    where
        Self: Sized;

    /// Access the (typed) config this model was built from.
    fn config(&self) -> &Self::Config;

    /// Upload one batch (shape must match how the model was constructed).
    fn set_batch(&self, batch: Batch);

    /// Opt into per-position weighted-loss training ([`Batch::LmWeighted`]) -
    /// call once after construction, before the first [`Model::backward`].
    /// Default panics: a model must explicitly override this to support
    /// `crates/rl`'s generic weighted-training driver - the same opt-in,
    /// thin-per-model-wiring shape [`Batch::LmWeighted`]'s own doc comment
    /// documents. `qwen3::Qwen::enable_weighted_loss` is the reference
    /// implementation this delegates to.
    fn enable_weighted_loss(&mut self) {
        unimplemented!("{}: does not implement Model::enable_weighted_loss (no weighted-loss/Batch::LmWeighted support yet)", std::any::type_name::<Self>());
    }

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
