// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Architecture-agnostic model seam (ADR 0001).
//!
//! Swedish Embedded AB implements training and inference stacks that do not
//! have to be rewritten for each new architecture, for teams whose model
//! roadmap is longer than one model. If your team needs expertise in building
//! an engine where a new architecture is composed from existing pieces instead
//! of forked from the last one, you can procure our services by sending an
//! email to info@swedishembedded.com.
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
pub mod adapter;
pub mod anchor_memory;
pub mod kv_offload;
pub mod layer_stream;
pub mod offload;
pub mod kv_tier;
pub mod kvcalib;
pub mod memplan;
pub mod attninject;
pub mod block;
pub mod dispatch;
pub mod collective;
pub mod acceptance;
pub mod conformance;
pub mod conformance_adapter;
pub mod conformance_grad;
pub mod conformance_head;
#[cfg(not(target_arch = "wasm32"))]
pub mod finetune;
pub mod flow;
pub mod session;
pub mod paged;
pub mod policy;
pub mod readout;
pub mod rowemit;
pub mod serve;
pub mod snapshot_prefix;
#[cfg(not(target_arch = "wasm32"))]
pub mod distributed;
#[cfg(not(target_arch = "wasm32"))]
pub mod job;
pub mod fp8;
pub mod gdn;
pub mod gdn_mixer;
pub mod gdn_segment;
pub mod gdn_verify;
pub mod gqa_mixer;
pub mod grid;
pub mod head_ce;
/// bf16 pack/unpack (B4's storage tier) - see this module's own doc comment.
pub mod half;
pub mod eigen;
pub mod hostmath;
pub mod int4;
pub mod int8;
pub mod kquant;
pub mod logitscale;
pub mod logprobs;
pub mod lora;
pub mod lossw;
pub mod lut4;
pub mod moe;
pub mod moe_i8;
pub mod objective;
#[cfg(not(target_arch = "wasm32"))]
pub mod score;
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
pub mod projector;
/// Real per-device, per-dtype GEMM throughput measurement - see the module doc.
pub mod probe;
pub mod hostopt;
pub mod resident_ft;
pub mod rollout;
pub mod saved;
#[cfg(not(target_arch = "wasm32"))]
pub mod shard;
pub mod streamed;
pub mod tape;
pub mod train;
pub mod trainprec;
pub mod training;
pub mod vit;
pub mod vlm;
pub mod rope_scaling;
pub mod yarn;

pub use collective::{Collective, HostCollective};
pub use collective::{payload_bytes, CollectiveProfile, OpKind, OpSnapshot, ShapedCollective, SimulatedLink, Snapshot, TimedCollective};
pub use optim::{grad_multiplier, Adam};
#[cfg(not(target_arch = "wasm32"))]
pub use distributed::{federated_average, reduce_grads, DdpOptimizer};
#[cfg(not(target_arch = "wasm32"))]
pub use netcollective::NetworkCollective;
#[cfg(not(target_arch = "wasm32"))]
pub use job::{CheckpointCommit, CommitFile, JobContext, JobPlan};
pub use grid::{Coord, Grid, LocalGroups};
#[cfg(not(target_arch = "wasm32"))]
pub use parallel::DataParallel;
pub use plan::{plan_tp, Hardware, ModelShape, TpPlan};
#[cfg(not(target_arch = "wasm32"))]
pub use shard::{plan_balanced, Pipeline, PipelineModel, Shard, ShardCost, Shardable, StreamPlan};

pub use objective::{Objective, StepVerdict};
pub use trainprec::{apply_train_precision, OptimizerParts, OptimizerState, ResidentRequest, StateSupport, TrainPrecision, TrainPrecisionSupport};
#[cfg(not(target_arch = "wasm32"))]
pub use score::{score_base, score_dataset, score_tuned, supervised_positions, ScoredSet};
pub use train::{cosine_lr, generate, FitOpts, LrSchedule, IGNORE};
#[cfg(not(target_arch = "wasm32"))]
pub use train::{build_or_resume, causal_lm, fit, fit_controlled, fit_controlled_sharded, fit_controlled_sync, fit_distributed, fit_from, fit_with, load_dataset, load_dataset_with_itos, AdapterSnapshots, CheckpointIdentity, FitControl, FitReport, GradSync, StepReport, StepWork};

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
    /// A model opts into this variant through
    /// [`training::TokenObjectives::enable_weighted_loss`] (e.g.
    /// `qwen3::Qwen::enable_weighted_loss` / `gpt2::Gpt::enable_weighted_loss`,
    /// both thin wrappers over the shared [`lossw::WeightedCe`] primitive,
    /// following the same allocate-buffer/rebuild-steps pattern as
    /// `enable_mrope`/`enable_mm_splice`) - NOT by every model handling this
    /// variant by default, so ordinary (unweighted) training pays zero extra
    /// kernel dispatches. A model that has not opted in may treat this like
    /// any other unsupported `Batch` variant (panic, matching the existing
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

/// Training state a model keeps off the device and commits transactionally: one generation per optimizer step,
/// each atomically published, so a resume names the step it continues from and gets exactly that state or a
/// refusal.
pub trait OffDeviceState {
    /// Atomically commit the state as of optimizer step `step` (the number of updates absorbed).
    fn commit(&self, dir: &std::path::Path, step: u32) -> std::io::Result<()>;
    /// Restore the state committed at exactly `step` into the model, or fail when no such generation exists.
    fn restore(&self, dir: &std::path::Path, step: u32) -> std::io::Result<()>;
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

    /// Build for FORWARD passes only, sized for batch `b` x seq `t`.
    ///
    /// [`Model::new`] builds the TRAINING shape: backward scratch, and one
    /// copy of every per-layer activation, because a backward pass reads
    /// every layer's. A caller that only ever asks for [`Model::logits_all`]
    /// (scoring, evaluation, rollout) pays for both and reads neither, and
    /// on a real model that is most of the allocation rather than a margin.
    ///
    /// Defaults to [`Model::new`], so a model with no separate inference
    /// shape is correct without implementing it, and an implementation that
    /// overrides it must return the same numbers: this is an allocation
    /// decision, never a numerical one.
    fn new_inference(cfg: Self::Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self
    where
        Self: Sized,
    {
        Self::new(cfg, b, t, init)
    }

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

    /// Tokens the most recent [`Self::forward`] actually computed, when the
    /// model trims a batch to what its targets need (a causal model computes
    /// nothing past the last supervised position of a row): rows times the run
    /// length each row got. `None` (the default) means the model computes the
    /// whole batch, which is what the step accounting
    /// ([`crate::StepWork`]) then assumes.
    fn computed_tokens(&self) -> Option<u64> {
        None
    }

    /// Forward and backward every batch of `batches` as one accumulation,
    /// the gradients adding up, and return the summed loss: for a model that
    /// can overlap the micro-batches (a pipeline's stages working on
    /// neighbouring ones at once). `None` (the default) means it cannot, and
    /// the caller runs them one after another with [`Model::set_batch`],
    /// [`Model::forward`] and [`Model::backward`].
    fn accumulate_overlapped(&self, _batches: &[Batch]) -> Option<f32> {
        None
    }

    /// The per-token objective capability ([`training::TokenObjectives`]):
    /// weighted loss, per-token log-probabilities, loss denominator and logit
    /// scale. `None` (the default) means the model does not provide it; an
    /// objective that needs it refuses with [`training::Unsupported`] when it
    /// is configured. Read-only view, for the calls a forward/backward pair
    /// makes through `&self`.
    fn token_objectives(&self) -> Option<&dyn training::TokenObjectives> {
        None
    }

    /// [`Model::token_objectives`] for the calls that reconfigure the model
    /// (`enable_weighted_loss`, `set_logit_scale`).
    fn token_objectives_mut(&mut self) -> Option<&mut dyn training::TokenObjectives> {
        None
    }

    /// Run forward; return the scalar objective loss that `backward` differentiates.
    fn forward(&self) -> f32;
    /// Accumulate analytic gradients for the current batch into the ParamStore.
    fn backward(&self);
    fn zero_grads(&self);

    /// One AdamW step with the run's `adam` hyperparameters (with optional
    /// global-norm clip and a grad-accum scale).
    fn adamw_step(&self, t: u32, lr: f32, wd: f32, adam: Adam, clip: Option<f32>, extra_scale: f32);

    /// The global gradient norm the latest clipped [`Model::adamw_step`] measured, after its
    /// `extra_scale` (the norm the clip bounds). `None` (the default) for a model that does not
    /// measure it or a step that did not clip.
    fn last_grad_norm(&self) -> Option<f32> {
        None
    }

    /// Block until submitted device work completes (memory-aperture hygiene).
    fn poll_wait(&self);

    /// Key the stochastic rounding of the optimizer state on the run's `seed` (a model whose optimizer rounds
    /// stochastically draws from it), so a run resumed with the same seed replays the draws of the original. The
    /// default is for a model with no stochastic state.
    fn pin_rounding_seed(&self, _seed: u64) {}

    // ---- optimizer state (exact resume) ----
    /// The parameters [`Model::adamw_step`] optimises on the device, when
    /// this model can hand over and take back their AdamW moments through
    /// [`Model::read_moments`]/[`Model::write_moments`] - what
    /// [`crate::train::fit_controlled`] needs to resume a run exactly rather
    /// than with its moments reset. `None` (the default) for a model that
    /// cannot, including one whose moments live off the device.
    fn optimized_params(&self) -> Option<Vec<String>> {
        // The device-optimised parameters of the store; offloaded ones keep their state in the host optimizer, which
        // the fit loop commits beside the resume file.
        self.optimizer_parts().map(|p| p.store.opt_params().iter().map(|(n, _)| n.clone()).collect())
    }

    /// Training state kept off the device that a resume must restore beside [`Model::optimized_params`]:
    /// master copies and optimizer state of the parameters trained against them (resident low-precision full
    /// fine-tuning). `None` (the default) when the model has none.
    fn off_device_state(&self) -> Option<&dyn OffDeviceState> {
        None
    }

    /// The parameter store and the device optimizer [`Model::adamw_step`] updates it with, so the run's optimizer-state
    /// precision ([`FitOpts::precision`](crate::train::FitOpts)) reaches any model through one generic path
    /// ([`trainprec::apply_train_precision`], called by the fit loop before the first step). `None` (the default) for
    /// a model that keeps no such pair; such a model refuses a non-default optimizer-state precision by name.
    fn optimizer_state(&mut self) -> Option<OptimizerState<'_>> {
        None
    }

    /// Stream this model's per-layer tensors through a device window under `cfg` (a full fine-tune whose weights,
    /// their f32 values on the host and their optimizer state there, need not all fit the device; see
    /// [`streamed::StreamedStore`]). Called before the first forward when the run asks for resident weights
    /// (`weights=`); the model's [`Model::off_device_state`] then reports the streamed state. The default refuses by
    /// name, never training every layer on the device instead.
    fn set_resident(&mut self, _cfg: &resident_ft::ResidentFtConfig) -> Result<(), String> {
        Err("this model has no layer-streamed full fine-tune".into())
    }

    /// Hold this model's optimizer state as `precision` names when its parameters and its update live on the host
    /// ([`hostopt::HostAdam`]): the fit loop calls it for a model with no device [`OptimizerState`] when the run asks
    /// for a state other than f32. The default refuses by name, never training with f32 moments instead.
    fn set_host_optimizer(&mut self, _precision: paramstore::precision::ParamPrecision) -> Result<(), String> {
        Err("this model does not expose its optimizer state, so it trains only with f32 moments".into())
    }

    /// [`Model::new`] for a run that streams this model's layers under `resident` from the start: a model that
    /// builds its layers straight into the window (`paramstore::Role::Streamed`, [`streamed::StreamedStore::attach`])
    /// never holds them all on the device, so it trains a model larger than the device. The default builds every
    /// layer and then streams them ([`Model::set_resident`]): the window bounds what training adds, not the load.
    fn new_resident(cfg: Self::Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, resident: &resident_ft::ResidentFtConfig) -> Result<Self, String>
    where
        Self: Sized,
    {
        let mut model = Self::new(cfg, b, t, init);
        model.set_resident(resident)?;
        Ok(model)
    }

    /// Keep what `policy` names of the forward for the backward ([`saved::SavedPolicy`]): every layer's intermediates,
    /// or the residual between layers with each layer replayed before its backward. Called before the first forward
    /// (the fit loop applies the run's `activations=` precision). The default keeps everything and refuses any other
    /// policy by name, never ignoring it.
    fn set_saved_policy(&mut self, policy: saved::SavedPolicy) -> Result<(), String> {
        match policy {
            saved::SavedPolicy::Keep => Ok(()),
            other => Err(format!("this model keeps every layer's activations; {other:?} is not supported")),
        }
    }

    /// [`Model::optimizer_state`] read-only: what the fit loop commits and restores the host-held optimizer state of
    /// an offloaded store through, for a model that keeps no [`OffDeviceState`] of its own.
    fn optimizer_parts(&self) -> Option<OptimizerParts<'_>> {
        None
    }

    /// `name`'s AdamW `(m, v)` moments; `None` when it has none to give.
    fn read_moments(&self, name: &str) -> Option<(Vec<f32>, Vec<f32>)> {
        self.optimizer_parts().and_then(|p| optim::read_moments(p.gpu, p.store, name))
    }

    /// Restore `name`'s AdamW moments, as [`Model::read_moments`] returned them.
    fn write_moments(&self, name: &str, m: &[f32], v: &[f32]) -> Result<(), String> {
        let p = self.optimizer_parts().ok_or_else(|| format!("{name}: this model does not expose its optimizer state"))?;
        optim::write_moments(p.gpu, p.store, name, m, v)
    }

    // ---- parameter access (also satisfies gradcheck::CheckModel) ----
    fn param_names(&self) -> Vec<String>;

    /// Adapter tensors the model holds but does not optimise (LoRA-FA's `A`):
    /// they have no gradient, so [`Self::param_names`] omits them, yet a saved
    /// adapter is incomplete without them. None unless the model can freeze
    /// part of an adapter.
    fn frozen_adapter_param_names(&self) -> Vec<String> {
        Vec::new()
    }

    /// The linears an adapter may correct, with their dimensions: what a saved
    /// adapter records so a reader can shape factors that carry no shape of
    /// their own (LoKr's). Empty for a model without adapter sites.
    fn adapter_sites(&self) -> Vec<adapter::LinearSite> {
        Vec::new()
    }

    fn read_weight(&self, name: &str) -> Vec<f32>;
    fn write_weight(&self, name: &str, data: &[f32]);
    fn read_grad(&self, name: &str) -> Vec<f32>;

    /// Replace `name`'s gradient, as a data-parallel reduction needs to. The
    /// default refuses; a model that cannot expose its gradients cannot be
    /// trained through a [`crate::Collective`].
    fn write_grad(&self, name: &str, _data: &[f32]) -> Result<(), String> {
        Err(format!("{name}: this model does not expose its gradients for a reduction"))
    }

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
