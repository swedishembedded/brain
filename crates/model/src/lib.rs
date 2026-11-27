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

pub mod act_quant;
pub mod actstats;
pub mod adapter;
pub mod anchor_memory;
pub mod kv_offload;
pub mod layer_stream;
pub mod offload;
pub mod kv_tier;
pub mod kvcalib;
pub mod memplan;
pub mod message_gate;
pub mod attninject;
pub mod block;
pub mod dispatch;
pub mod collective;
pub mod depth_attn;
pub mod acceptance;
pub mod conformance;
pub mod conformance_adapter;
#[cfg(not(target_arch = "wasm32"))]
pub mod conformance_faults;
pub mod conformance_grad;
pub mod conformance_head;
pub mod conformance_trainable;
#[cfg(not(target_arch = "wasm32"))]
pub mod conformance_zero;
#[cfg(not(target_arch = "wasm32"))]
pub mod finetune;
pub mod flash_ext;
pub mod flow;
pub mod session;
pub mod paged;
pub mod paged_sparse;
pub mod policy;
pub mod readout;
pub mod rowemit;
pub mod serve;
pub mod snapshot_prefix;
#[cfg(not(target_arch = "wasm32"))]
pub mod distributed;
#[cfg(not(target_arch = "wasm32"))]
pub mod job;
pub mod jobfault;
pub mod job_params;
pub mod fp8;
pub mod gdn;
pub mod gdn_mixer;
pub mod gdn_segment;
pub mod gdn_segments;
pub mod gdn_verify;
pub mod gqa_mixer;
pub mod grid;
pub mod head_ce;
pub mod head_transform;
pub mod half;
pub mod eigen;
pub mod hostmath;
pub mod cq;
pub mod int4;
pub mod int8;
pub mod kquant;
pub mod logitscale;
pub mod logprobs;
pub mod looped;
pub mod lora;
pub mod lsa;
pub mod lossw;
pub mod lut4;
pub mod glu;
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
pub mod devcollective;
pub mod gradbucket;
#[cfg(not(target_arch = "wasm32"))]
pub mod netcollective;
#[cfg(not(target_arch = "wasm32"))]
pub mod sealedwire;
#[cfg(not(target_arch = "wasm32"))]
pub mod parallel;
pub mod plan;
pub mod postable;
pub mod projector;
pub mod prunable;
pub mod surgery;
pub mod probe;
pub mod hostopt;
pub mod resident_ft;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod resident_shard;
pub mod rollout;
pub mod routing;
pub mod saved;
pub mod segmented;
#[cfg(not(target_arch = "wasm32"))]
pub mod shard;
pub mod short_conv;
pub mod streamed;
pub mod stepgate;
pub mod train;
pub mod trainprec;
pub mod training;
pub mod training_plan;
pub mod vit;
pub mod vlm;
pub mod rope_scaling;
pub mod rope_table;
pub mod yarn;
pub mod zero;

pub use collective::{broadcast_words, Collective, HostCollective, TransportReport};
pub use collective::{payload_bytes, CollectiveProfile, OpKind, OpSnapshot, ShapedCollective, SimulatedLink, Snapshot, TimedCollective};
pub use optim::{grad_multiplier, Adam};
#[cfg(not(target_arch = "wasm32"))]
pub use distributed::{federated_average, reduce_grads, reduce_grads_weighted, DdpOptimizer};
#[cfg(not(target_arch = "wasm32"))]
pub use netcollective::NetworkCollective;
#[cfg(not(target_arch = "wasm32"))]
pub use devcollective::{CollectivePath, JobCollective};
#[cfg(not(target_arch = "wasm32"))]
pub use job::{CheckpointCommit, CommitFile, JobContext, JobPlan};
#[cfg(not(target_arch = "wasm32"))]
pub use sealedwire::JobSecret;
pub use grid::{Axis, Coord, Grid, LocalGroups};
#[cfg(not(target_arch = "wasm32"))]
pub use parallel::DataParallel;
pub use plan::{plan_tp, Hardware, ModelShape, TpPlan};
#[cfg(not(target_arch = "wasm32"))]
pub use shard::{plan_balanced, plan_stage_layout, Pipeline, PipelineModel, Shard, ShardCost, Shardable, StreamPlan};

pub use objective::{Objective, StepVerdict, Supply};
/// A model's gradients on the device ([`Model::device_grads`]): the card and, per optimised parameter in
/// [`Model::param_names`] order, its gradient buffer, element count and element type.
pub struct DeviceGrads {
    pub gpu: gpu_core::Gpu,
    pub tensors: Vec<(gpu_core::DeviceBuffer, u64, gpu_core::CollectiveDtype)>,
}

impl DeviceGrads {
    /// The gradients of `names` from the parameter store, and from `resident` for the names it holds: what a model
    /// that keeps a resident full fine-tune beside its store answers [`Model::device_grads`] with.
    pub fn from_store_and_resident(gpu: &gpu_core::Gpu, store: &paramstore::ParamStore, resident: Option<&resident_ft::ResidentFullFt>, names: &[String]) -> Result<DeviceGrads, String> {
        let held: HashMap<String, (gpu_core::DeviceBuffer, u64, gpu_core::CollectiveDtype)> = match resident {
            Some(r) => r.device_grads()?.into_iter().map(|(name, b, n, d)| (name, (b, n, d))).collect(),
            None => HashMap::new(),
        };
        let tensors = names
            .iter()
            .map(|n| match (held.get(n), store.grad.get(n)) {
                (Some(t), _) => Ok(t.clone()),
                (None, Some(g)) => Ok((g.clone(), store.numel(n) as u64, gpu_core::CollectiveDtype::F32)),
                (None, None) => Err(format!("{n}: its gradient is in neither the parameter store nor the resident state")),
            })
            .collect::<Result<_, _>>()?;
        Ok(DeviceGrads { gpu: gpu.share(), tensors })
    }

    /// The fp32 gradients of `names` in `store`, or which one it does not hold.
    pub fn from_store(gpu: &gpu_core::Gpu, store: &paramstore::ParamStore, names: &[String]) -> Result<DeviceGrads, String> {
        DeviceGrads::from_store_and_resident(gpu, store, None, names)
    }
}

pub use trainprec::{apply_train_precision, OptimizerParts, OptimizerState, ResidentRequest, StateSupport, TrainPrecision, TrainPrecisionSupport};
#[cfg(not(target_arch = "wasm32"))]
pub use score::{example_rows, score_base, score_dataset, score_tuned, supervised_positions, ScoredSet};
pub use train::{cosine_lr, generate, FitOpts, LrSchedule, NativeSparse, IGNORE};
#[cfg(not(target_arch = "wasm32"))]
pub use train::{build_or_warm_start, StartKind, causal_lm, causal_lm_examples, causal_lm_packed, fit, fit_controlled, fit_controlled_sharded, fit_controlled_sync, fit_distributed, fit_from, fit_with, load_dataset, load_dataset_with_images, load_dataset_with_itos, load_rollout_dataset, AdapterPublish, AdapterSnapshots, CheckpointIdentity, FitControl, FitReport, GradSync, LoopTime, StepReport, StepWork};

/// What a batch looks like for a given model. Decoder-LM and seq2seq differ in
/// whether there is a separate source sequence; this enum keeps `set_batch`
/// uniform without forcing every model to accept encoder inputs it ignores.
#[derive(Clone, Copy)]
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
    /// [`Batch::Lm`] whose rows are dataset examples that bring inputs of their own beyond tokens (the images of a
    /// vision-language record): `examples[i]` names the example of `split` laid out in row `i` (`None`: a row of
    /// padding). The trainer sends this variant only when the run asks for it
    /// ([`train::causal_lm_examples`]); a model that reads such inputs looks them up by `(split, example)`.
    LmExamples { split: Split, tokens: &'a [u32], targets: &'a [u32], examples: &'a [Option<usize>] },
    /// [`Batch::Lm`] whose rows each pack several documents ([`data::packing`]): `segments[i]` is token `i`'s document
    /// id, never decreasing within a row, a new document starting wherever it changes and at every row start
    /// ([`data::packing::document_positions`] validates it and gives each token's position inside its document).
    /// `targets` already carry IGNORE where a target would cross into the next document. A model that reads it
    /// ([`Model::reads_packed_rows`]) resets everything carried along the sequence at each document start -
    /// attention reaches only the query's own document, every convolution or recurrent state restarts, positions
    /// count from 0 - so each document trains exactly as it would alone. Sent only by a run that asks for packing
    /// ([`train::causal_lm_packed`]), which refuses a model that does not read it; a model that does not must
    /// refuse this variant by name, never train on it as plain rows.
    LmPacked { tokens: &'a [u32], targets: &'a [u32], segments: &'a [u32] },
}

/// The part of a prepared dataset a batch is drawn from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Train,
    Val,
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

    /// [`Model::new`] with the arithmetic of the frozen-weight GEMMs chosen: what the shared trainer builds a run's
    /// model with ([`train::build_for`], from [`train::FitOpts::gemm_precision`]). A model that runs its GEMMs through
    /// [`ops::Ops`] records the products the choice names (its passes are recorded at construction, so the choice is
    /// made here, not afterwards); the default is [`Model::new`], for a model whose GEMMs have one arithmetic, as
    /// `FitOpts::gemm_precision` documents.
    fn new_with_gemm(cfg: Self::Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, _gemm: gpu_core::provider::library::GemmPrecision) -> Self
    where
        Self: Sized,
    {
        Self::new(cfg, b, t, init)
    }

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
    /// neighbouring ones at once). `loss_denominator`, when given, divides
    /// every batch's loss and gradient in place of its own count of supervised
    /// positions, as [`training::TokenObjectives::set_loss_denominator`] does
    /// for one batch. A `None` return (the default) means it cannot, and
    /// the caller runs them one after another with [`Model::set_batch`],
    /// [`Model::forward`] and [`Model::backward`].
    fn accumulate_overlapped(&self, _batches: &[Batch], _loss_denominator: Option<f32>) -> Option<f32> {
        None
    }

    /// Whether [`Model::set_batch`] reads [`Batch::LmPacked`]: rows packed with several documents, attention, every
    /// state carried along the sequence and the positions reset at each document start. `false` (the default): a
    /// packing run ([`train::causal_lm_packed`]) refuses the model with [`training::Unsupported`] naming
    /// [`training::Capability::PackedRows`] before any step.
    fn reads_packed_rows(&self) -> bool {
        false
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

    /// Whether this model updates itself from a gradient norm and a multiplier a caller decides
    /// ([`Model::local_grad_sum_sq`], [`Model::local_step`]): the protocol a [`Pipeline`] drives when every stage
    /// owns its optimizer (resident low-precision state, off-device master and moments) instead of handing its
    /// gradients to the pipeline's host AdamW. `false` (the default) for a model whose optimizer is the pipeline's.
    fn stage_local_optimizer(&self) -> bool {
        false
    }

    /// The sum of squares of every gradient this model's own optimizer updates, accumulated over the micro-batches
    /// since [`Model::zero_grads`]: a stage's share of the global norm a clip bounds. Only for a
    /// [`Model::stage_local_optimizer`].
    fn local_grad_sum_sq(&self) -> Option<f64> {
        None
    }

    /// One update step `t` of every parameter this model owns with its gradients multiplied by `multiplier` (the
    /// accumulation scale and the clip, decided globally): [`Model::adamw_step`] with the norm taken elsewhere.
    fn local_step(&self, _t: u32, _lr: f32, _wd: f32, _adam: Adam, _multiplier: f32) -> Result<(), String> {
        Err("this model has no stage-local optimizer".into())
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

    /// Weigh the entropy of the per-token exit distribution into the objective of a model whose depth is chosen by
    /// exit gates (a looped decoder): `loss - beta * H(p)` over every supervised position, the regulariser toward a
    /// uniform prior over exit steps that keeps the exit mass from collapsing onto one depth. `0` is the plain
    /// objective. Called before the first step (the fit loop applies [`FitOpts::exit_entropy`]); a model without exit
    /// gates takes `0` and refuses any other weight by name.
    fn set_exit_entropy(&mut self, beta: f32) -> Result<(), String> {
        if beta == 0.0 {
            Ok(())
        } else {
            Err(format!("this model has no exit gates; an exit-entropy weight of {beta} has nothing to regularise"))
        }
    }

    /// Train the global-attention layers with native sparse attention ([`NativeSparse`]), before the first step. A
    /// model that has such layers and the kernels overrides this; the default refuses by name.
    fn enable_native_sparse_attention(&mut self, _sparse: NativeSparse) -> Result<(), String> {
        Err("this model has no native sparse attention training".to_string())
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

    /// Every optimised parameter's gradient where it lives on the device, in [`Model::param_names`] order, so a
    /// data-parallel reduction can sum them without staging them through the host. The default answers from the
    /// parameter store ([`Model::optimizer_parts`]) when it holds every one of them; a model that keeps gradients
    /// elsewhere (a resident full fine-tune) adds its own, and one that holds them nowhere a reduction can reach
    /// answers why, so the run reduces through the host instead.
    fn device_grads(&self) -> Result<DeviceGrads, String> {
        let parts = self.optimizer_parts().ok_or("this model keeps no parameter store a reduction can reach")?;
        let names = self.param_names();
        DeviceGrads::from_store(parts.gpu, parts.store, &names)
    }

    /// Every optimised parameter's value and gradient where they live on the device, in [`Model::param_names`]
    /// order: what a sharded optimizer ([`zero::ZeroOptimizer`]) reduces, updates and gathers back. The default
    /// answers from the parameter store ([`Model::optimizer_parts`]) when it holds every one of them; a model whose
    /// parameters live elsewhere answers why, and a sharded run of it is refused by that reason.
    #[cfg(not(target_arch = "wasm32"))]
    fn device_params(&self) -> Result<zero::DeviceParams, String> {
        let parts = self.optimizer_parts().ok_or("this model keeps no parameter store a sharded optimizer can reach")?;
        let names = self.param_names();
        zero::DeviceParams::from_store(parts.gpu, parts.store, &names)
    }

    /// The group this model shards its parameters over (ZeRO-3, [`zero::Sharding::Parameters`]), when it was built
    /// that way: it gathers, reduces and updates its own units, so a data-parallel fit loop neither reduces its
    /// gradients nor shards an optimizer of its own around it. `None` (the default) for a model that holds every
    /// parameter.
    #[cfg(not(target_arch = "wasm32"))]
    fn parameter_shards(&self) -> Option<zero::ShardGroup> {
        None
    }

    /// The cause of a failure of the job's collective inside work this model drives itself (a parameter-sharded build
    /// gathers, reduces and updates its own units in its forward, backward and optimizer step), `None` while there is
    /// none. Once set, the model's state is no step's result: the fit loop ends the run as the job's failure, and the
    /// job restarts from its last commit. `None` (the default) for a model that runs no collective of its own.
    fn collective_failure(&self) -> Option<String> {
        None
    }

    /// Write the trained model of a parameter-sharded run ([`Model::parameter_shards`]): every rank of the group calls
    /// it at the same point (each block is gathered over the group), the rank given `path` writes the checkpoint the
    /// model's [`Model::save`] would, its sharded tensors in the encoding they train in, streamed so no rank holds the
    /// whole model. Refused by a model that does not shard its parameters.
    fn save_sharded(&self, _path: Option<&str>) -> Result<(), String> {
        Err("this model does not shard its parameters".into())
    }

    /// This rank's share of the next step's items (`local / total`), by which a parameter-sharded model weighs the
    /// gradients it reduces. Refused by a model that does not shard its parameters.
    fn set_shard_weight(&self, _share: f32) -> Result<(), String> {
        Err("this model does not shard its parameters".into())
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

    /// The stacked expert banks an adapter may correct expert by expert ([`adapter::sites::ExpertSite`]): what a saved
    /// adapter's rank-3 pairs are shaped by. Empty for a model without expert sites.
    fn adapter_expert_sites(&self) -> Vec<adapter::sites::ExpertSite> {
        Vec::new()
    }

    /// Every linear whose storage tier can be chosen on its own: what a
    /// precision search places a tier on. Defaults to [`Self::adapter_sites`];
    /// a model whose adapters target fewer linears than it stores (a
    /// mixture-of-experts model's routed experts) lists the rest too.
    fn linear_sites(&self) -> Vec<adapter::LinearSite> {
        self.adapter_sites()
    }

    /// Groups of stored tensors this model's deployed engine holds as ONE
    /// weight (a fused projection: Q/K/V concatenated into one GEMM), which
    /// therefore take one storage tier together. Empty (the default) when
    /// every tensor is held on its own. A precision search assigns a group one
    /// tier; a serving policy that split a group would be refused by the engine.
    fn coupled_tensors(&self) -> Vec<Vec<String>> {
        Vec::new()
    }

    /// The dimensions whose channels a structured-pruning search may remove
    /// ([`prunable::PrunableDim`]). Empty (the default) when the model declares
    /// none.
    fn prunable_dims(&self) -> Vec<prunable::PrunableDim> {
        Vec::new()
    }

    fn read_weight(&self, name: &str) -> Vec<f32>;
    fn write_weight(&self, name: &str, data: &[f32]);
    fn read_grad(&self, name: &str) -> Vec<f32>;

    /// Whether every accumulated gradient of the optimised parameters is finite; `None` when this
    /// model does not expose them. Reads the gradients back to the host, so a caller that asks every
    /// step bounds the size of the set first ([`stepgate::MAX_SCANNED_GRAD_FLOATS`]).
    fn gradients_finite(&self) -> Option<bool> {
        let names = self.optimized_params()?;
        Some(names.iter().all(|n| self.read_grad(n).iter().all(|g| g.is_finite())))
    }

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
