// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Ripple as a trainable [`model::Model`]: the forward of [`crate::layer`] with a backward through every op, on
//! the shared training paths (`Ops` products at any weight tier, `model::block` norms, RoPE and grouped-query
//! attention backward, `model::saved` boundary residuals, `optim` AdamW, `model::adapter` LoRA).
//!
//! Swedish Embedded AB implements solutions for fine-tuning very large dense decoders on the memory a team
//! actually has, from full-weight training to adapters over a 4-bit base. If your team needs expertise in
//! training hybrid-attention language models then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # What the backward differentiates
//!
//! ```text
//! x0 = embed(tokens) * embed_scale                       (every layer reads x0 as the value-residual source)
//! per layer, h the residual stream:
//!   xn = rmsnorm(h)                  v_in = lerp(x0, xn, sigmoid(value_res_lambda))
//!   q, k = rmsnorm_per_head(q_proj xn), rmsnorm_per_head(k_proj xn)        v = v_proj v_in
//!   sliding layers: RoPE on q, k;  window keys only        global layers: NoPE, whole causal prefix
//!   a = attention(q, k, v) * sigmoid(g_proj xn)                            (the gate is computed in f32)
//!   h = h + rmsnorm(o_proj a)
//!   m = rmsnorm(h);  h = h + rmsnorm(down(silu(min(gate m, T)) * clamp(up m, -T, T)))
//! logits = lm_head rmsnorm(h)
//! ```
//!
//! The gradient of `x0` is the sum over all layers' value paths plus the stream's own, and
//! `value_res_lambda` has one scalar gradient per layer; a clamp passes the gradient only inside its closed
//! interval. Attention backward is the materialised softmax chain (`gqa_bwd_*`): the windowed layers' masked
//! probabilities are exactly zero outside the window, so the same kernels differentiate both layer types.
//!
//! # Memory
//!
//! The forward keeps only the residual between layers ([`SavedPolicy`] recompute, optionally held in bf16 or
//! FP8) and the backward rebuilds one layer at a time into one shared scratch set, so activation memory is the
//! residual stream plus one layer. With an adapter the base is frozen and each linear may be held packed
//! (bf16, FP8, int8, NVFP4): the backward reads the same packed bytes the forward did.

pub mod adapter;
mod config;
mod graph;
mod init;
mod io;
mod nlsa;
mod objectives;
mod resident;
mod shard;

use std::cell::Cell;
use std::collections::HashMap;

use gpu_core::provider::library::GemmPrecision;
use gpu_core::select::Dtype;
use gpu_core::{DeviceBuffer, Gpu};
use model::adapter::device::LoraCfg;
use model::adapter::exec::AdapterRuntime;
use model::ops::{Ops, TierPolicy, Weight};
use model::saved::{Residuals, SavedFormat, SavedPolicy};
use model::segmented::SegmentedPass;
use model::Shard;
use paramstore::{ParamStore, Role};

pub use config::{default_targets, expand_modules_to_save, linear_suffix, TrainConfig, LORA_LEAVES, SAVED_GROUPS};
pub use init::AdapterInit;
pub use resident::{plan_resident, resident_specs, stage_costs, StageCost};
use resident::check_resident_fits;
pub use shard::estimate_stage_bytes;
use graph::{Grad, Kernels, LayerParams, Lin, Scratch};

use crate::config::{LayerType, RicursiveConfig};

/// Cross-entropy ignore index (unsupervised positions).
pub const IGNORE: u32 = model::IGNORE;

/// How a training model is built beyond its configuration.
#[derive(Clone, Debug)]
pub struct TrainOpts {
    /// The tier each linear is held at. `F32` (the default) keeps it in the parameter store; any other tier needs an
    /// adapter (the base is then frozen) and is read packed by the forward and the backward alike.
    pub tiers: TierPolicy,
    /// The arithmetic of the dense products.
    pub gemm: GemmPrecision,
    /// What the forward keeps for the backward: the boundary residuals in f32, bf16 or FP8.
    pub saved: SavedPolicy,
    /// Rows of logits the training head holds at once; `None`: as many as fit one GiB (all of them for a small batch).
    /// A multiple of 64 unless it covers the whole batch.
    pub head_chunk_rows: Option<u32>,
    /// A resident low-precision full fine-tune: the linears and the two tables live on the device in the encoding the
    /// plan's configuration names with their master copy and optimizer state off it (`model::resident_ft`), optionally
    /// streamed through a window of layers. A full fine-tune only; the plan is checked against the device before
    /// anything is allocated. Built by [`plan_resident`].
    pub resident: Option<model::resident_ft::ResidentPlan>,
}

impl Default for TrainOpts {
    fn default() -> TrainOpts {
        TrainOpts { tiers: TierPolicy::uniform(Dtype::F32), gemm: GemmPrecision::Fp32Reference, saved: SavedPolicy::recompute_with(SavedFormat::F32), head_chunk_rows: None, resident: None }
    }
}

/// The kernels a training device registers: the shared façade set, the tier decodes, Ripple's own elementwise
/// kernels with their backward, the attention chain, AdamW, boundary packing and the adapter runtime's.
pub fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let own: [(&'static str, &'static str); 37] = [
            ("rmsnorm", kernels::RMSNORM),
            ("rms_inv", kernels::RMS_INV),
            ("rmsnorm_dx", kernels::RMSNORM_DX),
            ("rmsnorm_dw", kernels::RMSNORM_DW),
            ("rope_base", kernels::ROPE_BASE),
            ("rope_base_bwd", kernels::ROPE_BASE_BWD),
            ("gqa_scores_win", kernels::GQA_SCORES_WIN),
            ("attn_softmax", kernels::ATTN_SOFTMAX),
            ("gqa_apply", kernels::GQA_APPLY),
            ("gqa_bwd_dscores", kernels::GQA_BWD_DSCORES),
            ("gqa_bwd_dv", kernels::GQA_BWD_DV),
            ("gqa_bwd_dq", kernels::GQA_BWD_DQ),
            ("gqa_bwd_dk", kernels::GQA_BWD_DK),
            ("sigmoid", kernels::SIGMOID),
            ("sigmoid_bwd", kernels::SIGMOID_BWD),
            ("mul", kernels::MUL),
            ("add2", kernels::ADD2),
            ("silu_mul", kernels::SILU_MUL),
            ("silu_bwd_da", kernels::SILU_BWD_DA),
            ("silu_bwd_db", kernels::SILU_BWD_DB),
            ("clamp_range", kernels::CLAMP_RANGE),
            ("clamp_range_bwd", kernels::CLAMP_RANGE_BWD),
            ("value_lerp", kernels::VALUE_LERP),
            ("value_lerp_bwd", kernels::VALUE_LERP_BWD),
            ("value_lerp_dlam", kernels::VALUE_LERP_DLAM),
            ("bias_grad", kernels::BIAS_GRAD),
            ("axpy", kernels::AXPY),
            ("grad_scale", kernels::GRAD_SCALE),
            ("ce_value", kernels::CE_VALUE_MASKED),
            ("ce_grad", kernels::CE_GRAD_MASKED),
            ("emb_bwd", kernels::EMB_BWD),
            ("adamw", kernels::ADAMW),
            ("gradnorm_sq", kernels::GRADNORM_SQ),
            ("clip_coef", kernels::CLIP_COEF),
            ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
            ("matmul_gemv", kernels::MATMUL_GEMV),
            ("scale_row", kernels::SCALE_ROW),
        ];
        let mut table: Vec<(&'static str, &'static str)> = Vec::new();
        let add = |table: &mut Vec<(&'static str, &'static str)>, list: &[(&'static str, &'static str)]| {
            for &entry in list {
                if !table.iter().any(|(name, _)| *name == entry.0) {
                    table.push(entry);
                }
            }
        };
        add(&mut table, &own);
        add(&mut table, model::ops::training_kernel_list());
        add(&mut table, model::saved::kernels());
        add(&mut table, model::head_ce::kernels());
        add(&mut table, model::ops::frozen_base_kernels());
        add(&mut table, optim::MOMENT_KERNELS);
        add(&mut table, &model::lsa::device::KERNELS);
        add(&mut table, &model::lsa::native::KERNELS);
        add(&mut table, &gpu_core::sort::KERNELS);
        model::adapter::exec::add_kernels(&mut table);
        table
    })
}

/// Ripple on a device with a backward, ready to take batches. Built by [`Ripple::build`] (or `Model::new`).
pub struct Ripple {
    gpu: Gpu,
    ops: Ops,
    cfg: TrainConfig,
    shard: Shard,
    b: u32,
    t: u32,
    train: bool,
    count: Cell<f32>,
    ps: ParamStore,
    /// Every linear by weight name, at the tier it is held in (a trainable one shares the store's buffer).
    weights: HashMap<String, Weight>,
    layers: Vec<LayerParams>,
    head: Option<Lin>,
    adapter: Option<AdapterRuntime>,
    k: Kernels,
    tokens: DeviceBuffer,
    targets: DeviceBuffer,
    /// The value-residual source `x0`, which is also the stream entering layer 0.
    x0: DeviceBuffer,
    /// The stream leaving layer `l` is `res[l + 1]`; `res[0]` is an unused slot (layer 0 reads `x0`).
    res: Residuals,
    hs: DeviceBuffer,
    logits: DeviceBuffer,
    ce_rows: DeviceBuffer,
    scratch: Scratch,
    grad: Option<Grad>,
    ce_grad_uniform: DeviceBuffer,
    /// `1 / count` of the batch in one word, what the chunked head scales its loss gradient by.
    inv_count: DeviceBuffer,
    /// The training head: the loss and its gradient a chunk of rows at a time; `None` on an inference build.
    head_ce: Option<model::head_ce::HeadCe>,
    /// The recorded passes, in segments that name the layer each runs: a layer-streamed build acts between the layers.
    fwd: SegmentedPass,
    bwd: SegmentedPass,
    /// The linears and the two tables of a resident full fine-tune, their master copies and optimizer state.
    resident: Option<std::cell::RefCell<model::resident_ft::ResidentFullFt>>,
    /// The global gradient norm of the latest clipped resident step.
    resident_norm: Cell<Option<f32>>,
    /// A layer-streamed backward runs inside the update: set by `backward`, taken by `adamw_step`.
    backward_pending: Cell<bool>,
    /// The distinct token ids of the batch, padded: what the embedding gradient of a resident build scatters by.
    tok_uniq: Option<DeviceBuffer>,
    opt: Option<optim::Optim>,
    nlsa: Option<nlsa::NativeLsa>,
    /// Per-position loss weights, once [`model::training::TokenObjectives::enable_weighted_loss`] opted in.
    weighted: Option<model::lossw::WeightedCe>,
    /// The factor the logits are multiplied by before the softmax (the sampler-temperature correction).
    logit_scale: f32,
}

impl Ripple {
    /// Build over `src`: the checkpoint (or an in-memory map) the weights are read from by their canonical names.
    ///
    /// A full fine-tune (`cfg.lora == None`) trains every tensor and needs f32 linears. An adapter build freezes the
    /// base, trains the factors, and may hold each base linear at any tier `opts.tiers` names.
    pub fn build(cfg: TrainConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource, opts: &TrainOpts, train: bool) -> Result<Ripple, String> {
        cfg.validate()?;
        graph::build(cfg, b, t, src, opts, train)
    }

    /// Rows per step, `b * t`.
    /// Stage `shard` of the model for a pipeline: only its layers, and the embedding, norm and head it needs.
    pub fn build_shard(cfg: TrainConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource, opts: &TrainOpts, train: bool, shard: Shard) -> Result<Ripple, String> {
        cfg.validate()?;
        graph::build_shard(cfg, b, t, src, opts, train, shard)
    }

    fn rows(&self) -> u32 {
        self.b * self.t
    }

    /// The model's own configuration.
    pub fn model_config(&self) -> &RicursiveConfig {
        &self.cfg.model
    }

    /// The device the model lives on.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// Upload a causal-LM batch (`targets` predict the next token; [`IGNORE`] masks a row of the loss).
    pub fn set_tokens_targets(&self, tokens: &[u32], targets: &[u32]) {
        let n = self.rows() as usize;
        assert_eq!(tokens.len(), n, "tokens must be b*t");
        assert_eq!(targets.len(), n, "targets must be b*t");
        let vocab = self.cfg.model.vocab_size as u32;
        assert!(tokens.iter().all(|&x| x < vocab), "ricursive: a token id is outside the vocabulary of {vocab}");
        assert!(targets.iter().all(|&x| x == IGNORE || x < vocab), "ricursive: a target id is outside the vocabulary of {vocab}");
        self.gpu.write(&self.tokens, tokens);
        if let Some(uniq) = &self.tok_uniq {
            self.gpu.write(uniq, &model::ops::uniq_padded(tokens, n));
        }
        self.gpu.write(&self.targets, targets);
        let count = targets.iter().filter(|&&v| v != IGNORE).count();
        self.count.set(count.max(1) as f32);
    }

    /// The stream the final norm reads (`[rows, hidden]`), after the last forward.
    pub fn read_final_residual(&self) -> Vec<f32> {
        self.gpu.read(&self.res[self.cfg.model.num_hidden_layers], self.rows() as usize * self.cfg.model.hidden_size)
    }

    /// The logits of the last forward, `[rows, vocab]`.
    pub fn read_logits(&self) -> Vec<f32> {
        self.read_logit_rows(self.rows() as usize, self.cfg.model.vocab_size)
    }

    /// Bytes of device memory the activations of the forward and backward hold (not the parameters).
    pub fn activation_bytes(&self) -> u64 {
        graph::activation_bytes(&self.cfg, self.b, self.t, self.res.format(), self.train)
    }

    /// The tier linear `name` is held at (`F32` for one in the parameter store), `None` for a name that is no linear.
    pub fn weight_tier(&self, name: &str) -> Option<Dtype> {
        self.weights.get(name).map(Weight::dtype)
    }

    /// The adapter this model trains, if any.
    pub fn lora(&self) -> Option<&LoraCfg> {
        self.cfg.lora.as_ref()
    }

    /// The layer type of layer `layer`.
    pub fn layer_type(&self, layer: usize) -> LayerType {
        self.cfg.model.layer_type(layer)
    }

    /// Does this model hold a parameter named `name`?
    pub fn has_param(&self, name: &str) -> bool {
        self.ps.weight.contains_key(name) || self.resident_index(name).is_some()
    }

    /// Names a [`Role::Trainable`] parameter of the store has.
    fn is_trainable(&self, name: &str) -> bool {
        self.ps.grad.contains_key(name)
    }

    /// Role of every parameter in a build with `lora`: only adapter factors train when there is an adapter, and
    /// the base tensors it also saves in full (`modules_to_save`).
    fn role(cfg: &TrainConfig, name: &str) -> Role {
        match &cfg.lora {
            Some(lora) if lora.saves(name) && !model::adapter::device::is_adapter_param(name) => Role::Trainable,
            Some(lora) => model::adapter::device::role_of(name, &lora.hp()),
            None => Role::Trainable,
        }
    }
}
