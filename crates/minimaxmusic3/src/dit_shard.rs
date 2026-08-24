// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pipeline-parallel sharding for the flow-matching DiT ([`DitStage`]),
//! wiring it onto the generic `model::Shardable` seam (`model::shard`'s
//! module doc: cut placement via `model::plan_balanced`, orchestration via
//! `model::Pipeline`) - the same seam `crates/ltxv`'s `LtxDit` already
//! wires a diffusion transformer onto, and the precedent this module
//! follows closely.
//!
//! # Who computes what, and why nothing but the residual crosses the wire
//!
//! `dit::forward`'s op sequence has a part that runs ONCE (the preprocess
//! conv + `proj_in`, and the Fourier timestep embedding) and a part that
//! runs PER BLOCK (partial-RoPE bidirectional attention, the gated FFN). A
//! naive per-layer cut would also need to ship the timestep token across
//! every stage boundary - extra wire traffic on top of the residual
//! `model::shard`'s own doc says should be the ONLY thing that crosses a
//! cut. Instead, every stage loads its OWN copy of the (tiny)
//! `time_proj.*`/`time_embed.*` weights and independently recomputes its
//! own timestep token from the batch - the same "every stage gets the
//! whole batch, some of its own weights are REPLICATED" shape `ltxv`'s
//! `adaln_single.*` already uses for the identical reason. Only
//! `preprocess_conv.*`/`proj_in.*` (embed-stage-only: they produce the
//! first stage's initial residual) and `proj_out.*`/`postprocess_conv.*`
//! (head-stage-only: the final projection) are NOT replicated - see
//! [`shard_owns_weight`]'s doc for the exact partition.
//!
//! # `model::Batch` does not fit a diffusion batch - `DitStageBatch` does
//!
//! [`DitStage`] cannot implement `Model::set_batch` meaningfully:
//! `model::Batch`'s variants carry LM tokens, seq2seq pairs, or an image
//! splice, never a continuous latent + condition + scalar timestep. Per
//! `ltxv::shard`'s own doc (the same situation, same fix), `Model::set_batch`
//! is therefore a documented no-op and the real seam is
//! [`DitStage::load_shard_batch`] (owned data, survives past one call).
//!
//! # What is real here, and what is an explicit, tracked gap
//!
//! Real and tested (`crates/minimaxmusic3/tests/dit_shard_parity.rs`):
//! * [`shard_cost`](Shardable::shard_cost) - a FLOP-shaped per-block cost
//!   (QKVO + the O(T^2) score/weighted-sum term, the fused gated FFN's two
//!   projections), fed to `model::plan_balanced`.
//! * [`Shardable::new_shard`] genuinely loads only its block range's weight
//!   subset ([`shard_weights`]), not the full 36-layer stack.
//! * The single-shard degenerate case (`Shard::whole`) runs for real and is
//!   checked bit-for-bit-close against `dit::forward` on identical
//!   weights/inputs.
//! * A genuine two-stage split (real block-range partition, NOT both
//!   stages secretly owning everything) with the boundary handed off
//!   through [`DitStage::write_in_res`]/[`DitStage::read_out_res`], run
//!   sequentially on a single device, composed and checked against the
//!   same non-sharded reference.
//!
//! Explicit gaps, not silently glossed over (identical in kind to
//! `ltxv::shard`'s own documented gaps for `LtxDit`):
//! * **No real multi-device execution proven by this crate's own test
//!   coverage.** This machine has no discrete GPU at all, let alone two -
//!   a single-GPU (indeed single-CPU-JIT-backend) run cannot exercise two
//!   stages resident on two different physical devices. The two-stage
//!   test above proves the BOUNDARY HANDOFF is correct, not that two real
//!   cards agree.
//! * **No backward pass through the pipeline.** [`DitStage`] has no
//!   gradient/training machinery - `crate::dit_train::Trainer` is a
//!   SEPARATE, single-device, non-sharded training path this pipeline seam
//!   does not build on (and does not need to: this crate's DiT training
//!   story - full fine-tune, gradcheck, LoRA - is already served
//!   end-to-end by `dit_train::Trainer` on one device; `Shardable` exists
//!   here purely to let a too-big-for-one-card DiT run split for
//!   INFERENCE, the actual reason a diffusion transformer this size needs
//!   multiple GPUs at all). `Shardable::run_backward_stage`/
//!   `read_in_dres`/`write_out_dres` and `Model::backward`/`write_weight`/
//!   `read_grad` all `unimplemented!()` loudly rather than returning a
//!   silently-wrong zero gradient.
//! * **`model::Pipeline<DitStage>` type-checks but is not a usable
//!   end-to-end entry point**, for the same reason `model::Pipeline<LtxDit>`
//!   is not: `Pipeline::forward`/`pipelined_fwd_bwd` drive a stage purely
//!   through `Model::set_batch`, which is a documented no-op here (see
//!   above). The tests in this module hand-drive `Shardable::new_shard`/
//!   `run_forward_stage`/`read_out_res`/`write_in_res` directly instead.

use std::cell::RefCell;
use std::collections::HashMap;

use gpu_core::Gpu;
use model::{Batch, Model, ModelConfig, Shard, ShardCost, Shardable};

use crate::config::DitConfig;
use crate::dit::{self, AttnW, BlockW, DitWeights};

/// Every real tensor name + shape `dit::from_tensors`'s own `get()` calls
/// read - the `Model`/`Shardable` boundary's flat representation
/// (`ModelConfig::param_list`, `Model::init_weights`, `Shardable::new_shard`'s
/// `init` map), mirrored here exactly rather than derived from
/// `dit::from_tensors` (that function reads FROM a map; this one has to
/// describe the map's own shape, so one direction has to be spelled out by
/// hand - `crate::dit::DitWeights::linear_mut`'s own names deliberately did
/// NOT reuse these, since those are internal short names and these are the
/// real checkpoint names `dit::import` actually reads).
pub fn dit_tensor_manifest(cfg: &DitConfig) -> Vec<(String, Vec<usize>)> {
    let inner = cfg.inner_dim() as usize;
    let ff_inner = cfg.ff_inner_dim as usize;
    let concat = cfg.concat_channels() as usize;
    let cin = cfg.in_channels as usize;
    let fourier = cfg.fourier_embedding_dim as usize;
    let mut m = vec![
        ("time_proj.weight".to_string(), vec![fourier / 2, 1]),
        ("time_embed.linear_1.weight".to_string(), vec![inner, fourier]),
        ("time_embed.linear_1.bias".to_string(), vec![inner]),
        ("time_embed.linear_2.weight".to_string(), vec![inner, inner]),
        ("time_embed.linear_2.bias".to_string(), vec![inner]),
        ("preprocess_conv.weight".to_string(), vec![concat, concat, 1]),
        ("proj_in.weight".to_string(), vec![inner, concat]),
    ];
    for i in 0..cfg.num_layers as usize {
        let p = format!("transformer_blocks.{i}");
        m.push((format!("{p}.norm1.weight"), vec![inner]));
        m.push((format!("{p}.norm1.bias"), vec![inner]));
        m.push((format!("{p}.attn.to_q.weight"), vec![inner, inner]));
        m.push((format!("{p}.attn.to_k.weight"), vec![inner, inner]));
        m.push((format!("{p}.attn.to_v.weight"), vec![inner, inner]));
        m.push((format!("{p}.attn.to_out.0.weight"), vec![inner, inner]));
        m.push((format!("{p}.norm2.weight"), vec![inner]));
        m.push((format!("{p}.norm2.bias"), vec![inner]));
        m.push((format!("{p}.ff_in.weight"), vec![2 * ff_inner, inner]));
        m.push((format!("{p}.ff_in.bias"), vec![2 * ff_inner]));
        m.push((format!("{p}.ff_out.weight"), vec![inner, ff_inner]));
        m.push((format!("{p}.ff_out.bias"), vec![inner]));
    }
    m.push(("proj_out.weight".to_string(), vec![cin, inner]));
    m.push(("postprocess_conv.weight".to_string(), vec![cin, cin, 1]));
    m
}

/// The exact inverse of [`dit_tensor_manifest`]'s naming, over an actual
/// [`DitWeights`] - the flat map `Model::init_weights`/`Shardable::new_shard`
/// consume, and what `dit_train::random_weights` is turned into for
/// `DitStage::init_weights`.
pub fn flatten_weights(w: &DitWeights) -> HashMap<String, Vec<f32>> {
    let mut m = HashMap::new();
    m.insert("time_proj.weight".to_string(), w.time_proj_weight.clone());
    m.insert("time_embed.linear_1.weight".to_string(), w.time_embed_l1_w.clone());
    m.insert("time_embed.linear_1.bias".to_string(), w.time_embed_l1_b.clone());
    m.insert("time_embed.linear_2.weight".to_string(), w.time_embed_l2_w.clone());
    m.insert("time_embed.linear_2.bias".to_string(), w.time_embed_l2_b.clone());
    m.insert("preprocess_conv.weight".to_string(), w.preprocess_conv_w.clone());
    m.insert("proj_in.weight".to_string(), w.proj_in_w.clone());
    for (i, b) in w.blocks.iter().enumerate() {
        let p = format!("transformer_blocks.{i}");
        m.insert(format!("{p}.norm1.weight"), b.norm1_w.clone());
        m.insert(format!("{p}.norm1.bias"), b.norm1_b.clone());
        m.insert(format!("{p}.attn.to_q.weight"), b.attn.wq.clone());
        m.insert(format!("{p}.attn.to_k.weight"), b.attn.wk.clone());
        m.insert(format!("{p}.attn.to_v.weight"), b.attn.wv.clone());
        m.insert(format!("{p}.attn.to_out.0.weight"), b.attn.wo.clone());
        m.insert(format!("{p}.norm2.weight"), b.norm2_w.clone());
        m.insert(format!("{p}.norm2.bias"), b.norm2_b.clone());
        m.insert(format!("{p}.ff_in.weight"), b.ff_in_w.clone());
        m.insert(format!("{p}.ff_in.bias"), b.ff_in_b.clone());
        m.insert(format!("{p}.ff_out.weight"), b.ff_out_w.clone());
        m.insert(format!("{p}.ff_out.bias"), b.ff_out_b.clone());
    }
    m.insert("proj_out.weight".to_string(), w.proj_out_w.clone());
    m.insert("postprocess_conv.weight".to_string(), w.postprocess_conv_w.clone());
    m
}

/// Whether pipeline stage `shard` needs weight `name` at all - the ONLY
/// weights a stage loads, so a partial shard never materializes the full
/// 36-block stack just to discard most of it (see [`shard_weights`]).
/// `transformer_blocks.{l}.*` follows the block range (`shard.owns(l)`);
/// `time_proj.*`/`time_embed.*` is REPLICATED (this module's own doc);
/// `preprocess_conv.*`/`proj_in.*` are embed-only (they produce the FIRST
/// stage's initial residual); `proj_out.*`/`postprocess_conv.*` are
/// head-only (the final projection).
pub(crate) fn shard_owns_weight(shard: &Shard, name: &str) -> bool {
    if let Some(rest) = name.strip_prefix("transformer_blocks.") {
        let l: usize = rest.split('.').next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
        return shard.owns(l);
    }
    match name {
        "time_proj.weight" | "time_embed.linear_1.weight" | "time_embed.linear_1.bias" | "time_embed.linear_2.weight" | "time_embed.linear_2.bias" => true,
        "preprocess_conv.weight" | "proj_in.weight" => shard.embed,
        "proj_out.weight" | "postprocess_conv.weight" => shard.head,
        _ => false,
    }
}

/// Load only `shard`'s own weight subset from a flat checkpoint - see
/// [`shard_owns_weight`]'s doc for exactly which names that is.
pub(crate) fn shard_weights(cfg: &DitConfig, init: &HashMap<String, Vec<f32>>, shard: &Shard) -> HashMap<String, Vec<f32>> {
    dit_tensor_manifest(cfg)
        .into_iter()
        .filter(|(name, _)| shard_owns_weight(shard, name))
        .map(|(name, shape)| {
            let data = init.get(&name).unwrap_or_else(|| panic!("minimaxmusic3 dit shard: missing weight {name}")).clone();
            let want: usize = shape.iter().product();
            assert_eq!(data.len(), want, "minimaxmusic3 dit shard: {name} wrong length ({} vs {want})", data.len());
            (name, data)
        })
        .collect()
}

/// Reconstruct block `l`'s (absolute layer index) [`BlockW`] from a flat
/// weight map - the inverse of [`flatten_weights`]'s per-block entries,
/// scoped to one block, since a shard's own `[start, end)` range needs
/// exactly this, never the full `Vec<BlockW>` `dit::from_tensors` builds.
fn block_from_flat(w: &HashMap<String, Vec<f32>>, l: usize) -> BlockW {
    let p = format!("transformer_blocks.{l}");
    let g = |n: &str| w.get(n).unwrap_or_else(|| panic!("minimaxmusic3 dit shard: missing weight {n}")).clone();
    BlockW {
        norm1_w: g(&format!("{p}.norm1.weight")),
        norm1_b: g(&format!("{p}.norm1.bias")),
        attn: AttnW { wq: g(&format!("{p}.attn.to_q.weight")), wk: g(&format!("{p}.attn.to_k.weight")), wv: g(&format!("{p}.attn.to_v.weight")), wo: g(&format!("{p}.attn.to_out.0.weight")) },
        norm2_w: g(&format!("{p}.norm2.weight")),
        norm2_b: g(&format!("{p}.norm2.bias")),
        ff_in_w: g(&format!("{p}.ff_in.weight")),
        ff_in_b: g(&format!("{p}.ff_in.bias")),
        ff_out_w: g(&format!("{p}.ff_out.weight")),
        ff_out_b: g(&format!("{p}.ff_out.bias")),
    }
}

impl ModelConfig for DitConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        dit_tensor_manifest(self).into_iter().map(|(n, shape)| (n, shape.iter().product())).collect()
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "in_channels": self.in_channels,
            "condition_dim": self.condition_dim,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "attention_head_dim": self.attention_head_dim,
            "ff_inner_dim": self.ff_inner_dim,
            "rotary_dim": self.rotary_dim,
            "fourier_embedding_dim": self.fourier_embedding_dim,
        })
    }
    fn from_json(v: &serde_json::Value) -> DitConfig {
        let u = |k: &str| v[k].as_u64().unwrap_or_else(|| panic!("DitConfig::from_json: missing {k}")) as u32;
        DitConfig {
            in_channels: u("in_channels"),
            condition_dim: u("condition_dim"),
            num_layers: u("num_layers"),
            num_attention_heads: u("num_attention_heads"),
            attention_head_dim: u("attention_head_dim"),
            ff_inner_dim: u("ff_inner_dim"),
            rotary_dim: u("rotary_dim"),
            fourier_embedding_dim: u("fourier_embedding_dim"),
        }
    }
    fn vocab(&self) -> u32 {
        0 // no token vocabulary - a diffusion transformer over continuous latents
    }
    fn block_size(&self) -> u32 {
        0 // no fixed context window - the chunk length is set per call
    }
    fn finalize_for_dataset(self, _vocab: u32, _block_size: u32) -> DitConfig {
        self // nothing dataset-derived to apply
    }
}

/// One pipeline-stage batch for [`DitStage`] - see this module's doc
/// ("`model::Batch` does not fit a diffusion batch") for why this (owned)
/// seam exists instead of `model::Batch`.
pub struct DitStageBatch {
    /// `[in_channels, length]` NCL - only read on the embed stage (a
    /// non-embed stage's input comes from [`DitStage::write_in_res`]
    /// instead).
    pub latents: Vec<f32>,
    /// `[length, condition_dim]` - only read on the embed stage.
    pub condition: Vec<f32>,
    /// Every stage needs this (it recomputes its own replicated timestep
    /// token) - see this module's doc.
    pub timestep: f32,
    pub length: usize,
    /// `[in_channels, length]` NCL training target (flow-matching
    /// velocity); `None` for a forward-only run - [`DitStage::run_stage_forward`]
    /// then returns `None` even on the head stage.
    pub target: Option<Vec<f32>>,
}

/// One pipeline stage of the flow-matching DiT - `Shard::whole` for the
/// ordinary, non-sharded path every other entry point in this crate uses.
/// Weights are host-resident (a flat name -> data map, only this stage's
/// own subset - see [`shard_weights`]); device residency is per-call
/// (`dit::block_fwd`'s own convention, matching `dit::forward`'s style).
pub struct DitStage {
    cfg: DitConfig,
    w: HashMap<String, Vec<f32>>,
    shard: Shard,
    batch: RefCell<Option<DitStageBatch>>,
    /// This stage's INPUT-side residual (`res[shard.start]`), written by
    /// the previous stage via [`Self::write_in_res`]; read instead of
    /// running the embed preamble on a non-embed stage.
    res_in: RefCell<Option<Vec<f32>>>,
    /// This stage's OUTPUT-side residual (`res[shard.end]`, pre-head-stage
    /// epilogue) - set by [`Self::run_stage_forward`], read by
    /// [`Self::read_out_res`].
    res_out: RefCell<Option<Vec<f32>>>,
    /// The head stage's last epilogue result - the model's actual output,
    /// distinct from `res_out` (which never runs through the head-stage
    /// epilogue).
    stage_out: RefCell<Option<Vec<f32>>>,
}

impl DitStage {
    /// The whole (non-sharded) model, from a typed [`DitWeights`] - the
    /// ordinary entry point every caller outside this module's own tests
    /// uses.
    pub fn new(cfg: DitConfig, weights: &DitWeights) -> DitStage {
        let shard = Shard::whole(cfg.num_layers as usize);
        DitStage::build(cfg, flatten_weights(weights), shard)
    }

    fn build(cfg: DitConfig, w: HashMap<String, Vec<f32>>, shard: Shard) -> DitStage {
        DitStage { cfg, w, shard, batch: RefCell::new(None), res_in: RefCell::new(None), res_out: RefCell::new(None), stage_out: RefCell::new(None) }
    }

    /// Build one pipeline stage from a flat checkpoint (`Model`/`Shardable`'s
    /// own representation) - `Shardable::new_shard`/`Model::new` both
    /// delegate here (`Shard::whole` for the latter).
    pub(crate) fn from_flat_weights(cfg: DitConfig, init: &HashMap<String, Vec<f32>>, shard: Shard) -> DitStage {
        DitStage::build(cfg, shard_weights(&cfg, init, &shard), shard)
    }

    pub fn config(&self) -> &DitConfig {
        &self.cfg
    }
    pub fn shard(&self) -> &Shard {
        &self.shard
    }

    /// Set this stage's diffusion batch - see [`DitStageBatch`]'s doc.
    pub fn load_shard_batch(&self, b: DitStageBatch) {
        *self.batch.borrow_mut() = Some(b);
    }

    /// The head stage's last epilogue result (after [`Self::run_stage_forward`]
    /// has run).
    pub fn take_stage_output(&self) -> Vec<f32> {
        self.stage_out.borrow().clone().expect("DitStage::take_stage_output: run_stage_forward has not produced a head-stage output yet")
    }

    pub(crate) fn weight_names(&self) -> Vec<String> {
        self.w.keys().cloned().collect()
    }
    pub(crate) fn weight(&self, name: &str) -> Vec<f32> {
        self.w.get(name).unwrap_or_else(|| panic!("DitStage: no such weight {name:?}")).clone()
    }

    pub(crate) fn read_out_res(&self) -> Vec<f32> {
        self.res_out.borrow().clone().expect("DitStage::read_out_res: run_stage_forward has not run yet")
    }
    pub(crate) fn write_in_res(&self, data: &[f32]) {
        *self.res_in.borrow_mut() = Some(data.to_vec());
    }

    /// Run this stage's forward: the embed preamble (`preprocess_conv` +
    /// `proj_in`, embed stage only, from `batch.latents`/`batch.condition`)
    /// or the previous stage's residual ([`Self::write_in_res`]) -> this
    /// stage's own block range (`dit::block_fwd`, reused unchanged from the
    /// served path) -> the head epilogue (`proj_out` + `postprocess_conv`,
    /// head stage only). Every stage independently recomputes its own
    /// timestep token from `batch.timestep` and its OWN (replicated)
    /// `time_proj`/`time_embed` weights - the only thing that actually
    /// crosses the stage boundary is the residual (see [`shard_owns_weight`]'s
    /// doc). Returns the head stage's MSE loss against `batch.target`
    /// (`None` with no target, or on a non-head stage).
    pub fn run_stage_forward(&self) -> Option<f32> {
        let cfg = &self.cfg;
        let inner = cfg.inner_dim() as usize;
        let batch_ref = self.batch.borrow();
        let batch = batch_ref.as_ref().expect("DitStage::run_stage_forward: no batch (call load_shard_batch first)");
        let length = batch.length;
        let rows = length + 1;
        let gpu = Gpu::new_cpu(dit::PIPELINES);

        let x0 = if self.shard.embed {
            let hidden_lc = dit::preprocess_hidden_lc(&gpu, cfg, &self.weight("preprocess_conv.weight"), &batch.latents, &batch.condition, length);
            let temb = dit::timestep_embed(
                &self.weight("time_proj.weight"),
                &self.weight("time_embed.linear_1.weight"),
                &self.weight("time_embed.linear_1.bias"),
                &self.weight("time_embed.linear_2.weight"),
                &self.weight("time_embed.linear_2.bias"),
                cfg,
                batch.timestep,
            );
            let proj_rows = dit::proj_in_rows(cfg, &self.weight("proj_in.weight"), &hidden_lc, length);
            let mut x_host = vec![0.0f32; rows * inner];
            x_host[..inner].copy_from_slice(&temb);
            x_host[inner..].copy_from_slice(&proj_rows);
            x_host
        } else {
            self.res_in.borrow().clone().expect("DitStage::run_stage_forward: non-embed stage needs write_in_res first")
        };
        assert_eq!(x0.len(), rows * inner, "DitStage::run_stage_forward: residual length mismatch");

        let (cos_t, sin_t) = dit::rope_tables(rows, cfg.rotary_dim as usize, 10000.0);
        let cos_b = gpu.storage_init("rope.cos", &cos_t);
        let sin_b = gpu.storage_init("rope.sin", &sin_t);

        let local_blocks: Vec<BlockW> = (self.shard.start..self.shard.end).map(|l| block_from_flat(&self.w, l)).collect();
        let device_blocks = dit::upload_blocks(&gpu, &local_blocks);
        let mut x = gpu.storage_init("x_stage_in", &x0);
        for db in &device_blocks {
            x = dit::block_fwd(&gpu, cfg, db, &x, &cos_b, &sin_b, rows);
        }
        let x_final = gpu.read(&x, rows * inner);
        *self.res_out.borrow_mut() = Some(x_final.clone());

        if self.shard.head {
            let out = dit::proj_out_postprocess(&gpu, cfg, &self.weight("proj_out.weight"), &self.weight("postprocess_conv.weight"), &x_final[inner..], length);
            *self.stage_out.borrow_mut() = Some(out.clone());
            batch.target.as_ref().map(|target| {
                assert_eq!(out.len(), target.len(), "DitStage::run_stage_forward: target length mismatch");
                out.iter().zip(target).map(|(o, g)| (o - g) * (o - g)).sum::<f32>() / out.len().max(1) as f32
            })
        } else {
            None
        }
    }
}

impl Model for DitStage {
    type Config = DitConfig;

    fn new(cfg: DitConfig, _b: u32, _t: u32, init: &HashMap<String, Vec<f32>>) -> DitStage {
        DitStage::from_flat_weights(cfg, init, Shard::whole(cfg.num_layers as usize))
    }
    fn init_weights(cfg: &DitConfig, seed: u64) -> HashMap<String, Vec<f32>> {
        flatten_weights(&crate::dit_train::random_weights(cfg, seed))
    }
    fn config(&self) -> &DitConfig {
        DitStage::config(self)
    }

    /// A documented no-op - see this module's doc ("`model::Batch` does not
    /// fit a diffusion batch"). Use [`DitStage::load_shard_batch`].
    fn set_batch(&self, _batch: Batch) {}

    fn forward(&self) -> f32 {
        self.run_stage_forward().unwrap_or(0.0)
    }
    fn backward(&self) {
        unimplemented!("minimaxmusic3::DitStage: no backward pass exists in this pipeline-sharding slice (dit_train::Trainer serves this crate's real, single-device training story) - see this module's doc");
    }
    fn zero_grads(&self) {
        // Nothing to zero - there is no gradient accumulator (see `backward`'s doc).
    }
    fn adamw_step(&self, _t: u32, _lr: f32, _wd: f32, _clip: Option<f32>, _extra_scale: f32) {
        // No-op: with no working `backward`, there is never a real gradient to apply here.
    }
    fn poll_wait(&self) {
        // Nothing to wait for: `run_stage_forward` ends in a blocking
        // `gpu.read` of the stage residual (and, on the head shard, several
        // more inside `dit::proj_out_postprocess`), so every stage this
        // trait can observe has already drained its queue by the time it
        // returns. If a stage ever stops reading its own result back to the
        // host, this becomes a real wait and must call `Gpu::poll_wait`.
        // (`dit::block_fwd` used to be the thing that guaranteed this, per
        // block; it no longer reads anything back - it only `flush`es, which
        // does not wait.)
    }

    fn param_names(&self) -> Vec<String> {
        self.weight_names()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.weight(name)
    }
    fn write_weight(&self, _name: &str, _data: &[f32]) {
        unimplemented!("minimaxmusic3::DitStage: weights are immutable post-construction in this slice (no backward/optimizer path exists) - see this module's doc");
    }
    fn read_grad(&self, _name: &str) -> Vec<f32> {
        unimplemented!("minimaxmusic3::DitStage: no gradients exist yet - see this module's doc");
    }

    fn logits_all(&self, _tokens: &[u32]) -> Option<Vec<f32>> {
        None // no token-classification head - a continuous-latent diffusion transformer
    }

    fn save(&self, _path: &str) {
        // No-op: `dit::import`/`dit_train::random_weights` already cover
        // this crate's own weight construction paths.
    }
    fn config_json(&self) -> serde_json::Value {
        DitStage::config(self).to_json()
    }
}

impl Shardable for DitStage {
    fn shard_cost(cfg: &DitConfig, b: u32, t: u32) -> ShardCost {
        let dim = cfg.inner_dim() as f64;
        let ff_inner = cfg.ff_inner_dim as f64;
        let concat = cfg.concat_channels() as f64;
        let cin = cfg.in_channels as f64;
        let fourier = cfg.fourier_embedding_dim as f64;
        // `t + 1`: every stage's own row count includes the prepended
        // timestep token (`DitStage::run_stage_forward`'s own `rows`).
        let tt = (b.max(1) as f64) * (t as f64 + 1.0);
        // Self-attention: QKVO projections (4*dim^2 per token) plus the
        // O(T^2) score/weighted-sum term (this DiT's attention is always
        // the full bidirectional T-token attention, no KV-cache
        // amortization, matching `ltxv::shard::shard_cost`'s own reasoning
        // for its diffusion transformer).
        let self_attn = 4.0 * dim * dim * tt + 2.0 * tt * tt * dim;
        // Fused gated FFN: `ff_in` (dim -> 2*ff_inner, the gate+up fusion)
        // plus `ff_out` (ff_inner -> dim).
        let ffn = (2.0 * dim * ff_inner + ff_inner * dim) * tt;
        let per_layer = self_attn + ffn;
        // Embed stage: `preprocess_conv` (concat -> concat) + `proj_in`
        // (concat -> dim) + the Fourier timestep MLP (fourier -> dim -> dim).
        let embed = concat * concat + concat * dim + fourier * dim + dim * dim;
        // Head stage: `proj_out` (dim -> cin) + `postprocess_conv` (cin -> cin).
        let head = dim * cin + cin * cin;
        ShardCost { n_layers: cfg.num_layers as usize, per_layer, embed, head, boundary_words: (b.max(1) * (t + 1)) as usize * cfg.inner_dim() as usize }
    }

    fn new_shard(cfg: DitConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, shard: Shard) -> DitStage {
        let _ = (b, t); // see `DitStage::from_flat_weights`'s doc: construction needs only the weight subset
        DitStage::from_flat_weights(cfg, init, shard)
    }

    fn replicated_params(&self) -> Vec<String> {
        // Every stage loads its own copy of `time_proj.*`/`time_embed.*`
        // (this module's doc: "who computes what"), so these names appear
        // in every stage's `param_names()` and would need their gradient
        // SUMMED across holders if a backward pass existed.
        Model::param_names(self).into_iter().filter(|n| n.starts_with("time_proj.") || n.starts_with("time_embed.")).collect()
    }

    fn run_forward_stage(&self) -> Option<f32> {
        self.run_stage_forward()
    }
    fn run_backward_stage(&self) {
        unimplemented!("minimaxmusic3::DitStage: no backward pass exists yet - see this module's doc for the tracked gap");
    }
    fn read_out_res(&self) -> Vec<f32> {
        DitStage::read_out_res(self)
    }
    fn write_in_res(&self, data: &[f32]) {
        DitStage::write_in_res(self, data)
    }
    fn read_in_dres(&self) -> Vec<f32> {
        unimplemented!("minimaxmusic3::DitStage: no backward pass exists yet - see this module's doc for the tracked gap");
    }
    fn write_out_dres(&self, _data: &[f32]) {
        unimplemented!("minimaxmusic3::DitStage: no backward pass exists yet - see this module's doc for the tracked gap");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_json_round_trips() {
        let cfg = DitConfig::tiny();
        let v = cfg.to_json();
        let back = DitConfig::from_json(&v);
        assert_eq!(cfg.in_channels, back.in_channels);
        assert_eq!(cfg.condition_dim, back.condition_dim);
        assert_eq!(cfg.num_layers, back.num_layers);
        assert_eq!(cfg.num_attention_heads, back.num_attention_heads);
        assert_eq!(cfg.attention_head_dim, back.attention_head_dim);
        assert_eq!(cfg.ff_inner_dim, back.ff_inner_dim);
        assert_eq!(cfg.rotary_dim, back.rotary_dim);
        assert_eq!(cfg.fourier_embedding_dim, back.fourier_embedding_dim);
    }

    /// Every name in [`dit_tensor_manifest`] is claimed by exactly the
    /// stage(s) [`shard_owns_weight`] says it should be - a whole shard
    /// (`Shard::whole`) must cover EVERY name exactly once (the filter is
    /// exhaustive, nothing silently dropped), and a genuine partial shard
    /// must load STRICTLY FEWER floats than the whole model (`new_shard`
    /// really does skip the rest of the stack, not build it and discard it).
    #[test]
    fn new_shard_loads_only_its_own_weight_subset() {
        let cfg = DitConfig::tiny();
        let init = <DitStage as Model>::init_weights(&cfg, 1);

        let whole = Shard::whole(cfg.num_layers as usize);
        let full = <DitStage as Shardable>::new_shard(cfg, 1, 4, &init, whole);
        let full_names: std::collections::HashSet<String> = Model::param_names(&full).into_iter().collect();
        let manifest_names: std::collections::HashSet<String> = dit_tensor_manifest(&cfg).into_iter().map(|(n, _)| n).collect();
        assert_eq!(full_names, manifest_names, "a whole shard must load exactly the full manifest, no more, no less");

        assert!(cfg.num_layers >= 2, "test assumes >=2 layers so a partial shard is meaningful");
        let partial_shard = Shard { start: 0, end: 1, embed: true, head: false, gpu_index: Shard::ANY_GPU };
        let partial = <DitStage as Shardable>::new_shard(cfg, 1, 4, &init, partial_shard);
        let partial_total: usize = Model::param_names(&partial).iter().map(|n| Model::read_weight(&partial, n).len()).sum();
        let full_total: usize = Model::param_names(&full).iter().map(|n| Model::read_weight(&full, n).len()).sum();
        assert!(partial_total < full_total, "a 1-of-{}-layer shard ({partial_total} floats) must be smaller than the whole model ({full_total} floats)", cfg.num_layers);

        for name in Model::param_names(&partial) {
            if let Some(rest) = name.strip_prefix("transformer_blocks.") {
                let l: usize = rest.split('.').next().unwrap().parse().unwrap();
                assert_eq!(l, 0, "partial shard [0,1) must not own block {l}'s weights ({name})");
            }
        }
    }

    fn assert_well_formed(shards: &[Shard], n_layers: usize) {
        assert_eq!(shards[0].start, 0, "first stage must start at layer 0");
        assert_eq!(shards.last().unwrap().end, n_layers, "last stage must end at n_layers");
        for w in shards.windows(2) {
            assert_eq!(w[0].end, w[1].start, "stage layer ranges must be contiguous with no gap or overlap");
            assert!(w[0].end > w[0].start, "no stage may be empty");
        }
        assert!(shards.last().unwrap().end > shards.last().unwrap().start, "no stage may be empty");
        assert_eq!(shards.iter().filter(|s| s.embed).count(), 1, "exactly one embed stage");
        assert_eq!(shards.iter().filter(|s| s.head).count(), 1, "exactly one head stage");
    }

    #[test]
    fn plan_balanced_is_well_formed_for_the_tiny_config() {
        let cfg = DitConfig::tiny();
        let cost = <DitStage as Shardable>::shard_cost(&cfg, 1, 3);
        assert_eq!(cost.n_layers, cfg.num_layers as usize);
        let shards = model::plan_balanced(&cost, &[0, 1]);
        assert_eq!(shards.len(), 2);
        assert_well_formed(&shards, cfg.num_layers as usize);
    }

    /// The plan can be COMPUTED for the real 36-layer, 2048-wide config even
    /// though it cannot be built or run on this machine (no discrete GPU) -
    /// `shard_cost` only reads `cfg`'s plain numeric fields.
    #[test]
    fn plan_balanced_is_well_formed_for_the_real_config_shape() {
        let cfg = DitConfig::real();
        let cost = <DitStage as Shardable>::shard_cost(&cfg, 1, 200);
        assert_eq!(cost.n_layers, 36);
        for k in [2usize, 4, 8] {
            let gpus: Vec<usize> = (0..k).collect();
            let shards = model::plan_balanced(&cost, &gpus);
            assert_eq!(shards.len(), k);
            assert_well_formed(&shards, 36);
        }
    }
}
