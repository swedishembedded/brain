// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Ripple as a [`model::Shardable`]: the generic pipeline cuts its 80 layers over several devices.
//!
//! Swedish Embedded AB implements solutions for training very large dense decoders across the GPUs a team
//! actually has, with the same result as one device. If your team needs expertise in pipeline-parallel training
//! of hybrid-attention language models then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! One thing crosses the cut besides the residual stream: the value residual reads `x0`, the scaled embedding,
//! in every layer. Rather than ship it, each stage holds the embedding table and recomputes `x0` from the
//! tokens (every stage is given the batch); the backward of a later stage turns its share of `x0`'s gradient
//! into an embedding gradient of its own, and the pipeline sums the copies' gradients as it does for any
//! parameter held by several stages. A model without the value residual keeps the embedding on the first
//! stage alone.

use std::collections::HashMap;

use gpu_core::select::Dtype;
use model::ops::TierPolicy;
use model::{Model, Shard, ShardCost, Shardable};

use super::{Ripple, TrainConfig, TrainOpts};


/// Device bytes training the stage `shard` of `cfg` needs: its linears at the tier `tiers` holds them (f32 and four
/// times over, with the gradient and both AdamW moments, when they train), the f32 tensors beside them, the
/// adapter (weight, gradient, moments), and the activations of a `b x t` batch through its layers. A planning
/// figure with the bounds of one: weights are exact, the activations are the build's own count.
pub fn estimate_stage_bytes(cfg: &TrainConfig, shard: &Shard, tiers: &TierPolicy, b: u32, t: u32) -> u64 {
    use model::ops::Weight;
    let training_full = cfg.lora.is_none();
    let (mut weights, mut trained) = (0u64, 0u64);
    let mut linear = std::collections::HashSet::new();
    for (name, n, k) in crate::weights::all_linears(&cfg.model).into_iter().filter(|(name, ..)| cfg.shard_owns(name, shard)) {
        let elems = (n * k) as u64;
        if training_full {
            weights += 4 * elems;
            trained += 12 * elems;
        } else {
            weights += Weight::bytes_for(tiers.want(&name), n, k);
        }
        linear.insert(name);
    }
    for (name, numel) in cfg.shard_param_list(shard).into_iter().filter(|(name, _)| !linear.contains(name)) {
        weights += 4 * numel as u64;
        if training_full || model::adapter::device::is_adapter_param(&name) {
            trained += 12 * numel as u64;
        }
    }
    let mut stage_cfg = cfg.clone();
    stage_cfg.model.num_hidden_layers = shard.end - shard.start;
    weights + trained + super::graph::activation_bytes(&stage_cfg, b, t, model::saved::SavedFormat::F32, true)
}

impl Shardable for Ripple {
    fn shard_cost(cfg: &TrainConfig, b: u32, t: u32) -> ShardCost {
        let c = &cfg.model;
        let (d, ff) = (c.hidden_size as f64, c.intermediate_size as f64);
        let vocab = c.vocab_size as f64;
        // Parameter count of a layer's linears: weight memory is what the cuts balance.
        let per_layer = 2.0 * c.q_dim() as f64 * d + 2.0 * c.kv_dim() as f64 * d + if c.use_gated_attention { c.q_dim() as f64 * d } else { 0.0 } + 3.0 * d * ff;
        // With the value residual every later stage holds the embedding as well, so it costs the same everywhere
        // but the head: the cut cannot move it.
        ShardCost { n_layers: c.num_hidden_layers, per_layer, embed: vocab * d, head: vocab * d, boundary_words: (b * t) as usize * c.hidden_size }
    }

    fn new_shard(cfg: TrainConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, shard: Shard) -> Ripple {
        Ripple::build_shard(cfg, b, t, init, &TrainOpts::default(), true, shard).unwrap_or_else(|e| panic!("{e}"))
    }

    fn new_shard_from(cfg: TrainConfig, b: u32, t: u32, init: &dyn checkpoint::TensorSource, shard: Shard, dt: Dtype) -> Ripple {
        let opts = TrainOpts { tiers: TierPolicy::uniform(dt), ..TrainOpts::default() };
        Ripple::build_shard(cfg, b, t, init, &opts, true, shard).unwrap_or_else(|e| panic!("{e}"))
    }

    /// The stages' tensors, each once (the embedding is held by every stage and equal in all of them after a step), in
    /// the file a one-device run writes. A pipeline of an adapter run holds no base to write: refused.
    fn save_stages(stages: &[Ripple], path: &str) -> Result<(), String> {
        let first = stages.first().ok_or("ricursive: a pipeline of no stages")?;
        if first.cfg.lora.is_some() || stages.iter().any(Ripple::external_base) {
            return Err("ricursive: a pipeline of an adapter run holds no base to write; save the adapter".to_string());
        }
        let mut seen = std::collections::HashSet::new();
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = stages.iter().flat_map(Ripple::saved_tensors).filter(|(name, ..)| seen.insert(name.clone())).collect();
        let card = checkpoint::st::ModelCard::new("brain/ricursive", "ricursive");
        checkpoint::save_carded(path, first.cfg.to_json(), &tensors, &card);
        Ok(())
    }

    fn run_forward_stage(&self) -> Option<f32> {
        if self.shard.head {
            Some(Model::forward(self))
        } else {
            self.run_forward();
            None
        }
    }

    fn run_backward_stage(&self) {
        self.run_backward();
    }

    fn read_out_res(&self) -> Vec<f32> {
        self.gpu.read(&self.res[self.shard.end], self.rows() as usize * self.cfg.model.hidden_size)
    }

    fn write_in_res(&self, data: &[f32]) {
        assert!(self.shard.start > 0, "ricursive: the first stage takes tokens, not a residual");
        self.res.ingest(&self.gpu, self.shard.start, data);
    }

    fn read_in_dres(&self) -> Vec<f32> {
        let g = self.grad.as_ref().expect("a backward needs the training build");
        self.gpu.read(&g.dres[self.shard.start % 2], self.rows() as usize * self.cfg.model.hidden_size)
    }

    fn write_out_dres(&self, data: &[f32]) {
        let g = self.grad.as_ref().expect("a backward needs the training build");
        self.gpu.write_f32(&g.dres[self.shard.end % 2], data);
    }
}
