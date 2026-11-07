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
