// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **An ADAPTER, not a sharding implementation.** This file contains no
//! device-placement or partitioning logic of its own and no other crate
//! should ever import sharding from here: the generic mechanism - how layers
//! are cut into per-device stages, both the training-time
//! `model::shard::plan_balanced` and the capacity-aware
//! `model::shard::plan_by_capacity`/`plan_fewest_devices` a resident model
//! places real weights with - lives in `crates/model/src/shard.rs`, which
//! depends on no model family and is what `omni` (and anything added later)
//! uses too. All this file does is describe Qwen to it.
//!
//! Qwen implements the generic [`model::Shardable`] seam so the shared
//! [`model::Pipeline`] can pipeline-parallel it across GPUs. All the heavy
//! lifting — the layer-range parameterisation of the forward/backward graph and
//! the cross-stage residual buffers — lives in [`crate::model`]; this file just
//! wires those onto the trait. Qwen's tied `tok.weight` is reported as a
//! replicated parameter so the pipeline sums its gradient across the embed and
//! head stages.

use model::{Shard, ShardCost, Shardable};

use crate::config::QwenConfig;
use crate::model::Qwen;

impl Shardable for Qwen {
    fn shard_cost(cfg: &QwenConfig, b: u32, t: u32) -> ShardCost {
        let d = cfg.d_model as f64;
        let ff = cfg.d_ff as f64;
        let vocab = cfg.vocab as f64;
        // Parameter-count proxy (balances weight memory across cards).
        let per_layer = 2.0 * cfg.q_dim() as f64 * d      // wq + wo
            + 2.0 * cfg.kv_dim() as f64 * d               // wk + wv
            + 3.0 * d * ff;                                // gate + up + down
        // tok.weight (embedding). Tied: the head stage holds a replica of it, so
        // both endpoints carry ~vocab*d — hence embed == head here.
        ShardCost {
            n_layers: cfg.n_layers as usize,
            per_layer,
            embed: vocab * d,
            head: vocab * d,
            boundary_words: (b * t) as usize * cfg.d_model as usize,
        }
    }

    fn new_shard(cfg: QwenConfig, b: u32, t: u32, init: &std::collections::HashMap<String, Vec<f32>>, shard: Shard) -> Qwen {
        Qwen::new_shard(cfg, b, t, init, true, shard)
    }

    fn replicated_params(&self) -> Vec<String> {
        if self.cfg.head_weight() == "tok.weight" {
            vec!["tok.weight".to_string()]
        } else {
            Vec::new()
        }
    }

    fn run_forward_stage(&self) -> Option<f32> {
        if self.shard.head {
            Some(self.forward())
        } else {
            self.run_forward();
            None
        }
    }
    fn run_backward_stage(&self) {
        self.run_backward();
    }
    fn read_out_res(&self) -> Vec<f32> {
        Qwen::read_out_res(self)
    }
    fn write_in_res(&self, data: &[f32]) {
        Qwen::write_in_res(self, data)
    }
    fn read_in_dres(&self) -> Vec<f32> {
        Qwen::read_in_dres(self)
    }
    fn write_out_dres(&self, data: &[f32]) {
        Qwen::write_out_dres(self, data)
    }
}
