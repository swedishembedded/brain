// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepseekV2 implements the generic [`model::Shardable`] seam so the shared
//! [`model::Pipeline`] can pipeline-parallel it across GPUs. The layer-range
//! parameterisation of the forward/backward graph and the `res`/`dres`
//! cross-stage boundary buffers all live in [`crate::model`]; this file just
//! wires those onto the trait. Mirrors `qwen35moe::shard`'s and
//! `gpt2::shard`'s shape exactly (see either for the reasoning behind each
//! method - it does not differ here).
//!
//! Two consumers share this decoder (DeepSeek-OCR and DeepSeek-OCR-2, see
//! `crate`'s own lib doc) and neither wired sharding for itself before this -
//! `scripts/gates/check-multi-gpu-sharding.sh`'s `deepseek2ocr` allow-list row
//! is removed in the same change that adds this file, since the gate walks
//! `arch!()` rows back to the CRATE that implements `Shardable`, and this is
//! that crate for both.
//!
//! `tok.weight`/`lm_head.weight` are **untied** for the real checkpoint
//! (`DeepseekV2Config::head_weight` only returns `"tok.weight"` when
//! `shape.tie_embeddings` is set, which the real config never does), so
//! [`Shardable::replicated_params`] reports nothing to sum - the same
//! untied branch `qwen35moe::Qwen35`'s own impl takes.

use model::{Shard, ShardCost, Shardable};

use crate::config::DeepseekV2Config;
use crate::model::DeepseekV2;

impl Shardable for DeepseekV2 {
    fn shard_cost(cfg: &DeepseekV2Config, b: u32, t: u32) -> ShardCost {
        let d = cfg.d_model() as f64;
        let vocab = cfg.vocab() as f64;
        let dense_ff = cfg.ffn_hidden() as f64;
        let shared_ff = cfg.shared_ff() as f64;
        let moe_ff = cfg.moe_ff() as f64;
        let n_experts = cfg.n_experts() as f64;

        // Plain MHA mixer: q/k/v/o, all square [d,d] (this crate's own
        // `new_on` asserts `q_dim == d_model`).
        let cost_mha = 4.0 * d * d;

        // MLP sublayer: dense on the leading blocks, MoE (router + experts +
        // fused shared expert) on the rest. Folded into one frequency-
        // weighted average per layer, same reasoning as `qwen35moe::shard`'s
        // own averaging over its two mixer types - `plan_balanced`'s cost
        // model is one scalar per layer, and MoE's cost dominates by a wide
        // margin regardless of how the average comes out.
        let n_layers = cfg.n_layers() as f64;
        let n_dense = 1.0f64.min(n_layers).max(0.0); // leading_dense_block_count = 1
        let n_moe = (n_layers - n_dense).max(0.0);
        let cost_dense_mlp = 3.0 * d * dense_ff; // gate+up+down
        let cost_moe_mlp = n_experts * d // router
            + n_experts * 3.0 * d * moe_ff // routed experts (gate+up+down)
            + 3.0 * d * shared_ff; // fused shared expert (gate+up+down)
        let mlp_avg = (n_dense * cost_dense_mlp + n_moe * cost_moe_mlp) / n_layers.max(1.0);
        let cost_ln = 2.0 * d; // ln1 + ln2

        ShardCost {
            n_layers: cfg.n_layers() as usize,
            per_layer: cost_mha + mlp_avg + cost_ln,
            embed: vocab * d,
            head: vocab * d,
            boundary_words: (b * t) as usize * cfg.d_model() as usize,
        }
    }

    fn new_shard(cfg: DeepseekV2Config, b: u32, t: u32, init: &std::collections::HashMap<String, Vec<f32>>, shard: Shard) -> DeepseekV2 {
        DeepseekV2::new_shard(cfg, b, t, init, shard)
    }

    fn run_forward_stage(&self) -> Option<f32> {
        if self.shard.head {
            Some(self.forward())
        } else {
            self.forward_submit();
            None
        }
    }
    fn run_backward_stage(&self) {
        self.backward();
    }
    fn read_out_res(&self) -> Vec<f32> {
        DeepseekV2::read_out_res(self)
    }
    fn write_in_res(&self, data: &[f32]) {
        DeepseekV2::write_in_res(self, data)
    }
    fn read_in_dres(&self) -> Vec<f32> {
        DeepseekV2::read_in_dres(self)
    }
    fn write_out_dres(&self, data: &[f32]) {
        DeepseekV2::write_out_dres(self, data)
    }
}
