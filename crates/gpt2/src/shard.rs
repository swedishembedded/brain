// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GPT implements the generic [`model::Shardable`] seam so the shared
//! [`model::Pipeline`] can pipeline-parallel it across GPUs. The layer-range
//! parameterisation of the forward/backward graph lives in [`crate::model`];
//! this just wires it onto the trait. GPT's lm_head is untied, so no parameter is
//! replicated across stages (`replicated_params` is empty).

use model::{Shard, ShardCost, Shardable};

use crate::model::{Gpt, GptConfig};

impl Shardable for Gpt {
    fn shard_cost(cfg: &GptConfig, b: u32, t: u32) -> ShardCost {
        let d = cfg.d_model as f64;
        let ff = cfg.d_ff as f64;
        let vocab = cfg.vocab as f64;
        // Parameter-count proxy (balances weight memory across cards).
        let per_layer = 4.0 * d * d + 2.0 * d * ff; // qkv + out + fc + proj (biases ~negligible)
        ShardCost {
            n_layers: cfg.n_layers as usize,
            per_layer,
            embed: vocab * d + cfg.block_size as f64 * d, // tok + pos embeddings
            head: vocab * d,                              // untied lm_head
            boundary_words: (b * t) as usize * cfg.d_model as usize,
        }
    }

    fn new_shard(cfg: GptConfig, b: u32, t: u32, init: &std::collections::HashMap<String, Vec<f32>>, shard: Shard) -> Gpt {
        Gpt::new_shard(cfg, b, t, init, shard)
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
        Gpt::read_out_res(self)
    }
    fn write_in_res(&self, data: &[f32]) {
        Gpt::write_in_res(self, data)
    }
    fn read_in_dres(&self) -> Vec<f32> {
        Gpt::read_in_dres(self)
    }
    fn write_out_dres(&self, data: &[f32]) {
        Gpt::write_out_dres(self, data)
    }
}
