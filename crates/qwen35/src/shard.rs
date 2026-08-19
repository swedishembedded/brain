// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen35 implements the generic [`model::Shardable`] seam so the shared
//! [`model::Pipeline`] can pipeline-parallel it across GPUs. All the heavy
//! lifting - the layer-range parameterisation of the forward/backward graph
//! and the cross-stage residual/gradient boundary buffers - lives in
//! [`crate::model`]; this file just wires those onto the trait. Mirrors
//! `qwen35moe::shard`'s shape exactly (see that file for the reasoning
//! behind each method).
//!
//! `tok.weight`/`lm_head.weight` are **untied** for this model
//! (`Qwen35Config::tie_embeddings` is `false` in both [`Qwen35Config::tiny`]
//! and the real [`Qwen35Config::qwen38_27b`] shape), so
//! [`Shardable::replicated_params`] reports nothing to sum. `cfg.mtp`
//! requires a whole shard (`Qwen35::new_impl_on`'s own assert) - the MTP
//! head needs `res[n_layers]` and the shared `lm_head`, both only valid on a
//! whole shard; pipeline-sharding a model with MTP enabled panics loudly at
//! construction rather than silently producing a wrong MTP loss.

use model::{Shard, ShardCost, Shardable};

use crate::config::{LayerType, Qwen35Config};
use crate::model::Qwen35;

impl Shardable for Qwen35 {
    fn shard_cost(cfg: &Qwen35Config, b: u32, t: u32) -> ShardCost {
        let d = cfg.d_model as f64;
        let vocab = cfg.vocab as f64;

        // GDN (Linear) mixer: in_proj_{qkv,z,b,a} + conv1d + out_proj (norm/
        // A_log/dt_bias are tiny, folded in for completeness, not because
        // they move the needle).
        let conv_dim = cfg.linear_conv_dim() as f64;
        let vdim = cfg.linear_value_dim() as f64;
        let nvh = cfg.linear_num_value_heads as f64;
        let k = cfg.linear_conv_kernel_dim as f64;
        let hvd = cfg.linear_value_head_dim as f64;
        let cost_gdn = conv_dim * d          // in_proj_qkv
            + vdim * d                        // in_proj_z
            + 2.0 * nvh * d                   // in_proj_b + in_proj_a
            + conv_dim * k                     // conv1d.weight
            + 2.0 * nvh                        // A_log + dt_bias
            + hvd                              // norm.weight
            + d * vdim; // out_proj

        // GQA (Full) mixer: q_proj (doubled value+gate) + k_proj + v_proj +
        // q/k norm + o_proj.
        let hqp = cfg.q_proj_dim() as f64;
        let hq = cfg.q_dim() as f64;
        let hkv = cfg.kv_dim() as f64;
        let hd = cfg.head_dim as f64;
        let cost_gqa = hqp * d + 2.0 * hkv * d + 2.0 * hd + d * hq;

        // Dense SwiGLU MLP: identical on every layer regardless of mixer
        // type (`Qwen35Config::param_list`'s own doc) - no MoE experts/
        // router here, unlike qwen35moe.
        let ff = cfg.intermediate_size as f64;
        let cost_mlp = 3.0 * ff * d; // gate + up + down
        let cost_ln = 2.0 * d; // ln1 + ln2, both layer types

        // `plan_balanced`'s cost model is a single scalar per layer (its DP
        // treats every layer as identical cost), so the alternating GDN/GQA
        // mixer cost is folded into one frequency-weighted average.
        let types = cfg.layer_types();
        let n_gdn = types.iter().filter(|t| **t == LayerType::Linear).count().max(1) as f64;
        let n_gqa = types.iter().filter(|t| **t == LayerType::Full).count() as f64;
        let n_layers_f = (n_gdn + n_gqa).max(1.0);
        let mixer_avg = (n_gdn * cost_gdn + n_gqa * cost_gqa) / n_layers_f;
        let per_layer = mixer_avg + cost_mlp + cost_ln;

        // tok.weight / lm_head.weight: untied (see this module's own doc),
        // so each is its own vocab*d tensor.
        ShardCost {
            n_layers: cfg.n_layers as usize,
            per_layer,
            embed: vocab * d,
            head: vocab * d,
            boundary_words: (b * t) as usize * cfg.d_model as usize,
        }
    }

    fn new_shard(cfg: Qwen35Config, b: u32, t: u32, init: &std::collections::HashMap<String, Vec<f32>>, shard: Shard) -> Qwen35 {
        Qwen35::new_shard(cfg, b, t, init, shard)
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
        self.backward();
    }
    fn read_out_res(&self) -> Vec<f32> {
        Qwen35::read_out_res(self)
    }
    fn write_in_res(&self, data: &[f32]) {
        Qwen35::write_in_res(self, data)
    }
    fn read_in_dres(&self) -> Vec<f32> {
        Qwen35::read_in_dres(self)
    }
    fn write_out_dres(&self, data: &[f32]) {
        Qwen35::write_out_dres(self, data)
    }
}
