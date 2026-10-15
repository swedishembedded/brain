// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! What a Qwen3.5 training build asks the device for, measured with
//! `Gpu::charged_bytes` (the device-bytes-requested accounting) rather than
//! derived from the code that allocates. Each measurement runs in a child
//! process because the memory ceiling that turns the accounting on is a
//! process-wide `OnceLock`.

mod common;

use common::*;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

/// Every layer full-attention, so a decode KV cache is the biggest thing a
/// training build could wrongly keep per token.
fn all_gqa() -> Qwen35Config {
    Qwen35Config { n_layers: 16, full_attention_interval: 1, block_size: 512, lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny() }
}

/// A layer stack whose GDN recurrent state dwarfs its weights (wide heads, a
/// narrow model), so state a training build keeps shows up above the weights.
fn wide_gdn() -> Qwen35Config {
    Qwen35Config {
        d_model: 32,
        n_layers: 12,
        full_attention_interval: 12,
        linear_num_key_heads: 1,
        linear_num_value_heads: 1,
        linear_key_head_dim: 256,
        linear_value_head_dim: 256,
        lora: Some(lora_cfg(3, 6.0)),
        ..Qwen35Config::tiny()
    }
}

fn charged(cfg: &Qwen35Config, t: u32, base: Dtype) -> u64 {
    let init = qwen35::init::init_weights(cfg, 5); // seed: only the shapes matter here
    let m = Qwen35::new_train_opts_on(Gpu::new_cpu(pipelines()), cfg.clone(), 1, t, &init, &opts(base));
    m.gpu.charged_bytes()
}

#[test]
#[ignore = "child process helper, driven by a_training_build_keeps_no_per_token_decode_cache"]
fn child_charged_bytes_at_two_lengths() {
    publish_ceiling();
    let cfg = all_gqa();
    mark("short", charged(&cfg, 256, Dtype::F32));
    mark("long", charged(&cfg, 512, Dtype::F32));
}

/// A training build never decodes, so it must not hold the per-GQA-layer f32
/// KV cache a decode step reads (2 planes x kv_dim x 4 B per token per GQA
/// layer). The per-token slope of what the build requests is bounded by the
/// buffers a training step does need.
#[test]
fn a_training_build_keeps_no_per_token_decode_cache() {
    let m = measure_in_child("child_charged_bytes_at_two_lengths");
    let per_token = (m["long"] - m["short"]) / 256;
    let c = all_gqa();
    let (d, v, r) = (c.d_model as u64, c.vocab as u64, 3u64);
    let max_out = [c.linear_conv_dim(), c.linear_value_dim(), c.d_model, c.q_proj_dim(), c.kv_dim(), c.intermediate_size].into_iter().max().unwrap() as u64;
    // tokens, targets, ce | logits | residual boundaries | two boundary-grad
    // slabs | lora scratch | rope tables.
    let needed = 4 * (3 + v + (c.n_layers as u64 + 1) * d + 2 * d + 2 * r + max_out + c.rotary_dim() as u64);
    let kv = 4 * 2 * c.kv_dim() as u64 * c.n_layers as u64;
    assert!(per_token <= needed + kv / 2, "{per_token} B/token requested, {needed} needed by training; the decode KV cache would add {kv}");
}

#[test]
#[ignore = "child process helper, driven by a_training_build_keeps_no_recurrent_decode_state"]
fn child_charged_bytes_wide_gdn() {
    publish_ceiling();
    let cfg = wide_gdn();
    mark("charged", charged(&cfg, 12, Dtype::F32));
    let params: u64 = cfg.param_list().iter().map(|(_, n)| *n as u64 * 4).sum();
    mark("params", params);
    let gdn_layers = cfg.layer_types().iter().filter(|t| **t == qwen35::config::LayerType::Linear).count() as u64;
    mark("state", cfg.gdn_state_len() as u64 * 4 * gdn_layers);
}

/// The GDN recurrent state and conv history are decode-only too.
#[test]
fn a_training_build_keeps_no_recurrent_decode_state() {
    let m = measure_in_child("child_charged_bytes_wide_gdn");
    let (charged, params, state) = (m["charged"], m["params"], m["state"]);
    // Weights, the adapters' grads/moments and a few small buffers, but not `state`.
    assert!(charged < params + state / 2, "requested {charged} B for {params} B of parameters: the {state} B recurrent state is being kept");
}
