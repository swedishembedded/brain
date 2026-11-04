// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The serving engine runs the Qwen2 and Llama config variants of the decoder
//! exactly as the model does: q/k/v biases (Qwen2), no QK-norm (both), full
//! multi-head attention with an untied head and a declared RoPE scaling
//! (Llama). `Qwen`'s KV-cache decode is the reference; the engine must match
//! it with fp32 KV, and within int8 quantization with int8 KV.

use std::collections::HashMap;

use model::paged::BlockTable;
use model::rope_scaling::RopeScaling;
use qwen3::model::PrefillInput;
use qwen3::{Qwen, QwenConfig};

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

const PROMPT: [u32; 6] = [2, 9, 4, 11, 7, 3];

/// `init_weights` with every q/k/v bias made non-zero, so a bias that is not
/// added shows up.
fn weights(cfg: &QwenConfig) -> HashMap<String, Vec<f32>> {
    let mut w = qwen3::init_weights(cfg, 5);
    for (name, v) in w.iter_mut() {
        if name.ends_with(".bias") {
            for (i, x) in v.iter_mut().enumerate() {
                *x = 0.2 * ((i as f32) * 0.71 + name.len() as f32).sin();
            }
        }
    }
    w
}

fn variants() -> [(QwenConfig, &'static str); 2] {
    [
        (QwenConfig { qk_norm: false, attn_bias: true, ..QwenConfig::tiny() }, "qwen2"),
        (
            QwenConfig {
                qk_norm: false,
                n_kv_heads: 4,
                tie_embeddings: false,
                rope_scaling: Some(RopeScaling::Linear { factor: 4.0 }),
                ..QwenConfig::tiny()
            },
            "llama",
        ),
    ]
}

#[test]
fn the_engine_serves_every_decoder_variant_as_the_model_computes_it() {
    if gpu_disabled() {
        return;
    }
    for (cfg, name) in variants() {
        let w = weights(&cfg);
        let dec = Qwen::from_tensors_decode(cfg.clone(), &w, cfg.block_size);
        let inputs: Vec<PrefillInput> = PROMPT.iter().map(|&t| PrefillInput::Token(t)).collect();
        let want = dec.prefill(&inputs);
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for (kv_int8, tol) in [(false, 1e-4), (true, 5e-2)] {
            let mut eng = qwen3::serve::Engine::from_map(cfg.clone(), &w, 4, 32, 1, 12, 8, kv_int8, false);
            let got = eng.prefill_for_perf(&mut BlockTable::new(), &PROMPT);
            let err = got.iter().zip(&want).fold(0.0f32, |m, (a, b)| m.max((a - b).abs())) / scale;
            assert!(err < tol, "{name} (kv_int8 {kv_int8}): engine differs from the model by {err:e} relative");
        }
    }
}
