// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Every qwen3 path rotates at a declared RoPE scaling: the batched forward
//! (training and `logits_all`), the KV-cache decode, and the serving engine.
//!
//! The serving engine reads the scaled table through its own kernel, so it is
//! the oracle for the model's two paths, and the model's two paths are each
//! other's. A path that ignored the scaling would fall out of step with the
//! others, and `scaling_changes_the_output` makes sure there is a difference
//! to fall out of step with.

use model::paged::BlockTable;
use model::rope_scaling::RopeScaling;
use qwen3::model::PrefillInput;
use qwen3::{Qwen, QwenConfig};

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

const PROMPT: [u32; 7] = [1, 5, 3, 9, 2, 7, 4];

fn cfg(scaling: Option<RopeScaling>) -> QwenConfig {
    QwenConfig { rope_scaling: scaling, ..QwenConfig::tiny() }
}

fn scalings() -> [(RopeScaling, &'static str); 2] {
    [
        (RopeScaling::Linear { factor: 4.0 }, "linear"),
        (RopeScaling::Llama3 { factor: 8.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max_position_embeddings: 4 }, "llama3"),
    ]
}

/// `(decode hidden, decode logits, forward logits of the last row)`.
fn model_outputs(cfg: &QwenConfig, init: &std::collections::HashMap<String, Vec<f32>>) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dec = Qwen::from_tensors_decode(cfg.clone(), init, cfg.block_size);
    let inputs: Vec<PrefillInput> = PROMPT.iter().map(|&t| PrefillInput::Token(t)).collect();
    let hidden = dec.prefill(&inputs);
    let dec_logits = dec.decode_logits();
    let full = Qwen::new(cfg.clone(), 1, cfg.block_size, init);
    let v = cfg.vocab as usize;
    let all = full.logits_all(&PROMPT);
    (hidden, dec_logits, all[(PROMPT.len() - 1) * v..].to_vec())
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn every_path_rotates_at_the_declared_scaling() {
    if gpu_disabled() {
        return;
    }
    for (scaling, name) in scalings() {
        let cfg = cfg(Some(scaling));
        let init = qwen3::init_weights(&cfg, 3);
        let (hidden, dec_logits, fwd_logits) = model_outputs(&cfg, &init);

        let mut eng = qwen3::serve::Engine::from_map(cfg.clone(), &init, 4, 32, 1, 12, 8, false, false);
        let served = eng.prefill_for_perf(&mut BlockTable::new(), &PROMPT);

        let (e_dec, e_fwd) = (worst(&served, &hidden), worst(&dec_logits, &fwd_logits));
        assert!(e_dec < 1e-4, "{name}: decode hidden differs from the serving engine by {e_dec:e}");
        assert!(e_fwd < 1e-4, "{name}: batched forward differs from decode by {e_fwd:e}");
    }
}

#[test]
fn scaling_changes_the_output() {
    if gpu_disabled() {
        return;
    }
    let plain = cfg(None);
    let init = qwen3::init_weights(&plain, 3);
    let (base, _, _) = model_outputs(&plain, &init);
    for (scaling, name) in scalings() {
        let (scaled, _, _) = model_outputs(&cfg(Some(scaling)), &init);
        assert!(worst(&base, &scaled) > 1e-3, "{name}: a declared scaling must change the decode");
    }
}
