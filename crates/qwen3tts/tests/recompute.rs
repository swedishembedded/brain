// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of text-to-speech models for its clients. If your team
// needs expertise in training speech language models and their policy updates within a fixed device budget, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Saved-activation policy of the Qwen3-TTS Talker. Its decoder is a `qwen3::Qwen` whose backward always rebuilds
//! every layer from its input boundary (a one-layer scratch serves all of them), so `TalkerModel` and the
//! policy-gradient `TalkerTrainer` accept recomputing as what they already do and refuse keeping every layer and
//! narrow saved formats by name, never ignoring them.

use model::saved::{SavedFormat, SavedPolicy};
use qwen3tts::train::TalkerTrainer;
use qwen3tts::{TalkerConfig, TalkerModel};

fn config() -> TalkerConfig {
    TalkerConfig::tiny()
}

/// A Talker checkpoint on disk: the tiny decoder plus the untied codec embedding and head the trainer loads.
fn write_talker(path: &str, seed: u64) {
    let cfg = config();
    let qcfg = cfg.to_qwen(8);
    let mut init = qwen3::init_weights(&qcfg, seed);
    let mut rng = data::rng::Lcg::new(seed ^ 1);
    let (d, vocab) = (cfg.d_model as usize, cfg.vocab as usize);
    init.insert("tok.weight".to_string(), rng.vec_scaled(vocab * d, 0.05));
    init.insert("lm_head.weight".to_string(), rng.vec_scaled(vocab * d, 0.05));
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = init.into_iter().map(|(k, v)| (k, vec![v.len() as u64], v)).collect();
    checkpoint::save(path, qcfg.to_json(), &tensors);
}

/// The answers the contract asks of a model whose backward always recomputes.
fn check_always_recomputes(name: &str, mut set: impl FnMut(SavedPolicy) -> Result<(), String>) {
    set(SavedPolicy::Recompute).unwrap_or_else(|e| panic!("{name}: recompute refused: {e}"));
    for refused in [SavedPolicy::Keep, SavedPolicy::Quantized(SavedFormat::Bf16)] {
        let err = set(refused).expect_err("a policy the backward cannot honour must be refused");
        assert!(err.contains(name), "{name}: the refusal of {refused:?} does not name the model: {err}");
    }
}

#[test]
fn the_talker_accepts_recompute_and_refuses_what_it_cannot_do() {
    let seed = data::rng::random_seed();
    let model = TalkerModel::new_trainable(config(), 1, 6, seed);
    check_always_recomputes("qwen3tts talker", |p| model.set_saved_policy(p));

    let dir = std::env::temp_dir().join(format!("qwen3tts-recompute-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("talker.safetensors").to_str().unwrap().to_string();
    write_talker(&path, seed);
    let trainer = TalkerTrainer::load(&path, 8, 2, 4.0, seed).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    check_always_recomputes("qwen3tts talker", |p| trainer.set_saved_policy(p));
}
