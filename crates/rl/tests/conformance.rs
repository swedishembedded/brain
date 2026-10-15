// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Runs the training-capability conformance suite (`model::conformance`) over
//! every registered model. Adding a model is one `subject!` line plus its
//! `Subject` constructor; the suite never names an architecture.
//!
//! Swedish Embedded AB implements solutions for verifying that every model in
//! a training stack honours the same fine-tuning contract, for its clients. If
//! your team needs expertise in conformance testing for ML training code you
//! can procure our services by sending an email to info@swedishembedded.com.

use model::conformance::{Declares, Subject};

/// Distinct, non-square dims: d_model 16, q_dim 24, kv_dim 12, d_ff 40, vocab 23.
fn qwen3_subject() -> Subject<qwen3::Qwen> {
    fn config(adapter_rank: Option<u32>) -> qwen3::QwenConfig {
        let lora = adapter_rank.map(|rank| qwen3::LoraCfg { rank, alpha: rank as f32, targets: ["wq", "wk", "wv", "wo", "gate", "up", "down"].iter().map(|s| s.to_string()).collect() });
        qwen3::QwenConfig { head_dim: 6, d_ff: 40, lora, ..qwen3::QwenConfig::tiny() }
    }
    Subject { name: "qwen3", config, declares: Declares { token_objectives: true, adapters: true } }
}

/// Distinct dims: d_model 16, 3*d 48, d_ff 40, vocab 23. GPT-2 has no adapters.
fn gpt2_subject() -> Subject<gpt2::Gpt> {
    fn config(adapter_rank: Option<u32>) -> gpt2::GptConfig {
        assert!(adapter_rank.is_none(), "gpt2 declares no adapters");
        gpt2::GptConfig { vocab: 23, block_size: 12, n_layers: 2, d_model: 16, n_heads: 2, d_ff: 40 }
    }
    Subject { name: "gpt2", config, declares: Declares { token_objectives: true, adapters: false } }
}

/// Every registered model, by name. The suite refuses to pass over none.
const REGISTERED: &[&str] = &["qwen3", "gpt2"];

macro_rules! subject {
    ($test:ident, $subject:expr) => {
        #[test]
        fn $test() {
            let subject = $subject;
            assert!(REGISTERED.contains(&subject.name), "{} is not in REGISTERED", subject.name);
            // seed: fixed so a conformance failure reproduces
            let seed = 41;
            model::conformance::run(&subject, seed).unwrap_or_else(|e| panic!("{e}"));
        }
    };
}

subject!(qwen3_conforms, qwen3_subject());
subject!(gpt2_conforms, gpt2_subject());

#[test]
fn at_least_one_model_is_registered() {
    assert!(!REGISTERED.is_empty());
}
