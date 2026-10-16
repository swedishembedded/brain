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

use model::conformance::{Declares, Subject, Tier};

/// Distinct, non-square dims: d_model 64, q_dim 32, kv_dim 16, d_ff 96, vocab 23
/// (every contraction width a whole number of int8 groups).
fn qwen3_subject() -> Subject<qwen3::Qwen> {
    fn config(adapter_rank: Option<u32>) -> qwen3::QwenConfig {
        let lora = adapter_rank.map(|rank| qwen3::LoraCfg { rank, alpha: rank as f32, targets: ["wq", "wk", "wv", "wo", "gate", "up", "down"].iter().map(|s| s.to_string()).collect() });
        qwen3::QwenConfig { lora, ..qwen3::QwenConfig::tiny_i8() }
    }
    Subject {
        name: "qwen3",
        config,
        declares: Declares { token_objectives: true, adapters: true },
        load: |path, rows, seq| qwen3::Qwen::load(path, rows, seq),
        tiers: vec![
            Tier { name: "bf16", build: |cfg, rows, seq, init| qwen3::Qwen::new_lora_dt(cfg, rows, seq, init, qwen3::Dtype::BF16), grad_rel_tol: 5e-2 },
            Tier { name: "int8", build: |cfg, rows, seq, init| qwen3::Qwen::new_lora_dt(cfg, rows, seq, init, qwen3::Dtype::I8), grad_rel_tol: 2e-1 },
        ],
    }
}

/// Distinct dims: d_model 16, 3*d 48, d_ff 40, vocab 23. GPT-2 has no adapters.
fn gpt2_subject() -> Subject<gpt2::Gpt> {
    fn config(adapter_rank: Option<u32>) -> gpt2::GptConfig {
        assert!(adapter_rank.is_none(), "gpt2 declares no adapters");
        gpt2::GptConfig { vocab: 23, block_size: 12, n_layers: 2, d_model: 16, n_heads: 2, d_ff: 40 }
    }
    Subject { name: "gpt2", config, declares: Declares { token_objectives: true, adapters: false }, load: |path, rows, seq| gpt2::Gpt::load(path, rows, seq), tiers: Vec::new() }
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
            rl::conformance::run_objectives(&subject, seed).unwrap_or_else(|e| panic!("{e}"));
        }
    };
}

subject!(qwen3_conforms, qwen3_subject());
subject!(gpt2_conforms, gpt2_subject());

#[test]
fn at_least_one_model_is_registered() {
    assert!(!REGISTERED.is_empty());
}

/// A model that does not provide token objectives yet. It is not registered
/// for the full suite (its trainer is still arriving); this keeps the
/// declaration honest and shows every rl objective refusing it with the typed
/// error. When it gains the capability this test fails on the declaration,
/// which is the cue to register it in `REGISTERED` and delete this.
#[test]
fn a_model_without_token_objectives_is_refused_by_every_objective() {
    fn config(_adapter_rank: Option<u32>) -> qwen35::config::Qwen35Config {
        qwen35::config::Qwen35Config::tiny()
    }
    let subject: Subject<qwen35::model::Qwen35> = Subject {
        name: "qwen35",
        config,
        declares: Declares { token_objectives: false, adapters: false },
        load: |_, _, _| panic!("not registered for the full suite"),
        tiers: Vec::new(),
    };
    // seed: fixed so a failure reproduces
    let seed = 41;
    model::conformance::check_declaration(&subject, seed).unwrap_or_else(|e| panic!("{e}"));
    rl::conformance::run_objectives(&subject, seed).unwrap_or_else(|e| panic!("{e}"));
}

/// The suite fails a model whose declaration is false: here qwen3 claiming
/// not to offer what it offers.
#[test]
fn a_false_declaration_fails_the_suite() {
    let mut subject = qwen3_subject();
    subject.declares.token_objectives = false;
    // seed: fixed so a failure reproduces
    let err = model::conformance::run(&subject, 41).expect_err("a model that offers the capability but declares it absent must fail");
    assert!(err.contains("declares token_objectives=false"), "{err}");
}

/// C8: a chunked LM-head cross-entropy equals the unchunked one (loss,
/// log-probabilities and gradients). The shared chunked head lands with
/// lane A1's `head_ce`; until then qwen3's own chunked head is covered by
/// `qwen3/tests/chunked_head.rs` and this stays a visible, failing-when-run gap.
#[test]
#[ignore = "waits for the shared chunked head cross-entropy (head_ce)"]
fn chunked_head_ce_equals_unchunked() {
    panic!("C8 is not implemented: register the chunked-head hook on Subject once head_ce lands");
}
