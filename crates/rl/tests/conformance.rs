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

/// A Qwen3.5 hybrid decoder: three Gated-DeltaNet layers and one
/// full-attention layer, every contraction width a whole number of
/// quantisation groups, and dimensions pairwise distinct and non-square
/// (d_model 96, q_dim 128, kv_dim 32, GDN value dim 160, key dim 48, d_ff 192,
/// vocab 29).
fn qwen35_subject() -> Subject<qwen35::model::Qwen35> {
    use gpu_core::select::Dtype;
    use model::ops::TierPolicy;

    fn config(adapter_rank: Option<u32>) -> qwen35::config::Qwen35Config {
        qwen35::config::Qwen35Config {
            n_heads: 4,
            head_dim: 32,
            mrope_section: [2, 1, 1],
            linear_key_head_dim: 24,
            linear_num_value_heads: 4,
            linear_value_head_dim: 40,
            intermediate_size: 192,
            lora: adapter_rank.map(|rank| qwen35::config::lora_cfg(rank, rank as f32)),
            ..qwen35::config::Qwen35Config::tiny()
        }
    }
    fn frozen(base: Dtype, cfg: qwen35::config::Qwen35Config, rows: u32, seq: u32, init: &std::collections::HashMap<String, Vec<f32>>) -> qwen35::model::Qwen35 {
        let opts = qwen35::model::TrainOpts { base: TierPolicy::uniform(base), ..Default::default() };
        qwen35::model::Qwen35::new_train_opts_on(gpu_core::Gpu::new(qwen35::model::pipelines()), cfg, rows, seq, init, &opts)
    }
    Subject {
        name: "qwen35",
        config,
        declares: Declares { token_objectives: true, adapters: true },
        load: |path, rows, seq| qwen35::model::Qwen35::load(path, rows, seq),
        tiers: vec![
            Tier { name: "bf16", build: |cfg, rows, seq, init| frozen(Dtype::BF16, cfg, rows, seq, init), grad_rel_tol: 5e-2 },
            Tier { name: "nf4", build: |cfg, rows, seq, init| frozen(Dtype::NF4, cfg, rows, seq, init), grad_rel_tol: 4e-1 },
        ],
    }
}

/// A Qwen3.5 mixture-of-experts decoder: its tiny config already has
/// pairwise-distinct widths; four layers (three Gated-DeltaNet, one
/// full-attention) keep the finite-difference checks short. Only the nine
/// mixer linears carry adapters; the base is fp32 (the experts have no narrow
/// training tier).
fn qwen35moe_subject() -> Subject<qwen35moe::model::Qwen35> {
    fn config(adapter_rank: Option<u32>) -> qwen35moe::config::Qwen35Config {
        qwen35moe::config::Qwen35Config {
            n_layers: 4,
            lora: adapter_rank.map(|rank| qwen35moe::config::lora_cfg(rank, rank as f32)),
            ..qwen35moe::config::Qwen35Config::tiny()
        }
    }
    Subject {
        name: "qwen35moe",
        config,
        declares: Declares { token_objectives: true, adapters: true },
        load: |path, rows, seq| qwen35moe::model::Qwen35::load(path, rows, seq),
        tiers: Vec::new(),
    }
}

/// Every registered model, by name. The suite refuses to pass over none.
const REGISTERED: &[&str] = &["qwen3", "gpt2", "qwen35", "qwen35moe"];
/// Qwen3.5 trains a batch of more than one row through its Gated-DeltaNet
/// layers correctly on the accelerator backends (the suite's finite-difference
/// check passes), but its backward corrupts the heap on the CPU backend. The
/// suite fails with that reason instead of aborting the process.
fn require_accelerator(model: &str) {
    assert!(
        gpu_core::backend_name() != "cpu",
        "{model}: batched Gated-DeltaNet training corrupts memory on the CPU backend; run the suite on a GPU backend"
    );
}

macro_rules! subject {
    ($test:ident, $subject:expr) => {
        #[test]
        fn $test() {
            let subject = $subject;
            if subject.name == "qwen35" || subject.name == "qwen35moe" {
                require_accelerator(subject.name);
            }
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
subject!(qwen35_conforms, qwen35_subject());
subject!(qwen35moe_conforms, qwen35moe_subject());

#[test]
fn at_least_one_model_is_registered() {
    assert!(!REGISTERED.is_empty());
}

/// A model that does not provide token objectives. It is not registered for
/// the full suite; this keeps the declaration honest and shows every rl
/// objective refusing it with the typed error. It computes nothing: only
/// construction and the configuration reach it.
struct Fake;

#[derive(Clone)]
struct FakeConfig;

impl model::ModelConfig for FakeConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        Vec::new()
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({})
    }
    fn from_json(_v: &serde_json::Value) -> Self {
        FakeConfig
    }
    fn vocab(&self) -> u32 {
        8
    }
    fn block_size(&self) -> u32 {
        8
    }
    fn finalize_for_dataset(self, _vocab: u32, _block_size: u32) -> Self {
        self
    }
}

impl model::Model for Fake {
    type Config = FakeConfig;
    fn new(_cfg: FakeConfig, _b: u32, _t: u32, _init: &std::collections::HashMap<String, Vec<f32>>) -> Self {
        Fake
    }
    fn init_weights(_cfg: &FakeConfig, _seed: u64) -> std::collections::HashMap<String, Vec<f32>> {
        Default::default()
    }
    fn config(&self) -> &FakeConfig {
        &FakeConfig
    }
    fn set_batch(&self, _batch: model::Batch) {
        unreachable!("a refused model is never trained")
    }
    fn forward(&self) -> f32 {
        unreachable!("a refused model is never trained")
    }
    fn backward(&self) {
        unreachable!("a refused model is never trained")
    }
    fn zero_grads(&self) {}
    fn adamw_step(&self, _t: u32, _lr: f32, _wd: f32, _adam: model::Adam, _clip: Option<f32>, _extra_scale: f32) {}
    fn poll_wait(&self) {}
    fn param_names(&self) -> Vec<String> {
        Vec::new()
    }
    fn read_weight(&self, _name: &str) -> Vec<f32> {
        Vec::new()
    }
    fn write_weight(&self, _name: &str, _data: &[f32]) {}
    fn read_grad(&self, _name: &str) -> Vec<f32> {
        Vec::new()
    }
    fn logits_all(&self, _tokens: &[u32]) -> Option<Vec<f32>> {
        None
    }
    fn save(&self, _path: &str) {}
    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}

#[test]
fn a_model_without_token_objectives_is_refused_by_every_objective() {
    fn config(_adapter_rank: Option<u32>) -> FakeConfig {
        FakeConfig
    }
    let subject: Subject<Fake> = Subject { name: "fake", config, declares: Declares { token_objectives: false, adapters: false }, load: |_, _, _| Fake, tiers: Vec::new() };
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
