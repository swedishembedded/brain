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

use model::conformance::{Budget, ChunkedHead, Declares, Subject, Tier};

/// Tokens of the C8 row: three full 64-row head chunks and a partial one.
const CHUNK_SEQ: u32 = 200;

/// Distinct, non-square dims: d_model 64, q_dim 32, kv_dim 16, d_ff 96, vocab 23
/// (every contraction width a whole number of int8 groups).
fn qwen3_subject() -> Subject<qwen3::Qwen> {
    fn config(adapter_rank: Option<u32>) -> qwen3::QwenConfig {
        let lora = adapter_rank.map(|rank| qwen3::LoraCfg::new(rank, rank as f32, ["wq", "wk", "wv", "wo", "gate", "up", "down"].iter().map(|s| s.to_string()).collect()));
        qwen3::QwenConfig { lora, ..qwen3::QwenConfig::tiny_i8() }
    }
    fn chunked(adapter_rank: Option<u32>, head_budget_words: u64, seed: u64) -> qwen3::Qwen {
        let cfg = qwen3::QwenConfig { block_size: CHUNK_SEQ, max_position_embeddings: CHUNK_SEQ, ..config(adapter_rank) };
        let init = qwen3::init_weights(&cfg, seed);
        model::block::with_tile_budget_words(head_budget_words, || qwen3::Qwen::new(cfg, 1, CHUNK_SEQ, &init))
    }
    Subject {
        name: "qwen3",
        config,
        declares: Declares { token_objectives: true, adapters: true, sparse_targets: true },
        load: |path, rows, seq| qwen3::Qwen::load(path, rows, seq),
        tiers: vec![
            Tier { name: "bf16", build: |cfg, rows, seq, init| qwen3::Qwen::new_lora_dt(cfg, rows, seq, init, qwen3::Dtype::BF16), grad_rel_tol: 5e-2 },
            Tier { name: "int8", build: |cfg, rows, seq, init| qwen3::Qwen::new_lora_dt(cfg, rows, seq, init, qwen3::Dtype::I8), grad_rel_tol: 2e-1 },
            Tier { name: "nf4", build: |cfg, rows, seq, init| qwen3::Qwen::new_lora_dt(cfg, rows, seq, init, qwen3::Dtype::NF4), grad_rel_tol: 4e-1 },
            Tier { name: "fp8", build: |cfg, rows, seq, init| qwen3::Qwen::new_lora_dt(cfg, rows, seq, init, qwen3::Dtype::F8E4M3), grad_rel_tol: 2e-1 },
        ],
        budget: Budget::default(),
        chunked_head: Some(ChunkedHead { seq: CHUNK_SEQ, build: chunked }),
    }
}

/// Distinct dims: d_model 16, 3*d 48, d_ff 40, vocab 23. GPT-2 has no adapters.
fn gpt2_subject() -> Subject<gpt2::Gpt> {
    fn config(adapter_rank: Option<u32>) -> gpt2::GptConfig {
        assert!(adapter_rank.is_none(), "gpt2 declares no adapters");
        gpt2::GptConfig { vocab: 23, block_size: 12, n_layers: 2, d_model: 16, n_heads: 2, d_ff: 40 }
    }
    Subject {
        name: "gpt2",
        config,
        declares: Declares { token_objectives: true, adapters: false, sparse_targets: false },
        load: |path, rows, seq| gpt2::Gpt::load(path, rows, seq),
        tiers: Vec::new(),
        budget: Budget::default(),
        chunked_head: None,
    }
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
    fn chunked(adapter_rank: Option<u32>, head_budget_words: u64, seed: u64) -> qwen35::model::Qwen35 {
        let cfg = qwen35::config::Qwen35Config { block_size: CHUNK_SEQ, max_position_embeddings: CHUNK_SEQ, ..config(adapter_rank) };
        let init = <qwen35::model::Qwen35 as model::Model>::init_weights(&cfg, seed);
        let opts = qwen35::model::TrainOpts { head_budget_words: Some(head_budget_words), ..Default::default() };
        qwen35::model::Qwen35::new_train_opts_on(gpu_core::Gpu::new(qwen35::model::pipelines()), cfg, 1, CHUNK_SEQ, &init, &opts)
    }
    Subject {
        name: "qwen35",
        config,
        declares: Declares { token_objectives: true, adapters: true, sparse_targets: true },
        load: |path, rows, seq| qwen35::model::Qwen35::load(path, rows, seq),
        tiers: vec![
            Tier { name: "bf16", build: |cfg, rows, seq, init| frozen(Dtype::BF16, cfg, rows, seq, init), grad_rel_tol: 5e-2 },
            Tier { name: "nf4", build: |cfg, rows, seq, init| frozen(Dtype::NF4, cfg, rows, seq, init), grad_rel_tol: 5e-1 },
            Tier { name: "fp8", build: |cfg, rows, seq, init| frozen(Dtype::F8E4M3, cfg, rows, seq, init), grad_rel_tol: 2e-1 },
        ],
        // A finite-difference probe is two forward passes of a four-layer hybrid
        // decoder: ten tensors spread over the parameter list, two directions each.
        budget: Budget { tensors: Some(10), directions: 2 },
        chunked_head: Some(ChunkedHead { seq: CHUNK_SEQ, build: chunked }),
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
    fn chunked(adapter_rank: Option<u32>, head_budget_words: u64, seed: u64) -> qwen35moe::model::Qwen35 {
        let cfg = qwen35moe::config::Qwen35Config { block_size: CHUNK_SEQ, max_position_embeddings: CHUNK_SEQ, ..config(adapter_rank) };
        let init = <qwen35moe::model::Qwen35 as model::Model>::init_weights(&cfg, seed);
        model::block::with_tile_budget_words(head_budget_words, || qwen35moe::model::Qwen35::new_train(cfg, 1, CHUNK_SEQ, &init))
    }
    Subject {
        name: "qwen35moe",
        config,
        declares: Declares { token_objectives: true, adapters: true, sparse_targets: true },
        load: |path, rows, seq| qwen35moe::model::Qwen35::load(path, rows, seq),
        tiers: Vec::new(),
        budget: Budget { tensors: Some(10), directions: 2 },
        chunked_head: Some(ChunkedHead { seq: CHUNK_SEQ, build: chunked }),
    }
}

/// A Ripple: three layers (sliding, sliding, global) with a window that excludes keys, a value residual, a gated
/// attention and a clipped feed-forward, at widths that are pairwise distinct and non-square (hidden 64, q 128, kv 32,
/// feed-forward 192, vocabulary 32). Its frozen bases are held packed: the adapter rides over them unchanged.
fn ricursive_subject() -> Subject<ricursive::Ripple> {
    use gpu_core::select::Dtype;
    use model::ops::TierPolicy;
    use ricursive::{Ripple, TrainConfig, TrainOpts};

    fn config(adapter_rank: Option<u32>) -> TrainConfig {
        let cfg = TrainConfig::tiny();
        match adapter_rank {
            Some(rank) => cfg.with_lora(model::adapter::device::LoraCfg::new(rank, rank as f32, ricursive::train::default_targets())),
            None => cfg,
        }
    }
    fn frozen(base: Dtype, cfg: TrainConfig, rows: u32, seq: u32, init: &std::collections::HashMap<String, Vec<f32>>) -> Ripple {
        let opts = TrainOpts { tiers: TierPolicy::uniform(base), ..TrainOpts::default() };
        Ripple::build(cfg, rows, seq, init, &opts, true).unwrap_or_else(|e| panic!("{e}"))
    }
    // The tolerances are the quantization's own: the queries' and keys' factors see the base through the attention
    // softmax, which a random-weight tiny model amplifies (measured 0.27 at FP8 and 0.58 at NF4 on the worst tensor).
    // That the packed tiers are executed exactly is pinned elsewhere: `ricursive`'s `train_tiers` holds each to an
    // f32 build over the values its codec decodes to, with no difference at all.
    // The int8 tier is executed natively or not at all: a backend that would hold it as f32 refuses it by name, and the
    // check says so rather than passing over it.
    let device = gpu_core::Gpu::new(ricursive::train::pipelines());
    let int8_runs = model::ops::Ops::new(device.share()).is_ok_and(|ops| Dtype::I8.promote(&ops.caps().numeric) == Dtype::I8);
    let mut subject = Subject {
        name: "ricursive",
        config,
        declares: Declares { token_objectives: true, adapters: true, sparse_targets: true },
        load: |path, rows, seq| Ripple::load(path, rows, seq),
        tiers: Vec::new(),
        budget: Budget { tensors: Some(12), directions: 2 },
        chunked_head: None,
    };
    subject.tiers = vec![
        Tier { name: "bf16", build: |cfg, rows, seq, init| frozen(Dtype::BF16, cfg, rows, seq, init), grad_rel_tol: 5e-2 },
        Tier { name: "nf4", build: |cfg, rows, seq, init| frozen(Dtype::NF4, cfg, rows, seq, init), grad_rel_tol: 7e-1 },
        Tier { name: "fp8", build: |cfg, rows, seq, init| frozen(Dtype::F8E4M3, cfg, rows, seq, init), grad_rel_tol: 4e-1 },
    ];
    if int8_runs {
        subject.tiers.push(Tier { name: "int8", build: |cfg, rows, seq, init| frozen(Dtype::I8, cfg, rows, seq, init), grad_rel_tol: 2e-1 });
    } else {
        brain_testutil::skip_unavailable("ricursive int8 frozen base: this backend has no packed-int8 product (the other tiers run)");
    }
    subject
}

/// The tiny looped Inkling 2 (`inkling2::train::tiny_config`): a core of two depth-attention blocks run three times
/// with messages between passes, sliding and full layers, two experts both routed. Its adapter covers every linear and merged
/// site (the float expert banks of a small configuration take no expert pair); its frozen base is held at f32 here.
fn inkling2_subject() -> Subject<inkling2::Inkling2Train> {
    use inkling2::{Inkling2Train, TrainConfig};

    fn config(adapter_rank: Option<u32>) -> TrainConfig {
        // Every token routes to both experts: the suite's finite differences and single steps cannot pin a top-k
        // selection, and a flipped one is another function (the gradient checks with a pinned routing on the
        // four-expert config are `gradcheck`'s `inkling2_fd`).
        let cfg = inkling2::train::tiny_config_every_expert_routed();
        match adapter_rank {
            Some(rank) => cfg.with_lora(model::adapter::device::LoraCfg::new(rank, rank as f32, inkling2::train::targets(false))).unwrap_or_else(|e| panic!("{e}")),
            None => cfg,
        }
    }
    Subject {
        name: "inkling2",
        config,
        declares: Declares { token_objectives: true, adapters: true, sparse_targets: true },
        load: |path, rows, seq| Inkling2Train::load(path, rows, seq).unwrap_or_else(|e| panic!("{e}")),
        // A bf16 base rounds the router's logits, so a near-tie selects other experts than the f32 base and the
        // gradients are of another function: the tier is held to the f32 base with the routing pinned, in
        // `inkling2/tests/it/training.rs` (`a_bf16_base_trains_the_adapters_the_f32_base_does`).
        tiers: Vec::new(),
        budget: Budget { tensors: Some(12), directions: 2 },
        chunked_head: None,
    }
}

/// Every registered model, by name. The suite refuses to pass over none.
const REGISTERED: &[&str] = &["qwen3", "gpt2", "qwen35", "qwen35moe", "ricursive", "inkling2"];

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
subject!(qwen35_conforms, qwen35_subject());
subject!(qwen35moe_conforms, qwen35moe_subject());
subject!(ricursive_conforms, ricursive_subject());
subject!(inkling2_conforms, inkling2_subject());

/// What the asynchronous policy-gradient suite (A49, [`rl::conformance_async`]) must answer for each architecture whose
/// trainer is registered: `(name, online policy gradient, score centring, policy publication)`. The capability gate
/// (`scripts/gates/training-capabilities.py`) holds the declaration's `online_pg`, `score_centring` and
/// `policy_publication` cells to this table.
const ASYNC_MATRIX: &[(&str, bool, bool, bool)] = &[
    ("qwen3", true, true, true),
    ("qwen35", true, true, true),
    ("qwen35moe", false, false, false),
    ("ricursive", true, true, true),
    ("inkling2", true, true, true),
];

/// The serving bank's verdict on a published adapter of `family`: read, bound to the base its card names and verified
/// as `--serve-adapters` does, then attached to a bank of the default settings.
fn bank_takes(family: llmserve::adapter_release::ServedFamily, base_id: &'static str, base_file: std::path::PathBuf) -> impl Fn(&std::path::Path) -> Result<(), String> {
    use llmserve::adapter_registry::{AdapterBankSettings, AdapterRegistry};
    use llmserve::adapter_release::{verify_adapter_among, ServedBase};
    move |path| {
        let base = ServedBase::of_family(family.clone(), base_file.clone(), Some(base_id.to_string()));
        let (_, loaded) = verify_adapter_among(path, std::slice::from_ref(&base)).map_err(|e| e.to_string())?;
        AdapterRegistry::new(Some(AdapterBankSettings::default()), base).attach(loaded).map(drop)
    }
}

/// Runs A49 on `subject` with its trainer's declaration, publishing to `serving`'s bank when it serves adapters, and
/// holds the answer to [`ASYNC_MATRIX`].
fn async_conforms<M: model::Model>(subject: model::conformance::Subject<M>, policy_gradient: bool, serving: Option<llmserve::adapter_release::ServedFamily>) {
    use rl::conformance_async::{run_async, AsyncDeclares, Publication, Verdict};
    let name = subject.name;
    let &(_, online, centring, publication) = ASYNC_MATRIX.iter().find(|row| row.0 == name).unwrap_or_else(|| panic!("{name} is not in ASYNC_MATRIX"));
    assert_eq!(policy_gradient, online, "{name}: its ChatTrainer::POLICY_GRADIENT is not what ASYNC_MATRIX says");
    let dir = brain_testutil::Scratch::dir(&format!("async-{name}"));
    let base_file = dir.join("base.safetensors");
    let (base, _) = model::conformance::build(&subject, None, 1, 41);
    base.save(base_file.to_str().unwrap());
    let base_id: &'static str = Box::leak(format!("conformance/{name}").into_boxed_str());
    let card_id = format!("{base_id}:conformance:policy:v1");
    let check = serving.clone().map(|family| bank_takes(family, base_id, base_file.clone()));
    let tag = serving.as_ref().map(|f| f.tag);
    let publish = check.as_ref().map(|check| Publication { card_id: &card_id, base_id, family: tag.unwrap(), check });
    let declares = AsyncDeclares { online_policy_gradient: policy_gradient, policy_publication: serving.is_some() };
    // seed: fixed so a conformance failure reproduces
    let report = run_async(&subject, declares, publish.as_ref(), &dir, 41).unwrap_or_else(|e| panic!("{e}"));
    let holds = |v: &Verdict| *v == Verdict::Holds;
    // Support replay is the head's extra targets, as score centring is: it holds exactly where centring does.
    assert_eq!((holds(&report.online_policy_gradient), holds(&report.score_centring), holds(&report.support_replay), holds(&report.policy_publication)), (online, centring, centring, publication), "{name}: {report:?}");
    for (verdict, what) in [(&report.online_policy_gradient, "online policy gradient"), (&report.score_centring, "score centring"), (&report.support_replay, "support replay"), (&report.policy_publication, "policy publication")] {
        if let Verdict::Refused(why) = verdict {
            assert!(!why.is_empty(), "{name}: {what} is refused without a reason");
        }
    }
}

#[test]
fn qwen3_trains_online() {
    async_conforms(qwen3_subject(), <qwen3::finetune::Qwen3Trainer as model::finetune::ChatTrainer>::POLICY_GRADIENT, Some(llmserve::adapter_release::ServedFamily::qwen3()));
}

#[test]
fn qwen35_trains_online() {
    async_conforms(qwen35_subject(), <qwen35::finetune::Qwen35Trainer as model::finetune::ChatTrainer>::POLICY_GRADIENT, Some(qwen35::spec::served_family()));
}

#[test]
fn qwen35moe_trains_online() {
    async_conforms(qwen35moe_subject(), <qwen35moe::finetune::Qwen35MoeTrainer as model::finetune::ChatTrainer>::POLICY_GRADIENT, None);
}

#[test]
fn ricursive_trains_online() {
    async_conforms(ricursive_subject(), <ricursive::finetune::RippleTrainer as model::finetune::ChatTrainer>::POLICY_GRADIENT, Some(ricursive::resident::family()));
}

#[test]
fn inkling2_trains_online() {
    let cfg = (inkling2_subject().config)(Some(model::conformance::RANK));
    async_conforms(inkling2_subject(), <inkling2::finetune::Inkling2Trainer as model::finetune::ChatTrainer>::POLICY_GRADIENT, Some(inkling2::serve::served_family(&cfg.cfg)));
}

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
    let subject: Subject<Fake> = Subject { name: "fake", config, declares: Declares { token_objectives: false, adapters: false, sparse_targets: false }, load: |_, _, _| Fake, tiers: Vec::new(), budget: Budget::default(), chunked_head: None };
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

/// C10 fails a model whose adapter sites are not what it declares: a config that ignores the rank gives a model
/// that trains its base tensors where an adapter model must train adapter tensors only.
#[test]
fn an_adapter_model_that_trains_base_tensors_fails_the_trainable_audit() {
    let mut subject = qwen3_subject();
    subject.config = |_| qwen3::QwenConfig::tiny_i8();
    // seed: fixed so a failure reproduces
    let err = model::conformance_trainable::trainable_set_matches_declaration(&subject, 41).expect_err("the adapter model trains base tensors");
    assert!(err.contains("qwen3") && err.contains("not an adapter tensor"), "{err}");
}
