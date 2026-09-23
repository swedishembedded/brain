// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `rl::reader::ModelLearner` against a real checkpoint.
//!
//! `brain-audit`'s own suite proves the ORCHESTRATION with a fake that
//! counts calls. This proves the four methods that fake stands in for
//! actually do what the loop assumes, on the same tiny CPU-runnable Qwen3
//! fixture the document study uses: a randomly initialised two-layer decoder
//! over a byte tokenizer, with a LoRA overlay.
//!
//! Nothing here asserts that the model LEARNED anything. The fixture is
//! random weights trained for a handful of steps, so any accuracy number is
//! noise and saying otherwise would be inventing a result. What is under
//! test is the contract: a loss that is a real per-token figure rather than
//! a placeholder, a training run that produces adapter bytes a pool can
//! store, and a scoring pass that returns one score per probe plus the
//! entropy the degeneracy bar needs, from the checkpoint that would actually
//! be served.
//!
//! Swedish Embedded AB builds the bindings that turn a tested,
//! device-independent training policy into one that runs against real
//! weights without the policy having to know it. If your team needs
//! expertise in keeping orchestration testable while the model stays real,
//! you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::{Path, PathBuf};

use audit::bank::{Probe, ProbeFamily, ProbeId};
use audit::reader::{Arm, Learner};
use checkpoint::gguf::GgufTokenizer;
use data::qwen_tokenizer::QwenBpe;
use model::rollout::RolloutParams;
use model::serve::SampleParams;
use model::FitOpts;
use qwen3::config::{LoraCfg, QwenConfig};
use rl::reader::ModelLearner;

const VOCAB: u32 = 256;
const BLOCK: u32 = 64;
const RANK: u32 = 4;
const ALPHA: f32 = 8.0;

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn byte_tokenizer() -> QwenBpe {
    let tokens: Vec<String> = data::bpe::bytes_to_unicode().iter().map(|c| c.to_string()).collect();
    let gt = GgufTokenizer { model: "gpt2".into(), tokens, ..Default::default() };
    QwenBpe::from_gguf(&gt).expect("byte tokenizer builds")
}

fn cfg() -> QwenConfig {
    QwenConfig { vocab: VOCAB, block_size: BLOCK, max_position_embeddings: BLOCK, lora: Some(LoraCfg::attn(RANK, ALPHA)), ..QwenConfig::tiny() }
}

fn base_checkpoint(path: &Path, c: &QwenConfig, seed: u64) {
    let init = qwen3::init_weights(c, seed);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = c
        .param_list()
        .into_iter()
        .map(|(name, n)| (name.clone(), vec![n as u64], init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone()))
        .collect();
    checkpoint::save(path.to_str().expect("utf-8 path"), c.to_json(), &tensors);
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("brain-reader-learner-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("temp dir");
    d
}

fn fit_opts() -> FitOpts {
    FitOpts { steps: 3, batch_size: 1, block_size: BLOCK, lr: 1e-3, eval_interval: 0, eval_batches: 0, checkpoint_secs: 0, ..FitOpts::default() }
}

/// `ProbeId` is constructed by the bank from an episode's own content; a
/// test that only needs two distinct ids builds them by round-tripping the
/// public shape rather than reaching for a constructor that exists for the
/// bank's benefit.
fn probe_id(i: usize) -> ProbeId {
    serde_json::from_value(serde_json::Value::String(format!("test-probe-{i}"))).expect("a ProbeId is a string")
}

fn probe(i: usize, prompt: &str, expected: &str) -> Probe {
    Probe {
        id: probe_id(i),
        family: ProbeFamily::Literal,
        prompt: prompt.to_string(),
        expected: expected.to_string(),
        line: i,
        blind: false,
        baseline: None,
    }
}

fn rows() -> Vec<String> {
    (0..8).map(|i| format!("--flag{i:03} VALUE sets option {i:03}\n")).collect()
}

/// The three contracts the loop relies on, in one run so the fixture is
/// built once: a real loss, adapter bytes a pool can store, and a scoring
/// pass of the right shape from the served checkpoint.
#[test]
fn the_learner_reports_a_real_loss_trains_an_adapter_and_scores_both_arms() {
    if gpu_disabled() {
        return;
    }
    let dir = tmp("contract");
    let base = dir.join("base.safetensors");
    base_checkpoint(&base, &cfg(), 7);
    let tok = byte_tokenizer();
    let rollout = RolloutParams { max_new: 12, sample: SampleParams::greedy(), eos: None };
    let mut learner: ModelLearner<'_, qwen3::model::Qwen, QwenBpe> =
        ModelLearner::new(&base, &dir.join("work"), &tok, cfg(), fit_opts(), rollout, RANK, ALPHA).expect("learner");

    // A real per-token cross entropy: finite, positive, and no larger than
    // uniform over the vocabulary, which is what an untrained model is.
    let loss = learner.loss("--flag000 VALUE sets option 000\n--flag001 VALUE sets option 001\n");
    assert!(loss.is_finite() && loss > 0.0, "loss must be a real figure, got {loss}");
    assert!(loss < (VOCAB as f64).ln() * 1.5, "an untrained model should be near uniform, not beyond it: {loss}");

    // A one-token episode has nothing to predict FROM, and must read as out
    // of reach rather than as a suspiciously perfect score.
    assert!(learner.loss("x").is_infinite(), "a text with nothing to predict from must not read as well understood");

    let owned = rows();
    let row_refs: Vec<&str> = owned.iter().map(String::as_str).collect();
    let adapter = learner.train(&row_refs, 11);
    assert!(!adapter.is_empty(), "training must produce adapter bytes for the pool to store");
    assert!(adapter.len() < 4 * 1024 * 1024, "the pool stores an ADAPTER, not the base weights again: {} bytes", adapter.len());

    // The adapter's card must record the rank it was actually trained at.
    // `fold_adapter_into` READS the rank from there to do the fold, so a
    // card that understated it would fold silently wrongly rather than fail.
    let written = checkpoint::st::load_safetensors(dir.join("work").join("adapter.safetensors").to_str().expect("utf-8")).expect("adapter reads back");
    let card = written.card().expect("an adapter carries a card");
    let descriptor = card.adapter.as_ref().expect("and an adapter descriptor");
    assert_eq!(descriptor.rank, Some(RANK), "the card must record the rank the adapter was trained at");
    assert_eq!(descriptor.alpha, Some(ALPHA));

    let probes = [probe(0, "--flag000 VALUE", " sets option 000"), probe(1, "--flag001 VALUE", " sets option 001")];
    let refs: Vec<&Probe> = probes.iter().collect();
    for arm in [Arm::Incumbent, Arm::Candidate] {
        let scored = learner.score(arm, &refs);
        assert_eq!(scored.scores.len(), refs.len(), "one score per probe, in order");
        assert!(scored.scores.iter().all(|s| (0.0..=1.0).contains(s)), "exact match is 0 or 1: {:?}", scored.scores);
        assert!(scored.mean_entropy.is_finite(), "the degeneracy bar needs a real entropy, got {}", scored.mean_entropy);
    }
}

/// A promotion is the candidate becoming what is served, and the incumbent
/// must not move before one.
#[test]
fn promoting_replaces_what_is_served_and_nothing_else_does() {
    if gpu_disabled() {
        return;
    }
    let dir = tmp("promote");
    let base = dir.join("base.safetensors");
    base_checkpoint(&base, &cfg(), 3);
    let tok = byte_tokenizer();
    let rollout = RolloutParams { max_new: 8, sample: SampleParams::greedy(), eos: None };
    let work = dir.join("work");
    let mut learner: ModelLearner<'_, qwen3::model::Qwen, QwenBpe> =
        ModelLearner::new(&base, &work, &tok, cfg(), fit_opts(), rollout, RANK, ALPHA).expect("learner");

    let before = std::fs::read(work.join("incumbent.safetensors")).expect("incumbent exists from the base");
    let owned = rows();
    let row_refs: Vec<&str> = owned.iter().map(String::as_str).collect();
    learner.train(&row_refs, 5);
    let after_training = std::fs::read(work.join("incumbent.safetensors")).expect("incumbent");
    assert_eq!(before, after_training, "training a candidate must not change what is served");

    learner.promote().expect("promote");
    let served = std::fs::read(work.join("incumbent.safetensors")).expect("incumbent");
    let candidate = std::fs::read(work.join("candidate.safetensors")).expect("candidate");
    assert_eq!(served, candidate, "a promotion makes the candidate what is served");
}
