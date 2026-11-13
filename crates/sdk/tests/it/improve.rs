// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "study")]

//! `brain::Improve` end to end: one GRPO cycle over a caller-supplied
//! `Environment`/`Verifier`, against a tiny CPU-runnable Qwen3 fixture - no
//! downloaded weights, no tokenizer, no chat template (`Improve` resolves a
//! bare checkpoint path directly, unlike `DocumentStudy`'s curriculum path,
//! which needs both).
//!
//! This exercises the mechanism only, per `ImproveOptions`'s own doc
//! comment: GRPO's measured performance on the one task family tried so far
//! is WORSE than the SFT-driven `DocumentStudy` path, so nothing here
//! asserts the candidate learned anything - only that the loop runs, gates,
//! and publishes an adapter iff (and only iff) it promoted.
//!
//! Swedish Embedded AB builds the operator-facing surfaces that turn a
//! gated training loop into something a team can actually run, read and
//! act on - one call, one typed verdict, one adapter a live server picks
//! up. If your team needs expertise shipping continuous learning as an
//! operable product rather than a notebook, you can procure our services
//! by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use brain::{Environment, Improve, ImproveOptions, Reward, Step, Task, Verifier};
use qwen3::config::QwenConfig;

fn skip() -> bool {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS is set: an improve cycle needs a real training/decode backend");
        return true;
    }
    false
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-improve-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A bare checkpoint - `Improve::run` resolves a base weights FILE directly,
/// no tokenizer or chat template needed (unlike `DocumentStudy`'s
/// curriculum, which reads both out of a model directory).
fn write_base(path: &Path) {
    let cfg = QwenConfig { block_size: 32, max_position_embeddings: 32, ..QwenConfig::tiny() };
    let init = qwen3::init_weights(&cfg, 11);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
        cfg.param_list().into_iter().map(|(name, n)| (name.clone(), vec![n as u64], init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone())).collect();
    checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
}

/// One fixed prompt, one fixed target - the simplest possible verifiable
/// task, scored by fractional token match. No model-as-judge anywhere.
struct FixedTargetEnv {
    prompt: Vec<u32>,
    target: Vec<u32>,
}

impl Environment for FixedTargetEnv {
    fn name(&self) -> &str {
        "fixed-target"
    }
    fn tasks(&self, seed: u64) -> Vec<Task> {
        vec![Task { id: format!("fixed-target-{seed}"), prompt: self.prompt.clone(), answer: serde_json::json!(self.target) }]
    }
}

#[derive(Clone)]
struct FracMatchVerifier;

impl Verifier for FracMatchVerifier {
    fn verify(&self, task: &Task, _transcript: &[Step], completion: &[u32]) -> Reward {
        let target: Vec<u32> = serde_json::from_value(task.answer.clone()).expect("answer is a token array");
        let matches = completion.iter().zip(target.iter()).filter(|(a, b)| a == b).count();
        Reward { value: matches as f32 / target.len().max(1) as f32, parts: Default::default() }
    }
}

/// One full round trip: a real GRPO cycle over a toy environment, a typed
/// verdict out, and - only if the gate promoted - an adapter published
/// under the name the serving-side watcher looks for.
#[test]
fn an_improve_cycle_runs_gates_and_publishes_an_adapter_only_on_promote() {
    if skip() {
        return;
    }
    let dir = tmp("round-trip");
    let base = dir.join("base.safetensors");
    write_base(&base);

    let adapters = dir.join("adapters");
    let env = FixedTargetEnv { prompt: vec![1, 2, 3], target: vec![4, 5, 6] };
    let outcome = Improve::from_pretrained(base.to_string_lossy().into_owned())
        .expect("from_pretrained")
        .adapter_dir(&adapters)
        .work_dir(dir.join("work"))
        .lora(4)
        .seed(brain::random_seed())
        .held_out_seed(brain::random_seed())
        // `max_new` must leave room for `prompt.len() + max_new <= block_size`
        // (32 here): the prompt is 3 tokens, so 8 is comfortably inside it.
        .run(env, FracMatchVerifier, &ImproveOptions { steps: 8, max_new: 8, ..ImproveOptions::default() })
        .expect("the improve cycle must run");

    assert!(outcome.p_value.is_finite() && (0.0..=1.0).contains(&outcome.p_value), "p_value must be a probability, got {}", outcome.p_value);
    assert!(outcome.effect_size.is_finite());
    assert_eq!(outcome.anchor_delta, None, "a single improve cycle has no anchor suite to pool");
    assert_eq!(outcome.worst_block_delta, None, "a single improve cycle has no anchor blocks");

    let published = rl::improve::latest_adapter(&adapters).expect("the adapter directory must exist either way");
    match outcome.decision {
        brain::Decision::Promote => {
            let (version, path) = published.expect("a promoted cycle must leave an adapter the serving-side watcher can find");
            assert_eq!(version, 0, "the first adapter published into an empty directory is version 0");
            assert_eq!(path.file_name().unwrap(), "adapter-000000.safetensors");
            assert!(path.metadata().expect("the published adapter must be a real file").len() > 0);
            assert_eq!(outcome.adapter_path.as_deref(), Some(path.as_path()), "the outcome must name the adapter it published");
        }
        brain::Decision::Reject(_) => {
            assert!(published.is_none(), "a rejected cycle must publish no adapter at all");
            assert!(outcome.adapter_path.is_none());
        }
    }
}

/// The serving engine samples the groups by default; the model's own
/// forward stays available as the oracle, and both run a cycle.
#[test]
fn a_cycle_can_sample_with_the_models_own_forward_instead_of_the_engine() {
    if skip() {
        return;
    }
    let dir = tmp("own-rollouts");
    let base = dir.join("base.safetensors");
    write_base(&base);
    assert!(!ImproveOptions::default().model_rollouts, "the serving engine is the default");
    let outcome = Improve::from_pretrained(base.to_string_lossy().into_owned())
        .expect("from_pretrained")
        .adapter_dir(dir.join("adapters"))
        .work_dir(dir.join("work"))
        .lora(4)
        .seed(brain::random_seed())
        .held_out_seed(brain::random_seed())
        .run(FixedTargetEnv { prompt: vec![1, 2, 3], target: vec![4, 5, 6] }, FracMatchVerifier, &ImproveOptions { steps: 4, max_new: 8, model_rollouts: true, ..ImproveOptions::default() })
        .expect("the improve cycle must run");
    assert!(outcome.p_value.is_finite());
}

/// An unregistered `--arch` is refused by name, listing the ones that are -
/// the same contract `DocumentStudy` gives (`crates/sdk/tests/
/// document_study.rs`).
#[test]
fn an_unregistered_architecture_is_refused_and_names_the_registered_ones() {
    let dir = tmp("bad-arch");
    let base = dir.join("base.safetensors");
    write_base(&base);
    let err = Improve::from_pretrained(base.to_string_lossy().into_owned())
        .expect("from_pretrained")
        .arch("not-a-real-arch")
        .adapter_dir(dir.join("adapters"))
        .run(FixedTargetEnv { prompt: vec![1], target: vec![2] }, FracMatchVerifier, &ImproveOptions::default());
    let msg = format!("{}", err.expect_err("an unregistered arch must be refused"));
    assert!(msg.contains("not-a-real-arch"), "{msg}");
    for arch in Improve::architectures() {
        assert!(msg.contains(arch), "the message must name the registered architectures: {msg}");
    }
}

/// A cap on the training row, shorter than the model's own context, still runs
/// a cycle to a verdict: the rows are as long as the prompt and the completion
/// need, not as long as the model could take.
#[test]
fn a_cycle_trains_on_rows_capped_below_the_models_context() {
    if skip() {
        return;
    }
    let dir = tmp("capped");
    let base = dir.join("base.safetensors");
    write_base(&base);
    let outcome = Improve::from_pretrained(base.to_string_lossy().into_owned())
        .expect("from_pretrained")
        .adapter_dir(dir.join("adapters"))
        .work_dir(dir.join("work"))
        .lora(4)
        .seed(brain::random_seed())
        .held_out_seed(brain::random_seed())
        // The model's context is 32; a prompt of 3 and 8 new tokens need 11.
        .run(FixedTargetEnv { prompt: vec![1, 2, 3], target: vec![4, 5, 6] }, FracMatchVerifier, &ImproveOptions { steps: 4, max_new: 8, max_seq_len: Some(16), ..ImproveOptions::default() })
        .expect("a capped cycle must run");
    assert!(outcome.p_value.is_finite());
}

/// A cap that cannot hold the prompt and the completion is an error that says
/// so, before anything trains.
#[test]
fn a_cap_too_small_for_the_prompt_and_the_completion_is_refused_by_name() {
    let dir = tmp("cap-too-small");
    let base = dir.join("base.safetensors");
    write_base(&base);
    let err = Improve::from_pretrained(base.to_string_lossy().into_owned())
        .expect("from_pretrained")
        .adapter_dir(dir.join("adapters"))
        .work_dir(dir.join("work"))
        .run(FixedTargetEnv { prompt: vec![1, 2, 3], target: vec![4] }, FracMatchVerifier, &ImproveOptions { steps: 1, max_new: 8, max_seq_len: Some(8), ..ImproveOptions::default() })
        .expect_err("a cap below prompt + max_new must be refused");
    let msg = format!("{err}");
    assert!(msg.contains("max_seq_len") && msg.contains("11"), "{msg}");
}

/// The default leaves the cap off: a cycle trains on the model's own context
/// unless the caller says otherwise.
#[test]
fn the_cap_is_off_by_default() {
    assert_eq!(ImproveOptions::default().max_seq_len, None);
}

/// Prompts "The capital of <country> is" and the first token of the capital's name: the base answers
/// most but not all of them at temperature 1, so a group of samples earns different rewards.
struct CapitalEnv {
    tasks: Vec<(Vec<u32>, u32)>,
}

impl Environment for CapitalEnv {
    fn name(&self) -> &str {
        "capitals"
    }
    fn tasks(&self, seed: u64) -> Vec<Task> {
        self.tasks.iter().enumerate().map(|(i, (prompt, answer))| Task { id: format!("capital-{seed}-{i}"), prompt: prompt.clone(), answer: serde_json::json!(answer) }).collect()
    }
}

#[derive(Clone)]
struct FirstTokenVerifier;

impl Verifier for FirstTokenVerifier {
    fn verify(&self, task: &Task, _transcript: &[Step], completion: &[u32]) -> Reward {
        let want: u32 = serde_json::from_value(task.answer.clone()).expect("answer is a token id");
        Reward { value: f32::from(completion.first() == Some(&want)), parts: Default::default() }
    }
}

/// One real-weight improve cycle on Qwen3-4B, on the serving engine and on the model's own sampler: the
/// numbers a ledger records (wall time of the cycle, which is dominated by weight loads at this length,
/// and the training steps). Ignored: it needs the checkpoint converted to a brain file
/// (`brain qwen3 import --hf <Qwen3-4B dir> --out testdata/improve/qwen3-4b.safetensors`) with the model's
/// `tokenizer.json` beside it, and a GPU with room for the f32 model twice.
#[test]
#[ignore = "a real-weight measurement, run on a GPU"]
fn a_real_4b_improve_cycle_runs_end_to_end_on_the_engine_path() {
    let base = brain_testutil::testdata_path("improve/qwen3-4b.safetensors");
    let tokenizer_path = brain_testutil::testdata_path("improve/tokenizer.json");
    if !base.is_file() || !tokenizer_path.is_file() {
        brain_testutil::skip_unavailable("testdata/improve holds no converted Qwen3-4B checkpoint and tokenizer.json");
        return;
    }
    let tokenizer = data::qwen_tokenizer::QwenBpe::from_file(tokenizer_path.to_str().expect("utf-8 path")).expect("a tokenizer");
    use data::Tokenizer as _;
    let pairs = [("France", " Paris"), ("Japan", " Tokyo"), ("Italy", " Rome"), ("Spain", " Madrid"), ("Canada", " Ottawa"), ("Australia", " Canberra"), ("Brazil", " Bras\u{ed}lia"), ("Egypt", " Cairo")];
    let tasks: Vec<(Vec<u32>, u32)> = pairs.iter().map(|(country, capital)| (tokenizer.encode(&format!("The capital of {country} is")), tokenizer.encode(capital)[0])).collect();
    for model_rollouts in [false, true] {
        let dir = tmp(if model_rollouts { "real-model" } else { "real-engine" });
        let started = std::time::Instant::now();
        let outcome = Improve::from_pretrained(base.to_string_lossy().into_owned())
            .expect("from_pretrained")
            .adapter_dir(dir.join("adapters"))
            .work_dir(dir.join("work"))
            .lora(8)
            .seed(7)
            .held_out_seed(9)
            .run(CapitalEnv { tasks: tasks.clone() }, FirstTokenVerifier, &ImproveOptions { group_size: 8, steps: 6, max_new: 4, max_seq_len: Some(32), temperature: 1.0, lr: 2e-4, model_rollouts, ..ImproveOptions::default() })
            .expect("the improve cycle runs");
        println!("REAL-IMPROVE rollouts={} wall={:.1}s decision={:?} p={:.3}", if model_rollouts { "model" } else { "engine" }, started.elapsed().as_secs_f64(), outcome.decision, outcome.p_value);
    }
}
