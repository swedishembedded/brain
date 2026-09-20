// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

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
        .seed(7)
        .held_out_seed(8)
        // `max_new` must leave room for `prompt.len() + max_new <= block_size`
        // (32 here): the prompt is 3 tokens, so 8 is comfortably inside it.
        .run(env, FracMatchVerifier, &ImproveOptions { steps: 8, max_new: 8, ..ImproveOptions::default() })
        .expect("the improve cycle must run");

    assert!(outcome.p_value.is_finite() && (0.0..=1.0).contains(&outcome.p_value), "p_value must be a probability, got {}", outcome.p_value);
    assert!(outcome.effect_size.is_finite());
    assert_eq!(outcome.anchor_delta, 0.0, "a single improve cycle has no retention suite to pool");
    assert_eq!(outcome.worst_block_delta, 0.0, "a single improve cycle has no anchor blocks");

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
