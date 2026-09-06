// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end plumbing test for `rl::objective::grpo::Grpo`: a real
//! `qwen3::Qwen`, a toy self-contained `Environment`/`Verifier` (a fixed
//! prompt with a fixed target continuation, scored by fraction of matching
//! tokens - no model-as-judge, deterministic, matching `rl::env`'s own
//! contract), driven through `model::fit_with` for a handful of steps. This
//! is NOT the gradient-correctness gate (`tests/grpo_gradcheck.rs` is) -
//! it proves the rollout -> verify -> group-advantage -> pack -> weighted
//! backward pipeline actually runs to completion against a real `Model`
//! without panicking and without producing non-finite losses, the thing a
//! pure-math unit test and a fixed-batch gradcheck harness cannot cover on
//! their own.

use qwen3::{Qwen, QwenConfig};
use rl::env::{Environment, Reward, Step, Task, Verifier};
use rl::objective::grpo::{Grpo, GrpoConfig};

/// A single fixed prompt whose target continuation is a fixed token
/// sequence - `Task::answer` carries the target (what a verifier needs to
/// RECOMPUTE correctness), never a label baked into the reward directly.
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

/// Programmatic, deterministic reward: fraction of the completion's tokens
/// that match the target at the same position.
struct FracMatchVerifier;

impl Verifier for FracMatchVerifier {
    fn verify(&self, task: &Task, _transcript: &[Step], completion: &[u32]) -> Reward {
        let target: Vec<u32> = serde_json::from_value(task.answer.clone()).expect("answer is a token array");
        let matches = completion.iter().zip(target.iter()).filter(|(a, b)| a == b).count();
        let value = matches as f32 / target.len().max(1) as f32;
        Reward { value, parts: Default::default() }
    }
}

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

/// GRPO (`group_size = 2`): real sampling variance across the group is what
/// exercises `group_advantages`'s non-degenerate z-score path, not just its
/// zero-variance drop.
#[test]
fn grpo_group_size_two_runs_end_to_end_through_fit_with() {
    if gpu_disabled() {
        return;
    }
    let cfg = QwenConfig::tiny();
    let init = qwen3::init_weights(&cfg, 7);
    let mut model = Qwen::new(cfg, 1, 8, &init);
    model::Model::enable_weighted_loss(&mut model);

    let env = FixedTargetEnv { prompt: vec![1, 2, 3], target: vec![4, 5, 6] };
    let grpo_cfg = GrpoConfig {
        group_size: 2,
        clip_eps: 0.2,
        kl_beta: 0.0,
        seq_len: 8,
        rollout: model::rollout::RolloutParams {
            max_new: 3,
            sample: model::serve::SampleParams { temp: 1.0, top_k: 0, top_p: 1.0 },
            eos: None,
        },
        max_attempts: 1,
    };
    let obj = Grpo::new(env, FracMatchVerifier, grpo_cfg);

    let opts = model::FitOpts {
        steps: 2,
        grad_accum: 2, // one full group's worth of rows per optimizer step
        lr: 1e-3,
        min_lr: 1e-4,
        warmup: 0,
        decay_iters: 10,
        weight_decay: 0.0,
        grad_clip: 1.0,
        eval_interval: 0,
        eval_batches: 0,
        seed: 123,
        checkpoint_secs: 0,
        ..model::FitOpts::default()
    };

    let (initial, last) = model::fit_with(model, obj, &opts, None).expect("fit_with");
    assert!(initial.is_finite(), "initial loss {initial} is not finite");
    assert!(last.is_finite(), "final loss {last} is not finite");
}

/// RFT/STaR (`group_size = 1`): rejection sampling with a small
/// `max_attempts`, exercising the code path that trains uniform-weight on a
/// single verified completion (or, when nothing verifies within the
/// attempt budget, a legitimate no-op micro-step).
#[test]
fn grpo_group_size_one_is_the_rft_star_path_and_runs_end_to_end() {
    if gpu_disabled() {
        return;
    }
    let cfg = QwenConfig::tiny();
    let init = qwen3::init_weights(&cfg, 7);
    let mut model = Qwen::new(cfg, 1, 8, &init);
    model::Model::enable_weighted_loss(&mut model);

    let env = FixedTargetEnv { prompt: vec![1, 2, 3], target: vec![4, 5, 6] };
    let grpo_cfg = GrpoConfig {
        group_size: 1,
        clip_eps: 0.2,
        kl_beta: 0.0,
        seq_len: 8,
        rollout: model::rollout::RolloutParams {
            max_new: 3,
            sample: model::serve::SampleParams { temp: 1.0, top_k: 0, top_p: 1.0 },
            eos: None,
        },
        max_attempts: 4,
    };
    let obj = Grpo::new(env, FracMatchVerifier, grpo_cfg);

    let opts = model::FitOpts {
        steps: 2,
        grad_accum: 1,
        lr: 1e-3,
        min_lr: 1e-4,
        warmup: 0,
        decay_iters: 10,
        weight_decay: 0.0,
        grad_clip: 1.0,
        eval_interval: 0,
        eval_batches: 0,
        seed: 321,
        checkpoint_secs: 0,
        ..model::FitOpts::default()
    };

    let (initial, last) = model::fit_with(model, obj, &opts, None).expect("fit_with");
    assert!(initial.is_finite(), "initial loss {initial} is not finite");
    assert!(last.is_finite(), "final loss {last} is not finite");
}
