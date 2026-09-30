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

/// A rollout that is not the trained model: it records what it is synced
/// with, and hands back fixed completions whose own logprobs are absurd, so
/// only a trainer that recomputes them gets the on-policy ratio of one.
struct RecordingRollout {
    syncs: std::rc::Rc<std::cell::RefCell<Vec<Vec<String>>>>,
}

impl model::rollout::Rollout for RecordingRollout {
    fn sample_n(&mut self, _prompt: &[u32], n: usize, params: &model::rollout::RolloutParams, _rng: &mut data::rng::Rng) -> Vec<model::rollout::Completion> {
        let fixed = [vec![4u32, 5, 6], vec![7, 7, 7]];
        (0..n)
            .map(|i| model::rollout::Completion { tokens: fixed[i % 2][..params.max_new].to_vec(), logprobs: vec![100.0; params.max_new], stop: model::rollout::StopReason::MaxNew })
            .collect()
    }
}

impl model::rollout::SyncRollout for RecordingRollout {
    fn sync(&mut self, trained: &std::collections::HashMap<String, Vec<f32>>) -> Result<(), String> {
        let mut names: Vec<String> = trained.keys().cloned().collect();
        names.sort();
        self.syncs.borrow_mut().push(names);
        Ok(())
    }
}

/// GRPO sampling through an external rollout (a serving engine on another
/// card, say): the rollout is synced with the trainer's trainable tensors
/// before its first group and then every `sync_every` groups, and the
/// trainer recomputes the sampled tokens' old logprobs itself, so a fresh
/// group's first update sees a probability ratio of exactly one.
#[test]
fn an_external_rollout_is_synced_on_schedule_and_its_logprobs_are_recomputed() {
    if gpu_disabled() {
        return;
    }
    let cfg = QwenConfig { lora: Some(qwen3::LoraCfg::attn(2, 4.0)), ..QwenConfig::tiny() };
    let init = qwen3::init_weights(&cfg, 7);
    let mut model = Qwen::new(cfg, 1, 8, &init);
    let syncs = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let grpo_cfg = GrpoConfig {
        group_size: 2,
        clip_eps: 0.2,
        kl_beta: 0.0,
        seq_len: 8,
        rollout: model::rollout::RolloutParams { max_new: 3, sample: model::serve::SampleParams { temp: 1.0, top_k: 0, top_p: 1.0 }, eos: None },
        max_attempts: 1,
    };
    let env = FixedTargetEnv { prompt: vec![1, 2, 3], target: vec![4, 5, 6] };
    let mut obj = Grpo::new(env, FracMatchVerifier, grpo_cfg).with_rollout(Box::new(RecordingRollout { syncs: syncs.clone() }), 2);
    use model::Objective;
    obj.prepare(&mut model);
    let mut rng = data::rng::Rng::new(5);
    // Each group yields two rows: eight micro-steps are four groups.
    let losses: Vec<f32> = (0..8).map(|_| obj.micro_step(&model, &mut rng)).collect();

    // The first row is the target (advantage +1): at a ratio of one its
    // per-token loss is -1.
    assert!((losses[0] + 1.0).abs() < 1e-3, "on-policy loss {} (the rollout's own logprobs were used)", losses[0]);
    let mut want: Vec<String> = model::Model::optimized_params(&model).expect("a LoRA model trains its adapters only");
    want.sort();
    let syncs = syncs.borrow();
    assert_eq!(syncs.len(), 2, "synced before groups 0 and 2");
    assert!(syncs.iter().all(|s| *s == want), "the payload is the trainable tensors: {:?}", syncs[0]);
    assert!(want.iter().all(|n| n.contains(".lora_")), "{want:?}");
}
