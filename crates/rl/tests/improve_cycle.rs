// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `rl::improve::cycle` end-to-end gate (self-improve roadmap P18): a real,
//! tiny `qwen3::Qwen`, driven through rollout (P10) over a self-contained
//! `Environment`/`Verifier` (P11), GRPO (P12) as "whichever objective is
//! configured", `model::fit_with` (P8), the paired-sign-test gate (P16), and
//! lineage stamping (P16) - end to end, producing a gated, lineage-stamped
//! adapter.
//!
//! Two scenarios prove the gate is a gate, not a rubber stamp:
//! `improve_cycle_end_to_end_promotes_a_genuinely_better_candidate` trains a
//! real candidate to reliably beat an untrained incumbent on the held-out
//! split and asserts `Decision::Promote` plus a real, loadable,
//! lineage-stamped adapter; `improve_cycle_rejects_a_deliberately_worse_
//! candidate` takes THAT promoted checkpoint as the new incumbent and runs a
//! second cycle whose "objective" deliberately randomizes every weight
//! before scoring (`Objective::micro_step` calling `Model::write_weight`
//! with fresh noise, no forward/backward at all - the literal "corrupt its
//! weights before gating" the roadmap gate criterion asks for), asserting
//! `Decision::Reject` and that no new adapter is produced - the incumbent is
//! retained.

use std::path::{Path, PathBuf};

use data::rng::Rng;
use model::rollout::RolloutParams;
use model::serve::SampleParams;
use model::{FitOpts, Model};
use qwen3::config::{LoraCfg, QwenConfig};
use qwen3::model::Qwen;
use rl::env::{Environment, Reward, Step, Task, Verifier};
use rl::gate::{Decision, GateConfig};
use rl::improve::{self, AdapterMeta, ProvenanceInput};
use rl::objective::grpo::{CycleLog, Grpo, GrpoConfig};

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-rl-improve-cycle-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A single fixed prompt whose target continuation is a fixed token
/// sequence - `Task::answer` carries the target (what a verifier needs to
/// RECOMPUTE correctness, never a label baked into the reward directly).
/// `seed` only ever affects the returned `Task::id`, never its content -
/// what lets [`improve::explore_anchor_split`]'s disjointness check use
/// plain non-overlapping seed ranges honestly: two different seed ranges
/// really do produce two disjoint sets of task IDENTITIES.
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
/// that match the target at the same position - no model-as-judge anywhere.
struct FracMatchVerifier;

impl Verifier for FracMatchVerifier {
    fn verify(&self, task: &Task, _transcript: &[Step], completion: &[u32]) -> Reward {
        let target: Vec<u32> = serde_json::from_value(task.answer.clone()).expect("answer is a token array");
        let matches = completion.iter().zip(target.iter()).filter(|(a, b)| a == b).count();
        let value = matches as f32 / target.len().max(1) as f32;
        Reward { value, parts: Default::default() }
    }
}

/// A "deliberately worse candidate" objective: every micro-step overwrites
/// EVERY weight with fresh noise via `Model::write_weight` - no forward, no
/// backward, no relationship whatsoever to the verifiable task - the
/// roadmap gate criterion's own literal example ("corrupting/randomizing its
/// weights before gating"), expressed generically through the same
/// `model::Objective` seam a real training regime uses.
struct Sabotage {
    seed: u64,
}

impl<M: Model> model::Objective<M> for Sabotage {
    fn regime(&self) -> &'static str {
        "sabotage"
    }
    fn micro_step(&mut self, model: &M, _rng: &mut Rng) -> f32 {
        let mut rng = Rng::new(self.seed);
        self.seed = self.seed.wrapping_add(1);
        for name in model.param_names() {
            let n = model.read_weight(&name).len();
            let noise: Vec<f32> = (0..n).map(|_| rng.uniform(-2.0, 2.0) as f32).collect();
            model.write_weight(&name, &noise);
        }
        0.0
    }
}

fn write_base_checkpoint(path: &Path, seed: u64) -> QwenConfig {
    let cfg = QwenConfig { lora: Some(LoraCfg::attn(4, 8.0)), ..QwenConfig::tiny() };
    let init = qwen3::init_weights(&cfg, seed);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, n)| {
            let v = init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone();
            (name, vec![n as u64], v)
        })
        .collect();
    checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
    cfg
}

fn rollout_params() -> RolloutParams {
    RolloutParams { max_new: 3, sample: SampleParams { temp: 1.0, top_k: 0, top_p: 1.0 }, eos: None }
}

fn greedy_rollout_params() -> RolloutParams {
    RolloutParams { max_new: 3, sample: SampleParams::greedy(), eos: None }
}

fn env_and_held_out() -> (FixedTargetEnv, Vec<Task>) {
    let env = FixedTargetEnv { prompt: vec![1, 2, 3], target: vec![4, 5, 6] };
    // Disjoint seed ranges -> disjoint task-id sets (structural assertion
    // #2), hashed and asserted by `explore_anchor_split`, not assumed.
    let split = improve::explore_anchor_split(&env, &[0, 1, 2, 3, 4], &[10_000, 10_001, 10_002, 10_003, 10_004, 10_005]);
    (env, split.anchor)
}

#[test]
fn improve_cycle_end_to_end_promotes_a_genuinely_better_candidate() {
    if gpu_disabled() {
        return;
    }
    let dir = tmp("promote");
    let base_path = dir.join("base.safetensors");
    write_base_checkpoint(&base_path, 1);

    let (env, held_out) = env_and_held_out();
    let log = CycleLog::new();
    let grpo_cfg = GrpoConfig { group_size: 2, clip_eps: 0.2, kl_beta: 0.0, seq_len: 12, rollout: rollout_params(), max_attempts: 1 };
    let objective = Grpo::new(env, FracMatchVerifier, grpo_cfg).with_log(log.clone());

    let opts = FitOpts {
        steps: 120,
        grad_accum: 2, // one full GRPO group's worth of rows per optimizer step
        lr: 5e-3,
        min_lr: 5e-4,
        warmup: 0,
        decay_iters: 120,
        weight_decay: 0.0,
        grad_clip: 1.0,
        eval_interval: 0,
        eval_batches: 0,
        seed: 7,
        checkpoint_secs: 0,
        ..FitOpts::default()
    };

    let adapter_dir = dir.join("adapters");
    let train_out = dir.join("train.safetensors");
    let targets: Vec<String> = LoraCfg::attn(4, 8.0).targets;
    let outcome = improve::cycle::<Qwen, _>(
        &base_path,
        objective,
        &held_out,
        &FracMatchVerifier,
        &greedy_rollout_params(),
        &opts,
        &train_out,
        &adapter_dir,
        AdapterMeta { rank: 4, alpha: 8.0, targets: &targets, family: "qwen", base_id: "base", dataset_id: None },
        ProvenanceInput { regime: "grpo".to_string(), seed: 7, hyperparams: serde_json::json!({"group_size": 2, "clip_eps": 0.2}), environment: "cpu/test".to_string(), cycle: 0 },
        &GateConfig { min_entropy_ratio: 0.0, ..GateConfig::default() },
    )
    .expect("cycle");

    assert_eq!(outcome.decision, Decision::Promote, "report: {:?}", outcome.report);
    let adapter_path = outcome.adapter_path.expect("Promote must produce an adapter path");
    assert!(adapter_path.exists(), "the adapter file must actually exist on disk");

    // Structural assertion #1: every completion span this cycle's objective
    // wrote into a training row is a member of the multiset it actually
    // sampled - no label (e.g. the verifier's own `target`) ever entered
    // training directly.
    improve::assert_trained_spans_were_sampled(&log.trained(), &log.sampled());
    assert!(!log.trained().is_empty(), "a 120-step run must have trained on at least one sampled span");

    // The adapter is a real, loadable LoRA adapter with lineage stamped.
    let st = checkpoint::st::load_safetensors(adapter_path.to_str().unwrap()).expect("load adapter");
    let card = st.card().expect("adapter must carry a ModelCard");
    let training = card.training.expect("Promote must stamp TrainingProvenance");
    assert_eq!(training.regime, "grpo");
    assert_eq!(training.cycle, 0);
    assert_eq!(training.seed, 7);
    let gate_outcome = training.gate.expect("a gated cycle must record its GateOutcome");
    assert_eq!(gate_outcome.decision, "promote");

    let base = checkpoint::load(base_path.to_str().unwrap());
    let mut folded = base.by_role("");
    let before = folded.clone();
    qwen3::lora::fold_adapter_into(&mut folded, adapter_path.to_str().unwrap()).expect("fold_adapter_into");
    let any_changed = before.iter().any(|(name, v)| folded.get(name).map(|after| after != v).unwrap_or(false));
    assert!(any_changed, "the promoted adapter must actually change the folded base weights");
}

#[test]
fn improve_cycle_rejects_a_deliberately_worse_candidate_and_retains_the_incumbent() {
    if gpu_disabled() {
        return;
    }
    // Reuse the SAME good, already-promoted checkpoint a real training cycle
    // produces as this cycle's incumbent - a clean, already-trained baseline
    // the sabotaged candidate must be measured against, not an untrained one.
    let dir = tmp("reject");
    let base_path = dir.join("base.safetensors");
    write_base_checkpoint(&base_path, 1);

    let (env, held_out) = env_and_held_out();
    let grpo_cfg = GrpoConfig { group_size: 2, clip_eps: 0.2, kl_beta: 0.0, seq_len: 12, rollout: rollout_params(), max_attempts: 1 };
    let objective = Grpo::new(env, FracMatchVerifier, grpo_cfg);
    let opts = FitOpts {
        steps: 120,
        grad_accum: 2,
        lr: 5e-3,
        min_lr: 5e-4,
        warmup: 0,
        decay_iters: 120,
        weight_decay: 0.0,
        grad_clip: 1.0,
        eval_interval: 0,
        eval_batches: 0,
        seed: 7,
        checkpoint_secs: 0,
        ..FitOpts::default()
    };
    let targets: Vec<String> = LoraCfg::attn(4, 8.0).targets;
    let good = improve::cycle::<Qwen, _>(
        &base_path,
        objective,
        &held_out,
        &FracMatchVerifier,
        &greedy_rollout_params(),
        &opts,
        &dir.join("train.safetensors"),
        &dir.join("adapters"),
        AdapterMeta { rank: 4, alpha: 8.0, targets: &targets, family: "qwen", base_id: "base", dataset_id: None },
        ProvenanceInput { regime: "grpo".to_string(), seed: 7, hyperparams: serde_json::json!({}), environment: "cpu/test".to_string(), cycle: 0 },
        &GateConfig { min_entropy_ratio: 0.0, ..GateConfig::default() },
    )
    .expect("cycle");
    assert_eq!(good.decision, Decision::Promote, "the setup cycle producing this test's incumbent must itself promote: {:?}", good.report);

    // Fold the promoted adapter into the base to get a real, full,
    // servable "incumbent" checkpoint on disk - the same shape
    // `base_checkpoint` always is.
    let incumbent_path = dir.join("incumbent.safetensors");
    let base = checkpoint::load(base_path.to_str().unwrap());
    let mut folded = base.by_role("");
    qwen3::lora::fold_adapter_into(&mut folded, good.adapter_path.as_ref().unwrap().to_str().unwrap()).expect("fold_adapter_into");
    let cfg = QwenConfig { lora: Some(LoraCfg::attn(4, 8.0)), ..QwenConfig::tiny() };
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg.param_list().into_iter().map(|(name, n)| (name.clone(), vec![n as u64], folded.get(&name).unwrap().clone())).collect();
    checkpoint::save(incumbent_path.to_str().unwrap(), cfg.to_json(), &tensors);

    let adapter_dir_2 = dir.join("adapters2");
    let sabotage = Sabotage { seed: 999 };
    let outcome = improve::cycle::<Qwen, _>(
        &incumbent_path,
        sabotage,
        &held_out,
        &FracMatchVerifier,
        &greedy_rollout_params(),
        &FitOpts { steps: 2, grad_accum: 1, eval_interval: 0, eval_batches: 0, seed: 999, checkpoint_secs: 0, ..FitOpts::default() },
        &dir.join("train_sabotaged.safetensors"),
        &adapter_dir_2,
        AdapterMeta { rank: 4, alpha: 8.0, targets: &targets, family: "qwen", base_id: "incumbent", dataset_id: None },
        ProvenanceInput { regime: "sabotage".to_string(), seed: 999, hyperparams: serde_json::json!({}), environment: "cpu/test".to_string(), cycle: 1 },
        &GateConfig::default(),
    )
    .expect("cycle");

    assert!(matches!(outcome.decision, Decision::Reject(_)), "a deliberately corrupted candidate must be rejected, got {:?}", outcome.report);
    assert!(outcome.adapter_path.is_none(), "a rejected cycle must not produce a new adapter - the incumbent is retained");
    assert!(!adapter_dir_2.exists() || std::fs::read_dir(&adapter_dir_2).unwrap().next().is_none(), "no adapter file may appear in the out dir for a rejected cycle");
}
