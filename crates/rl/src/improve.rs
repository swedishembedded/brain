// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `rl::improve::cycle` (self-improve roadmap P18): the generic integration
//! that generalizes P5/P6a's qwen3-only `rl::continuous::run_cycle` into
//! rollout (P10) over an environment's explore split (P11) -> train via
//! whichever objective is configured (P12-P15) -> gate against the
//! incumbent (P16) -> stamp lineage (P16) -> version the checkpoint.
//! Generic over any `model::Model` that has adopted weighted-loss support
//! and device-adapter LoRA save/fold - not qwen3-specific.
//!
//! ## Two structural assertions, not two claims
//!
//! [`explore_anchor_split`] asserts (by hashing every task's own `id`, never
//! merely assuming disjoint seed ranges) that the held-out/anchor split
//! shares no task with the explore split - a held-out score is only honest
//! if the policy never trained on the thing it is being scored against.
//!
//! [`assert_trained_spans_were_sampled`] is the "no label ever enters
//! training" check: every completion span [`cycle`]'s objective actually
//! wrote into a training row must be a member of the MULTISET of completions
//! that objective actually sampled this cycle (see
//! [`crate::objective::grpo::CycleLog`]) - a reference answer spliced
//! straight from `Task::answer` into a training batch, bypassing the policy
//! entirely, would not appear in that multiset and this assertion would
//! catch it.
//!
//! Swedish Embedded AB builds the training-lineage machinery that turns
//! "the model got better" from a claim into a checked property - rollout,
//! verifiable reward, and a real promote/reject gate wired end to end. If
//! your team needs a self-improvement loop it can actually trust, you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use data::rng::Rng;
use model::rollout::{ModelRollout, Rollout, RolloutParams};
use model::{FitOpts, Model, ModelConfig, Objective};

use crate::env::{Environment, Task, Verifier};
use crate::gate::{gate, Decision, GateConfig, GateInput, GateReport};

/// Explore vs. held-out/anchor task split for one improve cycle. `anchor` is
/// asserted disjoint from `explore` (hashed by [`Task::id`]) by
/// [`explore_anchor_split`] - never merely assumed from non-overlapping seed
/// ranges, since an `Environment` is free to map two different seeds onto
/// the same task identity.
pub struct Split {
    pub explore: Vec<Task>,
    pub anchor: Vec<Task>,
}

/// Build [`Split`] from `env`, one task per seed in each of `explore_seeds`/
/// `anchor_seeds`, then assert the two resulting task-id sets are disjoint.
/// Panics (loudly, naming every colliding id) on a violation rather than
/// silently letting a held-out score leak an explore-time task - see the
/// module doc comment.
pub fn explore_anchor_split<E: Environment>(env: &E, explore_seeds: &[u64], anchor_seeds: &[u64]) -> Split {
    let explore = tasks_for_seeds(env, explore_seeds);
    let anchor = tasks_for_seeds(env, anchor_seeds);
    let explore_ids: HashSet<&str> = explore.iter().map(|t| t.id.as_str()).collect();
    let overlap: Vec<&str> = anchor.iter().map(|t| t.id.as_str()).filter(|id| explore_ids.contains(id)).collect();
    assert!(
        overlap.is_empty(),
        "rl::improve::explore_anchor_split: anchor/held-out task(s) {overlap:?} are not disjoint from the explore split - \
         a held-out score is only honest if the policy never trained on the task it is scored against"
    );
    Split { explore, anchor }
}

fn tasks_for_seeds<E: Environment>(env: &E, seeds: &[u64]) -> Vec<Task> {
    seeds
        .iter()
        .map(|&s| env.tasks(s).into_iter().next().expect("rl::improve: Environment::tasks produced no task for this seed"))
        .collect()
}

/// The "no label ever enters training" structural check: every span in
/// `trained` must be a member of the MULTISET `sampled` (an item used twice
/// in `trained` needs two matching occurrences in `sampled`, not one).
/// Panics naming the first offending span - a span present in `trained` but
/// absent from `sampled` did not come from the policy's own rollout.
pub fn assert_trained_spans_were_sampled(trained: &[Vec<u32>], sampled: &[Vec<u32>]) {
    let mut remaining: Vec<&Vec<u32>> = sampled.iter().collect();
    for span in trained {
        let pos = remaining.iter().position(|&s| s == span);
        match pos {
            Some(i) => {
                remaining.remove(i);
            }
            None => panic!(
                "rl::improve::assert_trained_spans_were_sampled: a trained completion span {span:?} is not a member of \
                 the multiset of completions the policy actually sampled this cycle - a label (or otherwise \
                 non-sampled data) reached training"
            ),
        }
    }
}

/// The programmatic promote/reject decision plus everything a caller needs
/// to act on it. `adapter_path` is `Some` only on [`Decision::Promote`] - a
/// rejected candidate produces no new servable artifact, the incumbent is
/// simply retained.
pub struct CycleOutcome {
    pub decision: Decision,
    pub report: GateReport,
    pub adapter_path: Option<PathBuf>,
}

/// Everything [`cycle`] needs to save a promoted candidate as a versioned,
/// loadable LoRA adapter via [`model::lora::device_adapter::save_adapter`] -
/// architecture-specific bits the CALLER supplies, so this module stays
/// generic over `M: Model` rather than hardcoding qwen3's `LoraCfg`.
pub struct AdapterMeta<'a> {
    pub rank: u32,
    pub alpha: f32,
    pub targets: &'a [String],
    /// The `ModelCard.family` tag (e.g. `"qwen"`, `"qwen35"`).
    pub family: &'a str,
    pub base_id: &'a str,
    pub dataset_id: Option<&'a str>,
}

/// The training-lineage facts [`cycle`] cannot derive on its own (regime
/// name, hyperparameters, which environment) - everything else
/// ([`checkpoint::st::TrainingProvenance::code_revision`], the gate outcome,
/// `trained_from`) is filled in by [`cycle`] itself.
pub struct ProvenanceInput {
    pub regime: String,
    pub seed: u64,
    pub hyperparams: serde_json::Value,
    pub environment: String,
    pub cycle: u64,
}

/// One `adapter-{n:06}.safetensors` version past the highest one already in
/// `dir` (0 if none yet). Deliberately NOT `read_dir().count()` - deleting an
/// old adapter must not shift every later version down and silently
/// overwrite the next one produced. Shared by [`cycle`] and (behind the
/// `qwen3` feature) `crate::continuous::run_cycle`, which used to carry its
/// own copy of exactly this logic.
pub(crate) fn next_adapter_version(dir: &Path) -> std::io::Result<u32> {
    let max = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .filter_map(|name| name.strip_prefix("adapter-")?.strip_suffix(".safetensors")?.parse::<u32>().ok())
        .max();
    Ok(max.map(|v| v + 1).unwrap_or(0))
}

/// `git rev-parse --short HEAD`, `-dirty`-suffixed when the working tree is
/// not clean, `"unknown"` if git is unavailable - what
/// [`checkpoint::st::TrainingProvenance::code_revision`] records: reproducing
/// a promoted checkpoint means checking out this exact revision.
fn code_revision() -> String {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());
    let Some(commit) = commit else { return "unknown".to_string() };
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    if dirty {
        format!("{commit}-dirty")
    } else {
        commit
    }
}

/// Mean softmax entropy (nats) of the policy's OWN per-position distribution
/// over `completion`'s span, read back via [`Model::logits_all`] - not the
/// logprob of the token that happened to be sampled ([`model::rollout::
/// Completion::logprobs`] only carries that), the full categorical
/// distribution's entropy at each decoded position. This is [`gate`]'s
/// non-degeneracy signal: a policy collapsing onto one output regardless of
/// input has near-zero entropy here even though it may still decode
/// correctly.
fn mean_completion_entropy<M: Model>(model: &M, prompt: &[u32], completion: &[u32]) -> f64 {
    if completion.is_empty() || prompt.is_empty() {
        return 0.0;
    }
    let full: Vec<u32> = prompt.iter().chain(completion.iter()).copied().collect();
    let Some(logits) = model.logits_all(&full) else { return 0.0 };
    let vocab = logits.len() / full.len();
    let mut total = 0.0f64;
    for i in 0..completion.len() {
        let pos = prompt.len() - 1 + i;
        let row = &logits[pos * vocab..(pos + 1) * vocab];
        total += softmax_entropy(row) as f64;
    }
    total / completion.len() as f64
}

fn softmax_entropy(logits: &[f32]) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(f32::EPSILON);
    -exps.iter().map(|&e| e / sum).filter(|&p| p > 0.0).map(|p| p * p.ln()).sum::<f32>()
}

/// Load `path` as an `M`, greedily decode every `held_out` task's prompt,
/// score each completion with `verifier`, and return `(per-task scores, mean
/// completion entropy)`. Both arms of [`cycle`]'s gate call this on
/// disk-reloaded checkpoints - never the freshly-trained in-memory instance
/// (the `lora_learning_gate` lesson [`crate::gate`] itself documents: what is
/// actually served must be what is scored).
fn score_checkpoint<M: Model>(path: &Path, held_out: &[Task], verifier: &dyn Verifier, rollout: &RolloutParams) -> (Vec<f64>, f64) {
    let c = checkpoint::load(path.to_str().expect("utf-8 path"));
    let cfg = M::Config::from_json(&c.header["config"]);
    let init = c.by_role("");
    let model = M::new(cfg.clone(), 1, cfg.block_size(), &init);
    let mut roll = ModelRollout::new(&model);
    let mut rng = Rng::new(0); // greedy: rollout::RolloutParams::sample must be SampleParams::greedy()
    let mut scores = Vec::with_capacity(held_out.len());
    let mut entropies = Vec::with_capacity(held_out.len());
    for task in held_out {
        let completion = roll.sample_n(&task.prompt, 1, rollout, &mut rng).pop().expect("sample_n(1) returns exactly one completion");
        scores.push(verifier.verify(task, &[], &completion.tokens).value as f64);
        entropies.push(mean_completion_entropy(&model, &task.prompt, &completion.tokens));
    }
    let mean_entropy = if entropies.is_empty() { 0.0 } else { entropies.iter().sum::<f64>() / entropies.len() as f64 };
    (scores, mean_entropy)
}

/// Score both arms (both decoded greedily from disk-reloaded checkpoints,
/// same held-out tasks, same order - see [`score_checkpoint`]) and run
/// [`gate::gate`] over the result. The anchor-suite headline
/// ([`GateInput::anchor_candidate`]/`anchor_incumbent`) is this same
/// held-out set's mean score: a deliberate simplification for a single-
/// environment cycle (a caller wiring a real, separate anchor suite - e.g.
/// a retention-matrix benchmark run across many task families - passes its
/// own headline scores through [`gate::gate`] directly instead of calling
/// this).
fn score_and_gate<M: Model>(
    incumbent_path: &Path,
    candidate_path: &Path,
    held_out: &[Task],
    verifier: &dyn Verifier,
    rollout: &RolloutParams,
    cfg: &GateConfig,
) -> GateReport {
    let (incumbent_scores, incumbent_entropy) = score_checkpoint::<M>(incumbent_path, held_out, verifier, rollout);
    let (candidate_scores, candidate_entropy) = score_checkpoint::<M>(candidate_path, held_out, verifier, rollout);
    let anchor_incumbent = mean(&incumbent_scores);
    let anchor_candidate = mean(&candidate_scores);
    let input = GateInput {
        candidate_scores: &candidate_scores,
        incumbent_scores: &incumbent_scores,
        anchor_candidate,
        anchor_incumbent,
        entropy_candidate: candidate_entropy,
        entropy_incumbent: incumbent_entropy,
    };
    gate(&input, cfg)
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn gate_outcome(report: &GateReport) -> checkpoint::st::GateOutcome {
    let decision = match report.decision {
        Decision::Promote => "promote".to_string(),
        Decision::Reject(cause) => format!("reject: {cause:?}"),
    };
    checkpoint::st::GateOutcome {
        decision,
        p_value: report.p_value,
        effect_size: report.effect_size,
        anchor_delta: report.anchor_delta,
        entropy_ratio: report.entropy_ratio,
    }
}

/// One `rl::improve` cycle: train `objective` (already configured over
/// whichever `Environment`/`Verifier` and P12-P15 objective the caller
/// chose) starting from `base_checkpoint`'s weights, gate the result against
/// that same incumbent on `held_out`, and - only on [`Decision::Promote`] -
/// save a versioned, lineage-stamped LoRA adapter. A rejected candidate's
/// full training checkpoint (`train_out`) is left on disk for inspection but
/// no adapter is produced - the incumbent (`base_checkpoint`) is what stays
/// servable.
#[allow(clippy::too_many_arguments)]
pub fn cycle<M: Model, O: Objective<M>>(
    base_checkpoint: &Path,
    objective: O,
    held_out: &[Task],
    verifier: &dyn Verifier,
    rollout: &RolloutParams,
    opts: &FitOpts,
    train_out: &Path,
    adapter_out_dir: &Path,
    adapter: AdapterMeta,
    provenance: ProvenanceInput,
    gate_cfg: &GateConfig,
) -> std::io::Result<CycleOutcome> {
    let base = checkpoint::load(base_checkpoint.to_str().expect("utf-8 path"));
    let base_cfg = M::Config::from_json(&base.header["config"]);
    let init = base.by_role("");
    let model = M::new(base_cfg.clone(), 1, base_cfg.block_size(), &init);

    model::fit_with(model, objective, opts, Some(train_out))?;

    let report = score_and_gate::<M>(base_checkpoint, train_out, held_out, verifier, rollout, gate_cfg);

    let adapter_path = match report.decision {
        Decision::Promote => {
            let trained = checkpoint::load(train_out.to_str().expect("utf-8 path"));
            let trained_cfg = M::Config::from_json(&trained.header["config"]);
            let trained_init = trained.by_role("");
            let trained_model = M::new(trained_cfg.clone(), 1, trained_cfg.block_size(), &trained_init);

            std::fs::create_dir_all(adapter_out_dir)?;
            let version = next_adapter_version(adapter_out_dir)?;
            let card_id = format!("adapter-{version:06}");
            let path = adapter_out_dir.join(format!("{card_id}.safetensors"));
            model::lora::device_adapter::save_adapter(
                path.to_str().expect("utf-8 path"),
                &trained_model,
                adapter.rank,
                adapter.alpha,
                adapter.targets,
                &card_id,
                adapter.base_id,
                adapter.family,
                adapter.dataset_id,
            )?;

            let training = checkpoint::st::TrainingProvenance {
                code_revision: code_revision(),
                regime: provenance.regime,
                seed: provenance.seed,
                hyperparams: provenance.hyperparams,
                environment: provenance.environment,
                gate: Some(gate_outcome(&report)),
                trained_from: Some(adapter.base_id.to_string()),
                cycle: provenance.cycle,
            };
            stamp_lineage(&path, training)?;
            Some(path)
        }
        Decision::Reject(_) => None,
    };

    Ok(CycleOutcome { decision: report.decision, report, adapter_path })
}

/// Attach `training` to `path`'s already-written [`checkpoint::st::
/// ModelCard`] (round-tripping the file once) - the generic post-processing
/// step that turns any [`model::lora::device_adapter::save_adapter`] output
/// into a lineage-stamped artifact, independent of which `Model` produced it.
fn stamp_lineage(path: &Path, training: checkpoint::st::TrainingProvenance) -> std::io::Result<()> {
    let st = checkpoint::st::load_safetensors(path.to_str().expect("utf-8 path"))?;
    let mut card = st
        .card()
        .unwrap_or_else(|| panic!("rl::improve::stamp_lineage: {} has no ModelCard to stamp lineage onto", path.display()));
    card.training = Some(training);
    let config = st.config();
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = st.tensors.into_iter().map(|(name, data)| (name.clone(), vec![data.len() as u64], data)).collect();
    checkpoint::st::save_safetensors(path.to_str().expect("utf-8 path"), &tensors, &config, Some(&card))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoEnv;
    impl Environment for EchoEnv {
        fn name(&self) -> &str {
            "toy-echo"
        }
        fn tasks(&self, seed: u64) -> Vec<Task> {
            vec![Task { id: format!("echo-{seed}"), prompt: vec![1, 2, 3], answer: serde_json::json!([4, 5]) }]
        }
    }

    #[test]
    fn explore_anchor_split_accepts_a_genuinely_disjoint_split() {
        let split = explore_anchor_split(&EchoEnv, &[0, 1, 2], &[100, 101]);
        assert_eq!(split.explore.len(), 3);
        assert_eq!(split.anchor.len(), 2);
    }

    #[test]
    #[should_panic(expected = "are not disjoint from the explore split")]
    fn explore_anchor_split_panics_on_a_colliding_seed() {
        // Seed 2 appears in both lists -> the SAME task id ("echo-2") on
        // both sides - exactly the leak this assertion exists to catch.
        let _ = explore_anchor_split(&EchoEnv, &[0, 1, 2], &[2, 3]);
    }

    #[test]
    fn assert_trained_spans_were_sampled_accepts_a_true_subset_multiset() {
        let sampled = vec![vec![1, 2], vec![3, 4], vec![1, 2]];
        let trained = vec![vec![1, 2], vec![1, 2]];
        assert_trained_spans_were_sampled(&trained, &sampled); // must not panic
    }

    #[test]
    #[should_panic(expected = "is not a member of")]
    fn assert_trained_spans_were_sampled_rejects_a_span_that_was_never_sampled() {
        let sampled = vec![vec![1, 2], vec![3, 4]];
        // [9, 9] never appeared in `sampled` - stands in for a label that
        // reached training without ever being sampled from the policy.
        let trained = vec![vec![1, 2], vec![9, 9]];
        assert_trained_spans_were_sampled(&trained, &sampled);
    }

    #[test]
    #[should_panic(expected = "is not a member of")]
    fn assert_trained_spans_were_sampled_rejects_using_the_same_sample_twice() {
        // Only ONE [1, 2] was ever sampled - training on it twice must not
        // be silently accepted as "it was sampled" (multiset, not set).
        let sampled = vec![vec![1, 2], vec![3, 4]];
        let trained = vec![vec![1, 2], vec![1, 2]];
        assert_trained_spans_were_sampled(&trained, &sampled);
    }

    #[test]
    fn softmax_entropy_is_zero_for_a_one_hot_distribution_and_max_for_uniform() {
        let one_hot = [100.0, -100.0, -100.0, -100.0];
        assert!(softmax_entropy(&one_hot) < 1e-3);

        let uniform = [0.0, 0.0, 0.0, 0.0];
        let expected_max = (4.0f32).ln();
        assert!((softmax_entropy(&uniform) - expected_max).abs() < 1e-4);
    }
}
