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

/// One arm's per-task scores from the gate's own decode pass, kept rather
/// than collapsed to a mean. A multi-cycle caller (see [`crate::continual`])
/// recovers a whole retention-matrix row from these at ZERO extra decode
/// cost: the gate already had to decode every anchor task on both arms to
/// make its decision, so re-decoding them for a separate "retention report"
/// would be measuring the same model twice.
#[derive(Clone, Debug, Default)]
pub struct ArmScores {
    /// Per task, in [`Evaluation::held_out`] order.
    pub held_out: Vec<f64>,
    /// Per task, in [`Evaluation::anchor`] order (empty when there is no
    /// separate anchor suite).
    pub anchor: Vec<f64>,
    /// Mean completion entropy over the decoded tasks - [`gate`]'s
    /// non-degeneracy signal.
    pub mean_entropy: f64,
    /// How many DISTINCT completions the arm produced across all decoded
    /// tasks, out of `held_out.len() + anchor.len()`.
    ///
    /// The direct, assumption-free mode-collapse measurement, and the one
    /// [`crate::gate::Cause::Degenerate`]'s entropy ratio cannot make on a
    /// task family with exactly ONE correct completion per prompt: there, a
    /// policy that has actually SOLVED the task decodes greedily with
    /// near-zero entropy, so a low entropy ratio is the signature of success
    /// and of collapse alike. Counting distinct completions separates them
    /// outright - a collapsed policy emits one completion for every prompt,
    /// a correct one emits a different completion per prompt.
    pub distinct_completions: usize,
}

/// The programmatic promote/reject decision plus everything a caller needs
/// to act on it. `adapter_path` is `Some` only on [`Decision::Promote`] - a
/// rejected candidate produces no new servable artifact, the incumbent is
/// simply retained.
pub struct CycleOutcome {
    pub decision: Decision,
    pub report: GateReport,
    pub adapter_path: Option<PathBuf>,
    /// The candidate arm's own per-task scores, exactly as the gate saw them.
    pub candidate: ArmScores,
    /// The incumbent arm's, likewise.
    pub incumbent: ArmScores,
}

/// What [`cycle`] scores its two arms on. Grouping these five together (they
/// travel as a unit and are never varied independently) is what retires the
/// eleven-positional-argument `cycle` signature this replaces.
pub struct Evaluation<'a> {
    /// The gate's primary suite: the tasks the candidate must beat the
    /// incumbent on.
    pub held_out: &'a [Task],
    /// The RETENTION suite: tasks from earlier cycles this cycle must not
    /// destroy. EMPTY means "no separate anchor" and the held-out set doubles
    /// as the anchor headline - exactly the single-environment behavior this
    /// module shipped with, preserved for existing callers.
    pub anchor: &'a [Task],
    pub verifier: &'a dyn Verifier,
    /// Must be greedy ([`model::serve::SampleParams::greedy`]): both arms are
    /// scored by the same deterministic decode, or the comparison is noise.
    pub rollout: &'a RolloutParams,
    pub gate_cfg: &'a GateConfig,
}

/// Where [`cycle`] writes, and what it stamps on what it writes.
pub struct CycleArtifacts<'a> {
    /// The full trained checkpoint, written whether or not the gate promotes
    /// (a rejected candidate is left on disk for inspection).
    pub train_out: &'a Path,
    /// Directory the versioned adapter lands in, on promotion only.
    pub adapter_out_dir: &'a Path,
    pub adapter: AdapterMeta<'a>,
    pub provenance: ProvenanceInput,
}

/// Everything [`cycle`] needs to save a promoted candidate as a versioned,
/// loadable LoRA adapter via [`model::lora::device_adapter::save_adapter`] -
/// architecture-specific bits the CALLER supplies, so this module stays
/// generic over `M: Model` rather than hardcoding qwen3's `LoraCfg`.
///
/// `Copy`: a multi-cycle caller ([`crate::continual`]) hands the SAME adapter
/// description to every cycle, and a description that could not be handed out
/// twice would force either a clone-per-cycle dance or a per-cycle
/// reconstruction that could drift.
#[derive(Clone, Copy)]
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

/// Load `path` as an `M`, greedily decode every task's prompt, score each
/// completion with `verifier`, and return `(per-task scores, mean completion
/// entropy)`. Both arms of [`cycle`]'s gate call this on disk-reloaded
/// checkpoints - never the freshly-trained in-memory instance (the
/// `lora_learning_gate` lesson [`crate::gate`] itself documents: what is
/// actually served must be what is scored).
///
/// Public so a multi-cycle caller can score a checkpoint the gate never saw
/// (a control arm, an untrained baseline) through exactly the same decode
/// path the gate uses - the alternative, a second scoring routine, is how
/// two arms of one comparison quietly stop being comparable.
pub fn score_checkpoint<M: Model>(path: &Path, tasks: &[Task], verifier: &dyn Verifier, rollout: &RolloutParams) -> (Vec<f64>, f64) {
    let (scores, entropy, _) = decode_checkpoint::<M>(path, tasks, verifier, rollout);
    (scores, entropy)
}

/// [`score_checkpoint`] plus the decoded completions themselves - what the
/// distinct-completion collapse check ([`ArmScores::distinct_completions`])
/// is computed from, without a second decode pass.
pub fn decode_checkpoint<M: Model>(path: &Path, tasks: &[Task], verifier: &dyn Verifier, rollout: &RolloutParams) -> (Vec<f64>, f64, Vec<Vec<u32>>) {
    let c = checkpoint::load(path.to_str().expect("utf-8 path"));
    let cfg = M::Config::from_json(&c.header["config"]);
    let init = c.by_role("");
    let model = M::new(cfg.clone(), 1, cfg.block_size(), &init);
    let mut roll = ModelRollout::new(&model);
    let mut rng = Rng::new(0); // greedy: rollout::RolloutParams::sample must be SampleParams::greedy()
    let mut scores = Vec::with_capacity(tasks.len());
    let mut entropies = Vec::with_capacity(tasks.len());
    let mut completions = Vec::with_capacity(tasks.len());
    for task in tasks {
        let completion = roll.sample_n(&task.prompt, 1, rollout, &mut rng).pop().expect("sample_n(1) returns exactly one completion");
        scores.push(verifier.verify(task, &[], &completion.tokens).value as f64);
        entropies.push(mean_completion_entropy(&model, &task.prompt, &completion.tokens));
        completions.push(completion.tokens);
    }
    let mean_entropy = if entropies.is_empty() { 0.0 } else { entropies.iter().sum::<f64>() / entropies.len() as f64 };
    (scores, mean_entropy, completions)
}

/// Score both arms (both decoded greedily from disk-reloaded checkpoints,
/// the same `held_out ++ anchor` task list in the same order, ONE decode pass
/// per arm - see [`score_checkpoint`]) and run [`gate::gate`] over the
/// result.
///
/// The anchor-suite headline ([`GateInput::anchor_candidate`]/
/// `anchor_incumbent`) is the mean over [`Evaluation::anchor`] when that
/// suite is non-empty - which is what makes [`crate::gate::Cause::
/// AnchorRegressed`] load-bearing rather than decorative - and otherwise the
/// held-out set's own mean, byte-for-byte the single-environment behavior
/// this module shipped with.
fn score_and_gate<M: Model>(incumbent_path: &Path, candidate_path: &Path, eval: &Evaluation) -> (GateReport, ArmScores, ArmScores) {
    let all: Vec<Task> = eval.held_out.iter().chain(eval.anchor.iter()).cloned().collect();
    let n_held = eval.held_out.len();
    let split = |(scores, entropy, completions): (Vec<f64>, f64, Vec<Vec<u32>>)| -> ArmScores {
        let distinct: HashSet<Vec<u32>> = completions.into_iter().collect();
        ArmScores { held_out: scores[..n_held].to_vec(), anchor: scores[n_held..].to_vec(), mean_entropy: entropy, distinct_completions: distinct.len() }
    };
    let incumbent = split(decode_checkpoint::<M>(incumbent_path, &all, eval.verifier, eval.rollout));
    let candidate = split(decode_checkpoint::<M>(candidate_path, &all, eval.verifier, eval.rollout));

    let headline = |a: &ArmScores| if a.anchor.is_empty() { mean(&a.held_out) } else { mean(&a.anchor) };
    let input = GateInput {
        candidate_scores: &candidate.held_out,
        incumbent_scores: &incumbent.held_out,
        anchor_candidate: headline(&candidate),
        anchor_incumbent: headline(&incumbent),
        entropy_candidate: candidate.mean_entropy,
        entropy_incumbent: incumbent.mean_entropy,
    };
    (gate(&input, eval.gate_cfg), candidate, incumbent)
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

/// The `(rows, block)` to build the training model at, taken from the
/// objective itself ([`Objective::batch_shape`]) and defaulting to the one-row
/// shape every rollout-driven objective uploads - the value this used to
/// hardcode, so a GRPO-shaped caller constructs exactly the model it always
/// did.
///
/// `block` is asserted equal to the checkpoint's own, because it is not free
/// to differ: the checkpoint's positional tables are sized for it, and a
/// mismatch would be a wrong-shape forward rather than a smaller one.
pub(crate) fn objective_shape<M: Model, O: Objective<M>>(objective: &O, cfg: &M::Config) -> (u32, u32) {
    match objective.batch_shape() {
        None => (1, cfg.block_size()),
        Some((rows, block)) => {
            assert!(rows >= 1, "rl::improve: an objective declared a zero-row batch shape");
            assert_eq!(
                block,
                cfg.block_size(),
                "rl::improve: the objective uploads {block}-token rows but the checkpoint's block_size is {} - \
                 the model's token buffer is sized b*t, so this would forward over stale or overrun rows rather than fail",
                cfg.block_size()
            );
            (rows, block)
        }
    }
}

/// One `rl::improve` cycle: train `objective` (already configured over
/// whichever `Environment`/`Verifier` and P12-P15 objective the caller
/// chose) starting from `base_checkpoint`'s weights, gate the result against
/// that same incumbent on `eval`, and - only on [`Decision::Promote`] -
/// save a versioned, lineage-stamped LoRA adapter. A rejected candidate's
/// full training checkpoint (`train_out`) is left on disk for inspection but
/// no adapter is produced - the incumbent (`base_checkpoint`) is what stays
/// servable.
pub fn cycle<M: Model, O: Objective<M>>(
    base_checkpoint: &Path,
    objective: O,
    eval: &Evaluation,
    opts: &FitOpts,
    artifacts: CycleArtifacts,
) -> std::io::Result<CycleOutcome> {
    let CycleArtifacts { train_out, adapter_out_dir, adapter, provenance } = artifacts;
    let base = checkpoint::load(base_checkpoint.to_str().expect("utf-8 path"));
    let base_cfg = M::Config::from_json(&base.header["config"]);
    let init = base.by_role("");
    let (rows, block) = objective_shape(&objective, &base_cfg);
    let model = M::new(base_cfg.clone(), rows, block, &init);

    model::fit_with(model, objective, opts, Some(train_out))?;

    let (report, candidate, incumbent) = score_and_gate::<M>(base_checkpoint, train_out, eval);

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

    Ok(CycleOutcome { decision: report.decision, report, adapter_path, candidate, incumbent })
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
