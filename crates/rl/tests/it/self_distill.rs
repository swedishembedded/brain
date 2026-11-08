// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for dense per-token credit on a language model's own rollouts, for its
// clients. If your team needs expertise in on-policy self-distillation with privileged information then you can
// procure our services by sending an email to info@swedishembedded.com.

//! On-policy self-distillation with privileged information ([`rl::objective::external`]): a record's teacher is the
//! frozen model shown extra context, and each sampled token carries `A_t = log q_teacher - log p_old`, trained through
//! the one clipped policy-gradient step. On a tiny `qwen35` LoRA model: a token the teacher prefers is raised, one it
//! rejects lowered, an identical view gives zero advantage, the collapse guards bound what a step sees, and a dataset
//! whose teacher cannot be scored is refused by name.

use std::path::Path;

use data::chat::{EncodedSample, EncodedSplit};
use data::rng::Rng;
use model::training;
use model::{fit_controlled, Batch, FitControl, FitOpts, Model};
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};
use rl::objective::external::{BehaviourSource, ExternalPolicyGradient, PolicyGradientConfig, ReferencePolicy};
use rl::objective::pg_batch::with_adapters_disabled;

const T: usize = 12;
const ANSWER_FROM: usize = 3;

struct Row {
    ids: Vec<u32>,
    /// The teacher's tokens: the student's with a privileged prefix, or `None` for an imitation-free scalar record.
    teacher: Option<Vec<u32>>,
}

fn write_dataset(dir: &Path, rows: &[Row]) {
    let mut split = EncodedSplit::default();
    for row in rows {
        let mask: Vec<bool> = (0..row.ids.len()).map(|i| i >= ANSWER_FROM).collect();
        let teacher = row.teacher.as_ref().map(|ids| {
            let lead = ids.len() - row.ids.len() + ANSWER_FROM;
            (ids.clone(), (0..ids.len()).map(|i| i >= lead).collect::<Vec<bool>>())
        });
        split.push_rollout(&EncodedSample { ids: row.ids.clone(), mask, behavior: vec![f32::NAN; row.ids.len()], teacher }, &[], Some(0.0));
    }
    data::chat::write_split(dir, "train", &split).unwrap();
    data::chat::write_split(dir, "val", &split).unwrap();
    std::fs::write(dir.join("meta.json"), data::binio::Meta::vocab_only(29)).unwrap();
}

fn lora_model(seed: u64) -> Qwen35 {
    let cfg = Qwen35Config { lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny() };
    let init = qwen35::init::init_weights(&cfg, seed);
    let m = Qwen35::new_train_on(gpu_core::Gpu::new(pipelines()), cfg, 1, T as u32, &init);
    let mut rng = Rng::new(seed);
    for name in m.param_names().into_iter().filter(|n| n.contains(".lora_")) {
        let w: Vec<f32> = (0..m.read_weight(&name).len()).map(|_| 1.5 * (rng.next_f32() - 0.5)).collect();
        m.write_weight(&name, &w);
    }
    m
}

fn opts(steps: u32, lr: f32) -> FitOpts {
    FitOpts { steps, batch_size: 1, block_size: T as u32, lr, grad_accum: 1, seed: 17, min_lr: 0.0, ..Default::default() }
}

fn config() -> PolicyGradientConfig {
    PolicyGradientConfig { kl_beta: 0.0, log_kl: false, clip_eps: 0.5, behaviour: BehaviourSource::StartOfRunProxy, ..Default::default() }
}

/// The model's log-probability of each supervised token of `ids` (answer from [`ANSWER_FROM`]).
fn answer_logprobs(m: &Qwen35, ids: &[u32]) -> Vec<f32> {
    let (mut tokens, mut targets) = (vec![0u32; T], vec![model::IGNORE; T]);
    tokens[..ids.len()].copy_from_slice(ids);
    let (from, to) = (ANSWER_FROM - 1, ids.len() - 1);
    targets[from..to].copy_from_slice(&ids[from + 1..to + 1]);
    Model::set_batch(m, Batch::Lm { tokens: &tokens, targets: &targets });
    let _ = m.forward();
    let lp = training::objectives(m).batch_token_logprobs();
    (from..to).map(|t| lp[t]).collect()
}

fn objective_on(rows: &[Row], cfg: PolicyGradientConfig, run: &FitOpts) -> (std::io::Result<ExternalPolicyGradient>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    write_dataset(dir.path(), rows);
    (ExternalPolicyGradient::open(dir.path(), run, cfg), dir)
}

fn same_view() -> Vec<Row> {
    let ids = vec![1, 2, 3, 4, 5, 6];
    vec![Row { teacher: Some(ids.clone()), ids }]
}

/// The first optimiser step's statistics of a run of one step.
fn first_step(rows: &[Row], cfg: PolicyGradientConfig, m: Qwen35) -> (rl::objective::external::PolicyGradientStep, Qwen35) {
    let run = opts(1, 0.0);
    let (obj, _dir) = objective_on(rows, cfg, &run);
    let mut obj = obj.unwrap();
    let (_, m) = fit_controlled(m, &mut obj, &run, None, FitControl::default()).unwrap();
    (obj.stats().steps[0].clone(), m)
}

#[test]
fn an_identical_view_gives_every_token_zero_advantage() {
    let m = lora_model(3);
    let (step, _) = first_step(&same_view(), PolicyGradientConfig { teacher: ReferencePolicy::StartOfRun, ..config() }, m);
    assert_eq!(step.token_advantage_max, 0.0, "{step:?}");
    assert_eq!(step.token_advantage_mean, 0.0);
    assert_eq!(step.token_advantage_clip_fraction, 0.0);
}

#[test]
fn a_token_the_teacher_prefers_is_raised_and_one_it_rejects_is_lowered() {
    let ids = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
    let m = lora_model(5);
    let student = answer_logprobs(&m, &ids);
    let base = with_adapters_disabled(&m, || answer_logprobs(&m, &ids)).unwrap();
    let advantage: Vec<f32> = base.iter().zip(&student).map(|(b, s)| b - s).collect();
    assert!(advantage.iter().any(|a| *a > 0.02) && advantage.iter().any(|a| *a < -0.02), "the fixture needs tokens of both signs: {advantage:?}");

    let rows = [Row { teacher: Some(ids.clone()), ids: ids.clone() }];
    let run = opts(8, 2e-2);
    let (obj, _dir) = objective_on(&rows, config(), &run);
    let mut obj = obj.unwrap();
    let (_, m) = fit_controlled(m, &mut obj, &run, None, FitControl::default()).unwrap();
    let first = &obj.stats().steps[0];
    let mean = advantage.iter().sum::<f32>() / advantage.len() as f32;
    assert!((first.token_advantage_mean - mean).abs() < 1e-4, "step reports {} for a hand-computed {mean}", first.token_advantage_mean);
    assert!((first.token_advantage_max - advantage.iter().fold(0.0f32, |a, x| a.max(x.abs()))).abs() < 1e-4);

    let after = answer_logprobs(&m, &ids);
    let moved = |sign: f32| -> f32 { (0..advantage.len()).filter(|&t| advantage[t] * sign > 0.0).map(|t| after[t] - student[t]).sum() };
    println!("raised by {:+.4} where the teacher prefers, {:+.4} where it rejects", moved(1.0), moved(-1.0));
    assert!(moved(1.0) > 1e-3, "tokens the teacher prefers did not rise");
    assert!(moved(-1.0) < -1e-3, "tokens the teacher rejects did not fall");
}

#[test]
fn a_privileged_prefix_changes_the_teacher_and_so_the_advantage() {
    // The same answer after a different (privileged) prefix: the teacher scores it differently from the student.
    let student = vec![1u32, 2, 3, 4, 5, 6];
    let teacher = vec![9u32, 10, 11, 1, 2, 3, 4, 5, 6];
    let (step, _) = first_step(&[Row { ids: student, teacher: Some(teacher) }], PolicyGradientConfig { teacher: ReferencePolicy::StartOfRun, ..config() }, lora_model(7));
    assert!(step.token_advantage_max > 1e-3, "a different context must change the teacher's view: {step:?}");
}

#[test]
fn the_advantage_bound_and_the_sequence_cap_limit_what_a_step_sees() {
    let ids = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
    let unbounded = first_step(&[Row { teacher: Some(ids.clone()), ids: ids.clone() }], config(), lora_model(5)).0;
    assert!(unbounded.token_advantage_max > 0.05 && unbounded.token_advantage_clip_fraction == 0.0);

    let bound = unbounded.token_advantage_max / 2.0;
    let bounded = first_step(&[Row { teacher: Some(ids.clone()), ids: ids.clone() }], PolicyGradientConfig { advantage_bound: Some(bound), ..config() }, lora_model(5)).0;
    assert!(bounded.token_advantage_max <= bound + 1e-6, "{bounded:?}");
    assert!(bounded.token_advantage_clip_fraction > 0.0, "{bounded:?}");

    // Five supervised tokens, so a cap of one allows at most 1 of summed magnitude over the sequence.
    let cap = 0.1;
    let capped = first_step(&[Row { teacher: Some(ids.clone()), ids }], PolicyGradientConfig { sequence_advantage_cap: Some(cap), ..config() }, lora_model(5)).0;
    assert!(capped.token_advantage_max <= cap + 1e-6 && capped.token_advantage_max < unbounded.token_advantage_max, "{capped:?}");
}

#[test]
fn a_teacher_that_cannot_be_scored_is_refused_by_name() {
    let run = opts(1, 0.0);
    let ids = vec![1u32, 2, 3, 4, 5, 6];
    let too_long = Row { ids: ids.clone(), teacher: Some((0..T as u32 + 3).map(|i| i % 20).collect()) };
    let (why, _d) = objective_on(&[too_long], config(), &run);
    let why = why.err().expect("a teacher longer than the row is refused").to_string();
    assert!(why.contains("teacher") && why.contains("example 0") && why.contains(&format!("{}", T + 3)), "{why}");

    // Its supervised tokens do not number the student's.
    let mut split = EncodedSplit::default();
    let mask: Vec<bool> = (0..ids.len()).map(|i| i >= ANSWER_FROM).collect();
    let teacher = (ids.clone(), vec![true; ids.len()]);
    split.push_rollout(&EncodedSample { ids: ids.clone(), mask, behavior: vec![f32::NAN; ids.len()], teacher: Some(teacher) }, &[], Some(0.0));
    let dir = tempfile::tempdir().unwrap();
    data::chat::write_split(dir.path(), "train", &split).unwrap();
    data::chat::write_split(dir.path(), "val", &split).unwrap();
    std::fs::write(dir.path().join("meta.json"), data::binio::Meta::vocab_only(29)).unwrap();
    let why = ExternalPolicyGradient::open(dir.path(), &run, config()).err().expect("refused").to_string();
    assert!(why.contains("teacher") && why.contains("example 0") && why.contains("supervises"), "{why}");
}

#[test]
fn collapse_guards_with_no_meaning_are_refused() {
    for bad in [PolicyGradientConfig { advantage_bound: Some(0.0), ..config() }, PolicyGradientConfig { advantage_bound: Some(f32::NAN), ..config() }, PolicyGradientConfig { sequence_advantage_cap: Some(-1.0), ..config() }] {
        assert!(bad.validate().is_err(), "{bad:?}");
    }
    assert!(PolicyGradientConfig { advantage_bound: Some(5.0), sequence_advantage_cap: Some(20.0), ..config() }.validate().is_ok());
}
