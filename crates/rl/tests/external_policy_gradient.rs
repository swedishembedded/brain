// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models from the rollouts an agent collected, for its
// clients. If your team needs expertise in reinforcement learning from externally collected agent trajectories then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Policy-gradient training on externally collected rollouts ([`rl::objective::external`]), on a tiny
//! `qwen35::Qwen35` with a rank-3 LoRA adapter: the direction an advantage pushes an action's probability, the clip,
//! the KL term against its definition, the on-policy start, and that a record without an advantage trains as the
//! supervised fine-tune does.

use std::path::Path;

use data::chat::EncodedSplit;
use data::rng::Rng;
use model::training;
use model::{fit_controlled, Batch, FitControl, FitOpts, Model, Objective, StepVerdict};
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};
use rl::objective::external::{ExternalPolicyGradient, PolicyGradientConfig, ReferencePolicy};
use rl::objective::grpo::LossAgg;
use rl::objective::grpo_term::k3_kl;

const T: usize = 8;

/// One sequence of the dataset: its tokens, where its supervised answer starts, and what its rollout recorded.
struct Row {
    ids: Vec<u32>,
    answer_from: usize,
    advantage: Option<f32>,
    /// One value per supervised token.
    behavior: Option<Vec<f32>>,
}

impl Row {
    fn new(ids: &[u32], answer_from: usize, advantage: Option<f32>) -> Row {
        Row { ids: ids.to_vec(), answer_from, advantage, behavior: None }
    }
}

fn write_dataset(dir: &Path, rows: &[Row]) {
    let mut split = EncodedSplit::default();
    for row in rows {
        let mask: Vec<bool> = (0..row.ids.len()).map(|i| i >= row.answer_from).collect();
        let mut behavior = vec![f32::NAN; row.ids.len()];
        if let Some(b) = &row.behavior {
            assert_eq!(b.len(), row.ids.len() - row.answer_from);
            behavior[row.answer_from..].copy_from_slice(b);
        }
        split.push_rollout(&data::chat::EncodedSample { ids: row.ids.clone(), mask, behavior }, &[], row.advantage);
    }
    data::chat::write_split(dir, "train", &split).unwrap();
    data::chat::write_split(dir, "val", &split).unwrap();
    std::fs::write(dir.join("meta.json"), data::binio::Meta::vocab_only(29)).unwrap();
}

fn lora_model(seed: u64) -> Qwen35 {
    let cfg = Qwen35Config { lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny() };
    let init = qwen35::init::init_weights(&cfg, seed);
    let m = Qwen35::new_train_on(gpu_core::Gpu::new(pipelines()), cfg, 1, T as u32, &init);
    // Factors large enough that the adapter moves the tiny model's distribution measurably off its base.
    let mut rng = Rng::new(seed);
    for name in m.param_names().into_iter().filter(|n| n.contains(".lora_")) {
        let w: Vec<f32> = (0..m.read_weight(&name).len()).map(|_| 1.5 * (rng.next_f32() - 0.5)).collect();
        m.write_weight(&name, &w);
    }
    m
}

fn opts(steps: u32, grad_accum: u32, lr: f32) -> FitOpts {
    FitOpts { steps, batch_size: 1, block_size: T as u32, lr, grad_accum, seed: 17, ..Default::default() }
}

/// The model's log-probability of each supervised token of `row`.
fn answer_logprobs(m: &Qwen35, row: &Row) -> Vec<f32> {
    let (mut tokens, mut targets) = (vec![0u32; T], vec![model::IGNORE; T]);
    tokens[..row.ids.len()].copy_from_slice(&row.ids);
    let (from, to) = (row.answer_from - 1, row.ids.len() - 1);
    targets[from..to].copy_from_slice(&row.ids[from + 1..to + 1]);
    Model::set_batch(m, Batch::Lm { tokens: &tokens, targets: &targets });
    let _ = m.forward();
    let lp = training::objectives(m).batch_token_logprobs();
    (row.answer_from - 1..row.ids.len() - 1).map(|t| lp[t]).collect()
}

fn grad_l1(m: &Qwen35) -> f32 {
    m.param_names().iter().map(|n| m.read_grad(n).iter().map(|g| g.abs()).sum::<f32>()).sum()
}

/// An objective opened on `rows`, prepared on `m`, positioned at the first draw: the state `fit_controlled` hands
/// the first step.
fn started(rows: &[Row], cfg: PolicyGradientConfig, m: &mut Qwen35) -> (ExternalPolicyGradient, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    write_dataset(dir.path(), rows);
    let mut obj = ExternalPolicyGradient::open(dir.path(), &opts(1, 1, 1e-2), cfg).unwrap();
    Objective::<Qwen35>::prepare(&mut obj, m).unwrap();
    Objective::<Qwen35>::begin(&mut obj, 17, 0);
    (obj, dir)
}

fn one_step(obj: &mut ExternalPolicyGradient, m: &Qwen35) -> f32 {
    m.zero_grads();
    let loss = Objective::<Qwen35>::micro_steps(obj, m, &mut Rng::new(1), 1);
    m.poll_wait();
    assert!(matches!(Objective::<Qwen35>::step_verdict(obj), StepVerdict::Apply));
    loss
}

fn no_kl() -> PolicyGradientConfig {
    PolicyGradientConfig { kl_beta: 0.0, log_kl: false, ..Default::default() }
}

#[test]
fn a_positive_advantage_raises_and_a_negative_one_lowers_the_probability_of_the_action() {
    let seed = data::rng::random_seed();
    println!("seed {seed}");
    let mut m = lora_model(seed);
    let (good, bad) = (Row::new(&[1, 2, 3, 4, 5, 6], 3, Some(1.0)), Row::new(&[7, 8, 9, 10, 11, 12], 3, Some(-1.0)));
    let (good_before, bad_before) = (answer_logprobs(&m, &good), answer_logprobs(&m, &bad));
    training::require_token_objectives(&mut m).unwrap().enable_weighted_loss();

    let dir = tempfile::tempdir().unwrap();
    write_dataset(dir.path(), &[Row { ..Row::new(&good.ids, 3, Some(1.0)) }, Row::new(&bad.ids, 3, Some(-1.0))]);
    let run = opts(6, 2, 2e-2);
    let mut obj = ExternalPolicyGradient::open(dir.path(), &run, PolicyGradientConfig::default()).unwrap();
    let (report, m) = fit_controlled(m, &mut obj, &run, None, FitControl::default()).unwrap();
    assert!(!report.interrupted);

    let (good_after, bad_after) = (answer_logprobs(&m, &good), answer_logprobs(&m, &bad));
    let delta = |after: &[f32], before: &[f32]| after.iter().sum::<f32>() - before.iter().sum::<f32>();
    let (up, down) = (delta(&good_after, &good_before), delta(&bad_after, &bad_before));
    println!("log-prob of the rewarded action {up:+.4}, of the penalised action {down:+.4}");
    assert!(up > 1e-3, "the rewarded action became no more likely: {up}");
    assert!(down < -1e-3, "the penalised action became no less likely: {down}");

    let stats = obj.stats();
    assert_eq!(stats.steps.len(), 6);
    assert_eq!(stats.steps[0].sequences, 2);
    assert!((stats.steps[0].ratio_mean - 1.0).abs() < 1e-4, "the first step is on-policy: {:?}", stats.steps[0]);
    assert!(stats.steps.iter().all(|s| s.loss.is_finite() && s.kl.is_some_and(f32::is_finite)));
}

#[test]
fn a_ratio_outside_the_band_gives_no_gradient_in_the_clipped_direction_only() {
    let mut m = lora_model(data::rng::random_seed());
    let ids = [1, 2, 3, 4, 5, 6];
    let now = answer_logprobs(&m, &Row::new(&ids, 3, None));
    // The sampler gave each token `shift` less (more) log-probability than the policy gives now: the ratio is e^shift.
    for (shift, advantage, clipped) in [(1.0f32, 1.0f32, true), (1.0, -1.0, false), (-1.0, 1.0, false), (-1.0, -1.0, true)] {
        let mut row = Row::new(&ids, 3, Some(advantage));
        row.behavior = Some(now.iter().map(|lp| (lp - shift).min(0.0)).collect());
        // A behaviour log-probability is at most zero; keep the ratio outside the band whatever the clamp did.
        let ratio: Vec<f32> = now.iter().zip(row.behavior.as_ref().unwrap()).map(|(n, b)| (n - b).exp()).collect();
        assert!(ratio.iter().all(|r| (*r > 1.25) == (shift > 0.0) && (*r < 0.8) == (shift < 0.0)), "{ratio:?}");
        let (mut obj, _dir) = started(&[row], no_kl(), &mut m);
        one_step(&mut obj, &m);
        let step = &obj.stats().steps[0];
        if clipped {
            assert_eq!(grad_l1(&m), 0.0, "shift {shift}, advantage {advantage}: a clipped token carries no gradient");
            assert_eq!(step.clip_fraction, 1.0);
        } else {
            assert!(grad_l1(&m) > 0.0, "shift {shift}, advantage {advantage}: the unclipped side still learns");
            assert_eq!(step.clip_fraction, 0.0);
        }
    }
}

#[test]
fn the_kl_term_is_the_k3_estimator_against_the_reference_it_names() {
    let mut m = lora_model(data::rng::random_seed());
    let row = Row::new(&[1, 2, 3, 4, 5, 6], 3, Some(0.0));
    let policy = answer_logprobs(&m, &row);
    // The base: every adapter's B factor zeroed.
    let b_names: Vec<String> = m.param_names().into_iter().filter(|n| n.ends_with(".lora_b")).collect();
    let saved: Vec<Vec<f32>> = b_names.iter().map(|n| m.read_weight(n)).collect();
    b_names.iter().zip(&saved).for_each(|(n, w)| m.write_weight(n, &vec![0.0; w.len()]));
    let base = answer_logprobs(&m, &row);
    b_names.iter().zip(&saved).for_each(|(n, w)| m.write_weight(n, w));
    let want: f32 = base.iter().zip(&policy).map(|(r, p)| k3_kl(*r, *p)).sum::<f32>() / policy.len() as f32;
    assert!(want > 1e-4, "the fixture's adapter must move the policy off the base: {want}");

    let beta = 0.5;
    let cfg = PolicyGradientConfig { kl_beta: beta, reference: ReferencePolicy::Base, ..Default::default() };
    let (mut obj, _dir) = started(std::slice::from_ref(&row), cfg, &mut m);
    let loss = one_step(&mut obj, &m);
    let step = &obj.stats().steps[0];
    let kl = step.kl.expect("a measured reference");
    assert!((kl - want).abs() < 1e-4 * want.max(1.0), "reported {kl}, by definition {want}");
    // A zero advantage leaves only the KL term in the loss: beta * mean k3.
    assert!((loss - beta * want).abs() < 1e-4, "loss {loss}, beta * KL {}", beta * want);
    assert!((step.loss - beta * want).abs() < 1e-4);
    assert!(grad_l1(&m) > 0.0, "the KL term pulls the policy toward the base");

    // Against the policy the run started from, the first step has no KL at all.
    let cfg = PolicyGradientConfig { kl_beta: beta, reference: ReferencePolicy::StartOfRun, ..Default::default() };
    let (mut obj, _dir) = started(std::slice::from_ref(&row), cfg, &mut m);
    one_step(&mut obj, &m);
    assert!(obj.stats().steps[0].kl.unwrap().abs() < 1e-6);
    assert!(grad_l1(&m) < 1e-5, "the policy sits on its own reference, so nothing pulls it");
}

#[test]
fn without_recorded_behaviour_the_first_step_is_the_on_policy_estimator() {
    let mut m = lora_model(data::rng::random_seed());
    let row = Row::new(&[1, 2, 3, 4, 5, 6], 3, Some(2.0));
    let (mut obj, _dir) = started(std::slice::from_ref(&row), no_kl(), &mut m);
    assert_eq!(obj.stats().tokens_start_of_run_behavior, 3);
    assert_eq!(obj.stats().tokens_recorded_behavior, 0);
    let loss = one_step(&mut obj, &m);
    let step = &obj.stats().steps[0];
    assert!((step.ratio_min - 1.0).abs() < 1e-5 && (step.ratio_max - 1.0).abs() < 1e-5, "{step:?}");
    assert_eq!(step.clip_fraction, 0.0);
    // r = 1: the loss is -A * mean log-prob... of the ratio, i.e. -A per token.
    assert!((loss + 2.0).abs() < 1e-4, "{loss}");

    // The gradient is that of A * (-mean log-prob): the supervised fine-tune's gradient scaled by A.
    let policy_grad = m.param_names().iter().map(|n| m.read_grad(n)).collect::<Vec<_>>();
    m.zero_grads();
    let plain = Row::new(&row.ids, 3, None);
    let (mut sft, _dir2) = started(std::slice::from_ref(&plain), no_kl(), &mut m);
    one_step(&mut sft, &m);
    let sft_grad = m.param_names().iter().map(|n| m.read_grad(n)).collect::<Vec<_>>();
    // The on-policy gradient is -A * grad(log p) averaged over tokens; the cross-entropy's is -grad(log p) averaged.
    let (mut worst, mut scale) = (0.0f32, 0.0f32);
    for (a, b) in policy_grad.iter().zip(&sft_grad) {
        for (x, y) in a.iter().zip(b) {
            worst = worst.max((x - 2.0 * y).abs());
            scale = scale.max(x.abs());
        }
    }
    println!("on-policy gradient against A times the cross-entropy gradient: worst {worst:e} of scale {scale:e}");
    assert!(worst <= 1e-4 * scale.max(1e-6), "worst {worst} of scale {scale}");
}

#[test]
fn a_record_without_an_advantage_trains_as_the_supervised_fine_tune_does() {
    let mut m = lora_model(data::rng::random_seed());
    let row = Row::new(&[1, 2, 3, 4, 5, 6], 3, None);
    let plain_loss = {
        let (tokens, targets) = {
            let mut tokens = vec![0u32; T];
            tokens[..6].copy_from_slice(&row.ids);
            let mut targets = vec![model::IGNORE; T];
            (2..5).for_each(|t| targets[t] = row.ids[t + 1]);
            (tokens, targets)
        };
        Model::set_batch(&m, Batch::Lm { tokens: &tokens, targets: &targets });
        m.zero_grads();
        let loss = m.forward();
        m.backward();
        m.poll_wait();
        loss
    };
    let sft_grad: Vec<Vec<f32>> = m.param_names().iter().map(|n| m.read_grad(n)).collect();

    let (mut obj, _dir) = started(std::slice::from_ref(&row), PolicyGradientConfig::default(), &mut m);
    assert_eq!((obj.stats().sequences_imitation, obj.stats().sequences_with_advantage), (1, 0));
    let loss = one_step(&mut obj, &m);
    assert!((loss - plain_loss).abs() < 1e-5 * plain_loss.abs().max(1.0), "loss {loss} against the supervised {plain_loss}");
    for (name, want) in m.param_names().iter().zip(&sft_grad) {
        let got = m.read_grad(name);
        let scale = want.iter().fold(0.0f32, |a, w| a.max(w.abs())).max(1e-6);
        assert!(got.iter().zip(want).all(|(g, w)| (g - w).abs() <= 1e-5 * scale), "{name}");
    }
    assert!(obj.stats().steps[0].kl.is_none(), "an imitation token has no reference to diverge from");
}

#[test]
fn a_mistaken_configuration_is_refused_before_anything_trains() {
    let dir = tempfile::tempdir().unwrap();
    write_dataset(dir.path(), &[Row::new(&[1, 2, 3, 4], 2, Some(1.0))]);
    let run = opts(1, 1, 1e-2);
    for bad in [PolicyGradientConfig { clip_eps: 0.0, ..Default::default() }, PolicyGradientConfig { clip_eps: 1.5, ..Default::default() }, PolicyGradientConfig { kl_beta: -0.1, ..Default::default() }, PolicyGradientConfig { temperature: 0.0, ..Default::default() }] {
        let err = ExternalPolicyGradient::open(dir.path(), &run, bad).err().expect("refused");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput, "{err}");
    }
    let wide = FitOpts { batch_size: 2, ..run.clone() };
    let err = ExternalPolicyGradient::open(dir.path(), &wide, PolicyGradientConfig::default()).err().expect("a sequence mean needs one sequence per batch");
    assert!(err.to_string().contains("TokenSum"), "{err}");
    assert!(ExternalPolicyGradient::open(dir.path(), &wide, PolicyGradientConfig { loss_agg: LossAgg::TokenSum, ..Default::default() }).is_ok());
}

#[test]
fn a_supervised_trainer_refuses_a_dataset_that_carries_advantages() {
    let dir = tempfile::tempdir().unwrap();
    write_dataset(dir.path(), &[Row::new(&[1, 2, 3, 4], 2, Some(1.0))]);
    let err = model::load_dataset_with_images(dir.path(), &opts(1, 1, 1e-2)).err().expect("refused");
    assert!(err.to_string().contains("advantages") && err.to_string().contains("policy_gradient"), "{err}");
    assert!(model::load_rollout_dataset(dir.path(), &opts(1, 1, 1e-2)).is_ok());
}
