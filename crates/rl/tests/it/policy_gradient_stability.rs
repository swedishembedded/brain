// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for keeping reinforcement-learning runs on real models from diverging on one
// bad step, for its clients. If your team needs expertise in stable policy-gradient training of language models then you
// can procure our services by sending an email to info@swedishembedded.com.

//! What keeps a policy-gradient run stable and readable ([`rl::objective::external`]): the asymmetric clip band
//! ("clip-higher") and the clip fraction per edge, the engine-versus-trainer importance correction (truncated and
//! masked), the per-step entropy estimate and response length, and the step guard that skips, and logs, an update whose
//! KL or gradient norm is over its bound.

use model::{Objective, StepVerdict};
use qwen35::model::Qwen35;
use rl::objective::external::{BehaviourSource, ExternalPolicyGradient, ImportanceCorrection, PolicyGradientConfig, MAX_CONSECUTIVE_GUARD_SKIPS};

use super::external_policy_gradient::{answer_logprobs, grad_l1, lora_model, no_kl, one_step, started, Row};

const IDS: [u32; 6] = [1, 2, 3, 4, 5, 6];

/// A row whose recorded behaviour is `shift` below the policy's own log-probability of each answer token, so every
/// ratio against the record is `e^shift`.
fn shifted(m: &Qwen35, advantage: f32, shift: f32) -> Row {
    let now = answer_logprobs(m, &Row::new(&IDS, 3, None));
    Row { behavior: Some(now.iter().map(|lp| (lp - shift).min(0.0)).collect()), ..Row::new(&IDS, 3, Some(advantage)) }
}

fn grads(m: &Qwen35) -> Vec<Vec<f32>> {
    m.param_names().iter().map(|n| m.read_grad(n)).collect()
}

/// Clip-higher: a token the symmetric band clips (ratio e past 1.2, positive advantage) keeps its gradient under a wider
/// upper edge, and the clip fraction is read per edge.
#[test]
fn a_wider_upper_edge_keeps_the_gradient_of_a_token_the_symmetric_band_clips() {
    let mut m = lora_model(data::rng::random_seed());
    let row = shifted(&m, 1.0, 1.0);

    let (mut obj, _dir) = started(std::slice::from_ref(&row), no_kl(), &mut m);
    one_step(&mut obj, &m);
    let step = &obj.stats().steps[0];
    assert_eq!((step.clip_fraction, step.clip_fraction_high, step.clip_fraction_low), (1.0, 1.0, 0.0), "{step:?}");
    assert_eq!(grad_l1(&m), 0.0, "clipped at the upper edge");

    m.zero_grads();
    let cfg = PolicyGradientConfig { clip_eps_high: Some(2.0), ..no_kl() };
    let (mut obj, _dir) = started(std::slice::from_ref(&row), cfg, &mut m);
    one_step(&mut obj, &m);
    let step = &obj.stats().steps[0];
    assert_eq!((step.clip_fraction, step.clip_fraction_high), (0.0, 0.0), "{step:?}");
    assert!(grad_l1(&m) > 0.0, "the band reaches past ratio e, so the token still learns");

    // The lower edge is its own: a negative advantage under 1 - eps counts as a low clip.
    let low = shifted(&m, -1.0, -1.0);
    let (mut obj, _dir) = started(std::slice::from_ref(&low), cfg, &mut m);
    m.zero_grads();
    one_step(&mut obj, &m);
    let step = &obj.stats().steps[0];
    assert_eq!((step.clip_fraction_low, step.clip_fraction_high), (1.0, 0.0), "{step:?}");
}

/// The importance correction: the ratio is centred on the trainer's own start-of-run policy, and each token's policy
/// term is scaled by the engine's gap to it, truncated at the cap or dropped past it.
#[test]
fn an_importance_correction_weighs_the_engine_gap_and_the_mask_drops_what_is_too_far() {
    let mut m = lora_model(data::rng::random_seed());
    // The engine gave every token e^-1 less log-probability than the trainer: w = e, the ratio e against the record.
    let row = shifted(&m, 1.0, 1.0);

    // Uncorrected, the step starts clipped (ratio e).
    let (mut obj, _dir) = started(std::slice::from_ref(&row), PolicyGradientConfig { behaviour: BehaviourSource::Recorded, ..no_kl() }, &mut m);
    one_step(&mut obj, &m);
    assert_eq!(obj.stats().steps[0].clip_fraction, 1.0);

    // On-policy reference: the same row against the trainer's own log-probabilities.
    m.zero_grads();
    let (mut obj, _dir) = started(&[Row::new(&IDS, 3, Some(1.0))], no_kl(), &mut m);
    one_step(&mut obj, &m);
    let on_policy = grads(&m);

    m.zero_grads();
    let truncate = PolicyGradientConfig { importance: ImportanceCorrection::Truncate { cap: 2.0 }, behaviour: BehaviourSource::Recorded, ..no_kl() };
    let (mut obj, _dir) = started(std::slice::from_ref(&row), truncate, &mut m);
    one_step(&mut obj, &m);
    let step = &obj.stats().steps[0];
    assert!((step.ratio_mean - 1.0).abs() < 1e-4 && step.clip_fraction == 0.0, "the ratio is centred on the trainer: {step:?}");
    assert!((step.importance_weight_mean - 2.0).abs() < 1e-4, "e truncated at 2: {step:?}");
    let (mut worst, mut scale) = (0.0f32, 0.0f32);
    for (g, o) in grads(&m).iter().zip(&on_policy) {
        for (x, y) in g.iter().zip(o) {
            worst = worst.max((x - 2.0 * y).abs());
            scale = scale.max(x.abs());
        }
    }
    assert!(worst <= 1e-4 * scale.max(1e-6), "the truncated gradient is twice the on-policy one: worst {worst} of {scale}");

    m.zero_grads();
    let mask = PolicyGradientConfig { importance: ImportanceCorrection::Mask { cap: 2.0 }, behaviour: BehaviourSource::Recorded, ..no_kl() };
    let (mut obj, _dir) = started(std::slice::from_ref(&row), mask, &mut m);
    one_step(&mut obj, &m);
    let step = &obj.stats().steps[0];
    assert_eq!(step.importance_masked_fraction, 1.0, "e is outside [1/2, 2]: {step:?}");
    assert_eq!(grad_l1(&m), 0.0, "a masked token carries no policy gradient");
}

#[test]
fn the_step_reports_the_entropy_estimate_the_response_length_and_the_gradient_norm() {
    let mut m = lora_model(data::rng::random_seed());
    let row = Row::new(&IDS, 3, Some(1.0));
    let nll: f32 = -answer_logprobs(&m, &row).iter().sum::<f32>() / 3.0;
    let (mut obj, _dir) = started(std::slice::from_ref(&row), no_kl(), &mut m);
    one_step(&mut obj, &m);
    let step = &obj.stats().steps[0];
    assert!((step.entropy_estimate - nll).abs() < 1e-4, "{} against {nll}", step.entropy_estimate);
    assert_eq!(step.mean_response_tokens, 3.0);
    let norm = grads(&m).iter().flatten().map(|g| f64::from(*g).powi(2)).sum::<f64>().sqrt() as f32;
    assert!((step.grad_norm - norm).abs() < 1e-3 * norm.max(1.0), "reported {}, the gradient's own {norm}", step.grad_norm);
}

/// The step guard: a step whose KL to the reference or gradient norm is over its bound is skipped with its reason, in
/// the statistics and the log line; a step under the bounds is applied; a bound that skips every step ends the run.
#[test]
fn the_step_guard_skips_a_step_over_its_kl_or_gradient_norm_bound_and_ends_a_run_it_always_skips() {
    let verdict = |cfg: PolicyGradientConfig| {
        let mut m = lora_model(7);
        let (mut obj, _dir) = started(&[Row::new(&IDS, 3, Some(1.0))], cfg, &mut m);
        m.zero_grads();
        Objective::<Qwen35>::micro_steps(&mut obj, &m, &mut data::rng::Rng::new(1), 1);
        m.poll_wait();
        let v = Objective::<Qwen35>::step_verdict(&mut obj);
        (v, obj)
    };
    let base = PolicyGradientConfig { behaviour: BehaviourSource::StartOfRunProxy, ..Default::default() };

    let (v, obj) = verdict(base);
    assert!(matches!(v, StepVerdict::Apply), "no bound, no skip");
    assert!(obj.stats().steps[0].kl.unwrap() > 1e-4, "the fixture's adapter is off the base, so there is a KL to bound");

    let (v, obj) = verdict(PolicyGradientConfig { max_step_kl: Some(1e-6), ..base });
    let StepVerdict::Skip(why) = v else { panic!("a KL over the bound is skipped") };
    assert!(why.contains("KL") && why.contains("bound"), "{why}");
    assert_eq!((obj.stats().steps_skipped_guard, obj.stats().steps[0].skipped.as_deref()), (1, Some(why.as_str())));
    assert!(Objective::<Qwen35>::step_log(&obj).unwrap().contains("SKIPPED"));

    let (v, _) = verdict(PolicyGradientConfig { max_step_kl: Some(100.0), max_step_grad_norm: Some(1e9), ..base });
    assert!(matches!(v, StepVerdict::Apply), "bounds above the step let it through");

    let (v, _) = verdict(PolicyGradientConfig { max_step_grad_norm: Some(1e-9), ..base });
    let StepVerdict::Skip(why) = v else { panic!("a gradient norm over the bound is skipped") };
    assert!(why.contains("gradient norm"), "{why}");

    // Every step over the bound: skipped, then the run ends instead of idling.
    let mut m = lora_model(7);
    let (mut obj, _dir) = started(&[Row::new(&IDS, 3, Some(1.0))], PolicyGradientConfig { max_step_grad_norm: Some(1e-9), ..base }, &mut m);
    for _ in 0..MAX_CONSECUTIVE_GUARD_SKIPS - 1 {
        m.zero_grads();
        Objective::<Qwen35>::micro_steps(&mut obj, &m, &mut data::rng::Rng::new(1), 1);
        assert!(matches!(Objective::<Qwen35>::step_verdict(&mut obj), StepVerdict::Skip(_)));
    }
    m.zero_grads();
    Objective::<Qwen35>::micro_steps(&mut obj, &m, &mut data::rng::Rng::new(1), 1);
    assert!(matches!(Objective::<Qwen35>::step_verdict(&mut obj), StepVerdict::Abort(_)));
}

#[test]
fn the_stability_settings_are_validated_by_name() {
    let dir = tempfile::tempdir().unwrap();
    super::external_policy_gradient::write_dataset(dir.path(), &[Row::new(&IDS, 3, Some(1.0))]);
    let run = super::external_policy_gradient::opts(1, 1, 1e-2);
    let proxy = PolicyGradientConfig { behaviour: BehaviourSource::StartOfRunProxy, ..Default::default() };
    for (bad, named) in [
        (PolicyGradientConfig { clip_eps_high: Some(0.1), ..proxy }, "clip_eps_high"),
        (PolicyGradientConfig { clip_eps_high: Some(f32::NAN), ..proxy }, "clip_eps_high"),
        (PolicyGradientConfig { max_step_kl: Some(0.0), ..proxy }, "max_step_kl"),
        (PolicyGradientConfig { max_step_grad_norm: Some(-1.0), ..proxy }, "max_step_grad_norm"),
        (PolicyGradientConfig { max_step_kl: Some(0.1), kl_beta: 0.0, log_kl: false, ..proxy }, "max_step_kl"),
        (PolicyGradientConfig { importance: ImportanceCorrection::Truncate { cap: 1.0 }, behaviour: BehaviourSource::Recorded, ..Default::default() }, "cap"),
        (PolicyGradientConfig { importance: ImportanceCorrection::Mask { cap: 2.0 }, ..proxy }, "recorded"),
    ] {
        let err = ExternalPolicyGradient::open(dir.path(), &run, bad).err().unwrap_or_else(|| panic!("{named} accepted"));
        assert!(err.to_string().contains(named), "{named}: {err}");
    }
}

/// A truncated rollout is masked as a whole: never drawn, so no step trains on, scores or counts it, and a dataset of
/// nothing else is refused.
#[test]
fn a_truncated_rollout_is_never_drawn_and_a_dataset_of_only_those_is_refused() {
    use super::external_policy_gradient::{opts, write_dataset};
    let mut m = lora_model(data::rng::random_seed());
    let rows = [Row { truncated: true, ..Row::new(&[7, 8, 9, 10, 11, 12], 3, Some(5.0)) }, Row::new(&IDS, 3, Some(1.0)), Row { truncated: true, ..Row::new(&[2, 3, 4, 5, 6, 7], 3, Some(-5.0)) }];
    let (mut obj, _dir) = started(&rows, no_kl(), &mut m);
    assert_eq!((obj.stats().sequences_truncated, obj.stats().sequences_with_advantage), (2, 1));
    for _ in 0..5 {
        one_step(&mut obj, &m);
    }
    let steps = &obj.stats().steps;
    assert!(steps.iter().all(|s| s.sequences == 1 && (s.mean_advantage - 1.0).abs() < 1e-6), "only the finished rollout is trained: {steps:?}");

    let dir = tempfile::tempdir().unwrap();
    write_dataset(dir.path(), &[Row { truncated: true, ..Row::new(&IDS, 3, Some(1.0)) }]);
    let err = ExternalPolicyGradient::open(dir.path(), &opts(1, 1, 1e-2), no_kl()).err().expect("nothing to train on");
    assert!(err.to_string().contains("truncated"), "{err}");
}

/// The numerics gate: with the recorded log-probabilities equal to the trainer's own, the gap is zero, the effective
/// sample size one and the gate passes; recorded values a fixed shift below them give exactly that gap, a mismatch KL
/// of the k3 definition and a broken bound that names itself. No update is ever taken.
#[test]
fn the_numerics_gate_measures_the_serve_train_gap_takes_no_step_and_names_the_bound_it_broke() {
    use rl::objective::numerics::NumericsBounds;
    let gate = |bounds: NumericsBounds| PolicyGradientConfig { numerics: Some(bounds), behaviour: BehaviourSource::Recorded, kl_beta: 0.0, log_kl: false, ..Default::default() };
    let strict = NumericsBounds { max_mean_abs_gap: Some(1e-4), max_abs_gap: Some(1e-3), max_mismatch_kl: Some(1e-6), min_sequence_ess_fraction: Some(0.99), max_grad_norm: Some(1e6) };
    let run = |shift: f32| {
        let mut m = lora_model(31);
        let before: Vec<Vec<f32>> = m.param_names().iter().map(|n| m.read_weight(n)).collect();
        let other = [2u32, 3, 4, 5, 6, 7];
        let other_now = answer_logprobs(&m, &Row::new(&other, 3, None));
        let rows = [shifted(&m, 1.0, shift), Row { behavior: Some(other_now.iter().map(|lp| (lp - shift).min(0.0)).collect()), ..Row::new(&other, 3, Some(-1.0)) }];
        let (mut obj, _dir) = started(&rows, gate(strict), &mut m);
        for _ in 0..4 {
            m.zero_grads();
            Objective::<Qwen35>::micro_steps(&mut obj, &m, &mut data::rng::Rng::new(1), 1);
            m.poll_wait();
            assert!(matches!(Objective::<Qwen35>::step_verdict(&mut obj), StepVerdict::Skip(_)), "a measurement takes no optimiser step");
        }
        let after: Vec<Vec<f32>> = m.param_names().iter().map(|n| m.read_weight(n)).collect();
        assert_eq!(before, after);
        obj.stats().numerics.clone().expect("a gate run reports")
    };

    let exact = run(0.0);
    assert!(exact.passed(), "{:?}", exact.violations);
    assert!(exact.gap_max_abs < 1e-3 && (exact.ess_fraction_sequence - 1.0).abs() < 1e-3, "{exact:?}");
    assert_eq!(exact.grad_batches, 4);

    let off = run(0.3);
    assert!((off.gap_mean_abs - 0.3).abs() < 1e-3 && (off.gap_mean - 0.3).abs() < 1e-3, "a 0.3 shift: {off:?}");
    let kl = (0.3f32.exp() - 1.0 - 0.3).abs();
    assert!((off.mismatch_kl - kl).abs() < 1e-3, "k3 of a 0.3 gap is {kl}: {off:?}");
    assert!(!off.passed() && off.violations.iter().any(|v| v.contains("mean |log-prob gap|")) && off.violations.iter().any(|v| v.contains("mismatch KL")), "{:?}", off.violations);
}

#[test]
fn the_numerics_gate_refuses_what_would_make_it_meaningless() {
    use rl::objective::numerics::NumericsBounds;
    let dir = tempfile::tempdir().unwrap();
    super::external_policy_gradient::write_dataset(dir.path(), &[Row::new(&IDS, 3, Some(1.0))]);
    let run = super::external_policy_gradient::opts(1, 1, 1e-2);
    let gate = Some(NumericsBounds::default());
    for (bad, named) in [
        (PolicyGradientConfig { numerics: gate, behaviour: BehaviourSource::StartOfRunProxy, ..Default::default() }, "recorded"),
        (PolicyGradientConfig { numerics: gate, importance: ImportanceCorrection::Truncate { cap: 2.0 }, ..Default::default() }, "importance"),
        (PolicyGradientConfig { numerics: gate, max_step_kl: Some(0.1), ..Default::default() }, "guard"),
    ] {
        let err = ExternalPolicyGradient::open(dir.path(), &run, bad).err().unwrap_or_else(|| panic!("{named} accepted"));
        assert!(err.to_string().contains(named), "{named}: {err}");
    }
}
