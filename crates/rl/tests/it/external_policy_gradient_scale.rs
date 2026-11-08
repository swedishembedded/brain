// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for knowing exactly how far one rollout moves a language model, for its
// clients. If your team needs expertise in verifying the scale of reinforcement-learning updates then you can procure
// our services by sending an email to info@swedishembedded.com.

//! The update scale of the external policy gradient against an FP64 reference of its documented objective.
//!
//! A step's gradient is the mean over its records of each record's own per-token mean ([`LossAgg::SeqMean`]) or the
//! sum over every token divided by the step's token count ([`LossAgg::TokenSum`]). The reference here evaluates
//! those objectives in f64 from the model's own log-probabilities, independently of the objective's code, with
//! record lengths and advantage signs varied, and the step's accumulated gradient must be its directional derivative.

use data::rng::Rng;
use model::{FitOpts, Objective, StepVerdict};
use qwen35::model::Qwen35;
use rl::objective::external::{BehaviourSource, ExternalPolicyGradient, PolicyGradientConfig, ReferencePolicy};
use rl::objective::grpo::LossAgg;

use super::external_policy_gradient::{answer_logprobs, lora_model, opts, write_dataset, Row};

const CLIP_EPS: f64 = 0.2;
const KL_BETA: f64 = 0.1;

/// Records of unequal answer lengths (1, 2, 3 and 5 supervised tokens) and both advantage signs.
fn records(m: &Qwen35) -> Vec<Row> {
    let shapes: [(&[u32], usize, f32); 4] = [(&[1, 2, 3, 4, 5, 6, 7, 8], 3, 1.3), (&[7, 8, 9, 10, 11, 12], 4, -0.9), (&[2, 3, 4, 5, 6], 2, 0.4), (&[3, 4], 1, -1.6)];
    // Behaviour log-probabilities a little off the policy's own, so ratios sit inside the clip band but are not 1.
    let offsets = [0.06f32, -0.05, 0.04, -0.07, 0.05];
    shapes
        .iter()
        .map(|&(ids, from, advantage)| {
            let mut row = Row::new(ids, from, Some(advantage));
            let now = answer_logprobs(m, &row);
            row.behavior = Some(now.iter().enumerate().map(|(t, lp)| lp - offsets[t % offsets.len()]).collect());
            row
        })
        .collect()
}

/// The reference objective of one record's tokens, in f64: the clipped surrogate and the k3 KL to the base.
fn record_terms(policy: &[f32], behaviour: &[f32], base: &[f32], advantage: f32) -> Vec<f64> {
    (0..policy.len())
        .map(|t| {
            let (a, r) = (f64::from(advantage), (f64::from(policy[t]) - f64::from(behaviour[t])).exp());
            let u = f64::from(base[t]) - f64::from(policy[t]);
            -(r * a).min(r.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * a) + KL_BETA * (u.exp() - u - 1.0)
        })
        .collect()
}

fn base_logprobs(m: &Qwen35, rows: &[Row]) -> Vec<Vec<f32>> {
    let b_names: Vec<String> = m.param_names().into_iter().filter(|n| n.ends_with(".lora_b")).collect();
    let saved: Vec<Vec<f32>> = b_names.iter().map(|n| m.read_weight(n)).collect();
    b_names.iter().zip(&saved).for_each(|(n, w)| m.write_weight(n, &vec![0.0; w.len()]));
    let base = rows.iter().map(|r| answer_logprobs(m, r)).collect();
    b_names.iter().zip(&saved).for_each(|(n, w)| m.write_weight(n, w));
    base
}

/// The objective of the whole step, in f64, as its aggregation defines it.
fn reference_loss(m: &Qwen35, rows: &[Row], base: &[Vec<f32>], agg: LossAgg) -> f64 {
    let per_record: Vec<Vec<f64>> = rows.iter().zip(base).map(|(r, b)| record_terms(&answer_logprobs(m, r), r.behavior.as_ref().unwrap(), b, r.advantage.unwrap())).collect();
    match agg {
        LossAgg::SeqMean => per_record.iter().map(|t| t.iter().sum::<f64>() / t.len() as f64).sum::<f64>() / rows.len() as f64,
        LossAgg::TokenSum => per_record.iter().flatten().sum::<f64>() / per_record.iter().map(Vec::len).sum::<usize>() as f64,
    }
}

fn adapter_names(m: &Qwen35) -> Vec<String> {
    m.param_names().into_iter().filter(|n| n.contains(".lora_")).collect()
}

/// The step's accumulated gradient, divided by the accumulation count as the fit loop does before AdamW.
fn step_gradient(m: &mut Qwen35, rows: &[Row], agg: LossAgg) -> Vec<Vec<f32>> {
    let dir = tempfile::tempdir().unwrap();
    write_dataset(dir.path(), rows);
    let run = opts(1, rows.len() as u32, 1e-2);
    let cfg = PolicyGradientConfig { clip_eps: CLIP_EPS as f32, kl_beta: KL_BETA as f32, reference: ReferencePolicy::Base, behaviour: BehaviourSource::Recorded, loss_agg: agg, ..Default::default() };
    let mut obj = ExternalPolicyGradient::open(dir.path(), &run, cfg).unwrap();
    Objective::<Qwen35>::prepare(&mut obj, m).unwrap();
    Objective::<Qwen35>::begin(&mut obj, 17, 0);
    m.zero_grads();
    Objective::<Qwen35>::micro_steps(&mut obj, m, &mut Rng::new(1), rows.len() as u32);
    m.poll_wait();
    assert!(matches!(Objective::<Qwen35>::step_verdict(&mut obj), StepVerdict::Apply));
    let scale = 1.0 / rows.len() as f32;
    adapter_names(m).iter().map(|n| m.read_grad(n).iter().map(|g| g * scale).collect()).collect()
}

/// Both aggregations of a step over records of unequal length and mixed advantage sign are the gradient of the
/// objective they document: no record is weighted by its length under a sequence mean, every token is weighted
/// alike under a token sum.
#[test]
fn a_steps_gradient_is_the_documented_aggregation_over_unequal_records_of_both_advantage_signs() {
    for (agg, seed) in [(LossAgg::SeqMean, 3), (LossAgg::TokenSum, 3), (LossAgg::SeqMean, 5), (LossAgg::TokenSum, 5)] {
        let mut m = lora_model(seed);
        let rows = records(&m);
        let base = base_logprobs(&m, &rows);
        let grads = step_gradient(&mut m, &rows, agg);

        let names = adapter_names(&m);
        let mut rng = Rng::new(0x5ca1e);
        let direction: Vec<Vec<f32>> = grads.iter().map(|g| g.iter().map(|_| rng.next_f32() - 0.5).collect()).collect();
        let norm = direction.iter().flatten().map(|d| f64::from(*d).powi(2)).sum::<f64>().sqrt();
        let analytic: f64 = grads.iter().flatten().zip(direction.iter().flatten()).map(|(g, d)| f64::from(*g) * f64::from(*d)).sum::<f64>() / norm;

        let originals: Vec<Vec<f32>> = names.iter().map(|n| m.read_weight(n)).collect();
        let h = 5e-3;
        let shifted = |sign: f64| {
            for ((n, w), d) in names.iter().zip(&originals).zip(&direction) {
                let moved: Vec<f32> = w.iter().zip(d).map(|(w, d)| w + (sign * h * f64::from(*d) / norm) as f32).collect();
                m.write_weight(n, &moved);
            }
            reference_loss(&m, &rows, &base, agg)
        };
        let numeric = (shifted(1.0) - shifted(-1.0)) / (2.0 * h);
        for (n, w) in names.iter().zip(&originals) {
            m.write_weight(n, w);
        }
        println!("{agg:?}: directional derivative, brain {analytic:+.6e}, FP64 reference {numeric:+.6e}");
        assert!(analytic.abs() > 1e-4, "{agg:?}: the fixture must have a gradient to compare: {analytic}");
        assert!((analytic - numeric).abs() <= 3e-2 * analytic.abs().max(numeric.abs()), "{agg:?}: brain {analytic}, FP64 reference {numeric}");
    }
}

/// The scale a step moves the policy by: the mean absolute change of a record's token log-probabilities after one
/// optimiser step, per record of the batch, against the learning rate. AdamW normalises the gradient, so the scale
/// of the advantages and the aggregation do not change it; this test pins that invariance (the step the advantages
/// `A` give is the step `1000 A` gives), which is why a tiny learning rate moves a policy imperceptibly.
#[test]
fn rescaling_every_advantage_does_not_change_the_step() {
    let moved = |factor: f32| {
        let m = lora_model(7);
        let mut rows = records(&m);
        rows.iter_mut().for_each(|r| r.advantage = r.advantage.map(|a| a * factor));
        let before: Vec<Vec<f32>> = rows.iter().map(|r| answer_logprobs(&m, r)).collect();
        let dir = tempfile::tempdir().unwrap();
        write_dataset(dir.path(), &rows);
        let run = FitOpts { grad_clip: 0.0, ..opts(1, rows.len() as u32, 1e-3) };
        let cfg = PolicyGradientConfig { kl_beta: 0.0, log_kl: false, behaviour: BehaviourSource::Recorded, ..Default::default() };
        let mut obj = ExternalPolicyGradient::open(dir.path(), &run, cfg).unwrap();
        let (_, m) = model::fit_controlled(m, &mut obj, &run, None, model::FitControl::default()).unwrap();
        let after: Vec<Vec<f32>> = rows.iter().map(|r| answer_logprobs(&m, r)).collect();
        before.iter().flatten().zip(after.iter().flatten()).map(|(b, a)| a - b).collect::<Vec<f32>>()
    };
    let (small, large) = (moved(1.0), moved(1000.0));
    let mean = |v: &[f32]| v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
    println!("mean |change in token log-probability| after one step at lr 1e-3: {:.3e}", mean(&small));
    assert!(mean(&small) > 0.0);
    let worst = small.iter().zip(&large).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst <= 0.05 * small.iter().fold(0.0f32, |a, x| a.max(x.abs())), "a 1000x advantage changed the step by {worst}");
}
