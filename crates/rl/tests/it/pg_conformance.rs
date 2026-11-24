// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving that a reinforcement-learning objective computes what its
// paper says, for its clients. If your team needs expertise in verified policy-gradient training then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Numerical conformance of the policy-gradient objective against a golden fixture whose expected values are
//! independent of brain (`tests/fixtures/pg_objective.json`, written by `tools/goldens/pg_objective_reference.py`
//! from the definitions, gradients by float64 autograd; this test never runs Python).
//!
//! Three layers, each read through the code a training step runs:
//! - one token's term ([`rl::objective::pg_batch::weigh`]): the importance ratio, the ratio-weighted k3 KL, the clipped
//!   (and dual-clipped) surrogate's loss and its gradient weight, and the clip flags, on both sides of every edge and
//!   for both advantage signs;
//! - one group's advantages ([`rl::objective::advantage::group_advantages_with`]) under every normalisation, and the
//!   isolation of groups in a run's dataset ([`ExternalPolicyGradient::advantage`]);
//! - one optimiser step's aggregation ([`LossAgg::denominator`] over two one-row micro-batches with prompt and padding
//!   positions): the step's loss and the gradient of every position, zero where nothing is supervised.
//!
//! - the preference losses: DPO's pair term ([`pair_loss_term`]) and KTO's example term ([`feedback_term`]), their
//!   values and the per-token weights a trainer installs (the count times minus the loss's derivative in the summed
//!   log-probability, or in the log-ratio).
//!
//! A failure names the earliest divergence: the first case, in fixture order, and its first field, in the order a
//! step computes them (ratio, KL, loss, weight, flags). Every check also runs against deliberate mutants of the
//! objective written here (a reversed advantage, a wrong clip, the unweighted k3, ...) and must reject each one;
//! [`every_mutant_of_the_objective_is_killed`] prints the kill table.

use data::chat::EncodedSplit;
use model::IGNORE;
use rl::objective::advantage::{group_advantages_with, AdvantageNorm};
use rl::objective::dpo::{pair_loss_term, PairLoss, PairTerm, RowSums};
use rl::objective::feedback::{feedback_term, FeedbackConfig, FeedbackTerm};
use rl::objective::external::{ExternalPolicyGradient, PolicyGradientConfig};
use rl::objective::grpo::LossAgg;
use rl::objective::grpo_term::ClipBand;
use rl::objective::pg_batch::{weigh, PgBatch, PgSpec};
use rl::objective::pg_loss::PgLoss;
use serde_json::Value;

use super::external_policy_gradient::{no_kl, opts};

fn fixture() -> Value {
    let path = brain_testutil::crate_path("tests/fixtures/pg_objective.json");
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))).expect("the fixture is JSON")
}

fn f(v: &Value) -> f32 {
    v.as_f64().unwrap_or_else(|| panic!("{v} is not a number")) as f32
}

fn opt(v: &Value) -> Option<f32> {
    v.as_f64().map(|x| x as f32)
}

/// One token's inputs.
#[derive(Clone, Copy, Debug)]
struct Token {
    old: f32,
    new: f32,
    adv: f32,
    band: ClipBand,
    ref_lp: Option<f32>,
    beta: f32,
    w: f32,
}

/// What one token's objective computes, in the order a step computes it.
#[derive(Clone, Copy, Debug, Default)]
struct Term {
    ratio: f64,
    kl: Option<f64>,
    loss: f64,
    weight: f64,
    clipped: bool,
    dual_clipped: bool,
}

/// The objective as a training step runs it: one supervised position through [`weigh`].
fn brain_term(t: &Token) -> Term {
    let (targets, old, refs, adv, w) = ([1u32], [t.old], [t.ref_lp.unwrap_or(0.0)], [t.adv], [t.w]);
    let batch = PgBatch { targets: &targets, old_lp: &old, ref_lp: t.ref_lp.map(|_| &refs[..]), advantage: &adv, is_weight: Some(&w), top: None };
    let (weights, terms) = weigh(&[t.new], None, &batch, &PgSpec::clipped(t.band, t.beta));
    Term { ratio: terms.ratio_sum, kl: (terms.kl_tokens > 0).then_some(f64::from(terms.kl_sum)), loss: f64::from(terms.loss), weight: f64::from(weights.label[0]), clipped: terms.clipped == 1, dual_clipped: terms.dual_clipped == 1 }
}

fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 2e-5 * want.abs().max(1.0)
}

/// The first divergence of `term` from the fixture's token cases, named.
fn check_terms(fx: &Value, term: &TermFn) -> Result<(), String> {
    for case in fx["token_terms"].as_array().expect("token_terms") {
        let band = ClipBand { low: f(&case["low"]), high: f(&case["high"]), dual: opt(&case["dual"]) };
        let t = Token { old: f(&case["old"]), new: f(&case["new"]), adv: f(&case["adv"]), band, ref_lp: opt(&case["ref"]), beta: f(&case["beta"]), w: f(&case["w"]) };
        let (got, want, name) = (term(&t), &case["expected"], case["name"].as_str().unwrap());
        let diverged = |field: &str, g: String, w: String| Err(format!("token case {name}: {field} {g} against the reference {w} (inputs {t:?})"));
        if let Some(w) = want["ratio"].as_f64() {
            if !close(got.ratio, w) {
                return diverged("ratio", got.ratio.to_string(), w.to_string());
            }
        }
        if let Some(w) = want["kl"].as_f64() {
            if !got.kl.is_some_and(|k| close(k, w)) {
                return diverged("ratio-weighted k3", format!("{:?}", got.kl), w.to_string());
            }
        }
        for (field, g, w) in [("loss", got.loss, &want["loss"]), ("gradient weight", got.weight, &want["weight"])] {
            let w = w.as_f64().unwrap();
            if !close(g, w) {
                return diverged(field, g.to_string(), w.to_string());
            }
        }
        for (field, g, w) in [("clipped", got.clipped, &want["clipped"]), ("dual_clipped", got.dual_clipped, &want["dual_clipped"])] {
            if Some(g) != w.as_bool() {
                return diverged(field, g.to_string(), w.to_string());
            }
        }
    }
    Ok(())
}

fn norm_of(name: &str) -> AdvantageNorm {
    AdvantageNorm::parse(name).expect("a normalisation the fixture names")
}

/// The first divergence of `advantages` from the fixture's group cases, named.
fn check_advantages(fx: &Value, advantages: &AdvantagesFn) -> Result<(), String> {
    for case in fx["advantages"].as_array().expect("advantages") {
        let rewards: Vec<f64> = case["rewards"].as_array().unwrap().iter().map(|r| r.as_f64().unwrap()).collect();
        let want: Vec<f64> = case["expected"].as_array().unwrap().iter().map(|r| r.as_f64().unwrap()).collect();
        let got = advantages(norm_of(case["norm"].as_str().unwrap()), &rewards);
        let name = case["name"].as_str().unwrap();
        if got.len() != want.len() {
            return Err(format!("advantage case {name}: {} advantages for {} rewards", got.len(), want.len()));
        }
        if let Some(i) = (0..want.len()).find(|&i| !(got[i].is_finite() && (got[i] - want[i]).abs() <= 1e-9 * want[i].abs().max(1.0))) {
            return Err(format!("advantage case {name}: member {i} {} against the reference {} (rewards {rewards:?})", got[i], want[i]));
        }
    }
    Ok(())
}

/// One row of an aggregation case.
struct AggRow {
    supervised: Vec<bool>,
    adv: f32,
    old: Vec<f32>,
    new: Vec<f32>,
    ref_lp: Vec<f32>,
}

/// A step of one-row micro-batches as a training step divides it: each row's terms through [`weigh`], divided by the
/// aggregation's denominator (the row's own supervised count where it has none, as the model's loss does) and by the
/// micro-batch count (the fit loop's averaging). Returns the step's loss and every position's `d loss / d new_lp`.
/// That the model divides its installed weights by this denominator is the model's own contract, checked through a real
/// model by `external_policy_gradient_scale`.
type StepFn = dyn Fn(LossAgg, &PgSpec, &[AggRow]) -> (f64, Vec<Vec<f64>>);

fn brain_step(agg: LossAgg, spec: &PgSpec, rows: &[AggRow]) -> (f64, Vec<Vec<f64>>) {
    let k = rows.len();
    let total: usize = rows.iter().map(|r| r.supervised.iter().filter(|s| **s).count()).sum();
    let mut loss = 0.0f64;
    let mut grads = Vec::new();
    for row in rows {
        let targets: Vec<u32> = row.supervised.iter().map(|&s| if s { 1 } else { IGNORE }).collect();
        let adv = vec![row.adv; targets.len()];
        let batch = PgBatch { targets: &targets, old_lp: &row.old, ref_lp: Some(&row.ref_lp), advantage: &adv, is_weight: None, top: None };
        let (weights, terms) = weigh(&row.new, None, &batch, spec);
        let den = f64::from(agg.denominator(1, total, k).unwrap_or(terms.supervised.max(1) as f32));
        loss += f64::from(terms.loss) / den / k as f64;
        grads.push(weights.label.iter().map(|w| -f64::from(*w) / den / k as f64).collect());
    }
    (loss, grads)
}

fn check_aggregation(fx: &Value, step: &StepFn) -> Result<(), String> {
    for case in fx["aggregation"].as_array().expect("aggregation") {
        let name = case["name"].as_str().unwrap();
        let agg: LossAgg = case["agg"].as_str().unwrap().parse().expect("an aggregation the fixture names");
        let spec = PgSpec::clipped(ClipBand::new(f(&case["low"]), f(&case["high"])), f(&case["beta"]));
        let floats = |v: &Value| v.as_array().unwrap().iter().map(f).collect::<Vec<f32>>();
        let rows: Vec<AggRow> = case["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| AggRow { supervised: r["supervised"].as_array().unwrap().iter().map(|s| s.as_bool().unwrap()).collect(), adv: f(&r["adv"]), old: floats(&r["old"]), new: floats(&r["new"]), ref_lp: floats(&r["ref"]) })
            .collect();
        let (loss, grads) = step(agg, &spec, &rows);
        let want_loss = case["expected"]["loss"].as_f64().unwrap();
        for (b, want_row) in case["expected"]["grad"].as_array().unwrap().iter().enumerate() {
            for (t, want) in want_row.as_array().unwrap().iter().enumerate() {
                let (got, want) = (grads[b][t], want.as_f64().unwrap());
                let tolerance = if rows[b].supervised[t] { 2e-5 * want.abs().max(1e-3) } else { 0.0 };
                if (got - want).abs() > tolerance {
                    let what = if rows[b].supervised[t] { "a supervised position" } else { "a prompt or padding position" };
                    return Err(format!("aggregation case {name}: row {b} position {t} ({what}) gradient {got} against the reference {want}"));
                }
            }
        }
        if !close(loss, want_loss) {
            return Err(format!("aggregation case {name}: step loss {loss} against the reference {want_loss}"));
        }
    }
    Ok(())
}

/// Numerical: every token term the fixture lists, through the code a step runs.
#[test]
fn every_token_term_matches_the_independent_reference() {
    check_terms(&fixture(), &brain_term).unwrap();
}

/// Numerical: every group's advantages under every normalisation; equal rewards are zero, never NaN.
#[test]
fn every_groups_advantages_match_the_independent_reference() {
    check_advantages(&fixture(), &group_advantages_with).unwrap();
}

/// Numerical: a step's loss and every position's gradient under each aggregation; prompt and padding positions have
/// none.
#[test]
fn every_aggregation_divides_the_step_as_the_reference_does() {
    check_aggregation(&fixture(), &brain_step).unwrap();
}

/// A run's advantages for `(group, reward)` rollouts, one record each, read back as a run opened on them computes them.
fn run_advantages(rollouts: &[(&str, f64)], norm: AdvantageNorm) -> Vec<f64> {
    use data::chat_rollout::{Member, Outcome};
    let dir = tempfile::tempdir().unwrap();
    let mut split = EncodedSplit::default();
    for (k, &(group, reward)) in rollouts.iter().enumerate() {
        let ids: Vec<u32> = (0..6).map(|i| 1 + ((k + i) % 20) as u32).collect();
        let outcome = Outcome { reward, group: group.to_string(), member: Member::Named(format!("r{k}")), teacher_log_ratio: None };
        split.push_rollout(&data::chat::EncodedSample { ids, mask: (0..6).map(|i| i >= 3).collect(), behavior: vec![f32::NAN; 6], loss_weights: None, teacher: None, truncated: false, teacher_logprobs: Vec::new(), ..Default::default() }, &[], Some(Some(outcome)));
    }
    data::chat::write_split(dir.path(), "train", &split).unwrap();
    data::chat::write_split(dir.path(), "val", &split).unwrap();
    std::fs::write(dir.path().join("meta.json"), data::binio::Meta::vocab_only(29)).unwrap();
    let obj = ExternalPolicyGradient::open(dir.path(), &opts(1, 1, 1e-2), PolicyGradientConfig { loss: PgLoss { advantage_norm: norm, ..no_kl().loss }, ..no_kl() }).unwrap();
    (0..rollouts.len()).map(|e| f64::from(obj.advantage(e).unwrap())).collect()
}

/// Algorithmic: a rollout's advantage is computed from its own group alone. Group a = [1, 1, 0, 0] gives
/// [+1/2, +1/2, -1/2, -1/2] mean-centred and `+-0.5 / (0.5 + 1e-4)` z-scored, by hand, whatever group b holds; and
/// changing group b's rewards leaves group a's advantages bit for bit.
/// A run's advantages for `(group, reward)` rollouts under a normalisation.
type RunAdvantagesFn = dyn Fn(&[(&str, f64)], AdvantageNorm) -> Vec<f64>;

fn check_isolation(advantages: &RunAdvantagesFn) -> Result<(), String> {
    let a = [("a", 1.0), ("a", 1.0), ("a", 0.0), ("a", 0.0)];
    let with_b = |b: [f64; 3]| a.iter().copied().chain(b.into_iter().map(|r| ("b", r))).collect::<Vec<_>>();
    for (norm, want) in [(AdvantageNorm::MeanCentered, 0.5), (AdvantageNorm::ZScore, 0.5 / (0.5 + 1e-4)), (AdvantageNorm::LeaveOneOut, 2.0 / 3.0)] {
        let first = advantages(&with_b([9.0, 2.0, 2.5]), norm);
        let hand = [want, want, -want, -want];
        if let Some(i) = (0..4).find(|&i| (first[i] - hand[i]).abs() > 1e-6) {
            return Err(format!("{norm:?}: group a member {i} has the advantage {} where its own group gives {} (group b's rewards entered it)", first[i], hand[i]));
        }
        let second = advantages(&with_b([-4.0, 0.0, 100.0]), norm);
        if first[..4] != second[..4] {
            return Err(format!("{norm:?}: group a's advantages moved from {:?} to {:?} when only group b's rewards changed", &first[..4], &second[..4]));
        }
    }
    Ok(())
}

#[test]
fn a_groups_advantages_depend_on_its_own_rewards_alone() {
    check_isolation(&run_advantages).unwrap();
}

/// The tokens a preference row's weight is spread over (the forward's supervised count a model divides by).
const COUNT: f32 = 6.0;

fn pref_close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 1e-5 * want.abs() + 1e-9
}

type DpoFn = dyn Fn(f32, RowSums, RowSums) -> PairTerm;

fn brain_dpo(beta: f32, chosen: RowSums, rejected: RowSums) -> PairTerm {
    pair_loss_term(PairLoss::Dpo, beta, chosen, rejected, COUNT)
}

/// The first divergence of the DPO pair term from the fixture: the loss, then each row's weight, which is `COUNT`
/// times minus the loss's derivative in that row's summed log-probability.
fn check_dpo(fx: &Value, term: &DpoFn) -> Result<(), String> {
    for case in fx["dpo"].as_array().expect("dpo") {
        let name = case["name"].as_str().unwrap();
        let row = |new: &str, reference: &str| RowSums { new: f(&case[new]), reference: f(&case[reference]), count: 3 };
        let got = term(f(&case["beta"]), row("chosen", "chosen_ref"), row("rejected", "rejected_ref"));
        let want = &case["expected"];
        let count = f64::from(COUNT);
        for (field, g, w) in [("loss", f64::from(got.loss), want["loss"].as_f64().unwrap()), ("chosen weight", f64::from(got.weight_chosen), -count * want["d_chosen"].as_f64().unwrap()), ("rejected weight", f64::from(got.weight_rejected), -count * want["d_rejected"].as_f64().unwrap())] {
            if !pref_close(g, w) {
                return Err(format!("DPO case {name}: {field} {g} against the reference {w}"));
            }
        }
    }
    Ok(())
}

type KtoFn = dyn Fn(&FeedbackConfig, bool, f32, f32) -> FeedbackTerm;

fn brain_kto(cfg: &FeedbackConfig, desirable: bool, logratio: f32, z: f32) -> FeedbackTerm {
    feedback_term(cfg, desirable, logratio, z, COUNT)
}

/// The first divergence of the KTO term from the fixture: the loss, then the weight (`COUNT` times minus the loss's
/// derivative in the log-ratio, the reference point held fixed).
fn check_kto(fx: &Value, term: &KtoFn) -> Result<(), String> {
    for case in fx["kto"].as_array().expect("kto") {
        let name = case["name"].as_str().unwrap();
        let cfg = FeedbackConfig { beta: f(&case["beta"]), desirable_weight: f(&case["desirable_weight"]), undesirable_weight: f(&case["undesirable_weight"]), seq_len: 8 };
        let got = term(&cfg, case["desirable"].as_bool().unwrap(), f(&case["logratio"]), f(&case["z"]));
        let want = &case["expected"];
        for (field, g, w) in [("loss", f64::from(got.loss), want["loss"].as_f64().unwrap()), ("weight", f64::from(got.weight), -f64::from(COUNT) * want["d_logratio"].as_f64().unwrap())] {
            if !pref_close(g, w) {
                return Err(format!("KTO case {name}: {field} {g} against the reference {w}"));
            }
        }
    }
    Ok(())
}

/// Numerical: the DPO pair term and its row weights.
#[test]
fn the_dpo_term_matches_the_independent_reference() {
    check_dpo(&fixture(), &brain_dpo).unwrap();
}

/// Numerical: the KTO term and its weight, both kinds of example, either side of the reference point.
#[test]
fn the_kto_term_matches_the_independent_reference() {
    check_kto(&fixture(), &brain_kto).unwrap();
}

// ---- Mutants: each is the objective with one deliberate defect; every check above must reject at least one. ----

/// The KL part of a term alone (no policy term), from the real objective.
fn kl_only(t: &Token) -> Term {
    brain_term(&Token { adv: 0.0, ..*t })
}

fn add_policy(kl: Term, ratio: f64, loss: f64, weight: f64, clipped: bool) -> Term {
    Term { ratio, kl: kl.kl, loss: kl.loss + loss, weight: kl.weight + weight, clipped, dual_clipped: false }
}

type TermFn = dyn Fn(&Token) -> Term;
type AdvantagesFn = dyn Fn(AdvantageNorm, &[f64]) -> Vec<f64>;

fn term_mutants() -> Vec<(&'static str, Box<TermFn>)> {
    vec![
        ("reversed advantage sign", Box::new(|t: &Token| brain_term(&Token { adv: -t.adv, ..*t }))),
        (
            "clip without the pessimistic min (always the clipped term)",
            Box::new(|t: &Token| {
                let r = f64::from(t.new - t.old).exp();
                let (lo, hi) = (1.0 - f64::from(t.band.low), 1.0 + f64::from(t.band.high));
                let a = f64::from(t.adv) * f64::from(t.w);
                let inside = (lo..=hi).contains(&r);
                add_policy(kl_only(t), r, -r.clamp(lo, hi) * a, if inside { a * r } else { 0.0 }, !inside && a != 0.0)
            }),
        ),
        ("symmetric band (upper edge ignored)", Box::new(|t: &Token| brain_term(&Token { band: ClipBand { high: t.band.low, ..t.band }, ..*t }))),
        ("dual clip dropped", Box::new(|t: &Token| brain_term(&Token { band: ClipBand { dual: None, ..t.band }, ..*t }))),
        (
            "unweighted k3 (knowledge 537)",
            Box::new(|t: &Token| {
                let policy = brain_term(&Token { beta: 0.0, ref_lp: None, ..*t });
                let Some(ref_lp) = t.ref_lp.filter(|_| t.beta > 0.0) else { return policy };
                let u = f64::from(ref_lp) - f64::from(t.new);
                let beta = f64::from(t.beta);
                Term { kl: Some(u.exp() - u - 1.0), loss: policy.loss + beta * (u.exp() - u - 1.0), weight: policy.weight + beta * (u.exp() - 1.0), ..policy }
            }),
        ),
        ("KL term dropped from the loss", Box::new(|t: &Token| Term { kl: kl_only(t).kl, ..brain_term(&Token { beta: 0.0, ..*t }) })),
        ("ratio against the current policy (behaviour recomputed)", Box::new(|t: &Token| Term { ratio: f64::from(t.new - t.old).exp(), ..brain_term(&Token { old: t.new, ..*t }) })),
        ("policy weight ignored", Box::new(|t: &Token| brain_term(&Token { w: 1.0, ..*t }))),
    ]
}

fn advantage_mutants() -> Vec<(&'static str, Box<AdvantagesFn>)> {
    let stats = |r: &[f64], ddof: f64| {
        let g = r.len() as f64;
        let mean = r.iter().sum::<f64>() / g;
        (mean, (r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (g - ddof)).sqrt())
    };
    vec![
        ("reversed advantage sign", Box::new(|n, r: &[f64]| group_advantages_with(n, r).into_iter().map(|a| -a).collect())),
        (
            "sample std (G - 1) in the z-score",
            Box::new(move |n, r: &[f64]| match n {
                AdvantageNorm::ZScore if r.len() > 1 && stats(r, 0.0).1 >= 1e-6 => {
                    let (mean, std) = stats(r, 1.0);
                    r.iter().map(|x| (x - mean) / (std + 1e-4)).collect()
                }
                _ => group_advantages_with(n, r),
            }),
        ),
        (
            "a flat group is not dropped",
            Box::new(move |n, r: &[f64]| match n {
                AdvantageNorm::ZScore if r.len() > 1 => {
                    let (mean, std) = stats(r, 0.0);
                    r.iter().map(|x| (x - mean) / std).collect()
                }
                _ => group_advantages_with(n, r),
            }),
        ),
        ("leave-one-out includes the member itself", Box::new(|n, r: &[f64]| group_advantages_with(if n == AdvantageNorm::LeaveOneOut { AdvantageNorm::MeanCentered } else { n }, r))),
        ("a group of one gets no weight", Box::new(|n, r: &[f64]| if r.len() == 1 { vec![0.0] } else { group_advantages_with(n, r) })),
    ]
}

fn isolation_mutants() -> Vec<(&'static str, Box<RunAdvantagesFn>)> {
    vec![(
        "every rollout normalised against the whole dataset (wrong group)",
        Box::new(|rollouts: &[(&str, f64)], norm| {
            let rewards: Vec<f64> = rollouts.iter().map(|r| r.1).collect();
            group_advantages_with(norm, &rewards)
        }),
    )]
}

fn aggregation_mutants() -> Vec<(&'static str, Box<StepFn>)> {
    vec![
        (
            "padding and prompt positions counted in the denominator",
            Box::new(|agg, spec: &PgSpec, rows: &[AggRow]| {
                let (_, grads) = brain_step(agg, spec, rows);
                let (k, t) = (rows.len() as f64, rows[0].supervised.len() as f64);
                let scale: Vec<f64> = rows
                    .iter()
                    .map(|r| {
                        let own = r.supervised.iter().filter(|s| **s).count() as f64;
                        let total: f64 = rows.iter().map(|x| x.supervised.iter().filter(|s| **s).count() as f64).sum();
                        match agg {
                            LossAgg::TokenSum => total / (k * t),
                            LossAgg::SeqMean => own / t,
                            LossAgg::ConstantLength(_) => 1.0,
                        }
                    })
                    .collect();
                let grads: Vec<Vec<f64>> = grads.iter().zip(&scale).map(|(g, s)| g.iter().map(|x| x * s).collect()).collect();
                (brain_step(agg, spec, rows).0, grads)
            }),
        ),
        (
            "prompt and padding positions trained",
            Box::new(|agg, spec: &PgSpec, rows: &[AggRow]| {
                let all: Vec<AggRow> = rows.iter().map(|r| AggRow { supervised: vec![true; r.supervised.len()], adv: r.adv, old: r.old.clone(), new: r.new.clone(), ref_lp: r.ref_lp.clone() }).collect();
                brain_step(agg, spec, &all)
            }),
        ),
        ("every aggregation is a token sum", Box::new(|_, spec: &PgSpec, rows: &[AggRow]| brain_step(LossAgg::TokenSum, spec, rows))),
        ("every aggregation is a sequence mean", Box::new(|_, spec: &PgSpec, rows: &[AggRow]| brain_step(LossAgg::SeqMean, spec, rows))),
    ]
}

fn dpo_mutants() -> Vec<(&'static str, Box<DpoFn>)> {
    let no_ref = |r: RowSums| RowSums { reference: 0.0, ..r };
    vec![
        ("chosen and rejected swapped", Box::new(|beta, c, r| brain_dpo(beta, r, c))),
        ("reference dropped", Box::new(move |beta, c, r| brain_dpo(beta, no_ref(c), no_ref(r)))),
        ("beta outside the sigmoid", Box::new(|beta, c, r| {
            let t = brain_dpo(1.0, c, r);
            PairTerm { loss: beta * t.loss, weight_chosen: beta * t.weight_chosen, weight_rejected: beta * t.weight_rejected }
        })),
        ("rejected row not pushed down", Box::new(|beta, c, r| PairTerm { weight_rejected: 0.0, ..brain_dpo(beta, c, r) })),
    ]
}

fn kto_mutants() -> Vec<(&'static str, Box<KtoFn>)> {
    vec![
        ("desirable and undesirable swapped", Box::new(|cfg: &FeedbackConfig, d, x, z| brain_kto(cfg, !d, x, z))),
        ("reference point dropped", Box::new(|cfg: &FeedbackConfig, d, x, _z| brain_kto(cfg, d, x, 0.0))),
        ("one weight for both kinds", Box::new(|cfg: &FeedbackConfig, d, x, z| brain_kto(&FeedbackConfig { undesirable_weight: cfg.desirable_weight, ..*cfg }, d, x, z))),
        ("weight of the wrong sign", Box::new(|cfg: &FeedbackConfig, d, x, z| {
            let t = brain_kto(cfg, d, x, z);
            FeedbackTerm { weight: -t.weight, ..t }
        })),
    ]
}

/// Every mutant of the objective is rejected by the check that covers it, at a named earliest divergence. Prints the
/// kill table (`--nocapture`).
#[test]
fn every_mutant_of_the_objective_is_killed() {
    let fx = fixture();
    let mut table = Vec::new();
    for (name, m) in term_mutants() {
        table.push(("token term", name, check_terms(&fx, &*m).err()));
    }
    for (name, m) in advantage_mutants() {
        table.push(("advantages", name, check_advantages(&fx, &*m).err()));
    }
    for (name, m) in isolation_mutants() {
        table.push(("group isolation", name, check_isolation(&*m).err()));
    }
    for (name, m) in aggregation_mutants() {
        table.push(("aggregation", name, check_aggregation(&fx, &*m).err()));
    }
    for (name, m) in dpo_mutants() {
        table.push(("DPO", name, check_dpo(&fx, &*m).err()));
    }
    for (name, m) in kto_mutants() {
        table.push(("KTO", name, check_kto(&fx, &*m).err()));
    }
    let killed = table.iter().filter(|(_, _, e)| e.is_some()).count();
    for (area, name, divergence) in &table {
        println!("{area:16} {name:62} {}", divergence.as_deref().map_or("SURVIVED".to_string(), |d| format!("killed: {d}")));
    }
    println!("objective mutants killed: {killed}/{}", table.len());
    let survivors: Vec<_> = table.iter().filter(|(_, _, e)| e.is_none()).map(|(a, n, _)| format!("{a}: {n}")).collect();
    assert!(survivors.is_empty(), "mutants the conformance checks did not reject: {survivors:?}");
}
