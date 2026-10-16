// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Honest evaluation of a decision model on held-out, outcome-labelled
//! examples.
//!
//! A model scores each example of one split; this module turns the
//! probabilities into the numbers a promotion decision rests on, and every one
//! of them is defined here, once:
//!
//! * **accuracy** - the share of examples whose most probable option is the
//!   labelled one (ties go to the earlier option).
//! * **baseline** - what needs no model. The *majority baseline* always answers
//!   the most common label id of the `train` split (the evaluated split's own
//!   majority when the file has no `train` examples, which flatters the
//!   baseline and is reported as such); the *chance baseline* is the mean of
//!   `1/k` over the examples' option counts. The baseline accuracy is the
//!   larger of the two, and **lift** is accuracy minus baseline.
//! * **NLL** - mean `-ln p(label)`, probabilities floored at `1e-12`.
//! * **squared-error probability score** - mean over examples of the sum over options of
//!   `(p - 1{label})^2`.
//! * **ECE / AdaECE** - calibration of the top probability: examples binned by
//!   `p_max` into 15 equal-width bins (ECE) or 15 equal-count bins (AdaECE),
//!   the count-weighted mean gap between a bin's mean `p_max` and its accuracy.
//!   The reliability bins are reported so the gap can be read, not only summed.
//!   AdaECE cannot separate tied confidences: examples with equal `p_max` fall
//!   into bins in file order, so on a model that is confident to the same
//!   figure everywhere it measures that order rather than calibration; read
//!   ECE and the bins beside it.
//! * **coverage versus accuracy** - examples ranked by the model's `confidence`
//!   (`1 - H/ln k`, the figure the served API reports); the accuracy of the
//!   most confident tenth, two tenths, ... of them. A model whose confidence means
//!   something is more accurate at low coverage.
//! * **order instability** - every example is asked again with its options in
//!   a seeded random order; the share whose winning option *id* changes. A
//!   model that reads the options scores 0.
//! * **regret** - with a cost file, the cost of acting on the model's own
//!   confidence. The action is *trust* (take the top option; if it is wrong
//!   the right one was excluded, costing `false_exclusion`) or *defer* (keep
//!   every option; if the top option was right the extra context was wasted,
//!   costing `wasted_tokens`). The model trusts when its `p_max` makes trusting
//!   the cheaper expected action (`rlcd::cost::bayes_action`); the regret is
//!   the mean realized cost against hindsight. The two constant policies are
//!   reported beside it, and a model that cannot beat the cheaper constant
//!   policy has not learned when to trust itself.
//! * **per family** - the same accuracy, baseline, lift and NLL within each
//!   task family, so a gain in one cannot hide a loss in another.
//!
//! Swedish Embedded AB implements model evaluation that separates a real gain
//! from a lucky split: baselines, calibration and cost-weighted decisions
//! reported together. If your team needs expertise in evaluating decision
//! models before they reach production, you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use rlcd::cost::{bayes_action, regret, CostMatrix};
use rlcd::metrics::{ada_ece, brier_score, coverage_accuracy, ece, nll, reliability_bins};
use serde_json::{json, Value};

use crate::examples::{Dataset, DecisionExample, Split};

/// Bins for ECE, AdaECE and the reliability diagram.
pub const CALIBRATION_BINS: usize = 15;
/// The coverage levels reported on the coverage-versus-accuracy curve.
const COVERAGE_LEVELS: [f32; 10] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

/// Something that scores a question: a probability for every option.
pub trait Scorer {
    /// Probabilities over `options`, in their order, summing to 1.
    fn probabilities(&mut self, state: &str, instruction: &str, options: &[String]) -> Result<Vec<f32>, String>;
}

impl<F> Scorer for F
where
    F: FnMut(&str, &str, &[String]) -> Result<Vec<f32>, String>,
{
    fn probabilities(&mut self, state: &str, instruction: &str, options: &[String]) -> Result<Vec<f32>, String> {
        self(state, instruction, options)
    }
}

/// What a wrong answer and an unneeded fallback cost, in one unit the caller
/// chooses (tokens, dollars, minutes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostSpec {
    /// Cost of trusting a top option that was wrong: the right option was excluded.
    pub false_exclusion: f32,
    /// Cost of deferring when the top option was right: context kept for nothing.
    pub wasted_tokens: f32,
}

impl CostSpec {
    /// Parse `{"false_exclusion": 5.0, "wasted_tokens": 1.0}`; unknown keys are refused.
    pub fn parse(text: &str) -> Result<CostSpec, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| format!("the cost file is not valid JSON: {e}"))?;
        let obj = v.as_object().ok_or("the cost file must be a JSON object")?;
        if let Some(k) = obj.keys().find(|k| !matches!(k.as_str(), "false_exclusion" | "wasted_tokens")) {
            return Err(format!("the cost file has an unknown key {k:?}; expected false_exclusion and wasted_tokens"));
        }
        let cost = |key: &str| -> Result<f32, String> {
            let n = obj.get(key).and_then(Value::as_f64).ok_or_else(|| format!("the cost file needs a numeric `{key}`"))?;
            if !n.is_finite() || n < 0.0 {
                return Err(format!("`{key}` must be a finite number of at least 0, got {n}"));
            }
            Ok(n as f32)
        };
        let spec = CostSpec { false_exclusion: cost("false_exclusion")?, wasted_tokens: cost("wasted_tokens")? };
        if spec.false_exclusion == 0.0 && spec.wasted_tokens == 0.0 {
            return Err("the cost file must make at least one mistake cost something".to_string());
        }
        Ok(spec)
    }

    pub fn read(path: &Path) -> Result<CostSpec, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        CostSpec::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn to_json(&self) -> Value {
        json!({"false_exclusion": self.false_exclusion, "wasted_tokens": self.wasted_tokens})
    }

    /// Actions: 0 defer, 1 trust. Outcomes: 0 the top option is right, 1 it is wrong.
    fn matrix(&self) -> CostMatrix {
        CostMatrix::binary(self.wasted_tokens, self.false_exclusion)
    }
}

/// What to evaluate and how.
#[derive(Clone, Copy, Debug)]
pub struct EvalRequest {
    pub split: Split,
    /// Seeds the option permutation of the order-stability pass.
    pub order_seed: u64,
    /// When set, the report includes cost-weighted regret.
    pub costs: Option<CostSpec>,
}

/// The majority baseline and the chance baseline over one set of examples.
#[derive(Clone, Debug, PartialEq)]
pub struct Baseline {
    pub majority_label: Option<String>,
    pub majority_accuracy: f64,
    pub chance_accuracy: f64,
}

impl Baseline {
    /// The accuracy a model must beat.
    pub fn accuracy(&self) -> f64 {
        self.majority_accuracy.max(self.chance_accuracy)
    }
}

/// One reliability bin, as reported.
#[derive(Clone, Debug, PartialEq)]
pub struct Bin {
    pub lo: f32,
    pub hi: f32,
    pub count: usize,
    pub mean_confidence: Option<f32>,
    pub accuracy: Option<f32>,
}

/// Cost-weighted regret; see the module documentation.
#[derive(Clone, Debug, PartialEq)]
pub struct Regret {
    pub costs: CostSpec,
    /// Mean realized cost per example of the model's own trust-or-defer policy.
    pub model: f64,
    pub always_trust: f64,
    pub always_defer: f64,
    /// Share of examples on which the model chose to trust its top option.
    pub trusted_share: f64,
}

/// Accuracy within one task family.
#[derive(Clone, Debug, PartialEq)]
pub struct FamilyReport {
    pub examples: usize,
    pub accuracy: f64,
    pub baseline_accuracy: f64,
    pub nll: f64,
}

/// Everything measured on one split.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub split: Split,
    pub examples: usize,
    pub accuracy: f64,
    pub baseline: Baseline,
    /// Whether the majority label came from the `train` split (`true`) or from
    /// the evaluated split itself.
    pub baseline_from_train: bool,
    pub lift: f64,
    pub nll: f64,
    pub brier: f64,
    pub ece: f64,
    pub ada_ece: f64,
    pub reliability: Vec<Bin>,
    /// `(coverage, accuracy)` of the most confident share of examples.
    pub coverage_accuracy: Vec<(f64, f64)>,
    pub order_seed: u64,
    pub order_instability: f64,
    pub regret: Option<Regret>,
    pub families: BTreeMap<String, FamilyReport>,
}

/// The model's confidence in a distribution: `1 - H/ln k`, 1 for a certain
/// answer and 0 for a uniform one, whatever `k` is; 1 when there is no choice.
fn contract_confidence(p: &[f32]) -> f32 {
    if p.len() < 2 {
        return 1.0;
    }
    let h: f64 = p.iter().map(|&x| x as f64).filter(|&x| x > 0.0).map(|x| -x * x.ln()).sum();
    (1.0 - h / (p.len() as f64).ln()).clamp(0.0, 1.0) as f32
}

fn argmax(p: &[f32]) -> usize {
    p.iter().enumerate().fold(0, |best, (i, &x)| if x > p[best] { i } else { best })
}

/// A scored example: what the model said about it.
struct Scored<'a> {
    example: &'a DecisionExample,
    probabilities: Vec<f32>,
    top: usize,
}

impl Scored<'_> {
    fn correct(&self) -> bool {
        self.top == self.example.label
    }
}

fn score_all<'a>(scorer: &mut dyn Scorer, examples: &[&'a DecisionExample]) -> Result<Vec<Scored<'a>>, String> {
    examples
        .iter()
        .map(|&e| {
            let p = scorer.probabilities(&e.state, &e.instruction, &e.option_texts()).map_err(|m| format!("line {}: the model failed: {m}", e.line))?;
            if p.len() != e.options.len() {
                return Err(format!("line {}: the model returned {} probabilities for {} options", e.line, p.len(), e.options.len()));
            }
            if p.iter().any(|x| !x.is_finite() || *x < 0.0) {
                return Err(format!("line {}: the model returned a probability that is negative or not finite", e.line));
            }
            let sum: f32 = p.iter().sum();
            if (sum - 1.0).abs() > 1e-3 {
                return Err(format!("line {}: the probabilities sum to {sum}, not 1", e.line));
            }
            let top = argmax(&p);
            Ok(Scored { example: e, probabilities: p, top })
        })
        .collect()
}

/// The share of examples whose winning option id changes when the options are
/// asked about in a seeded random order.
fn order_instability(scorer: &mut dyn Scorer, scored: &[Scored], seed: u64) -> Result<f64, String> {
    let mut changed = 0usize;
    for (i, s) in scored.iter().enumerate() {
        let e = s.example;
        let mut order: Vec<usize> = (0..e.options.len()).collect();
        let mut rng = data::rng::Rng::new(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        for j in (1..order.len()).rev() {
            order.swap(j, (rng.next_u64() % (j as u64 + 1)) as usize);
        }
        let texts: Vec<String> = order.iter().map(|&j| e.options[j].text().to_string()).collect();
        let p = scorer.probabilities(&e.state, &e.instruction, &texts).map_err(|m| format!("line {}: the model failed: {m}", e.line))?;
        if p.len() != texts.len() {
            return Err(format!("line {}: the model returned {} probabilities for {} options", e.line, p.len(), texts.len()));
        }
        let winner = order[argmax(&p)];
        if winner != s.top {
            changed += 1;
        }
    }
    Ok(changed as f64 / scored.len() as f64)
}

fn baseline_of(examples: &[&DecisionExample], majority: Option<&str>) -> Baseline {
    let n = examples.len().max(1) as f64;
    let hits = majority.map_or(0, |m| examples.iter().filter(|e| e.label_id() == m).count());
    Baseline {
        majority_label: majority.map(str::to_string),
        majority_accuracy: hits as f64 / n,
        chance_accuracy: examples.iter().map(|e| 1.0 / e.options.len() as f64).sum::<f64>() / n,
    }
}

fn majority_label<'a>(examples: impl Iterator<Item = &'a DecisionExample>) -> Option<String> {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for e in examples {
        *counts.entry(e.label_id()).or_insert(0) += 1;
    }
    // First maximum in key order, so a tie resolves the same way every run.
    counts.iter().fold(None, |best: Option<(&&str, &usize)>, kv| match best {
        Some((_, b)) if b >= kv.1 => best,
        _ => Some(kv),
    }).map(|(k, _)| k.to_string())
}

fn regret_of(scored: &[Scored], spec: CostSpec) -> Regret {
    let costs = spec.matrix();
    let (mut model, mut trust, mut defer, mut trusted) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    for s in scored {
        let p_max = s.probabilities[s.top].clamp(0.0, 1.0);
        // The belief the model's own probability implies about its top option.
        let action = bayes_action(&[p_max, 1.0 - p_max], &costs).action;
        let outcome: [f32; 2] = if s.correct() { [1.0, 0.0] } else { [0.0, 1.0] };
        model += regret(&outcome, &costs, action) as f64;
        defer += regret(&outcome, &costs, 0) as f64;
        trust += regret(&outcome, &costs, 1) as f64;
        trusted += (action == 1) as usize;
    }
    let n = scored.len() as f64;
    Regret { costs: spec, model: model / n, always_trust: trust / n, always_defer: defer / n, trusted_share: trusted as f64 / n }
}

/// Evaluate `scorer` on one split of `dataset`.
pub fn evaluate(scorer: &mut dyn Scorer, dataset: &Dataset, request: &EvalRequest) -> Result<Report, String> {
    let examples: Vec<&DecisionExample> = dataset.split(request.split).collect();
    if examples.is_empty() {
        return Err(format!("the data has no `{}` examples to evaluate on", request.split.slug()));
    }
    let scored = score_all(scorer, &examples)?;
    let n = scored.len() as f64;

    let train_majority = majority_label(dataset.split(Split::Train));
    let baseline_from_train = train_majority.is_some();
    let majority = train_majority.or_else(|| majority_label(examples.iter().copied()));
    let baseline = baseline_of(&examples, majority.as_deref());

    let labels: Vec<usize> = scored.iter().map(|s| s.example.label).collect();
    let probs: Vec<Vec<f32>> = scored.iter().map(|s| s.probabilities.clone()).collect();
    let correct: Vec<bool> = scored.iter().map(Scored::correct).collect();
    let p_max: Vec<f32> = scored.iter().map(|s| s.probabilities[s.top]).collect();
    let confidence: Vec<f32> = probs.iter().map(|p| contract_confidence(p)).collect();

    let accuracy = correct.iter().filter(|&&c| c).count() as f64 / n;
    let curve = coverage_accuracy(&confidence, &correct);
    let coverage_accuracy = COVERAGE_LEVELS
        .iter()
        .map(|&level| {
            let at = ((level * curve.len() as f32).ceil() as usize).clamp(1, curve.len()) - 1;
            (curve[at].0 as f64, curve[at].1 as f64)
        })
        .fold(Vec::new(), |mut points: Vec<(f64, f64)>, point| {
            if points.last() != Some(&point) {
                points.push(point);
            }
            points
        });

    let mut families = BTreeMap::new();
    for family in examples.iter().map(|e| e.family.clone()).collect::<std::collections::BTreeSet<_>>() {
        let members: Vec<&Scored> = scored.iter().filter(|s| s.example.family == family).collect();
        let member_examples: Vec<&DecisionExample> = members.iter().map(|s| s.example).collect();
        let family_probs: Vec<Vec<f32>> = members.iter().map(|s| s.probabilities.clone()).collect();
        let family_labels: Vec<usize> = members.iter().map(|s| s.example.label).collect();
        families.insert(
            family,
            FamilyReport {
                examples: members.len(),
                accuracy: members.iter().filter(|s| s.correct()).count() as f64 / members.len() as f64,
                baseline_accuracy: baseline_of(&member_examples, majority.as_deref()).accuracy(),
                nll: nll(&family_probs, &family_labels) as f64,
            },
        );
    }

    Ok(Report {
        split: request.split,
        examples: scored.len(),
        accuracy,
        lift: accuracy - baseline.accuracy(),
        baseline,
        baseline_from_train,
        nll: nll(&probs, &labels) as f64,
        brier: brier_score(&probs, &labels) as f64,
        ece: ece(&p_max, &correct, CALIBRATION_BINS) as f64,
        ada_ece: ada_ece(&p_max, &correct, CALIBRATION_BINS) as f64,
        reliability: reliability_bins(&p_max, &correct, CALIBRATION_BINS)
            .into_iter()
            .map(|b| Bin { lo: b.lo, hi: b.hi, count: b.count, mean_confidence: (b.count > 0).then_some(b.mean_confidence), accuracy: (b.count > 0).then_some(b.accuracy) })
            .collect(),
        coverage_accuracy,
        order_seed: request.order_seed,
        order_instability: order_instability(scorer, &scored, request.order_seed)?,
        regret: request.costs.map(|c| regret_of(&scored, c)),
        families,
    })
}

impl Report {
    pub fn to_json(&self) -> Value {
        let bins: Vec<Value> = self.reliability.iter().map(|b| json!({"lo": b.lo, "hi": b.hi, "count": b.count, "mean_confidence": b.mean_confidence, "accuracy": b.accuracy})).collect();
        let coverage: Vec<Value> = self.coverage_accuracy.iter().map(|(c, a)| json!({"coverage": c, "accuracy": a})).collect();
        let families: serde_json::Map<String, Value> = self
            .families
            .iter()
            .map(|(name, f)| (name.clone(), json!({"examples": f.examples, "accuracy": f.accuracy, "baseline_accuracy": f.baseline_accuracy, "nll": f.nll})))
            .collect();
        let regret = self.regret.as_ref().map(|r| {
            json!({"costs": r.costs.to_json(), "model": r.model, "always_trust": r.always_trust, "always_defer": r.always_defer, "trusted_share": r.trusted_share})
        });
        json!({
            "split": self.split.slug(),
            "examples": self.examples,
            "accuracy": self.accuracy,
            "baseline": {
                "accuracy": self.baseline.accuracy(),
                "majority_label": self.baseline.majority_label,
                "majority_accuracy": self.baseline.majority_accuracy,
                "chance_accuracy": self.baseline.chance_accuracy,
                "majority_from_train": self.baseline_from_train,
            },
            "lift": self.lift,
            "nll": self.nll,
            "brier": self.brier,
            "ece": self.ece,
            "ada_ece": self.ada_ece,
            "reliability_bins": bins,
            "coverage_accuracy": coverage,
            "order_instability": {"seed": self.order_seed, "changed_share": self.order_instability},
            "regret": regret,
            "families": families,
        })
    }

    /// A short human-readable table.
    pub fn render(&self) -> String {
        let mut out = format!(
            "{} split, {} examples\n  accuracy     {:.3}   baseline {:.3} (majority {:?} {:.3} from {}, chance {:.3})   lift {:+.3}\n  nll {:.3}   brier {:.3}   ece {:.3}   ada_ece {:.3}\n  order instability {:.3} (seed {})\n",
            self.split.slug(),
            self.examples,
            self.accuracy,
            self.baseline.accuracy(),
            self.baseline.majority_label.as_deref().unwrap_or("-"),
            self.baseline.majority_accuracy,
            if self.baseline_from_train { "train" } else { "this split" },
            self.baseline.chance_accuracy,
            self.lift,
            self.nll,
            self.brier,
            self.ece,
            self.ada_ece,
            self.order_instability,
            self.order_seed,
        );
        if let Some(r) = &self.regret {
            out.push_str(&format!(
                "  regret {:.3} per example (always trust {:.3}, always defer {:.3}; trusts {:.0}% of the time)\n",
                r.model,
                r.always_trust,
                r.always_defer,
                r.trusted_share * 100.0
            ));
        }
        out.push_str("  coverage -> accuracy: ");
        out.push_str(&self.coverage_accuracy.iter().map(|(c, a)| format!("{c:.1}->{a:.2}")).collect::<Vec<_>>().join("  "));
        out.push_str("\n  family                examples  accuracy  baseline       nll\n");
        for (name, f) in &self.families {
            out.push_str(&format!("  {name:<20}  {:>8}  {:>8.3}  {:>8.3}  {:>8.3}\n", f.examples, f.accuracy, f.baseline_accuracy, f.nll));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{synthetic_jsonl, SyntheticTask};

    /// The desk a state's problem belongs to, as the synthetic task defines it.
    fn desk_of(state: &str) -> &'static str {
        [("invoice", "billing"), ("router", "network"), ("backup", "storage"), ("mailbox", "email")]
            .iter()
            .find(|(problem, _)| state.contains(problem))
            .map(|(_, desk)| *desk)
            .expect("a synthetic state names a problem")
    }

    fn dataset(task: &SyntheticTask) -> Dataset {
        Dataset::read(synthetic_jsonl(task).as_bytes()).unwrap()
    }

    /// Puts `p_right` on the right desk and the rest evenly elsewhere, except
    /// that every `wrong_every`-th question (counting from 1) is answered with
    /// the same confidence on a wrong desk.
    fn scripted(p_right: f32, wrong_every: usize) -> impl FnMut(&str, &str, &[String]) -> Result<Vec<f32>, String> {
        let mut asked = 0usize;
        move |state, _, options| {
            asked += 1;
            let right = options.iter().position(|o| o == desk_of(state)).unwrap();
            let top = if wrong_every > 0 && asked % wrong_every == 0 { (right + 1) % options.len() } else { right };
            let rest = (1.0 - p_right) / (options.len() - 1) as f32;
            Ok((0..options.len()).map(|i| if i == top { p_right } else { rest }).collect())
        }
    }

    fn request(split: Split) -> EvalRequest {
        EvalRequest { split, order_seed: data::rng::random_seed(), costs: None }
    }

    #[test]
    fn a_perfect_model_is_perfect_and_its_lift_is_over_the_larger_baseline() {
        let d = dataset(&SyntheticTask::new(40, 0, 20));
        let r = evaluate(&mut scripted(0.97, 0), &d, &request(Split::Test)).unwrap();
        assert_eq!((r.examples, r.accuracy), (20, 1.0));
        assert!(r.baseline.accuracy() >= 0.25 - 1e-9 && r.baseline.accuracy() < 0.6, "{}", r.baseline.accuracy());
        assert!(r.baseline_from_train);
        assert!((r.lift - (1.0 - r.baseline.accuracy())).abs() < 1e-12);
        assert!(r.nll < 0.05 && r.nll > 0.0);
    }

    #[test]
    fn calibration_is_the_gap_between_confidence_and_accuracy() {
        // Confidence 0.8 on every answer and 3 of every 4 right: acc 0.75, so
        // ECE = |0.75 - 0.8| = 0.05, with every answer in one bin.
        let d = dataset(&SyntheticTask::new(8, 0, 40));
        let r = evaluate(&mut scripted(0.8, 4), &d, &request(Split::Test)).unwrap();
        assert!((r.accuracy - 0.75).abs() < 1e-9, "{}", r.accuracy);
        assert!((r.ece - 0.05).abs() < 1e-4, "{}", r.ece);
        // Equal-count bins cannot separate tied confidences, so AdaECE is only bounded here.
        assert!(r.ada_ece >= 0.0 && r.ada_ece <= 1.0, "{}", r.ada_ece);
        let populated: Vec<&Bin> = r.reliability.iter().filter(|b| b.count > 0).collect();
        assert_eq!(populated.len(), 1);
        assert_eq!(populated[0].count, 40);
        // NLL: 3/4 of answers put 0.8 on the label, 1/4 put the even remainder 0.2/3.
        let expected = 0.75 * -(0.8f64).ln() + 0.25 * -((0.2f64 / 3.0).ln());
        assert!((r.nll - expected).abs() < 1e-3, "{} vs {expected}", r.nll);
        // squared-error probability score: right answers (0.2)^2 + 3 (0.2/3)^2, wrong ones (0.8)^2 + (1 - 0.2/3)^2 + 2 (0.2/3)^2.
        let rest = 0.2f64 / 3.0;
        let right = 0.2f64.powi(2) + 3.0 * rest.powi(2);
        let wrong = 0.8f64.powi(2) + (1.0 - rest).powi(2) + 2.0 * rest.powi(2);
        assert!((r.brier - (0.75 * right + 0.25 * wrong)).abs() < 1e-3, "{}", r.brier);
    }

    #[test]
    fn coverage_accuracy_rises_when_confidence_ranks_the_mistakes_last() {
        // A model that is sure when it is right and unsure when it is wrong.
        let d = dataset(&SyntheticTask::new(8, 0, 20));
        let mut asked = 0;
        let mut model = |state: &str, _: &str, options: &[String]| -> Result<Vec<f32>, String> {
            asked += 1;
            let right = options.iter().position(|o| o == desk_of(state)).unwrap();
            let wrong = asked % 5 == 0;
            let (top, p) = if wrong { ((right + 1) % 4, 0.4) } else { (right, 0.9) };
            Ok((0..4).map(|i| if i == top { p } else { (1.0 - p) / 3.0 }).collect())
        };
        let r = evaluate(&mut model, &d, &request(Split::Test)).unwrap();
        let first = r.coverage_accuracy.first().unwrap();
        let last = r.coverage_accuracy.last().unwrap();
        assert_eq!((first.1, last.0), (1.0, 1.0));
        assert!(last.1 < first.1, "full-coverage accuracy {} must be below top-tenth accuracy {}", last.1, first.1);
    }

    #[test]
    fn a_model_that_answers_by_position_is_unstable_and_one_that_reads_options_is_not() {
        let d = dataset(&SyntheticTask::new(8, 0, 60));
        let stable = evaluate(&mut scripted(0.9, 0), &d, &request(Split::Test)).unwrap();
        assert_eq!(stable.order_instability, 0.0);
        let mut by_position = |_: &str, _: &str, options: &[String]| -> Result<Vec<f32>, String> {
            Ok((0..options.len()).map(|i| if i == 0 { 0.7 } else { 0.1 }).collect())
        };
        let r = evaluate(&mut by_position, &d, &request(Split::Test)).unwrap();
        // The first option's id is the winner when asked in order; after a random
        // shuffle it wins again with probability 1/4, so about 3/4 change.
        assert!(r.order_instability > 0.5 && r.order_instability < 0.95, "{}", r.order_instability);
        let again = evaluate(&mut by_position, &d, &request(Split::Test)).unwrap();
        assert_eq!(r.order_instability, again.order_instability, "a seed fixes the permutations");
    }

    #[test]
    fn regret_is_the_realized_cost_of_trusting_or_deferring_on_the_models_own_confidence() {
        let d = dataset(&SyntheticTask::new(8, 0, 40));
        let costs = CostSpec { false_exclusion: 5.0, wasted_tokens: 1.0 };
        let req = EvalRequest { costs: Some(costs), ..request(Split::Test) };
        // Confident (0.9) and 3 of 4 right. Trusting costs 5 * P(wrong) = 0.5 < deferring (1 * 0.9),
        // so the model trusts every time: realized cost = 5 * 1/4 = 1.25.
        let r = evaluate(&mut scripted(0.9, 4), &d, &req).unwrap().regret.unwrap();
        assert!((r.model - 1.25).abs() < 1e-9 && (r.always_trust - 1.25).abs() < 1e-9, "{r:?}");
        assert!((r.always_defer - 0.75).abs() < 1e-9);
        assert_eq!(r.trusted_share, 1.0);
        // Unsure (0.3): trusting costs 5 * 0.7 = 3.5 > deferring 0.3, so it always defers: cost 1 * 3/4.
        let r = evaluate(&mut scripted(0.3, 4), &d, &req).unwrap().regret.unwrap();
        assert!((r.model - 0.75).abs() < 1e-9 && r.trusted_share == 0.0, "{r:?}");
        assert!(evaluate(&mut scripted(0.9, 4), &d, &request(Split::Test)).unwrap().regret.is_none());
    }

    #[test]
    fn families_are_reported_separately() {
        let mut lines = synthetic_jsonl(&SyntheticTask::new(4, 0, 6));
        let mut other = SyntheticTask::new(0, 0, 6);
        other.family = "other";
        other.seed = data::rng::random_seed();
        // Unique tasks, seeds and states for the second family.
        let other_lines = synthetic_jsonl(&other).replace("task-", "o-task-").replace("\"seed\":", "\"seed\":1000").replace("ticket ", "other ticket ");
        lines.push_str(&other_lines);
        let d = Dataset::read(lines.as_bytes()).unwrap();
        let r = evaluate(&mut scripted(0.9, 0), &d, &request(Split::Test)).unwrap();
        assert_eq!(r.families.keys().collect::<Vec<_>>(), ["desks", "other"]);
        assert_eq!(r.families["desks"].examples + r.families["other"].examples, r.examples);
    }

    #[test]
    fn a_missing_split_or_a_malformed_distribution_is_an_error_naming_the_line() {
        let d = dataset(&SyntheticTask::new(4, 0, 4));
        assert!(evaluate(&mut scripted(0.9, 0), &d, &request(Split::Validation)).unwrap_err().contains("validation"));
        let mut short = |_: &str, _: &str, _: &[String]| Ok(vec![1.0]);
        assert!(evaluate(&mut short, &d, &request(Split::Test)).unwrap_err().contains("1 probabilities for 4 options"));
        let mut nan = |_: &str, _: &str, o: &[String]| Ok(vec![f32::NAN; o.len()]);
        assert!(evaluate(&mut nan, &d, &request(Split::Test)).unwrap_err().contains("not finite"));
        let mut unnormalised = |_: &str, _: &str, o: &[String]| Ok(vec![0.5; o.len()]);
        assert!(evaluate(&mut unnormalised, &d, &request(Split::Test)).unwrap_err().contains("sum to"));
        let mut failing = |_: &str, _: &str, _: &[String]| Err("device lost".to_string());
        assert!(evaluate(&mut failing, &d, &request(Split::Test)).unwrap_err().contains("device lost"));
    }

    #[test]
    fn the_report_is_machine_readable_and_renders_a_table() {
        let d = dataset(&SyntheticTask::new(8, 0, 12));
        let req = EvalRequest { costs: Some(CostSpec { false_exclusion: 5.0, wasted_tokens: 1.0 }), ..request(Split::Test) };
        let r = evaluate(&mut scripted(0.9, 4), &d, &req).unwrap();
        let j = r.to_json();
        for key in ["accuracy", "baseline", "lift", "nll", "brier", "ece", "ada_ece", "reliability_bins", "coverage_accuracy", "order_instability", "regret", "families"] {
            assert!(!j[key].is_null(), "{key} missing: {j}");
        }
        assert_eq!(j["reliability_bins"].as_array().unwrap().len(), CALIBRATION_BINS);
        let table = r.render();
        assert!(table.contains("accuracy") && table.contains("regret") && table.contains("desks"), "{table}");
    }

    #[test]
    fn a_cost_file_is_validated() {
        assert_eq!(CostSpec::parse(r#"{"false_exclusion": 5, "wasted_tokens": 1.5}"#).unwrap(), CostSpec { false_exclusion: 5.0, wasted_tokens: 1.5 });
        for (bad, needle) in [
            ("nope", "valid JSON"),
            ("[]", "object"),
            (r#"{"false_exclusion": 5}"#, "wasted_tokens"),
            (r#"{"false_exclusion": -1, "wasted_tokens": 1}"#, "at least 0"),
            (r#"{"false_exclusion": 0, "wasted_tokens": 0}"#, "something"),
            (r#"{"false_exclusion": 1, "wasted_tokens": 1, "extra": 2}"#, "extra"),
        ] {
            let e = CostSpec::parse(bad).unwrap_err();
            assert!(e.contains(needle), "{bad}: {e}");
        }
    }
}
