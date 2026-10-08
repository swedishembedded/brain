// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Evaluation of a probability for a binary outcome observed in full (a
//! condition present or not at the examination, not a time to an event).
//!
//! Swedish Embedded AB implements validation of risk and screening models
//! for its clients. If your team needs expertise in deciding whether a score
//! is good enough to screen with, and saying so honestly, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! - [`auroc`] and [`average_precision`]: discrimination, weighted.
//! - [`brier`], [`recalibration`], [`risk_groups`]: calibration.
//! - [`threshold_for_specificity`], [`rates_at`], [`predictive_values`]:
//!   screening operating points. A threshold is chosen on one set of subjects
//!   and applied to another, and predictive values are restated at the
//!   prevalence of the population the screen is meant for.
//!
//! Weights are sampling weights: a subject of integer weight `w` contributes
//! as `w` copies would. A score at or above the threshold is a positive call.
//! Every function takes parallel slices and panics if their lengths differ or
//! if an outcome class is empty where the quantity needs both.

fn check(score: &[f64], label: &[bool], weight: &[f64]) {
    assert!(
        score.len() == label.len() && label.len() == weight.len(),
        "one score, label and weight per subject"
    );
}

fn totals(label: &[bool], weight: &[f64]) -> (f64, f64) {
    let pos: f64 = label
        .iter()
        .zip(weight)
        .filter(|(l, _)| **l)
        .map(|(_, w)| w)
        .sum();
    let neg: f64 = label
        .iter()
        .zip(weight)
        .filter(|(l, _)| !**l)
        .map(|(_, w)| w)
        .sum();
    assert!(pos > 0.0 && neg > 0.0, "both outcome classes need weight");
    (pos, neg)
}

/// Indices ordered by ascending score.
fn ascending(score: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..score.len()).collect();
    order.sort_by(|&a, &b| score[a].partial_cmp(&score[b]).expect("finite scores"));
    order
}

/// Runs of equal score over `order`, as `(positive weight, negative weight)`.
fn tie_groups(
    score: &[f64],
    label: &[bool],
    weight: &[f64],
    order: &[usize],
) -> Vec<(f64, f64, f64)> {
    let mut out: Vec<(f64, f64, f64)> = Vec::new();
    for &i in order {
        match out.last_mut() {
            Some(g) if g.0 == score[i] => {
                if label[i] {
                    g.1 += weight[i]
                } else {
                    g.2 += weight[i]
                }
            }
            _ => out.push((
                score[i],
                if label[i] { weight[i] } else { 0.0 },
                if label[i] { 0.0 } else { weight[i] },
            )),
        }
    }
    out
}

/// Area under the ROC curve (Mann-Whitney; a tie counts one half).
pub fn auroc(score: &[f64], label: &[bool], weight: &[f64]) -> f64 {
    check(score, label, weight);
    let (pos, neg) = totals(label, weight);
    let groups = tie_groups(score, label, weight, &ascending(score));
    let mut below_neg = 0.0;
    let mut wins = 0.0;
    for (_, p, n) in groups {
        wins += p * (below_neg + 0.5 * n);
        below_neg += n;
    }
    wins / (pos * neg)
}

/// Average precision: the step-wise area under the precision-recall curve,
/// one step per distinct score (scikit-learn's definition).
pub fn average_precision(score: &[f64], label: &[bool], weight: &[f64]) -> f64 {
    check(score, label, weight);
    let (pos, _) = totals(label, weight);
    let mut order = ascending(score);
    order.reverse();
    let (mut tp, mut fp, mut prev_recall, mut ap) = (0.0, 0.0, 0.0, 0.0);
    let mut groups = tie_groups(score, label, weight, &order);
    groups.sort_by(|a, b| b.0.partial_cmp(&a.0).expect("finite scores"));
    for (_, p, n) in groups {
        tp += p;
        fp += n;
        let recall = tp / pos;
        ap += (recall - prev_recall) * tp / (tp + fp);
        prev_recall = recall;
    }
    ap
}

/// Weighted mean squared difference between probability and outcome.
pub fn brier(prob: &[f64], label: &[bool], weight: &[f64]) -> f64 {
    check(prob, label, weight);
    let total: f64 = weight.iter().sum();
    prob.iter()
        .zip(label)
        .zip(weight)
        .map(|((p, l), w)| w * (p - f64::from(u8::from(*l))).powi(2))
        .sum::<f64>()
        / total
}

fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-9, 1.0 - 1e-9);
    (p / (1.0 - p)).ln()
}

/// Logistic recalibration `logit P(y) = a + b * logit(p)` by weighted
/// iteratively reweighted least squares; returns `(intercept, slope)`. A
/// perfectly calibrated probability gives `(0, 1)`; a slope below one means
/// the probabilities are too extreme.
pub fn recalibration(prob: &[f64], label: &[bool], weight: &[f64]) -> (f64, f64) {
    check(prob, label, weight);
    totals(label, weight);
    let x: Vec<f64> = prob.iter().map(|&p| logit(p)).collect();
    let (mut a, mut b) = (0.0_f64, 1.0_f64);
    for _ in 0..50 {
        let (mut h00, mut h01, mut h11, mut g0, mut g1) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..x.len() {
            let mu = 1.0 / (1.0 + (-(a + b * x[i])).exp());
            let y = f64::from(u8::from(label[i]));
            let v = weight[i] * mu * (1.0 - mu);
            h00 += v;
            h01 += v * x[i];
            h11 += v * x[i] * x[i];
            g0 += weight[i] * (y - mu);
            g1 += weight[i] * (y - mu) * x[i];
        }
        let det = h00 * h11 - h01 * h01;
        if det.abs() < 1e-300 {
            break;
        }
        let da = (h11 * g0 - h01 * g1) / det;
        let db = (h00 * g1 - h01 * g0) / det;
        a += da;
        b += db;
        if da.abs() < 1e-10 && db.abs() < 1e-10 {
            break;
        }
    }
    (a, b)
}

/// One equal-weight risk group.
#[derive(Clone, Debug, PartialEq)]
pub struct RiskGroup {
    pub weight: f64,
    /// Weighted mean predicted probability.
    pub expected: f64,
    /// Weighted share with the outcome.
    pub observed: f64,
}

/// Subjects ordered by probability and cut into `groups` equal-weight groups
/// (a subject goes to the group its weight midpoint falls in).
pub fn risk_groups(prob: &[f64], label: &[bool], weight: &[f64], groups: usize) -> Vec<RiskGroup> {
    check(prob, label, weight);
    assert!(groups >= 1, "at least one group");
    let total: f64 = weight.iter().sum();
    let mut out: Vec<(f64, f64, f64)> = vec![(0.0, 0.0, 0.0); groups];
    let mut seen = 0.0;
    for i in ascending(prob) {
        let g =
            ((((seen + weight[i] / 2.0) / total) * groups as f64).floor() as usize).min(groups - 1);
        seen += weight[i];
        out[g].0 += weight[i];
        out[g].1 += weight[i] * prob[i];
        out[g].2 += weight[i] * f64::from(u8::from(label[i]));
    }
    out.into_iter()
        .filter(|g| g.0 > 0.0)
        .map(|(w, e, o)| RiskGroup {
            weight: w,
            expected: e / w,
            observed: o / w,
        })
        .collect()
}

/// Sensitivity and specificity of the call `score >= threshold`.
pub fn rates_at(score: &[f64], label: &[bool], weight: &[f64], threshold: f64) -> (f64, f64) {
    check(score, label, weight);
    let (pos, neg) = totals(label, weight);
    let (mut tp, mut tn) = (0.0, 0.0);
    for i in 0..score.len() {
        match (label[i], score[i] >= threshold) {
            (true, true) => tp += weight[i],
            (false, false) => tn += weight[i],
            _ => {}
        }
    }
    (tp / pos, tn / neg)
}

/// The lowest threshold whose specificity is at least `specificity`, so that
/// it catches as many cases as that specificity allows. Returns a score that
/// occurs in the data, or infinity if only a call of nobody reaches it.
pub fn threshold_for_specificity(
    score: &[f64],
    label: &[bool],
    weight: &[f64],
    specificity: f64,
) -> f64 {
    check(score, label, weight);
    assert!(
        (0.0..=1.0).contains(&specificity),
        "a specificity is a share"
    );
    let (_, neg) = totals(label, weight);
    let groups = tie_groups(score, label, weight, &ascending(score));
    let mut below_neg = 0.0;
    for (s, _, n) in groups {
        // Calling positive from `s` upward leaves every negative below `s` correct.
        if below_neg / neg >= specificity {
            return s;
        }
        below_neg += n;
    }
    f64::INFINITY
}

/// Positive and negative predictive value of a screen with the given
/// sensitivity and specificity in a population with the given `prevalence`
/// (Bayes' rule). Returns `(ppv, npv)`.
pub fn predictive_values(sensitivity: f64, specificity: f64, prevalence: f64) -> (f64, f64) {
    assert!((0.0..=1.0).contains(&prevalence), "a prevalence is a share");
    let tp = sensitivity * prevalence;
    let fp = (1.0 - specificity) * (1.0 - prevalence);
    let tn = specificity * (1.0 - prevalence);
    let fn_ = (1.0 - sensitivity) * prevalence;
    (tp / (tp + fp), tn / (tn + fn_))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from scikit-learn 1.9.1 (roc_auc_score, average_precision_score,
    // brier_score_loss, all with sample_weight; the recalibration line is the exact maximum
    // likelihood from scipy BFGS, since scikit-learn's solver stops 1e-4 short).
    const S: [f64; 8] = [0.1, 0.4, 0.35, 0.8, 0.4, 0.7, 0.2, 0.9];
    const Y: [bool; 8] = [false, false, true, true, true, false, false, true];
    const W: [f64; 8] = [1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0];

    #[test]
    fn matches_scikit_learn_with_ties_and_weights() {
        assert!((auroc(&S, &Y, &W) - 0.861111111111111).abs() < 1e-12);
        assert!((average_precision(&S, &Y, &W) - 0.8819444444444444).abs() < 1e-12);
        assert!((brier(&S, &Y, &W) - 0.15104166666666666).abs() < 1e-12);
        let (a, b) = recalibration(&S, &Y, &W);
        assert!(
            (a + 0.01233228).abs() < 1e-7 && (b - 1.40706213).abs() < 1e-7,
            "{a} {b}"
        );
    }

    #[test]
    fn integer_weights_equal_copies() {
        let (mut s, mut y) = (Vec::new(), Vec::new());
        for i in 0..8 {
            for _ in 0..W[i] as usize {
                s.push(S[i]);
                y.push(Y[i]);
            }
        }
        let ones = vec![1.0; s.len()];
        assert!((auroc(&s, &y, &ones) - auroc(&S, &Y, &W)).abs() < 1e-12);
        assert!((average_precision(&s, &y, &ones) - average_precision(&S, &Y, &W)).abs() < 1e-12);
        let (a1, b1) = recalibration(&s, &y, &ones);
        let (a2, b2) = recalibration(&S, &Y, &W);
        assert!((a1 - a2).abs() < 1e-8 && (b1 - b2).abs() < 1e-8);
    }

    #[test]
    fn a_reversed_score_mirrors_the_auroc_and_a_perfect_one_scores_one() {
        let r: Vec<f64> = S.iter().map(|s| -s).collect();
        assert!((auroc(&r, &Y, &W) - (1.0 - auroc(&S, &Y, &W))).abs() < 1e-12);
        let perfect: Vec<f64> = Y.iter().map(|&y| if y { 0.9 } else { 0.1 }).collect();
        assert_eq!(auroc(&perfect, &Y, &W), 1.0);
        assert_eq!(average_precision(&perfect, &Y, &W), 1.0);
        assert_eq!(auroc(&[0.5; 8], &Y, &W), 0.5);
    }

    #[test]
    fn risk_groups_split_by_weight_and_report_observed_shares() {
        let g = risk_groups(
            &[0.1, 0.2, 0.8, 0.9],
            &[false, false, true, false],
            &[1.0; 4],
            2,
        );
        assert_eq!(g.len(), 2);
        assert!((g[0].expected - 0.15).abs() < 1e-12 && g[0].observed == 0.0);
        assert!((g[1].expected - 0.85).abs() < 1e-12 && (g[1].observed - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_threshold_is_chosen_for_a_specificity_and_applied_elsewhere() {
        // Negatives score 0.1..0.5, positives 0.4..0.8.
        let s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.8];
        let y = [
            false, false, false, false, false, true, true, true, true, true,
        ];
        let w = [1.0; 10];
        // 80% specificity: four of five negatives below the threshold -> threshold 0.5.
        let t = threshold_for_specificity(&s, &y, &w, 0.8);
        assert_eq!(t, 0.5);
        assert_eq!(rates_at(&s, &y, &w, t), (0.8, 0.8));
        // 100% specificity needs a threshold above every negative.
        assert_eq!(threshold_for_specificity(&s, &y, &w, 1.0), 0.6);
        // A negative at the very top leaves only calling nobody.
        assert_eq!(
            threshold_for_specificity(&[0.1, 0.9], &[true, false], &[1.0, 1.0], 1.0),
            f64::INFINITY
        );
        // The threshold carries to other subjects.
        assert_eq!(
            rates_at(&[0.45, 0.55], &[true, false], &[1.0, 1.0], t),
            (0.0, 0.0)
        );
    }

    #[test]
    fn predictive_values_follow_bayes_rule() {
        // 90% sensitive, 90% specific, 1% prevalence: PPV = 0.009 / (0.009 + 0.099).
        let (ppv, npv) = predictive_values(0.9, 0.9, 0.01);
        assert!((ppv - 0.009 / 0.108).abs() < 1e-12);
        assert!((npv - 0.891 / 0.892).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "both outcome classes")]
    fn a_class_without_weight_is_refused() {
        auroc(&[0.1, 0.2], &[true, true], &[1.0, 1.0]);
    }
}
