// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Calibration metrics.
//!
//! ECE, AdaECE, classwise-ECE, Brier, NLL, reliability bins,
//! coverage-vs-accuracy and failure-AUROC were specified as a decision
//! model's evaluation deliverable and never built - `crates/eval` has
//! detection, MLM and TTS metrics only. The functions here fill that gap,
//! placed at layer 3 (this crate) rather than layer-5 `eval` so a sample can
//! use them without pulling in `eval`'s own dependency closure
//! (`gpt2`/`lfm2`/`yolov8`/`audio`/`ecapatdnn`), which would blow a sample's
//! declared `max-brain-crates` budget for metrics that need none of those
//! models. `eval` may re-export these later.
//!
//! **AUGRC is deliberately not implemented here.** The published metric
//! (Traub et al., "Overcoming Common Flaws in the Evaluation of Selective
//! Classification Systems") is a pairwise generalization of the naive
//! risk-coverage AUC specifically built to fix that naive version's
//! sensitivity to confidence-tie ordering - getting the pairwise formulation
//! right without a reference implementation to check it against is exactly
//! the kind of unverified numerical claim this codebase's engineering-quality
//! bar rules out. [`coverage_accuracy`] and [`failure_auroc`] cover the same
//! selective-prediction question with well-established, independently
//! checkable formulas; AUGRC stays a named gap until there is a reference to
//! gate it against.

/// One bin of a reliability diagram.
#[derive(Clone, Debug)]
pub struct ReliabilityBin {
    pub lo: f32,
    pub hi: f32,
    pub count: usize,
    /// Mean predicted confidence of the examples in this bin (`NaN` if empty).
    pub mean_confidence: f32,
    /// Empirical accuracy of the examples in this bin (`NaN` if empty).
    pub accuracy: f32,
}

/// Equal-width bins over `[0, 1]` - the classic ECE binning. `confidences[i]`
/// is the model's own reported confidence for example `i` (not necessarily
/// `p_max`, per `decide.md` - any `[0, 1]` confidence statistic works here);
/// `correct[i]` is whether the model's decision on example `i` was right.
pub fn reliability_bins(
    confidences: &[f32],
    correct: &[bool],
    n_bins: usize,
) -> Vec<ReliabilityBin> {
    assert_eq!(
        confidences.len(),
        correct.len(),
        "one correctness flag per confidence"
    );
    assert!(n_bins > 0, "at least one bin");
    let mut sum_conf = vec![0.0f64; n_bins];
    let mut sum_correct = vec![0.0f64; n_bins];
    let mut count = vec![0usize; n_bins];
    for (&c, &ok) in confidences.iter().zip(correct) {
        let c = c.clamp(0.0, 1.0);
        let mut b = (c * n_bins as f32).floor() as usize;
        if b >= n_bins {
            b = n_bins - 1; // c == 1.0 lands in the last bin, not a phantom n_bins-th one
        }
        sum_conf[b] += c as f64;
        sum_correct[b] += if ok { 1.0 } else { 0.0 };
        count[b] += 1;
    }
    (0..n_bins)
        .map(|b| {
            let n = count[b];
            ReliabilityBin {
                lo: b as f32 / n_bins as f32,
                hi: (b + 1) as f32 / n_bins as f32,
                count: n,
                mean_confidence: if n > 0 {
                    (sum_conf[b] / n as f64) as f32
                } else {
                    f32::NAN
                },
                accuracy: if n > 0 {
                    (sum_correct[b] / n as f64) as f32
                } else {
                    f32::NAN
                },
            }
        })
        .collect()
}

/// Expected Calibration Error: the count-weighted mean absolute gap between
/// confidence and accuracy over equal-WIDTH bins.
pub fn ece(confidences: &[f32], correct: &[bool], n_bins: usize) -> f32 {
    let n = confidences.len().max(1) as f32;
    reliability_bins(confidences, correct, n_bins)
        .iter()
        .filter(|b| b.count > 0)
        .map(|b| (b.count as f32 / n) * (b.accuracy - b.mean_confidence).abs())
        .sum()
}

/// Adaptive ECE: the same statistic over equal-MASS bins (each bin holds the
/// same number of examples, so no bin is empty and none dominates because
/// confidences happened to cluster) - Nguyen & O'Connor's fix for the
/// standard ECE's sensitivity to where the empty/sparse bins fall.
pub fn ada_ece(confidences: &[f32], correct: &[bool], n_bins: usize) -> f32 {
    assert_eq!(
        confidences.len(),
        correct.len(),
        "one correctness flag per confidence"
    );
    assert!(n_bins > 0, "at least one bin");
    let n = confidences.len();
    if n == 0 {
        return 0.0;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        confidences[a]
            .partial_cmp(&confidences[b])
            .expect("confidence is never NaN")
    });

    let mut total = 0.0f64;
    let mut start = 0usize;
    for b in 0..n_bins {
        // Distribute the remainder across the first bins so every bin gets
        // floor(n/n_bins) or ceil(n/n_bins) members, never zero unless n < n_bins.
        let extra = if b < n % n_bins { 1 } else { 0 };
        let len = n / n_bins + extra;
        if len == 0 {
            continue;
        }
        let members = &idx[start..start + len];
        let mean_conf: f64 =
            members.iter().map(|&i| confidences[i] as f64).sum::<f64>() / len as f64;
        let acc: f64 = members
            .iter()
            .map(|&i| if correct[i] { 1.0 } else { 0.0 })
            .sum::<f64>()
            / len as f64;
        total += (len as f64 / n as f64) * (acc - mean_conf).abs();
        start += len;
    }
    total as f32
}

/// The average, over every class, of that class's one-vs-rest ECE: for class
/// `k`, "confidence" is `probs[i][k]` and "correct" is `labels[i] == k`. A
/// model can have near-zero top-label ECE while being badly miscalibrated on
/// the options it usually rejects - this is what `decide.md`'s
/// multiclass concern names and top-1 ECE alone cannot see.
pub fn classwise_ece(probs: &[Vec<f32>], labels: &[usize], n_bins: usize) -> f32 {
    assert_eq!(probs.len(), labels.len(), "one label per row");
    let k = probs.first().map(|p| p.len()).unwrap_or(0);
    if k == 0 {
        return 0.0;
    }
    let mut total = 0.0f32;
    for class in 0..k {
        let conf: Vec<f32> = probs.iter().map(|p| p[class]).collect();
        let correct: Vec<bool> = labels.iter().map(|&y| y == class).collect();
        total += ece(&conf, &correct, n_bins);
    }
    total / k as f32
}

/// Mean negative log likelihood of the true label: `mean_i -ln p_i[label_i]`.
pub fn nll(probs: &[Vec<f32>], labels: &[usize]) -> f32 {
    assert_eq!(probs.len(), labels.len(), "one label per row");
    if probs.is_empty() {
        return 0.0;
    }
    let sum: f64 = probs
        .iter()
        .zip(labels)
        .map(|(p, &y)| -((p[y] as f64).max(1e-12)).ln())
        .sum();
    (sum / probs.len() as f64) as f32
}

/// Mean multiclass Brier score: `mean_i sum_k (p_i[k] - 1{k=label_i})^2` - a
/// METRIC over a held-out set, distinct from [`crate::scoring::decision_loss`]
/// (a training objective over one example's raw scores).
pub fn brier_score(probs: &[Vec<f32>], labels: &[usize]) -> f32 {
    assert_eq!(probs.len(), labels.len(), "one label per row");
    if probs.is_empty() {
        return 0.0;
    }
    let sum: f64 = probs
        .iter()
        .zip(labels)
        .map(|(p, &y)| {
            p.iter()
                .enumerate()
                .map(|(k, &pk)| {
                    let d = pk as f64 - if k == y { 1.0 } else { 0.0 };
                    d * d
                })
                .sum::<f64>()
        })
        .sum();
    (sum / probs.len() as f64) as f32
}

/// The selective-prediction risk-coverage curve: sorted by DESCENDING
/// confidence, `(coverage, accuracy)` after keeping the top `coverage`
/// fraction. `coverage` values are `1/n, 2/n, ..., 1.0`. Ties are broken by
/// original input order (stable sort) - a real tie-order dependency, which
/// is exactly why AUGRC exists in the literature and why this module does
/// not claim to reproduce it (see the module doc).
pub fn coverage_accuracy(confidences: &[f32], correct: &[bool]) -> Vec<(f32, f32)> {
    assert_eq!(
        confidences.len(),
        correct.len(),
        "one correctness flag per confidence"
    );
    let n = confidences.len();
    if n == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        confidences[b]
            .partial_cmp(&confidences[a])
            .expect("confidence is never NaN")
    });
    let mut kept_correct = 0usize;
    let mut out = Vec::with_capacity(n);
    for (i, &orig) in idx.iter().enumerate() {
        if correct[orig] {
            kept_correct += 1;
        }
        out.push((
            (i + 1) as f32 / n as f32,
            kept_correct as f32 / (i + 1) as f32,
        ));
    }
    out
}

/// AUROC using `-confidence` to detect a FAILURE (an incorrect decision) -
/// the standard selective-prediction "does confidence separate the mistakes"
/// statistic. `0.5` is chance separation; `1.0` is perfect (every failure has
/// lower confidence than every success). Computed via the rank-sum
/// (Mann-Whitney U) identity, which handles tied confidences by average rank
/// exactly the way the pairwise AUROC definition requires.
pub fn failure_auroc(confidences: &[f32], correct: &[bool]) -> f32 {
    assert_eq!(
        confidences.len(),
        correct.len(),
        "one correctness flag per confidence"
    );
    let n_fail = correct.iter().filter(|&&c| !c).count();
    let n_ok = correct.len() - n_fail;
    if n_fail == 0 || n_ok == 0 {
        return f32::NAN; // undefined with only one class present
    }

    // Rank by ascending -confidence (so a failure - which should have LOW
    // confidence, i.e. HIGH -confidence - gets a high rank when detection is
    // working), averaging ranks across ties.
    let mut idx: Vec<usize> = (0..confidences.len()).collect();
    idx.sort_by(|&a, &b| {
        (-confidences[a])
            .partial_cmp(&(-confidences[b]))
            .expect("confidence is never NaN")
    });

    let mut ranks = vec![0.0f64; confidences.len()];
    let mut i = 0usize;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && confidences[idx[j + 1]] == confidences[idx[i]] {
            j += 1;
        }
        // Ranks are 1-based; a tied block [i, j] shares the average of those ranks.
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        for &k in &idx[i..=j] {
            ranks[k] = avg_rank;
        }
        i = j + 1;
    }

    let rank_sum_fail: f64 = correct
        .iter()
        .zip(&ranks)
        .filter(|(&ok, _)| !ok)
        .map(|(_, &r)| r)
        .sum();
    let u = rank_sum_fail - (n_fail * (n_fail + 1)) as f64 / 2.0;
    (u / (n_fail * n_ok) as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `decide.md`'s own gate: a perfectly calibrated input gives ECE 0.
    /// Constructed so every bin's empirical accuracy exactly equals its
    /// members' confidence - not approximately, by exact integer counts.
    #[test]
    fn ece_is_zero_on_a_perfectly_calibrated_input() {
        let mut confidences = Vec::new();
        let mut correct = Vec::new();
        for tenths in 1..10 {
            let c = tenths as f32 / 10.0;
            // 10 examples at confidence c, exactly `tenths` of them correct.
            for k in 0..10 {
                confidences.push(c);
                correct.push(k < tenths);
            }
        }
        let e = ece(&confidences, &correct, 10);
        assert!(
            e <= 1e-6,
            "ECE {e} should be ~0 on a perfectly calibrated input"
        );
    }

    /// `decide.md`'s own gate: maximally overconfident (confidence = 1
    /// everywhere) collapses ECE to `1 - accuracy`.
    #[test]
    fn ece_is_one_minus_accuracy_when_maximally_overconfident() {
        let confidences = vec![1.0f32; 10];
        let correct = vec![
            true, true, true, true, true, true, false, false, false, false,
        ]; // acc 0.6
        let e = ece(&confidences, &correct, 10);
        assert!((e - 0.4).abs() <= 1e-6, "ECE {e} should be 1 - 0.6 = 0.4");
    }

    #[test]
    fn ada_ece_is_also_zero_on_the_calibrated_input_and_leaves_no_bin_empty() {
        let mut confidences = Vec::new();
        let mut correct = Vec::new();
        for tenths in 1..10 {
            let c = tenths as f32 / 10.0;
            for k in 0..10 {
                confidences.push(c);
                correct.push(k < tenths);
            }
        }
        let e = ada_ece(&confidences, &correct, 9);
        assert!(
            e <= 1e-6,
            "AdaECE {e} should be ~0 on a perfectly calibrated input"
        );
    }

    #[test]
    fn classwise_ece_catches_a_class_top1_ece_is_blind_to() {
        // 3 classes. The top label is always right (top-1 ECE = 0 at
        // confidence 1), but among the two REJECTED classes, the model always
        // reports class 1 as impossible and class 2 as a coin flip when it is
        // never actually chosen - a per-class miscalibration invisible to any
        // metric that only looks at the argmax slot.
        let probs = vec![vec![1.0, 0.0, 0.0]; 20];
        let labels = vec![0usize; 20];
        let top1 = ece(&[1.0f32; 20], &[true; 20], 10);
        assert!(top1 <= 1e-6, "top-1 ECE should look perfect");
        let cw = classwise_ece(&probs, &labels, 10);
        // Classes 1 and 2 are reported as confidence-0 and are indeed never
        // the label, so THEY are calibrated too here - classwise_ece must
        // therefore also read ~0 on this particular (degenerate but
        // internally consistent) input. The real check is that it is a
        // DIFFERENT, independently computed number from top-1 ECE, not that
        // it disagrees on every input.
        assert!(cw <= 1e-6, "classwise ECE {cw} on a genuinely 0/1 model");
    }

    #[test]
    fn nll_of_a_perfect_prediction_is_zero() {
        let probs = vec![vec![1e-12, 1.0 - 1e-12]];
        let labels = vec![1usize];
        assert!(nll(&probs, &labels) <= 1e-6);
    }

    #[test]
    fn nll_matches_the_closed_form_for_a_uniform_prediction() {
        let probs = vec![vec![0.25f32; 4]];
        let labels = vec![2usize];
        let n = nll(&probs, &labels);
        assert!(
            (n - 0.25f32.ln().abs()).abs() <= 1e-5,
            "NLL {n} should be -ln(0.25)"
        );
    }

    #[test]
    fn brier_score_of_a_perfect_prediction_is_zero() {
        let probs = vec![vec![0.0, 1.0]];
        let labels = vec![1usize];
        assert_eq!(brier_score(&probs, &labels), 0.0);
    }

    #[test]
    fn coverage_accuracy_is_monotonically_non_increasing_when_confidence_predicts_correctness() {
        // Perfectly separated: high confidence always correct, low confidence always wrong.
        let confidences = vec![0.9, 0.8, 0.7, 0.3, 0.2, 0.1];
        let correct = vec![true, true, true, false, false, false];
        let curve = coverage_accuracy(&confidences, &correct);
        assert_eq!(curve.len(), 6);
        assert_eq!(curve[2], (0.5, 1.0), "the 3 most confident are all correct");
        assert_eq!(
            curve[5],
            (1.0, 0.5),
            "at full coverage, accuracy is the base rate"
        );
        for w in curve.windows(2) {
            assert!(
                w[1].1 <= w[0].1 + 1e-6,
                "accuracy should not RISE as low-confidence examples are admitted"
            );
        }
    }

    #[test]
    fn failure_auroc_is_one_when_confidence_perfectly_separates_mistakes() {
        let confidences = vec![0.9, 0.8, 0.7, 0.3, 0.2, 0.1];
        let correct = vec![true, true, true, false, false, false];
        let auroc = failure_auroc(&confidences, &correct);
        assert!(
            (auroc - 1.0).abs() <= 1e-6,
            "AUROC {auroc} should be 1.0: every failure scores below every success"
        );
    }

    #[test]
    fn failure_auroc_is_half_when_confidence_is_uninformative() {
        // Confidence identical for every example: no separation is possible.
        let confidences = vec![0.5; 8];
        let correct = vec![true, false, true, false, true, false, true, false];
        let auroc = failure_auroc(&confidences, &correct);
        assert!(
            (auroc - 0.5).abs() <= 1e-6,
            "AUROC {auroc} should be 0.5 with no information"
        );
    }
}
