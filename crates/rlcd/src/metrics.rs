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

/// Mean cross-entropy against the oracle's own DISTRIBUTION:
/// `mean_n -sum_i q_n[i] * ln p_n[i]`.
///
/// The soft-target counterpart of [`nll`], for a task whose targets are
/// exact posteriors rather than realized labels. Collapsing such a target to
/// `argmax` and scoring [`nll`] against it does not measure a weaker version
/// of the same thing - it measures a DIFFERENT thing, and rewards the model
/// for moving toward `[0, 1]` when the world says `[1/3, 2/3]`.
///
/// **Its floor is not zero.** A model that reproduces `q` exactly scores the
/// mean entropy of `q`, which is the irreducible uncertainty the world
/// actually has. Compare a run against that floor, never against 0; pair it
/// with [`posterior_kl`], whose floor IS zero, when the excess over the
/// oracle is the quantity of interest.
pub fn soft_nll(probs: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
    assert_eq!(probs.len(), targets.len(), "one target distribution per row");
    if probs.is_empty() {
        return 0.0;
    }
    let sum: f64 = probs
        .iter()
        .zip(targets)
        .map(|(p, q)| {
            assert_eq!(p.len(), q.len(), "a row and its target must have the same arity");
            q.iter()
                .zip(p)
                .map(|(&qi, &pi)| -(qi as f64) * (pi as f64).max(1e-12).ln())
                .sum::<f64>()
        })
        .sum();
    (sum / probs.len() as f64) as f32
}

/// Mean squared distance between the model's posterior and the oracle's:
/// `mean_n sum_i (p_n[i] - q_n[i])^2`.
///
/// The soft-target counterpart of [`brier_score`], and zero exactly when the
/// model reproduces the oracle. Against a one-hot `q` this reduces to
/// [`brier_score`] term for term.
pub fn posterior_error(probs: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
    assert_eq!(probs.len(), targets.len(), "one target distribution per row");
    if probs.is_empty() {
        return 0.0;
    }
    let sum: f64 = probs
        .iter()
        .zip(targets)
        .map(|(p, q)| {
            assert_eq!(p.len(), q.len(), "a row and its target must have the same arity");
            p.iter()
                .zip(q)
                .map(|(&pi, &qi)| {
                    let d = pi as f64 - qi as f64;
                    d * d
                })
                .sum::<f64>()
        })
        .sum();
    (sum / probs.len() as f64) as f32
}

/// Mean `KL(oracle || model)` - how much belief the model is missing,
/// in nats, with a floor of exactly zero.
///
/// This is [`soft_nll`] minus the oracle's own entropy, so it answers the
/// question `soft_nll` cannot on its own: is this run's number large because
/// the model is wrong, or because the world is genuinely uncertain.
pub fn posterior_kl(probs: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
    assert_eq!(probs.len(), targets.len(), "one target distribution per row");
    if probs.is_empty() {
        return 0.0;
    }
    let sum: f64 = probs
        .iter()
        .zip(targets)
        .map(|(p, q)| {
            assert_eq!(p.len(), q.len(), "a row and its target must have the same arity");
            q.iter()
                .zip(p)
                .filter(|(&qi, _)| qi > 0.0)
                .map(|(&qi, &pi)| {
                    let (qi, pi) = (qi as f64, (pi as f64).max(1e-12));
                    qi * (qi / pi).ln()
                })
                .sum::<f64>()
        })
        .sum();
    (sum / probs.len() as f64) as f32
}

/// Top-label calibration against a known oracle: bin by the model's reported
/// probability for the option it would REPORT, and compare each bin's mean
/// against the oracle's own probability that that option is correct.
///
/// [`ece`] needs a realized `correct[i]` flag, which is a single Bernoulli
/// draw from `q[argmax p]`. Where the oracle is known, that draw can be
/// replaced by its mean (the same statistic with the sampling noise
/// removed), and the metric then has the property [`ece`] does not: **a
/// model that reproduces the oracle scores exactly zero.**
///
/// Feeding [`ece`] an entropy-derived "confidence" instead measures neither.
/// `1 - H(p)/ln K` is a statement about how PEAKED a distribution is, not
/// about how often the reported option is right, so a correctly uncertain
/// model is scored as badly calibrated for being uncertain: on the
/// device-diagnosis world an exact Bayesian oracle scores 0.646 that way,
/// worse than the trained model it is supposed to bound.
pub fn soft_ece(probs: &[Vec<f32>], targets: &[Vec<f32>], n_bins: usize) -> f32 {
    assert_eq!(probs.len(), targets.len(), "one target distribution per row");
    assert!(n_bins > 0, "at least one bin");
    let n = probs.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum_conf = vec![0.0f64; n_bins];
    let mut sum_true = vec![0.0f64; n_bins];
    let mut count = vec![0usize; n_bins];
    for (p, q) in probs.iter().zip(targets) {
        assert_eq!(p.len(), q.len(), "a row and its target must have the same arity");
        let top = p
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) })
            .0;
        let conf = p[top].clamp(0.0, 1.0);
        let mut b = (conf * n_bins as f32).floor() as usize;
        if b >= n_bins {
            b = n_bins - 1; // conf == 1.0 lands in the last bin
        }
        sum_conf[b] += conf as f64;
        sum_true[b] += q[top] as f64;
        count[b] += 1;
    }
    (0..n_bins)
        .filter(|&b| count[b] > 0)
        .map(|b| (count[b] as f64 / n as f64) * ((sum_true[b] - sum_conf[b]) / count[b] as f64).abs())
        .sum::<f64>() as f32
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

    /// The device-diagnosis world's three exact posteriors, each appearing
    /// twice - the shape of `samples/learning/rlcd`'s own held-out split.
    fn oracle_rows() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let q: Vec<Vec<f32>> = [
            vec![0.8f32, 0.2],                       // no evidence yet
            vec![1.0 / 3.0, 2.0 / 3.0],              // diagnostic: positive
            vec![18.0 / 19.0, 1.0 / 19.0],           // diagnostic: negative
        ]
        .iter()
        .flat_map(|d| [d.clone(), d.clone()])
        .collect();
        (q.clone(), q)
    }

    /// The property the hard-label metrics do not have, and the reason these
    /// exist: a model that reproduces the oracle EXACTLY must be reported as
    /// having nothing left to fix.
    #[test]
    fn an_exact_oracle_scores_zero_on_every_soft_metric() {
        let (probs, targets) = oracle_rows();
        assert!(posterior_error(&probs, &targets).abs() <= 1e-6);
        assert!(posterior_kl(&probs, &targets).abs() <= 1e-6);
        assert!(soft_ece(&probs, &targets, 10).abs() <= 1e-6);
        // soft_nll's floor is the world's own entropy, NOT zero.
        let entropy: f32 = targets.iter().map(|q| -q.iter().map(|&x| x * x.ln()).sum::<f32>()).sum::<f32>() / targets.len() as f32;
        assert!((soft_nll(&probs, &targets) - entropy).abs() <= 1e-5, "soft NLL should bottom out at the oracle's entropy");
    }

    /// What the metrics this replaces reported for that same exact oracle.
    /// Pinned as a number, because "the old metric was wrong" is a claim and
    /// 0.646 is the evidence: an ECE fed `1 - H(p)/ln K` ranks a perfect
    /// Bayesian model WORSE than the 0.554 a real trained run scored.
    #[test]
    fn entropy_confidence_ece_punishes_the_exact_oracle() {
        let (probs, targets) = oracle_rows();
        let conf: Vec<f32> = probs
            .iter()
            .map(|p| {
                let h: f32 = -p.iter().filter(|&&x| x > 0.0).map(|&x| x * x.ln()).sum::<f32>();
                (1.0 - h / (p.len() as f32).ln()).clamp(0.0, 1.0)
            })
            .collect();
        let correct = vec![true; probs.len()]; // p == q, so the argmax always matches
        let old = ece(&conf, &correct, 10);
        assert!((old - 0.6459).abs() <= 1e-3, "entropy-confidence ECE on an exact oracle was {old}");
        assert!(soft_ece(&probs, &targets, 10) < old, "the replacement must not inherit the defect");
    }

    /// Against a one-hot target the soft metrics have to agree with the
    /// hard-label ones they generalize, or they are a second answer to the
    /// same question rather than a wider one.
    #[test]
    fn the_soft_metrics_reduce_to_the_hard_label_ones_on_a_one_hot_target() {
        let probs = vec![vec![0.7f32, 0.2, 0.1], vec![0.1, 0.1, 0.8]];
        let labels = vec![0usize, 2];
        let targets: Vec<Vec<f32>> = labels
            .iter()
            .map(|&y| (0..3).map(|k| if k == y { 1.0 } else { 0.0 }).collect())
            .collect();
        assert!((soft_nll(&probs, &targets) - nll(&probs, &labels)).abs() <= 1e-6);
        assert!((posterior_error(&probs, &targets) - brier_score(&probs, &labels)).abs() <= 1e-6);
    }

    /// A model that is confidently wrong has to be scored as miscalibrated,
    /// or the metric cannot tell the run that needs work from the one that
    /// does not.
    #[test]
    fn soft_ece_grows_with_overconfidence() {
        let targets = vec![vec![1.0f32 / 3.0, 2.0 / 3.0]; 4];
        let honest = vec![vec![1.0f32 / 3.0, 2.0 / 3.0]; 4];
        let cocky = vec![vec![0.02f32, 0.98]; 4];
        assert!(soft_ece(&cocky, &targets, 10) > soft_ece(&honest, &targets, 10));
        // 0.98 claimed where the oracle says 0.667: a gap of about 0.313.
        assert!((soft_ece(&cocky, &targets, 10) - 0.3133).abs() <= 1e-3);
    }

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
