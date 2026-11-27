// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for stable off-policy reinforcement learning of language models from
// asynchronously sampled rollouts for its clients. If your team needs expertise in policy-gradient estimators that stay
// unbiased when the sampler lags the learner then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The policy-gradient estimators of `rl::objective::pg_batch` against exact expectations, model-free.
//!
//! A categorical policy over `K` actions with host logits `theta` (the trainer, `pi = softmax(theta)`) and a sampler
//! `mu = softmax(phi)` that lags it. Every estimator reduces to cross-entropy weights ([`weigh`]), so its gradient in
//! the logits is `sum_tokens w (pi - e_a) + sum_j c_j (pi - e_{v_j})`, the same arithmetic the LM head runs. Spec:
//!
//! * with the sampler's whole distribution as its top-K, the score-centred estimate at every action equals
//!   `-A (f(r_a) (e_a - pi) - sum_v mu_v f(r_v) (e_v - pi))` written independently in f64, so its expectation under a
//!   constant advantage is zero, while the masked importance-sampling estimator without centring is biased once the
//!   band cuts (shown exactly and on a sampled set);
//! * with a truncated top-K the tail bucket centres the unlisted mass, so the expectation is zero again when every
//!   unlisted action's ratio is inside the band;
//! * for groups of `G` rewards with group-standardised advantages, each estimator's sampled mean converges to its own
//!   exact expectation, enumerated over all `K^G` groups; the score-centred one is zero under a constant advantage and
//!   the clipped surrogate is not (it clips off-policy).

use data::rng::Lcg;
use rl::objective::advantage::{group_advantages_with, AdvantageNorm};
use rl::objective::grpo_term::ClipBand;
use rl::objective::pg_batch::{weigh, PgBatch, PgEstimator, PgSpec, SamplerTop};
use rl::objective::score_centring::CentringBand;

fn softmax(z: &[f64]) -> Vec<f64> {
    let mx = z.iter().copied().fold(f64::MIN, f64::max);
    let e: Vec<f64> = z.iter().map(|v| (v - mx).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

/// A policy pair over `K` actions: the trainer `pi` and the sampler `mu`, as probabilities.
struct Pair {
    pi: Vec<f64>,
    mu: Vec<f64>,
}

impl Pair {
    fn new(theta: &[f64], phi: &[f64]) -> Pair {
        Pair { pi: softmax(theta), mu: softmax(phi) }
    }

    fn k(&self) -> usize {
        self.pi.len()
    }

    /// The gradient in the logits of the cross-entropy weights [`weigh`] gives `tokens` (each an action with its
    /// advantage), every token listing the sampler's `listed` actions as its top-K.
    fn gradient(&self, tokens: &[(usize, f32)], spec: &PgSpec, listed: &[usize]) -> Vec<f64> {
        let (n, k) = (tokens.len(), listed.len());
        let ln = |p: f64| p.ln() as f32;
        let targets: Vec<u32> = tokens.iter().map(|&(a, _)| a as u32).collect();
        let old: Vec<f32> = tokens.iter().map(|&(a, _)| ln(self.mu[a])).collect();
        let new: Vec<f32> = tokens.iter().map(|&(a, _)| ln(self.pi[a])).collect();
        let advantage: Vec<f32> = tokens.iter().map(|&(_, a)| a).collect();
        let ids: Vec<u32> = (0..n).flat_map(|_| listed.iter().map(|&v| v as u32)).collect();
        let sampler: Vec<f32> = (0..n).flat_map(|_| listed.iter().map(|&v| ln(self.mu[v]))).collect();
        let trainer: Vec<f32> = (0..n).flat_map(|_| listed.iter().map(|&v| ln(self.pi[v]))).collect();
        let batch = PgBatch { targets: &targets, old_lp: &old, ref_lp: None, advantage: &advantage, is_weight: None, top: Some(SamplerTop { k, ids: &ids, logprobs: &sampler }), support: None, scale: None };
        let (w, terms) = weigh(&new, Some(&trainer), &batch, spec);
        assert!(terms.finite);
        let mut g = vec![0f64; self.k()];
        let mut add = |weight: f64, at: usize| {
            for (v, gv) in g.iter_mut().enumerate() {
                *gv += weight * (self.pi[v] - if v == at { 1.0 } else { 0.0 });
            }
        };
        for (i, &(a, _)) in tokens.iter().enumerate() {
            add(f64::from(w.label[i]), a);
            if let Some(extra) = &w.extra {
                for (j, &v) in listed.iter().enumerate() {
                    add(f64::from(extra[i * k + j]), v);
                }
            }
        }
        g
    }
}

fn centred(band: CentringBand) -> PgSpec {
    PgSpec { band: ClipBand::new(0.2, 0.2), kl_beta: 0.0, estimator: PgEstimator::ScoreCentred(band) }
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// `f(r) = r 1[low <= r <= high]`.
fn masked_ratio(r: f64, band: CentringBand) -> f64 {
    if r >= f64::from(band.low) && r <= f64::from(band.high) {
        r
    } else {
        0.0
    }
}

/// Invariant 5 of the ledger: with the whole vocabulary as top-K, each action's estimate is the centred score, and the
/// expectation under a constant advantage is zero; the uncentred masked estimator's is not once the band cuts.
#[test]
fn with_the_whole_vocabulary_the_centred_estimate_is_the_exact_centred_score() {
    let pair = Pair::new(&[0.3, -1.2, 2.0, 0.1, -0.4], &[0.9, 0.2, 0.5, -1.5, 0.0]);
    let all: Vec<usize> = (0..pair.k()).collect();
    let advantage = 0.7f32;
    for band in [CentringBand { low: 0.0, high: 1e9 }, CentringBand { low: 0.3, high: 2.8 }] {
        let ratio: Vec<f64> = (0..pair.k()).map(|v| pair.pi[v] / pair.mu[v]).collect();
        let centring: Vec<f64> = (0..pair.k()).map(|v| (0..pair.k()).map(|u| pair.mu[u] * masked_ratio(ratio[u], band) * (if u == v { 1.0 } else { 0.0 } - pair.pi[v])).sum()).collect();
        let (mut expected, mut uncentred) = (vec![0f64; pair.k()], vec![0f64; pair.k()]);
        for a in 0..pair.k() {
            let got = pair.gradient(&[(a, advantage)], &centred(band), &all);
            let fa = masked_ratio(ratio[a], band);
            for v in 0..pair.k() {
                let score = if v == a { 1.0 } else { 0.0 } - pair.pi[v];
                let want = -f64::from(advantage) * (fa * score - centring[v]);
                assert!((got[v] - want).abs() < 1e-5, "band {band:?} action {a}: d/dz_{v} {} vs {want}", got[v]);
                expected[v] += pair.mu[a] * got[v];
                uncentred[v] += pair.mu[a] * -f64::from(advantage) * fa * score;
            }
        }
        assert!(norm(&expected) < 1e-5, "band {band:?}: the centred estimator's expectation under a constant advantage is {expected:?}");
        if band.low > 0.0 {
            assert!(ratio.iter().any(|&r| masked_ratio(r, band) == 0.0), "the fixture's band must cut an action");
            assert!(norm(&uncentred) > 1e-2, "band {band:?}: the uncentred masked estimator is unbiased only without a cut: {uncentred:?}");
        } else {
            assert!(norm(&uncentred) < 1e-5, "without a cut the uncentred estimator is unbiased too: {uncentred:?}");
        }
    }
}

/// The same on a sampled set: 20000 tokens drawn from the sampler, a constant advantage. The centred mean sits within
/// its standard error of zero; the uncentred masked mean is many standard errors away.
#[test]
fn on_a_sampled_set_centring_removes_the_bias_of_the_masked_estimator() {
    let pair = Pair::new(&[0.3, -1.2, 2.0, 0.1, -0.4], &[0.9, 0.2, 0.5, -1.5, 0.0]);
    let band = CentringBand { low: 0.3, high: 2.8 };
    let all: Vec<usize> = (0..pair.k()).collect();
    let mut rng = Lcg::random();
    let draw = |rng: &mut Lcg| {
        let u = f64::from(rng.unit());
        let mut acc = 0.0;
        pair.mu.iter().position(|&m| {
            acc += m;
            u < acc
        })
        .unwrap_or(pair.k() - 1)
    };
    let n = 20_000;
    let (mut sum_c, mut sq_c, mut sum_u, mut sq_u) = (vec![0f64; pair.k()], vec![0f64; pair.k()], vec![0f64; pair.k()], vec![0f64; pair.k()]);
    for _ in 0..n {
        let a = draw(&mut rng);
        let c = pair.gradient(&[(a, 1.0)], &centred(band), &all);
        let fa = masked_ratio(pair.pi[a] / pair.mu[a], band);
        for v in 0..pair.k() {
            let u = -fa * (if v == a { 1.0 } else { 0.0 } - pair.pi[v]);
            sum_c[v] += c[v];
            sq_c[v] += c[v] * c[v];
            sum_u[v] += u;
            sq_u[v] += u * u;
        }
    }
    let nf = n as f64;
    let (mut uncentred_far, mut worst_centred) = (0f64, 0f64);
    for v in 0..pair.k() {
        let se = |sum: f64, sq: f64| ((sq / nf - (sum / nf).powi(2)).max(0.0) / nf).sqrt().max(1e-12);
        worst_centred = worst_centred.max((sum_c[v] / nf).abs() / se(sum_c[v], sq_c[v]));
        uncentred_far = uncentred_far.max((sum_u[v] / nf).abs() / se(sum_u[v], sq_u[v]));
    }
    assert!(worst_centred < 5.0, "the centred mean is {worst_centred:.1} standard errors from zero");
    assert!(uncentred_far > 20.0, "the uncentred masked mean is only {uncentred_far:.1} standard errors from zero");
}

/// A truncated top-K: the unlisted mass is centred as one bucket, which is exact when every unlisted action's ratio is
/// inside the band (then `sum_{unlisted} pi grad log pi = -sum_listed pi grad log pi`), so the expectation under a
/// constant advantage is zero again; without the bucket (a band its ratio leaves) it is not.
#[test]
fn a_truncated_top_k_is_centred_by_its_tail_bucket() {
    // Actions 3 and 4 are unlisted; their ratios (pi/mu) are inside [0.5, 2.0], and so is their bucket's.
    let pair = Pair::new(&[0.3, -1.2, 2.0, 0.1, -0.4], &[0.9, 0.2, 0.5, 0.3, -0.1]);
    let band = CentringBand { low: 0.5, high: 2.0 };
    for v in [3usize, 4] {
        let r = pair.pi[v] / pair.mu[v];
        assert!(masked_ratio(r, band) > 0.0, "the fixture's unlisted action {v} must be in the band: ratio {r}");
    }
    let listed = [0usize, 1, 2];
    let expected = |band: CentringBand| -> Vec<f64> {
        let mut e = vec![0f64; pair.k()];
        for a in 0..pair.k() {
            for (v, g) in pair.gradient(&[(a, 1.0)], &centred(band), &listed).iter().enumerate() {
                e[v] += pair.mu[a] * g;
            }
        }
        e
    };
    assert!(norm(&expected(band)) < 1e-5, "the tail-centred expectation is {:?}", expected(band));
}

/// Fixed rewards per action, groups of `G` with z-scored advantages: each estimator's sampled mean over 4000 groups
/// approaches its exact expectation over all `K^G` groups (within 5 standard errors per coordinate), and under a
/// constant advantage the score-centred expectation is zero while the clipped surrogate's is not.
#[test]
fn sampled_group_estimates_converge_to_their_exact_expectations() {
    const G: usize = 3;
    let pair = Pair::new(&[0.6, -0.8, 1.1, 0.0], &[0.0, 0.3, -0.4, 0.5]);
    let reward = [1.0f64, 0.0, 0.5, -0.5];
    let all: Vec<usize> = (0..pair.k()).collect();
    let clipped = PgSpec::clipped(ClipBand::new(0.2, 0.28), 0.0);
    let centred = centred(CentringBand::DEFAULT);
    let group_tokens = |group: &[usize], constant: bool| -> Vec<(usize, f32)> {
        let adv = if constant { vec![1.0; G] } else { group_advantages_with(AdvantageNorm::ZScore, &group.iter().map(|&a| reward[a]).collect::<Vec<f64>>()) };
        group.iter().zip(adv).map(|(&a, adv)| (a, adv as f32)).collect()
    };
    // Token-sum over the group, as one optimiser step of G tokens divides it.
    let estimate = |spec: &PgSpec, group: &[usize], constant: bool| -> Vec<f64> { pair.gradient(&group_tokens(group, constant), spec, &all).into_iter().map(|g| g / G as f64).collect() };
    let groups: Vec<Vec<usize>> = (0..pair.k().pow(G as u32)).map(|mut i| (0..G).map(|_| { let a = i % pair.k(); i /= pair.k(); a }).collect()).collect();
    let exact = |spec: &PgSpec, constant: bool| -> Vec<f64> {
        let mut e = vec![0f64; pair.k()];
        for group in &groups {
            let p: f64 = group.iter().map(|&a| pair.mu[a]).product();
            for (v, g) in estimate(spec, group, constant).iter().enumerate() {
                e[v] += p * g;
            }
        }
        e
    };
    let mut rng = Lcg::random();
    let mut draw = || {
        let u = f64::from(rng.unit());
        let mut acc = 0.0;
        pair.mu.iter().position(|&m| {
            acc += m;
            u < acc
        })
        .unwrap_or(pair.k() - 1)
    };
    let samples: Vec<Vec<usize>> = (0..4000).map(|_| (0..G).map(|_| draw()).collect()).collect();
    for (name, spec) in [("clipped surrogate", &clipped), ("score-centred", &centred)] {
        let want = exact(spec, false);
        assert!(norm(&want) > 1e-3, "{name}: a fixture whose expected update vanishes checks nothing: {want:?}");
        let (mut sum, mut sq) = (vec![0f64; pair.k()], vec![0f64; pair.k()]);
        for group in &samples {
            for (v, g) in estimate(spec, group, false).iter().enumerate() {
                sum[v] += g;
                sq[v] += g * g;
            }
        }
        let m = samples.len() as f64;
        for v in 0..pair.k() {
            let mean = sum[v] / m;
            let se = ((sq[v] / m - mean * mean).max(0.0) / m).sqrt();
            assert!((mean - want[v]).abs() <= 5.0 * se + 1e-9, "{name}: coordinate {v}: sampled mean {mean} vs exact {} (standard error {se})", want[v]);
        }
    }
    assert!(norm(&exact(&centred, true)) < 1e-5, "score-centred under a constant advantage: {:?}", exact(&centred, true));
    assert!(norm(&exact(&clipped, true)) > 1e-3, "the clipped surrogate off-policy under a constant advantage: {:?}", exact(&clipped, true));
}
