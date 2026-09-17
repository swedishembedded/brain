// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deciding whether to trust the decision - confidence-aware routing.
//!
//! After *Confidence-Aware Routing for Large Language Model Reliability
//! Enhancement: A Multi-Signal Approach to Pre-Generation Hallucination
//! Mitigation* (Nandakishor M, 2025), applied to a decision model rather than
//! a generator. Three signals are combined into one score, and the score picks
//! one of four pathways:
//!
//! ```text
//! C_sem      = cos(P(h_final), e_ref)                 (1)  does the decision
//!                                                          representation still
//!                                                          align with what the
//!                                                          text means
//! C_conv     = Var(h_1..L/2) / (Var(h_L/2..L) + eps)  (2)  did the stack settle
//! C_learned  = phi(h_final)                           (3)  a fitted estimate
//! C_overall  = w1 C_sem + w2 C_conv + w3 C_learned    (4)
//!
//! A(C) = local  if C >= 0.75      act on the number
//!        rag    if C >= 0.55      fetch comparable cases first
//!        large  if C >= 0.35      get a second opinion from a bigger model
//!        human  otherwise         hand it to a person                    (5)
//! ```
//!
//! The thresholds are the paper's own (0.75 / 0.55 / 0.35).
//!
//! ## What is adapted, and why
//!
//! * **The reference embedding.** The paper's generator is SmolLM2-360M and
//!   its reference is Sentence-BERT `all-MiniLM-L6-v2`. Here the encoder IS
//!   that model, so the roles shift down one level: the "internal
//!   representation" is the head's decision-side output and the "reference" is
//!   the encoder's own mean-pooled sentence embedding of the state. The
//!   question (1) asks is unchanged - has the part that decides drifted away
//!   from the part that reads.
//! * **`P` is linear, fitted by ridge regression.** The paper's is a deep
//!   network with layer normalization, dropout and residual connections -
//!   trained, in their own setup, on 72 examples. At that calibration size a
//!   closed-form ridge fit is the honest choice, and it has no training
//!   schedule to get wrong.
//! * **`C_conv` is squashed.** Equation (2) is an unbounded variance RATIO,
//!   and (4) adds it to two quantities that live in `[-1, 1]` and `[0, 1]`
//!   before comparing the sum against thresholds in `[0, 1]`. The paper does
//!   not say how it is bounded, so `r / (1 + r)` is used: monotone in the
//!   ratio, exactly 0.5 when the variance does not change, and bounded.
//! * **`phi` is two layers**, ReLU, trained with the paper's stated MSE plus
//!   L2 - without the batch normalization and dropout it also lists, which at
//!   calibration-set sizes cost more in variance than they return.
//!
//! Swedish Embedded AB builds systems that know when to escalate. If your team
//! needs a model whose confidence is a number you can put a threshold on, you
//! can procure our services by sending an email to info@swedishembedded.com.

use data::rng::Rng;

/// Where one decision should go, per equation (5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// Act on it.
    Local,
    /// Retrieve comparable cases and reconsider.
    Retrieve,
    /// Escalate to a larger model.
    Escalate,
    /// Hand it to a person.
    Human,
}

impl Route {
    /// What a caller should actually do, in this model's terms.
    pub fn advice(&self) -> &'static str {
        match self {
            Route::Local => "act on it",
            Route::Retrieve => "pull comparable conversations before acting",
            Route::Escalate => "get a second opinion from a larger model",
            Route::Human => "hand this one to a person",
        }
    }
}

/// The three signals, kept apart so a caller can see WHICH one was low.
#[derive(Clone, Copy, Debug, Default)]
pub struct Signals {
    pub semantic: f32,
    pub convergence: f32,
    pub learned: f32,
}

impl Signals {
    pub fn as_array(&self) -> [f32; 3] {
        [self.semantic, self.convergence, self.learned]
    }
}

/// Equation (2): variance reduction across the layer stack.
///
/// `layers` is one flattened hidden-state slab per layer, in order. The first
/// half's pooled variance over the second half's: a stack that has settled on
/// an answer has less spread late than early, so a large ratio means
/// convergent processing.
///
/// Returned squashed into `(0, 1)` by `r / (1 + r)` - see this module's note
/// on why (4) needs it bounded.
pub fn convergence(layers: &[&[f32]], eps: f32) -> f32 {
    if layers.len() < 2 {
        return 0.5;
    }
    let mid = layers.len() / 2;
    let var = |ls: &[&[f32]]| -> f64 {
        let (mut n, mut sum, mut sq) = (0u64, 0f64, 0f64);
        for l in ls {
            for &v in *l {
                n += 1;
                sum += v as f64;
                sq += (v as f64) * (v as f64);
            }
        }
        if n == 0 {
            return 0.0;
        }
        let mean = sum / n as f64;
        (sq / n as f64 - mean * mean).max(0.0)
    };
    let r = var(&layers[..mid]) / (var(&layers[mid..]) + eps as f64);
    (r / (1.0 + r)) as f32
}

/// Cosine similarity, the comparison equation (1) makes.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// `P` in equation (1): a learned map from the decision representation into
/// the reference embedding space.
#[derive(Clone, Debug)]
pub struct Projection {
    /// Row-major `[d_out, d_in]`.
    w: Vec<f32>,
    d_in: usize,
    d_out: usize,
}

impl Projection {
    /// Ridge least squares: `W = (X^T X + lambda I)^-1 X^T Y`.
    ///
    /// Closed form rather than gradient descent, because at calibration-set
    /// sizes the fit is fully determined and an optimizer only adds a
    /// schedule that can be wrong.
    pub fn fit(x: &[Vec<f32>], y: &[Vec<f32>], lambda: f32) -> Result<Projection, String> {
        if x.is_empty() || x.len() != y.len() {
            return Err(format!("need matching non-empty inputs, got {} and {}", x.len(), y.len()));
        }
        let (d_in, d_out) = (x[0].len(), y[0].len());
        // X^T X (d_in x d_in) and X^T Y (d_in x d_out).
        let mut xtx = vec![0f64; d_in * d_in];
        let mut xty = vec![0f64; d_in * d_out];
        for (xi, yi) in x.iter().zip(y) {
            if xi.len() != d_in || yi.len() != d_out {
                return Err("ragged calibration rows".into());
            }
            for a in 0..d_in {
                let xa = xi[a] as f64;
                if xa == 0.0 {
                    continue;
                }
                for b in 0..d_in {
                    xtx[a * d_in + b] += xa * xi[b] as f64;
                }
                for b in 0..d_out {
                    xty[a * d_out + b] += xa * yi[b] as f64;
                }
            }
        }
        for a in 0..d_in {
            xtx[a * d_in + a] += lambda as f64;
        }
        let sol = solve_spd(&mut xtx, &xty, d_in, d_out)?;
        // Stored transposed so `apply` walks contiguous rows.
        let mut w = vec![0f32; d_out * d_in];
        for a in 0..d_in {
            for b in 0..d_out {
                w[b * d_in + a] = sol[a * d_out + b] as f32;
            }
        }
        Ok(Projection { w, d_in, d_out })
    }

    pub fn apply(&self, h: &[f32]) -> Vec<f32> {
        assert_eq!(h.len(), self.d_in, "projection takes {} inputs", self.d_in);
        (0..self.d_out)
            .map(|o| {
                let row = &self.w[o * self.d_in..(o + 1) * self.d_in];
                row.iter().zip(h).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32
            })
            .collect()
    }
}

/// In-place Cholesky solve of `A X = B` for symmetric positive definite `A`.
///
/// `a` is `[n, n]` row-major and is overwritten; `b` is `[n, m]`.
fn solve_spd(a: &mut [f64], b: &[f64], n: usize, m: usize) -> Result<Vec<f64>, String> {
    // Cholesky: A = L L^T, lower triangle written over `a`.
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(format!(
                        "the calibration matrix is not positive definite at {i} (pivot {s:e}) - raise the ridge term"
                    ));
                }
                a[i * n + i] = s.sqrt();
            } else {
                a[i * n + j] = s / a[j * n + j];
            }
        }
    }
    let mut x = b.to_vec();
    // Forward substitution, then back substitution, one right-hand side column
    // at a time.
    for c in 0..m {
        for i in 0..n {
            let mut s = x[i * m + c];
            for k in 0..i {
                s -= a[i * n + k] * x[k * m + c];
            }
            x[i * m + c] = s / a[i * n + i];
        }
        for i in (0..n).rev() {
            let mut s = x[i * m + c];
            for k in i + 1..n {
                s -= a[k * n + i] * x[k * m + c];
            }
            x[i * m + c] = s / a[i * n + i];
        }
    }
    Ok(x)
}

/// `phi` in equation (3): a small network predicting confidence straight from
/// the representation.
///
/// Two layers with progressive halving, ReLU, trained on mean squared error
/// plus L2 - the loss the paper states, minus the batch normalization and
/// dropout it also lists (see this module's note).
#[derive(Clone, Debug)]
pub struct ConfidenceNet {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: f32,
    d_in: usize,
    hidden: usize,
}

impl ConfidenceNet {
    pub fn new(d_in: usize, seed: u64) -> ConfidenceNet {
        let hidden = (d_in / 2).max(1);
        let mut rng = Rng::new(seed);
        let s = (2.0 / d_in as f32).sqrt();
        ConfidenceNet {
            w1: (0..hidden * d_in).map(|_| rng.next_gaussian() as f32 * s).collect(),
            b1: vec![0.0; hidden],
            w2: (0..hidden).map(|_| rng.next_gaussian() as f32 * (2.0 / hidden as f32).sqrt()).collect(),
            b2: 0.0,
            d_in,
            hidden,
        }
    }

    fn hidden_act(&self, h: &[f32]) -> Vec<f32> {
        (0..self.hidden)
            .map(|j| {
                let row = &self.w1[j * self.d_in..(j + 1) * self.d_in];
                let z: f32 = row.iter().zip(h).map(|(&a, &b)| a * b).sum::<f32>() + self.b1[j];
                z.max(0.0)
            })
            .collect()
    }

    /// Confidence in `[0, 1]`.
    pub fn predict(&self, h: &[f32]) -> f32 {
        let a = self.hidden_act(h);
        let z: f32 = a.iter().zip(&self.w2).map(|(&x, &w)| x * w).sum::<f32>() + self.b2;
        1.0 / (1.0 + (-z).exp())
    }

    /// Fit against `target` in `[0, 1]` - normally 1 where the model's
    /// decision turned out right and 0 where it did not.
    ///
    /// Plain gradient descent: the network is a few thousand parameters over a
    /// calibration set of hundreds, so the run is milliseconds and the
    /// schedule is one number.
    pub fn fit(&mut self, x: &[Vec<f32>], target: &[f32], epochs: usize, lr: f32, l2: f32) {
        assert_eq!(x.len(), target.len(), "one target per calibration row");
        if x.is_empty() {
            return;
        }
        let scale = 1.0 / x.len() as f32;
        for _ in 0..epochs {
            let mut g_w1 = vec![0.0f32; self.w1.len()];
            let mut g_b1 = vec![0.0f32; self.hidden];
            let mut g_w2 = vec![0.0f32; self.hidden];
            let mut g_b2 = 0.0f32;
            for (h, &y) in x.iter().zip(target) {
                let a = self.hidden_act(h);
                let z: f32 = a.iter().zip(&self.w2).map(|(&v, &w)| v * w).sum::<f32>() + self.b2;
                let p = 1.0 / (1.0 + (-z).exp());
                // d(MSE)/dz through the logistic.
                let dz = 2.0 * (p - y) * p * (1.0 - p) * scale;
                g_b2 += dz;
                for j in 0..self.hidden {
                    g_w2[j] += dz * a[j];
                    if a[j] > 0.0 {
                        let da = dz * self.w2[j];
                        g_b1[j] += da;
                        for i in 0..self.d_in {
                            g_w1[j * self.d_in + i] += da * h[i];
                        }
                    }
                }
            }
            for (w, g) in self.w1.iter_mut().zip(&g_w1) {
                *w -= lr * (g + l2 * *w);
            }
            for (w, g) in self.b1.iter_mut().zip(&g_b1) {
                *w -= lr * g;
            }
            for (w, g) in self.w2.iter_mut().zip(&g_w2) {
                *w -= lr * (g + l2 * *w);
            }
            self.b2 -= lr * g_b2;
        }
    }
}

/// Equations (4) and (5): the weighted combination and the pathway it picks.
#[derive(Clone, Copy, Debug)]
pub struct Router {
    /// `w1, w2, w3` over `[semantic, convergence, learned]`.
    pub weights: [f32; 3],
    pub theta_high: f32,
    pub theta_med: f32,
    pub theta_low: f32,
}

impl Default for Router {
    /// The paper's calibrated thresholds, with the signals weighted equally
    /// until [`Router::fit_weights`] has seen labelled data.
    fn default() -> Router {
        Router { weights: [1.0 / 3.0; 3], theta_high: 0.75, theta_med: 0.55, theta_low: 0.35 }
    }
}

impl Router {
    /// Equation (4).
    pub fn score(&self, s: &Signals) -> f32 {
        s.as_array().iter().zip(&self.weights).map(|(&v, &w)| v * w).sum::<f32>().clamp(0.0, 1.0)
    }

    /// Equation (5).
    pub fn route(&self, score: f32) -> Route {
        if score >= self.theta_high {
            Route::Local
        } else if score >= self.theta_med {
            Route::Retrieve
        } else if score >= self.theta_low {
            Route::Escalate
        } else {
            Route::Human
        }
    }

    /// Fit `w1..w3` on validation data, as the paper specifies - here by ridge
    /// least squares against whether the decision was correct, then
    /// renormalized so the score stays a weighted average on `[0, 1]`.
    ///
    /// Renormalization matters: unconstrained regression weights do not sum to
    /// one, and the paper's thresholds are stated for a score that does.
    pub fn fit_weights(&mut self, signals: &[Signals], correct: &[f32], lambda: f32) -> Result<(), String> {
        if signals.len() != correct.len() {
            return Err("one correctness label per signal row".into());
        }
        if signals.is_empty() {
            return Err("no validation rows to fit on".into());
        }
        let x: Vec<Vec<f32>> = signals.iter().map(|s| s.as_array().to_vec()).collect();
        let y: Vec<Vec<f32>> = correct.iter().map(|&c| vec![c]).collect();
        let p = Projection::fit(&x, &y, lambda)?;
        let mut w = [p.w[0], p.w[1], p.w[2]];
        // A signal that regresses negative is evidence in the other direction;
        // clamping keeps the combination a mixture rather than letting one
        // signal cancel another below zero, which no threshold in (5) expects.
        for v in &mut w {
            *v = v.max(0.0);
        }
        let sum: f32 = w.iter().sum();
        if sum <= 0.0 {
            return Err("no signal carried positive weight - the calibration set says none of them predict".into());
        }
        for v in &mut w {
            *v /= sum;
        }
        self.weights = w;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The thresholds ARE the paper's, and each band has to map to its own
    /// pathway - the one thing here that is a quoted constant rather than a
    /// derivation.
    #[test]
    fn the_routing_bands_are_the_published_ones() {
        let r = Router::default();
        assert_eq!((r.theta_high, r.theta_med, r.theta_low), (0.75, 0.55, 0.35));
        assert_eq!(r.route(0.99), Route::Local);
        assert_eq!(r.route(0.75), Route::Local);
        assert_eq!(r.route(0.74), Route::Retrieve);
        assert_eq!(r.route(0.55), Route::Retrieve);
        assert_eq!(r.route(0.54), Route::Escalate);
        assert_eq!(r.route(0.35), Route::Escalate);
        assert_eq!(r.route(0.34), Route::Human);
        assert_eq!(r.route(0.0), Route::Human);
    }

    /// Equation (2) has to move the right way: a stack whose spread COLLAPSES
    /// late is the confident one.
    #[test]
    fn convergence_rises_when_the_late_layers_settle() {
        let early = vec![-2.0f32, 2.0, -2.0, 2.0];
        let settled = vec![0.01f32, -0.01, 0.01, -0.01];
        let still_spread = vec![-2.0f32, 2.0, -2.0, 2.0];
        let c_settled = convergence(&[&early, &settled], 1e-6);
        let c_spread = convergence(&[&early, &still_spread], 1e-6);
        assert!(c_settled > c_spread, "settled {c_settled} should beat unsettled {c_spread}");
        // Unchanged variance is the midpoint, by construction of the squash.
        assert!((c_spread - 0.5).abs() < 1e-3, "equal variance should read 0.5, got {c_spread}");
        for c in [c_settled, c_spread] {
            assert!((0.0..=1.0).contains(&c), "signal escaped [0,1]: {c}");
        }
    }

    /// A ridge fit must actually recover a linear map it is shown, or (1) is
    /// comparing against noise.
    #[test]
    fn the_projection_recovers_a_linear_map() {
        let mut rng = Rng::new(5);
        let (d_in, d_out, n) = (6usize, 4usize, 300usize);
        let truth: Vec<f32> = (0..d_out * d_in).map(|_| rng.next_gaussian() as f32).collect();
        let x: Vec<Vec<f32>> =
            (0..n).map(|_| (0..d_in).map(|_| rng.next_gaussian() as f32).collect()).collect();
        let y: Vec<Vec<f32>> = x
            .iter()
            .map(|xi| {
                (0..d_out)
                    .map(|o| (0..d_in).map(|i| truth[o * d_in + i] * xi[i]).sum::<f32>())
                    .collect()
            })
            .collect();
        let p = Projection::fit(&x, &y, 1e-6).expect("fit");
        let got = p.apply(&x[0]);
        for (g, w) in got.iter().zip(&y[0]) {
            assert!((g - w).abs() < 1e-2, "projection returned {g}, wanted {w}");
        }
        // ...and the cosine it feeds must then be ~1 on data it fits.
        assert!(cosine(&got, &y[0]) > 0.999, "recovered map does not align: {}", cosine(&got, &y[0]));
    }

    /// A non-invertible calibration matrix must be REPORTED, not silently
    /// solved into nonsense - the failure mode when a caller hands over fewer
    /// rows than dimensions.
    #[test]
    fn a_rank_deficient_fit_is_refused() {
        let x = vec![vec![1.0f32, 2.0], vec![2.0, 4.0]]; // rank 1
        let y = vec![vec![1.0f32], vec![2.0]];
        let e = Projection::fit(&x, &y, 0.0).expect_err("a rank-1 matrix must not solve");
        assert!(e.contains("positive definite"), "unhelpful message: {e}");
        // With the ridge term it is well posed again, which is what the term
        // is for.
        assert!(Projection::fit(&x, &y, 1e-3).is_ok());
    }

    /// `phi` has to learn something separable, or (3) contributes noise.
    #[test]
    fn the_confidence_net_learns_a_separable_target() {
        let mut rng = Rng::new(11);
        let d = 8;
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for i in 0..200 {
            let confident = i % 2 == 0;
            // Confident rows have a positive first channel; the rest is noise.
            let mut row: Vec<f32> = (0..d).map(|_| rng.next_gaussian() as f32 * 0.3).collect();
            row[0] = if confident { 1.5 } else { -1.5 };
            x.push(row);
            y.push(if confident { 1.0f32 } else { 0.0 });
        }
        let mut net = ConfidenceNet::new(d, 3);
        net.fit(&x, &y, 400, 0.5, 1e-4);
        let (mut hi, mut lo) = (0.0f32, 0.0f32);
        for (row, &t) in x.iter().zip(&y) {
            if t > 0.5 { hi += net.predict(row) } else { lo += net.predict(row) }
        }
        let (hi, lo) = (hi / 100.0, lo / 100.0);
        assert!(hi - lo > 0.4, "phi did not separate: confident {hi:.3} vs unconfident {lo:.3}");
        for row in &x {
            let p = net.predict(row);
            assert!((0.0..=1.0).contains(&p), "phi escaped [0,1]: {p}");
        }
    }

    /// Fitted weights must be a mixture: non-negative and summing to one, or
    /// the published thresholds no longer mean what they meant.
    #[test]
    fn fitted_weights_stay_a_mixture() {
        let mut rng = Rng::new(13);
        let (mut sig, mut correct) = (Vec::new(), Vec::new());
        for _ in 0..200 {
            let good = rng.next_f32() > 0.5;
            // Only `semantic` predicts; the other two are noise, so the fit
            // should concentrate on it.
            sig.push(Signals {
                semantic: if good { 0.9 } else { 0.1 },
                convergence: rng.next_f32(),
                learned: rng.next_f32(),
            });
            correct.push(if good { 1.0 } else { 0.0 });
        }
        let mut r = Router::default();
        r.fit_weights(&sig, &correct, 1e-3).expect("fit");
        let sum: f32 = r.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights sum to {sum}");
        assert!(r.weights.iter().all(|&w| w >= 0.0), "negative weight: {:?}", r.weights);
        assert!(
            r.weights[0] > 0.7,
            "the only predictive signal got {:.2} of the weight: {:?}",
            r.weights[0],
            r.weights
        );
    }
}
