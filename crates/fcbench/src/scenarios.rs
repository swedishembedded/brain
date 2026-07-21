// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Synthetic forecasting scenarios — seeded, guaranteed unseen by construction,
//! each with a known data-generating process so the *optimal* forecast is
//! computable rather than merely observed.
//!
//! These prove implementation correctness, not financial skill: a model that
//! infers an AR(1) coefficient or a GARCH volatility path is working; that says
//! nothing about markets. The **random-walk scenario is a negative control** —
//! its optimal forecast is the last value, so any model that materially beats
//! the naive baseline on it is overfitting, and the harness treats that as a
//! failure.

use crate::rng::Rng;

/// One generated window: the observed `context` and the realised `future` (the
/// ground truth to score against).
#[derive(Clone, Debug)]
pub struct Window {
    pub context: Vec<f32>,
    pub future: Vec<f32>,
}

/// A data-generating process with a known structure.
pub trait Scenario: Send + Sync {
    /// Stable name (also the benchmark id).
    fn name(&self) -> &str;
    /// Capability axis this probes (`"trend"`, `"autoregressive"`,
    /// `"volatility"`, `"regime"`, `"control"`, `"tails"`).
    fn axis(&self) -> &str;
    /// Context length.
    fn context_len(&self) -> usize;
    /// Forecast horizon.
    fn horizon(&self) -> usize;
    /// Seasonal period, for seasonal-naive and MASE scaling (1 = none).
    fn season(&self) -> usize {
        1
    }
    /// Generate one seeded window.
    fn generate(&self, seed: u64) -> Window;
    /// The oracle (optimal) point forecast given a context, in closed form, when
    /// one exists. `None` for purely simulated processes with no analytic mean.
    fn oracle(&self, _context: &[f32]) -> Option<Vec<f32>> {
        None
    }
    /// True if this is a negative control: no model should beat the naive
    /// baseline here, and the harness fails if one does.
    fn is_negative_control(&self) -> bool {
        false
    }
}

// ---- deterministic seasonal + trend ----------------------------------------

/// A clean sinusoid plus linear trend plus small noise. Tests whether a model
/// can extrapolate obvious structure at all. Oracle = the noiseless continuation.
pub struct SeasonalTrend {
    pub period: usize,
    pub slope: f32,
    pub amp: f32,
    pub noise: f32,
    pub context: usize,
    pub horizon: usize,
}

impl Default for SeasonalTrend {
    fn default() -> Self {
        SeasonalTrend { period: 12, slope: 0.05, amp: 1.0, noise: 0.05, context: 120, horizon: 24 }
    }
}

impl SeasonalTrend {
    fn signal(&self, t: f32) -> f32 {
        self.slope * t + self.amp * (2.0 * std::f32::consts::PI * t / self.period as f32).sin()
    }
}

impl Scenario for SeasonalTrend {
    fn name(&self) -> &str {
        "seasonal_trend"
    }
    fn axis(&self) -> &str {
        "trend"
    }
    fn context_len(&self) -> usize {
        self.context
    }
    fn horizon(&self) -> usize {
        self.horizon
    }
    fn season(&self) -> usize {
        self.period
    }
    fn generate(&self, seed: u64) -> Window {
        let mut r = Rng::new(seed);
        let n = self.context + self.horizon;
        // random phase offset per seed so windows differ
        let off = (r.uniform() * self.period as f32).floor();
        let all: Vec<f32> = (0..n)
            .map(|i| self.signal(i as f32 + off) + r.normal_with(0.0, self.noise))
            .collect();
        Window { context: all[..self.context].to_vec(), future: all[self.context..].to_vec() }
    }
    fn oracle(&self, _c: &[f32]) -> Option<Vec<f32>> {
        // The oracle needs the phase; scored via generate() in the harness, so we
        // expose the noiseless mean as guidance only when the harness recomputes.
        None
    }
}

// ---- AR(1) -----------------------------------------------------------------

/// A mean-zero AR(1): `x_t = phi * x_{t-1} + eps`. Oracle forecast is
/// `phi^h * x_last` (geometric decay toward the mean). Tests optimal
/// autoregressive point forecasting.
pub struct Ar1 {
    pub phi: f32,
    pub sigma: f32,
    pub context: usize,
    pub horizon: usize,
}

impl Default for Ar1 {
    fn default() -> Self {
        Ar1 { phi: 0.7, sigma: 1.0, context: 200, horizon: 10 }
    }
}

impl Scenario for Ar1 {
    fn name(&self) -> &str {
        "ar1"
    }
    fn axis(&self) -> &str {
        "autoregressive"
    }
    fn context_len(&self) -> usize {
        self.context
    }
    fn horizon(&self) -> usize {
        self.horizon
    }
    fn generate(&self, seed: u64) -> Window {
        let mut r = Rng::new(seed);
        let n = self.context + self.horizon;
        let mut x = 0.0f32;
        // burn-in so the process is at its stationary distribution
        for _ in 0..200 {
            x = self.phi * x + r.normal_with(0.0, self.sigma);
        }
        let all: Vec<f32> = (0..n)
            .map(|_| {
                x = self.phi * x + r.normal_with(0.0, self.sigma);
                x
            })
            .collect();
        Window { context: all[..self.context].to_vec(), future: all[self.context..].to_vec() }
    }
    fn oracle(&self, context: &[f32]) -> Option<Vec<f32>> {
        let last = *context.last().unwrap_or(&0.0);
        Some((0..self.horizon).map(|h| self.phi.powi(h as i32 + 1) * last).collect())
    }
}

// ---- GARCH(1,1) volatility -------------------------------------------------

/// A price random walk whose return volatility follows GARCH(1,1). Tests
/// conditional-volatility forecasting and interval calibration. Oracle mean is
/// the last price (returns are mean-zero).
pub struct GarchVol {
    pub omega: f32,
    pub alpha: f32,
    pub beta: f32,
    pub context: usize,
    pub horizon: usize,
}

impl Default for GarchVol {
    fn default() -> Self {
        GarchVol { omega: 0.05, alpha: 0.1, beta: 0.85, context: 300, horizon: 10 }
    }
}

impl Scenario for GarchVol {
    fn name(&self) -> &str {
        "garch_vol"
    }
    fn axis(&self) -> &str {
        "volatility"
    }
    fn context_len(&self) -> usize {
        self.context
    }
    fn horizon(&self) -> usize {
        self.horizon
    }
    fn generate(&self, seed: u64) -> Window {
        let mut r = Rng::new(seed);
        let n = self.context + self.horizon;
        let uncond = self.omega / (1.0 - self.alpha - self.beta).max(1e-3);
        let mut s2 = uncond;
        let mut price = 100.0f32;
        let mut prices = Vec::with_capacity(n);
        // burn-in
        for _ in 0..200 {
            let e = r.normal() * s2.sqrt();
            s2 = self.omega + self.alpha * e * e + self.beta * s2;
        }
        for _ in 0..n {
            let e = r.normal() * s2.sqrt();
            price += e;
            prices.push(price);
            s2 = self.omega + self.alpha * e * e + self.beta * s2;
        }
        Window {
            context: prices[..self.context].to_vec(),
            future: prices[self.context..].to_vec(),
        }
    }
    fn oracle(&self, context: &[f32]) -> Option<Vec<f32>> {
        let last = *context.last().unwrap_or(&0.0);
        Some(vec![last; self.horizon])
    }
}

// ---- regime switching ------------------------------------------------------

/// A two-state Markov regime switch: the mean/drift flips between regimes with a
/// small transition probability. Tests adaptation to breaks. No simple closed
/// form for the future (depends on unobserved regime path).
pub struct RegimeSwitch {
    pub drift_a: f32,
    pub drift_b: f32,
    pub sigma: f32,
    pub p_switch: f32,
    pub context: usize,
    pub horizon: usize,
}

impl Default for RegimeSwitch {
    fn default() -> Self {
        RegimeSwitch {
            drift_a: 0.1,
            drift_b: -0.1,
            sigma: 0.3,
            p_switch: 0.02,
            context: 200,
            horizon: 10,
        }
    }
}

impl Scenario for RegimeSwitch {
    fn name(&self) -> &str {
        "regime_switch"
    }
    fn axis(&self) -> &str {
        "regime"
    }
    fn context_len(&self) -> usize {
        self.context
    }
    fn horizon(&self) -> usize {
        self.horizon
    }
    fn generate(&self, seed: u64) -> Window {
        let mut r = Rng::new(seed);
        let n = self.context + self.horizon;
        let mut state_a = r.uniform() < 0.5;
        let mut level = 0.0f32;
        let all: Vec<f32> = (0..n)
            .map(|_| {
                if r.uniform() < self.p_switch {
                    state_a = !state_a;
                }
                let drift = if state_a { self.drift_a } else { self.drift_b };
                level += drift + r.normal_with(0.0, self.sigma);
                level
            })
            .collect();
        Window { context: all[..self.context].to_vec(), future: all[self.context..].to_vec() }
    }
}

// ---- random walk (NEGATIVE CONTROL) ----------------------------------------

/// A driftless random walk: `x_t = x_{t-1} + eps`. The optimal forecast is the
/// last value; *nothing should beat the naive baseline here*. This is the
/// harness's negative control against false skill.
pub struct RandomWalkScenario {
    pub sigma: f32,
    pub context: usize,
    pub horizon: usize,
}

impl Default for RandomWalkScenario {
    fn default() -> Self {
        RandomWalkScenario { sigma: 1.0, context: 200, horizon: 10 }
    }
}

impl Scenario for RandomWalkScenario {
    fn name(&self) -> &str {
        "random_walk"
    }
    fn axis(&self) -> &str {
        "control"
    }
    fn context_len(&self) -> usize {
        self.context
    }
    fn horizon(&self) -> usize {
        self.horizon
    }
    fn generate(&self, seed: u64) -> Window {
        let mut r = Rng::new(seed);
        let n = self.context + self.horizon;
        let mut x = 0.0f32;
        let all: Vec<f32> = (0..n)
            .map(|_| {
                x += r.normal_with(0.0, self.sigma);
                x
            })
            .collect();
        Window { context: all[..self.context].to_vec(), future: all[self.context..].to_vec() }
    }
    fn oracle(&self, context: &[f32]) -> Option<Vec<f32>> {
        let last = *context.last().unwrap_or(&0.0);
        Some(vec![last; self.horizon])
    }
    fn is_negative_control(&self) -> bool {
        true
    }
}

// ---- jump diffusion (heavy tails) ------------------------------------------

/// Brownian motion with occasional large jumps — heavy-tailed returns. Tests
/// tail-quantile calibration. Oracle mean is the last value.
pub struct JumpDiffusion {
    pub sigma: f32,
    pub jump_prob: f32,
    pub jump_scale: f32,
    pub context: usize,
    pub horizon: usize,
}

impl Default for JumpDiffusion {
    fn default() -> Self {
        JumpDiffusion {
            sigma: 0.5,
            jump_prob: 0.02,
            jump_scale: 5.0,
            context: 250,
            horizon: 10,
        }
    }
}

impl Scenario for JumpDiffusion {
    fn name(&self) -> &str {
        "jump_diffusion"
    }
    fn axis(&self) -> &str {
        "tails"
    }
    fn context_len(&self) -> usize {
        self.context
    }
    fn horizon(&self) -> usize {
        self.horizon
    }
    fn generate(&self, seed: u64) -> Window {
        let mut r = Rng::new(seed);
        let n = self.context + self.horizon;
        let mut x = 0.0f32;
        let all: Vec<f32> = (0..n)
            .map(|_| {
                let mut step = r.normal_with(0.0, self.sigma);
                if r.uniform() < self.jump_prob {
                    step += r.normal_with(0.0, self.jump_scale);
                }
                x += step;
                x
            })
            .collect();
        Window { context: all[..self.context].to_vec(), future: all[self.context..].to_vec() }
    }
    fn oracle(&self, context: &[f32]) -> Option<Vec<f32>> {
        let last = *context.last().unwrap_or(&0.0);
        Some(vec![last; self.horizon])
    }
}

/// The default scenario battery for the P0 comparison (all univariate).
/// Multivariate / known-future-covariate scenarios land with Chronos-2, which
/// is the first model that consumes covariates.
pub fn default_battery() -> Vec<Box<dyn Scenario>> {
    vec![
        Box::new(SeasonalTrend::default()),
        Box::new(Ar1::default()),
        Box::new(GarchVol::default()),
        Box::new(RegimeSwitch::default()),
        Box::new(RandomWalkScenario::default()),
        Box::new(JumpDiffusion::default()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_are_deterministic_and_correctly_sized() {
        let s = Ar1::default();
        let w1 = s.generate(5);
        let w2 = s.generate(5);
        assert_eq!(w1.context, w2.context);
        assert_eq!(w1.future, w2.future);
        assert_eq!(w1.context.len(), s.context_len());
        assert_eq!(w1.future.len(), s.horizon());
        // different seed -> different data
        assert_ne!(s.generate(6).future, w1.future);
    }

    #[test]
    fn ar1_oracle_decays_geometrically() {
        let s = Ar1 { phi: 0.5, sigma: 0.0, context: 10, horizon: 3 };
        // with sigma 0 after burn-in x is 0; use a synthetic context
        let ctx = vec![0.0, 0.0, 4.0];
        let o = s.oracle(&ctx).unwrap();
        assert!((o[0] - 2.0).abs() < 1e-5); // 0.5^1 * 4
        assert!((o[1] - 1.0).abs() < 1e-5); // 0.5^2 * 4
        assert!((o[2] - 0.5).abs() < 1e-5); // 0.5^3 * 4
    }

    #[test]
    fn random_walk_is_flagged_as_negative_control() {
        assert!(RandomWalkScenario::default().is_negative_control());
        assert!(!Ar1::default().is_negative_control());
        // its oracle is the flat last value
        let o = RandomWalkScenario::default().oracle(&[1.0, 2.0, 3.0]).unwrap();
        assert!(o.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }

    #[test]
    fn battery_has_a_negative_control_and_distinct_names() {
        let b = default_battery();
        assert!(b.iter().any(|s| s.is_negative_control()));
        let names: std::collections::HashSet<&str> = b.iter().map(|s| s.name()).collect();
        assert_eq!(names.len(), b.len(), "scenario names must be unique");
    }
}
