// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Flow-matching (rectified-flow) Euler-discrete scheduler.
//!
//! A faithful port of diffusers' `FlowMatchEulerDiscreteScheduler` for the
//! static-shift case used by Z-Image and FLUX.2 (`use_dynamic_shifting=false`).
//! Reference: `resources/image-models/common/diffusers/src/diffusers/schedulers/
//! scheduling_flow_match_euler_discrete.py`.
//!
//! ## Model of the process
//! Rectified flow defines the noisy latent as a straight line between data `x0`
//! and noise `ε`: `x_σ = (1 - σ)·x0 + σ·ε`, with `σ ∈ [0, 1]`. The denoiser
//! predicts the velocity `v = dx/dσ = ε - x0`. Sampling integrates that ODE
//! backwards from `σ_max` toward 0 with explicit Euler steps:
//! `x_{next} = x + (σ_next - σ)·v`.
//!
//! ## Schedule construction (static shift)
//! The pipeline supplies a monotonically-decreasing `sigmas_in ∈ (0, 1]`
//! (Z-Image uses `linspace(1, 1/N, N)`; see [`default_z_image_sigmas`]). Each is
//! shifted by `σ' = shift·σ / (1 + (shift-1)·σ)`, a terminal `0` is appended
//! (so the last Euler step lands exactly on the clean latent), and the discrete
//! `timesteps` are `σ'·num_train_timesteps` (before the terminal).

/// Static config for the flow-match scheduler (the fields Z-Image/FLUX.2 set).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlowMatchConfig {
    /// Training-time discretization (Z-Image/FLUX.2: 1000). Maps `σ → t = σ·N`.
    pub num_train_timesteps: u32,
    /// Resolution-independent schedule shift (Z-Image/FLUX.2: 3.0).
    pub shift: f32,
}

impl Default for FlowMatchConfig {
    fn default() -> Self {
        FlowMatchConfig { num_train_timesteps: 1000, shift: 3.0 }
    }
}

/// The Z-Image default input-sigma spacing: `linspace(1.0, 1/n, n)` (inclusive
/// of both endpoints), matching `get_default_z_image_sigmas` in the pipeline.
pub fn default_z_image_sigmas(n: usize) -> Vec<f32> {
    linspace(1.0, 1.0 / n as f32, n)
}

/// `numpy.linspace(a, b, n)`: `n` points from `a` to `b` inclusive.
fn linspace(a: f32, b: f32, n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![a];
    }
    let step = (b - a) / (n - 1) as f32;
    (0..n).map(|i| a + step * i as f32).collect()
}

/// Flow-matching Euler-discrete scheduler (static shift). Stateful over a
/// sampling run: [`set_timesteps`](Self::set_timesteps) builds the schedule,
/// then [`step`](Self::step) is called once per timestep in order.
#[derive(Clone, Debug)]
pub struct FlowMatchEulerScheduler {
    cfg: FlowMatchConfig,
    /// `N+1` sigmas: the `N` shifted step sigmas plus a terminal `0`.
    sigmas: Vec<f32>,
    /// `N` discrete timesteps (`σ'·num_train_timesteps`), one per step.
    timesteps: Vec<f32>,
    /// Index of the next [`step`](Self::step) to take (into `sigmas`).
    step_index: usize,
}

impl FlowMatchEulerScheduler {
    /// Fresh scheduler with no schedule set yet (call [`set_timesteps`] next).
    pub fn new(cfg: FlowMatchConfig) -> Self {
        FlowMatchEulerScheduler { cfg, sigmas: Vec::new(), timesteps: Vec::new(), step_index: 0 }
    }

    /// Build the schedule from explicit input sigmas `∈ (0, 1]` (the pipeline's
    /// spacing). Applies the static shift, appends the terminal `0`, computes
    /// the discrete timesteps, and resets the step cursor.
    pub fn set_timesteps(&mut self, sigmas_in: &[f32]) {
        let shift = self.cfg.shift;
        let n_train = self.cfg.num_train_timesteps as f32;
        // Static shift, then discrete timesteps (before the terminal sigma).
        let shifted: Vec<f32> =
            sigmas_in.iter().map(|&s| shift * s / (1.0 + (shift - 1.0) * s)).collect();
        self.timesteps = shifted.iter().map(|&s| s * n_train).collect();
        // Append the terminal 0 so the final Euler step lands on the clean latent.
        self.sigmas = shifted;
        self.sigmas.push(0.0);
        self.step_index = 0;
    }

    /// The `N+1` sigmas (shifted step sigmas + terminal `0`).
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// The `N` discrete timesteps fed to the denoiser (`σ'·num_train_timesteps`).
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    /// One explicit-Euler flow-matching step: `x_next = x + (σ_next - σ)·v`,
    /// where `v = model_output` (predicted velocity). Advances the step cursor.
    /// `sample` and `model_output` must be the same length; returns `x_next`.
    pub fn step(&mut self, model_output: &[f32], sample: &[f32]) -> Vec<f32> {
        assert_eq!(
            sample.len(),
            model_output.len(),
            "sample/model_output length mismatch: {} != {}",
            sample.len(),
            model_output.len()
        );
        let i = self.step_index;
        assert!(
            i + 1 < self.sigmas.len(),
            "step() called {} times but schedule has only {} steps",
            i + 1,
            self.sigmas.len() - 1
        );
        let dt = self.sigmas[i + 1] - self.sigmas[i];
        let prev: Vec<f32> =
            sample.iter().zip(model_output).map(|(x, v)| x + dt * v).collect();
        self.step_index += 1;
        prev
    }
}

// ---- FLUX.2 dynamic (resolution- and step-count-dependent) shift ------------

/// FLUX.2's empirical `mu`: the exponential-shift strength as a function of
/// the generated-image token count and the step count (BFL
/// `compute_empirical_mu`, duplicated verbatim in the diffusers pipeline).
/// Below 4300 tokens the two empirical lines (fit at 10 and 200 steps) are
/// linearly interpolated in `num_steps` — a 4-step klein run gets a genuinely
/// different schedule than a 50-step base run at the same resolution.
pub fn empirical_mu(image_seq_len: usize, num_steps: usize) -> f32 {
    const A1: f64 = 8.73809524e-05;
    const B1: f64 = 1.89833333;
    const A2: f64 = 0.00016927;
    const B2: f64 = 0.45666666;
    let seq = image_seq_len as f64;
    if image_seq_len > 4300 {
        return (A2 * seq + B2) as f32;
    }
    let m200 = A2 * seq + B2;
    let m10 = A1 * seq + B1;
    let a = (m200 - m10) / 190.0;
    let b = m200 - 200.0 * a;
    (a * num_steps as f64 + b) as f32
}

/// Static (resolution-independent) sigma shift: `σ' = shift·σ / (1 + (shift-1)·σ)`.
///
/// **Not interchangeable with [`time_shift_exponential`]**, which is FLUX.2's
/// `mu` form `σ' = e^mu / (e^mu + (1/σ - 1))`, parameterised by a token-count-
/// and step-count-dependent `mu` rather than by a scalar `shift`. The two agree
/// only for `e^mu = shift`, and picking the wrong one silently changes every
/// sigma in the schedule. This one is what Wan2.1 uses (`shift = 5.0` for T2V,
/// `3.0` for I2V at 480p, `16` for FLF2V/VACE); the exponential one is what
/// FLUX.2 and Z-Image's `dynamic_shift` use.
///
/// In `f64` on purpose: the reference builds the whole schedule in numpy
/// `float64` and rounds to `f32` exactly once, at the end. Rounding earlier
/// moves the result by more than a ULP, so the flow-matching solvers keep the
/// pipeline in `f64` and cast where the reference casts.
/// ([`FlowMatchEulerScheduler::set_timesteps`] applies the same map in `f32`,
/// which is what the Z-Image/FLUX.2 pipelines do.)
pub fn flow_shift(shift: f64, sigmas: &[f64]) -> Vec<f64> {
    sigmas.iter().map(|&s| shift * s / (1.0 + (shift - 1.0) * s)).collect()
}

/// Exponential time shift: `σ' = e^mu / (e^mu + (1/σ - 1))`, with `σ = 0`
/// mapping to 0. This is diffusers' `_time_shift_exponential` (sigma
/// exponent 1.0) and the same map Z-Image's `dynamic_shift` applies.
pub fn time_shift_exponential(mu: f32, sigmas: &[f32]) -> Vec<f32> {
    let e = (mu as f64).exp();
    sigmas
        .iter()
        .map(|&s| {
            if s <= 0.0 {
                0.0
            } else {
                (e / (e + (1.0 / s as f64 - 1.0))) as f32
            }
        })
        .collect()
}

/// The FLUX.2 Klein sigma schedule: `linspace(1, 1/N, N)` exponentially
/// shifted by [`empirical_mu`], with the terminal 0 appended (`N+1` entries) —
/// feed straight into [`FlowMatchEulerScheduler::set_timesteps`] with
/// `shift: 1.0` (the shift is already applied).
pub fn klein_sigmas(num_steps: usize, image_seq_len: usize) -> Vec<f32> {
    let n = num_steps;
    let mu = empirical_mu(image_seq_len, n);
    let base: Vec<f32> = (0..n)
        .map(|i| 1.0 - i as f32 * (1.0 - 1.0 / n as f32) / (n.max(2) - 1).max(1) as f32)
        .collect();
    let mut out = time_shift_exponential(mu, &base);
    out.push(0.0);
    out
}
