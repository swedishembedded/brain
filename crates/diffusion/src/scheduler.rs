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
