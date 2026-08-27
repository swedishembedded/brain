// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `RestoreEDMSampler`'s scalar math - SUPIR's sampler, despite the "EDM"
//! name **not** Karras: a `LegacyDDPMDiscretization` (SDXL's own linear-β,
//! 1000-step chain, [`crate::discrete::DiscreteConfig::sdxl`]) driven with
//! `s_churn`/`s_noise` stochasticity, an optional restoration-guidance pull
//! toward a clean reference, and a `LinearCFG` guidance ramp.
//!
//! Pure host math, no `gpu_core` dependency, by the same design as
//! [`crate::discrete`] and [`crate::scheduler`]: a sampling loop moves
//! `O(latent)` floats per step and the schedule/step arithmetic here is
//! `O(steps)` scalars (plus one elementwise pass per call), so every piece
//! is unit-testable with no weights and no device.
//!
//! Two corrections over a naive reading of the reference, both load-bearing:
//!
//! 1. **The control-scale ramp uses the RAW pre-churn `sigma`**, not
//!    `sigma_hat`, even though it is computed after the churn step in the
//!    reference's own call order - see [`control_scale_ramp`].
//! 2. **`LinearCFG`'s constructor argument naming is the OPPOSITE of what a
//!    quick read suggests**: its first argument is the guidance scale AT
//!    `sigma_max`, its second (`scale_min`) is the scale AT `sigma -> 0` - so
//!    at SDXL's own defaults (`spt_linear_CFG = 1.0`, `s_cfg = 4.0`) the
//!    "min" argument ends up holding the LARGER number. [`linear_cfg_scale`]
//!    takes both endpoints as plain arguments so the caller supplies which
//!    upstream constant means which, rather than the function guessing.

use crate::discrete::DiscreteConfig;

/// σ's hard-coded upper bound in the reference - not derived from the
/// schedule (the actual 1000-step grid's max, [`DiscreteDenoiserWithControl`]'s
/// last entry, lands a few parts in 1e7 away from this).
pub const SIGMA_MAX: f32 = 14.6146;

/// Restoration guidance only applies while `next_sigma` is above this floor
/// (`restore_cfg_s_tmin` upstream) - it fades out entirely on the last few
/// steps rather than pulling the fully-denoised sample toward the guidance
/// target.
pub const RESTORE_CFG_S_TMIN: f32 = 0.05;

/// `DiscreteDenoiserWithControl`'s snapping grid: `num_idx = 1000` discrete
/// σ values, one per training timestep of
/// [`DiscreteConfig::sdxl`] - reused, not reimplemented, per that config's
/// own doc. Every σ the sampler touches (`sigma`, `sigma_hat`, ...) is
/// snapped to the nearest of these before the denoiser sees it, because the
/// network was trained at exactly these 1000 discrete noise levels.
pub struct DiscreteDenoiserWithControl {
    /// Ascending: `sigmas[0]` is smallest (least noise, training step 0),
    /// `sigmas[999]` is largest (training step 999, close to
    /// [`SIGMA_MAX`]) - σ grows monotonically with the training timestep.
    sigmas: Vec<f32>,
}

impl Default for DiscreteDenoiserWithControl {
    fn default() -> Self {
        Self::new()
    }
}

impl DiscreteDenoiserWithControl {
    /// Builds the 1000-entry grid from `DiscreteConfig::sdxl()`'s own
    /// `ᾱ_t`: `σ_t = sqrt((1-ᾱ_t)/ᾱ_t)`.
    pub fn new() -> Self {
        let sigmas =
            DiscreteConfig::sdxl().alphas_cumprod().into_iter().map(|abar| ((1.0 - abar) / abar).sqrt()).collect();
        DiscreteDenoiserWithControl { sigmas }
    }

    /// The 1000-entry discrete σ grid, ascending.
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// Snap `sigma` to the nearest of the 1000 discrete grid values. A value
    /// outside `[sigmas[0], sigmas[999]]` clamps to whichever end is
    /// nearest, matching a plain nearest-neighbour search over the whole
    /// grid (there is no separate out-of-range branch upstream either).
    pub fn snap(&self, sigma: f32) -> f32 {
        let mut best = self.sigmas[0];
        let mut best_d = (sigma - best).abs();
        for &s in &self.sigmas[1..] {
            let d = (sigma - s).abs();
            if d < best_d {
                best_d = d;
                best = s;
            }
        }
        best
    }
}

/// `γ = min(s_churn/(n_steps-1), √2-1)` - the churn strength for one step of
/// an `n_steps`-step run. Upstream's `s_tmin`/`s_tmax` default to `0`/`inf`,
/// so churn is active at every step whenever `s_churn > 0`; `s_churn <= 0`
/// makes this `0` with no separate branch, since `min(0, √2-1) = 0`.
///
/// `n_steps` must be `> 1` (a one-step run has no `n_steps - 1` to divide
/// by); every real sampler config satisfies this (`edm_steps` defaults to
/// 50).
pub fn churn_gamma(s_churn: f32, n_steps: usize) -> f32 {
    assert!(n_steps > 1, "churn_gamma: n_steps must be > 1, got {n_steps}");
    (s_churn / (n_steps as f32 - 1.0)).min(std::f32::consts::SQRT_2 - 1.0)
}

/// `σ̂ = σ·(γ+1)` - the churned noise level a step actually denoises at.
pub fn sigma_hat(sigma: f32, gamma: f32) -> f32 {
    sigma * (gamma + 1.0)
}

/// `x += noise·s_noise·sqrt(σ̂²-σ²)` - the churn noise injection. A no-op
/// when `gamma == 0` (`sigma_hat == sigma`, so the added term is exactly
/// zero for every element), which is why callers do not need a separate
/// "churn is off" branch around this.
pub fn apply_churn_noise(x: &[f32], noise: &[f32], sigma: f32, sigma_hat: f32, s_noise: f32) -> Vec<f32> {
    assert_eq!(x.len(), noise.len(), "apply_churn_noise: x and noise must be the same length");
    let extra = (sigma_hat * sigma_hat - sigma * sigma).max(0.0).sqrt();
    let scale = s_noise * extra;
    x.iter().zip(noise).map(|(&xi, &n)| xi + n * scale).collect()
}

/// The restoration-guidance pull: `denoised -= (denoised - x_center)·(σ/σ_max)^restore_cfg`,
/// applied only when `next_sigma > `[`RESTORE_CFG_S_TMIN`]` AND restore_cfg > 0`
/// - both gates must hold, or `denoised` passes through unchanged (SUPIR's
/// CLI default is `restore_cfg = -1`, i.e. this is OFF by default despite
/// the shipped YAML's `4.0`).
pub fn restore_guidance(denoised: &[f32], x_center: &[f32], sigma: f32, sigma_max: f32, restore_cfg: f32, next_sigma: f32) -> Vec<f32> {
    assert_eq!(denoised.len(), x_center.len(), "restore_guidance: denoised and x_center must be the same length");
    if !(next_sigma > RESTORE_CFG_S_TMIN && restore_cfg > 0.0) {
        return denoised.to_vec();
    }
    let ratio = (sigma / sigma_max).powf(restore_cfg);
    denoised.iter().zip(x_center).map(|(&d, &xc)| d - (d - xc) * ratio).collect()
}

/// `LinearCFG`'s guidance-scale ramp: `scale` is the value AT `sigma_max`,
/// `scale_min` is the value AT `sigma -> 0` (see the module doc's
/// correction #2 - upstream's ctor argument naming is the opposite of what
/// it suggests). Evaluated at `sigma_hat`, the CHURNED value, per the real
/// call site (`denoise(x, σ̂, ..., control_scale=s)` reads the ramp at `σ̂`).
pub fn linear_cfg_scale(scale: f32, scale_min: f32, sigma_hat: f32, sigma_max: f32) -> f32 {
    (scale - scale_min) * sigma_hat / sigma_max + scale_min
}

/// The optional `linear_s_stage2` control-scale ramp (off by default - most
/// callers pass a constant `s_stage2`/`control_scale` instead).
///
/// Correction #1 over the ledger's pseudocode: this reads the RAW, pre-churn
/// `sigma` - not `sigma_hat` - even though the reference computes it AFTER
/// the churn step in call order. `s_start` is the ramp's value at
/// `sigma_max`; `s` is both the ramp's value at `sigma -> 0` and the
/// constant this replaces when the ramp is off.
pub fn control_scale_ramp(s: f32, s_start: f32, sigma: f32, sigma_max: f32) -> f32 {
    (sigma / sigma_max) * (s_start - s) + s
}

/// The sampler's first-order Euler update: `d = (x - denoised)/σ̂`,
/// `x_next = x + d·(σ_next - σ̂)`.
pub fn euler_step(x: &[f32], denoised: &[f32], sigma_hat: f32, sigma_next: f32) -> Vec<f32> {
    assert_eq!(x.len(), denoised.len(), "euler_step: x and denoised must be the same length");
    let dt = sigma_next - sigma_hat;
    x.iter().zip(denoised).map(|(&xi, &di)| xi + (xi - di) / sigma_hat * dt).collect()
}

/// `RestoreEDMSampler`'s CLI defaults, reproduced faithfully - including the
/// trap: `s_stage1` (`restore_cfg`) defaults to `-1.0`, so restoration
/// guidance is OFF at these defaults despite the shipped YAML's `4.0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestoreEDMSamplerConfig {
    pub edm_steps: usize,
    /// `s_cfg` - the guidance scale at `sigma -> 0` (`LinearCFG`'s
    /// `scale_min` argument, per the module doc's correction #2).
    pub s_cfg: f32,
    /// `spt_linear_CFG` - the guidance scale at `sigma_max` (`LinearCFG`'s
    /// `scale` argument).
    pub spt_linear_cfg: f32,
    /// `s_stage2` - the control scale (`control_scale`).
    pub s_stage2: f32,
    pub s_churn: f32,
    pub s_noise: f32,
    /// `s_stage1` - `restore_cfg`. Negative means OFF.
    pub s_stage1: f32,
}

impl Default for RestoreEDMSamplerConfig {
    fn default() -> Self {
        RestoreEDMSamplerConfig {
            edm_steps: 50,
            s_cfg: 4.0,
            spt_linear_cfg: 1.0,
            s_stage2: 1.0,
            s_churn: 5.0,
            s_noise: 1.01,
            s_stage1: -1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_grid_has_1000_ascending_entries_ending_near_sigma_max() {
        let d = DiscreteDenoiserWithControl::new();
        assert_eq!(d.sigmas().len(), 1000);
        assert!(d.sigmas().windows(2).all(|w| w[0] <= w[1]), "the grid must be ascending");
        assert!((d.sigmas()[0] - 0.029167158).abs() < 1e-6, "{}", d.sigmas()[0]);
        assert!((d.sigmas()[500] - 1.6182788).abs() < 1e-5, "{}", d.sigmas()[500]);
        assert!((d.sigmas()[999] - SIGMA_MAX).abs() < 1e-3, "grid max {} vs hard-coded {SIGMA_MAX}", d.sigmas()[999]);
    }

    #[test]
    fn snap_finds_the_nearest_grid_point_and_clamps_out_of_range() {
        let d = DiscreteDenoiserWithControl::new();
        let exact = d.sigmas()[500];
        assert_eq!(d.snap(exact), exact, "snapping an exact grid value must be a no-op");
        // Below the whole grid: the nearest point is the smallest.
        assert_eq!(d.snap(-5.0), d.sigmas()[0]);
        // Above the whole grid: the nearest point is the largest.
        assert_eq!(d.snap(1000.0), d.sigmas()[999]);
        // Strictly between two neighbours: picks whichever is closer.
        let mid = (d.sigmas()[10] + d.sigmas()[11]) / 2.0 - 1e-4;
        assert_eq!(d.snap(mid), d.sigmas()[10]);
    }

    #[test]
    fn churn_gamma_matches_hand_computed_values() {
        assert!((churn_gamma(5.0, 50) - 0.10204082).abs() < 1e-6);
        // Large s_churn is capped at sqrt(2) - 1.
        assert!((churn_gamma(100.0, 50) - 0.41421356).abs() < 1e-6);
        // s_churn <= 0 turns churn off with no separate branch.
        assert_eq!(churn_gamma(0.0, 50), 0.0);
    }

    #[test]
    fn sigma_hat_matches_hand_computed_values() {
        assert!((sigma_hat(1.0, 0.5) - 1.5).abs() < 1e-6);
        // gamma = 0 (churn off) must be the identity.
        assert_eq!(sigma_hat(2.0, 0.0), 2.0);
    }

    #[test]
    fn apply_churn_noise_matches_hand_computed_values() {
        let got = apply_churn_noise(&[0.0], &[1.0], 1.0, 1.5, 1.01);
        assert!((got[0] - 1.129214328637394).abs() < 1e-5, "{}", got[0]);
        // gamma = 0: sigma_hat == sigma, so this must be a true no-op.
        let noop = apply_churn_noise(&[3.0, -2.0], &[9.0, 9.0], 2.0, 2.0, 1.01);
        assert_eq!(noop, vec![3.0, -2.0]);
    }

    #[test]
    fn restore_guidance_matches_hand_computed_values_and_both_gates() {
        let got = restore_guidance(&[2.0], &[1.0], 1.0, SIGMA_MAX, 1.5, 1.0);
        assert!((got[0] - 1.9821013778430658).abs() < 1e-5, "{}", got[0]);

        // Off at the CLI default (restore_cfg = -1): unchanged regardless of sigma.
        let cfg = RestoreEDMSamplerConfig::default();
        let unchanged = restore_guidance(&[2.0], &[1.0], 1.0, SIGMA_MAX, cfg.s_stage1, 1.0);
        assert_eq!(unchanged, vec![2.0]);

        // On (restore_cfg > 0) but next_sigma has faded below the floor: still unchanged.
        let faded = restore_guidance(&[2.0], &[1.0], 1.0, SIGMA_MAX, 1.5, 0.01);
        assert_eq!(faded, vec![2.0]);
    }

    #[test]
    fn linear_cfg_scale_ramps_from_scale_at_sigma_max_to_scale_min_at_zero() {
        let (scale, scale_min) = (1.0, 4.0); // spt_linear_CFG, s_cfg defaults
        assert!((linear_cfg_scale(scale, scale_min, SIGMA_MAX, SIGMA_MAX) - 1.0).abs() < 1e-5);
        assert!((linear_cfg_scale(scale, scale_min, 0.0, SIGMA_MAX) - 4.0).abs() < 1e-6);
        assert!((linear_cfg_scale(scale, scale_min, 7.3073, SIGMA_MAX) - 2.5).abs() < 1e-4);
    }

    #[test]
    fn control_scale_ramp_matches_hand_computed_values() {
        let got = control_scale_ramp(1.0, 0.5, 7.3073, SIGMA_MAX);
        assert!((got - 0.75).abs() < 1e-4, "{got}");
        // sigma -> 0 must land on `s` (the "off" constant this ramp replaces).
        assert!((control_scale_ramp(1.0, 0.5, 0.0, SIGMA_MAX) - 1.0).abs() < 1e-6);
        // sigma = sigma_max must land on s_start.
        assert!((control_scale_ramp(1.0, 0.5, SIGMA_MAX, SIGMA_MAX) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn euler_step_matches_hand_computed_values() {
        let got = euler_step(&[1.0], &[0.6], 1.5, 1.0);
        assert!((got[0] - 0.8666666666666667).abs() < 1e-6, "{}", got[0]);
    }

    #[test]
    fn cli_defaults_leave_restoration_guidance_off() {
        let cfg = RestoreEDMSamplerConfig::default();
        assert_eq!(cfg.edm_steps, 50);
        assert_eq!(cfg.s_cfg, 4.0);
        assert_eq!(cfg.spt_linear_cfg, 1.0);
        assert_eq!(cfg.s_stage2, 1.0);
        assert_eq!(cfg.s_churn, 5.0);
        assert_eq!(cfg.s_noise, 1.01);
        assert!(cfg.s_stage1 < 0.0, "restoration guidance must be off at the CLI default");
    }
}
