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
    /// Invert the shifted sigma schedule (`σ' → 1 - σ'`) and append a
    /// terminal `1.0` instead of `0.0` - diffusers' `invert_sigmas` flag.
    /// `false` for every caller before this field existed (Z-Image, FLUX.2,
    /// Wan, LTX): their terminal sigma is 0 (noise -> data). MiniMax Music 3
    /// is the first `true` caller: its DiT's own timestep convention runs
    /// the other way (`0` = noise, `1` = data - see its `Transformer1DModel`
    /// doc), and diffusers documents this exact flag ("only required in
    /// Mochi" before) for that case.
    pub invert_sigmas: bool,
}

impl Default for FlowMatchConfig {
    fn default() -> Self {
        FlowMatchConfig { num_train_timesteps: 1000, shift: 3.0, invert_sigmas: false }
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
    /// spacing). Applies the static shift, then either appends the terminal
    /// `0` (`invert_sigmas: false` - noise walks to data), or inverts every
    /// shifted sigma (`σ' → 1 - σ'`) and appends a terminal `1` instead
    /// (`invert_sigmas: true`); computes the discrete timesteps from
    /// whichever sigmas end up in effect, and resets the step cursor.
    pub fn set_timesteps(&mut self, sigmas_in: &[f32]) {
        let shift = self.cfg.shift;
        let n_train = self.cfg.num_train_timesteps as f32;
        // Static shift, then discrete timesteps (before the terminal sigma).
        let mut shifted: Vec<f32> =
            sigmas_in.iter().map(|&s| shift * s / (1.0 + (shift - 1.0) * s)).collect();
        let terminal = if self.cfg.invert_sigmas {
            for s in &mut shifted {
                *s = 1.0 - *s;
            }
            1.0
        } else {
            0.0
        };
        self.timesteps = shifted.iter().map(|&s| s * n_train).collect();
        // Append the terminal sigma so the final Euler step lands exactly on
        // the clean latent (or, inverted, on the fully-noised one).
        self.sigmas = shifted;
        self.sigmas.push(terminal);
        self.step_index = 0;
    }

    /// The `N+1` sigmas (shifted step sigmas + terminal `0`, or - when
    /// [`FlowMatchConfig::invert_sigmas`] is set - inverted sigmas + terminal
    /// `1`).
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
/// shifted by [`empirical_mu`], with the terminal 0 appended (`N+1` entries) -
/// feed straight into [`FlowMatchEulerScheduler::set_timesteps`] with
/// `shift: 1.0` (the shift is already applied).
///
/// The `base` below spells out the same ramp [`default_z_image_sigmas`] gives,
/// and unifying the two is a standing temptation. **Do not**, without moving
/// the FLUX.2 goldens deliberately and in the same change. The two spellings
/// associate their float ops differently - `(i * span) / (n-1)` here against
/// `((b - a) / (n-1)) * i` in [`linspace`] - and f32 multiply/divide is not
/// associative, so they disagree by one ULP (5.96e-8) for 1979 of the first
/// 2001 step counts. The handful that do agree includes the small round
/// numbers a spot check reaches for, which is exactly how this reads as
/// duplication when it is really a second, load-bearing rounding.
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

// ---- LTX-2.5 (`LTX2Scheduler`): token-count-dependent shift + terminal stretch ----

/// `ltx_core.components.schedulers.LTX2Scheduler`'s two calibration anchors -
/// the token counts at which `base_shift`/`max_shift` apply exactly; the
/// shift is LINEARLY interpolated between them (and extrapolated outside),
/// never clamped - `ltxv_schedule_dump_reference.py`'s golden cases span
/// below, at, between, and above both anchors specifically to prove this.
pub const LTX2_BASE_SHIFT_ANCHOR: f64 = 1024.0;
pub const LTX2_MAX_SHIFT_ANCHOR: f64 = 4096.0;

/// The real LTX-2.5 distilled-8-step sigma schedule, read verbatim from
/// `ltx_pipelines.utils.constants.DISTILLED_SIGMA_VALUES` (not derived from
/// [`ltx2_sigmas`] - this is a separate, hand-tuned constant table upstream
/// ships for its distilled checkpoint, cross-checked bit-for-bit against
/// `testdata/golden/ltxv/schedule/schedule.safetensors`'s
/// `distilled_8step.sigmas`).
pub const LTX2_DISTILLED_SIGMAS: [f32; 9] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0];
/// `STAGE_2_DISTILLED_SIGMA_VALUES` - the second stage of the distilled
/// two-stage pipeline (`ti2vid_two_stages*.py`), a suffix of
/// [`LTX2_DISTILLED_SIGMAS`].
pub const LTX2_STAGE2_DISTILLED_SIGMAS: [f32; 4] = [0.909375, 0.725, 0.421875, 0.0];
/// `TDP_DISTILLED_SIGMAS` - the temporal-duration-prediction pipeline's own
/// 2-step distilled schedule.
pub const LTX2_TDP_DISTILLED_SIGMAS: [f32; 3] = [0.625, 0.4, 0.0];

/// `LTX2Scheduler.execute`'s sigma vector (`steps+1` entries): a
/// resolution-independent `linspace(1,0,steps+1)` ramp, shifted by a
/// token-count-dependent Flux-style `mu` (linear interpolation between
/// `(BASE_SHIFT_ANCHOR, base_shift)` and `(MAX_SHIFT_ANCHOR, max_shift)`,
/// extrapolated outside that range - never clamped), optionally followed by a
/// terminal stretch so the last non-zero sigma lands exactly at `terminal`
/// (keeps the final denoise step from a schedule that would otherwise end too
/// close to fully clean, per `ltx_core`'s own docstring).
///
/// `token_count` is the LATENT token count (`lat_t * lh * lw` for video, no
/// patch-size division - `LTXModel`'s own `in_channels`-wide tokens), the
/// same quantity [`crate`]'s callers already compute for their DiT's `T`.
///
/// Computed in `f64` throughout, matching this module's other dynamic-shift
/// functions ([`flow_shift`]'s doc explains why) and the reference's own
/// numpy-`float64` closed-form self-check
/// (`ltxv_schedule_dump_reference.py`'s `expected_sigmas`) - only the
/// reference's OWN torch execution runs in f32, and the two are asserted to
/// agree to `< 1e-6` relative there, which is the bound
/// `crates/diffusion/tests/ltxv_schedule_parity.rs` re-checks against the
/// dumped golden.
pub fn ltx2_sigmas(token_count: usize, steps: usize, base_shift: f64, max_shift: f64, stretch: bool, terminal: f64) -> Vec<f64> {
    assert!(steps >= 1, "ltx2_sigmas: steps must be >= 1");
    // linspace(1.0, 0.0, steps+1) inclusive of both endpoints.
    let sigmas: Vec<f64> = (0..=steps).map(|i| 1.0 - i as f64 / steps as f64).collect();

    let mm = (max_shift - base_shift) / (LTX2_MAX_SHIFT_ANCHOR - LTX2_BASE_SHIFT_ANCHOR);
    let b = base_shift - mm * LTX2_BASE_SHIFT_ANCHOR;
    let sigma_shift = token_count as f64 * mm + b;
    let e = sigma_shift.exp();

    // `sigma == 0` (only the terminal linspace entry) maps to 0, never
    // through the exponential-shift formula (which would divide by zero) -
    // the reference's own `nz = sigmas != 0` mask.
    let mut out: Vec<f64> = sigmas.iter().map(|&s| if s == 0.0 { 0.0 } else { e / (e + (1.0 / s - 1.0)) }).collect();

    if stretch {
        // Stretch every NON-ZERO entry (post-shift) so the last of them lands
        // exactly at `terminal`: `scale = (1 - last)/(1 - terminal); out' = 1
        // - (1 - out)/scale` - the reference's own `non_zero_mask` (computed
        // on the SHIFTED sigmas, not the input ramp).
        let idxs: Vec<usize> = (0..out.len()).filter(|&i| out[i] != 0.0).collect();
        if let Some(&last) = idxs.last() {
            let scale = (1.0 - out[last]) / (1.0 - terminal);
            for &i in &idxs {
                out[i] = 1.0 - (1.0 - out[i]) / scale;
            }
        }
    }
    out
}

/// Rectified-flow ancestral Euler step
/// (`ltx_core.components.diffusion_steps.EulerAncestralDiffusionStep.step`,
/// verified against `scratchpad/reference/ltxv/packages/ltx-core/src/
/// ltx_core/components/diffusion_steps.py` rather than trusted from a
/// transcription): advance deterministically from `sigma` to an intermediate
/// `sigma_down <= sigma_next`, then renoise back up to `sigma_next`,
/// rescaling the signal component by `alpha_next/alpha_down` (`alpha = 1 -
/// sigma`) so the transition stays variance-preserving.
///
/// `x`/`denoised` are the current noisy sample and the model's denoised (x0)
/// prediction, same length. `eta` interpolates between a plain Euler step
/// (`eta=0`: `sigma_down == sigma_next`, `noise` unused and may be `None`)
/// and a fully ancestral step (`eta=1`, upstream's own distilled-pipeline
/// default - `ANCESTRAL_ETA`/`ANCESTRAL_S_NOISE` in `ltx_pipelines.
/// distilled`). `noise` must be standard-normal, one value per element of
/// `x`, and is REQUIRED (panics) whenever `eta > 0`.
///
/// `sigma_next == 0` is the schedule's terminal step: this returns
/// `denoised` directly (no formula applies at zero noise), same as upstream.
///
/// Deliberately NOT the DDIM/variance-exploding ancestral coefficients
/// (`_get_ancestral_step` in the same reference file) - that helper computes
/// a different `sigma_down` and a different injected-noise amount for the
/// same `eta`; the two agree only at `eta=0`. This is the rectified-flow
/// (`alpha = 1-sigma`) parameterization, the one LTX-2 uses.
pub fn euler_ancestral_step(x: &[f32], denoised: &[f32], sigma: f64, sigma_next: f64, eta: f64, s_noise: f64, noise: Option<&[f32]>) -> Vec<f32> {
    assert_eq!(x.len(), denoised.len(), "euler_ancestral_step: x/denoised length mismatch");
    if sigma_next == 0.0 {
        return denoised.to_vec();
    }
    assert!(eta <= 0.0 || noise.is_some(), "euler_ancestral_step: eta > 0 requires a noise tensor");

    let downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta;
    let sigma_down = sigma_next * downstep_ratio;
    let sigma_down_ratio = sigma_down / sigma;

    let mut out: Vec<f32> = x.iter().zip(denoised).map(|(&xi, &di)| (sigma_down_ratio * xi as f64 + (1.0 - sigma_down_ratio) * di as f64) as f32).collect();

    if eta > 0.0 {
        let noise = noise.expect("checked above");
        assert_eq!(noise.len(), x.len(), "euler_ancestral_step: noise length mismatch");
        let alpha_next = 1.0 - sigma_next;
        let alpha_down = 1.0 - sigma_down;
        let renoise_coeff = (sigma_next * sigma_next - sigma_down * sigma_down * alpha_next * alpha_next / (alpha_down * alpha_down)).max(0.0).sqrt();
        let ratio = alpha_next / alpha_down;
        for (o, &n) in out.iter_mut().zip(noise) {
            *o = (ratio * (*o as f64) + n as f64 * s_noise * renoise_coeff) as f32;
        }
    }
    out
}

#[cfg(test)]
mod ltx2_tests {
    use super::*;

    /// `eta=0` collapses the ancestral step to a plain Euler step
    /// (`sigma_down == sigma_next`, no noise), and needs no noise tensor even
    /// though `s_noise` is nonzero - the multiply is gated on `eta`, not
    /// `s_noise`, matching the reference's own `if self.eta > 0`.
    #[test]
    fn eta_zero_is_a_plain_euler_step_and_needs_no_noise() {
        let x = [1.0f32, -2.0, 0.5];
        let denoised = [0.0f32, 0.0, 0.0];
        let out = euler_ancestral_step(&x, &denoised, 0.8, 0.4, 0.0, 1.0, None);
        // sigma_down_ratio = sigma_next/sigma = 0.5, x_next = 0.5*x + 0.5*0.
        for (o, xi) in out.iter().zip(x) {
            assert!((*o as f64 - 0.5 * xi as f64).abs() < 1e-6, "{o} vs {xi}");
        }
    }

    /// The terminal step (`sigma_next == 0`) always returns the denoised
    /// prediction directly, whatever `eta`/`x` are - and never touches
    /// `noise`, so passing `None` at `eta>0` must not panic here.
    #[test]
    fn sigma_next_zero_returns_the_denoised_sample_directly() {
        let x = [3.0f32, -1.0];
        let denoised = [0.25f32, 0.75];
        let out = euler_ancestral_step(&x, &denoised, 0.3, 0.0, 1.0, 1.0, None);
        assert_eq!(out, denoised);
    }

    #[test]
    #[should_panic(expected = "requires a noise tensor")]
    fn eta_above_zero_without_noise_panics() {
        euler_ancestral_step(&[1.0], &[0.0], 0.5, 0.2, 1.0, 1.0, None);
    }

    /// The distilled constant tables are literal transcriptions - pin their
    /// lengths and endpoints (every LTX-2 schedule starts at sigma=1 and ends
    /// at sigma=0) so a future edit that fat-fingers a digit is caught even
    /// without the golden fixture.
    #[test]
    fn distilled_constants_are_well_formed() {
        assert_eq!(LTX2_DISTILLED_SIGMAS.len(), 9);
        assert_eq!(LTX2_DISTILLED_SIGMAS[0], 1.0);
        assert_eq!(*LTX2_DISTILLED_SIGMAS.last().unwrap(), 0.0);
        assert_eq!(LTX2_STAGE2_DISTILLED_SIGMAS.len(), 4);
        assert_eq!(*LTX2_STAGE2_DISTILLED_SIGMAS.last().unwrap(), 0.0);
        assert_eq!(LTX2_TDP_DISTILLED_SIGMAS.len(), 3);
        assert_eq!(*LTX2_TDP_DISTILLED_SIGMAS.last().unwrap(), 0.0);
    }

    /// Weight-free structural checks that hold for ANY valid
    /// `(base_shift, max_shift)` pair, independent of the golden fixture:
    /// the ramp always starts at 1 and ends at 0, and stretching never moves
    /// either endpoint (0 is never in the stretch mask; 1 stretches to
    /// exactly 1 because `out[0] == e/(e + 0) == 1` for every finite `e`).
    #[test]
    fn the_schedule_always_starts_at_one_and_ends_at_zero() {
        for &(tokens, steps, stretch) in &[(256usize, 8usize, true), (4096, 20, false), (8192, 50, true)] {
            let s = ltx2_sigmas(tokens, steps, 0.95, 2.05, stretch, 0.1);
            assert_eq!(s.len(), steps + 1);
            assert!((s[0] - 1.0).abs() < 1e-9, "tokens={tokens} steps={steps}: sigma[0] = {}", s[0]);
            assert_eq!(*s.last().unwrap(), 0.0, "tokens={tokens} steps={steps}: last sigma must be exactly 0");
            // Monotonically non-increasing.
            for w in s.windows(2) {
                assert!(w[0] >= w[1] - 1e-12, "not monotone: {w:?}");
            }
        }
    }
}
