// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Discrete (DDPM-parameterised) schedulers: DDIM, Euler, Euler-ancestral and
//! DPM-Solver++(2M), each with the **epsilon** and **v-prediction**
//! parameterisations.
//!
//! These are the schedulers the UNet family (SD / SDXL, `crates/unet`) samples
//! with, and they are a different family from [`crate::scheduler`]'s
//! rectified-flow Euler: there the noise level *is* the integration variable
//! `σ ∈ [0,1]` and the model predicts a velocity; here the forward process is
//! DDPM's variance-preserving chain
//!
//! ```text
//! x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε ,   ᾱ_t = prod_{s<=t} (1 - β_s)
//! ```
//!
//! and everything below is a way of integrating it backwards. Two coordinate
//! systems appear, and mixing them is the classic bug:
//!
//! * **ᾱ (`alphas_cumprod`) space**, indexed by an *integer* training timestep.
//!   DDIM works here directly.
//! * **σ space**, `σ_t = sqrt((1-ᾱ_t)/ᾱ_t)` — the "k-diffusion" variance-exploding
//!   reparameterisation. The Euler family and DPM-Solver work here, and they
//!   need [`Sigmas::scale_model_input`] (`x / sqrt(σ²+1)`) *before* each denoiser
//!   call because the network was trained on variance-preserving inputs.
//!   Forgetting that scaling is silent: the image is merely bad, nothing errors.
//!
//! Reference: diffusers `scheduling_{ddim,euler_discrete,
//! euler_ancestral_discrete,dpmsolver_multistep}.py`. Papers:
//! DDIM 2010.02502, DPM-Solver++ 2211.01095.
//!
//! Everything here is host math with no `gpu_core` dependency, exactly like
//! [`crate::scheduler`] — a sampling loop moves `O(latent)` floats per step and
//! the schedule arithmetic is `O(steps)` scalars.
//!
//! ## Noise
//! The ancestral step is stochastic. Rather than embed an RNG (which would make
//! the scheduler untestable against a dumped reference), the caller supplies the
//! noise: [`EulerAncestralScheduler::step_with_noise`]. `step` is the
//! deterministic `noise = 0` case, which is *not* the same sampler — it is
//! offered only so a parity test can isolate the deterministic part.

/// The β schedule the training run used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BetaSchedule {
    /// `linspace(beta_start, beta_end, N)`.
    Linear,
    /// `linspace(sqrt(beta_start), sqrt(beta_end), N)²` — latent diffusion
    /// (SD 1.x/2.x, SDXL).
    ScaledLinear,
    /// The GLIDE cosine schedule (`betas_for_alpha_bar` with the
    /// `cos((t+0.008)/1.008 · π/2)²` alpha-bar).
    SquaredcosCapV2,
}

/// What the denoiser's output means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prediction {
    /// The network predicts the noise `ε`. SD 1.5 / SDXL-base.
    Epsilon,
    /// The network predicts `x_0` directly.
    Sample,
    /// The network predicts `v = sqrt(ᾱ)·ε - sqrt(1-ᾱ)·x_0` (SD 2.1-v, SDXL
    /// refiner-style distillations).
    VPrediction,
}

/// How the `N` inference timesteps are drawn out of the `num_train_timesteps`
/// training grid. Table 2 of 2305.08891 — the three options give visibly
/// different images, so this is config, not a detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimestepSpacing {
    /// `linspace(0, N_train-1, N)` reversed.
    Linspace,
    /// `arange(0, N)·(N_train // N)` reversed, plus `steps_offset`. SDXL's
    /// `scheduler_config.json` (with `steps_offset = 1`).
    Leading,
    /// `arange(N_train, 0, -N_train/N)` minus 1.
    Trailing,
}

/// The DDPM chain a discrete scheduler integrates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiscreteConfig {
    pub num_train_timesteps: u32,
    pub beta_start: f32,
    pub beta_end: f32,
    pub beta_schedule: BetaSchedule,
    pub prediction: Prediction,
    pub timestep_spacing: TimestepSpacing,
    pub steps_offset: u32,
    /// DDIM only: use `ᾱ = 1` for the step below the first timestep (`true`) or
    /// `ᾱ_0` (`false`). SDXL ships `set_alpha_to_one: false`.
    pub set_alpha_to_one: bool,
}

impl Default for DiscreteConfig {
    /// diffusers' own defaults (`DDIMScheduler()` with no arguments).
    fn default() -> Self {
        DiscreteConfig {
            num_train_timesteps: 1000,
            beta_start: 1e-4,
            beta_end: 0.02,
            beta_schedule: BetaSchedule::Linear,
            prediction: Prediction::Epsilon,
            timestep_spacing: TimestepSpacing::Leading,
            steps_offset: 0,
            set_alpha_to_one: true,
        }
    }
}

impl DiscreteConfig {
    /// SDXL-base-1.0's `scheduler/scheduler_config.json`, verbatim.
    pub fn sdxl() -> DiscreteConfig {
        DiscreteConfig {
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            prediction: Prediction::Epsilon,
            timestep_spacing: TimestepSpacing::Leading,
            steps_offset: 1,
            set_alpha_to_one: false,
        }
    }

    /// The same chain re-parameterised for a v-prediction checkpoint.
    pub fn with_prediction(mut self, p: Prediction) -> DiscreteConfig {
        self.prediction = p;
        self
    }

    /// `β_t`, `t = 0..N_train`, computed in f64 and stored f32.
    ///
    /// Precision note, measured against torch on the SDXL schedule: torch
    /// builds this with `torch.linspace(..., dtype=torch.float32)`, whose
    /// `start + i·step` chain rounds every entry in f32. Reproducing that chain
    /// bit-for-bit is not achievable from outside ATen (its vectorised kernel
    /// splits at the halfway point and computes `step` at a precision the
    /// public API does not expose — both variants were tried and both leave a
    /// residual 7.5e-9 on the sqrt-space grid). So this computes the grid in
    /// f64 instead, which is the *more* accurate of the two and leaves a
    /// bounded, measured gap: `max |Δᾱ|/ᾱ = 9.5e-7` and `max |Δσ|/σ = 8.7e-6`
    /// versus torch over all 1000 entries. Quote those numbers rather than
    /// claiming bit-equality.
    pub fn betas(&self) -> Vec<f32> {
        let n = self.num_train_timesteps as usize;
        match self.beta_schedule {
            BetaSchedule::Linear => {
                linspace64(self.beta_start as f64, self.beta_end as f64, n)
                    .into_iter()
                    .map(|b| b as f32)
                    .collect()
            }
            BetaSchedule::ScaledLinear => linspace64(
                (self.beta_start as f64).sqrt(),
                (self.beta_end as f64).sqrt(),
                n,
            )
            .into_iter()
            .map(|b| (b * b) as f32)
            .collect(),
            BetaSchedule::SquaredcosCapV2 => {
                // betas_for_alpha_bar with the cosine alpha_bar, max_beta 0.999.
                let bar = |t: f64| ((t + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2).cos().powi(2);
                (0..n)
                    .map(|i| {
                        let t1 = i as f64 / n as f64;
                        let t2 = (i + 1) as f64 / n as f64;
                        (1.0 - bar(t2) / bar(t1)).min(0.999) as f32
                    })
                    .collect()
            }
        }
    }

    /// `ᾱ_t = prod_{s<=t}(1 - β_s)`, accumulated in **f64** and rounded to f32.
    ///
    /// The f64 accumulator is not gold-plating: torch's `cumprod` is a parallel
    /// scan, so a naive sequential f32 product drifts from it — measured over
    /// this exact 1000-entry schedule (fed torch's OWN betas so only the
    /// accumulation differs), `max |Δ|/ᾱ = 9.45e-7`, terminal 0.0046600914 vs
    /// torch's 0.004660095. With an f64 accumulator the same input is
    /// **bit-identical to torch on all 1000 entries**. The output stays f32 so
    /// σ and √ᾱ round exactly as they do in the reference.
    pub fn alphas_cumprod(&self) -> Vec<f32> {
        let mut acc = 1.0f64;
        self.betas()
            .into_iter()
            .map(|b| {
                acc *= 1.0 - b as f64;
                acc as f32
            })
            .collect()
    }

    /// The `N` inference timesteps, descending, as diffusers' integer-valued
    /// schedule. Returned as `f32` because the σ-space schedulers interpolate
    /// with them; DDIM rounds them back to indices.
    pub fn timesteps(&self, num_inference_steps: usize) -> Vec<f32> {
        let n_train = self.num_train_timesteps as usize;
        assert!(num_inference_steps > 0, "num_inference_steps must be > 0");
        assert!(
            num_inference_steps <= n_train,
            "num_inference_steps {num_inference_steps} > num_train_timesteps {n_train}"
        );
        match self.timestep_spacing {
            TimestepSpacing::Linspace => {
                let mut t = linspace64_round(0.0, (n_train - 1) as f64, num_inference_steps);
                t.reverse();
                t.into_iter().map(|v| v as f32).collect()
            }
            TimestepSpacing::Leading => {
                let ratio = n_train / num_inference_steps;
                (0..num_inference_steps)
                    .rev()
                    .map(|i| (i * ratio + self.steps_offset as usize) as f32)
                    .collect()
            }
            TimestepSpacing::Trailing => {
                // np.arange(N_train, 0, -N_train/N).round() - 1
                let step = n_train as f64 / num_inference_steps as f64;
                (0..num_inference_steps)
                    .map(|i| (round_half_even(n_train as f64 - i as f64 * step) - 1.0) as f32)
                    .collect()
            }
        }
    }
}

/// `numpy.linspace(a, b, n)` in f64.
fn linspace64(a: f64, b: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![a],
        _ => {
            let step = (b - a) / (n - 1) as f64;
            (0..n).map(|i| a + step * i as f64).collect()
        }
    }
}

/// `numpy.linspace(a, b, n).round()` — the *integer* timestep grid. Split from
/// [`linspace64`] because rounding the β grid would zero it; the two callers
/// look alike and sharing one rounding helper is how that mistake lands.
fn linspace64_round(a: f64, b: f64, n: usize) -> Vec<f64> {
    linspace64(a, b, n).into_iter().map(round_half_even).collect()
}

/// numpy's `round` (banker's rounding: halves go to the nearest EVEN integer).
/// `f64::round` rounds halves away from zero, which differs at exactly `x.5` —
/// reachable in `Trailing` spacing whenever `N_train/N` is a half-integer.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && (r % 2.0) != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// `numpy.interp(x, xp=0..len(fp), fp)` for the monotone integer grid the
/// schedulers use. `x` outside `[0, len(fp)-1]` clamps to the endpoints, as
/// numpy does.
fn interp_grid(x: f32, fp: &[f32]) -> f32 {
    let n = fp.len();
    if n == 0 {
        return 0.0;
    }
    let x = x as f64;
    if x <= 0.0 {
        return fp[0];
    }
    if x >= (n - 1) as f64 {
        return fp[n - 1];
    }
    let i = x.floor() as usize;
    let w = x - i as f64;
    (fp[i] as f64 + w * (fp[i + 1] as f64 - fp[i] as f64)) as f32
}

// ---------------------------------------------------------------------------
// σ-space schedule (shared by Euler, Euler-ancestral and DPM-Solver++)
// ---------------------------------------------------------------------------

/// The σ table and the input scaling that every k-diffusion-style scheduler
/// shares: `σ_i = interp(t_i, sqrt((1-ᾱ)/ᾱ))`, with a terminal σ appended.
///
/// Kept as its own type because the three σ-space schedulers below differ only
/// in their `step`, and duplicating the table construction is exactly how the
/// terminal-σ convention drifts between them (DPM-Solver++'s
/// `final_sigmas_type` is a config knob; the Euler family always terminates
/// at 0).
#[derive(Clone, Debug)]
pub struct Sigmas {
    /// `N+1` values: the `N` step sigmas plus the terminal one.
    pub sigmas: Vec<f32>,
    /// `N` denoiser timesteps.
    pub timesteps: Vec<f32>,
}

impl Sigmas {
    /// Build from a config and a step count. `sigma_last` is the appended
    /// terminal value (`0.0` for the Euler family and for DPM-Solver++'s
    /// default `final_sigmas_type="zero"`).
    pub fn new(cfg: &DiscreteConfig, num_inference_steps: usize, sigma_last: f32) -> Sigmas {
        let acp = cfg.alphas_cumprod();
        let full: Vec<f32> = acp.iter().map(|&a| (((1.0 - a) / a) as f64).sqrt() as f32).collect();
        let timesteps = cfg.timesteps(num_inference_steps);
        let mut sigmas: Vec<f32> = timesteps.iter().map(|&t| interp_grid(t, &full)).collect();
        sigmas.push(sigma_last);
        Sigmas { sigmas, timesteps }
    }

    /// `x / sqrt(σ²+1)` — the variance-preserving rescale the denoiser expects.
    /// MUST be applied to the latent before every model call in σ space.
    pub fn scale_model_input(&self, step_index: usize, sample: &[f32]) -> Vec<f32> {
        let s = self.sigmas[step_index];
        let k = 1.0 / ((s as f64 * s as f64 + 1.0).sqrt() as f32);
        sample.iter().map(|&x| x * k).collect()
    }

    /// The standard deviation of the initial latent: `x_T = ε · init_noise_sigma`.
    ///
    /// diffusers (`EulerDiscreteScheduler.init_noise_sigma`, and the identical
    /// property on `EulerAncestralDiscreteScheduler`):
    ///
    /// ```text
    /// if timestep_spacing in ["linspace", "trailing"]: return sigmas.max()
    /// return (sigmas.max()**2 + 1) ** 0.5
    /// ```
    ///
    /// So `Leading` — which is what SDXL ships — is the branch that takes the
    /// `sqrt(σ²+1)` lift, and `Linspace`/`Trailing` return the bare σ_max. The
    /// two are easy to swap because both are "the big one" and they differ by
    /// only a fraction of a percent (11.0736 vs 11.0283 on the SDXL 20-step schedule), which no
    /// image inspection would ever reveal — so it is gated by
    /// `discrete_parity` against the dumped reference value rather than left to
    /// a reading of the source.
    ///
    /// `sigmas[0]` is `sigmas.max()`: the table is built descending and
    /// `sigma_table_is_descending_with_terminal_zero` pins that.
    pub fn init_noise_sigma(&self, spacing: TimestepSpacing) -> f32 {
        let s = self.sigmas[0] as f64;
        match spacing {
            TimestepSpacing::Linspace | TimestepSpacing::Trailing => s as f32,
            TimestepSpacing::Leading => (s * s + 1.0).sqrt() as f32,
        }
    }
}

/// `x_0` from a denoiser output in σ space (`sample` is the UNSCALED latent,
/// i.e. the one `scale_model_input` was applied to on the way in).
fn x0_from_sigma(pred: Prediction, sigma: f32, model_output: &[f32], sample: &[f32]) -> Vec<f32> {
    let s = sigma as f64;
    match pred {
        Prediction::Epsilon => {
            sample.iter().zip(model_output).map(|(&x, &e)| ((x as f64) - s * e as f64) as f32).collect()
        }
        Prediction::Sample => model_output.to_vec(),
        Prediction::VPrediction => {
            // c_out·v + c_skip·x  =  -σ/sqrt(σ²+1)·v + 1/(σ²+1)·x
            let denom = s * s + 1.0;
            let c_out = -s / denom.sqrt();
            let c_skip = 1.0 / denom;
            sample
                .iter()
                .zip(model_output)
                .map(|(&x, &v)| (c_out * v as f64 + c_skip * x as f64) as f32)
                .collect()
        }
    }
}

/// `x_0` from a denoiser output in the **variance-preserving** convention
/// `x_t = α_t·x_0 + σ_t·ε`, with `(α_t, σ_t)` derived from a k-diffusion σ by
/// `α_t = 1/sqrt(σ²+1)`, `σ_t = σ·α_t`.
///
/// This is NOT [`x0_from_sigma`], and the difference is the DPM-Solver bug this
/// function exists to prevent: the Euler family treats the latent as
/// variance-EXPLODING (`x = x_0 + σ·ε`, which is why it must pre-divide the
/// model input by `sqrt(σ²+1)`), while DPM-Solver keeps the latent
/// variance-preserving and converts inside. Feeding the VE formula to
/// DPM-Solver was measured at `max_rel 1.1` against diffusers — a completely
/// different trajectory, not a rounding difference.
fn x0_from_vp(pred: Prediction, sigma: f32, model_output: &[f32], sample: &[f32]) -> Vec<f32> {
    let s = sigma as f64;
    let alpha_t = 1.0 / (s * s + 1.0).sqrt();
    let sigma_t = s * alpha_t;
    match pred {
        Prediction::Epsilon => sample
            .iter()
            .zip(model_output)
            .map(|(&x, &e)| ((x as f64 - sigma_t * e as f64) / alpha_t) as f32)
            .collect(),
        Prediction::Sample => model_output.to_vec(),
        Prediction::VPrediction => sample
            .iter()
            .zip(model_output)
            .map(|(&x, &v)| (alpha_t * x as f64 - sigma_t * v as f64) as f32)
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// DDIM
// ---------------------------------------------------------------------------

/// DDIM (2010.02502 eq. 12) with `eta = 0` — the deterministic implicit
/// sampler. Works directly in ᾱ space, so it needs **no**
/// [`Sigmas::scale_model_input`].
#[derive(Clone, Debug)]
pub struct DdimScheduler {
    cfg: DiscreteConfig,
    alphas_cumprod: Vec<f32>,
    final_alpha_cumprod: f32,
    timesteps: Vec<f32>,
    num_inference_steps: usize,
    step_index: usize,
}

impl DdimScheduler {
    pub fn new(cfg: DiscreteConfig) -> DdimScheduler {
        let alphas_cumprod = cfg.alphas_cumprod();
        let final_alpha_cumprod = if cfg.set_alpha_to_one { 1.0 } else { alphas_cumprod[0] };
        DdimScheduler {
            cfg,
            alphas_cumprod,
            final_alpha_cumprod,
            timesteps: Vec::new(),
            num_inference_steps: 0,
            step_index: 0,
        }
    }

    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        self.timesteps = self.cfg.timesteps(num_inference_steps);
        self.num_inference_steps = num_inference_steps;
        self.step_index = 0;
    }

    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    pub fn alphas_cumprod(&self) -> &[f32] {
        &self.alphas_cumprod
    }

    /// One DDIM step (`eta = 0`, no clipping/thresholding). Returns
    /// `(prev_sample, pred_original_sample)`; the second is what a preview
    /// decoder or a `pred_x0` guidance term wants, and returning it costs
    /// nothing since it is computed either way.
    pub fn step(&mut self, model_output: &[f32], sample: &[f32]) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(sample.len(), model_output.len(), "sample/model_output length mismatch");
        assert!(
            self.step_index < self.timesteps.len(),
            "step() called {} times but the schedule has {} steps",
            self.step_index + 1,
            self.timesteps.len()
        );
        let t = self.timesteps[self.step_index] as i64;
        // diffusers derives the previous timestep arithmetically, NOT from the
        // schedule: `prev = t - N_train // N`. With `Leading` spacing and a
        // steps_offset that is the same as timesteps[i+1], but with `Trailing`
        // or `Linspace` it is NOT, so reading the next schedule entry here
        // would silently change the sampler.
        let prev = t - (self.cfg.num_train_timesteps as i64) / (self.num_inference_steps as i64);
        let a_t = self.alphas_cumprod[t as usize] as f64;
        let a_prev = if prev >= 0 {
            self.alphas_cumprod[prev as usize] as f64
        } else {
            self.final_alpha_cumprod as f64
        };
        let b_t = 1.0 - a_t;
        let (sa, sb) = (a_t.sqrt(), b_t.sqrt());
        let (sap, sbp) = (a_prev.sqrt(), (1.0 - a_prev).sqrt());

        let mut x0 = Vec::with_capacity(sample.len());
        let mut prev_sample = Vec::with_capacity(sample.len());
        for (&x, &m) in sample.iter().zip(model_output) {
            let (x, m) = (x as f64, m as f64);
            let (p0, eps) = match self.cfg.prediction {
                Prediction::Epsilon => ((x - sb * m) / sa, m),
                Prediction::Sample => (m, (x - sa * m) / sb),
                Prediction::VPrediction => (sa * x - sb * m, sa * m + sb * x),
            };
            x0.push(p0 as f32);
            prev_sample.push((sap * p0 + sbp * eps) as f32);
        }
        self.step_index += 1;
        (prev_sample, x0)
    }
}

// ---------------------------------------------------------------------------
// Euler (discrete) and Euler-ancestral
// ---------------------------------------------------------------------------

/// SDXL's shipped default: deterministic Euler in σ space.
#[derive(Clone, Debug)]
pub struct EulerScheduler {
    cfg: DiscreteConfig,
    sched: Sigmas,
    step_index: usize,
}

impl EulerScheduler {
    pub fn new(cfg: DiscreteConfig) -> EulerScheduler {
        EulerScheduler { cfg, sched: Sigmas { sigmas: Vec::new(), timesteps: Vec::new() }, step_index: 0 }
    }

    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        self.sched = Sigmas::new(&self.cfg, num_inference_steps, 0.0);
        self.step_index = 0;
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sched.sigmas
    }

    pub fn timesteps(&self) -> &[f32] {
        &self.sched.timesteps
    }

    pub fn init_noise_sigma(&self) -> f32 {
        self.sched.init_noise_sigma(self.cfg.timestep_spacing)
    }

    /// `x / sqrt(σ²+1)` at the current cursor — call before every denoiser
    /// evaluation.
    pub fn scale_model_input(&self, sample: &[f32]) -> Vec<f32> {
        self.sched.scale_model_input(self.step_index, sample)
    }

    pub fn step(&mut self, model_output: &[f32], sample: &[f32]) -> Vec<f32> {
        let i = self.step_index;
        assert!(i + 1 < self.sched.sigmas.len(), "step() past the end of the schedule");
        let sigma = self.sched.sigmas[i];
        let x0 = x0_from_sigma(self.cfg.prediction, sigma, model_output, sample);
        let dt = (self.sched.sigmas[i + 1] - sigma) as f64;
        let inv_sigma = 1.0 / sigma as f64;
        let out = sample
            .iter()
            .zip(&x0)
            .map(|(&x, &p)| (x as f64 + (x as f64 - p as f64) * inv_sigma * dt) as f32)
            .collect();
        self.step_index += 1;
        out
    }
}

/// Euler with ancestral (stochastic) resampling: each step lands at `σ_down`
/// deterministically and then re-injects `σ_up · noise`.
#[derive(Clone, Debug)]
pub struct EulerAncestralScheduler {
    cfg: DiscreteConfig,
    sched: Sigmas,
    step_index: usize,
}

impl EulerAncestralScheduler {
    pub fn new(cfg: DiscreteConfig) -> EulerAncestralScheduler {
        EulerAncestralScheduler {
            cfg,
            sched: Sigmas { sigmas: Vec::new(), timesteps: Vec::new() },
            step_index: 0,
        }
    }

    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        self.sched = Sigmas::new(&self.cfg, num_inference_steps, 0.0);
        self.step_index = 0;
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sched.sigmas
    }

    pub fn timesteps(&self) -> &[f32] {
        &self.sched.timesteps
    }

    pub fn init_noise_sigma(&self) -> f32 {
        self.sched.init_noise_sigma(self.cfg.timestep_spacing)
    }

    pub fn scale_model_input(&self, sample: &[f32]) -> Vec<f32> {
        self.sched.scale_model_input(self.step_index, sample)
    }

    /// `(σ_up, σ_down)` for the current cursor — the ancestral split of
    /// `σ_to`. Exposed because a caller that draws its own noise needs
    /// `σ_up` to size it, and because it is the piece a parity test pins.
    pub fn ancestral_split(&self) -> (f32, f32) {
        let i = self.step_index;
        let (from, to) = (self.sched.sigmas[i] as f64, self.sched.sigmas[i + 1] as f64);
        let up = (to * to * (from * from - to * to) / (from * from)).sqrt();
        let down = (to * to - up * up).sqrt();
        (up as f32, down as f32)
    }

    /// The full ancestral step. `noise` must be the same length as `sample`
    /// (the caller owns the RNG — see the module header).
    pub fn step_with_noise(&mut self, model_output: &[f32], sample: &[f32], noise: &[f32]) -> Vec<f32> {
        assert_eq!(sample.len(), model_output.len(), "sample/model_output length mismatch");
        assert_eq!(sample.len(), noise.len(), "sample/noise length mismatch");
        let i = self.step_index;
        assert!(i + 1 < self.sched.sigmas.len(), "step() past the end of the schedule");
        let sigma = self.sched.sigmas[i];
        let x0 = x0_from_sigma(self.cfg.prediction, sigma, model_output, sample);
        let (up, down) = self.ancestral_split();
        let dt = (down - sigma) as f64;
        let inv_sigma = 1.0 / sigma as f64;
        let up = up as f64;
        let out = sample
            .iter()
            .zip(&x0)
            .zip(noise)
            .map(|((&x, &p), &z)| {
                (x as f64 + (x as f64 - p as f64) * inv_sigma * dt + up * z as f64) as f32
            })
            .collect();
        self.step_index += 1;
        out
    }

    /// The deterministic part alone (`noise = 0`). NOT the ancestral sampler —
    /// see the module header.
    pub fn step(&mut self, model_output: &[f32], sample: &[f32]) -> Vec<f32> {
        let zeros = vec![0.0f32; sample.len()];
        self.step_with_noise(model_output, sample, &zeros)
    }
}

// ---------------------------------------------------------------------------
// DPM-Solver++ (multistep)
// ---------------------------------------------------------------------------

/// Second-order update flavour (2211.01095 §3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverType {
    Midpoint,
    Heun,
}

/// DPM-Solver++ multistep, orders 1 and 2 (`algorithm_type = "dpmsolver++"`,
/// i.e. the **data-prediction** formulation, which is the one that is stable
/// with guidance).
///
/// Multistep: the previous step's converted model output is reused instead of
/// evaluating the network twice, so a 2nd-order step costs one denoiser call.
/// The first step (and, with `lower_order_final`, the last) necessarily falls
/// back to the 1st-order update.
#[derive(Clone, Debug)]
pub struct DpmSolverPlusPlusScheduler {
    cfg: DiscreteConfig,
    sched: Sigmas,
    order: usize,
    solver_type: SolverType,
    /// Converted (`x0`-space) model outputs, oldest first; `order` slots.
    outputs: Vec<Option<Vec<f32>>>,
    lower_order_nums: usize,
    step_index: usize,
}

impl DpmSolverPlusPlusScheduler {
    /// diffusers' defaults for `DPMSolverMultistepScheduler`: order 2,
    /// midpoint, `lower_order_final = true`, `final_sigmas_type = "zero"`.
    pub fn new(cfg: DiscreteConfig) -> DpmSolverPlusPlusScheduler {
        DpmSolverPlusPlusScheduler {
            cfg,
            sched: Sigmas { sigmas: Vec::new(), timesteps: Vec::new() },
            order: 2,
            solver_type: SolverType::Midpoint,
            outputs: vec![None, None],
            lower_order_nums: 0,
            step_index: 0,
        }
    }

    pub fn with_order(mut self, order: usize) -> DpmSolverPlusPlusScheduler {
        assert!(order == 1 || order == 2, "DPM-Solver++ orders 1 and 2 are implemented, not {order}");
        self.order = order;
        self.outputs = vec![None; order];
        self
    }

    pub fn with_solver_type(mut self, t: SolverType) -> DpmSolverPlusPlusScheduler {
        self.solver_type = t;
        self
    }

    /// Note the `+1` inside: diffusers builds `num_inference_steps + 1`
    /// timesteps and then drops the last one, which makes DPM-Solver's grid
    /// genuinely different from DDIM's at the same step count. Reproducing
    /// that is required for parity, not stylistic.
    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        let n_train = self.cfg.num_train_timesteps as usize;
        let timesteps: Vec<f32> = match self.cfg.timestep_spacing {
            TimestepSpacing::Linspace => {
                let mut t = linspace64_round(0.0, (n_train - 1) as f64, num_inference_steps + 1);
                t.reverse();
                t.pop();
                t.into_iter().map(|v| v as f32).collect()
            }
            TimestepSpacing::Leading => {
                let ratio = n_train / (num_inference_steps + 1);
                let mut t: Vec<f32> = (0..=num_inference_steps)
                    .rev()
                    .map(|i| (i * ratio + self.cfg.steps_offset as usize) as f32)
                    .collect();
                t.pop();
                t
            }
            TimestepSpacing::Trailing => {
                let step = n_train as f64 / num_inference_steps as f64;
                (0..num_inference_steps)
                    .map(|i| (round_half_even(n_train as f64 - i as f64 * step) - 1.0) as f32)
                    .collect()
            }
        };
        let acp = self.cfg.alphas_cumprod();
        let full: Vec<f32> = acp.iter().map(|&a| (((1.0 - a) / a) as f64).sqrt() as f32).collect();
        let mut sigmas: Vec<f32> = timesteps.iter().map(|&t| interp_grid(t, &full)).collect();
        sigmas.push(0.0); // final_sigmas_type = "zero"
        self.sched = Sigmas { sigmas, timesteps };
        self.outputs = vec![None; self.order];
        self.lower_order_nums = 0;
        self.step_index = 0;
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sched.sigmas
    }

    pub fn timesteps(&self) -> &[f32] {
        &self.sched.timesteps
    }

    /// **The identity**, deliberately, and it exists so a sampling loop can
    /// call the same method on every scheduler. DPM-Solver keeps the latent in
    /// the variance-preserving parameterisation and converts inside
    /// [`x0_from_vp`], so there is nothing to rescale — diffusers'
    /// `DPMSolverMultistepScheduler.scale_model_input` is `return sample` for
    /// exactly this reason. Applying the Euler family's `1/sqrt(σ²+1)` here
    /// measured `max_rel 0.81` against the reference on the very first step.
    pub fn scale_model_input(&self, sample: &[f32]) -> Vec<f32> {
        sample.to_vec()
    }

    /// `(α_t, σ_t)` of the variance-preserving chain from a k-diffusion σ.
    fn alpha_sigma(sigma: f32) -> (f64, f64) {
        let s = sigma as f64;
        let alpha = 1.0 / (s * s + 1.0).sqrt();
        (alpha, s * alpha)
    }

    pub fn step(&mut self, model_output: &[f32], sample: &[f32]) -> Vec<f32> {
        assert_eq!(sample.len(), model_output.len(), "sample/model_output length mismatch");
        let i = self.step_index;
        let n = self.sched.timesteps.len();
        assert!(i < n, "step() past the end of the schedule");

        // `convert_model_output`: everything becomes an x0 prediction, in the
        // VARIANCE-PRESERVING convention (see `x0_from_vp`).
        let x0 = x0_from_vp(self.cfg.prediction, self.sched.sigmas[i], model_output, sample);
        for j in 0..self.order.saturating_sub(1) {
            self.outputs[j] = self.outputs[j + 1].take();
        }
        let last = self.order - 1;
        self.outputs[last] = Some(x0);

        // diffusers' stability rule, reproduced exactly: the LAST step drops to
        // first order whenever the schedule terminates at σ=0, which it always
        // does here (`final_sigmas_type = "zero"`). That changes the trajectory,
        // so it is not optional.
        //
        // diffusers' sibling rule `lower_order_second` (drop the SECOND-to-last
        // step to second order when `len(timesteps) < 15`) is deliberately
        // absent: it only ever selects the 2nd-order branch over the 3rd-order
        // one, and orders 1 and 2 are all this scheduler implements. Adding a
        // 3rd-order update means adding that rule with it.
        let lower_order_final = i == n - 1;

        let (a_t, s_t) = Self::alpha_sigma(self.sched.sigmas[i + 1]);
        let (a_s0, s_s0) = Self::alpha_sigma(self.sched.sigmas[i]);
        let lam_t = a_t.ln() - s_t.ln();
        let lam_s0 = a_s0.ln() - s_s0.ln();
        let h = lam_t - lam_s0;

        let use_first = self.order == 1 || self.lower_order_nums < 1 || lower_order_final;
        let out: Vec<f32> = if use_first {
            let m0 = self.outputs[last].as_ref().expect("current output stored");
            let c_x = s_t / s_s0;
            let c_m = -a_t * ((-h).exp() - 1.0);
            sample.iter().zip(m0).map(|(&x, &m)| (c_x * x as f64 + c_m * m as f64) as f32).collect()
        } else {
            // Second order.
            let (a_s1, s_s1) = Self::alpha_sigma(self.sched.sigmas[i - 1]);
            let lam_s1 = a_s1.ln() - s_s1.ln();
            let h0 = lam_s0 - lam_s1;
            let r0 = h0 / h;
            let m0 = self.outputs[last].as_ref().expect("current output stored");
            let m1 = self.outputs[last - 1].as_ref().expect("previous output stored");
            let c_x = s_t / s_s0;
            let e = (-h).exp() - 1.0;
            let (c_d0, c_d1) = match self.solver_type {
                SolverType::Midpoint => (-a_t * e, -0.5 * a_t * e),
                SolverType::Heun => (-a_t * e, a_t * (e / h + 1.0)),
            };
            sample
                .iter()
                .zip(m0)
                .zip(m1)
                .map(|((&x, &a), &b)| {
                    let d0 = a as f64;
                    let d1 = (a as f64 - b as f64) / r0;
                    (c_x * x as f64 + c_d0 * d0 + c_d1 * d1) as f32
                })
                .collect()
        };

        if self.lower_order_nums < self.order {
            self.lower_order_nums += 1;
        }
        self.step_index += 1;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdxl_beta_schedule_endpoints() {
        let cfg = DiscreteConfig::sdxl();
        let b = cfg.betas();
        assert_eq!(b.len(), 1000);
        assert!((b[0] - 0.00085).abs() < 1e-9, "beta[0] = {}", b[0]);
        assert!((b[999] - 0.012).abs() < 1e-7, "beta[999] = {}", b[999]);
        let a = cfg.alphas_cumprod();
        assert!(a[0] > a[999], "alphas_cumprod must decrease");
        // torch's value is 0.004660095; brain computes the beta grid in f64,
        // so agreement is to the documented ~1e-6 relative, not bit-exact.
        assert!(
            ((a[999] - 0.004660095) / 0.004660095).abs() < 2e-6,
            "terminal alphabar = {}",
            a[999]
        );
    }

    #[test]
    fn leading_spacing_matches_sdxl_config() {
        // 50 steps, N_train 1000, steps_offset 1 -> 981, 961, ..., 1.
        let cfg = DiscreteConfig::sdxl();
        let t = cfg.timesteps(50);
        assert_eq!(t.len(), 50);
        assert_eq!(t[0], 981.0);
        assert_eq!(t[49], 1.0);
    }

    #[test]
    fn sigma_table_is_descending_with_terminal_zero() {
        let cfg = DiscreteConfig::sdxl();
        let s = Sigmas::new(&cfg, 20, 0.0);
        assert_eq!(s.sigmas.len(), 21);
        assert_eq!(*s.sigmas.last().expect("terminal"), 0.0);
        for w in s.sigmas[..20].windows(2) {
            assert!(w[0] > w[1], "sigmas must decrease: {} then {}", w[0], w[1]);
        }
    }

    /// A zero model output leaves DDIM's x0 proportional to the sample, which
    /// is the one closed form available without a reference dump — a cheap
    /// guard on the ᾱ indexing.
    #[test]
    fn ddim_zero_epsilon_is_a_pure_rescale() {
        let cfg = DiscreteConfig::sdxl();
        let mut s = DdimScheduler::new(cfg);
        s.set_timesteps(10);
        let x = vec![1.0f32, -2.0, 0.5];
        let (prev, x0) = s.step(&[0.0; 3], &x);
        let t = 901usize;
        let a_t = s.alphas_cumprod()[t] as f64;
        let a_prev = s.alphas_cumprod()[801] as f64;
        let k = (a_prev / a_t).sqrt();
        for (i, &v) in prev.iter().enumerate() {
            assert!((v as f64 - k * x[i] as f64).abs() < 1e-5, "prev[{i}] = {v}");
        }
        for (i, &v) in x0.iter().enumerate() {
            assert!((v as f64 - x[i] as f64 / a_t.sqrt()).abs() < 1e-4, "x0[{i}] = {v}");
        }
    }

    /// v-prediction and epsilon must agree when the outputs are related by the
    /// definition `v = sqrt(ᾱ)·ε - sqrt(1-ᾱ)·x0` — a property test that would
    /// catch a swapped sign in either branch.
    #[test]
    fn ddim_v_prediction_agrees_with_epsilon() {
        let cfg = DiscreteConfig::sdxl();
        let mut eps_s = DdimScheduler::new(cfg);
        let mut v_s = DdimScheduler::new(cfg.with_prediction(Prediction::VPrediction));
        eps_s.set_timesteps(10);
        v_s.set_timesteps(10);
        let x: Vec<f32> = (0..8).map(|i| (i as f32 * 0.37).sin()).collect();
        let eps: Vec<f32> = (0..8).map(|i| (i as f32 * 1.13).cos()).collect();
        let t = eps_s.timesteps()[0] as usize;
        let a = eps_s.alphas_cumprod()[t] as f64;
        let (sa, sb) = (a.sqrt(), (1.0 - a).sqrt());
        // x0 implied by (x, eps), then the v that encodes the same pair.
        let v: Vec<f32> = x
            .iter()
            .zip(&eps)
            .map(|(&xx, &ee)| {
                let x0 = (xx as f64 - sb * ee as f64) / sa;
                (sa * ee as f64 - sb * x0) as f32
            })
            .collect();
        let (pa, _) = eps_s.step(&eps, &x);
        let (pb, _) = v_s.step(&v, &x);
        for i in 0..8 {
            assert!((pa[i] - pb[i]).abs() < 2e-5, "step {i}: {} vs {}", pa[i], pb[i]);
        }
    }

    /// The ancestral split must satisfy `σ_up² + σ_down² = σ_to²`.
    #[test]
    fn ancestral_split_is_a_variance_split() {
        let cfg = DiscreteConfig::sdxl();
        let mut s = EulerAncestralScheduler::new(cfg);
        s.set_timesteps(12);
        for _ in 0..11 {
            let to = s.sigmas()[s.step_index + 1] as f64;
            let (up, down) = s.ancestral_split();
            let lhs = (up as f64).powi(2) + (down as f64).powi(2);
            // Relative, not absolute: sigma reaches ~14.6 at the top of the
            // SDXL schedule, so an f32-rounding-sized error is ~1e-6 absolute.
            assert!((lhs - to * to).abs() <= 1e-6 * (to * to).max(1.0), "{lhs} != {}", to * to);
            let x = vec![0.0f32; 4];
            s.step(&x, &x);
        }
    }

    /// DPM-Solver++ drops to first order on the last step, so a schedule that
    /// terminates at σ=0 must land exactly on the final x0 prediction there
    /// (`σ_t/σ_s0 = 0`, `−α_t(e^{−h}−1) = 1` in the limit).
    #[test]
    fn dpmpp_last_step_lands_on_x0() {
        let cfg = DiscreteConfig::sdxl();
        let mut s = DpmSolverPlusPlusScheduler::new(cfg);
        s.set_timesteps(4);
        let x = vec![1.0f32, 2.0, 3.0];
        for _ in 0..3 {
            let _ = s.step(&[0.1f32; 3], &x);
        }
        let out = s.step(&[0.1f32; 3], &x);
        // At σ_next = 0 the first-order update is exactly the x0 prediction,
        // in the VARIANCE-PRESERVING convention: x0 = (x - σ_t·ε)/α_t.
        let sigma = s.sigmas()[3] as f64;
        let alpha_t = 1.0 / (sigma * sigma + 1.0).sqrt();
        for (i, &v) in out.iter().enumerate() {
            let want = (x[i] as f64 - sigma * alpha_t * 0.1) / alpha_t;
            assert!((v as f64 - want).abs() < 1e-4, "out[{i}] = {v}, want {want}");
        }
    }
}
