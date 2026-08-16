// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Multistep solvers in the **flow-matching** parameterisation: UniPC and
//! DPM-Solver++, the two samplers Wan2.1 ships (`--sample_solver unipc|dpm++`,
//! default `unipc`).
//!
//! This is a third scheduler family, alongside [`crate::scheduler`]'s
//! single-step Euler and [`crate::discrete`]'s DDPM chain, and it is a separate
//! family for a hard reason: [`crate::discrete::DpmSolverPlusPlusScheduler`] is
//! built on `alphas_cumprod` (`α_t = 1/sqrt(σ²+1)`, the variance-preserving
//! chain), while here the forward process is rectified flow, so
//!
//! ```text
//! α_t = 1 - σ ,   x_σ = (1-σ)·x_0 + σ·ε ,   model predicts v = ε - x_0
//! ```
//!
//! There is no β schedule and no `alphas_cumprod` to point at: the two families
//! share the *shape* of the update but not a single input.
//!
//! ## Schedules
//! Both solvers integrate `σ: σ_max → 0` over `N+1` sigmas (the terminal `0` is
//! `final_sigmas_type = "zero"`), each shifted by
//! [`crate::scheduler::flow_shift`]. They do **not** share the starting sigma:
//!
//! * UniPC starts at the top of the *training* grid, `1 - 1/N_train = 0.999`
//!   (first timestep 999), because the reference derives `sigma_max` from
//!   `linspace(1, 1/N_train, N_train)`;
//! * DPM++ builds its own `linspace(1, 0, N+1)[:N]` and starts at exactly `1.0`
//!   (first timestep 1000).
//!
//! Reusing one vector for both is wrong for one of them, so
//! [`unipc_sigmas`] and [`sampling_sigmas`] are separate functions.
//!
//! The schedule is computed in `f64` and rounded to `f32` exactly once at the
//! end, matching numpy's `astype(np.float32)` in the reference; the discrete
//! timesteps are `trunc(σ·N_train)` (an int64 cast, not a round).
//!
//! Reference: `wan/utils/fm_solvers_unipc.py` and `wan/utils/fm_solvers.py`
//! (UniPC 2302.04867, DPM-Solver++ 2211.01095). Host math only, like its two
//! sibling modules.

use crate::discrete::SolverType;
use crate::scheduler::flow_shift;

/// `numpy.linspace(a, b, n)` in `f64`, including numpy's exact-endpoint fixup
/// (`y[i] = i·step + a`, then `y[n-1] = b`). The schedule has to agree with the
/// reference to the ULP, so the evaluation order is not free.
fn linspace64(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![a];
    }
    let step = (b - a) / (n - 1) as f64;
    let mut v: Vec<f64> = (0..n).map(|i| i as f64 * step + a).collect();
    v[n - 1] = b;
    v
}

/// UniPC's shifted sigma grid: `linspace(σ_max, 0, N+1)` with the last entry
/// dropped, then [`flow_shift`]. `σ_max` is the top of the training grid
/// **after the f32 round-trip the reference constructor performs** - for
/// `N_train = 1000` that is `0.99900001` and not `1.0`, which is why the first
/// timestep of a Wan UniPC run is 999.
pub fn unipc_sigmas(num_train_timesteps: u32, num_inference_steps: usize, shift: f64) -> Vec<f64> {
    let sigma_max = ((1.0 - 1.0 / num_train_timesteps as f64) as f32) as f64;
    let mut base = linspace64(sigma_max, 0.0, num_inference_steps + 1);
    base.pop();
    flow_shift(shift, &base)
}

/// DPM++'s shifted sigma grid (`get_sampling_sigmas`): `linspace(1, 0, N+1)`
/// truncated to `N` entries, then [`flow_shift`]. Starts at exactly `1.0`.
pub fn sampling_sigmas(num_inference_steps: usize, shift: f64) -> Vec<f64> {
    let mut base = linspace64(1.0, 0.0, num_inference_steps + 1);
    base.truncate(num_inference_steps);
    flow_shift(shift, &base)
}

/// `(sigmas + terminal 0, timesteps)` in f32, from an already-shifted f64 grid.
fn finalize(shifted: &[f64], num_train_timesteps: u32) -> (Vec<f32>, Vec<f32>) {
    let n_train = num_train_timesteps as f64;
    // `torch.Tensor.to(int64)` truncates toward zero; the sigmas are positive,
    // so `trunc` is the whole of it. Kept as f32 (integral, exactly
    // representable) so every schedule vector in this crate has one type.
    let timesteps: Vec<f32> = shifted.iter().map(|&s| (s * n_train).trunc() as f32).collect();
    let mut sigmas: Vec<f32> = shifted.iter().map(|&s| s as f32).collect();
    sigmas.push(0.0);
    (sigmas, timesteps)
}

/// `λ = log(α) - log(σ)` with `α = 1 - σ`: the log-SNR the solvers integrate in.
/// At the terminal `σ = 0` this is `+∞`, which is load-bearing rather than a
/// degenerate case - it is what collapses the last step to `x = x0_pred`.
fn lambda_of(sigma: f64) -> f64 {
    (1.0 - sigma).ln() - sigma.ln()
}

/// Solve the (tiny, dense) `R·ρ = b` system the B(h) coefficients come from.
/// Gaussian elimination with partial pivoting, the same factorisation
/// `torch.linalg.solve` uses.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| {
            a[i][col].abs().partial_cmp(&a[j][col].abs()).expect("finite pivots")
        });
        let piv = piv.expect("non-empty column");
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        for row in (col + 1)..n {
            let f = a[row][col] / d;
            let (above, from_row) = a.split_at_mut(row);
            for (dst, &src) in from_row[0].iter_mut().zip(&above[col]).skip(col) {
                *dst -= f * src;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut s = b[row];
        for k in (row + 1)..n {
            s -= a[row][k] * x[k];
        }
        x[row] = s / a[row][row];
    }
    x
}

/// The `B(h)` variant of the UniPC update (2302.04867 §3.2). `bh2` is the
/// reference default and what Wan runs; `bh1` is recommended upstream only for
/// unconditional sampling under 10 steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BhSolver {
    /// `B(h) = h`.
    Bh1,
    /// `B(h) = e^h - 1`.
    Bh2,
}

/// The knobs `wan/text2video.py` leaves at their reference defaults, made
/// explicit because a port that guesses them differently is silently wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlowUniPcConfig {
    /// Training-time discretization (Wan: 1000). Maps `σ → t = trunc(σ·N)`.
    pub num_train_timesteps: u32,
    /// UniPC order `p`; the effective order of accuracy is `p+1` thanks to the
    /// corrector. Wan uses the default 2.
    pub solver_order: usize,
    /// Run the update on the predicted `x_0` (`true`, the default and what Wan
    /// uses) rather than on `ε`.
    pub predict_x0: bool,
    pub solver_type: BhSolver,
    /// Drop to a lower order in the final steps. Always `true` upstream, and it
    /// is not cosmetic: with the terminal `σ = 0` the last step *must* be first
    /// order or `λ` diverges into the `D1` differences.
    pub lower_order_final: bool,
}

impl Default for FlowUniPcConfig {
    /// Wan2.1's construction: `FlowUniPCMultistepScheduler(num_train_timesteps
    /// =1000, shift=1, use_dynamic_shifting=False)` plus the class defaults.
    /// The `shift=1` there is deliberate - the real shift arrives at
    /// `set_timesteps`, and applying it in both places would square it.
    fn default() -> Self {
        FlowUniPcConfig {
            num_train_timesteps: 1000,
            solver_order: 2,
            predict_x0: true,
            solver_type: BhSolver::Bh2,
            lower_order_final: true,
        }
    }
}

/// UniPC multistep in the flow-matching parameterisation - Wan2.1's default
/// sampler.
///
/// Each [`step`](Self::step) is a **corrector then predictor** pair: the fresh
/// model output first corrects the sample the previous step predicted (UniC),
/// and only then is a new sample predicted (UniP). That ordering means one
/// denoiser call per step, and it is why the corrector runs at the order chosen
/// by the *previous* step.
#[derive(Clone, Debug)]
pub struct FlowUniPcScheduler {
    cfg: FlowUniPcConfig,
    /// `N+1` sigmas: the `N` shifted step sigmas plus the terminal `0`.
    sigmas: Vec<f32>,
    /// `N` discrete timesteps (`trunc(σ·num_train_timesteps)`).
    timesteps: Vec<f32>,
    /// Converted (`x0`- or `ε`-space) model outputs, oldest first.
    outputs: Vec<Option<Vec<f32>>>,
    /// The sample the last predictor started from, i.e. what the next
    /// corrector corrects.
    last_sample: Option<Vec<f32>>,
    /// The order the last predictor ran at; the next corrector reuses it.
    this_order: usize,
    lower_order_nums: usize,
    step_index: usize,
}

impl FlowUniPcScheduler {
    pub fn new(cfg: FlowUniPcConfig) -> FlowUniPcScheduler {
        assert!(cfg.solver_order >= 1, "solver_order must be >= 1");
        FlowUniPcScheduler {
            cfg,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            outputs: vec![None; cfg.solver_order],
            last_sample: None,
            this_order: 0,
            lower_order_nums: 0,
            step_index: 0,
        }
    }

    /// Build the schedule the way Wan drives it: `set_timesteps(steps,
    /// shift=shift)` on a scheduler that was constructed with `shift = 1`.
    pub fn set_timesteps(&mut self, num_inference_steps: usize, shift: f64) {
        let s = unipc_sigmas(self.cfg.num_train_timesteps, num_inference_steps, shift);
        self.set_sigmas(&s);
    }

    /// Same, from an explicit already-shifted sigma grid (the `sigmas=` path of
    /// the reference's `set_timesteps`).
    pub fn set_sigmas(&mut self, shifted: &[f64]) {
        let (sigmas, timesteps) = finalize(shifted, self.cfg.num_train_timesteps);
        self.sigmas = sigmas;
        self.timesteps = timesteps;
        self.outputs = vec![None; self.cfg.solver_order];
        self.last_sample = None;
        self.this_order = 0;
        self.lower_order_nums = 0;
        self.step_index = 0;
    }

    /// The `N+1` sigmas (shifted step sigmas + terminal `0`).
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// The `N` discrete timesteps fed to the denoiser.
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    /// `convert_model_output`: the denoiser predicts flow, and the solver wants
    /// `x_0` (or `ε`). `x_0 = x - σ·v`, `ε = x - (1-σ)·v`.
    fn convert(&self, model_output: &[f32], sample: &[f32]) -> Vec<f32> {
        let sigma = self.sigmas[self.step_index] as f64;
        let k = if self.cfg.predict_x0 { sigma } else { 1.0 - sigma };
        sample.iter().zip(model_output).map(|(&x, &v)| (x as f64 - k * v as f64) as f32).collect()
    }

    /// The `(R, b)` system and the scalars shared by predictor and corrector.
    /// `rks` holds `(λ_si - λ_s0)/h` for each history entry, with a trailing
    /// `1.0`; `b` the `h·φ_k` coefficients.
    fn bh_system(&self, h: f64, rks: &[f64], order: usize) -> (Vec<Vec<f64>>, Vec<f64>, f64, f64) {
        let hh = if self.cfg.predict_x0 { -h } else { h };
        let h_phi_1 = hh.exp_m1();
        let b_h = match self.cfg.solver_type {
            BhSolver::Bh1 => hh,
            BhSolver::Bh2 => hh.exp_m1(),
        };
        let mut h_phi_k = h_phi_1 / hh - 1.0;
        let mut factorial_i = 1.0f64;
        let mut r = Vec::with_capacity(order);
        let mut b = Vec::with_capacity(order);
        for i in 1..=order {
            r.push(rks.iter().map(|&x| x.powi(i as i32 - 1)).collect::<Vec<f64>>());
            b.push(h_phi_k * factorial_i / b_h);
            factorial_i *= (i + 1) as f64;
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i;
        }
        (r, b, h_phi_1, b_h)
    }

    /// `(c_x, c_m0, scale)` of the update `c_x·x + c_m0·m0 + scale·Σ ρ_k·D1_k`,
    /// shared by predictor and corrector. The `predict_x0` branch integrates in
    /// data space and the other in noise space, which swaps the roles of `α`
    /// and `σ` throughout.
    fn update_coeffs(
        &self,
        sigma_t: f64,
        sigma_s0: f64,
        h_phi_1: f64,
        b_h: f64,
    ) -> (f64, f64, f64) {
        let (alpha_t, alpha_s0) = (1.0 - sigma_t, 1.0 - sigma_s0);
        if self.cfg.predict_x0 {
            (sigma_t / sigma_s0, -alpha_t * h_phi_1, -alpha_t * b_h)
        } else {
            (alpha_t / alpha_s0, -sigma_t * h_phi_1, -sigma_t * b_h)
        }
    }

    /// `rks` for the `order-1` history entries preceding index `newest`, plus
    /// the trailing `1.0` the reference appends.
    fn rks(&self, newest: usize, lambda_s0: f64, h: f64, order: usize) -> Vec<f64> {
        let mut rks = Vec::with_capacity(order);
        for i in 1..order {
            let lambda_si = lambda_of(self.sigmas[newest - i] as f64);
            rks.push((lambda_si - lambda_s0) / h);
        }
        rks.push(1.0);
        rks
    }

    /// UniP (B(h)): predict `x` at `σ_{i+1}` from `x` at `σ_i`.
    fn uni_p(&self, sample: &[f32], order: usize) -> Vec<f32> {
        let i = self.step_index;
        let (sigma_t, sigma_s0) = (self.sigmas[i + 1] as f64, self.sigmas[i] as f64);
        let h = lambda_of(sigma_t) - lambda_of(sigma_s0);

        let rks = self.rks(i, lambda_of(sigma_s0), h, order);
        let (r, b, h_phi_1, b_h) = self.bh_system(h, &rks, order);

        // `rhos_p` for order 2 is the reference's hard-coded 0.5, NOT
        // `solve(R[:-1,:-1], b[:-1])`. The two are only asymptotically equal
        // (as `h → 0`), so the simplification is part of the algorithm.
        let rhos_p: Vec<f64> = match order {
            1 => Vec::new(),
            2 => vec![0.5],
            _ => {
                let sub: Vec<Vec<f64>> =
                    r[..order - 1].iter().map(|row| row[..order - 1].to_vec()).collect();
                solve(sub, b[..order - 1].to_vec())
            }
        };

        // x_t = (σ_t/σ_s0)·x - α_t·h_φ_1·m0 - α_t·B_h·Σ ρ_k·(m_k - m0)/r_k
        // (predict_x0; the ε branch swaps the α/σ roles).
        let (c_x, c_m0, scale) = self.update_coeffs(sigma_t, sigma_s0, h_phi_1, b_h);
        let coeffs: Vec<f64> = rhos_p.iter().zip(&rks).map(|(&p, &rk)| scale * p / rk).collect();
        self.combine(sample, c_x, c_m0, &coeffs)
    }

    /// UniC (B(h)): correct the sample the previous predictor produced, now
    /// that the model has been evaluated at it.
    fn uni_c(&self, this_output: &[f32], last_sample: &[f32], order: usize) -> Vec<f32> {
        let i = self.step_index;
        let (sigma_t, sigma_s0) = (self.sigmas[i] as f64, self.sigmas[i - 1] as f64);
        let h = lambda_of(sigma_t) - lambda_of(sigma_s0);

        // The corrector's history is one step further back than the
        // predictor's: `si = step_index - (i+1)`.
        let rks = self.rks(i - 1, lambda_of(sigma_s0), h, order);
        let (r, b, h_phi_1, b_h) = self.bh_system(h, &rks, order);

        // Mirror of the predictor's shortcut, at the other end of the order
        // range: order 1 is the hard-coded 0.5, everything else a full solve.
        let rhos_c: Vec<f64> = if order == 1 { vec![0.5] } else { solve(r, b) };

        let (c_x, c_m0, scale) = self.update_coeffs(sigma_t, sigma_s0, h_phi_1, b_h);
        let coeffs: Vec<f64> =
            rhos_c[..order - 1].iter().zip(&rks).map(|(&p, &rk)| scale * p / rk).collect();
        let mut out = self.combine(last_sample, c_x, c_m0, &coeffs);
        // The corrector's extra term: the difference between the model output
        // at the predicted sample and the one at the step it came from.
        let m0 = self.outputs[self.outputs.len() - 1].as_ref().expect("previous output");
        let c = scale * rhos_c[order - 1];
        for ((o, &t), &m) in out.iter_mut().zip(this_output).zip(m0) {
            *o = (*o as f64 + c * (t as f64 - m as f64)) as f32;
        }
        out
    }

    /// `c_x·x + c_m0·m0 + Σ coeffs[k]·(m_k - m0)`, where `m_k` walks the stored
    /// outputs from newest to oldest (one `coeffs` entry per history term, so
    /// the caller's order choice is what bounds the walk).
    fn combine(&self, x: &[f32], c_x: f64, c_m0: f64, coeffs: &[f64]) -> Vec<f32> {
        let n = self.outputs.len();
        let m0 = self.outputs[n - 1].as_ref().expect("current output stored");
        let mut out: Vec<f32> = x
            .iter()
            .zip(m0)
            .map(|(&v, &m)| (c_x * v as f64 + c_m0 * m as f64) as f32)
            .collect();
        for (k, &c) in coeffs.iter().enumerate() {
            let mk = self.outputs[n - 2 - k].as_ref().expect("history output stored");
            for ((o, &a), &b) in out.iter_mut().zip(mk).zip(m0) {
                *o = (*o as f64 + c * (a as f64 - b as f64)) as f32;
            }
        }
        out
    }

    /// One UniPC step. `model_output` is the (CFG-combined) flow prediction at
    /// the current timestep, `sample` the current latent; returns the next
    /// latent. Call once per entry of [`timesteps`](Self::timesteps), in order.
    pub fn step(&mut self, model_output: &[f32], sample: &[f32]) -> Vec<f32> {
        assert_eq!(sample.len(), model_output.len(), "sample/model_output length mismatch");
        let i = self.step_index;
        let n = self.timesteps.len();
        assert!(i < n, "step() past the end of the schedule");

        let converted = self.convert(model_output, sample);
        // `disable_corrector` is empty in every Wan configuration, so the only
        // gate is "there is a previous prediction to correct".
        let sample: Vec<f32> = match (i > 0, self.last_sample.take()) {
            (true, Some(last)) => self.uni_c(&converted, &last, self.this_order),
            _ => sample.to_vec(),
        };

        for j in 0..self.outputs.len() - 1 {
            self.outputs[j] = self.outputs[j + 1].take();
        }
        let last_slot = self.outputs.len() - 1;
        self.outputs[last_slot] = Some(converted);

        // Order for this step: capped by what is left of the schedule when
        // `lower_order_final` is on, and by how much history has accumulated.
        let this_order = if self.cfg.lower_order_final {
            self.cfg.solver_order.min(n - i)
        } else {
            self.cfg.solver_order
        };
        self.this_order = this_order.min(self.lower_order_nums + 1);
        assert!(self.this_order > 0, "solver order collapsed to zero");

        let prev = self.uni_p(&sample, self.this_order);
        self.last_sample = Some(sample);
        if self.lower_order_nums < self.cfg.solver_order {
            self.lower_order_nums += 1;
        }
        self.step_index += 1;
        prev
    }
}

// ---------------------------------------------------------------------------
// DPM-Solver++ (multistep), flow-matching parameterisation
// ---------------------------------------------------------------------------

/// `FlowDPMSolverMultistepScheduler`'s defaults, which are the ones Wan's
/// `--sample_solver dpm++` path gets: `algorithm_type = "dpmsolver++"` (data
/// prediction), `solver_type = "midpoint"`, order 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlowDpmSolverConfig {
    pub num_train_timesteps: u32,
    pub solver_order: usize,
    pub solver_type: SolverType,
}

impl Default for FlowDpmSolverConfig {
    fn default() -> Self {
        FlowDpmSolverConfig {
            num_train_timesteps: 1000,
            solver_order: 2,
            solver_type: SolverType::Midpoint,
        }
    }
}

/// DPM-Solver++(2M) in the flow-matching parameterisation - Wan2.1's
/// `--sample_solver dpm++`.
///
/// Same update as [`crate::discrete::DpmSolverPlusPlusScheduler`] once written
/// in `(α_t, σ_t)`, but reached from `α = 1 - σ` instead of from
/// `alphas_cumprod`, and fed a different schedule (see the module header).
#[derive(Clone, Debug)]
pub struct FlowDpmSolverPlusPlusScheduler {
    cfg: FlowDpmSolverConfig,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    outputs: Vec<Option<Vec<f32>>>,
    lower_order_nums: usize,
    step_index: usize,
}

impl FlowDpmSolverPlusPlusScheduler {
    pub fn new(cfg: FlowDpmSolverConfig) -> FlowDpmSolverPlusPlusScheduler {
        assert!(
            cfg.solver_order == 1 || cfg.solver_order == 2,
            "orders 1 and 2 are implemented, not {}",
            cfg.solver_order
        );
        FlowDpmSolverPlusPlusScheduler {
            cfg,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            outputs: vec![None; cfg.solver_order],
            lower_order_nums: 0,
            step_index: 0,
        }
    }

    /// Wan's path: [`sampling_sigmas`] then the reference's `retrieve_timesteps
    /// (sigmas=...)`. The second shift the reference applies there is the
    /// identity (the scheduler is constructed with `shift = 1`), so the grid
    /// arrives already shifted.
    pub fn set_timesteps(&mut self, num_inference_steps: usize, shift: f64) {
        let s = sampling_sigmas(num_inference_steps, shift);
        self.set_sigmas(&s);
    }

    pub fn set_sigmas(&mut self, shifted: &[f64]) {
        let (sigmas, timesteps) = finalize(shifted, self.cfg.num_train_timesteps);
        self.sigmas = sigmas;
        self.timesteps = timesteps;
        self.outputs = vec![None; self.cfg.solver_order];
        self.lower_order_nums = 0;
        self.step_index = 0;
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    pub fn step(&mut self, model_output: &[f32], sample: &[f32]) -> Vec<f32> {
        assert_eq!(sample.len(), model_output.len(), "sample/model_output length mismatch");
        let i = self.step_index;
        let n = self.timesteps.len();
        assert!(i < n, "step() past the end of the schedule");

        // `convert_model_output` for `dpmsolver++`: the data prediction.
        let sigma = self.sigmas[i] as f64;
        let x0: Vec<f32> = sample
            .iter()
            .zip(model_output)
            .map(|(&x, &v)| (x as f64 - sigma * v as f64) as f32)
            .collect();
        for j in 0..self.outputs.len() - 1 {
            self.outputs[j] = self.outputs[j + 1].take();
        }
        let last = self.outputs.len() - 1;
        self.outputs[last] = Some(x0);

        // The terminal `σ = 0` (`final_sigmas_type = "zero"`) makes the last
        // step first-order unconditionally, independent of the step count.
        let lower_order_final = i == n - 1;

        let (sigma_t, sigma_s0) = (self.sigmas[i + 1] as f64, self.sigmas[i] as f64);
        let alpha_t = 1.0 - sigma_t;
        let h = lambda_of(sigma_t) - lambda_of(sigma_s0);
        let e = (-h).exp() - 1.0;
        let c_x = sigma_t / sigma_s0;

        let m0 = self.outputs[last].as_ref().expect("current output stored");
        let out: Vec<f32> = if self.cfg.solver_order == 1
            || self.lower_order_nums < 1
            || lower_order_final
        {
            let c_m = -alpha_t * e;
            sample.iter().zip(m0).map(|(&x, &m)| (c_x * x as f64 + c_m * m as f64) as f32).collect()
        } else {
            let h0 = lambda_of(self.sigmas[i] as f64) - lambda_of(self.sigmas[i - 1] as f64);
            let r0 = h0 / h;
            let m1 = self.outputs[last - 1].as_ref().expect("previous output stored");
            let (c_d0, c_d1) = match self.cfg.solver_type {
                SolverType::Midpoint => (-alpha_t * e, -0.5 * alpha_t * e),
                SolverType::Heun => (-alpha_t * e, alpha_t * (e / h + 1.0)),
            };
            sample
                .iter()
                .zip(m0)
                .zip(m1)
                .map(|((&x, &a), &b)| {
                    let d1 = (a as f64 - b as f64) / r0;
                    (c_x * x as f64 + c_d0 * a as f64 + c_d1 * d1) as f32
                })
                .collect()
        };

        if self.lower_order_nums < self.cfg.solver_order {
            self.lower_order_nums += 1;
        }
        self.step_index += 1;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Wan defaults, from `generate.py`'s argument defaults: T2V runs 50
    /// steps at shift 5.0, I2V at 480p runs 40 at 3.0.
    #[test]
    fn wan_schedule_endpoints() {
        let mut s = FlowUniPcScheduler::new(FlowUniPcConfig::default());
        s.set_timesteps(50, 5.0);
        assert_eq!(s.sigmas().len(), 51);
        assert_eq!(s.timesteps().len(), 50);
        assert_eq!(s.timesteps()[0], 999.0, "UniPC starts one below the training top");
        assert_eq!(*s.sigmas().last().expect("terminal"), 0.0);
        for w in s.sigmas().windows(2) {
            assert!(w[0] > w[1], "sigmas must decrease: {} then {}", w[0], w[1]);
        }

        // The other solver's grid starts at exactly 1.0, so its first timestep
        // is 1000 and not 999.
        let mut d = FlowDpmSolverPlusPlusScheduler::new(FlowDpmSolverConfig::default());
        d.set_timesteps(40, 3.0);
        assert_eq!(d.timesteps()[0], 1000.0);
        assert_eq!(d.sigmas()[0], 1.0);
    }

    /// `shift = 1` is the identity, and every larger shift pushes mass toward
    /// the noisy end - the property the whole parameter exists for.
    #[test]
    fn shift_is_monotone_in_shift() {
        let base = unipc_sigmas(1000, 20, 1.0);
        let up = unipc_sigmas(1000, 20, 5.0);
        for (i, (&a, &b)) in base.iter().zip(&up).enumerate() {
            assert!(b >= a, "shift 5 must not lower sigma {i}: {b} < {a}");
        }
    }

    /// The first step of both solvers is a plain flow-matching Euler step -
    /// `x + (σ_next - σ)·v` - because the multistep history is empty and the
    /// `x0` conversion cancels. Derived by hand; shares no code with the B(h)
    /// machinery it checks.
    #[test]
    fn first_step_is_euler() {
        let x = [0.3f32, -1.2, 0.75, 2.0];
        let v = [0.11f32, 0.4, -0.9, 0.02];
        for &(shift, steps) in &[(5.0, 50), (3.0, 40)] {
            let mut s = FlowUniPcScheduler::new(FlowUniPcConfig::default());
            s.set_timesteps(steps, shift);
            let dt = (s.sigmas()[1] - s.sigmas()[0]) as f64;
            let got = s.step(&v, &x);
            for (i, &g) in got.iter().enumerate() {
                let want = x[i] as f64 + dt * v[i] as f64;
                assert!((g as f64 - want).abs() < 1e-6, "unipc[{i}]: {g} vs {want}");
            }

            let mut d = FlowDpmSolverPlusPlusScheduler::new(FlowDpmSolverConfig::default());
            d.set_timesteps(steps, shift);
            let dt = (d.sigmas()[1] - d.sigmas()[0]) as f64;
            let got = d.step(&v, &x);
            for (i, &g) in got.iter().enumerate() {
                let want = x[i] as f64 + dt * v[i] as f64;
                assert!((g as f64 - want).abs() < 1e-6, "dpm++[{i}]: {g} vs {want}");
            }
        }
    }

    /// The terminal `σ = 0` step must land exactly on the `x0` prediction: the
    /// `λ` there is `+∞`, and the arithmetic has to survive it rather than
    /// produce a NaN.
    #[test]
    fn final_step_lands_on_x0() {
        let x = [0.3f32, -1.2, 0.75];
        let v = [0.11f32, 0.4, -0.9];
        let mut s = FlowUniPcScheduler::new(FlowUniPcConfig::default());
        s.set_timesteps(4, 5.0);
        let mut cur = x.to_vec();
        for _ in 0..3 {
            cur = s.step(&v, &cur);
        }
        let sigma = *s.sigmas().get(3).expect("last step sigma") as f64;
        let out = s.step(&v, &cur);
        for (i, &g) in out.iter().enumerate() {
            let want = cur[i] as f64 - sigma * v[i] as f64;
            assert!((g as f64 - want).abs() < 1e-6, "[{i}]: {g} vs {want}");
        }
    }

    /// A 3x3 solve against a hand-checked system, since order-3 UniPC is the
    /// only user of the general path and nothing else in the crate solves.
    #[test]
    fn linear_solve_is_correct() {
        let a = vec![vec![2.0, 1.0, -1.0], vec![-3.0, -1.0, 2.0], vec![-2.0, 1.0, 2.0]];
        let x = solve(a, vec![8.0, -11.0, -3.0]);
        for (got, want) in x.iter().zip(&[2.0, 3.0, -1.0]) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
    }
}
