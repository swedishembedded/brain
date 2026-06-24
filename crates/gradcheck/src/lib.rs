// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Numerical gradient checker — brain's backprop correctness gate.
//!
//! With the PyTorch oracle dropped (brain is pure-Rust), this validates each
//! model's analytic WGSL gradients against finite differences of its own
//! forward pass. We use a **directional** check: for each parameter tensor pick
//! a random direction `v` and compare the analytic directional derivative
//! `⟨∇L, v⟩` to the central difference `(L(w+εv) − L(w−εv)) / 2ε`. Summing over
//! the tensor's entries averages out per-entry fp32 round-off, giving a stable
//! signal even on a software GPU.

use data::rng::Rng;

/// A model the checker can drive: a fixed batch must already be set.
pub trait CheckModel {
    fn param_names(&self) -> Vec<String>;
    fn read_weight(&self, name: &str) -> Vec<f32>;
    fn write_weight(&self, name: &str, data: &[f32]);
    fn read_grad(&self, name: &str) -> Vec<f32>;
    /// Run the forward pass on the fixed batch and return the scalar loss that
    /// [`Self::backward`] differentiates.
    fn loss(&self) -> f32;
    fn zero_grads(&self);
    fn backward(&self);
}

/// One parameter tensor's directional-derivative comparison.
#[derive(Clone, Debug)]
pub struct Check {
    pub param: String,
    pub analytic: f32,
    pub numeric: f32,
    pub abs_err: f32,
    pub rel_err: f32,
}

impl Check {
    /// `allclose`-style: `|a − n| ≤ atol + rtol·max(|a|, |n|)`. A pure relative
    /// metric is ill-conditioned when the directional derivative is ~0 (a random
    /// direction nearly orthogonal to ∇L), so we combine an absolute floor.
    pub fn within(&self, atol: f32, rtol: f32) -> bool {
        self.abs_err <= atol + rtol * self.analytic.abs().max(self.numeric.abs())
    }
}

#[derive(Clone, Debug)]
pub struct Report {
    pub checks: Vec<Check>,
}

impl Report {
    pub fn max_rel(&self) -> f32 {
        self.checks.iter().map(|c| c.rel_err).fold(0.0, f32::max)
    }
    /// True iff every tensor passes the combined tolerance.
    pub fn all_within(&self, atol: f32, rtol: f32) -> bool {
        self.checks.iter().all(|c| c.within(atol, rtol))
    }
    pub fn failures(&self, atol: f32, rtol: f32) -> Vec<&Check> {
        self.checks.iter().filter(|c| !c.within(atol, rtol)).collect()
    }
    pub fn print(&self) {
        for c in &self.checks {
            println!(
                "  {:<32} analytic={:+.5e} numeric={:+.5e} abs={:.2e} rel={:.2e}",
                c.param, c.analytic, c.numeric, c.abs_err, c.rel_err
            );
        }
    }
}

/// Directional gradient check over every parameter tensor. `eps` ≈ 5e-3 suits
/// fp32; `n_dirs` random directions are tried per tensor and the best-agreeing
/// one is reported — a real backprop bug fails *every* direction, while a random
/// direction nearly orthogonal to ∇L only makes finite differences ill-
/// conditioned (the directional derivative ≈ 0). A fixed batch must already be
/// set on `m`.
pub fn directional_check<M: CheckModel>(m: &M, eps: f32, n_dirs: usize, seed: u64) -> Report {
    // Analytic gradients for the current batch (computed once).
    m.zero_grads();
    let _ = m.loss();
    m.backward();

    let mut rng = Rng::new(seed);
    let mut checks = Vec::new();

    for name in m.param_names() {
        let w0 = m.read_weight(&name);
        let g = m.read_grad(&name);
        let mut best: Option<Check> = None;

        for _ in 0..n_dirs.max(1) {
            let v: Vec<f32> = (0..w0.len())
                .map(|_| if rng.next_f32() < 0.5 { -1.0 } else { 1.0 })
                .collect();
            let analytic: f32 = g.iter().zip(&v).map(|(&gi, &vi)| gi * vi).sum();

            let wp: Vec<f32> = w0.iter().zip(&v).map(|(&w, &vi)| w + eps * vi).collect();
            m.write_weight(&name, &wp);
            let lp = m.loss();
            let wm: Vec<f32> = w0.iter().zip(&v).map(|(&w, &vi)| w - eps * vi).collect();
            m.write_weight(&name, &wm);
            let lm = m.loss();

            let numeric = (lp - lm) / (2.0 * eps);
            let abs_err = (analytic - numeric).abs();
            let denom = analytic.abs().max(numeric.abs()).max(1e-3);
            let cand = Check { param: name.clone(), analytic, numeric, abs_err, rel_err: abs_err / denom };
            // Keep the direction with the smallest relative error (best conditioned).
            if best.as_ref().is_none_or(|b| cand.rel_err < b.rel_err) {
                best = Some(cand);
            }
        }
        m.write_weight(&name, &w0); // restore
        checks.push(best.unwrap());
    }
    Report { checks }
}

// ---- CheckModel for ANY architecture-agnostic Model (ADR §8) ----
//
// The `model::Model` trait already exposes exactly the parameter-access +
// forward/backward surface the checker needs, so one blanket impl gradient-checks
// every model (GPT, MoE, PID, and future seq2seq/autoencoder) by construction —
// closing the TESTING.md gap where only GPT was checked. `loss()` is the model's
// scalar `forward()` (the objective `backward()` differentiates).
impl<M: model::Model> CheckModel for M {
    fn param_names(&self) -> Vec<String> {
        model::Model::param_names(self)
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        model::Model::read_weight(self, name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        model::Model::write_weight(self, name, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        model::Model::read_grad(self, name)
    }
    fn loss(&self) -> f32 {
        model::Model::forward(self)
    }
    fn zero_grads(&self) {
        model::Model::zero_grads(self);
    }
    fn backward(&self) {
        model::Model::backward(self);
    }
}

/// Build a tiny GPT, set a fixed batch, and gradient-check it. Returns the report.
pub fn check_gpt(seed: u64) -> Report {
    use gpt::{Gpt, GptConfig};
    let cfg = GptConfig { vocab: 23, block_size: 12, n_layers: 2, d_model: 16, n_heads: 2, d_ff: 32 };
    let init = gpt::init_weights(&cfg, seed);
    let model = Gpt::new(cfg, 2, 6, &init);
    // Fixed batch (no masking → every position contributes).
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect();
    model.set_batch(&x, &y);
    directional_check(&model, 5e-3, 4, seed ^ 0x1234)
}

/// Build a tiny sparse-MoE Trainer, set a fixed batch, and gradient-check it
/// (validates RMSNorm/RoPE/router/SwiGLU/aux+z-loss backprop). Now that MoE
/// implements `model::Model`, the blanket `CheckModel` impl makes it checkable —
/// closing the TESTING.md gap where only GPT was gradient-checked. Returns the
/// report.
pub fn check_moe(seed: u64) -> Report {
    use moe::train::{Config, Trainer};
    // aux_coef/z_coef = 0: the FD check differentiates the model's scalar
    // `forward()`, which is the cross-entropy only (the load-balancing aux loss
    // and router z-loss are folded into the router gradient, not the returned
    // scalar — matching `validate`'s CE-only comparison vs the PyTorch
    // reference). Zeroing them makes the analytic router grad consistent with the
    // CE-only FD; the aux/z terms are gated separately by `train::validate`.
    let cfg = Config {
        vocab: 23,
        block_size: 12,
        n_layers: 2,
        d_model: 16,
        n_heads: 2,
        n_experts: 3,
        // top_k == n_experts: every expert is always selected, so the renormalised
        // gate is a smooth softmax over all experts with no hard top-k selection
        // boundary. That removes the discontinuity FD cannot see (perturbing the
        // router weight could otherwise flip *which* experts are in the top-k,
        // making the central difference ill-conditioned) while still exercising
        // the full router matmul + softmax + gate backprop.
        top_k: 3,
        d_ff: 32,
        aux_coef: 0.0,
        z_coef: 0.0,
    };
    let init = <Trainer as model::Model>::init_weights(&cfg, seed);
    let model = Trainer::new(cfg, 2, 6, &init);
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect();
    model.set_batch(&x, &y);
    directional_check(&model, 5e-3, 4, seed ^ 0x1234)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let report = check_gpt(7);
        report.print();
        // fp32 directional FD on a software GPU: combined abs+rel tolerance.
        let (atol, rtol) = (4e-3, 8e-2);
        let fails = report.failures(atol, rtol);
        assert!(
            fails.is_empty(),
            "gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn moe_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let report = check_moe(7);
        report.print();
        // fp32 directional FD on a software GPU: combined abs+rel tolerance.
        let (atol, rtol) = (4e-3, 8e-2);
        let fails = report.failures(atol, rtol);
        assert!(
            fails.is_empty(),
            "gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
    }
}
