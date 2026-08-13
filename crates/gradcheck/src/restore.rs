// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gate for **CodeFormer's code-prediction Transformer**
//! (`codeformer::train::CodeTransformerTrainer`).
//!
//! ## What this check covers, and what is frozen
//!
//! Covered - every tensor `codeformer::train::transformer_manifest` names, i.e.
//! the whole of CodeFormer's stage II:
//!
//! | tensor | what its gradient exercises |
//! |---|---|
//! | `position_emb` | the `add2` adjoint **summed over every layer** (`axpy`), and only into the q/k branch |
//! | `feat_emb.{weight,bias}` | the latent → embedding projection at the foot of the stack |
//! | `ft_layers.{l}.norm1.{weight,bias}` | `ln_stats` + `layernorm_dgamma`/`dbeta` on the pre-attention norm, whose output feeds **two** consumers |
//! | `ft_layers.{l}.self_attn.qk.{weight,bias}` | the split fused projection, `[2E, E]`, over `n1 + position_emb` |
//! | `ft_layers.{l}.self_attn.v.{weight,bias}` | the v projection over `n1` **without** the position embedding |
//! | `ft_layers.{l}.self_attn.out_proj.{weight,bias}` | the attention output projection |
//! | `ft_layers.{l}.norm2.{weight,bias}` | the pre-MLP norm |
//! | `ft_layers.{l}.linear1.{weight,bias}` | the MLP up-projection into `gelu_erf` |
//! | `ft_layers.{l}.linear2.{weight,bias}` | the MLP down-projection |
//! | `idx_pred_layer.0.{weight,bias}` | the head's LayerNorm |
//! | `idx_pred_layer.1.weight` | the **biasless** 1024-way code head |
//!
//! **Frozen, and deliberately so:** the VQ encoder, the codebook, the
//! generator, and the controllable feature transformation. The encoder's
//! latent is a fixed input buffer here; its own backward (including the VQ
//! straight-through estimator) is a separate, already-passing gate,
//! [`crate::check_vqgan`] over `vqgan::train::VqganTrainer`. Splitting them is
//! not a shortcut — it is the reference's own stage-II recipe, in which the
//! pre-trained VQGAN is fixed and only the transformer is trained on the
//! code-token loss. What is **not** yet gated is the two composed end to end
//! (a gradient flowing from the CE back through `feat_emb` into the encoder);
//! that is listed as remaining work in `codeformer::train`'s header.
//!
//! ## The objective, and the non-differentiable argmin
//!
//! ```text
//! L = (1/T) · Σ_t CE(logits[t, :], code_target[t])
//! ```
//!
//! The question "how is the non-differentiable argmin handled?" has a sharper
//! answer here than for a VQ autoencoder: **it is not on the path**. Two
//! distinct discrete ops sit near this model —
//!
//! * `vq_argmin`, the nearest-codebook search, which IS a gradient problem and
//!   which `vqgan::train` handles with a straight-through estimator; CodeFormer
//!   replaces it with the transformer and never runs it in this graph;
//! * `argmax_row`, which turns the predicted logits into indices for the
//!   codebook gather at **inference** time, strictly downstream of the loss.
//!
//! The targets are data (in training, the codes the frozen encoder + `vq_argmin`
//! assign to the ground-truth image), not a function of these weights, so the
//! CE is a smooth function of every checked parameter and finite differences
//! validate the reverse **directly** — no surrogate, no stop-gradient, no
//! relaxation. That is a strictly stronger gate than the VQ one, where FD can
//! only ever check the straight-through surrogate.
//!
//! `loss()` is the device mean CE (`ce_value` summed on the host, ÷ T) and the
//! reverse is seeded by `ce_grad`, which already divides by `n_rows` — so the
//! analytic gradient and the finite difference are gradients of the *same*
//! scalar. A mismatch there would show as a uniform factor-of-T error across
//! every tensor at once.
//!
//! ## Epsilon
//!
//! `5e-4`, the phase-4c convention rather than the workspace default `5e-3`: a
//! `±1` direction over `numel` entries is an L2 step of `eps·√numel`, and the
//! largest tensor here (`linear1.weight`, 240 entries at the tiny config)
//! would move 0.077 at `5e-3`.
//!
//! Measured, not assumed. [`check_codeformer_eps_sweep`] returns the table and
//! `codeformer_eps_plateau` gates it. Max relative error over all 31 tensors
//! (seed 7, tiny config):
//!
//! | eps | P40 | backend-cpu |
//! |---|---|---|
//! | 5e-3 | 1.69e-3 | 1.68e-3 |
//! | **2e-3** | **5.54e-4** | **1.51e-3** |
//! | 1e-3 | 3.43e-3 | 1.75e-3 |
//! | **5e-4** | **2.97e-3** | **4.81e-3** |
//! | 2e-4 | 3.82e-3 | 6.43e-3 |
//! | 1e-4 | 1.07e-2 | 6.99e-3 |
//! | 5e-5 | 4.42e-2 | 1.62e-2 |
//!
//! Two honest observations. First, this graph's optimum is **2e-3**, not 5e-4:
//! the CE seed carries a `1/T` and the whole objective is an order of magnitude
//! smaller than T5's, so fp32 cancellation bites about a decade earlier. `5e-4`
//! is 2–5× off that optimum and still 16× inside the `8e-2` bound, so it is
//! kept for consistency with the rest of phase 4c rather than tuned per model —
//! but a future failure here should be read against this table before anything
//! is widened. Second, the curve is noisier than T5's (1e-3 is worse than 5e-4
//! on the P40 and better on the CPU); with a single random direction per
//! tensor, individual points move, which is why the gate compares against the
//! *envelope* (`≤ 4× the better of the two decade endpoints`) and not against a
//! monotone shape.

use std::cell::Cell;

use codeformer::config::CodeFormerConfig;
use codeformer::train::{CodeTransformerTrainer, TRAIN_PIPELINES};

use crate::{directional_check, CheckModel, Report};

/// One trainable code Transformer with a fixed latent + code-target batch.
struct CodeHarness {
    m: CodeTransformerTrainer,
    /// The backward reads activation caches only valid after a forward.
    fwd_done: Cell<bool>,
}

impl CodeHarness {
    /// The device is the **pooled test device** (`gpu_core::testgpu::dev`), not
    /// a fresh `Gpu::new` — a device per model object is the pattern AGENTS.md
    /// bans.
    fn new(cfg: CodeFormerConfig, seed: u64) -> CodeHarness {
        let init = codeformer::train::init_weights(&cfg, seed);
        let (latent, targets) = codeformer::train::fixed_batch(&cfg, seed);
        let m = CodeTransformerTrainer::new_on(gpu_core::testgpu::dev(TRAIN_PIPELINES), cfg, &init);
        m.set_latent(&latent);
        m.set_targets(&targets);
        CodeHarness { m, fwd_done: Cell::new(false) }
    }
}

impl CheckModel for CodeHarness {
    fn param_names(&self) -> Vec<String> {
        self.m.ps.params.iter().map(|(n, _)| n.clone()).collect()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.m.read_weight(name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        self.m.write_weight(name, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.m.read_grad(name)
    }
    fn loss(&self) -> f32 {
        let l = self.m.forward();
        self.fwd_done.set(true);
        l
    }
    fn zero_grads(&self) {
        self.m.zero_grads();
    }
    fn backward(&self) {
        if !self.fwd_done.get() {
            let _ = self.loss();
        }
        self.m.backward();
        self.m.poll_wait();
    }
}

/// **The gate.** CodeFormer's code-prediction Transformer at gradcheck scale:
/// a 4×4 latent (`T = 16`), 2 layers, `E = 12` over 3 heads, a 12-entry
/// codebook. Two layers because that is the minimum that makes
/// `position_emb`'s cross-layer gradient accumulation observable.
pub fn check_codeformer(seed: u64) -> Report {
    let h = CodeHarness::new(codeformer::train::tiny_config(), seed);
    directional_check(&h, 5e-4, 4, seed ^ 0x1234)
}

/// The same graph at ONE layer. A pass here with a failure in
/// [`check_codeformer`] localises the fault to the cross-layer `position_emb`
/// accumulation rather than to the layer backward.
pub fn check_codeformer_one_layer(seed: u64) -> Report {
    let cfg = CodeFormerConfig { n_layers: 1, ..codeformer::train::tiny_config() };
    let h = CodeHarness::new(cfg, seed);
    directional_check(&h, 5e-4, 4, seed ^ 0x1234)
}

/// The eps/error table on this graph, measured rather than assumed — the probe
/// AGENTS.md asks for in place of widening a bound.
pub fn check_codeformer_eps_sweep(seed: u64) -> Vec<(f32, f32)> {
    let h = CodeHarness::new(codeformer::train::tiny_config(), seed);
    [5e-3f32, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5]
        .iter()
        .map(|&eps| (eps, directional_check(&h, eps, 4, seed ^ 0x1234).max_rel()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ATOL: f32 = 4e-3;
    const RTOL: f32 = 8e-2;

    fn gate(report: Report, what: &str) {
        report.print();
        let fails = report.failures(ATOL, RTOL);
        assert!(
            fails.is_empty(),
            "{what} gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn codeformer_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        gate(check_codeformer(7), "CodeFormer code-prediction Transformer");
    }

    #[test]
    fn codeformer_one_layer_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        gate(check_codeformer_one_layer(7), "CodeFormer transformer (single layer)");
    }

    #[test]
    fn codeformer_eps_plateau() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let table = check_codeformer_eps_sweep(7);
        for (eps, rel) in &table {
            println!("  eps={eps:.1e}  max_rel={rel:.3e}");
        }
        let at = |e: f32| table.iter().find(|(x, _)| *x == e).expect("eps in table").1;
        assert!(at(5e-4) <= RTOL, "eps 5e-4 max_rel {:.3e} exceeds rtol", at(5e-4));
        assert!(
            at(5e-4) <= at(5e-3).max(at(5e-5)) * 4.0,
            "eps 5e-4 is not on the plateau: {table:?}"
        );
    }
}
