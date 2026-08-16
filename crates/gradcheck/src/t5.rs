// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gate for the **T5 encoder**'s backward
//! (`t5encoder::train::T5Trainer`).
//!
//! ## What this check covers, and what is frozen
//!
//! **Nothing is frozen.** The harness walks `ParamStore`'s full parameter list,
//! which is exactly [`t5encoder::config::T5Config::tensor_manifest`] - for
//! [`t5encoder::train::tiny_config`] that is 17 tensors:
//!
//! | tensor | shape | what its gradient exercises |
//! |---|---|---|
//! | `shared.weight` | `[vocab, D]` | the token-embedding scatter (`emb_bwd`) |
//! | `rel_bias.weight` | `[buckets, heads]` | **the learned relative-position bias** — `attn_bwd_dbias` summed over the batch, `axpy`-accumulated over the block stack, `nchw_nlc` (the adjoint of the forward's `nlc_nchw` permute), then the `emb_bwd` scatter over the bucket ids |
//! | `blocks.{l}.attn_norm.weight` | `[D]` | `rms_inv_eps` + `rmsnorm_dw` on the pre-attention norm |
//! | `blocks.{l}.qkv.weight` | `[3·heads·d_kv, D]` | the fused q‖k‖v GEMM's `matmul_dw`, fed by all three `attn_bwd_d{q,k,v}` regions |
//! | `blocks.{l}.o.weight` | `[D, heads·d_kv]` | the attention output projection |
//! | `blocks.{l}.ff_norm.weight` | `[D]` | the pre-FFN norm gain |
//! | `blocks.{l}.wi_0.weight` | `[d_ff, D]` | the gated-GELU **gate** branch (`gelu_bwd`, tanh form) |
//! | `blocks.{l}.wi_1.weight` | `[d_ff, D]` | the gated-GELU **up** branch (`mul`'s two adjoints) |
//! | `blocks.{l}.wo.weight` | `[D, d_ff]` | the FFN output projection |
//! | `final_norm.weight` | `[D]` | the encoder-output norm gain |
//!
//! Between them the sweep differentiates every kernel the reverse dispatches.
//! Three of those are T5-specific and are the reason this check exists rather
//! than being assumed from `check_clip`:
//!
//! * **no `1/√d_kv` attention scale** — the backward `attn_bwd_d{q,k}_bias`
//!   takes `scale` as a Param and must be given the forward's `1.0`. At
//!   `d_kv = 6` a wrongly-scaled `d_q`/`d_k` is off by 2.45×, which shows up in
//!   `qkv.weight` and nowhere else;
//! * **the relative-position bias is a learned embedding shared by every
//!   block** — `attn_bwd_dbias` *assigns*, so an implementation that dispatched
//!   it straight into one accumulator would keep only one block's contribution.
//!   **[`check_t5`] does not catch that**, and that is measured, not assumed:
//!   with the `axpy` fold deleted the tensor's gradient is 33 % wrong and this
//!   check still reports `rel = 6.2e-4` at seed 1 / `5.3e-2` at seed 7, both
//!   inside `(4e-3, 8e-2)`. `directional_check` contracts the tensor onto one
//!   ±1 direction and keeps the *best* of four, which is the wrong selection
//!   rule for a *partial* gradient error. The fold is covered by
//!   [`check_t5_rel_bias_elementwise`], a per-ENTRY check, and by nothing else;
//! * **RMSNorm with a runtime epsilon and no bias** — `rms_inv_eps` /
//!   `rmsnorm_dx_eps`, not the eps-hardcoded `rmsnorm_dx`.
//!
//! ## The objective
//!
//! ```text
//! L = <r, final_layer_norm(x_L)>
//! ```
//! with `r` a fixed random `[B·T, D]` direction. `L` is exactly linear in the
//! encoder output, so `backward()` seeds the output grad with `r` directly —
//! the standard `dL/dy = r` trick that turns the whole graph into one
//! differentiable scalar without inventing a head T5's encoder does not have
//! (it has no LM head; FLUX consumes `last_hidden_state` as-is). One seed, not
//! CLIP's two, because T5 has exactly one consumer.
//!
//! ## Epsilon
//!
//! `5e-4`, not the workspace default `5e-3`. A `±1` direction over `numel`
//! entries is an L2 step of `eps·√numel`; the largest tensor here is
//! `qkv.weight` at 864 entries, where `5e-3` would be a 0.147 step in weight
//! space — well outside the region where a two-layer stack with an unscaled
//! softmax is locally linear. `5e-4` puts it at 0.0147.
//!
//! That is not asserted, it is **measured** — [`check_t5_eps_sweep`] returns
//! the whole table and `t5_eps_plateau` gates it. On this graph (seed 7, tiny
//! config), max relative error over all 17 tensors:
//!
//! | eps | P40 | backend-cpu |
//! |---|---|---|
//! | 5e-3 | 2.92e-3 | 2.90e-3 |
//! | 2e-3 | 4.81e-4 | 4.61e-4 |
//! | **1e-3** | **2.21e-4** | **2.27e-4** |
//! | **5e-4** | **5.54e-4** | **3.05e-4** |
//! | 2e-4 | 3.12e-3 | 6.85e-4 |
//! | 1e-4 | 1.57e-3 | 2.82e-3 |
//! | 5e-5 | 1.89e-3 | 2.90e-3 |
//!
//! The U is the textbook one: truncation error dominates above 2e-3, fp32
//! cancellation below 2e-4, and 1e-3–5e-4 is the floor. `5e-4` sits inside it
//! on both backends — unlike the phase-4b depth/QARep block, where even 5e-4
//! was still above the truncation knee.

use std::cell::Cell;

use data::rng::Rng;

use t5encoder::config::T5Config;
use t5encoder::train::{T5Trainer, TRAIN_PIPELINES};

use crate::{directional_check, CheckModel, Report};

/// One trainable encoder, a fixed token batch, and the fixed proxy direction
/// that defines `L`.
struct T5Harness {
    m: T5Trainer,
    /// `[B*T, D]` — the proxy direction on `final_layer_norm(x_L)`.
    r: Vec<f32>,
    /// The backward reads activation caches that are only valid after a
    /// forward. `directional_check` always calls `loss()` first, but a caller
    /// driving the harness by hand might not.
    fwd_done: Cell<bool>,
}

impl T5Harness {
    /// The device is the **pooled test device** (`gpu_core::testgpu::dev`), not
    /// a fresh `Gpu::new`: several entry points share one test binary, and a
    /// device per model object is the pattern AGENTS.md bans.
    fn new(cfg: T5Config, b: u32, t: u32, seed: u64) -> T5Harness {
        let init = t5encoder::train::init_weights(&cfg, seed);
        let ids = t5encoder::train::fixed_tokens(&cfg, b, t);
        let n = (b * t) as usize * cfg.d_model as usize;
        let m = T5Trainer::new_on(gpu_core::testgpu::dev(TRAIN_PIPELINES), cfg, b, t, &init);
        m.set_tokens(&ids);
        let mut rng = Rng::new(seed ^ 0x715);
        T5Harness { r: (0..n).map(|_| rng.next_f32() - 0.5).collect(), m, fwd_done: Cell::new(false) }
    }
}

impl CheckModel for T5Harness {
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
        self.m.forward();
        self.fwd_done.set(true);
        // Accumulate in f64. The sum is a host reduction over `B·T·D` terms and
        // it is differenced by finite differences, so an f32 accumulator's
        // round-off lands directly in the numerator of `(L(w+ε) − L(w−ε))`.
        // `elementwise_check` perturbs ONE entry, so that difference is ~1e-3 of
        // `L` and the accumulator's noise is the binding error term.
        let dot: f64 =
            self.m.read_hidden().iter().zip(&self.r).map(|(y, r)| *y as f64 * *r as f64).sum();
        dot as f32
    }
    fn zero_grads(&self) {
        self.m.zero_grads();
    }
    fn backward(&self) {
        if !self.fwd_done.get() {
            let _ = self.loss();
        }
        // dL/d(hidden) = r — L is linear in the encoder output.
        self.m.backward(&self.r);
        self.m.poll_wait();
    }
}

/// **The gate.** The T5 encoder backward at gradcheck scale: 2 blocks, B=2,
/// T=6, and dims chosen so `heads ≠ d_kv` and `heads·d_kv ≠ d_model` (3, 6, 16)
/// — at XXL all three are 64/64/4096 and a swapped index would be invisible.
pub fn check_t5(seed: u64) -> Report {
    let h = T5Harness::new(t5encoder::train::tiny_config(), 2, 6, seed);
    directional_check(&h, 5e-4, 4, seed ^ 0x1234)
}

/// The same graph at a **single** block: a failure here is local to one block,
/// so it separates "the block backward is wrong" from anything cross-block.
///
/// It is a *localiser*, not a detector of the shared-bias fold — with
/// `layers = 1` the `axpy` fold is a no-op, and [`check_t5`] does not detect a
/// broken fold either (see the module header). [`check_t5_rel_bias_elementwise`]
/// is the detector.
pub fn check_t5_one_block(seed: u64) -> Report {
    let cfg = T5Config { layers: 1, ..t5encoder::train::tiny_config() };
    let h = T5Harness::new(cfg, 2, 6, seed);
    directional_check(&h, 5e-4, 4, seed ^ 0x1234)
}

/// The graph at dimensions that make `block::pick_gemm` select the
/// **register-tiled** backward GEMMs (`matmul_dx_reg` / `matmul_dw_reg`,
/// 128×128 output tile, 256 threads) instead of the naive per-output kernels:
/// every backward GEMM here has both output dims ≥ 128 (`B·T = 256`,
/// `d_model = 128`, `3·inner = 384`, `d_ff = 256`).
///
/// One block, because the point is kernel selection, not depth.
pub fn check_t5_tiled(seed: u64) -> Report {
    let cfg = T5Config {
        vocab: 23,
        d_model: 128,
        d_ff: 256,
        d_kv: 16,
        layers: 1,
        heads: 8,
        rel_buckets: 8,
        rel_max_distance: 6,
        eps: 1e-6,
        // The trainer implements T5 v1.1's shared bias and the unmasked
        // contract only; umT5's per-block variant has no backward.
        per_block_rel_bias: false,
        masked: false,
    };
    let h = T5Harness::new(cfg, 2, 128, seed);
    directional_check(&h, 5e-4, 4, seed ^ 0x1234)
}

/// **The gate that actually covers the cross-block fold.** Per-ENTRY finite
/// differences on `rel_bias.weight`, the one parameter in this graph whose
/// gradient is summed over the whole block stack.
///
/// [`check_t5`] does NOT cover that fold, and this is measured, not assumed:
/// deleting the `axpy` in `t5encoder::train::build_bwd_steps` - so `attn_bwd_dbias`
/// assigns straight into the accumulator and only the last-written block
/// survives — leaves a **33 %** error in this tensor's gradient (L2 of the
/// difference 0.672 against a gradient norm of 2.044 at seed 7) and
/// [`check_t5`] still passes: `rel = 6.2e-4` at seed 1 and `5.3e-2` at seed 7,
/// both inside `(4e-3, 8e-2)`. `directional_check` contracts the tensor onto one
/// ±1 direction and keeps the *best* of four, which is exactly the wrong
/// selection rule for a partial gradient error — see its rustdoc.
///
/// 24 entries at the tiny config, so this is 48 extra forwards of a
/// `d_model = 16`, 2-block graph.
///
/// `eps = 1e-2`, twenty times [`check_t5`]'s: a single-entry step has no
/// `√numel` amplification, so the loss difference here is `eps·|∂L/∂wᵢ|` with
/// `|∂L/∂wᵢ| ~ 0.3`, and below ~2e-3 fp32 cancellation dominates. Measured
/// max relative error over the 24 entries (seed 7):
///
/// | eps | 2e-2 | **1e-2** | 5e-3 | 2e-3 | 1e-3 | 5e-4 |
/// |---|---|---|---|---|---|---|
/// | P40 | 1.61e-2 | **1.67e-3** | 2.92e-3 | 4.12e-3 | 1.08e-1 | 3.29e-2 |
/// | backend-cpu | 1.41e-2 | **6.86e-3** | 1.52e-2 | 7.70e-3 | 9.48e-2 | 1.60e-1 |
///
/// `t5_rel_bias_elementwise_eps_plateau` gates that table. The relative metric
/// is dominated by the smallest-gradient entries (`|∂L/∂wᵢ| ~ 4e-2`); the gate
/// itself is `Check::within(4e-3, 8e-2)`, whose absolute floor is what those
/// entries are actually judged on.
pub fn check_t5_rel_bias_elementwise(seed: u64) -> Report {
    let h = T5Harness::new(t5encoder::train::tiny_config(), 2, 6, seed);
    crate::elementwise_check(&h, "rel_bias.weight", 1e-2)
}

/// The eps table behind [`check_t5_rel_bias_elementwise`]'s `5e-3`.
pub fn check_t5_rel_bias_eps_sweep(seed: u64) -> Vec<(f32, f32)> {
    let h = T5Harness::new(t5encoder::train::tiny_config(), 2, 6, seed);
    [2e-2f32, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4]
        .iter()
        .map(|&eps| (eps, crate::elementwise_check(&h, "rel_bias.weight", eps).max_rel()))
        .collect()
}

/// The eps/error relationship on this graph, measured rather than assumed.
///
/// Returns `(eps, max_rel_err)` over the whole tiny-config sweep. AGENTS.md's
/// rule when a gradcheck fails is to PROBE this table and report it, never to
/// widen the bound — the phase-4b depth/QARep finding was that even `5e-4` can
/// sit above the finite-difference truncation knee for some graphs, and the
/// only way to know is to measure.
pub fn check_t5_eps_sweep(seed: u64) -> Vec<(f32, f32)> {
    let h = T5Harness::new(t5encoder::train::tiny_config(), 2, 6, seed);
    [5e-3f32, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5]
        .iter()
        .map(|&eps| (eps, directional_check(&h, eps, 4, seed ^ 0x1234).max_rel()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// fp32 directional FD on a device: the workspace-standard combined
    /// tolerance.
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
    fn t5_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        gate(check_t5(7), "T5 encoder");
    }

    #[test]
    fn t5_one_block_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        gate(check_t5_one_block(7), "T5 encoder (single block)");
    }

    /// The eps probe, run as a gate rather than left as a comment: it asserts
    /// that the chosen `5e-4` is not sitting on a knee, i.e. that the max
    /// relative error at `5e-4` is no worse than at the decade on either side.
    /// If a future change to the graph moves the knee, this fails and prints
    /// the table — which is the reporting AGENTS.md asks for instead of a
    /// widened bound.
    #[test]
    fn t5_eps_plateau() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let table = check_t5_eps_sweep(7);
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

    /// The cross-block `rel_bias` fold, per entry. See
    /// [`check_t5_rel_bias_elementwise`] for why [`check_t5`] is not enough.
    #[test]
    fn t5_rel_bias_grad_is_the_sum_over_blocks() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        gate(check_t5_rel_bias_elementwise(7), "T5 rel_bias (per entry)");
    }

    /// The eps probe for the per-entry check, gated the same way as
    /// [`t5_eps_plateau`].
    #[test]
    fn t5_rel_bias_elementwise_eps_plateau() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let table = check_t5_rel_bias_eps_sweep(7);
        for (eps, rel) in &table {
            println!("  eps={eps:.1e}  max_rel={rel:.3e}");
        }
        let at = |e: f32| table.iter().find(|(x, _)| *x == e).expect("eps in table").1;
        assert!(at(1e-2) <= RTOL, "eps 1e-2 max_rel {:.3e} exceeds rtol", at(1e-2));
        assert!(
            at(1e-2) <= at(2e-2).max(at(1e-3)) * 4.0,
            "eps 1e-2 is not on the plateau: {table:?}"
        );
    }

    #[test]
    fn t5_tiled_gemm_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        gate(check_t5_tiled(7), "T5 encoder (register-tiled backward GEMMs)");
    }
}
