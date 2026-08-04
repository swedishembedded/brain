// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gate for the **CLIP text tower**'s backward
//! (`clip::model::ClipText::new_train_on`).
//!
//! `ClipText` does not implement `model::Model` — its batch is token ids plus a
//! derived EOS-pooling row set, and its objective is not a token head — so the
//! blanket `impl<M: model::Model> CheckModel for M` does not apply. This file
//! supplies a direct [`CheckModel`] instead, in the shape of
//! `crates/yolo/tests/p2_blocks.rs`: a harness owning one model, one fixed
//! batch, and one fixed random proxy vector per output.
//!
//! ## The objective
//!
//! ```text
//! L = <r_hidden, final_layer_norm(x_L)>  +  <r_out, tower_output>
//! ```
//! with `tower_output = text_projection(pooled)` when the config projects and
//! `pooled` otherwise. Two seeds rather than one because CLIP genuinely has two
//! consumers (SDXL conditions on the sequence AND on the pooled/projected
//! embedding), and because it is what makes `d_hidden` an ACCUMULATION — the
//! sequence seed plus the EOS gather's `emb_bwd` scatter. A single-seed proxy
//! would leave that add untested.
//!
//! `L` is exactly linear in the outputs, so `backward()` seeds the two output
//! grads with `r_hidden` / `r_out` directly — the standard `L = <r, y>` trick
//! (`d L / d y = r`) that turns a whole graph into one differentiable scalar
//! without inventing a loss head the model does not have.
//!
//! ## What each entry point covers
//!
//! | fn | config | what it adds |
//! |---|---|---|
//! | [`check_clip`] | CLIP-L shape: `quick_gelu`, no projection | the default gate |
//! | [`check_clip_bigg`] | OpenCLIP-bigG shape: `gelu_erf` + `text_projection` | the second activation and the projection head |
//! | [`check_clip_tiled`] | `hidden = 128`, `B*T = 128` | forces `block::pick_gemm` onto `matmul_{dx,dw}_reg` — the tiny configs only ever select the naive backward GEMMs |
//!
//! Between them these differentiate: the token embedding scatter (`emb_bwd`),
//! the learned positional scatter (`pos_bwd`, whose `max_positions > T` tail
//! rows must stay zero), causal MHA (`attn_bwd_{dscores,dv,dq,dk}` over the
//! per-layer `probs` cache — no recompute), the fused qkv bias row-sum, both
//! MLP activations (**including the new `quick_gelu_bwd`**), every LayerNorm's
//! `dgamma`/`dbeta`/`dx`, both residual re-joins, the EOS row gather's adjoint,
//! and the optional projection.
//!
//! ## Epsilon
//!
//! `5e-3` for the tiny configs (the workspace default for a device model), but
//! `5e-4` for [`check_clip_tiled`]: a `±1` direction over `numel` elements is an
//! L2 step of `eps·sqrt(numel)`, and at 49 152 elements `5e-3` lands 1.1 away in
//! weight space — far outside the region where the model is locally linear. The
//! same reason `crates/yolo/tests/p3_gradcheck.rs` drops to `5e-4`.

use std::cell::Cell;

use data::rng::Rng;

use clip::config::{ClipTextConfig, TextAct};
use clip::model::{ClipText, TEXT_PIPELINES};

use crate::{directional_check, CheckModel, Report};

/// One trainable text tower, a fixed token batch, and the two fixed proxy
/// vectors that define `L`.
struct TextHarness {
    m: ClipText,
    /// `[B*T, H]` — the proxy direction on `final_layer_norm(x_L)`.
    r_hidden: Vec<f32>,
    /// `[B, P]` (projection) or `[B, H]` (pooled) — the proxy on the output.
    r_out: Vec<f32>,
    /// The activation caches the backward reads are only valid after a forward.
    /// `directional_check` always calls `loss()` before `backward()`, but a
    /// caller driving the harness by hand might not.
    fwd_done: Cell<bool>,
}

impl TextHarness {
    /// The device is the **pooled test device** (`gpu_core::testgpu::dev`), not
    /// a fresh `Gpu::new`: three entry points in one test binary, each building
    /// its own device, is the per-model-device pattern AGENTS.md bans — it
    /// deadlocks the driver under `--test-threads=8` and crashes on process
    /// exit. `testgpu::dev` keys on the pipeline slice address, so all three
    /// share one handle and it dies with the last of them.
    fn new(gpu: gpu_core::Gpu, cfg: ClipTextConfig, b: u32, t: u32, seed: u64) -> TextHarness {
        let init = clip::init::init_text_weights(&cfg, seed);
        let ids = clip::init::fixed_tokens(&cfg, b, t);
        let out_w = cfg.projection.unwrap_or(cfg.hidden) as usize;
        let n = (b * t) as usize;
        let h = cfg.hidden as usize;
        let m = ClipText::new_train_on(gpu, cfg, b, t, &init);
        m.set_tokens(&ids);
        let mut rng = Rng::new(seed ^ 0xC11F);
        let mut rand = |k: usize| -> Vec<f32> { (0..k).map(|_| rng.next_f32() - 0.5).collect() };
        TextHarness {
            r_hidden: rand(n * h),
            r_out: rand(b as usize * out_w),
            m,
            fwd_done: Cell::new(false),
        }
    }

    /// The tower output the second proxy vector multiplies.
    fn tower_out(&self) -> Vec<f32> {
        match self.m.read_text_embeds() {
            Some(te) => te,
            None => self.m.read_pooled(),
        }
    }
}

impl CheckModel for TextHarness {
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
        let seq: f32 =
            self.m.read_hidden().iter().zip(&self.r_hidden).map(|(y, r)| y * r).sum();
        let out: f32 = self.tower_out().iter().zip(&self.r_out).map(|(y, r)| y * r).sum();
        seq + out
    }
    fn zero_grads(&self) {
        self.m.zero_grads();
    }
    fn backward(&self) {
        if !self.fwd_done.get() {
            let _ = self.loss();
        }
        // dL/d(hidden) = r_hidden, dL/d(output) = r_out — L is linear in both.
        self.m.backward(&self.r_hidden, &self.r_out);
        self.m.poll_wait();
    }
}

/// Tiny CLIP text config. `heads = 2` over `hidden = 16` gives `head_dim = 8`;
/// `max_positions > t` on purpose, so `pos_bwd`'s untouched tail rows are part
/// of the checked tensor.
fn tiny(act: TextAct, projection: Option<u32>) -> ClipTextConfig {
    ClipTextConfig {
        hidden: 16,
        intermediate: 32,
        layers: 2,
        heads: 2,
        max_positions: 10,
        vocab: 23,
        act,
        eps: 1e-5,
        projection,
        bos_id: 21,
        // eos must be the largest id in the vocabulary — `ClipText::set_tokens`
        // pools at the row argmax and asserts it lands on eos. `pad == eos` is
        // the CLIP-L convention (`<|endoftext|>` is also the pad token), which
        // is the harder case: every pad slot ties the max.
        eos_id: 22,
        pad_id: 22,
    }
}

/// **The gate.** CLIP-L's text tower at gradcheck scale: `quick_gelu`, no
/// projection, 2 layers, B=2, T=8.
///
/// This is the entry point that covers `quick_gelu_bwd` (added with this
/// backward — `quick_gelu` had no derivative in the tree at all), the causal
/// attention quartet against the per-layer `probs` cache, and the token +
/// positional embedding scatters.
pub fn check_clip(seed: u64) -> Report {
    let h = TextHarness::new(gpu_core::testgpu::dev(TEXT_PIPELINES), tiny(TextAct::QuickGelu, None), 2, 8, seed);
    directional_check(&h, 5e-3, 4, seed ^ 0x1234)
}

/// OpenCLIP-bigG's shape at gradcheck scale: the **exact-erf** MLP activation
/// (`gelu_erf_bwd`) and the `text_projection` head on top of EOS pooling. The
/// two activations are NOT interchangeable — see
/// `crates/gradcheck/tests/quick_gelu_fd.rs`.
pub fn check_clip_bigg(seed: u64) -> Report {
    let h = TextHarness::new(gpu_core::testgpu::dev(TEXT_PIPELINES), tiny(TextAct::GeluErf, Some(12)), 2, 8, seed);
    directional_check(&h, 5e-3, 4, seed ^ 0x1234)
}

/// The same graph at dimensions that make `block::pick_gemm` select the
/// **register-tiled** backward GEMMs (`matmul_dx_reg` / `matmul_dw_reg`,
/// 128x128 output tile, 256 threads) instead of the naive per-output kernels:
/// every backward GEMM here has both output dims >= 128 (`B*T = 128`,
/// `hidden = 128`, `3*hidden = 384`, `intermediate = 256`).
///
/// One layer, because the point is kernel selection, not depth. `eps = 5e-4`:
/// see this module's header.
pub fn check_clip_tiled(seed: u64) -> Report {
    let cfg = ClipTextConfig {
        hidden: 128,
        intermediate: 256,
        layers: 1,
        heads: 4,
        max_positions: 70,
        vocab: 23,
        act: TextAct::QuickGelu,
        eps: 1e-5,
        projection: None,
        bos_id: 21,
        eos_id: 22,
        pad_id: 22,
    };
    let h = TextHarness::new(gpu_core::testgpu::dev(TEXT_PIPELINES), cfg, 2, 64, seed);
    directional_check(&h, 5e-4, 4, seed ^ 0x1234)
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
    fn clip_text_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        gate(check_clip(7), "CLIP-L text tower");
    }

    #[test]
    fn clip_bigg_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        gate(check_clip_bigg(7), "OpenCLIP-bigG text tower (gelu_erf + projection)");
    }

    #[test]
    fn clip_tiled_gemm_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        gate(check_clip_tiled(7), "CLIP text tower (register-tiled backward GEMMs)");
    }
}
