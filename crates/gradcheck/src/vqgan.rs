// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `check_vqgan` — the finite-difference gate on the VQGAN training graph.
//!
//! What it covers, and why each piece is here:
//!
//! * the **VQ straight-through estimator** — the one genuinely new gradient
//!   form in the imaging workstream. `vq_argmin` is piecewise constant and has
//!   no derivative; the graph emits `z_q_st = z + (z_q - z)` with the second
//!   term detached, so the generator's gradient reaches the encoder unchanged.
//!   The check is what distinguishes that from "the encoder gets nothing"
//!   (which trains to a plausible loss and never moves the encoder).
//! * the **codebook** and **commitment** terms, whose stop-gradients point in
//!   opposite directions (`L_cb = beta·||sg[z]-q||²/n` reaches the codebook
//!   only, `L_com = ||z-sg[q]||²/n` reaches the encoder only). Swapping them is
//!   invisible to a loss curve and visible to this check: `quantize.embedding.
//!   weight`'s only gradient path is `L_cb`, so a swap zeroes it.
//!
//!   **What this check CANNOT see** is which term `beta` multiplies: finite
//!   differences gate the backward against whatever forward is emitted, so a
//!   mis-weighted objective is self-consistent and passes. That half is pinned
//!   by reading `vqgan_arch.py:55`, and `beta` sits on the CODEBOOK term there
//!   (see `vqgan::train`'s module docs) — not on the commitment term the
//!   reference's own line-29 comment claims.
//! * every **encoder and generator** parameter through `vae::blocks::grad` —
//!   conv weight/bias (`conv2d_dw` + `nchw_nlc`→`bias_grad`, `conv2d_dx`),
//!   fused GroupNorm `gb[2C]` (`gn_dsum`/`gn_dx`/`gn_dgamma`/`gn_dbeta` over
//!   the retained `stats`), SiLU, the residual and shortcut fan-outs, the
//!   nearest-2× upsample (`upsample2_dx`), and the single-head spatial
//!   attention (`attn_bwd_{dscores,dv,dq,dk}_bidir` via
//!   `model::block::bidir_bwd`) whose q/k/v are one fused `qkv.w`/`qkv.b`.
//!
//! **The code assignment is frozen for the sweep.** `argmin` is piecewise
//! constant, so an `±eps` step on an encoder weight can move a latent row into a
//! different Voronoi cell and make the central difference straddle two pieces —
//! the same non-smoothness `check_moe` avoids by setting `top_k == n_experts`.
//! `VqganTrainer::latch_assignment` pins the indices once; every forward in the
//! sweep gathers those fixed rows, which makes the whole objective smooth.
//!
//! `eps = 5e-4`, not the workspace-default `5e-3`: a `±1` direction over `numel`
//! elements is an L2 step of `eps·sqrt(numel)`, and the largest tensor here is a
//! 3×3 conv whose `5e-3` step would land well inside the SiLU/GroupNorm
//! curvature. Reconstruction is **L1** (the real recipe's data term); the
//! perceptual (LPIPS) and adversarial terms of the published VQGAN recipe need
//! models this workspace does not have and are out of scope — they would attach
//! as further contributions to the same `d_out` seed.

use crate::{directional_check, CheckModel, Report};
use data::rng::Rng;
use std::cell::Cell;
use vae::blocks::Tensors;
use vqgan::{VqganConfig, VqganTrainer};

/// A tiny VQGAN — 10 encoder blocks and 10 generator blocks over an 8×8 image,
/// with an attention block on each side and one down/up level. The check is
/// about correctness, not scale; every kernel on the CodeFormer preset's path
/// is dispatched here at least once.
///
/// `attn_resolutions = [4]` with `img_size = 8` puts an `AttnBlock` after the
/// second encoder residual level and inside the generator, and `ch_mult =
/// [1,2]` gives exactly one `Downsample`/`Upsample` pair (so `conv2d_dx` runs
/// at stride 2 with the reference's asymmetric pad, and `upsample2_dx` runs).
pub fn tiny_config() -> VqganConfig {
    VqganConfig {
        in_channels: 3,
        out_channels: 3,
        nf: 4,
        ch_mult: vec![1, 2],
        res_blocks: 1,
        attn_resolutions: vec![4],
        img_size: 8,
        codebook_size: 6,
        emb_dim: 4,
        beta: 0.25,
        norm_groups: 2,
        norm_eps: 1e-6,
    }
}

/// Random weights for every tensor in `cfg.tensor_manifest()`.
///
/// Conv kernels get `U(-1,1)/sqrt(fan_in)`, GroupNorm gammas `1 ± 0.1` and
/// biases `±0.1` — a network whose activations stay O(1) through 20 blocks, so
/// the finite difference is not measuring an exponent. The codebook is
/// `U(-0.6, 0.6)`, the range the encoder's own output lands in, so the frozen
/// assignment spreads over several codes instead of collapsing onto one.
pub fn init_weights(cfg: &VqganConfig, seed: u64) -> Tensors {
    let mut rng = Rng::new(seed);
    let mut t = Tensors::new();
    for (name, shape) in cfg.tensor_manifest() {
        let n: usize = shape.iter().product();
        let u = |rng: &mut Rng| 2.0 * rng.next_f32() - 1.0;
        let data: Vec<f32> = match shape.len() {
            // GroupNorm affine: [C] weight (gamma) or bias (beta).
            1 if name.ends_with(".weight") => (0..n).map(|_| 1.0 + 0.1 * u(&mut rng)).collect(),
            1 => (0..n).map(|_| 0.1 * u(&mut rng)).collect(),
            // The codebook [K, D].
            2 => (0..n).map(|_| 0.6 * u(&mut rng)).collect(),
            // Conv weight [Cout, Cin, K, K]: fan_in = Cin*K*K.
            _ => {
                let s = 1.0 / ((n / shape[0]) as f32).sqrt();
                (0..n).map(|_| s * u(&mut rng)).collect()
            }
        };
        t.insert(name, (shape, data));
    }
    t
}

/// `CheckModel` over [`VqganTrainer`]. `fwd` mirrors `yolo`'s block harness:
/// `write_weight` invalidates the activation cache, and `backward` re-runs the
/// forward if the checker did not (it does, but a caller might not).
struct Harness {
    m: VqganTrainer,
    fwd: Cell<bool>,
}

impl CheckModel for Harness {
    fn param_names(&self) -> Vec<String> {
        self.m.param_names().into_iter().map(|(n, _)| n).collect()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.m.read_weight(name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        self.m.write_weight(name, data);
        self.fwd.set(false);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.m.read_grad(name)
    }
    fn loss(&self) -> f32 {
        let l = self.m.loss();
        self.fwd.set(true);
        l
    }
    fn zero_grads(&self) {
        self.m.zero_grads();
    }
    fn backward(&self) {
        if !self.fwd.get() {
            let _ = self.loss();
        }
        self.m.backward();
    }
}

/// [`tiny_config`] with the channel count pushed OVER `vae::blocks::grad`'s
/// GEMM-lowering threshold (`GEMM_CONV_BWD_MIN_COUT = 32`): `tiny_config`'s
/// max `cout` is 8, so every FD sweep through it exercises only the naive
/// `conv2d_dw`/`conv2d_dx` path - the `im2col_at` + `matmul_dw_reg`(+split-K) + `col2im`
/// composition a REAL 128–512-channel training step takes had zero
/// finite-difference coverage anywhere (audit F14). One level, no attention,
/// minimal blocks: the point is crossing the threshold, not re-covering what
/// `tiny_config` already covers.
pub fn lowered_config() -> VqganConfig {
    VqganConfig {
        in_channels: 3,
        out_channels: 3,
        nf: 32,
        ch_mult: vec![1],
        res_blocks: 1,
        attn_resolutions: vec![],
        img_size: 8,
        codebook_size: 6,
        emb_dim: 4,
        beta: 0.25,
        norm_groups: 2,
        norm_eps: 1e-6,
    }
}

/// Build the tiny VQGAN, install a fixed `(image, target)` pair, freeze the code
/// assignment, and gradient-check every parameter tensor. Returns the report.
///
/// Gate at the workspace tolerance `(atol, rtol) = (4e-3, 8e-2)`.
pub fn check_vqgan(seed: u64) -> Report {
    check_config(tiny_config(), seed, 5e-4)
}

/// [`check_vqgan`] over [`lowered_config`] — the finite-difference gate for
/// the GEMM-lowered conv backward. Only meaningful on a backend with
/// `workgroup_reductions` (the lowering is gated on it); on the CPU JIT this
/// re-runs the naive path, which the caller should skip (see the test).
///
/// `eps = 1.25e-4`, a quarter of [`check_vqgan`]'s: the same `eps·sqrt(numel)`
/// argument the module doc makes — `lowered_config`'s 32×32×3×3 convs have
/// 16× `tiny_config`'s elements, so the ±1 direction's L2 step is 4× longer
/// at equal eps. Measured on a P40: at 5e-4 one conv sat at rel 8.8e-2
/// (right at the 8e-2 gate, curvature not a wrong gradient); at 1.25e-4 the
/// report's max_rel is 1.24e-2 — the roughly-quadratic shrink
/// central-difference truncation predicts, which a genuinely wrong gradient
/// would not show.
pub fn check_vqgan_lowered(seed: u64) -> Report {
    check_config(lowered_config(), seed, 1.25e-4)
}

fn check_config(cfg: VqganConfig, seed: u64, eps: f32) -> Report {
    let tensors = init_weights(&cfg, seed);
    let gpu = gpu_core::testgpu::dev(vqgan::TRAIN_PIPELINES);
    let (h, w) = (cfg.img_size, cfg.img_size);
    let m = VqganTrainer::new(cfg.clone(), &tensors, h, w, gpu);

    // A fixed batch in the [-1,1] range VQGAN images live in, and a target that
    // is NOT the input (a perfect-reconstruction target would sit on the L1
    // kink for every pixel at once).
    let mut rng = Rng::new(seed ^ 0xC0DE_B00C);
    let n_in = (cfg.in_channels * h * w) as usize;
    let n_out = (cfg.out_channels * h * w) as usize;
    let image: Vec<f32> = (0..n_in).map(|_| 2.0 * rng.next_f32() - 1.0).collect();
    let target: Vec<f32> = (0..n_out).map(|_| 2.0 * rng.next_f32() - 1.0).collect();
    m.set_batch(&image, &target);

    let h = Harness { m, fwd: Cell::new(false) };
    // eps: caller-chosen per config — the `eps·sqrt(numel)` L2-step argument
    // in the module docs (5e-4 suits tiny_config's 576-element convs;
    // lowered_config's 16x-larger tensors need a quarter of that).
    directional_check(&h, eps, 3, seed ^ 0x1234)
}

#[cfg(test)]
mod tests {
    /// The gate. Lives in this file (rather than `tests/`) so it arrives with
    /// the entry point it gates and cannot become an orphan.
    #[test]
    fn vqgan_gradients_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let r = super::check_vqgan(7);
        r.print();
        let (atol, rtol) = (4e-3, 8e-2);
        println!("check_vqgan: {} tensors, max_rel = {:.3e}", r.checks.len(), r.max_rel());
        let bad = r.failures(atol, rtol);
        assert!(bad.is_empty(), "{} tensors outside tolerance: {:?}", bad.len(), bad);
    }

    /// The GEMM-lowered conv backward under FD (audit F14): `tiny_config`'s
    /// max cout of 8 never crosses `GEMM_CONV_BWD_MIN_COUT`, so the
    /// `im2col_at` + `matmul_dw_reg`(+split-K) + `col2im` path real training
    /// takes was reachable from NO finite-difference gate. Skipped where the
    /// lowering itself is gated off (`workgroup_reductions == false`, the CPU
    /// JIT) — there the lowered branch structurally cannot run, and re-running
    /// the naive path would be a vacuous green.
    #[test]
    fn vqgan_lowered_conv_backward_matches_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let gpu = gpu_core::testgpu::dev(vqgan::TRAIN_PIPELINES);
        if !gpu.caps().workgroup_reductions {
            eprintln!("skipping: backend has no workgroup_reductions, the GEMM-lowered conv backward cannot be selected here");
            return;
        }
        drop(gpu);
        let r = super::check_vqgan_lowered(7);
        r.print();
        let (atol, rtol) = (4e-3, 8e-2);
        println!("check_vqgan_lowered: {} tensors, max_rel = {:.3e}", r.checks.len(), r.max_rel());
        let bad = r.failures(atol, rtol);
        assert!(bad.is_empty(), "{} tensors outside tolerance: {:?}", bad.len(), bad);
    }
}
