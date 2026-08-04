// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `check_arcface` — the finite-difference gate on `crates/facenet`'s ArcFace
//! training graph.
//!
//! # What this gates
//!
//! A tiny IResNet ([`facenet::train::ArcFaceTrainConfig::tiny`]: 5 residual
//! blocks — `layers = [2, 1, 1, 1]`, so FOUR strided blocks with a `downsample`
//! shortcut conv and ONE stride-1 block with an IDENTITY shortcut, which is a
//! separate arm of the block backward and 45 of IResNet-100's 49 blocks —
//! 32×32 input, embedding 8, 5 identities, batch 4) under the REAL
//! additive-angular-margin cross-entropy, driven through `directional_check`
//! over every trainable tensor. That covers, in one gate:
//!
//! * the **folded** conv backward (`conv2d_dx` / `conv2d_dw` + `bias_grad`) —
//!   the release folds BatchNorm into the convolutions, so the conv weight AND
//!   its bias are what move;
//! * **PReLU's learned per-channel slope** (`prelu_bwd` / `prelu_bwd_wg`), the
//!   one parameter in this architecture whose gradient is a per-channel
//!   REDUCTION and therefore the one that a barrier-only kernel gets silently
//!   wrong. `vision::PReLU::backward` selects on the QUERIED
//!   `DeviceCaps::workgroup_reductions`, so this check must be run on BOTH a GPU
//!   (`workgroup_reductions = true`, `prelu_bwd_wg`, `C*64` threads) and
//!   `BRAIN_DEVICE=cpu` (`false`, `prelu_bwd`, `C` threads) — a GPU-only run
//!   passes while every slope on the CPU backend stays frozen at its init value;
//! * **train-mode BatchNorm** (`bn_dstats`/`bn_dx`/`bn_dgamma`/`bn_dbeta`) on
//!   the three BNs the released graph keeps (`bn1` per block, `bn2`, `features`);
//! * the **pre-activation residual join** — IResNet's shortcut reads the block
//!   INPUT, not `bn1`'s output, so its gradient must be summed into the block's
//!   input grad and not into `bn1`'s;
//! * the `fc` matmul + bias (`matmul_dx`/`matmul_dw`/`bias_grad`);
//! * the margin head: `l2norm_scale_dx` on BOTH the embedding and the class
//!   centres, `matmul_dx`/`matmul_dw` for the cosine table, the new
//!   `arcface_margin_bwd`, and `ce_grad`.
//!
//! # What it does NOT gate
//!
//! * **SCRFD** and the **5-point alignment warp**. Those are preprocessing: the
//!   reference ArcFace recipe trains on pre-aligned crops, so the detector
//!   carries no recognition gradient. `crates/facenet` has no detector backward
//!   and this check does not pretend otherwise.
//! * `run_mean` / `run_var`. Train-mode BatchNorm never reads them and they
//!   carry no gradient; they are `Role::Frozen` and so are not in
//!   `param_names()`.
//! * The running-stat EMA (`bn_running`), which is off — it mutates state during
//!   the forward and would make `loss()` non-deterministic, which a central
//!   difference cannot survive.
//!
//! # Why the device style
//!
//! `facenet` is a device model composed from shipped WGSL, so this is
//! `check_seq2seq`'s style (fp32, on `--device`, `CheckModel` over the real
//! graph) and not `check_flux2`'s host-f64 oracle — a host oracle would be a
//! second implementation and would prove nothing about the kernels that ship.
//! `facenet` does not implement `model::Model` (there is no image/label `Batch`
//! variant), so this is a direct `CheckModel` impl, as in
//! `crates/yolo/tests/p2_blocks.rs`.

use data::rng::Rng;
use facenet::train::{ArcFaceTrainConfig, ArcFaceTrainer};

use crate::{directional_check, CheckModel, Report};

/// Newtype so the `CheckModel` impl can live here — `CheckModel` is this
/// crate's trait and `ArcFaceTrainer` is `facenet`'s type, so the impl has to
/// sit in one of the two (the same orphan-rule shape as
/// `crates/tts/tests/talker.rs`). It forwards; it computes nothing.
pub struct ArcFaceCheck(ArcFaceTrainer);

impl CheckModel for ArcFaceCheck {
    fn param_names(&self) -> Vec<String> {
        self.0.param_names()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.0.read_weight(name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        self.0.write_weight(name, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.0.read_grad(name)
    }
    fn loss(&self) -> f32 {
        self.0.loss()
    }
    fn zero_grads(&self) {
        self.0.zero_grads();
    }
    fn backward(&self) {
        self.0.backward();
    }
}

/// Per-tensor initialisation. Deliberately name-driven rather than one global
/// std: a BatchNorm `gamma` at ~0 collapses the whole activation, and a PReLU
/// slope at ~0 makes the negative half of the activation (the ONLY half that
/// contributes to `da`) numerically invisible — the check would pass while
/// telling you nothing about the slope gradient.
fn init_tensor(rng: &mut Rng, name: &str, numel: usize) -> Vec<f32> {
    if name.ends_with(".running_mean") {
        return vec![0.0; numel];
    }
    if name.ends_with(".running_var") {
        return vec![1.0; numel];
    }
    if name.ends_with("bn1.weight") || name == "bn2.weight" || name == "features.weight" {
        // BN gamma ~ 1
        return (0..numel).map(|_| 1.0 + 0.1 * (2.0 * rng.next_f32() - 1.0)).collect();
    }
    if name.ends_with("bn1.bias") || name == "bn2.bias" || name == "features.bias" {
        return (0..numel).map(|_| 0.1 * (2.0 * rng.next_f32() - 1.0)).collect();
    }
    if name.ends_with("prelu.weight") {
        // torch's nn.PReLU default is 0.25, jittered so the slopes are distinct
        // (identical slopes would hide a per-channel indexing bug in `da`).
        return (0..numel).map(|_| 0.25 + 0.1 * (2.0 * rng.next_f32() - 1.0)).collect();
    }
    if name.ends_with(".bias") {
        return (0..numel).map(|_| 0.05 * (2.0 * rng.next_f32() - 1.0)).collect();
    }
    // Conv / fc / head weights: a fan-in-free small uniform. The tensors here
    // are tiny (≤ 320 elements), so the usual Kaiming scaling would only make
    // the directional step harder to condition.
    (0..numel).map(|_| 0.3 * (2.0 * rng.next_f32() - 1.0)).collect()
}

/// Build the tiny ArcFace trainer on a fixed image+label batch and
/// gradient-check every trainable tensor.
///
/// Loss: **ArcFace additive angular margin cross-entropy**, `s = 8.0`,
/// `m = 0.5 rad` — the paper's margin, and a reduced scale (the paper's is 64)
/// because the margin kernels are exactly linear in `s` while `s = 64` over 5
/// classes saturates the softmax to the point where the central difference of
/// the mean CE is round-off. See `ArcFaceTrainConfig::tiny`.
///
/// `eps = 4e-4`, **measured, not guessed**. The directional probe perturbs EVERY
/// element of a tensor at once along a ±1 direction, so the L2 step is
/// `eps·√numel`. This objective is badly curved for its size — train-mode
/// BatchNorm makes the loss depend on the batch's own statistics, and a PReLU
/// puts a kink under every activation — so the usable window between truncation
/// error and fp32 round-off is narrow, and `eps` has to be placed in it rather
/// than picked. Two measurements on the P40, both at `(atol 4e-3, rtol 8e-2)`:
///
/// Failures over five independent DIRECTION seeds:
///
/// | eps | one direction | best of 3 |
/// |---|---|---|
/// | 1e-3 | 16 | **1** (`stem.conv.weight`) |
/// | 3e-4 | 12 | **0** |
/// | 1e-4 | 21 | 9 |
/// | 3e-5 | 41 | 19 |
///
/// and, over eight direction seeds at `n_dirs = 3`, the CLOSEST any comparison
/// came to failing (`abs_err / (atol + rtol·max(|a|,|n|))`, so 1.0 is a failure):
///
/// | eps | 2e-4 | 3e-4 | **4e-4** | 5e-4 | 7e-4 | 1e-3 |
/// |---|---|---|---|---|---|---|
/// | worst approach | 1.087 ✗ | 0.962 | **0.775** | 0.932 | 0.681 | 1.146 ✗ |
///
/// The passing band is `[3e-4, 7e-4]`; `4e-4` sits inside it with margin. The
/// two walls are different failure modes and both are real:
///
/// * **Above** the band, truncation. `1e-3` — the value this check originally
///   shipped with — passes on seed 7 but is SEED-LUCKY: it leans on the best-of-3
///   selection to hide a per-direction absolute error of order 1.
/// * **Below** it, round-off. The structurally-zero conv biases (a per-channel
///   constant shift that the following BatchNorm's mean subtraction annihilates,
///   so the true gradient IS zero) have a numeric derivative that is nothing but
///   quantised loss: at `4e-4` one fp32 ulp of a loss near 5.9 is
///   `4.77e-7 / 8e-4 = 6.0e-4`, and `atol` is only ~7 of those.
///
/// The directions that fail at `1e-3` are NOT a wrong backward — an eps sweep on
/// them converges to the analytic value (`layer1.0.conv2.weight`, analytic
/// −2.70e-1: −1.945 at 1e-2, −0.854 at 1e-3, −0.350 at 1e-4, −0.278 at 3e-5).
/// `arcface_gradients_are_robust_to_the_probe_direction` pins that the choice
/// stays inside the band. (`check_gpt`'s 5e-3 is calibrated for 16-element
/// tensors; `crates/yolo`'s 5e-4 for ~1e5.)
pub fn check_arcface(seed: u64) -> Report {
    let cfg = ArcFaceTrainConfig::tiny();
    // `ArcFaceTrainer::new` takes an `&dyn Fn` (the parameter LIST is only known
    // once the graph is built), so the sequential RNG rides in a RefCell.
    let rng = std::cell::RefCell::new(Rng::new(seed));
    let trainer =
        ArcFaceTrainer::new(cfg.clone(), &|name, numel| init_tensor(&mut rng.borrow_mut(), name, numel));

    // Fixed batch: a deterministic pseudo-image per sample and one identity each
    // (distinct labels, so every class column of `head.weight` is a target for
    // exactly one row and the margin branch is exercised on all of them).
    let mut brng = Rng::new(seed ^ 0xA5A5_5A5A);
    let n = (cfg.batch * 3 * cfg.arc.image_size * cfg.arc.image_size) as usize;
    let img: Vec<f32> = (0..n).map(|_| 2.0 * brng.next_f32() - 1.0).collect();
    let labels: Vec<u32> = (0..cfg.batch).map(|i| i % cfg.classes).collect();
    trainer.set_batch(&img, &labels);

    directional_check(&ArcFaceCheck(trainer), 4e-4, 3, seed ^ 0x1234)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The workspace-standard gate, same as every other brain model.
    const ATOL: f32 = 4e-3;
    const RTOL: f32 = 8e-2;

    /// MUST be run twice: once on the GPU and once with `BRAIN_DEVICE=cpu`.
    /// `vision::PReLU::backward` picks `prelu_bwd_wg` vs `prelu_bwd` on the
    /// queried `DeviceCaps::workgroup_reductions`, and the barrier variant
    /// returns an ALL-ZERO `da` on the CPU backend's split-at-barrier JIT — a
    /// GPU-only run of this test cannot see that.
    #[test]
    fn arcface_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let t0 = std::time::Instant::now();
        let report = check_arcface(7);
        println!(
            "check_arcface: {} tensors, worst rel-err {:.3e}, elapsed {:.2?}",
            report.checks.len(),
            report.max_rel(),
            t0.elapsed()
        );
        report.print();
        let fails = report.failures(ATOL, RTOL);
        assert!(
            fails.is_empty(),
            "arcface gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
    }

    /// A single passing run of `check_arcface(7)` proves less than it looks like:
    /// `directional_check` keeps the BEST-agreeing of its three random directions,
    /// so one lucky direction per tensor is enough to make the gate green. At the
    /// `eps = 1e-3` this check originally shipped with, that is exactly what was
    /// happening — 16 of the per-tensor comparisons fail when a single direction
    /// has to carry the check, and one still fails best-of-3 on an alternative
    /// direction seed. Re-run the whole gate on FIVE independent direction seeds
    /// and require every one to be clean; that is what makes `eps = 4e-4` a
    /// measured choice rather than the first value that happened to pass.
    ///
    /// This is the test that catches a re-tuned `eps` drifting off the floor of
    /// the truncation/round-off U-curve, in either direction.
    #[test]
    fn arcface_gradients_are_robust_to_the_probe_direction() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let cfg = ArcFaceTrainConfig::tiny();
        let rng = std::cell::RefCell::new(Rng::new(7));
        let trainer =
            ArcFaceTrainer::new(cfg.clone(), &|n, k| init_tensor(&mut rng.borrow_mut(), n, k));
        let mut brng = Rng::new(7 ^ 0xA5A5_5A5A);
        let n = (cfg.batch * 3 * cfg.arc.image_size * cfg.arc.image_size) as usize;
        let img: Vec<f32> = (0..n).map(|_| 2.0 * brng.next_f32() - 1.0).collect();
        let labels: Vec<u32> = (0..cfg.batch).map(|i| i % cfg.classes).collect();
        trainer.set_batch(&img, &labels);

        let m = ArcFaceCheck(trainer);
        for dir_seed in [0x11u64, 0x22, 0x33, 0x44, 0x55] {
            let report = crate::directional_check(&m, 4e-4, 3, dir_seed);
            let fails = report.failures(ATOL, RTOL);
            assert!(
                fails.is_empty(),
                "direction seed {dir_seed:#x}: {:?}",
                fails.iter().map(|c| (&c.param, c.analytic, c.numeric, c.rel_err)).collect::<Vec<_>>()
            );
        }
    }

    /// The PReLU slope gradient is the one this architecture can get silently
    /// wrong per backend, so pin it directly as well: every slope tensor must
    /// come back NON-ZERO. `prelu_bwd_wg` on the CPU backend returns all zeros
    /// while `dx` stays correct, which trains to a plausible loss.
    #[test]
    fn every_prelu_slope_gradient_is_nonzero() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let cfg = ArcFaceTrainConfig::tiny();
        let rng = std::cell::RefCell::new(Rng::new(11));
        let t = ArcFaceTrainer::new(cfg.clone(), &|n, k| init_tensor(&mut rng.borrow_mut(), n, k));

        let mut brng = Rng::new(0x1234);
        let n = (cfg.batch * 3 * cfg.arc.image_size * cfg.arc.image_size) as usize;
        let img: Vec<f32> = (0..n).map(|_| 2.0 * brng.next_f32() - 1.0).collect();
        let labels: Vec<u32> = (0..cfg.batch).map(|i| i % cfg.classes).collect();
        t.set_batch(&img, &labels);

        t.zero_grads();
        let _ = t.loss();
        t.backward();
        let mut seen = 0;
        for name in t.param_names().iter().filter(|n| n.ends_with("prelu.weight")) {
            let g = t.read_grad(name);
            assert!(
                g.iter().any(|v| v.abs() > 1e-9),
                "{name}: every per-channel PReLU slope gradient is zero — the `da` \
                 reduction ran the wrong kernel for this backend"
            );
            seen += 1;
        }
        assert_eq!(seen, 6, "stem + one PReLU per block (tiny() is [2,1,1,1] = 5 blocks)");
    }
}


