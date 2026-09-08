// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gate for the RRDBNet (Real-ESRGAN) backward.
//!
//! Every kernel this backward needs already existed (`leaky_relu_bwd`,
//! `scale_add_dexp`, plus the ordinary conv/add/concat/upsample adjoints
//! `vae::blocks::grad` already carries) - the defect was that `crate::model`'s
//! `lrelu`/`residual` helpers called `Builder::push_step` for the LeakyReLU
//! activation and the `x + 0.2 * f(x)` residual, which records nothing on the
//! reverse-mode tape. `Trace::backward` silently skips any op whose output no
//! consumer claimed, so every conv weight upstream of an activation - which is
//! every weight in the net - got a silent ZERO gradient. This is the SECOND
//! occurrence of that exact bug class in this repo (the first was SDXL's
//! transformer half, `check_unet`'s own doc); the fix gave `vae::blocks::
//! Builder` real recorders (`leaky_relu`, `residual_scale`) and routed
//! `crate::model` through them.
//!
//! # Two checks, deliberately
//!
//! [`check_rrdbnet`] is a directional check over every trainable tensor - the
//! coverage gate, and [`Report::checks`]'s length is asserted against the tiny
//! config's own `param_list().len()` so a near-empty tape (the failure mode a
//! `push_step` bug produces) cannot pass "by accident".
//!
//! [`check_rrdbnet_elementwise`] is required IN ADDITION, on
//! `body.0.rdb1.conv5.weight` specifically - the FIRST RRDB's first dense
//! block's LAST conv, i.e. the residual input FARTHEST from the loss. Every
//! one of the net's `~4 * num_block` residual sites reads the SAME one-element
//! `_scale` buffer (`vae::blocks::Op::ScaleAdd`'s `scale`), which is exactly
//! the shared/folded-parameter shape `gradcheck::t5`'s cross-block `rel_bias`
//! precedent shows a directional check can pass on while being partially
//! wrong: the contraction onto a random ±1 direction can stay small even when
//! a share of the true gradient is missing, and best-of-`n_dirs` actively
//! selects the direction where it is smallest. `tiny_config`'s `num_block =
//! 2` is what makes this far site exist to test at all - at `num_block = 1`
//! there is only one RRDB and "farthest from the loss" collapses to "every
//! site", which would not distinguish a correct fold from a lucky one.
//!
//! # Tolerances - the OPPOSITE direction from `check_vqgan`'s tuning
//!
//! `tiny_config`'s largest tensor is `body.*.rdb*.conv5.weight` at
//! `[num_feat, num_feat + 4*num_grow_ch, 3, 3]` = `[8, 24, 3, 3]` = 1728
//! elements, three times `check_vqgan::tiny_config`'s largest (576) - so the
//! `eps·sqrt(numel)` L2-step argument (`gradcheck::vqgan`'s module doc makes it
//! in full) predicted starting from `check_vqgan`'s `5e-4` and going SMALLER,
//! roughly `5e-4 / sqrt(3) ≈ 2.9e-4`.
//!
//! That prediction was measured wrong, in the informative direction. At
//! `eps = 2.5e-4`, `check_rrdbnet` measured `max_rel = 8.917e-2` - right at the
//! `8e-2` boundary, several `numeric` values reading exactly `0.0` against a
//! nonzero `analytic`. Halving eps (the standard next move when a directional
//! check sits near the boundary, per `check_vqgan_lowered`'s own precedent) made
//! it WORSE, not better: `eps = 1.25e-4` measured `max_rel = 2.023e-1`, with
//! `body.0.rdb2.conv5.weight` at `rel = 2.02e-1`. A halved eps that INCREASES
//! max_rel is not central-difference truncation shrinking - that would predict
//! the opposite - it is the other side of the finite-difference U-curve:
//! `tiny_config`'s per-weight gradients are themselves tiny (many `analytic`
//! values sit at 1e-6 to 1e-5, an order below `check_vqgan`'s), so `eps ·
//! ∂L/∂wᵢ` is small enough that the `mse_value` loss sum's own fp32 rounding
//! floor dominates it - `lp - lm` quantizes to whole ULPs at this magnitude,
//! which is exactly why so many `numeric` readings above are precisely `0.0`
//! or a round multiple of `~1.19e-4`. Going LARGER instead - `eps = 1e-3` -
//! clears that floor: `max_rel = 4.864e-2`, comfortably inside tolerance, and
//! [`check_rrdbnet_elementwise`] at the same `eps = 1e-3` measures
//! `max_rel = 8.877e-2` on `body.0.rdb1.conv5.weight`'s 1728 entries (a
//! per-entry check's atol floor is what makes THAT number, alone, not the
//! pass/fail signal - see [`crate::elementwise_check`]'s own doc). **The thing
//! to carry**: `eps·sqrt(numel)` predicts the TRUNCATION side of the tradeoff
//! correctly, but says nothing about the ROUND-OFF side, and a small-graph tiny
//! config with small per-weight gradients can be round-off-bound at an eps a
//! larger model would still find safely truncation-bound at - halving on a
//! near-boundary result is the right FIRST move to try, but confirm which side
//! of the U-curve you are on before concluding a smaller eps is the fix.

use std::cell::Cell;

use data::rng::Rng;
use rrdbnet::train::{RrdbTrainer, TRAIN_PIPELINES};
use rrdbnet::RrdbConfig;

use crate::{directional_check, elementwise_check, CheckModel, Report};

/// The image size the checks run at. `H != W` so a swapped H/W anywhere in the
/// conv/upsample plumbing cannot hide behind matching shapes.
const H: u32 = 4;
const W: u32 = 6;

/// A tiny RRDBNet: `num_feat = 8`, `num_grow_ch = 4` (differ, so a width swap
/// cannot hide - the same reasoning `rrdbnet::tests::parity` states for its own
/// tiny config), `scale = 4` (both upsample stages run), and **`num_block =
/// 2`** - non-negotiable, see [`check_rrdbnet_elementwise`]'s doc for why one
/// block would not exercise a "farthest from the loss" site at all.
pub fn tiny_config() -> RrdbConfig {
    RrdbConfig { in_channels: 3, out_channels: 3, num_feat: 8, num_grow_ch: 4, num_block: 2, scale: 4 }
}

/// Random weights for every tensor in `cfg.param_list()`. Conv weights get
/// `U(-1,1)/sqrt(fan_in)`, biases a small `U(-0.05,0.05)` - activations stay
/// O(1) through the trunk, so the finite difference measures a gradient, not
/// an exponent.
pub fn init_weights(cfg: &RrdbConfig, seed: u64) -> vae::blocks::Tensors {
    let mut rng = Rng::new(seed);
    let mut t = vae::blocks::Tensors::new();
    for (name, shape) in cfg.param_list() {
        let n: usize = shape.iter().product();
        let u = |rng: &mut Rng| 2.0 * rng.next_f32() - 1.0;
        let data: Vec<f32> = if name.ends_with(".bias") {
            (0..n).map(|_| 0.05 * u(&mut rng)).collect()
        } else {
            let fan_in = (n / shape[0]).max(1);
            let s = 1.0 / (fan_in as f32).sqrt();
            (0..n).map(|_| s * u(&mut rng)).collect()
        };
        t.insert(name, (shape, data));
    }
    t
}

/// Build a trainer at [`tiny_config`] with deterministic weights and a fixed
/// `(image, target)` batch. The target is NOT the network's own output at
/// init (a zero residual would make every gradient zero and the check
/// vacuously green).
fn trainer(seed: u64) -> RrdbTrainer {
    let cfg = tiny_config();
    let tensors = init_weights(&cfg, seed);
    let gpu = gpu_core::testgpu::dev(TRAIN_PIPELINES);
    let m = RrdbTrainer::new(gpu, cfg.clone(), &tensors, H, W);

    let mut rng = Rng::new(seed ^ 0x5DEC_0DE5);
    let mut r = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.next_f32()).collect() };
    let image = r((cfg.in_channels * H * W) as usize);
    let (oh, ow) = m.out_hw();
    let target = r((cfg.out_channels * oh * ow) as usize);
    m.set_inputs(&image, &target);
    m
}

struct Harness {
    m: RrdbTrainer,
    fwd: Cell<bool>,
}

impl CheckModel for Harness {
    fn param_names(&self) -> Vec<String> {
        self.m.params().iter().map(|(n, _)| n.clone()).collect()
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
        let l = self.m.forward();
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

/// Directional finite-difference check over every trainable tensor in the
/// graph. `eps = 1e-3` - LARGER than `check_vqgan`'s `5e-4` despite this
/// config's smaller tensors, because this config's per-weight gradients are
/// themselves smaller: see the module doc's tolerance section for the
/// round-off-vs-truncation tuning story (`2.5e-4` and `1.25e-4` were tried
/// first and both measured worse).
pub fn check_rrdbnet(seed: u64) -> Report {
    let h = Harness { m: trainer(seed), fwd: Cell::new(false) };
    directional_check(&h, 1e-3, 3, seed ^ 0x1234)
}

/// Per-ENTRY central differences on `body.0.rdb1.conv5.weight` - the residual
/// input farthest from the loss, whose gradient path reads the shared `_scale`
/// buffer at every one of the ~4·num_block residual sites between it and the
/// output. See the module doc for why `directional_check` alone cannot gate
/// this the way it gates an ordinary conv.
pub fn check_rrdbnet_elementwise(seed: u64) -> Report {
    let h = Harness { m: trainer(seed), fwd: Cell::new(false) };
    elementwise_check(&h, "body.0.rdb1.conv5.weight", 1e-3)
}

#[cfg(test)]
mod tests {
    /// The gate. Lives beside the entry point it gates (this repo's
    /// convention for a gradcheck entry point, matching `check_unet`'s
    /// placement) so it cannot become an orphan.
    #[test]
    fn rrdbnet_gradients_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let r = super::check_rrdbnet(7);
        r.print();
        let (atol, rtol) = (4e-3, 8e-2);
        println!("check_rrdbnet: {} tensors, max_rel = {:.3e}", r.checks.len(), r.max_rel());
        // Computed from the tiny config's ACTUAL trainable-tensor count, not
        // guessed: a near-empty tape (the push_step failure mode this backward
        // closes) must not pass "by accident" just because a handful of
        // tensors happen to check out.
        let expected = super::tiny_config().param_list().len();
        assert!(
            r.checks.len() > expected - 1,
            "only {} of {expected} tensors checked - the tape is not covering the graph",
            r.checks.len()
        );
        let bad = r.failures(atol, rtol);
        assert!(bad.is_empty(), "{} tensors outside tolerance: {:?}", bad.len(), bad);
    }

    /// The shared-`_scale`-buffer half of the gate - see
    /// [`super::check_rrdbnet_elementwise`]'s doc for why a directional check
    /// cannot replace it.
    #[test]
    fn rrdbnet_far_residual_gradient_matches_per_entry_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let r = super::check_rrdbnet_elementwise(11);
        println!("check_rrdbnet_elementwise: {} entries, max_rel = {:.3e}", r.checks.len(), r.max_rel());
        let bad = r.failures(4e-3, 8e-2);
        assert!(bad.is_empty(), "{} entries outside tolerance: {:?}", bad.len(), bad);
    }
}
