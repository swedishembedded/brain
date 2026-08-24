// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gate for the SDXL `UNet2DConditionModel` backward.
//!
//! Closing this was cheap in the sense that mattered - not one new kernel - but
//! not in the sense "the forward is all existing blocks, so the backward
//! composes existing adjoints" implies. The conv half WAS already
//! differentiable through `vae::blocks`. The transformer half was not, because
//! `sdxlunet::model::Rec` emitted its LayerNorms, linears, GEGLU,
//! self-attention, cross-attention and timestep broadcast with
//! `Builder::push_step`, which records nothing on the reverse-mode tape - so
//! every parameter upstream of one of those stages had a silent ZERO gradient,
//! not merely an unchecked one. The shared builder grew real recorders for
//! those stages, and this checks the result.
//!
//! # Why the whole graph is checked, not a block
//!
//! The UNet's parameters are not independent the way a transformer block's are.
//! Three things make a per-block check unable to see a real defect here:
//!
//! * **The timestep embedding is SHARED by all 17 resnets.** `time_embedding`
//!   and `add_embedding` feed one `silu(emb)` that every resnet's
//!   `time_emb_proj` consumes, so their gradients accumulate over 17
//!   contributors. That is exactly the folded/shared-parameter class where a
//!   partially-wrong gradient survives a directional check - see
//!   [`check_unet_conditioning_elementwise`], which is why it exists.
//! * **The skip connections cross the graph.** Every down-path output is
//!   consumed twice: once by the next down stage, once by an up-path concat
//!   nine stages later. A gradient that reaches only one of those consumers is
//!   a plausible number, not a shape error.
//! * **The text encoding feeds every cross-attention site.** Its `kv`
//!   projection is per-site, but `encoder_hidden_states` itself is one buffer
//!   read at up to 10 sites.
//!
//! # Tolerances
//!
//! `eps = 2.5e-4`. The `eps·sqrt(numel)` argument `gradcheck::vqgan`'s module
//! doc makes applies unchanged: `UNetConfig::tiny`'s largest tensor is the
//! `[128, 128]` time-embedding linear, between `tiny_config`'s and
//! `lowered_config`'s scale there, so the eps sits between their two.
//! Measured `max_rel` over all 263 tensors at this eps: 9.5e-2.

use std::cell::Cell;

use data::rng::Rng;
use sdxlunet::config::{UNetConfig, N_TIME_IDS};
use sdxlunet::train::{UnetTrainer, TRAIN_PIPELINES};

use crate::{directional_check, elementwise_check, CheckModel, Report};

/// The latent the checks run at. Small, but still a multiple of the 2x
/// downscale `UNetConfig::tiny`'s two levels need.
const H: u32 = 8;
const W: u32 = 8;
/// Text tokens. Deliberately NOT equal to `H*W` at either level, so a swapped
/// query/key length in cross-attention cannot hide behind matching shapes.
const T_ENC: u32 = 5;

/// [`UNetConfig::tiny`] with a much narrower conditioning chain, for the
/// per-ENTRY check only.
///
/// `elementwise_check` costs `2·numel` forwards. At `time_embed_dim = 128`
/// that is tens of thousands of full UNet forwards per conditioning tensor -
/// hours, i.e. a gate nobody would run, which this repo counts as no gate at
/// all. At 16 it is ~2k, and it catches the SAME defect class: the point is
/// that all 17 resnets consume `silu(emb)`, and dropping some of those
/// contributors is just as visible in a 16-wide embedding as a 128-wide one.
fn narrow_conditioning() -> UNetConfig {
    UNetConfig { time_embed_dim: 16, ..UNetConfig::tiny() }
}

/// Build a trainer at [`UNetConfig::tiny`] with deterministic weights and a
/// fixed batch.
fn trainer(seed: u64) -> UnetTrainer {
    trainer_with(UNetConfig::tiny(), seed)
}

fn trainer_with(cfg: UNetConfig, seed: u64) -> UnetTrainer {
    let tensors = sdxlunet::init::init_weights(&cfg, seed);
    let gpu = gpu_core::testgpu::dev(TRAIN_PIPELINES);
    let m = UnetTrainer::new(gpu, cfg.clone(), &tensors, H, W, T_ENC);

    let mut rng = Rng::new(seed ^ 0x5DEC_0DE5);
    let mut r = |n: usize| -> Vec<f32> { (0..n).map(|_| 2.0 * rng.next_f32() - 1.0).collect() };
    let sample = r((cfg.in_channels * H * W) as usize);
    let enc = r((T_ENC * cfg.cross_attention_dim) as usize);
    let pooled = r(cfg.pooled_dim() as usize);
    let time_ids = r(N_TIME_IDS as usize);
    // A target that is NOT the model's own output: a zero residual would make
    // every gradient zero and the check vacuously green.
    let target = r((cfg.out_channels * H * W) as usize);
    // A mid-schedule timestep. 0 would collapse half the sinusoids to a
    // constant and hide a wrong `time_embedding` gradient.
    m.set_inputs(&sample, &enc, 137.0, &pooled, &time_ids, &target);
    m
}

struct Harness {
    m: UnetTrainer,
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
/// graph: conv, GroupNorm, LayerNorm, every linear, both attentions and the
/// conditioning chain.
pub fn check_unet(seed: u64) -> Report {
    let h = Harness { m: trainer(seed), fwd: Cell::new(false) };
    directional_check(&h, 2.5e-4, 3, seed ^ 0x1234)
}

/// Per-ENTRY central differences over the two SHARED conditioning weights.
///
/// `directional_check` alone cannot gate these. It contracts a tensor onto one
/// ±1 direction and keeps the best-agreeing of `n_dirs`, and
/// `time_embedding.linear_2` / `add_embedding.linear_2` feed a single
/// `silu(emb)` consumed by all 17 resnets - a folded parameter whose gradient
/// accumulates over many contributors. Dropping some of those contributors
/// leaves a contraction that can be small, and best-of-n actively selects the
/// direction where it is smallest. That is the measured failure mode T5's
/// cross-block `axpy` fold showed (33% wrong, every directional check green),
/// which is why `elementwise_check` exists.
///
/// Scoped to the two `linear_2` weights, not the whole conditioning chain.
///
/// Those are where the fold actually happens: each produces half of `emb`, and
/// `emb` becomes the ONE `silu(emb)` all 17 resnets consume. The `linear_1`s
/// feed them and have exactly one consumer each, so a dropped contributor is
/// already visible here. That scoping is a runtime decision as much as a
/// coverage one - per-entry differencing is `2·numel` forwards, the four-tensor
/// version measured over five minutes, and `make test`'s fast lane runs this.
/// A gate that eats a large slice of the suite's budget is one people start
/// skipping, which this repo counts as no gate at all.
pub fn check_unet_conditioning_elementwise(seed: u64) -> Report {
    let h = Harness { m: trainer_with(narrow_conditioning(), seed), fwd: Cell::new(false) };
    let names = ["time_embedding.linear_2.weight", "add_embedding.linear_2.weight"];
    let mut checks = Vec::new();
    for n in names {
        checks.extend(elementwise_check(&h, n, 2.5e-4).checks);
    }
    Report { checks }
}

#[cfg(test)]
mod tests {
    /// The gate. Lives beside the entry point it gates so it cannot become an
    /// orphan (an entry point not wired into a test is not a gate).
    #[test]
    fn unet_gradients_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let r = super::check_unet(7);
        r.print();
        let (atol, rtol) = (4e-3, 8e-2);
        println!("check_unet: {} tensors, max_rel = {:.3e}", r.checks.len(), r.max_rel());
        assert!(r.checks.len() > 50, "only {} tensors checked - the tape is not covering the graph", r.checks.len());
        let bad = r.failures(atol, rtol);
        assert!(bad.is_empty(), "{} tensors outside tolerance: {:?}", bad.len(), bad);
    }

    /// The shared-parameter half of the gate - see
    /// [`super::check_unet_conditioning_elementwise`]'s doc for why a
    /// directional check cannot replace it.
    #[test]
    fn unet_conditioning_gradients_match_per_entry_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let r = super::check_unet_conditioning_elementwise(11);
        r.print();
        let bad = r.failures(4e-3, 8e-2);
        assert!(bad.is_empty(), "{} tensors outside tolerance: {:?}", bad.len(), bad);
    }
}
