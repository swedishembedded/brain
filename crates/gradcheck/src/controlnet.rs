// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gate for the SDXL ControlNet backward:
//! `controlnet::train::ControlNetTrainer`'s trainable copy + zero-convs +
//! residual injection into the frozen backbone, recorded onto one
//! `vae::blocks::grad::Trace`.
//!
//! Swedish Embedded AB implements correctness gates for numerical training
//! code, for teams porting a control-adapter whose backward composes a
//! trainable copy of a frozen backbone with a residual injection between the
//! two. If your team needs expertise in validating a ported model's backward
//! pass against a reference, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! # What is checked, and why the backbone is out of scope
//!
//! [`check_controlnet`] checks only the tensors that are genuinely NEW here:
//! ControlNet's own trainable copy (`"controlnet."`-prefixed - see
//! `controlnet::train`'s module doc for why that prefix exists) and the
//! zero-convs. The frozen backbone's OWN weights (unprefixed names) are
//! deliberately excluded from the CHECKED set, not from the graph - exactly
//! [`gradcheck::check_supir`]'s own scoping, for the same reason:
//! [`gradcheck::check_unet`] already proves that exact backward (the SAME
//! `vae::blocks::grad::Trace` walk, over the SAME conv/GroupNorm/transformer
//! adjoints `sdxlunet::model::Rec` emits) on the identical block code.
//! Gradient still genuinely FLOWS through the backbone during this check -
//! every zero-conv output is added into a backbone skip or the mid hidden
//! state, so a broken backbone backward would corrupt the residuals this
//! check DOES inspect (a wrong `conv2d_dx` in the up path, say, would show up
//! as a wrong `dL/d(zero-conv output)`, which propagates straight back into
//! `controlnet_down_blocks.k`'s gradient).
//!
//! # Why the closed loop, not the named fallback
//!
//! The isolated-trainer fallback (loss directly on the zero-conv outputs, no
//! backbone injection) would prove the trainable copy + zero-conv + `scale_chan`
//! backward internally consistent, but never that `Op::Add2`'s injection at
//! each site is wired at the RIGHT site with the RIGHT sign into the RIGHT
//! backbone tensor - exactly the class of defect `crate::supir`'s own doc
//! warns a trunk-only check cannot see. `controlnet::train::ControlNetTrainer`
//! records the injection directly onto the backbone's own tape, so this file
//! takes the full closed loop.
//!
//! # Why [`check_controlnet_elementwise`] targets the two `linear_2` weights
//!
//! Same fold [`gradcheck::check_unet_conditioning_elementwise`] documents,
//! restated for ControlNet's OWN copy: `controlnet.time_embedding.linear_2`
//! and `controlnet.add_embedding.linear_2` each produce half of ControlNet's
//! own `emb`, which becomes the ONE `silu(emb)` every resnet in the
//! trainable copy consumes - a folded parameter whose gradient accumulates
//! over every one of those resnets, exactly the shared-activation class a
//! directional check's random-direction contraction can miss (the T5
//! `rel_bias` precedent both sibling modules cite: a third of the gradient
//! dropped, every directional check still green). `conditioning_scale`
//! is NOT a candidate here - see `vae::blocks::Op::ScaleChan`'s doc: it is a
//! per-request `ParamSpec`, absent from `ControlNetConfig::tensor_manifest`,
//! so a `read_weight`/`read_grad` lookup on it would simply fail; nothing in
//! this tree trains it, and `Builder::scale_chan`'s adjoint deliberately has
//! no `dscale` for exactly that reason.
//!
//! # Tolerances and cost
//!
//! `eps = 2.5e-4`, [`gradcheck::check_unet`]'s own starting value - the two
//! graphs share the same op mix (conv/GroupNorm/transformer) and the same
//! `UNetConfig::tiny`-derived scale, just doubled (trainable copy + frozen
//! backbone, the same "~2-3x a plain UNet's op count" [`gradcheck::check_supir`]'s
//! own doc measures for its own trunk+adaptors+backbone graph). It needed no
//! adjustment: measured `max_rel` over all 150 checked tensors at this eps is
//! `1.395e-1`, comparable to `check_unet`'s own `9.5e-2` over 263. `time_embed_dim`
//! narrows to 8 (not [`gradcheck::unet`]'s 16) for the elementwise check only,
//! keeping `2·numel` (128 entries, both tensors) full forwards over the
//! ~2x-costlier merged graph inside a comparable wall-clock budget to
//! `check_unet_conditioning_elementwise`'s own (measured `max_rel = 8.205e-1`
//! here - every outlier entry checked is one where both the analytic and the
//! numeric derivative are within fp32 noise of zero, e.g.
//! `add_embedding.linear_2.weight[51]`: analytic `-1.32e-4`, numeric `0`, abs
//! error `1.32e-4` against the `4e-3` floor - the same "relative error alone
//! is ill-conditioned near zero" reasoning [`crate::Check::within`]'s own doc
//! states, not a sign either check is missing a contribution).

use std::cell::Cell;

use controlnet::config::ControlNetConfig;
use controlnet::train::{ControlNetTrainer, TRAIN_PIPELINES};
use data::rng::Rng;
use sdxlunet::config::N_TIME_IDS;

use crate::{directional_check, elementwise_check, CheckModel, Report};

// Matches `gradcheck::unet`/`gradcheck::supir`'s own scale.
const H: u32 = 8;
const W: u32 = 8;
const T_ENC: u32 = 5;
/// `conditioning_scale`. Must be NON-ZERO: `Op::ScaleChan`'s adjoint is
/// `dx = dy * scale[c]`, so a zero scale would zero every gradient reaching
/// the trainable copy through the zero-convs, making the whole check
/// vacuously pass regardless of whether the injection is wired correctly.
const COND_SCALE: f32 = 0.7;

/// [`ControlNetConfig::tiny`] with a narrower conditioning chain, for the
/// per-ENTRY check only - see the module doc's cost note.
fn narrow_conditioning() -> ControlNetConfig {
    let mut cfg = ControlNetConfig::tiny();
    cfg.backbone.time_embed_dim = 8;
    cfg
}

/// Build a trainer at `cfg` with deterministic weights (frozen backbone +
/// ControlNet delta, merged via `controlnet::train::tensors_for` - see that
/// function's doc for the name-collision handling) and a fixed batch.
fn trainer_at(cfg: ControlNetConfig, seed: u64) -> ControlNetTrainer {
    let tensors = controlnet::train::tensors_for(&cfg, seed);
    let gpu = gpu_core::testgpu::dev(TRAIN_PIPELINES);
    let m = ControlNetTrainer::new(gpu, cfg.clone(), &tensors, H, W, T_ENC);

    let bb = &cfg.backbone;
    let ds = cfg.cond_downscale();
    let mut rng = Rng::new(seed ^ 0x5DEC_0DE5);
    let mut r = |n: usize| -> Vec<f32> { (0..n).map(|_| 2.0 * rng.next_f32() - 1.0).collect() };
    let sample = r((bb.in_channels * H * W) as usize);
    let cond = r((cfg.conditioning_channels * (H * ds) * (W * ds)) as usize);
    let enc = r((T_ENC * bb.cross_attention_dim) as usize);
    let pooled = r(bb.pooled_dim() as usize);
    let time_ids = r(N_TIME_IDS as usize);
    // A target that is NOT the model's own output: a zero residual would make
    // every gradient zero and the check vacuously green - the same reasoning
    // `gradcheck::unet`/`gradcheck::supir` state.
    let target = r((bb.out_channels * H * W) as usize);
    m.set_inputs(&sample, 601.0, &enc, &pooled, &time_ids, &cond, COND_SCALE, &target);
    m
}

fn trainer(seed: u64) -> ControlNetTrainer {
    trainer_at(ControlNetConfig::tiny(), seed)
}

/// The tensors [`check_controlnet`]/[`check_controlnet_elementwise`] check -
/// see the module doc for why the frozen backbone's own (unprefixed) names
/// are excluded.
fn trainable_names(m: &ControlNetTrainer) -> Vec<String> {
    m.params().iter().map(|(n, _)| n.clone()).filter(|n| n.starts_with("controlnet.")).collect()
}

struct Harness {
    m: ControlNetTrainer,
    names: Vec<String>,
    fwd: Cell<bool>,
}

impl CheckModel for Harness {
    fn param_names(&self) -> Vec<String> {
        self.names.clone()
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

/// Directional finite-difference check over every ControlNet-owned trainable
/// tensor: the conditioning chain, the conditioning-image embedder, the
/// trainable down/mid copy and the zero-convs.
pub fn check_controlnet(seed: u64) -> Report {
    let m = trainer(seed);
    let names = trainable_names(&m);
    let h = Harness { m, names, fwd: Cell::new(false) };
    directional_check(&h, 2.5e-4, 3, seed ^ 0x1234)
}

/// Per-ENTRY central differences over ControlNet's own two SHARED
/// conditioning weights - see the module doc for why a directional check
/// cannot replace this.
pub fn check_controlnet_elementwise(seed: u64) -> Report {
    let m = trainer_at(narrow_conditioning(), seed);
    let names = trainable_names(&m);
    let h = Harness { m, names, fwd: Cell::new(false) };
    let targets = ["controlnet.time_embedding.linear_2.weight", "controlnet.add_embedding.linear_2.weight"];
    let mut checks = Vec::new();
    for n in targets {
        checks.extend(elementwise_check(&h, n, 2.5e-4).checks);
    }
    Report { checks }
}

#[cfg(test)]
mod tests {
    /// The gate. Lives beside the entry point it gates, per this workspace's
    /// own convention.
    #[test]
    fn controlnet_gradients_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let r = super::check_controlnet(7);
        r.print();
        let (atol, rtol) = (4e-3, 8e-2);
        println!("check_controlnet: {} tensors, max_rel = {:.3e}", r.checks.len(), r.max_rel());
        // `ControlNetTrainer`'s trace carries exactly 150 ControlNet-owned
        // (`"controlnet."`-prefixed) trainable tensors at
        // `ControlNetConfig::tiny()`, measured directly (lower than
        // `ControlNetConfig::tensor_manifest().len()` because GroupNorm's
        // `weight`/`bias` pair and an attention's q/k/v fuse into ONE trace
        // entry each - see `vae::blocks::grad::Trace::params`'s own doc).
        // `K = 100` leaves the same order of headroom below the actual count
        // `check_unet`'s own floor does (50 against a measured 263) while
        // still failing hard on the regression this floor exists to catch:
        // given the starting state is a fully discarded tape (`model::
        // ControlNet::new`'s own `finish()` used to throw it away before
        // this item), a near-empty `checks` list passing "by accident" is
        // the single most likely way that regresses silently.
        const K: usize = 100;
        assert!(r.checks.len() > K, "only {} tensors checked - the tape is not covering ControlNet's own graph", r.checks.len());
        let bad = r.failures(atol, rtol);
        assert!(bad.is_empty(), "{} tensors outside tolerance: {:?}", bad.len(), bad);
    }

    /// The shared/folded-activation half of the gate - see this module's own
    /// doc for why a directional check alone cannot replace it.
    #[test]
    fn controlnet_conditioning_gradients_match_per_entry_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let r = super::check_controlnet_elementwise(11);
        r.print();
        println!("check_controlnet_elementwise: {} entries, max_rel = {:.3e}", r.checks.len(), r.max_rel());
        assert!(!r.checks.is_empty(), "check_controlnet_elementwise: no entries checked");
        let bad = r.failures(4e-3, 8e-2);
        assert!(bad.is_empty(), "{} entries outside tolerance: {:?}", bad.len(), bad);
    }
}
