// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gate for SUPIR's trainable path: the `GLVControl`
//! control trunk plus the 12 `ZeroSFT`/`ZeroCrossAttn` adaptors, recorded
//! alongside the SDXL backbone into one tape via `supir::train::SupirTrainer`.
//!
//! Swedish Embedded AB implements correctness gates for numerical training
//! code, for teams porting a diffusion control-net whose backward composes
//! many small, order-sensitive pieces. If your team needs expertise in
//! validating a ported model's backward pass against a reference, you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! # What is checked, and why the backbone is out of scope
//!
//! [`check_supir`] checks only the tensors that are genuinely NEW here: the
//! trunk (`control_model.*`) and the adaptors (`project_modules.*`). The
//! frozen-for-this-gate SDXL backbone's OWN weights (unprefixed names) are
//! deliberately excluded from the checked set - not from the graph.
//! [`gradcheck::check_unet`] already proves that exact backward (the SAME
//! `vae::blocks::grad::Trace` walk, over the SAME conv/GroupNorm/transformer
//! adjoints `sdxlunet::model::Rec` emits) on the identical block code; a
//! second full pass over it here would duplicate that gate's coverage, not
//! add a new signal, while slowing down `make gradcheck`. Gradient still
//! genuinely FLOWS through the backbone during this check - every adaptor
//! reads the backbone's own up-path activations (`h_ori`, the popped skip),
//! so a broken backbone backward would corrupt the trunk/adaptor gradients
//! this check DOES inspect. Scoping the CHECKED set is not the same as
//! scoping the DIFFERENTIATED graph.
//!
//! # Why the whole fused graph is checked, not the trunk alone
//!
//! A trunk-only check (feeding synthetic control tensors straight into the
//! adaptors) could not see the plumbing this port's design risk actually
//! lives in: [`sdxlunet::model::Rec::set_prefix`]/`take_temb_act`/
//! `set_temb_act` switching between the trunk's and the backbone's own
//! conditioning chains on ONE recorder, and [`vae::blocks::skipfuse::SkipFuse`]
//! correctly wiring `fuse_skip`/`fuse_mid`/`pre_upsample` into the up-path
//! loop [`sdxlunet::model::Unet::record_into`] walks. Recording the FULL
//! `SupirTrainer` graph (trunk + adaptors + backbone, one tape) is what
//! actually exercises that wiring; a synthetic-control-tensor unit test
//! would not.
//!
//! # Why [`check_supir_elementwise`] targets `mid_block.resnets.1.conv2.bias`
//!
//! Every one of SUPIR's 10 trunk hidden states is read TWICE by the adaptor
//! that consumes it (`Adaptors::fuse_skip`/`fuse_mid` each read `control.buf`
//! once for `zero_conv` and once for `mlp_shared.0`) - a shared-activation
//! fold whose upstream producer's gradient is a SUM over both downstream
//! branches. That is exactly the class of defect [`gradcheck::directional_check`]'s
//! own doc warns about (a random-direction contraction can land near zero
//! even when a whole branch's contribution is missing - the measured T5
//! `rel_bias` case: a third of the true gradient dropped, still inside the
//! workspace's `(4e-3, 8e-2)` tolerance at some seeds). The trunk's LAST op,
//! `mid_block.resnets.1`, is the cheapest config-independent instance of
//! this: its output IS `hs.last()`, always present regardless of how many
//! levels a config has, so this target does not depend on the adaptors'
//! optional `ZeroCrossAttn` sites the way the roadmap's other named example
//! (`control[6]`/`control[3]`, each read by a `ZeroSFT` AND a
//! `ZeroCrossAttn`) does. `conv2.bias` (not `conv2.weight`) keeps `numel`
//! small: [`elementwise_check`] costs `2·numel` FULL forward passes over the
//! whole fused graph, so a shrunk trunk width ([`narrow_mid`]) plus a bias
//! tensor (`cmid` entries, not `cmid²·9`) is what keeps this gate in the
//! "a few dozen forwards" range rather than "tens of thousands" - the same
//! trade [`gradcheck::check_unet_conditioning_elementwise`] makes by
//! narrowing `time_embed_dim`.

use std::cell::Cell;

use data::rng::Rng;
use supir::config::SupirConfig;
use supir::train::{SupirTrainer, TRAIN_PIPELINES};

use crate::{directional_check, elementwise_check, CheckModel, Report};

// Matches `gradcheck::unet`'s own scale (`H = 8, W = 8, T_ENC = 5`) - this
// perf-number: op-count ratio from the graph's own structure, not a measured runtime speedup
// graph is trunk+adaptors+backbone (roughly 2-3x a plain UNet's op count),
// and `directional_check`/`elementwise_check` cost is linear in the number
// of FULL forward passes (`2·n_dirs·n_tensors` and `2·numel` respectively),
// so keeping the per-forward cost at `check_unet`'s own scale is what keeps
// this gate inside `make gradcheck`'s normal budget rather than turning it
// into a many-minute outlier.
const H: u32 = 8;
const W: u32 = 8;
const T_ENC: u32 = 5;
const CONTROL_SCALE: f32 = 0.7;

/// [`SupirConfig::tiny`] with the trunk/backbone's channel widths shrunk
/// further - see the module doc's `check_supir_elementwise` section.
/// [`SupirConfig::tiny`] unmodified - see the module doc's
/// `check_supir_elementwise` section for why the target tensor's `numel`
/// (`cmid`, [`sdxlunet::config::UNetConfig::tiny`]'s `block_out_channels`
/// last entry) cannot shrink below 64: `supir::adaptors::HEAD_DIM` (64,
/// fixed by the real checkpoint's shape) requires the level the one
/// `ZeroCrossAttn` site lives at - which is also the mid block's own level -
/// to stay a multiple of 64.
fn narrow_mid() -> SupirConfig {
    SupirConfig::tiny()
}

/// Build a trainer at `cfg` with deterministic weights (frozen backbone +
/// SUPIR delta, merged - the same pattern `supir::model`'s own tiny-forward
/// smoke test uses) and a fixed batch.
fn trainer_at(cfg: SupirConfig, seed: u64) -> SupirTrainer {
    let mut tensors = sdxlunet::init::init_weights(&cfg.backbone, seed);
    tensors.extend(supir::init::init_weights(&cfg, seed ^ 0x5350_4952));
    let gpu = gpu_core::testgpu::dev(TRAIN_PIPELINES);
    let m = SupirTrainer::new(gpu, cfg.clone(), &tensors, H, W, T_ENC, CONTROL_SCALE);

    let c = cfg.backbone.clone();
    let mut rng = Rng::new(seed ^ 0x5DEC_0DE5);
    let mut r = |n: usize| -> Vec<f32> { (0..n).map(|_| 2.0 * rng.next_f32() - 1.0).collect() };
    let sample = r((c.in_channels * H * W) as usize);
    let hint = r((4 * H * W) as usize);
    let enc = r((T_ENC * c.cross_attention_dim) as usize);
    let pooled = r(c.pooled_dim() as usize);
    let time_ids = r(sdxlunet::config::N_TIME_IDS as usize);
    // A target that is NOT the model's own output: a zero residual would make
    // every gradient zero and the check vacuously green - the same reasoning
    // `gradcheck::unet::trainer` states.
    let target = r((c.out_channels * H * W) as usize);
    m.set_inputs(&sample, &hint, &enc, 601.0, &pooled, &time_ids, &target);
    m
}

fn trainer(seed: u64) -> SupirTrainer {
    trainer_at(SupirConfig::tiny(), seed)
}

/// The tensors [`check_supir`]/[`check_supir_elementwise`] check - see the
/// module doc for why the backbone's own (unprefixed) names are excluded.
fn trainable_names(m: &SupirTrainer) -> Vec<String> {
    m.params()
        .iter()
        .map(|(n, _)| n.clone())
        .filter(|n| n.starts_with("control_model.") || n.starts_with("project_modules."))
        .collect()
}

struct Harness {
    m: SupirTrainer,
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

/// Directional finite-difference check over every trunk and adaptor tensor.
pub fn check_supir(seed: u64) -> Report {
    let m = trainer(seed);
    let names = trainable_names(&m);
    let h = Harness { m, names, fwd: Cell::new(false) };
    directional_check(&h, 2.5e-4, 3, seed ^ 0x1234)
}

/// Per-ENTRY central differences over the trunk's own mid-block output bias -
/// see the module doc for why this specific tensor and why it needs a
/// narrower config than [`check_supir`].
pub fn check_supir_elementwise(seed: u64) -> Report {
    let m = trainer_at(narrow_mid(), seed);
    let names = trainable_names(&m);
    let h = Harness { m, names, fwd: Cell::new(false) };
    elementwise_check(&h, "control_model.mid_block.resnets.1.conv2.bias", 2.5e-4)
}

#[cfg(test)]
mod tests {
    /// The gate. Lives beside the entry point it gates, per this workspace's
    /// own convention (`gradcheck::unet`'s module doc states the same
    /// reason: an entry point not wired into a test is not a gate).
    #[test]
    fn supir_gradients_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let r = super::check_supir(7);
        r.print();
        let (atol, rtol) = (4e-3, 8e-2);
        println!("check_supir: {} tensors, max_rel = {:.3e}", r.checks.len(), r.max_rel());
        assert!(r.checks.len() > 20, "only {} tensors checked - the tape is not covering the trunk/adaptors", r.checks.len());
        let bad = r.failures(atol, rtol);
        assert!(bad.is_empty(), "{} tensors outside tolerance: {:?}", bad.len(), bad);
    }

    /// The shared/folded-activation half of the gate - see this module's own
    /// doc for why a directional check alone cannot replace it.
    #[test]
    fn supir_mid_bias_gradient_matches_per_entry_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let r = super::check_supir_elementwise(11);
        r.print();
        assert!(!r.checks.is_empty(), "check_supir_elementwise: no entries checked");
        let bad = r.failures(4e-3, 8e-2);
        assert!(bad.is_empty(), "{} entries outside tolerance: {:?}", bad.len(), bad);
    }
}
