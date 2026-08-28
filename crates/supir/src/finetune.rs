// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Two SUPIR fine-tuning modes over [`crate::train::SupirTrainer`]'s already
//! gradient-checked graph:
//!
//! * **Adaptor-only** ([`Finetuner::adaptor_only`]) - upstream's own recipe:
//!   the frozen SDXL backbone's ENCODER pass (`conv_in`, `time_embedding`,
//!   `add_embedding`, `down_blocks.*`, `mid_block.*`, all unprefixed) runs
//!   under an effective `torch.no_grad()` - its Adam state is never touched,
//!   so it never moves - while the backbone's DECODER half (`up_blocks.*`,
//!   `conv_norm_out.*`, `conv_out.*`), the control trunk (`control_model.*`)
//!   and the 12 adaptors (`project_modules.*`) all train. This is a real
//!   simplification of upstream's actual `torch.no_grad()`: this trainer
//!   still COMPUTES a gradient for every frozen tensor (the shared reverse
//!   walk has no per-tensor stop-gradient), it just never applies one - the
//!   freeze here is "no Adam state, no write-back", not "no backward work".
//!   Correct for a training-correctness gate; a production trainer wanting
//!   the compute saving upstream gets from `no_grad` would need a stop-
//!   gradient at the down-path/mid-block boundary, out of scope here.
//! * **Full-backbone** ([`Finetuner::full_backbone`]) - every recorded
//!   parameter trains, trunk and backbone alike.
//!
//! Both modes share ONE optimizer step ([`Finetuner::step`]), reusing
//! [`model::lora::adam`] - the same Adam implementation every trainer in
//! this workspace steps with, not a second copy of the formula - over
//! whichever parameter names the installed freeze predicate lets through.
//!
//! ## Scale
//! Every test in this file runs at [`crate::config::SupirConfig::tiny`]. The
//! combined trunk+adaptors+backbone graph's fp32 resident set is already
//! documented (`crate::int8`'s module doc) to exceed a single card's
//! per-buffer binding cap at real-checkpoint scale. "Full-backbone" here
//! means "every parameter in this SMALL graph trains", not a claim that the
//! real 15 GB SUPIR checkpoint fits a full-backbone fine-tune on one card -
//! that remains a residency/int8 problem for a later pass, exactly as the
//! roadmap's own Phase 4 framing states.
//!
//! ## "Batch"
//! `crates/supir` records its graph at batch 1 (the roadmap's own Deferred
//! section: "Batch > 1... the SDXL graph is recorded at batch 1"), so a
//! "batch" overfit gate here means cycling a small FIXED set of independent
//! single-sample examples through the same trainer, one gradient step per
//! example per round - proving the optimizer drives down loss averaged over
//! more than one example, not a single-tensor batch dimension.

use std::collections::HashMap;

use crate::train::SupirTrainer;

/// Which parameters [`Finetuner::adaptor_only`] freezes - see the module doc.
pub fn is_frozen_for_adaptor_only(name: &str) -> bool {
    if name.starts_with("control_model.") || name.starts_with("project_modules.") {
        return false;
    }
    if name.starts_with("up_blocks.") || name.starts_with("conv_norm_out.") || name.starts_with("conv_out.") {
        return false;
    }
    true
}

/// A [`SupirTrainer`] plus host-side Adam moments and a freeze predicate.
pub struct Finetuner {
    trainer: SupirTrainer,
    moments: HashMap<String, (Vec<f32>, Vec<f32>)>,
    t: u64,
    frozen: fn(&str) -> bool,
}

impl Finetuner {
    /// Upstream's own recipe - see [`is_frozen_for_adaptor_only`].
    pub fn adaptor_only(trainer: SupirTrainer) -> Finetuner {
        Finetuner { trainer, moments: HashMap::new(), t: 0, frozen: is_frozen_for_adaptor_only }
    }

    /// Every recorded parameter trains.
    pub fn full_backbone(trainer: SupirTrainer) -> Finetuner {
        Finetuner { trainer, moments: HashMap::new(), t: 0, frozen: |_| false }
    }

    pub fn trainer(&self) -> &SupirTrainer {
        &self.trainer
    }

    /// One step: forward (returns the PRE-update loss), backward, then an
    /// Adam update over every unfrozen parameter.
    pub fn step(&mut self, lr: f32) -> f32 {
        self.trainer.zero_grads();
        let loss = self.trainer.forward();
        self.trainer.backward();
        self.t += 1;
        let t = self.t;
        let names: Vec<String> = self.trainer.params().iter().map(|(n, _)| n.clone()).collect();
        for name in names {
            if (self.frozen)(&name) {
                continue;
            }
            let g = self.trainer.read_grad(&name);
            let mut w = self.trainer.read_weight(&name);
            let (m, v) =
                self.moments.entry(name.clone()).or_insert_with(|| (vec![0.0f32; g.len()], vec![0.0f32; g.len()]));
            model::lora::adam(&mut w, m, v, &g, lr, t);
            self.trainer.write_weight(&name, &w);
        }
        loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Rng;

    /// One fixed, deterministic example: `(sample, hint, enc, pooled,
    /// time_ids, target)`, all finite and none identically zero.
    struct Example {
        sample: Vec<f32>,
        hint: Vec<f32>,
        enc: Vec<f32>,
        pooled: Vec<f32>,
        time_ids: Vec<f32>,
        target: Vec<f32>,
    }

    // Matches `gradcheck::supir`'s own scale: each training step is a FULL
    // forward+backward of the fused trunk+adaptors+backbone graph, which
    // dominates this file's wall-clock at anything larger. `H=W=8` is still
    // SDXL-shaped - a multiple of `UNetConfig::tiny`'s 2x downscale, an
    // architecture constant, not a tuned number - and keeps the whole file's
    // test suite inside a normal `cargo test` budget.
    const H: u32 = 8;
    const W: u32 = 8;
    const T_ENC: u32 = 5;
    const CONTROL_SCALE: f32 = 0.7;

    /// The learning rate every overfit gate in this file steps at, and the
    /// only genuinely calibrated number here - see [`overfit_single`] for
    /// what it is calibrated FOR (stability, not speed).
    const LR: f32 = 0.005;

    /// Steps excluded from [`overfit_single`]'s stability check. Adam's
    /// first updates are taken against bias-corrected moments estimated from
    /// one or two gradients, so a transient above `l0` in the first handful
    /// of steps is the optimizer warming up, not the instability this gate
    /// is looking for.
    const WARMUP: usize = 20;

    fn example(cfg: &crate::config::SupirConfig, seed: u64) -> Example {
        let c = &cfg.backbone;
        let mut rng = Rng::new(seed);
        let mut r = |n: usize| -> Vec<f32> { (0..n).map(|_| 2.0 * rng.next_f32() - 1.0).collect() };
        Example {
            sample: r((c.in_channels * H * W) as usize),
            hint: r((4 * H * W) as usize),
            enc: r((T_ENC * c.cross_attention_dim) as usize),
            pooled: r(c.pooled_dim() as usize),
            time_ids: vec![64.0, 64.0, 0.0, 0.0, 64.0, 64.0],
            target: r((c.out_channels * H * W) as usize),
        }
    }

    fn set(trainer: &SupirTrainer, ex: &Example) {
        trainer.set_inputs(&ex.sample, &ex.hint, &ex.enc, 601.0, &ex.pooled, &ex.time_ids, &ex.target);
    }

    fn new_trainer(seed: u64) -> (crate::config::SupirConfig, SupirTrainer) {
        let (cfg, tensors) = crate::train::tiny_setup(seed);
        let gpu = gpu_core::testgpu::dev(crate::train::TRAIN_PIPELINES);
        let trainer = SupirTrainer::new(gpu, cfg.clone(), &tensors, H, W, T_ENC, CONTROL_SCALE);
        (cfg, trainer)
    }

    /// The freeze predicate must do what it says: a frozen (backbone
    /// encoder) weight is BIT-IDENTICAL before and after training, while a
    /// trained (trunk) weight moves. Cheap (5 steps) - a correctness check
    /// on the mechanism, not an overfit gate.
    #[test]
    fn adaptor_only_freezes_the_encoder_and_trains_the_rest() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let (cfg, trainer) = new_trainer(41);
        set(&trainer, &example(&cfg, 42));
        let frozen_name = "down_blocks.0.resnets.0.conv1.weight";
        let trunk_name = "control_model.conv_in.weight";
        assert!(is_frozen_for_adaptor_only(frozen_name));
        assert!(!is_frozen_for_adaptor_only(trunk_name));
        let before_frozen = trainer.read_weight(frozen_name);
        let before_trunk = trainer.read_weight(trunk_name);

        let mut f = Finetuner::adaptor_only(trainer);
        for _ in 0..5 {
            f.step(0.02);
        }

        let after_frozen = f.trainer().read_weight(frozen_name);
        let after_trunk = f.trainer().read_weight(trunk_name);
        assert_eq!(before_frozen, after_frozen, "a frozen (backbone encoder) weight moved under adaptor-only training");
        assert_ne!(before_trunk, after_trunk, "the trunk weight did not move under adaptor-only training");
    }

    /// Overfit-one-sample, both modes through one loop.
    ///
    /// ## What [`LR`] is calibrated for: stability, not speed
    /// [`model::lora::adam`]'s update is scale-FREE - every unfrozen entry
    /// moves by about `lr` per step whatever its own gradient's magnitude -
    /// so `lr` here is not "how fast" but "how big a step relative to a
    /// weight whose init scale is `1/sqrt(fan_in)`", which at
    /// [`sdxlunet::config::UNetConfig::tiny`]'s widths is a few hundredths.
    /// Past a threshold the trainer enters a limit cycle: the loss still
    /// trends down between excursions, but the excursions reach ABOVE `l0`,
    /// so a gate reading ONE step's loss reads the cycle's phase rather than
    /// the training. Below it, both modes drive this single example to a
    /// literal near-zero floor, monotonically after a short warm-up - which
    /// is why the assertions below are four orders of magnitude, not a
    /// "clear descent" fraction.
    ///
    /// That threshold is a property of WHICH parameters are unfrozen, not of
    /// how many: unfreezing the backbone encoder moves the input every later
    /// stage is conditioned on, so full-backbone leaves the stable regime
    /// first even though it has strictly more capacity.
    ///
    /// Re-run either caller with `--nocapture` for the current per-step
    /// trajectory on your own hardware.
    ///
    /// Returns `(l0, last, tail_max)`. `tail_max` - the worst loss seen after
    /// the [`WARMUP`] steps - is what makes the gate see the regime and not
    /// just the endpoint: an every-20th-step print of a diverging run still
    /// looks monotone, and a single endpoint below a threshold can be one
    /// descending step of a cycle whose peaks are above `l0`.
    fn overfit_single(mode_full: bool, seed: u64, steps: usize, lr: f32) -> (f32, f32, f32) {
        let (cfg, trainer) = new_trainer(seed);
        set(&trainer, &example(&cfg, seed ^ 0xF00D));
        let mut f = if mode_full { Finetuner::full_backbone(trainer) } else { Finetuner::adaptor_only(trainer) };
        let mut last = f.step(lr);
        let l0 = last;
        let mut tail_max = 0.0f32;
        for step in 1..steps {
            last = f.step(lr);
            if step >= WARMUP {
                tail_max = tail_max.max(last);
            }
            if step % 20 == 0 {
                eprintln!("  step {step:3}: loss = {last:.4e}");
            }
        }
        (l0, last, tail_max)
    }

    #[test]
    fn adaptor_only_overfits_a_single_sample() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let (l0, last, tail_max) = overfit_single(false, 101, 120, LR);
        eprintln!("adaptor-only overfit: {l0:.4e} -> {last:.4e} (worst after warm-up {tail_max:.4e})");
        assert!(tail_max < l0, "adaptor-only training left the stable regime: {l0:.4e} -> peak {tail_max:.4e}");
        assert!(last < l0 * 1e-4, "adaptor-only training did not overfit: {l0:.4e} -> {last:.4e}");
    }

    #[test]
    fn full_backbone_overfits_a_single_sample() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let (l0, last, tail_max) = overfit_single(true, 202, 120, LR);
        eprintln!("full-backbone overfit: {l0:.4e} -> {last:.4e} (worst after warm-up {tail_max:.4e})");
        assert!(tail_max < l0, "full-backbone training left the stable regime: {l0:.4e} -> peak {tail_max:.4e}");
        assert!(last < l0 * 1e-4, "full-backbone training did not overfit: {l0:.4e} -> {last:.4e}");
    }

    /// The "batch" gate - see the module doc for why this is a small FIXED
    /// set of independent examples cycled through one trainer, not a batch
    /// dimension in the graph. Adaptor-only only: the optimizer mechanism is
    /// identical for full-backbone (only the freeze predicate differs, and
    /// that is already covered by the single-sample gate above), so a
    /// second heavy multi-example loop would add wall-clock time without a
    /// new correctness signal. Steps at the same [`LR`] and carries the same
    /// stability check as [`overfit_single`], for the same reason - a
    /// per-round average smooths the excursions a diverging run makes but
    /// does not remove them.
    #[test]
    fn adaptor_only_overfits_a_small_dataset() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let (cfg, trainer) = new_trainer(303);
        let examples: Vec<Example> = (0..3).map(|i| example(&cfg, 400 + i)).collect();
        let mut f = Finetuner::adaptor_only(trainer);

        let avg_loss = |f: &mut Finetuner, lr: f32| -> f32 {
            let mut total = 0.0;
            for ex in &examples {
                set(f.trainer(), ex);
                total += f.step(lr);
            }
            total / examples.len() as f32
        };

        let l0 = avg_loss(&mut f, LR);
        let mut last = l0;
        let mut tail_max = 0.0f32;
        for round in 1..40 {
            last = avg_loss(&mut f, LR);
            // `WARMUP` counts STEPS, and a round is one step per example.
            if round * examples.len() >= WARMUP {
                tail_max = tail_max.max(last);
            }
            if round % 10 == 0 {
                eprintln!("  round {round:3}: avg loss = {last:.4e}");
            }
        }
        eprintln!(
            "adaptor-only small-dataset overfit ({} examples): {l0:.4e} -> {last:.4e} (worst after warm-up {tail_max:.4e})",
            examples.len()
        );
        assert!(tail_max < l0, "small-dataset training left the stable regime: {l0:.4e} -> peak {tail_max:.4e}");
        assert!(last < l0 * 1e-3, "small-dataset training did not overfit: {l0:.4e} -> {last:.4e}");
    }
}
