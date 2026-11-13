// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training recipes for image tokenizers for its clients. If your team needs expertise in
// reproducing a published VQGAN objective (perceptual, adversarial and codebook terms) on your data, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The VQGAN **training objective**: which terms the generator is trained on and with what weights, and the options of
//! one optimiser step.
//!
//! The objective [`crate::train::VqganTrainer`] minimises is
//!
//! ```text
//! L_G = L_rec(x_hat, x) + perceptual_weight * LPIPS(x_hat, x)
//!       + L_cb + L_com (+ entropy_loss_ratio * L_ent)
//!       + disc_weight * lambda * ( -mean D(x_hat) )                  once step >= disc_start
//! lambda = clamp( ||grad_W L_nll|| / (||grad_W L_adv|| + 1e-4), 0, adaptive_weight_max )     detached,
//!          or 1 when the recipe does not adapt it
//! ```
//!
//! with `W` the generator's last convolution weight, `L_nll = L_rec + perceptual_weight * LPIPS`, `L_adv = -mean
//! D(x_hat)`, and the discriminator trained at the same steps on the hinge loss `0.5 (mean relu(1 - D(x)) + mean
//! relu(1 + D(x_hat)))` (taming-transformers' `VQLPIPSWithDiscriminator`, which both recipes below follow), scaled by
//! `disc_weight` where the recipe does. `L_cb`, `L_com` are the VQ terms the trainer already weighted with
//! [`crate::VqganConfig::beta`], and `L_ent` is [`crate::entropy`]'s (run on the device).
//!
//! Every field below is one a trainer reads; none is decoration. A weight of zero turns its term off entirely (no
//! LPIPS network, no discriminator is built for a zero weight).
//!
//! # Presets, and where their values come from
//!
//! Both recipes use LPIPS v0.1 on the **VGG16** trunk ([`lpips::Trunk::Vgg16`]), not AlexNet.
//!
//! * [`Objective::llamagen_vq16`] - LlamaGen `tokenizer/tokenizer_image/vq_loss.py` (`VQLoss`) and `vq_train.py`:
//!   L2 reconstruction (`reconstruction_loss='l2'`), `perceptual_weight 1.0`, `disc_weight 0.5` (the CLI default;
//!   `VQLoss`'s own default is 1.0), `disc_start 20000`, hinge GAN losses, a PatchGAN of `disc_dim 64` filters and
//!   `disc_num_layers 3`, `entropy_loss_ratio 0.0`. `disc_adaptive_weight` is False (`vq_train.py` does not pass it),
//!   so the generator's adversarial term is `disc_weight * gen_adv_loss` with `lambda = 1`; the discriminator's loss is
//!   `disc_weight * hinge_d_loss` (`disc_weight` adopted by step like the generator's, 0 before `disc_start`).
//!   Optimiser: Adam, `lr 1e-4`, betas `(0.9, 0.95)`.
//! * [`Objective::codeformer`] - CodeFormer `options/VQGAN_512_ds32_nearest_stage1.yml`: `pixel_opt` L1 weight 1.0,
//!   `LPIPSLoss` weight 1.0, `gan_opt` hinge with `loss_weight 1.0` and `adaptive_weighting` on, `network_d`
//!   `VQGANDiscriminator` with `ndf 64`, `net_d_start_iter 30001`, Adam `lr 7e-5`, `weight_decay 0`, betas
//!   `(0.9, 0.99)` for generator and discriminator. The yml does not state the discriminator's depth: `disc_layers 4`
//!   and the adaptive clamp `1e4` (taming's) are brain's choices there. Its discriminator loss is unscaled.
//! * [`Objective::reconstruction_only`] - this trainer before the full objective: L1 and the VQ terms, the objective
//!   `gradcheck::check_vqgan` differentiates. Every value that is not zero (the Adam settings, taming's adaptive
//!   weight and its `1e4` clamp, a 64-filter 3-layer discriminator) is brain's own default, not a recipe's.

use lpips::Trunk;
use optim::Adam;

/// The pixel reconstruction loss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReconstructionLoss {
    /// `mean |x_hat - x|`: `basicsr`/CodeFormer's `pixel_opt`, taming's `pixelloss_weight` term.
    L1,
    /// `mean (x_hat - x)^2`: LlamaGen's `--reconstruction-loss l2` default.
    L2,
}

/// Which terms the generator trains on, and with what weights. See the module docs for the objective and the presets.
#[derive(Clone, Debug, PartialEq)]
pub struct Objective {
    pub reconstruction: ReconstructionLoss,
    /// Weight of `LPIPS(x_hat, x)` (`crates/lpips`). Nonzero needs the LPIPS weights of [`Self::perceptual_net`]; zero
    /// builds no network.
    pub perceptual_weight: f32,
    /// The LPIPS v0.1 trunk the perceptual term runs on.
    pub perceptual_net: Trunk,
    /// Weight of the adversarial term, the discriminator's `disc_weight` of taming. Zero builds no discriminator.
    pub disc_weight: f32,
    /// The step from which the generator's adversarial term and the discriminator's updates begin.
    pub disc_start: u32,
    /// The discriminator's width (`ndf`) and depth (`n_layers`), as [`crate::patchgan::PatchGanConfig`].
    pub disc_ndf: u32,
    pub disc_layers: u32,
    /// Whether the generator's adversarial term is scaled by taming's adaptive `lambda` (CodeFormer) or by 1
    /// (LlamaGen's `disc_adaptive_weight=False`).
    pub adaptive_weight: bool,
    /// The clamp ceiling of the adaptive weight `lambda`; read only with [`Self::adaptive_weight`].
    pub adaptive_weight_max: f32,
    /// Whether the discriminator's own loss is `disc_weight * hinge` (LlamaGen) rather than the bare hinge.
    pub scale_disc_loss: bool,
    /// Learning rate of both optimisers (generator and discriminator).
    pub lr: f32,
    /// Adam's `(beta1, beta2)` for both optimisers.
    pub adam_betas: (f32, f32),
    /// Seed of the discriminator's initial weights; `None` draws one from the environment (printed by the trainer's
    /// constructor so a run reproduces).
    pub disc_seed: Option<u64>,
    /// Weight of the codebook entropy term ([`crate::entropy`]); zero computes nothing.
    pub entropy_loss_ratio: f32,
    /// The entropy term's softmax temperature (LlamaGen's `entropy_loss_temperature`).
    pub entropy_temperature: f32,
}

impl Objective {
    /// L1 and the VQ terms alone: no perceptual term, no discriminator, no entropy term.
    pub fn reconstruction_only() -> Objective {
        Objective {
            reconstruction: ReconstructionLoss::L1,
            perceptual_weight: 0.0,
            perceptual_net: Trunk::Vgg16,
            disc_weight: 0.0,
            disc_start: 0,
            disc_ndf: 64,
            disc_layers: 3,
            adaptive_weight: true,
            adaptive_weight_max: 1e4,
            scale_disc_loss: false,
            lr: 1e-4,
            adam_betas: (0.9, 0.999),
            disc_seed: None,
            entropy_loss_ratio: 0.0,
            entropy_temperature: 0.01,
        }
    }

    /// CodeFormer's VQGAN stage-1 recipe (module docs).
    pub fn codeformer() -> Objective {
        Objective {
            perceptual_weight: 1.0,
            disc_weight: 1.0,
            disc_start: 30_001,
            disc_layers: 4,
            adaptive_weight: true,
            lr: 7e-5,
            adam_betas: (0.9, 0.99),
            ..Objective::reconstruction_only()
        }
    }

    /// LlamaGen's VQ-16 recipe (module docs); its entropy ratio is 0 in the released flags.
    pub fn llamagen_vq16() -> Objective {
        Objective {
            reconstruction: ReconstructionLoss::L2,
            perceptual_weight: 1.0,
            disc_weight: 0.5,
            disc_start: 20_000,
            adaptive_weight: false,
            scale_disc_loss: true,
            lr: 1e-4,
            adam_betas: (0.9, 0.95),
            ..Objective::reconstruction_only()
        }
    }

    /// The optimiser settings of a step under this objective: Adam at [`Self::lr`] and [`Self::adam_betas`], no
    /// weight decay (both recipes use plain Adam) and no gradient clip.
    pub fn step_options(&self) -> StepOptions {
        StepOptions { lr_generator: self.lr, lr_discriminator: self.lr, weight_decay: 0.0, adam: Adam { beta1: self.adam_betas.0, beta2: self.adam_betas.1, ..Adam::default() }, clip: None }
    }

    /// Whether the discriminator exists at all.
    pub fn adversarial(&self) -> bool {
        self.disc_weight > 0.0
    }

    /// Whether the adversarial term and the discriminator's updates are live at `step`.
    pub fn adversarial_at(&self, step: u32) -> bool {
        self.adversarial() && step >= self.disc_start
    }

    /// Refuse a weight that is negative or not finite, a discriminator with no layers, a non-positive clamp or
    /// temperature: each would train something other than the recipe it names.
    pub fn validate(&self) -> Result<(), String> {
        let weights = [
            ("perceptual_weight", self.perceptual_weight),
            ("disc_weight", self.disc_weight),
            ("entropy_loss_ratio", self.entropy_loss_ratio),
        ];
        for (name, v) in weights {
            if !v.is_finite() || v < 0.0 {
                return Err(format!("vqgan objective: {name} is {v}, a weight must be finite and not negative"));
            }
        }
        for (name, v) in [("adaptive_weight_max", self.adaptive_weight_max), ("entropy_temperature", self.entropy_temperature), ("lr", self.lr)] {
            if !v.is_finite() || v <= 0.0 {
                return Err(format!("vqgan objective: {name} is {v}, it must be finite and positive"));
            }
        }
        let (b1, b2) = self.adam_betas;
        if !(0.0..1.0).contains(&b1) || !(0.0..1.0).contains(&b2) {
            return Err(format!("vqgan objective: adam_betas ({b1}, {b2}) must each be in [0, 1)"));
        }
        if self.adversarial() && (self.disc_ndf == 0 || self.disc_layers == 0) {
            return Err(format!("vqgan objective: a discriminator of {} filters and {} layers does not exist", self.disc_ndf, self.disc_layers));
        }
        Ok(())
    }
}

/// The optimiser settings of one [`crate::train::VqganTrainer::train_step`]: AdamW for the generator (encoder,
/// generator and codebook) and for the discriminator.
#[derive(Clone, Copy, Debug)]
pub struct StepOptions {
    pub lr_generator: f32,
    pub lr_discriminator: f32,
    pub weight_decay: f32,
    pub adam: Adam,
    /// Global gradient-norm clip of the generator's update.
    pub clip: Option<f32>,
}

/// What one step measured. Every generator term is the value that entered the loss, weight included.
#[derive(Clone, Debug, PartialEq)]
pub struct Losses {
    pub reconstruction: f64,
    pub perceptual: f64,
    pub codebook: f64,
    pub commitment: f64,
    pub entropy: f64,
    /// `disc_weight * lambda * (-mean D(x_hat))`, when the adversarial term was live.
    pub adversarial: Option<f64>,
    /// `lambda`, before `disc_weight`, when the adversarial term was live and the recipe adapts it.
    pub adaptive_weight: Option<f32>,
    /// The discriminator's loss as optimised ([`Objective::scale_disc_loss`]), when it updated.
    pub discriminator: Option<f64>,
}

impl Losses {
    /// The generator's objective: every term above but the discriminator's own loss.
    pub fn generator(&self) -> f64 {
        self.reconstruction + self.perceptual + self.codebook + self.commitment + self.entropy + self.adversarial.unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec: the presets carry the published values (module docs) and are valid objectives.
    #[test]
    fn the_presets_hold_the_published_recipes() {
        let l = Objective::llamagen_vq16();
        assert_eq!((l.reconstruction, l.perceptual_weight, l.perceptual_net), (ReconstructionLoss::L2, 1.0, Trunk::Vgg16));
        assert_eq!((l.disc_weight, l.disc_start, l.disc_ndf, l.disc_layers), (0.5, 20_000, 64, 3));
        assert_eq!((l.adaptive_weight, l.scale_disc_loss, l.entropy_loss_ratio, l.entropy_temperature), (false, true, 0.0, 0.01));
        assert_eq!((l.lr, l.adam_betas), (1e-4, (0.9, 0.95)));
        let c = Objective::codeformer();
        assert_eq!((c.reconstruction, c.perceptual_weight, c.perceptual_net), (ReconstructionLoss::L1, 1.0, Trunk::Vgg16));
        assert_eq!((c.disc_weight, c.disc_start, c.disc_ndf), (1.0, 30_001, 64));
        assert_eq!((c.adaptive_weight, c.scale_disc_loss), (true, false));
        assert_eq!((c.lr, c.adam_betas), (7e-5, (0.9, 0.99)));
        let r = Objective::reconstruction_only();
        assert!(!r.adversarial() && r.perceptual_weight == 0.0 && r.entropy_loss_ratio == 0.0);
        for o in [l, c, r] {
            o.validate().expect("a preset is a valid objective");
        }
    }

    /// Spec: the step's optimiser settings are the objective's, for both optimisers.
    #[test]
    fn the_step_options_follow_the_objective() {
        let o = Objective::codeformer().step_options();
        assert_eq!((o.lr_generator, o.lr_discriminator, o.weight_decay), (7e-5, 7e-5, 0.0));
        assert_eq!((o.adam.beta1, o.adam.beta2), (0.9, 0.99));
    }

    /// Spec: the discriminator's updates and the adversarial term begin at `disc_start`, and never with a zero weight.
    #[test]
    fn the_adversarial_term_starts_at_disc_start() {
        let o = Objective { disc_weight: 0.5, disc_start: 10, ..Objective::reconstruction_only() };
        assert!(!o.adversarial_at(9) && o.adversarial_at(10) && o.adversarial_at(11));
        assert!(!Objective { disc_weight: 0.0, ..o }.adversarial_at(100));
    }

    #[test]
    fn a_weight_that_is_negative_or_not_finite_is_refused() {
        for bad in [-0.1, f32::NAN, f32::INFINITY] {
            for o in [
                Objective { perceptual_weight: bad, ..Objective::reconstruction_only() },
                Objective { disc_weight: bad, ..Objective::reconstruction_only() },
                Objective { entropy_loss_ratio: bad, ..Objective::reconstruction_only() },
            ] {
                assert!(o.validate().is_err(), "{o:?}");
            }
        }
        assert!(Objective { entropy_temperature: 0.0, ..Objective::reconstruction_only() }.validate().unwrap_err().contains("entropy_temperature"));
        assert!(Objective { disc_weight: 1.0, disc_layers: 0, ..Objective::reconstruction_only() }.validate().is_err());
    }
}
