// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements adversarial fine-tuning of neural audio decoders for its clients. If your team needs
// expertise in GAN vocoder training, you can procure our services by sending an email to info@swedishembedded.com.

//! The joint generator-discriminator loop of the vocoder's adversarial fine-tune ([`GanTrainer`]).
//!
//! The generator is [`crate::train::Trainer`] (the vocoder, its device weights and backward); the discriminators are
//! [`crate::discriminator`]'s STFT-magnitude PatchGANs, one per resolution of [`GanConfig::mrd`]. Each step, in the
//! HiFi-GAN order:
//!
//! 1. the generator's forward gives the generated audio, one mono waveform per `(batch, channel)` row;
//! 2. once `disc_start_step` steps have passed, the discriminators take an LSGAN step on real (the target) against
//!    generated audio, with their own AdamW (`disc_lr`);
//! 3. the generator's objective `rec_weight * MSE + adv_weight * LSGAN-G + fm_weight * feature-matching` is
//!    differentiated against the UPDATED discriminators: the adversarial and feature-matching gradients on each
//!    waveform ([`generator_adversarial`], through the STFT) join the MSE gradient at the vocoder's output
//!    ([`Trainer::backward_joint`]) and the vocoder's own backward runs on from there;
//! 4. the generator takes an AdamW step (`gen_lr`).
//!
//! Before `disc_start_step` the run is reconstruction only. Per-row losses are averaged over the `batch * 2` rows.
//!
//! Defaults are the published BigVGAN/HiFi-GAN recipe's, because MiniMax publishes no vocoder training code:
//! LSGAN, feature-matching weight 2 (HiFi-GAN), learning rate 2e-4, and three STFT resolutions with `hop = n_fft / 4`
//! (BigVGAN's multi-resolution discriminator uses three `n_fft` in a 4:2:1 ratio). `disc_start_step` defaults to 0,
//! as in HiFi-GAN, which trains both from the first step; a run that wants the discriminator to meet a converged
//! generator sets it to the MSE-only phase's length.

use gpu_core::Gpu;
use model::hostopt::HostAdam;
use model::resident_ft::StepArgs;
use model::Adam;
use model::memplan::ParamPrecision;
use std::collections::HashMap;

use crate::config::VocoderConfig;
use crate::discriminator::{self, generator_adversarial, mrd_discriminator_step, random_mrd_weights, DiscConfig, DiscWeights, MrdConfig, StftConfig};
use crate::train::Trainer;
use crate::vocoder::VocoderWeights;

/// Everything the joint loop is parameterised by.
#[derive(Clone, Debug)]
pub struct GanConfig {
    pub mrd: MrdConfig,
    pub rec_weight: f32,
    pub adv_weight: f32,
    pub fm_weight: f32,
    /// Steps of reconstruction-only training before the discriminators and the adversarial terms start.
    pub disc_start_step: u32,
    pub gen_lr: f32,
    pub disc_lr: f32,
    pub adam: Adam,
    /// Seed of the discriminators' initial weights; `None` draws a random one (`data::rng::random_seed`).
    pub disc_seed: Option<u64>,
}

impl Default for GanConfig {
    fn default() -> GanConfig {
        let resolutions = [512usize, 1024, 2048].into_iter().map(|n_fft| StftConfig { n_fft, hop: n_fft / 4 }).collect();
        GanConfig {
            mrd: MrdConfig { resolutions, disc: DiscConfig { c1: 16, c2: 32, k: 3 } },
            rec_weight: 1.0,
            adv_weight: 1.0,
            fm_weight: 2.0,
            disc_start_step: 0,
            gen_lr: 2e-4,
            disc_lr: 2e-4,
            adam: Adam::default(),
            disc_seed: None,
        }
    }
}

/// The generator's objective terms on the current batch (unweighted); `adv` and `fm` are 0 when the discriminators
/// are not in play.
#[derive(Clone, Copy, Debug)]
pub struct Objective {
    pub rec: f32,
    pub adv: f32,
    pub fm: f32,
}

impl Objective {
    pub fn total(&self, cfg: &GanConfig) -> f32 {
        cfg.rec_weight * self.rec + cfg.adv_weight * self.adv + cfg.fm_weight * self.fm
    }
}

/// What one [`GanTrainer::step`] measured: the generator objective at the step's start, and the discriminator's loss
/// (`None` before its start step).
#[derive(Clone, Copy, Debug)]
pub struct GanStats {
    pub objective: Objective,
    pub disc_loss: Option<f32>,
}

pub struct GanTrainer {
    cfg: GanConfig,
    generator: Trainer,
    disc_gpu: Gpu,
    disc: Vec<DiscWeights>,
    gen_opt: HostAdam,
    disc_opt: HostAdam,
    gen_t: u32,
    disc_t: u32,
    steps_done: u32,
}

fn args(step: u32, lr: f32, adam: Adam) -> StepArgs {
    StepArgs { step, lr, wd: 0.0, adam, clip: None, extra_scale: 1.0 }
}

impl GanTrainer {
    /// A joint trainer over a vocoder at `w` and a fixed `(latents, target)` batch, the discriminators drawn from a
    /// random seed.
    pub fn new(cfg: GanConfig, vocoder_cfg: VocoderConfig, w: &VocoderWeights, latents: Vec<f32>, batch: usize, length: usize, target: Vec<f32>) -> GanTrainer {
        let generator = Trainer::new(vocoder_cfg, w, latents, batch, length, target);
        let mut disc = random_mrd_weights(&cfg.mrd, cfg.disc_seed.unwrap_or_else(data::rng::random_seed));
        let gen_params: Vec<(String, usize)> = generator.param_names().into_iter().map(|n| (n.clone(), generator.read_weight(&n).len())).collect();
        let disc_params: Vec<(String, usize)> = disc
            .iter_mut()
            .enumerate()
            .flat_map(|(r, d)| d.tensors_mut().into_iter().enumerate().map(|(t, v)| (format!("disc{r}.{t}"), v.len())).collect::<Vec<_>>())
            .collect();
        let optimiser = |p: &[(String, usize)]| HostAdam::new(p, ParamPrecision::FP32).expect("an f32 host optimiser is always accepted");
        GanTrainer {
            gen_opt: optimiser(&gen_params),
            disc_opt: optimiser(&disc_params),
            disc_gpu: Gpu::new_cpu(discriminator::PIPELINES),
            cfg,
            generator,
            disc,
            gen_t: 0,
            disc_t: 0,
            steps_done: 0,
        }
    }

    pub fn config(&self) -> &GanConfig {
        &self.cfg
    }

    /// The vocoder under training (its weights, gradients and last forward).
    pub fn generator(&self) -> &Trainer {
        &self.generator
    }

    pub fn discriminator_weights(&self) -> &[DiscWeights] {
        &self.disc
    }

    fn rows<'a>(&self, wave: &'a [f32]) -> std::slice::Chunks<'a, f32> {
        wave.chunks(self.generator.out_len())
    }

    /// The generator objective on the current weights with the discriminators fixed; `adversarial` false is the
    /// reconstruction term alone. Leaves the generator's forward cached.
    pub fn objective(&self, adversarial: bool) -> Objective {
        let rec = self.generator.loss();
        if !adversarial {
            return Objective { rec, adv: 0.0, fm: 0.0 };
        }
        let fake = self.generator.output();
        let (adv, fm, _) = self.adversarial_terms(&fake);
        Objective { rec, adv, fm }
    }

    /// Zero the generator's gradients and run its backward on the joint objective; the gradients stay in
    /// [`GanTrainer::generator`]. Returns the objective they are the gradient of.
    pub fn generator_gradients(&self, adversarial: bool) -> Objective {
        self.generator.zero_grads();
        let rec = self.generator.loss();
        if !adversarial {
            self.generator.backward();
            return Objective { rec, adv: 0.0, fm: 0.0 };
        }
        let fake = self.generator.output();
        let (adv, fm, d_fake) = self.adversarial_terms(&fake);
        self.generator.backward_joint(self.cfg.rec_weight, &d_fake);
        Objective { rec, adv, fm }
    }

    /// Row-averaged `(adversarial loss, feature-matching loss, weighted gradient on `fake`)`.
    fn adversarial_terms(&self, fake: &[f32]) -> (f32, f32, Vec<f32>) {
        let n_rows = (fake.len() / self.generator.out_len()) as f32;
        let (mut adv, mut fm) = (0.0, 0.0);
        let mut d_fake = Vec::with_capacity(fake.len());
        for (real_row, fake_row) in self.rows(self.generator.target()).zip(self.rows(fake)) {
            let g = generator_adversarial(&self.disc_gpu, &self.disc, &self.cfg.mrd, real_row, fake_row, self.cfg.adv_weight, self.cfg.fm_weight);
            adv += g.adv_loss / n_rows;
            fm += g.fm_loss / n_rows;
            d_fake.extend(g.d_fake.iter().map(|d| d / n_rows));
        }
        (adv, fm, d_fake)
    }

    /// One discriminator update on the generator's current output against the target, regardless of
    /// `disc_start_step`; returns its row-averaged LSGAN loss. The generator is not touched.
    pub fn discriminator_step(&mut self) -> f32 {
        self.generator.loss();
        let fake = self.generator.output();
        let n_rows = (fake.len() / self.generator.out_len()) as f32;
        let mut grads: Vec<Vec<f32>> = self.disc.iter_mut().flat_map(|d| d.tensors_mut().into_iter().map(|t| vec![0.0; t.len()])).collect();
        let mut loss = 0.0;
        for (real_row, fake_row) in self.rows(self.generator.target()).zip(self.rows(&fake)) {
            let step = mrd_discriminator_step(&self.disc_gpu, &self.disc, &self.cfg.mrd, real_row, fake_row);
            loss += step.loss / n_rows;
            let flat = step.steps.iter().flat_map(|s| s.tensors());
            for (acc, g) in grads.iter_mut().zip(flat) {
                acc.iter_mut().zip(g).for_each(|(a, g)| *a += g / n_rows);
            }
        }
        self.disc_t += 1;
        let mut tensors: Vec<(&mut [f32], &[f32])> = self.disc.iter_mut().flat_map(|d| d.tensors_mut()).zip(grads.iter().map(Vec::as_slice)).collect();
        self.disc_opt.step(&mut tensors, &args(self.disc_t, self.cfg.disc_lr, self.cfg.adam)).expect("the optimiser was built from these tensors");
        loss
    }

    /// One joint step (module docs); the stats' objective is measured before the updates.
    pub fn step(&mut self) -> GanStats {
        let adversarial = self.steps_done >= self.cfg.disc_start_step;
        let disc_loss = adversarial.then(|| self.discriminator_step());
        let objective = self.generator_gradients(adversarial);
        self.update_generator();
        self.steps_done += 1;
        GanStats { objective, disc_loss }
    }

    fn update_generator(&mut self) {
        let names = self.generator.param_names();
        let mut values: HashMap<String, Vec<f32>> = names.iter().map(|n| (n.clone(), self.generator.read_weight(n))).collect();
        let grads: HashMap<String, Vec<f32>> = names.iter().map(|n| (n.clone(), self.generator.read_grad(n))).collect();
        self.gen_t += 1;
        self.gen_opt.step_named(&mut values, &grads, &args(self.gen_t, self.cfg.gen_lr, self.cfg.adam)).expect("the optimiser was built from these tensors");
        for (n, v) in &values {
            self.generator.write_weight(n, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VocoderConfig;
    use crate::discriminator::{DiscConfig, MrdConfig, StftConfig};
    use crate::train::{random_weights, Trainer};
    use brain_testutil::{fd::Directional, rounding};
    use data::rng::Lcg;

    const CHAIN_ROUNDINGS: usize = 1024;

    fn tiny_mrd() -> MrdConfig {
        MrdConfig { resolutions: vec![StftConfig { n_fft: 8, hop: 4 }, StftConfig { n_fft: 16, hop: 8 }], disc: DiscConfig { c1: 2, c2: 2, k: 3 } }
    }

    /// A tiny vocoder over a fixed (latents, target) batch: 8 latent frames upsample 4x to 32 samples per channel.
    fn tiny_gan(cfg: GanConfig) -> GanTrainer {
        tiny_gan_seeded(cfg, data::rng::random_seed())
    }

    fn tiny_gan_seeded(cfg: GanConfig, seed: u64) -> GanTrainer {
        let vc = VocoderConfig::tiny();
        let w = random_weights(&vc, 5);
        let (batch, length) = (1usize, 8usize);
        let mut r = Lcg::new(seed);
        let latents = r.vec_scaled(batch * vc.latent_channels as usize * length, 0.5);
        let out_len = length * vc.upsampling_ratios.iter().product::<u32>() as usize;
        let target = r.vec_scaled(batch * 2 * out_len, 0.5);
        GanTrainer::new(cfg, vc, &w, latents, batch, length, target)
    }

    fn cfg() -> GanConfig {
        GanConfig { mrd: tiny_mrd(), rec_weight: 1.0, adv_weight: 1.0, fm_weight: 2.0, disc_start_step: 0, gen_lr: 1e-2, disc_lr: 1e-2, ..GanConfig::default() }
    }

    /// Spec: the generator gradient of `rec_weight * MSE + adv_weight * LSGAN + fm_weight * FM` (through the
    /// discriminator's STFT, convolutions and the vocoder's whole stack) is the derivative of that objective.
    ///
    /// Tolerance: the objective is a sum of non-negative terms (`S` = its value), each accumulated over at most
    /// `K` binary32 roundings along the longest chain: the vocoder's twelve or so reductions of at most 56 terms, the
    /// discriminator's three convolutions of at most 36, a 16-term DFT and the loss sums, under 1024 in all. Two evaluations differ from the exact loss by at most `gamma_K * S` each, so the
    /// central difference over `2 eps` is off by at most `agreement_bound(K, S) / (2 eps)`, and the one-sided difference the kink test
    /// compares by `agreement_bound(K, S) / eps`, the larger, which `Directional` uses for both. The rest is the
    /// truncation of the central difference, allowed as a relative 2%.
    #[test]
    fn the_joint_generator_gradient_matches_central_differences() {
        let gan = tiny_gan(cfg());
        let base = gan.generator_gradients(true);
        let eps = 2e-3f32;
        let abs_tol = (rounding::agreement_bound(CHAIN_ROUNDINGS, f64::from(base.total(&gan.config()))) / f64::from(eps)) as f32;
        let check = Directional { eps, abs_tol, rel_tol: 2e-2 };
        let mut dirs = Lcg::new(data::rng::random_seed());
        let mut checked = 0;
        for name in gan.generator().param_names() {
            let w0 = gan.generator().read_weight(&name);
            let grad = gan.generator().read_grad(&name);
            let mut loss_at = |w: &[f32]| {
                gan.generator().write_weight(&name, w);
                f64::from(gan.objective(true).total(&gan.config()))
            };
            let res = check.check(&grad, &w0, || (0..w0.len()).map(|_| if dirs.unit() < 0.5 { -1.0 } else { 1.0 }).collect(), &mut loss_at);
            gan.generator().write_weight(&name, &w0);
            res.unwrap_or_else(|e| panic!("{name}: {e}"));
            checked += 1;
        }
        assert!(checked > 30, "expected every vocoder parameter to be checked, got {checked}");
    }

    /// The check above is only worth its tolerance if a wrong gradient fails it. The realistic bug is a backward that
    /// forgets the adversarial path: the reconstruction-only gradient, checked against the joint objective on the
    /// parameter the two differ most on (along the direction of their difference), must be rejected.
    #[test]
    fn the_gradient_check_rejects_a_backward_that_drops_the_adversarial_terms() {
        let gan = tiny_gan(cfg());
        let joint = gan.generator_gradients(true);
        let abs_tol = (rounding::agreement_bound(CHAIN_ROUNDINGS, f64::from(joint.total(&gan.config()))) / 2e-3) as f32;
        let names = gan.generator().param_names();
        let full: Vec<Vec<f32>> = names.iter().map(|n| gan.generator().read_grad(n)).collect();
        gan.generator_gradients(false);
        let rec_only: Vec<Vec<f32>> = names.iter().map(|n| gan.generator().read_grad(n)).collect();
        let gap = |i: usize| full[i].iter().zip(&rec_only[i]).map(|(f, r)| (f - r).abs()).sum::<f32>();
        let worst = (0..names.len()).max_by(|&a, &b| gap(a).total_cmp(&gap(b))).unwrap();
        let name = &names[worst];
        let w0 = gan.generator().read_weight(name);
        let dir: Vec<f32> = full[worst].iter().zip(&rec_only[worst]).map(|(f, r)| if f >= r { 1.0 } else { -1.0 }).collect();
        let check = Directional { eps: 2e-3, abs_tol, rel_tol: 2e-2 };
        let mut loss_at = |w: &[f32]| {
            gan.generator().write_weight(name, w);
            f64::from(gan.objective(true).total(&gan.config()))
        };
        let res = check.check(&rec_only[worst], &w0, || dir.clone(), &mut loss_at);
        assert!(res.is_err(), "{name}: a gradient without the adversarial terms passed the check: {res:?}");
    }

    /// Spec: with the generator fixed, discriminator updates alone lower its LSGAN loss on real vs generated audio.
    #[test]
    fn the_discriminator_loss_falls_when_only_it_trains() {
        let mut gan = tiny_gan(cfg());
        let first = gan.discriminator_step();
        let mut last = first;
        for _ in 0..60 {
            last = gan.discriminator_step();
        }
        assert!(last < first * 0.7, "discriminator loss did not fall: {first} -> {last}");
    }

    /// Spec: the generator trained against the discriminator lowers the joint objective, and its reconstruction term
    /// falls with it. Fixed data and discriminator seeds, so the bar is a bar rather than a draw (as the MSE-only
    /// vocoder overfit test); the discriminator learns slowly, so the generator is chasing a nearly fixed critic.
    #[test]
    fn the_joint_objective_falls_over_a_short_overfit() {
        let mut gan = tiny_gan_seeded(GanConfig { gen_lr: 3e-2, disc_lr: 1e-4, disc_seed: Some(17), ..cfg() }, 31);
        let start = gan.objective(true);
        let mut last = start;
        for _ in 0..400 {
            last = gan.step().objective;
        }
        let (t0, t1) = (start.total(&gan.config()), last.total(&gan.config()));
        assert!(t1 < t0 * 0.85, "joint objective did not fall: {t0} -> {t1}");
        assert!(last.rec < start.rec * 0.5, "reconstruction did not fall: {} -> {}", start.rec, last.rec);
    }

    /// Spec: before `disc_start_step` the run is reconstruction only: the discriminator's weights do not move and the
    /// adversarial terms are not in the step's objective.
    #[test]
    fn the_discriminator_waits_for_its_start_step() {
        let mut gan = tiny_gan(GanConfig { disc_start_step: 2, ..cfg() });
        let before = gan.discriminator_weights()[0].conv1.weight.clone();
        for _ in 0..2 {
            let s = gan.step();
            assert!(s.disc_loss.is_none(), "the discriminator trained before its start step");
            assert_eq!((s.objective.adv, s.objective.fm), (0.0, 0.0));
        }
        assert_eq!(gan.discriminator_weights()[0].conv1.weight, before);
        let s = gan.step();
        assert!(s.disc_loss.is_some(), "the discriminator did not start at its start step");
        assert_ne!(gan.discriminator_weights()[0].conv1.weight, before);
    }

    #[test]
    fn the_default_recipe_is_three_resolutions() {
        let c = GanConfig::default();
        assert_eq!(c.mrd.resolutions.len(), 3);
        assert!(c.mrd.resolutions.iter().all(|r| r.hop < r.n_fft));
        let _: &Trainer = tiny_gan(cfg()).generator();
    }
}
