// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The SDXL denoise loop, factored out from `pipeline::Sdxl` and
//! `controlnet::caps::Controlled` - it used to be two near-identical copies,
//! because `Sdxl` built its `Unet` with the plain constructor and had no seam
//! for `controlnet`'s per-step residual. [`Denoiser`] is that seam: a caller
//! provides ONE forward per step, and [`sample`] owns everything else
//! (seeding, sigma scaling, the CFG pair, the scheduler step, progress).
//!
//! Nothing here composes the two CLIP towers into `prompt_embeds`/`pooled`
//! (see [`crate::textenc`]) or runs the VAE - a caller builds a
//! [`Conditioning`](crate::textenc::Conditioning) pair and decodes the
//! returned latent itself, the same as before this was factored out.
//!
//! # Classifier-free guidance is two forwards, not a batched one
//!
//! `sdxlunet` records its graph for one sample, so the conditional and
//! unconditional passes are two [`Denoiser::eval`] calls rather than a batch
//! of two. That is a cost (two forwards per step) and not a correctness
//! question; batching would need a graph recorded at `b = 2`.
//!
//! # The micro-conditioning is not decoration
//!
//! SDXL's `add_time_ids` is `[orig_h, orig_w, crop_top, crop_left, target_h,
//! target_w]`, projected and added to the timestep embedding. [`sample`]
//! reproduces diffusers' defaults (`original_size = target_size = the
//! generated size`, `crops_coords_top_left = (0,0)`), because those values
//! genuinely change the composition - they are how SDXL was taught that a
//! crop is a crop.

use diffusion::discrete::{DiscreteConfig, EulerScheduler};

use crate::textenc::Conditioning;

/// One denoise step's shared context - the same for every [`Denoiser::eval`]
/// call within a step (both the conditional and unconditional branch, when
/// CFG is on).
pub struct StepCtx<'a> {
    pub step: usize,
    pub n_steps: usize,
    pub sigma: f32,
    pub timestep: f32,
    /// The scaled latent this step's forward passes both read.
    pub scaled: &'a [f32],
}

/// One backbone forward: a scaled latent plus one prompt's conditioning in,
/// that step's noise prediction out. `Sdxl` implements this as a plain
/// `Unet::run`; `controlnet::caps::Controlled` implements it as a
/// `ControlNet::run` feeding `Unet::run_with_control` - the per-step residual
/// [`Denoiser::eval`] exists for.
pub trait Denoiser {
    fn eval(&self, ctx: &StepCtx<'_>, enc: &[f32], pooled: &[f32], time_ids: &[f32]) -> Result<Vec<f32>, String>;
}

/// How the latent is seeded and how many steps to take.
pub struct SamplerOptions {
    pub steps: usize,
    /// Classifier-free guidance scale. 1.0 disables CFG and halves the work.
    pub guidance: f32,
    pub seed: u64,
    /// Generated size in pixels - feeds `add_time_ids`, not the latent shape
    /// (the caller already sized `n_latent` for that).
    pub height: u32,
    pub width: u32,
}

/// Run the denoise loop and return the final SCALED latent (the caller
/// divides by the VAE's `scaling_factor` before decoding, as before).
///
/// `n_latent` is `in_channels * lh * lw`; `uncond` is `None` to skip CFG
/// entirely (matching `guidance <= 1.0`'s existing meaning at both call
/// sites).
///
/// `cancel` is checked on entry and before every step, so a fired token ends
/// the run with `Err("cancelled")` before the next forward starts; the step
/// in flight when it fires is the only one that completes.
pub fn sample(
    d: &dyn Denoiser,
    n_latent: usize,
    cond: &Conditioning,
    uncond: Option<&Conditioning>,
    o: &SamplerOptions,
    cancel: &capability::CancelToken,
    on_step: &mut dyn FnMut(u32, u32),
) -> Result<Vec<f32>, String> {
    cancel.check()?;
    let mut sched = EulerScheduler::new(DiscreteConfig::sdxl());
    sched.set_timesteps(o.steps);

    // diffusers' micro-conditioning defaults: the generated size is both the
    // "original" and the "target", with no crop.
    let time_ids = vec![o.height as f32, o.width as f32, 0.0, 0.0, o.height as f32, o.width as f32];

    let mut lat = model::hostmath::gaussian(n_latent, o.seed);
    let s0 = sched.init_noise_sigma();
    for v in &mut lat {
        *v *= s0;
    }

    let timesteps: Vec<f32> = sched.timesteps().to_vec();
    for (i, &t) in timesteps.iter().enumerate() {
        cancel.check()?;
        let scaled = sched.scale_model_input(&lat);
        let ctx = StepCtx { step: i, n_steps: timesteps.len(), sigma: sched.sigmas()[i], timestep: t, scaled: &scaled };

        let c = d.eval(&ctx, &cond.0, &cond.1, &time_ids)?;
        let eps = match uncond {
            None => c,
            Some((ue, up)) => {
                let u = d.eval(&ctx, ue, up, &time_ids)?;
                // guided = uncond + g * (cond - uncond)
                u.iter().zip(&c).map(|(a, b)| a + o.guidance * (b - a)).collect()
            }
        };
        lat = sched.step(&eps, &lat);
        on_step(i as u32 + 1, timesteps.len() as u32);
    }

    Ok(lat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::{Invocation, Outcome, Progress};
    use capability_mock::cancellation::{certify, Bound};
    use std::cell::RefCell;

    /// A backbone that returns zeros and reports each forward as one progress
    /// event, so the real loop runs without a checkpoint.
    struct Counting<'a, 'b> {
        progress: &'a RefCell<&'b mut dyn FnMut(Progress)>,
        steps: usize,
    }

    impl Denoiser for Counting<'_, '_> {
        fn eval(&self, ctx: &StepCtx<'_>, _enc: &[f32], _pooled: &[f32], _time_ids: &[f32]) -> Result<Vec<f32>, String> {
            (self.progress.borrow_mut())(Progress::step(ctx.step as u32, self.steps as u32, ""));
            Ok(vec![0.0; ctx.scaled.len()])
        }
    }

    #[test]
    fn sample_stops_within_one_step_of_its_token_firing() {
        const STEPS: usize = 12;
        let run = |inv: &Invocation, progress: &mut dyn FnMut(Progress)| {
            let sink = RefCell::new(progress);
            let denoiser = Counting { progress: &sink, steps: STEPS };
            let cond = (vec![0.0; 4], vec![0.0; 2]);
            let so = SamplerOptions { steps: STEPS, guidance: 1.0, seed: 1, height: 16, width: 16 };
            sample(&denoiser, 8, &cond, None, &so, &inv.cancel, &mut |_, _| {}).map(|_| Outcome::new())
        };
        certify("sdxl sampler", &Invocation::new, &run, Bound::ONE_STEP);
    }

    /// Every denoise step is reported once, in order, ending at the total:
    /// the per-step progress `Session::run` and the residents forward.
    #[test]
    fn every_denoise_step_is_reported() {
        const STEPS: usize = 6;
        let mut quiet = |_: Progress| {};
        let sink: RefCell<&mut dyn FnMut(Progress)> = RefCell::new(&mut quiet);
        let denoiser = Counting { progress: &sink, steps: STEPS };
        let cond = (vec![0.0; 4], vec![0.0; 2]);
        let so = SamplerOptions { steps: STEPS, guidance: 1.0, seed: 1, height: 16, width: 16 };
        let mut seen = Vec::new();
        sample(&denoiser, 8, &cond, None, &so, &capability::CancelToken::default(), &mut |done, total| seen.push((done, total))).expect("sample");
        assert_eq!(seen, (1..=STEPS as u32).map(|d| (d, STEPS as u32)).collect::<Vec<_>>());
    }
}
