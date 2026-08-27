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
pub fn sample(
    d: &dyn Denoiser,
    n_latent: usize,
    cond: &Conditioning,
    uncond: Option<&Conditioning>,
    o: &SamplerOptions,
) -> Result<Vec<f32>, String> {
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
        if i % 5 == 0 || i + 1 == timesteps.len() {
            eprintln!("  step {}/{}  sigma {:.4}", i + 1, timesteps.len(), sched.sigmas()[i]);
        }
    }

    Ok(lat)
}
