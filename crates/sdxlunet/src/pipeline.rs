// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The SDXL text-to-image pipeline: a prompt in, an image out.
//!
//! Everything this assembles was already parity-gated on its own — the two CLIP
//! towers (`crates/clip`, 148 stage checks), the UNet (165 comparisons), the VAE
//! (`crates/vae`) and the discrete schedulers (`crates/diffusion`, 66 checks).
//! What was missing was the loop that puts them together, which is why nothing
//! in the imaging workstream could produce a picture. The loop itself is
//! [`crate::sampler::sample`]; the conditioning is [`crate::textenc::TextEncoders`].
//! This module is a thin [`crate::sampler::Denoiser`] impl around the plain
//! [`Unet`] plus the load/decode glue - see those modules for the sampling and
//! conditioning documentation.

use std::path::Path;

use gpu_core::Gpu;
use vae::config::VaeConfig;
use vae::VaeDecoder;

use crate::config::UNetConfig;
use crate::model::{Unet, KERNELS};
use crate::sampler::{sample, Denoiser, SamplerOptions, StepCtx};
use crate::textenc::{read_any_safetensors, read_json, TextEncoders, CONTEXT};

/// How the latent is seeded and how many steps to take.
pub struct GenerateOptions {
    pub steps: usize,
    /// Classifier-free guidance scale. 1.0 disables CFG and halves the work.
    pub guidance: f32,
    pub seed: u64,
    /// Generated size in pixels; must be a multiple of the VAE's 8x downscale.
    pub height: u32,
    pub width: u32,
    pub negative: String,
}

impl Default for GenerateOptions {
    fn default() -> GenerateOptions {
        GenerateOptions {
            steps: 30,
            guidance: 5.0,
            seed: 0,
            height: 1024,
            width: 1024,
            negative: String::new(),
        }
    }
}

/// A loaded SDXL stack.
///
/// # Only the UNet stays resident, and that is not an optimisation
///
/// SDXL is ~3.5 B parameters across four models — about 14 GB at fp32 — and a
/// non-ReBAR Pascal card carries roughly 2x resident overhead per storage
/// buffer, so holding the UNet, both text encoders and the VAE at once does not
/// fit 24 GB. It OOMs, which is how this was found.
///
/// The two text encoders are needed ONCE per generation and the VAE once at the
/// end, while the UNet runs every step — so the encoders are built for the
/// encode and dropped (via [`crate::textenc::TextEncoders`]), and the VAE is
/// built for the decode and dropped. Same tiering as FLUX.1, done by
/// construction here rather than through `crates/residency`, because this
/// pipeline owns its own lifetimes.
pub struct Sdxl {
    gpu: Gpu,
    root: std::path::PathBuf,
    unet: Unet,
    vae_cfg: VaeConfig,
    ucfg: UNetConfig,
    hw: (u32, u32),
}

impl Sdxl {
    /// Load from a diffusers checkpoint root (the released SDXL layout).
    ///
    /// `h`/`w` are the generated size: the UNet's graph is recorded for one
    /// latent resolution, so a different size needs a different `Sdxl`.
    pub fn load(root: &str, h: u32, w: u32) -> Result<Sdxl, String> {
        let r = Path::new(root);
        let scale = 8u32; // the SDXL VAE's spatial downscale
        if !h.is_multiple_of(scale) || !w.is_multiple_of(scale) {
            return Err(format!("sdxl: {w}x{h} is not a multiple of the VAE's {scale}x downscale"));
        }
        let (lh, lw) = (h / scale, w / scale);

        // ONE device, several kernel sets. Each model resolves kernel indices
        // against the list ITS crate registered, so building `ClipText` on a
        // `Gpu` made from `sdxlunet::KERNELS` binds the wrong pipelines - a wrong
        // index is silently wrong output, and here it happened to surface as a
        // bind-group arity error rather than a bad picture.
        //
        // `Gpu::new_like` is exactly this case: a different kernel set on the
        // same device (AGENTS.md "one GPU device per process").
        let gpu = Gpu::new(&KERNELS);

        // --- unet ----------------------------------------------------------
        let ucfg = UNetConfig::sdxl_base();
        let udir = r.join("unet");
        let utensors = crate::import::load(udir.to_str().ok_or("sdxl: non-UTF8 unet path")?, &ucfg)?;
        let unet = Unet::new(gpu.share(), ucfg.clone(), &utensors, lh, lw, CONTEXT as u32, false);

        // --- vae config only; the decoder is built at decode time -----------
        let vae_cfg = VaeConfig::from_json(&read_json(&r.join("vae/config.json"))?);

        Ok(Sdxl { gpu, root: r.to_path_buf(), unet, vae_cfg, ucfg, hw: (h, w) })
    }

    /// Generate one image. Returns HWC RGB in `[0,1]`.
    pub fn generate(&mut self, prompt: &str, o: &GenerateOptions) -> Result<Vec<f32>, String> {
        let (h, w) = self.hw;
        let (lh, lw) = (h / 8, w / 8);
        let n = (self.ucfg.in_channels * lh * lw) as usize;

        let te = TextEncoders::load(self.gpu.share(), &self.root)?;
        let do_cfg = o.guidance > 1.0;
        let mut enc = if do_cfg { te.encode_all(&[prompt, o.negative.as_str()])? } else { te.encode_all(&[prompt])? };
        let uncond = do_cfg.then(|| enc.pop().expect("negative encoded"));
        let cond = enc.pop().expect("prompt encoded");
        // Both towers drop here (`te`'s scope ends), before the UNet runs a
        // single step.

        let denoiser = SdxlDenoiser { unet: &self.unet };
        let so = SamplerOptions { steps: o.steps, guidance: o.guidance, seed: o.seed, height: h, width: w };
        let lat = sample(&denoiser, n, &cond, uncond.as_ref(), &so)?;

        // The VAE decodes the UNSCALED latent. Built here and dropped on return,
        // so it never shares the card with the encoders.
        let sf = self.vae_cfg.scaling_factor;
        let z: Vec<f32> = lat.iter().map(|v| v / sf).collect();
        let vt = read_any_safetensors(&self.root.join("vae"))?;
        let vmap: vae::blocks::Tensors = vt.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();
        // Decode on the CPU by default. The UNet is still resident (10 GB at
        // fp32) and the VAE decode at 768^2 pushed a 24 GB card over — it OOMed
        // AFTER all 24 steps had run, which is the worst possible moment. The
        // decode is ONE pass, so the CPU cost is small next to losing the run;
        // `BRAIN_SDXL_VAE_DEVICE=gpu` forces the card when there is room.
        let vdev = std::env::var("BRAIN_SDXL_VAE_DEVICE").unwrap_or_else(|_| "cpu".into());
        let vae = VaeDecoder::from_diffusers(self.vae_cfg.clone(), &vmap, lh, lw, Some(&vdev));
        let chw = vae.decode(&z);
        // diffusers maps the decoder's [-1,1] output to [0,1].
        let rgb: Vec<f32> = chw.iter().map(|v| ((v + 1.0) * 0.5).clamp(0.0, 1.0)).collect();
        Ok(imaging::pixels::chw_to_hwc(&rgb, 3, h as usize, w as usize))
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }
}

/// The plain (uncontrolled) SDXL forward as a [`Denoiser`].
struct SdxlDenoiser<'a> {
    unet: &'a Unet,
}

impl Denoiser for SdxlDenoiser<'_> {
    fn eval(&self, ctx: &StepCtx<'_>, enc: &[f32], pooled: &[f32], time_ids: &[f32]) -> Result<Vec<f32>, String> {
        Ok(self.unet.run(ctx.scaled, ctx.timestep, enc, pooled, time_ids))
    }
}
