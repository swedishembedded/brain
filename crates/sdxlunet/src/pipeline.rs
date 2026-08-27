// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The SDXL text-to-image pipeline: a prompt in, an image out.
//!
//! Everything this assembles was already parity-gated on its own — the two CLIP
//! towers (`crates/clip`, 148 stage checks), the UNet (165 comparisons), the VAE
//! (`crates/vae`) and the discrete schedulers (`crates/diffusion`, 66 checks).
//! What was missing was the loop that puts them together, which is why nothing
//! in the imaging workstream could produce a picture.
//!
//! # SDXL's conditioning is two encoders, and the layer index matters
//!
//! `prompt_embeds` is `concat(CLIP-L penultimate, OpenCLIP-bigG penultimate)`
//! along the feature axis — 768 + 1280 = 2048 — and `pooled_prompt_embeds` is
//! bigG's **projected** `text_embeds` alone. The PENULTIMATE hidden state, not
//! the last: diffusers passes `output_hidden_states=True` and takes
//! `hidden_states[-2]`. Taking the last layer instead runs, produces an image,
//! and is not SDXL.
//!
//! # Classifier-free guidance is two forwards, not a batched one
//!
//! `crates/sdxlunet` records its graph for one sample, so the conditional and
//! unconditional passes are two `run` calls rather than a batch of two. That is
//! a cost (two forwards per step) and not a correctness question; batching would
//! need a graph recorded at `b = 2`.
//!
//! # The micro-conditioning is not decoration
//!
//! SDXL's `add_time_ids` is `[orig_h, orig_w, crop_top, crop_left, target_h,
//! target_w]`, projected and added to the timestep embedding. The defaults here
//! reproduce diffusers' (`original_size = target_size = the generated size`,
//! `crops_coords_top_left = (0,0)`), because those values genuinely change the
//! composition — they are how SDXL was taught that a crop is a crop.

use std::path::Path;

use clip::config::ClipTextConfig;
use clip::model::ClipText;
use diffusion::discrete::{DiscreteConfig, EulerScheduler};
use gpu_core::Gpu;
use vae::config::VaeConfig;
use vae::VaeDecoder;

use crate::config::UNetConfig;
use crate::model::{Unet, KERNELS};

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
/// encode and dropped, and the VAE is built for the decode and dropped. Same
/// tiering as FLUX.1, done by construction here rather than through
/// `crates/residency`, because this pipeline owns its own lifetimes.
pub struct Sdxl {
    gpu: Gpu,
    root: std::path::PathBuf,
    tok_l: data::clip_bpe::ClipBpe,
    tok_g: data::clip_bpe::ClipBpe,
    unet: Unet,
    vae_cfg: VaeConfig,
    ucfg: UNetConfig,
    hw: (u32, u32),
}

const CONTEXT: usize = 77;

/// One prompt's SDXL conditioning: the `77 x 2048` sequence and the 1280-d
/// pooled vector.
pub type Conditioning = (Vec<f32>, Vec<f32>);

/// Read every `*.safetensors` in `dir`. The diffusers layout names a component's
/// weights after the component (`diffusion_pytorch_model[.fp16].safetensors`),
/// and a variant suffix is normal — so match on the extension, not the stem.
fn read_any_safetensors(dir: &Path) -> Result<Vec<checkpoint::safetensors::StTensor>, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("sdxl: reading {}: {e}", dir.display()))?;
    let mut files: Vec<std::path::PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("sdxl: no *.safetensors under {}", dir.display()));
    }
    let mut out = Vec::new();
    for f in files {
        out.extend(checkpoint::safetensors::read(f.to_str().ok_or("sdxl: non-UTF8 path")?)?);
    }
    Ok(out)
}

fn read_json(p: &Path) -> Result<serde_json::Value, String> {
    let s = std::fs::read_to_string(p).map_err(|e| format!("sdxl: reading {}: {e}", p.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("sdxl: parsing {}: {e}", p.display()))
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

        // --- tokenizers (cheap; the towers are built per encode) -------------
        let tok_l = data::clip_bpe::ClipBpe::from_dir(&r.join("tokenizer"))
            .map_err(|e| format!("sdxl: tokenizer: {e}"))?;
        let tok_g = data::clip_bpe::ClipBpe::from_dir(&r.join("tokenizer_2"))
            .map_err(|e| format!("sdxl: tokenizer_2: {e}"))?;

        // --- unet ----------------------------------------------------------
        let ucfg = UNetConfig::sdxl_base();
        let udir = r.join("unet");
        let utensors = crate::import::load(udir.to_str().ok_or("sdxl: non-UTF8 unet path")?, &ucfg)?;
        let unet = Unet::new(gpu.share(), ucfg.clone(), &utensors, lh, lw, CONTEXT as u32, false);

        // --- vae config only; the decoder is built at decode time -----------
        let vae_cfg = VaeConfig::from_json(&read_json(&r.join("vae/config.json"))?);
        let _ = (lh, lw);

        Ok(Sdxl { gpu, root: r.to_path_buf(), tok_l, tok_g, unet, vae_cfg, ucfg, hw: (h, w) })
    }

    /// SDXL's conditioning for one prompt: `(prompt_embeds[77*2048], pooled[1280])`.
    ///
    /// Both towers take the PENULTIMATE hidden state; bigG additionally supplies
    /// the projected pooled vector.
    fn tower(&self, sub: &str, cfg: &ClipTextConfig) -> Result<ClipText, String> {
        let t = clip::import::read_text_encoder(&self.root.join(sub))?;
        let init = clip::import::import_text(t, cfg)?;
        let map: std::collections::HashMap<String, Vec<f32>> =
            init.into_iter().map(|(k, (_, d))| (k, d)).collect();
        // `new_like`: a DIFFERENT kernel set on the SAME device. Each crate
        // resolves kernel indices against the list it registered, so building a
        // ClipText on a Gpu made from sdxlunet::KERNELS binds the wrong pipelines.
        Ok(ClipText::new_on(self.gpu.new_like(clip::model::TEXT_PIPELINES), cfg.clone(), 1, CONTEXT as u32, &map))
    }

    /// Encode every prompt in one pass, so the towers are built and dropped ONCE
    /// rather than once per prompt — the conditional and unconditional passes
    /// would otherwise pay for 3.3 GB of encoder twice.
    fn encode_all(&self, prompts: &[&str]) -> Result<Vec<Conditioning>, String> {
        let l_tower = self.tower("text_encoder", &ClipTextConfig::clip_l())?;
        let g_tower = self.tower("text_encoder_2", &ClipTextConfig::openclip_bigg())?;
        let out = prompts.iter().map(|p| self.encode_with(&l_tower, &g_tower, p)).collect();
        // Both towers drop here, before the UNet runs a single step.
        Ok(out)
    }

    fn encode_with(&self, l_tower: &ClipText, g_tower: &ClipText, prompt: &str) -> Conditioning {
        l_tower.set_tokens(&self.tok_l.encode_with_context(prompt, CONTEXT).ids);
        l_tower.forward();
        let l = l_tower.read_penultimate();

        g_tower.set_tokens(&self.tok_g.encode_with_context(prompt, CONTEXT).ids);
        g_tower.forward();
        let g = g_tower.read_penultimate();
        let pooled = g_tower.read_text_embeds().unwrap_or_else(|| g_tower.read_pooled());

        let (dl, dg) = (l.len() / CONTEXT, g.len() / CONTEXT);
        let mut embeds = Vec::with_capacity(CONTEXT * (dl + dg));
        for t in 0..CONTEXT {
            embeds.extend_from_slice(&l[t * dl..(t + 1) * dl]);
            embeds.extend_from_slice(&g[t * dg..(t + 1) * dg]);
        }
        (embeds, pooled)
    }

    /// Generate one image. Returns HWC RGB in `[0,1]`.
    pub fn generate(&mut self, prompt: &str, o: &GenerateOptions) -> Result<Vec<f32>, String> {
        let (h, w) = self.hw;
        let (lh, lw) = (h / 8, w / 8);
        let n = (self.ucfg.in_channels * lh * lw) as usize;

        let do_cfg = o.guidance > 1.0;
        let mut enc = if do_cfg {
            self.encode_all(&[prompt, o.negative.as_str()])?
        } else {
            self.encode_all(&[prompt])?
        };
        let uncond = do_cfg.then(|| enc.pop().expect("negative encoded"));
        let (cond, cond_pooled) = enc.pop().expect("prompt encoded");

        let mut sched = EulerScheduler::new(DiscreteConfig::sdxl());
        sched.set_timesteps(o.steps);

        // diffusers' micro-conditioning defaults: the generated size is both the
        // "original" and the "target", with no crop.
        let time_ids = vec![h as f32, w as f32, 0.0, 0.0, h as f32, w as f32];

        let mut lat = gaussian(n, o.seed);
        let s0 = sched.init_noise_sigma();
        for v in &mut lat {
            *v *= s0;
        }

        let timesteps: Vec<f32> = sched.timesteps().to_vec();
        for (i, &t) in timesteps.iter().enumerate() {
            let scaled = sched.scale_model_input(&lat);
            let c = self.unet.run(&scaled, t, &cond, &cond_pooled, &time_ids);
            let eps = match &uncond {
                None => c,
                Some((ue, up)) => {
                    let u = self.unet.run(&scaled, t, ue, up, &time_ids);
                    // guided = uncond + g * (cond - uncond)
                    u.iter().zip(&c).map(|(a, b)| a + o.guidance * (b - a)).collect()
                }
            };
            lat = sched.step(&eps, &lat);
            if i % 5 == 0 || i + 1 == timesteps.len() {
                eprintln!("  step {}/{}  sigma {:.4}", i + 1, timesteps.len(), sched.sigmas()[i]);
            }
        }

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

/// Box–Muller normal samples from the workspace's deterministic LCG, so a seed
/// reproduces a picture.
fn gaussian(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = data::rng::Rng::new(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = (rng.next_f32().abs()).max(1e-7);
        let u2 = rng.next_f32().abs();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = std::f32::consts::TAU * u2;
        out.push(r * th.cos());
        if out.len() < n {
            out.push(r * th.sin());
        }
    }
    out
}
