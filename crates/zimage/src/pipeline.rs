// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end Z-Image **text-to-image**: the individually-validated components
//! (Qwen-4B encoder, the S³-DiT, the FLUX VAE, the flow-match scheduler) wired
//! into one `generate`, following the diffusers `ZImagePipeline.__call__` recipe:
//!
//!   1. chat-template + tokenize the prompt, Qwen-4B → `hidden_states[-2]` (the
//!      caption features the DiT conditions on);
//!   2. a seeded Gaussian latent `[16, 1, H/8, W/8]`;
//!   3. an 8-step flow-match Euler sampler over the DiT — each step the DiT
//!      predicts velocity at `t = (1000 - t_sched)/1000`, the scheduler advances
//!      `x += (σ_next − σ)·v`; dynamic-shifted sigmas (mu from the sequence length);
//!   4. VAE decode of `latent/scaling + shift` → RGB, `[-1,1] → [0,1]`.
//!
//! Heavy compute stays on the GPU: the encoder runs INFERENCE-ONLY (Frozen, ~16 GB
//! weights — no train buffers) then drops; the DiT samples in int8 (13 GB); the VAE
//! decodes on-device. Peak VRAM is one model at a time — all under a 24 GB P40.
//! Models load sequentially and drop before the next, so peak VRAM is one model.

use std::collections::HashMap;

use data::qwen_tokenizer::QwenBpe;
use data::Tokenizer;
use diffusion::{default_z_image_sigmas, FlowMatchConfig, FlowMatchEulerScheduler};
use qwen::{Qwen, QwenConfig};
use vae::{VaeConfig, VaeDecoder, VaeEncoder};

use crate::import::import_comfy;
use crate::{ZImageConfig, ZImageDitI8, ZImageDitShard};

/// Filesystem locations of the four Z-Image components (never hard-coded — from
/// the environment, mirroring the crate's tests).
pub struct Paths {
    pub dit: String,
    pub vae: String,
    pub qwen: String,
    pub tokenizer: String,
}

impl Paths {
    pub fn from_env() -> Result<Paths, String> {
        let g = |k: &str| std::env::var(k).map_err(|_| format!("set {k} to the Z-Image {k} path"));
        Ok(Paths { dit: g("BRAIN_ZIMAGE_DIT")?, vae: g("BRAIN_ZIMAGE_VAE")?, qwen: g("BRAIN_ZIMAGE_QWEN")?, tokenizer: g("BRAIN_ZIMAGE_TOKENIZER")? })
    }
}

/// Generation options.
pub struct Opts {
    pub steps: u32,
    pub guidance: f32,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    /// High-fidelity DiT: `false` = int8 on one P40 (~13 GB, cosine 0.99, fast);
    /// `true` = full-precision fp32 sharded across both P40s (higher fidelity, no
    /// quantisation error, needs 2 GPUs).
    pub hifi: bool,
}

/// The denoiser backend chosen by [`Opts::hifi`]. Both expose the same
/// `forward(latent, cap, t)`; the sampler is identical either way.
enum DitEngine {
    I8(ZImageDitI8),
    Shard(ZImageDitShard),
}

impl DitEngine {
    fn build(hifi: bool, cfg: ZImageConfig, weights: crate::block::Tensors, lh: u32, lw: u32, cap_len: u32) -> DitEngine {
        if hifi {
            DitEngine::Shard(ZImageDitShard::build(cfg, weights, 1, lh, lw, cap_len))
        } else {
            DitEngine::I8(ZImageDitI8::build(cfg, weights, 1, lh, lw, cap_len))
        }
    }
    fn forward(&self, latent: &[f32], cap: &[f32], t: f32) -> Vec<f32> {
        match self {
            DitEngine::I8(d) => d.forward(latent, cap, t),
            DitEngine::Shard(d) => d.forward(latent, cap, t),
        }
    }
}

/// A **resident** text-to-image pipeline: the Qwen-4B encoder (CPU), the DiT
/// (int8 on one P40, or fp32 sharded across both), and the VAE decoder are built
/// ONCE for a fixed output size and caption length, then reused across many
/// generations — no ~20 GB reload per image. Each model keeps its own device
/// handle from build time, so [`generate`](Self::generate) just runs forwards.
/// Captions are padded/truncated to `cap_len` so the built graphs stay valid for
/// any prompt (padding repeats the last token, keeping features in-distribution).
pub struct HotPipeline {
    tok: QwenBpe,
    enc: Qwen,
    dit: DitEngine,
    vae: VaeDecoder,
    cap_len: u32,
    lh: u32,
    lw: u32,
    width: u32,
    height: u32,
    hifi: bool,
}

impl HotPipeline {
    pub fn cap_len(&self) -> u32 {
        self.cap_len
    }
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    pub fn hifi(&self) -> bool {
        self.hifi
    }

    /// Build the resident models for `width×height`, `cap_len` caption tokens, and
    /// the chosen precision. This is the slow one-time step (~weights load + int8
    /// quantise / shard build); `generate` afterwards is fast. `progress(msg)`
    /// streams the build stages.
    pub fn build(paths: &Paths, width: u32, height: u32, cap_len: u32, hifi: bool, mut progress: impl FnMut(&str)) -> Result<HotPipeline, String> {
        if width % 16 != 0 || height % 16 != 0 {
            return Err("width/height must be multiples of 16".into());
        }
        let (lh, lw) = (height / 8, width / 8);

        progress("loading tokenizer");
        let tok = QwenBpe::from_file(&paths.tokenizer)?;

        // Where the Qwen-4B encoder runs. `BRAIN_ZIMAGE_ENCODER_GPU=<i>` puts its
        // ~16.8 GB f32 weights on GPU `i` (e.g. the second card while the int8 DiT
        // owns the first) — the encode then runs in ~1-2 s on-device instead of
        // ~38 s on the CPU. Card-agnostic: you choose the index; unset ⇒ CPU. When
        // `hifi` already shards the fp32 DiT across both cards, there is no spare
        // GPU, so the encoder stays on the CPU regardless.
        let qcfg = QwenConfig::qwen3_4b();
        let qtensors = checkpoint::safetensors::read(&paths.qwen).map_err(|e| format!("read qwen: {e}"))?;
        let qinit = qwen::import::brain_init_from_hf(qtensors, &qcfg)?;
        let enc_gpu = std::env::var("BRAIN_ZIMAGE_ENCODER_GPU").ok().filter(|s| !s.is_empty());
        let dit_gpu = std::env::var("BRAIN_GPU_INDEX").ok(); // restore for the DiT/VAE (GPU 0)
        let build_enc = || Qwen::new_shard(qcfg.clone(), 1, cap_len, &qinit, false, qwen::Shard::whole(qcfg.n_layers as usize));
        let enc = match (&enc_gpu, hifi) {
            (Some(g), false) => {
                progress(&format!("building Qwen-4B encoder (GPU {g})"));
                std::env::set_var("BRAIN_GPU_INDEX", g);
                gpu_core::set_default_backend(gpu_core::Backend::Wgpu);
                let e = build_enc(); // Qwen's Gpu handle captures GPU `g`
                match &dit_gpu {
                    Some(d) => std::env::set_var("BRAIN_GPU_INDEX", d),
                    None => std::env::remove_var("BRAIN_GPU_INDEX"),
                }
                e
            }
            _ => {
                progress("building Qwen-4B encoder (CPU/AVX2)");
                gpu_core::set_default_backend(gpu_core::Backend::Cpu);
                let e = build_enc();
                gpu_core::set_default_backend(gpu_core::Backend::Wgpu);
                e
            }
        };
        drop(qinit);

        progress(if hifi { "building DiT (fp32, 2×GPU)" } else { "building DiT (int8, GPU)" });
        let zcfg = ZImageConfig::turbo();
        let weights = import_comfy(checkpoint::safetensors::read(&paths.dit).map_err(|e| format!("read dit: {e}"))?, &zcfg);
        let dit = DitEngine::build(hifi, zcfg, weights, lh, lw, cap_len);

        progress("building VAE decoder (GPU)");
        let vtensors = tensors_map(checkpoint::safetensors::read(&paths.vae).map_err(|e| format!("read vae: {e}"))?);
        let vae = VaeDecoder::from_diffusers(zimage_vae_config(), &vtensors, lh, lw, Some("gpu"));

        Ok(HotPipeline { tok, enc, dit, vae, cap_len, lh, lw, width, height, hifi })
    }

    /// Tokenize `prompt`, pad/truncate to `cap_len`, and run encode → DiT sampling
    /// → VAE decode — all on the resident models. Fast (no weight loads).
    pub fn generate(&self, prompt: &str, seed: u64, steps: u32, mut progress: impl FnMut(u32, u32, &str)) -> Image {
        let steps = steps.max(1);
        let total = steps + 2;

        // 1. tokenize + pad/truncate to the built cap_len.
        progress(1, total, "encoding prompt (Qwen-4B, CPU)");
        let templated = self.tok.apply_chat_template(&[("user", prompt)], true);
        let mut tokens = self.tok.encode(&templated);
        let cl = self.cap_len as usize;
        if tokens.len() > cl {
            tokens.truncate(cl);
        } else if tokens.len() < cl {
            let pad = *tokens.last().unwrap_or(&0);
            tokens.resize(cl, pad);
        }
        let cap = self.enc.encode(&tokens); // [cap_len · 2560]

        // 2. seeded latent + scheduler.
        let n = (16 * self.lh * self.lw) as usize;
        let mut lat = randn(n, seed);
        let seq_len = ((self.lh / 2) * (self.lw / 2)) as usize;
        let sigmas = dynamic_shift(&default_z_image_sigmas(steps as usize), calc_mu(seq_len));
        let mut sched = FlowMatchEulerScheduler::new(FlowMatchConfig { num_train_timesteps: 1000, shift: 1.0 });
        sched.set_timesteps(&sigmas);
        let ts = sched.timesteps().to_vec();
        let sig_full = sched.sigmas().to_vec();

        // 3. flow-match sampling on the resident DiT.
        for i in 0..steps as usize {
            progress(2 + i as u32, total, if self.hifi { "sampling (fp32, 2×GPU)" } else { "sampling" });
            let t_dit = (1000.0 - ts[i]) / 1000.0;
            let v: Vec<f32> = self.dit.forward(&lat, &cap, t_dit).iter().map(|&x| -x).collect();
            let dt = sig_full[i + 1] - sig_full[i];
            for (x, &vv) in lat.iter_mut().zip(&v) {
                *x += dt * vv;
            }
        }

        // 4. VAE decode + postprocess.
        progress(total, total, "decoding (VAE)");
        let dec_in: Vec<f32> = lat.iter().map(|&x| x / VAE_SCALE + VAE_SHIFT).collect();
        let chw = self.vae.decode(&dec_in);
        let (h, w) = (self.height as usize, self.width as usize);
        let mut hwc = vec![0f32; h * w * 3];
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    hwc[(y * w + x) * 3 + c] = (chw[(c * h + y) * w + x] * 0.5 + 0.5).clamp(0.0, 1.0);
                }
            }
        }
        Image { hwc, w, h }
    }
}

/// A generated image: interleaved-RGB HWC in `[0,1]`.
pub struct Image {
    pub hwc: Vec<f32>,
    pub w: usize,
    pub h: usize,
}

/// An input image (+ optional mask) to condition generation on — the shared
/// substrate of image2image, inpaint and outpaint. The image is VAE-encoded to a
/// latent, partially re-noised (per `strength`), and denoised the rest of the way.
pub struct Init<'a> {
    /// Source image, HWC interleaved RGB in `[0,1]`, size `opts.width×opts.height`.
    pub image: &'a [f32],
    /// `0` = keep the input unchanged, `1` = ignore it (full re-generation). Sets
    /// how far back into the noise schedule sampling starts.
    pub strength: f32,
    /// Optional inpaint mask, `height·width` single-channel in `[0,1]` (`1` =
    /// regenerate, `0` = keep). When present, kept regions are re-anchored to the
    /// (noised) input at every step so only the masked area changes.
    pub mask: Option<&'a [f32]>,
    /// Feather radius in **latent cells** (VAE 8× downscale). `0` = a hard mask
    /// edge; larger blurs the mask boundary so the regenerated region blends
    /// smoothly into the kept pixels instead of showing a seam.
    pub feather: u32,
}

/// Deterministic standard-normal samples via xorshift64* + Box–Muller — a fixed
/// seed always yields the same latent (so generation is reproducible).
fn randn(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        // to (0,1)
        ((s >> 11) as f64 / (1u64 << 53) as f64).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let (u1, u2) = (next(), next());
        let r = (-2.0 * u1.ln()).sqrt();
        out.push((r * (std::f64::consts::TAU * u2).cos()) as f32);
        if out.len() < n {
            out.push((r * (std::f64::consts::TAU * u2).sin()) as f32);
        }
    }
    out
}

/// diffusers `calculate_shift` for Z-Image (base_seq 256, max 4096, shifts 0.5..1.15).
fn calc_mu(seq_len: usize) -> f32 {
    let m = (1.15 - 0.5) / (4096.0 - 256.0);
    0.5 + m * (seq_len as f32 - 256.0)
}

/// diffusers exponential time-shift: `σ' = e^mu / (e^mu + 1/σ − 1)`.
fn dynamic_shift(sigmas: &[f32], mu: f32) -> Vec<f32> {
    let e = mu.exp();
    sigmas.iter().map(|&s| e / (e + 1.0 / s - 1.0)).collect()
}

fn tensors_map(v: Vec<checkpoint::safetensors::StTensor>) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    v.into_iter().map(|t| (t.name, (t.shape, t.data))).collect()
}

/// The FLUX-style 16-channel VAE config Z-Image ships (weights/vae/config.json).
fn zimage_vae_config() -> VaeConfig {
    VaeConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels: 16,
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        norm_num_groups: 32,
        norm_eps: 1e-6,
        mid_block_add_attention: true,
        scaling_factor: 0.3611,
        shift_factor: 0.1159,
    }
}

/// VAE scale/shift → DiT latent space. Decode is `z/scale + shift`; encode is the
/// inverse `(z − shift)·scale`. (Z-Image VAE: scale 0.3611, shift 0.1159.)
const VAE_SCALE: f32 = 0.3611;
const VAE_SHIFT: f32 = 0.1159;

/// Area-average-pool a full-res mask `[h·w]` down to latent resolution `[lh·lw]`
/// (VAE 8× downscale), keeping soft values in `[0,1]`.
fn downsample_mask(mask: &[f32], w: usize, h: usize, lw: usize, lh: usize) -> Vec<f32> {
    let (sx, sy) = (w / lw, h / lh);
    let mut out = vec![0f32; lw * lh];
    for ly in 0..lh {
        for lx in 0..lw {
            let mut s = 0.0;
            for yy in 0..sy {
                for xx in 0..sx {
                    s += mask[(ly * sy + yy) * w + (lx * sx + xx)];
                }
            }
            out[ly * lw + lx] = s / (sx * sy) as f32;
        }
    }
    out
}

/// Separable box blur of a latent-resolution mask `[lh·lw]`, `radius` cells each
/// side, clamped at the borders — feathers a hard mask into a smooth ramp so the
/// inpaint/outpaint boundary blends. `radius = 0` returns the mask unchanged.
fn feather_mask(mask: &[f32], lw: usize, lh: usize, radius: usize) -> Vec<f32> {
    if radius == 0 {
        return mask.to_vec();
    }
    let win = (2 * radius + 1) as f32;
    // horizontal
    let mut h = vec![0f32; lw * lh];
    for y in 0..lh {
        for x in 0..lw {
            let mut s = 0.0;
            for d in 0..=2 * radius {
                let xx = (x + d).saturating_sub(radius).min(lw - 1);
                s += mask[y * lw + xx];
            }
            h[y * lw + x] = s / win;
        }
    }
    // vertical
    let mut out = vec![0f32; lw * lh];
    for y in 0..lh {
        for x in 0..lw {
            let mut s = 0.0;
            for d in 0..=2 * radius {
                let yy = (y + d).saturating_sub(radius).min(lh - 1);
                s += h[yy * lw + x];
            }
            out[y * lw + x] = s / win;
        }
    }
    out
}

/// Generate an image from `prompt` (text-to-image). `progress(step, total, msg)`
/// streams updates.
pub fn generate(prompt: &str, opts: &Opts, paths: &Paths, progress: impl FnMut(u32, u32, &str)) -> Result<Image, String> {
    generate_core(prompt, opts, paths, None, progress)
}

/// Image-to-image / inpaint / outpaint: regenerate `init.image` toward `prompt`.
/// A mask (`init.mask`) restricts changes to the masked region (inpaint/outpaint).
pub fn generate_img(prompt: &str, opts: &Opts, paths: &Paths, init: Init, progress: impl FnMut(u32, u32, &str)) -> Result<Image, String> {
    generate_core(prompt, opts, paths, Some(init), progress)
}

/// Shared pipeline for all four actions. With `init = None` it starts from pure
/// noise (text2image); with an init image it VAE-encodes it, re-noises to the
/// `strength`-determined step, and (for inpaint) re-anchors the kept region each
/// step.
fn generate_core(prompt: &str, opts: &Opts, paths: &Paths, init: Option<Init>, mut progress: impl FnMut(u32, u32, &str)) -> Result<Image, String> {
    let total = opts.steps + 2; // encode + N sampling + decode
    if opts.width % 16 != 0 || opts.height % 16 != 0 {
        return Err("width/height must be multiples of 16".into());
    }
    let (lh, lw) = (opts.height / 8, opts.width / 8); // VAE downscale 8

    // 1. tokenize (chat template) --------------------------------------------
    let tok = QwenBpe::from_file(&paths.tokenizer)?;
    let templated = tok.apply_chat_template(&[("user", prompt)], true);
    let tokens = tok.encode(&templated);
    let cap_len = tokens.len() as u32;

    // 2. Qwen-4B encode → caption features (penultimate hidden). Dropped after. -
    //
    // The encoder is a SINGLE forward pass (not the heavy iterative compute), and
    // its ~16.8 GB of f32 weights plus Vulkan's upload staging bump a 24 GB P40's
    // ceiling. So we run just this one-shot on the CPU (AVX2+FMA matmul, a few
    // seconds) and keep the *heavy* work — the 8-step DiT and the VAE — on the GPU
    // (VRAM), which is where the repeated compute belongs. This is the intended
    // "CPU only as a fallback for the piece that doesn't fit" split.
    progress(1, total, "encoding prompt (Qwen-4B, CPU/AVX2)");
    let qcfg = QwenConfig::qwen3_4b();
    let qtensors = checkpoint::safetensors::read(&paths.qwen).map_err(|e| format!("read qwen: {e}"))?;
    let qinit = qwen::import::brain_init_from_hf(qtensors, &qcfg)?;
    let cap = {
        gpu_core::set_default_backend(gpu_core::Backend::Cpu); // encoder → CPU (AVX2)
        let enc = Qwen::new_shard(qcfg.clone(), 1, cap_len, &qinit, false, qwen::Shard::whole(qcfg.n_layers as usize));
        let c = enc.encode(&tokens); // [cap_len · 2560]
        gpu_core::set_default_backend(gpu_core::Backend::Wgpu); // heavy compute → GPU
        c
    };
    drop(qinit);

    // 3. scheduler (dynamic-shifted sigmas; brain applies shift=1 so we pre-shift).
    // `sig_full` is the N+1 sigmas (N shifted step sigmas + terminal 0).
    let seq_len = ((lh / 2) * (lw / 2)) as usize; // DiT patch 2
    let sigmas = dynamic_shift(&default_z_image_sigmas(opts.steps as usize), calc_mu(seq_len));
    let mut sched = FlowMatchEulerScheduler::new(FlowMatchConfig { num_train_timesteps: 1000, shift: 1.0 });
    sched.set_timesteps(&sigmas);
    let ts = sched.timesteps().to_vec();
    let sig_full = sched.sigmas().to_vec();

    // 4. starting latent -----------------------------------------------------
    // Fixed seeded noise. text2image starts from it directly (σ≈1). An init image
    // is VAE-encoded to a latent `lat0`, then re-noised to the strength-chosen
    // step: `x = (1−σ)·lat0 + σ·noise` (flow-matching forward). A mask + `lat0`
    // are kept for per-step re-anchoring of the un-masked region (inpaint).
    let n = (16 * lh * lw) as usize;
    let noise = randn(n, opts.seed);
    let plane = (lh * lw) as usize;
    let (mut lat, start_step, mask_lat, lat0): (Vec<f32>, usize, Option<Vec<f32>>, Option<Vec<f32>>) = match &init {
        None => (noise.clone(), 0, None, None),
        Some(init) => {
            progress(1, total, "encoding image (VAE)");
            let (h, w) = (opts.height as usize, opts.width as usize);
            if init.image.len() != 3 * h * w {
                return Err(format!("init image is {} floats, expected {} (HWC {w}×{h}×3)", init.image.len(), 3 * h * w));
            }
            // HWC [0,1] → CHW [-1,1] (VAE-native).
            let mut chw_in = vec![0f32; 3 * h * w];
            for c in 0..3 {
                for y in 0..h {
                    for x in 0..w {
                        chw_in[(c * h + y) * w + x] = init.image[(y * w + x) * 3 + c] * 2.0 - 1.0;
                    }
                }
            }
            let vtensors = tensors_map(checkpoint::safetensors::read(&paths.vae).map_err(|e| format!("read vae: {e}"))?);
            let mean = {
                let enc = VaeEncoder::from_diffusers(zimage_vae_config(), &vtensors, opts.height, opts.width, Some("gpu"));
                enc.encode_mean(&chw_in, lh, lw)
            };
            let lat0: Vec<f32> = mean.iter().map(|&z| (z - VAE_SHIFT) * VAE_SCALE).collect();

            let strength = init.strength.clamp(0.0, 1.0);
            let init_t = ((opts.steps as f32 * strength).round() as usize).min(opts.steps as usize);
            let start = (opts.steps as usize).saturating_sub(init_t);
            let sig = sig_full[start];
            let lat_init: Vec<f32> = lat0.iter().zip(&noise).map(|(&x0, &nz)| (1.0 - sig) * x0 + sig * nz).collect();

            let mask_lat = init.mask.map(|m| {
                let ds = downsample_mask(m, w, h, lw as usize, lh as usize);
                feather_mask(&ds, lw as usize, lh as usize, init.feather as usize)
            });
            (lat_init, start, mask_lat, Some(lat0))
        }
    };

    // 5. flow-match sampling over the DiT ------------------------------------
    // int8 on one P40 (default), or full-precision fp32 sharded across both P40s
    // when `hifi` — no quantisation error, at the cost of a second card.
    let zcfg = ZImageConfig::turbo();
    let weights = import_comfy(checkpoint::safetensors::read(&paths.dit).map_err(|e| format!("read dit: {e}"))?, &zcfg);
    {
        let dit = if opts.hifi {
            DitEngine::Shard(ZImageDitShard::build(zcfg, weights, 1, lh, lw, cap_len))
        } else {
            DitEngine::I8(ZImageDitI8::build(zcfg, weights, 1, lh, lw, cap_len))
        };
        for i in start_step..opts.steps as usize {
            progress(2 + i as u32, total, if opts.hifi { "sampling (fp32, 2×GPU)" } else { "sampling" });
            let t_dit = (1000.0 - ts[i]) / 1000.0;
            // The reference negates the DiT output before the Euler step
            // (`noise_pred = -noise_pred; scheduler.step(noise_pred, …)`): brain's
            // scheduler is the bare `x + (σ_next−σ)·v`, so we negate here to match.
            let v: Vec<f32> = dit.forward(&lat, &cap, t_dit).iter().map(|&x| -x).collect();
            let dt = sig_full[i + 1] - sig_full[i];
            for (x, &vv) in lat.iter_mut().zip(&v) {
                *x += dt * vv;
            }
            // Inpaint: re-anchor the KEPT region to the input latent noised to the
            // next step's σ, so only the masked region is freely regenerated.
            if let (Some(mask), Some(lat0)) = (&mask_lat, &lat0) {
                let snext = sig_full[i + 1];
                for c in 0..16 {
                    for p in 0..plane {
                        let idx = c * plane + p;
                        let keep = 1.0 - mask[p];
                        let orig = (1.0 - snext) * lat0[idx] + snext * noise[idx];
                        lat[idx] = mask[p] * lat[idx] + keep * orig;
                    }
                }
            }
        }
    } // dit dropped → free VRAM before the VAE

    // 6. VAE decode ----------------------------------------------------------
    //
    // On the GPU (VRAM): the decoder graph is built over the *latent* dims
    // (`lh × lw`); it upsamples ×8 internally to the `height × wpx` image. Passing
    // the latent dims keeps every buffer small (well under the P40's 2 GiB binding
    // limit), so this runs on-device alongside the DiT.
    progress(total, total, "decoding (VAE)");
    let vtensors = tensors_map(checkpoint::safetensors::read(&paths.vae).map_err(|e| format!("read vae: {e}"))?);
    let vae = VaeDecoder::from_diffusers(zimage_vae_config(), &vtensors, lh, lw, Some("gpu"));
    let dec_in: Vec<f32> = lat.iter().map(|&x| x / VAE_SCALE + VAE_SHIFT).collect();
    let chw = vae.decode(&dec_in); // [3 · H · W] in [-1, 1]

    // 7. postprocess: [-1,1] → [0,1], CHW → HWC ------------------------------
    let (h, w) = (opts.height as usize, opts.width as usize);
    let mut hwc = vec![0f32; h * w * 3];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                hwc[(y * w + x) * 3 + c] = (chw[(c * h + y) * w + x] * 0.5 + 0.5).clamp(0.0, 1.0);
            }
        }
    }
    Ok(Image { hwc, w, h })
}
