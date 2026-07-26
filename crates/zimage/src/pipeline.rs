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
use vae::{VaeConfig, VaeDecoder};

use crate::import::import_comfy;
use crate::{ZImageConfig, ZImageDitI8};

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
}

/// A generated image: interleaved-RGB HWC in `[0,1]`.
pub struct Image {
    pub hwc: Vec<f32>,
    pub w: usize,
    pub h: usize,
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

/// Generate an image from `prompt`. `progress(step, total, msg)` streams updates.
pub fn generate(prompt: &str, opts: &Opts, paths: &Paths, mut progress: impl FnMut(u32, u32, &str)) -> Result<Image, String> {
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
    progress(1, total, "encoding prompt (Qwen-4B)");
    let qcfg = QwenConfig::qwen3_4b();
    let qtensors = checkpoint::safetensors::read(&paths.qwen).map_err(|e| format!("read qwen: {e}"))?;
    let qinit = qwen::import::brain_init_from_hf(qtensors, &qcfg)?;
    let cap = {
        let enc = Qwen::new_shard(qcfg.clone(), 1, cap_len, &qinit, false, qwen::Shard::whole(qcfg.n_layers as usize));
        enc.encode(&tokens) // [cap_len · 2560]
    };
    drop(qinit);

    // 3. seeded latent -------------------------------------------------------
    let mut lat = randn((16 * lh * lw) as usize, opts.seed);

    // 4. scheduler (dynamic-shifted sigmas; brain applies shift=1 so we pre-shift)
    let seq_len = ((lh / 2) * (lw / 2)) as usize; // DiT patch 2
    let sigmas = dynamic_shift(&default_z_image_sigmas(opts.steps as usize), calc_mu(seq_len));
    let mut sched = FlowMatchEulerScheduler::new(FlowMatchConfig { num_train_timesteps: 1000, shift: 1.0 });
    sched.set_timesteps(&sigmas);
    let ts = sched.timesteps().to_vec();

    // 5. flow-match sampling over the DiT (int8) -----------------------------
    let zcfg = ZImageConfig::turbo();
    let weights = import_comfy(checkpoint::safetensors::read(&paths.dit).map_err(|e| format!("read dit: {e}"))?, &zcfg);
    {
        let dit = ZImageDitI8::build(zcfg, weights, 1, lh, lw, cap_len);
        for i in 0..opts.steps as usize {
            progress(2 + i as u32, total, "sampling");
            let t_dit = (1000.0 - ts[i]) / 1000.0;
            let v = dit.forward(&lat, &cap, t_dit);
            lat = sched.step(&v, &lat);
        }
    } // dit dropped → free VRAM before the VAE

    // 6. VAE decode ----------------------------------------------------------
    progress(total, total, "decoding (VAE)");
    let vtensors = tensors_map(checkpoint::safetensors::read(&paths.vae).map_err(|e| format!("read vae: {e}"))?);
    let vae = VaeDecoder::from_diffusers(zimage_vae_config(), &vtensors, opts.height, opts.width, Some("gpu"));
    let dec_in: Vec<f32> = lat.iter().map(|&x| x / 0.3611 + 0.1159).collect();
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
