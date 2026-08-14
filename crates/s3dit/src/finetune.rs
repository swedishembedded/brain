// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end LoRA fine-tuning for the real Z-Image DiT: turn a folder of
//! captioned images ([`crate::dataset`]) into a trained [`crate::lora::LoraAdapter`].
//!
//! Pipeline per run:
//!   1. VAE-encode each image to a latent, Qwen-encode each caption to features
//!      (both once, up front; encoders then dropped to free VRAM).
//!   2. Flow-matching loop: each step draws a σ, builds `x_σ = (1-σ)·x₁ + σ·x₀`
//!      with target velocity `x₁ - x₀`, forwards the adapter-applied frozen base
//!      through the streaming device trainer to get `dL/dW_eff`, and projects that
//!      into an Adam step on the low-rank `A,B` (see [`crate::lora`]).
//!   3. Save the adapter.
//!
//! The flow-matching convention matches inference exactly (verified against the
//! Euler integrator in [`crate::pipeline`]): the DiT's raw output is the
//! noise→clean velocity `x₁ - x₀`, its time input is `1 - σ`, and the loss is
//! velocity-MSE in patch space. Getting this consistent is what makes the trained
//! adapter usable by the unchanged generation path.

use std::path::Path;

use crate::lora::{LoraAdapter, LoraCfg};
use crate::modelgrad::{patchify, Cfg};
use crate::model::ZImageConfig;
use crate::pipeline::Paths;
use crate::train::Batch;
use data::qwen_tokenizer::QwenBpe;
use data::Tokenizer;
use dit::rope::{tables_for_ids, RopeConfig};
use qwen3::{Qwen, QwenConfig, Shard};
use vae::VaeEncoder;

/// Latent-space VAE scale/shift (FLUX VAE; must match [`crate::pipeline`]).
pub const VAE_SCALE: f32 = 0.3611;
pub const VAE_SHIFT: f32 = 0.1159;

/// Build the training [`Cfg`] for a real Z-Image checkpoint at latent size
/// `h×w` (latent pixels = image/8) and `cap_len` caption tokens.
/// One named tensor: `(name, shape, row-major f32 data)`.
pub type NamedTensor = (String, Vec<usize>, Vec<f32>);

pub fn train_cfg(z: &ZImageConfig, h: u32, w: u32, cap_len: u32) -> Cfg {
    Cfg {
        dim: z.dim as usize,
        nh: z.n_heads as usize,
        n_layers: z.n_layers as usize,
        n_refiner: z.n_refiner_layers as usize,
        cap_feat_dim: z.cap_feat_dim as usize,
        in_channels: z.in_channels as usize,
        patch: z.patch_size as usize,
        h: h as usize,
        w: w as usize,
        ncap: cap_len as usize,
        t_scale: 1000.0,
    }
}

/// A dataset sample after encoding: the clean DiT-space latent `x₁`
/// (`[in_channels·h·w]`) and caption features (`[ncap·cap_feat_dim]`), host f64.
#[derive(Clone)]
pub struct Encoded {
    pub latent: Vec<f64>,
    pub cap: Vec<f64>,
}

/// Convert a VAE-mean latent (`[in_channels·h·w]`) to DiT space: `(mean-shift)·scale`
/// — the same transform [`crate::pipeline`] inverts before decoding.
pub fn latent_to_dit(mean: &[f32]) -> Vec<f64> {
    mean.iter().map(|&z| ((z - VAE_SHIFT) * VAE_SCALE) as f64).collect()
}

/// Build one flow-matching training [`Batch`] from a clean latent + caption.
///
/// `sigma ∈ (0,1]` is the noise level (1 = pure noise, 0 = clean); `noise` is a
/// standard-normal sample the length of `latent`. The RoPE `rope` must carry the
/// Z-Image axes (`cfg.rope()`), and `cfg`'s `h/w/patch/in_channels/ncap` must match
/// `latent`/`cap`. Panics on a length mismatch (a wiring bug, not user input).
pub fn make_flow_batch(cfg: &Cfg, rope: &RopeConfig, latent: &[f64], cap: &[f64], sigma: f64, noise: &[f64]) -> Batch {
    assert_eq!(latent.len(), cfg.in_channels * cfg.h * cfg.w, "latent size");
    assert_eq!(noise.len(), latent.len(), "noise size");
    assert_eq!(cap.len(), cfg.ncap * cfg.cap_feat_dim, "caption size");

    // x_σ = (1-σ)·x₁ + σ·x₀ ; target velocity v = x₁ - x₀ (raw DiT-output convention).
    let x_t: Vec<f64> = latent.iter().zip(noise).map(|(&x1, &x0)| (1.0 - sigma) * x1 + sigma * x0).collect();
    let v: Vec<f64> = latent.iter().zip(noise).map(|(&x1, &x0)| x1 - x0).collect();
    let target = patchify(&v, cfg); // loss is in patch space on the image tokens

    // RoPE ids: caption token i → (1+i,0,0); image patch (f=0,hi,wi) → (cap_len+1,hi,wi).
    let (ht, wt) = ((cfg.h / cfg.patch) as u32, (cfg.w / cfg.patch) as u32);
    let cap_len = cfg.ncap as u32;
    let mut img_ids = Vec::with_capacity((ht * wt) as usize * 3);
    for hi in 0..ht {
        for wi in 0..wt {
            img_ids.extend_from_slice(&[cap_len + 1, hi, wi]);
        }
    }
    let mut cap_ids = Vec::with_capacity(cfg.ncap * 3);
    for i in 0..cap_len {
        cap_ids.extend_from_slice(&[1 + i, 0, 0]);
    }
    let img = tables_for_ids(rope, &img_ids, 3);
    let cap_rope = tables_for_ids(rope, &cap_ids, 3);
    let f = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| x as f64).collect() };

    Batch {
        latent: x_t,
        cap: cap.to_vec(),
        t: 1.0 - sigma, // model time input
        img_cos: f(&img.cos),
        img_sin: f(&img.sin),
        cap_cos: f(&cap_rope.cos),
        cap_sin: f(&cap_rope.sin),
        target,
    }
}

/// Tokenize `prompt` with the chat template and pad/truncate to `cap_len` (the
/// exact caption tokenization [`crate::pipeline`] uses at generation time).
fn tokenize_pad(tok: &QwenBpe, prompt: &str, cap_len: usize) -> Vec<u32> {
    let templated = tok.apply_chat_template(&[("user", prompt)], true);
    let mut tokens = tok.encode(&templated);
    if tokens.len() > cap_len {
        tokens.truncate(cap_len);
    } else if tokens.len() < cap_len {
        let pad = *tokens.last().unwrap_or(&0);
        tokens.resize(cap_len, pad);
    }
    tokens
}

/// Encode every dataset sample once: caption → Qwen features, image → DiT-space
/// latent. Both encoders are built, used, and **dropped** before the caller builds
/// the trainer, so their VRAM is reclaimed (sequential residency). `enc_gpu` is the
/// card for the int8 Qwen encoder (fast, ~2 s/caption); `size` is the square image
/// size (latent = size/8). `progress(done, total, stage)` streams per-item progress.
/// `cancel` is polled per item so a cancelled job aborts during this phase too.
#[allow(clippy::too_many_arguments)]
pub fn encode_samples(
    paths: &Paths,
    samples: &[crate::dataset::Sample],
    size: u32,
    cap_len: u32,
    enc_gpu: &str,
    cancel: &capability::CancelToken,
    mut progress: impl FnMut(usize, usize, &str),
) -> Result<Vec<Encoded>, String> {
    let n = samples.len();
    let tok = QwenBpe::from_file(&paths.tokenizer)?;

    // --- captions → Qwen-4B features (int8 encoder on `enc_gpu`) ---
    let caps: Vec<Vec<f64>> = {
        let qcfg = QwenConfig::qwen3_4b();
        let qtensors = checkpoint::safetensors::read(&paths.qwen).map_err(|e| format!("read qwen: {e}"))?;
        let qinit = qwen3::import::brain_init_from_hf(qtensors, &qcfg)?;
        gpu_core::set_default_backend(gpu_core::Backend::Wgpu);
        let nl = qcfg.n_layers as usize;
        // `enc_gpu` is user input; the canonical index in the shard is what
        // places the build (device registry), not env mutation.
        let bi = enc_gpu.parse().unwrap_or(1);
        let enc = Qwen::new_shard_i8(qcfg, 1, cap_len, &qinit, Shard { start: 0, end: nl - 1, embed: true, head: false, gpu_index: bi });
        let mut out = Vec::with_capacity(n);
        for (i, s) in samples.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err("cancelled".into());
            }
            progress(i, n, "encoding captions (Qwen-4B int8)");
            let tokens = tokenize_pad(&tok, &s.prompt, cap_len as usize);
            out.push(enc.encode(&tokens).iter().map(|&x| x as f64).collect());
        }
        out
    }; // enc + qinit dropped here → VRAM/RAM reclaimed

    // --- images → DiT-space latents (VAE encoder) ---
    let (lh, lw) = (size / 8, size / 8);
    let vtensors = crate::pipeline::tensors_map(checkpoint::safetensors::read(&paths.vae).map_err(|e| format!("read vae: {e}"))?);
    let venc = VaeEncoder::from_diffusers(crate::pipeline::zimage_vae_config(), &vtensors, size, size, Some("gpu"));
    let mut encoded = Vec::with_capacity(n);
    for (i, (s, cap)) in samples.iter().zip(caps).enumerate() {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        progress(i, n, "encoding images (VAE)");
        // HWC [0,1] → CHW [-1,1] (the VAE encoder's expected input range).
        let (h, w) = (size as usize, size as usize);
        let mut chw = vec![0f32; 3 * h * w];
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    chw[(c * h + y) * w + x] = s.hwc[(y * w + x) * 3 + c] * 2.0 - 1.0;
                }
            }
        }
        let mean = venc.encode_mean(&chw, lh, lw);
        encoded.push(Encoded { latent: latent_to_dit(&mean), cap });
    }
    Ok(encoded)
}

/// Save a LoRA adapter (its `to_tensors` output + rank/alpha) to brain's
/// checkpoint format. Reloadable by [`load_adapter_folded`] and inspectable via
/// `checkpoint::load`.
pub fn save_adapter(path: &str, tensors: &[(String, Vec<usize>, Vec<f32>)], rank: usize, alpha: f32) {
    let t: Vec<(String, Vec<u64>, Vec<f32>)> =
        tensors.iter().map(|(n, s, d)| (n.clone(), s.iter().map(|&x| x as u64).collect(), d.clone())).collect();
    checkpoint::save(path, serde_json::json!({"model": "z-image-lora", "rank": rank, "alpha": alpha}), &t);
}

/// Load an adapter saved by [`save_adapter`] and fold it into an inference tensor
/// map, so the generation path produces adapter-conditioned images unchanged.
pub fn load_adapter_folded(path: &str, cfg: &Cfg, tensors: &mut crate::block::Tensors) -> Result<(), String> {
    let c = checkpoint::load(path);
    let rank = c.header["config"]["rank"].as_u64().ok_or("adapter: missing rank in header")? as usize;
    let alpha = c.header["config"]["alpha"].as_f64().unwrap_or(rank as f64) as f32;
    let map: std::collections::HashMap<String, (Vec<usize>, Vec<f32>)> =
        c.tensors.into_iter().map(|t| (t.name, (Vec::new(), t.data))).collect();
    let ad = LoraAdapter::from_tensors(cfg, LoraCfg { rank, alpha, seed: 0 }, &map)?;
    ad.fold_into_comfy(tensors)
}

/// LoRA fine-tuning hyper-parameters.
pub struct TrainOpts {
    pub steps: u32,
    pub rank: usize,
    pub lr: f32,
    pub size: u32,
    pub cap_len: u32,
    pub seed: u64,
    /// Split the DiT across both GPUs (needed for the real 6B fp32 fwd+bwd).
    pub two_gpu: bool,
    /// Where to write the adapter (final + every `ckpt_every` steps, 0 = only final).
    pub save_path: String,
    pub ckpt_every: u32,
}

/// Fine-tune a LoRA adapter on `dir` (a captioned-image folder). Returns the adapter
/// as `(name, shape, data)` tensors ready to save. `progress(step, total, msg)`
/// streams encoding + per-step loss so a long run is not a black box. `cancel` is
/// polled every step (a multi-hour job must be abortable): a cancelled token
/// returns `Err("cancelled")` — periodic checkpoints already written remain.
pub fn run(
    paths: &Paths,
    dir: &Path,
    opts: &TrainOpts,
    cancel: &capability::CancelToken,
    mut progress: impl FnMut(u32, u32, String),
) -> Result<Vec<NamedTensor>, String> {
    if !opts.size.is_multiple_of(16) {
        return Err("size must be a multiple of 16".into());
    }
    // 1. dataset
    let samples = crate::dataset::load_dir(dir, opts.size, |w| progress(0, opts.steps + 1, format!("dataset: {w}")))?;
    progress(0, opts.steps + 1, format!("loaded {} images from {}", samples.len(), dir.display()));

    // 2. encode (encoders dropped before the trainer is built)
    let enc_gpu = std::env::var("BRAIN_S3DIT_ENCODER_GPU").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "1".to_string());
    let n_samples = samples.len();
    let encoded = encode_samples(paths, &samples, opts.size, opts.cap_len, &enc_gpu, cancel, |i, tot, stage| {
        progress(0, opts.steps + 1, format!("{stage} {}/{tot}", i + 1))
    })?;
    drop(samples);

    // 3. base weights → training format
    progress(0, opts.steps + 1, "loading DiT weights".into());
    let zcfg = ZImageConfig::turbo();
    let tensors = crate::import::import_comfy(checkpoint::safetensors::read(&paths.dit).map_err(|e| format!("read dit: {e}"))?, &zcfg);
    let (lh, lw) = (opts.size / 8, opts.size / 8);
    let cfg = train_cfg(&zcfg, lh, lw, opts.cap_len);
    let base = crate::import::model_weights_from_comfy(&tensors, &zcfg)?;
    drop(tensors);
    let rope = zcfg.rope();

    // 4. trainer + adapter
    std::env::set_var("BRAIN_OFFLOAD_ADAM", "1");
    let trainer = TrainerKind::new(cfg, opts.two_gpu);
    let mut adapter = LoraAdapter::new(&cfg, LoraCfg::new(opts.rank));
    let mut rng = (opts.seed ^ 0xf17e_7c7e_a5ad_a97e).wrapping_mul(2654435761).max(1);

    // 5. flow-matching loop
    let alpha = adapter.alpha();
    for step in 0..opts.steps {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        let s = &encoded[step as usize % n_samples];
        let sigma = uniform01(&mut rng).clamp(1e-3, 1.0) as f64;
        let noise: Vec<f64> = (0..s.latent.len()).map(|_| gauss(&mut rng)).collect();
        let batch = make_flow_batch(&cfg, &rope, &s.latent, &s.cap, sigma, &noise);
        let w_eff = adapter.apply(&base);
        let (loss, grads) = trainer.grads(&w_eff, &batch);
        adapter.step(&grads, opts.lr);
        progress(step + 1, opts.steps + 1, format!("step {}/{}  loss {loss:.5}", step + 1, opts.steps));
        // periodic checkpoint so a long run is resumable / inspectable mid-flight.
        if opts.ckpt_every > 0 && (step + 1) % opts.ckpt_every == 0 && step + 1 < opts.steps {
            save_adapter(&opts.save_path, &adapter.to_tensors(), opts.rank, alpha);
        }
    }
    let tensors = adapter.to_tensors();
    save_adapter(&opts.save_path, &tensors, opts.rank, alpha);
    progress(opts.steps + 1, opts.steps + 1, format!("saved adapter → {}", opts.save_path));
    Ok(tensors)
}

/// One- or two-GPU trainer behind a common `grads`.
enum TrainerKind {
    One(crate::train::DeviceTrainer),
    Two(crate::shard::ShardTrainer),
}
impl TrainerKind {
    fn new(cfg: Cfg, two_gpu: bool) -> TrainerKind {
        if two_gpu {
            TrainerKind::Two(crate::shard::ShardTrainer::new(cfg, cfg.n_layers / 2))
        } else {
            TrainerKind::One(crate::train::DeviceTrainer::new(cfg))
        }
    }
    fn grads(&self, w: &crate::modelgrad::ModelWeightsF32, b: &Batch) -> (f64, crate::modelgrad::ModelGradsF32) {
        match self {
            TrainerKind::One(t) => t.grads(w, b),
            TrainerKind::Two(t) => t.grads(w, b),
        }
    }
}

// Deterministic RNG (xorshift64*) — `Math.random`-free, reproducible from `seed`.
fn next_u64(s: &mut u64) -> u64 {
    *s ^= *s >> 12;
    *s ^= *s << 25;
    *s ^= *s >> 27;
    s.wrapping_mul(0x2545_f491_4f6c_dd1d)
}
fn uniform01(s: &mut u64) -> f32 {
    (next_u64(s) >> 40) as f32 / (1u64 << 24) as f32
}
fn gauss(s: &mut u64) -> f64 {
    // Box–Muller
    let (u1, u2) = (uniform01(s).max(1e-7) as f64, uniform01(s) as f64);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> Cfg {
        // head_dim must equal Σ axes_dims (32+48+48 = 128) for the RoPE tables to fit.
        Cfg { dim: 128, nh: 1, n_layers: 1, n_refiner: 1, cap_feat_dim: 4, in_channels: 2, patch: 2, h: 4, w: 4, ncap: 3, t_scale: 1000.0 }
    }

    #[test]
    fn flow_batch_shapes_and_convention() {
        let cfg = tiny_cfg();
        let rope = ZImageConfig::turbo().rope(); // axes [32,48,48] = head_dim 128
        let latent: Vec<f64> = (0..cfg.in_channels * cfg.h * cfg.w).map(|i| i as f64).collect();
        let cap = vec![0.5f64; cfg.ncap * cfg.cap_feat_dim];
        let noise = vec![1.0f64; latent.len()];

        // σ = 0 → x_t == latent, t == 1 (clean); target == x₁ - x₀ = latent - 1.
        let b = make_flow_batch(&cfg, &rope, &latent, &cap, 0.0, &noise);
        assert_eq!(b.latent, latent);
        assert_eq!(b.t, 1.0);
        assert_eq!(b.target.len(), cfg.n_img() * cfg.patch_dim());
        // σ = 1 → x_t == noise, t == 0 (pure noise).
        let b1 = make_flow_batch(&cfg, &rope, &latent, &cap, 1.0, &noise);
        assert_eq!(b1.latent, noise);
        assert_eq!(b1.t, 0.0);
        // RoPE tables sized [n_pos · head_dim/2] = [n · 64].
        assert_eq!(b.img_cos.len(), cfg.n_img() * 64);
        assert_eq!(b.cap_cos.len(), cfg.ncap * 64);
    }

    #[test]
    fn latent_transform_roundtrips_pipeline() {
        // DiT-space then inverse (as pipeline decodes) recovers the VAE mean.
        let mean = vec![0.2f32, -0.5, 1.3, 0.0];
        let dit = latent_to_dit(&mean);
        let back: Vec<f32> = dit.iter().map(|&x| x as f32 / VAE_SCALE + VAE_SHIFT).collect();
        for (a, b) in mean.iter().zip(&back) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}
