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

use crate::modelgrad::{patchify, Cfg};
use crate::model::ZImageConfig;
use crate::train::Batch;
use dit::rope::{tables_for_ids, RopeConfig};

/// Latent-space VAE scale/shift (FLUX VAE; must match [`crate::pipeline`]).
pub const VAE_SCALE: f32 = 0.3611;
pub const VAE_SHIFT: f32 = 0.1159;

/// Build the training [`Cfg`] for a real Z-Image checkpoint at latent size
/// `h×w` (latent pixels = image/8) and `cap_len` caption tokens.
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
