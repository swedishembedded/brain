// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DIAMOND as a playable [`wm_core::WorldModel`]: a context ring of the last
//! `num_steps_conditioning` frames+actions, and per step a Karras/Euler
//! denoising loop (default 3 steps) exactly mirroring the reference
//! `DiffusionSampler` (s_churn = 0, order 1) and `Denoiser.denoise`:
//!
//! - x starts as UNIT normal noise (a reference quirk — not sigma-scaled)
//! - per sigma: F = inner_model(c_in*x, c_noise, obs/sigma_data, act);
//!   denoised = clamp(c_skip*x + c_out*F, -1, 1) quantized to {0..255}
//!   (torch `.byte()` TRUNCATES — reproduced here);
//!   x += (x - denoised)/sigma * (sigma_next - sigma)
//! - frames live in [-1, 1] internally; the trait's [0,1] at the boundary.

use crate::cond::build_sigmas;
use crate::model::DiamondUNet;
use wm_core::WorldModel;

/// SplitMix64 + Box-Muller standard normals (deterministic per seed).
pub struct NormalRng {
    s: u64,
    spare: Option<f32>,
}

impl NormalRng {
    pub fn new(seed: u64) -> NormalRng {
        NormalRng { s: seed, spare: None }
    }
    fn next_u64(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f32 {
        // (0, 1]: avoids ln(0).
        (((self.next_u64() >> 40) + 1) as f32) / (1u64 << 24) as f32
    }
    pub fn normal(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u1 = self.uniform();
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * std::f32::consts::PI * u2;
        self.spare = Some(r * th.sin());
        r * th.cos()
    }
}

/// Reference output quantization: clamp to [-1,1], map to [0,255], TRUNCATE
/// to integer (torch `.byte()`), back to [-1,1].
pub fn quantize(v: f32) -> f32 {
    let c = v.clamp(-1.0, 1.0);
    let byte = ((c + 1.0) / 2.0 * 255.0) as u8; // `as` truncates like torch
    (byte as f32) / 255.0 * 2.0 - 1.0
}

pub struct DiamondWorldModel {
    unet: DiamondUNet,
    /// Context frames in [-1,1], oldest first: nsc * [ic*h*w].
    obs_ring: Vec<Vec<f32>>,
    act_ring: Vec<u32>,
    rng: NormalRng,
    /// The seed the RNG is (re)initialized from — kept so `reset`/`reset_initial`
    /// can rewind the noise stream to a reproducible start.
    seed: u64,
    /// Snapshot of the context rings captured at the last `reset`, so
    /// `reset_initial` (the interactive Enter key) can restore the exact
    /// starting frame instead of clearing to grey.
    init_obs: Vec<Vec<f32>>,
    init_act: Vec<u32>,
    num_denoise_steps: u32,
    sigmas: Vec<f32>,
    frame_len: usize,
}

pub const DEFAULT_DENOISE_STEPS: u32 = 3;
const SIGMA_MIN: f32 = 2e-3;
const SIGMA_MAX: f32 = 5.0;
const RHO: f32 = 7.0;

impl DiamondWorldModel {
    pub fn new(unet: DiamondUNet, seed: u64) -> DiamondWorldModel {
        let cfg = &unet.cfg;
        let frame_len = (cfg.img_channels * cfg.h * cfg.w) as usize;
        let nsc = cfg.num_steps_conditioning as usize;
        DiamondWorldModel {
            obs_ring: vec![vec![0.0; frame_len]; nsc],
            act_ring: vec![0; nsc],
            rng: NormalRng::new(seed),
            seed,
            init_obs: vec![vec![0.0; frame_len]; nsc],
            init_act: vec![0; nsc],
            num_denoise_steps: DEFAULT_DENOISE_STEPS,
            sigmas: build_sigmas(DEFAULT_DENOISE_STEPS, SIGMA_MIN, SIGMA_MAX, RHO),
            frame_len,
            unet,
        }
    }

    fn upload_context(&self) {
        let sd = self.unet.cfg.sigma_data;
        let mut obs = Vec::with_capacity(self.frame_len * self.obs_ring.len());
        for f in &self.obs_ring {
            obs.extend(f.iter().map(|v| v / sd));
        }
        self.unet.set_context(&obs);
    }

    /// One world-model step: append the action, denoise the next frame.
    /// The whole sampling loop runs on-device (see DiamondUNet::denoise_frame);
    /// only the unit-noise init goes up and the final frame comes back.
    fn generate(&mut self) -> Vec<f32> {
        self.upload_context();
        let x0: Vec<f32> = (0..self.frame_len).map(|_| self.rng.normal()).collect();
        self.unet.denoise_frame(&x0, &self.sigmas, &self.act_ring)
    }
}

impl WorldModel for DiamondWorldModel {
    fn frame_shape(&self) -> (u32, u32, u32) {
        let c = &self.unet.cfg;
        (c.img_channels, c.h, c.w)
    }

    fn num_actions(&self) -> u32 {
        self.unet.cfg.num_actions
    }

    /// Context frames arrive in trait convention ([0,1] CHW, oldest first);
    /// missing context stays zero ([-1,1] mid-grey). Re-seeds the noise stream
    /// to the construction seed and snapshots the resulting context as the
    /// initial state, so a subsequent [`reset_initial`](WorldModel::reset_initial)
    /// (the Enter key) rewinds to exactly this frame.
    fn reset(&mut self, ctx_frames: &[f32], ctx_actions: &[u32]) {
        let nsc = self.obs_ring.len();
        for f in self.obs_ring.iter_mut() {
            f.iter_mut().for_each(|v| *v = 0.0);
        }
        self.act_ring.iter_mut().for_each(|a| *a = 0);
        let n_frames = (ctx_frames.len() / self.frame_len).min(nsc);
        for k in 0..n_frames {
            let src = &ctx_frames[k * self.frame_len..(k + 1) * self.frame_len];
            let dst = nsc - n_frames + k;
            for (d, s) in self.obs_ring[dst].iter_mut().zip(src) {
                *d = s * 2.0 - 1.0;
            }
        }
        let n_act = ctx_actions.len().min(nsc);
        for (k, &a) in ctx_actions.iter().enumerate().take(n_act) {
            self.act_ring[nsc - n_act + k] = a.min(self.unet.cfg.num_actions - 1);
        }
        self.rng = NormalRng::new(self.seed);
        self.init_obs = self.obs_ring.clone();
        self.init_act = self.act_ring.clone();
    }

    /// Rewind to the snapshot captured at the last `reset`: restore the seed
    /// context rings and re-seed the noise stream, so the interactive Enter
    /// key replays the sequence from an identical first frame.
    fn reset_initial(&mut self) {
        self.obs_ring = self.init_obs.clone();
        self.act_ring = self.init_act.clone();
        self.rng = NormalRng::new(self.seed);
    }

    fn step(&mut self, action: u32) -> Vec<f32> {
        let a = action.min(self.unet.cfg.num_actions - 1);
        self.act_ring.rotate_left(1);
        *self.act_ring.last_mut().unwrap() = a;
        let frame = self.generate();
        self.obs_ring.rotate_left(1);
        *self.obs_ring.last_mut().unwrap() = frame.clone();
        frame.iter().map(|v| (v + 1.0) / 2.0).collect()
    }

    /// Quality code from the display layer: 0 = default (3 denoise steps),
    /// 1 -> 2 steps, >=2 -> 1 step.
    fn set_nfe(&mut self, code: u32) {
        let steps = match code {
            0 => DEFAULT_DENOISE_STEPS,
            1 => 2,
            _ => 1,
        };
        if steps != self.num_denoise_steps {
            self.num_denoise_steps = steps;
            self.sigmas = build_sigmas(steps, SIGMA_MIN, SIGMA_MAX, RHO);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn play_quantize_matches_torch_byte_truncation() {
        // v=0: (0+1)/2*255 = 127.5 -> byte 127 -> 127/255*2-1 = -0.0039216.
        assert!((quantize(0.0) + 0.003_921_6).abs() < 1e-6);
        assert_eq!(quantize(1.0), 1.0);
        assert_eq!(quantize(-1.0), -1.0);
        assert_eq!(quantize(2.0), 1.0); // clamps first
    }

    #[test]
    fn play_normal_rng_is_deterministic_and_roughly_standard() {
        let mut a = NormalRng::new(7);
        let mut b = NormalRng::new(7);
        let xs: Vec<f32> = (0..10_000).map(|_| a.normal()).collect();
        let ys: Vec<f32> = (0..10_000).map(|_| b.normal()).collect();
        assert_eq!(xs, ys);
        let mean = xs.iter().sum::<f32>() / xs.len() as f32;
        let var = xs.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / xs.len() as f32;
        assert!(mean.abs() < 0.05, "mean={mean}");
        assert!((var - 1.0).abs() < 0.05, "var={var}");
    }
}
