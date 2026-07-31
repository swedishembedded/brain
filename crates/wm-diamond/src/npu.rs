// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Intel-NPU (OpenVINO whole-graph) path for DIAMOND: export the UNet inner
//! model to fp32 ONNX (`brain wm export`) and play it back through
//! `npu::wm_topology::WmSession` with the sampler staying host-side.
//!
//! Split of responsibilities, mirroring `DiamondUNet::denoise_frame`:
//!   - host: EDM conditioners, Fourier/action/cond-MLP (`crate::cond`), the
//!     c_in pre-scale, the EDM wrap + `quantize`, and the Euler step;
//!   - device: ONE inference per sigma — the whole UNet forward
//!     `F(noisy_scaled, obs_rescaled, cond)`.
//!
//! [`DiamondNpuWorldModel`] duplicates the small context-ring logic of
//! [`crate::play::DiamondWorldModel`] (kept simple on purpose) so `brain wm
//! play/bench --model diamond --device npu --onnx F.onnx` is a drop-in.

use crate::cond::{build_sigmas, conditioners, CondNet};
use crate::config::DiamondConfig;
use crate::model::Tensors;
use crate::play::{quantize, NormalRng, DEFAULT_DENOISE_STEPS};
use npu::openvino::NpuConfig;
use npu::wm_topology::{WmSession, WmUnetConfig};
use wm_core::WorldModel;

// Karras schedule constants (same as crate::play).
const SIGMA_MIN: f32 = 2e-3;
const SIGMA_MAX: f32 = 5.0;
const RHO: f32 = 7.0;

fn unet_cfg(cfg: &DiamondConfig) -> WmUnetConfig {
    WmUnetConfig {
        img_channels: cfg.img_channels,
        num_steps_conditioning: cfg.num_steps_conditioning,
        cond_channels: cfg.cond_channels,
        depths: cfg.depths.clone(),
        channels: cfg.channels.clone(),
        attn_depths: cfg.attn_depths.clone(),
        h: cfg.h,
        w: cfg.w,
    }
}

fn cond_net(cfg: &DiamondConfig, tensors: &Tensors) -> CondNet {
    let get = |n: &str| -> Vec<f32> {
        tensors.get(n).unwrap_or_else(|| panic!("diamond: missing tensor {n}")).1.clone()
    };
    CondNet {
        cond_channels: cfg.cond_channels as usize,
        num_steps_conditioning: cfg.num_steps_conditioning as usize,
        fourier_w: get("noise_emb.weight"),
        act_emb: get("act_emb.0.weight"),
        num_actions: cfg.num_actions as usize,
        mlp0_w: get("cond_proj.0.weight"),
        mlp0_b: get("cond_proj.0.bias"),
        mlp2_w: get("cond_proj.2.weight"),
        mlp2_b: get("cond_proj.2.bias"),
    }
}

/// Build the fp32 ONNX bytes of the DIAMOND UNet inner model.
pub fn build_onnx_bytes(cfg: &DiamondConfig, tensors: &Tensors) -> Vec<u8> {
    let mut g = onnx::GraphBuilder::new("diamond_unet");
    npu::wm_topology::build_diamond_graph(&unet_cfg(cfg), tensors, &mut g);
    g.finish()
}

/// `brain wm export`: `.safetensors` -> fp32 ONNX at `out_path`.
pub fn export_onnx(weights_path: &str, out_path: &str) -> Result<DiamondConfig, String> {
    let (cfg, tensors) = crate::import::load(weights_path)?;
    std::fs::write(out_path, build_onnx_bytes(&cfg, &tensors))
        .map_err(|e| format!("write {out_path}: {e}"))?;
    Ok(cfg)
}

/// The compiled OpenVINO UNet + the host-side conditioning/sampler state.
pub struct DiamondNpu {
    session: WmSession,
    pub cfg: DiamondConfig,
    cond: CondNet,
}

impl DiamondNpu {
    /// Load `.safetensors` (config + host conditioning tensors) and compile
    /// `onnx_path` (from `brain wm export`) for the configured device.
    pub fn load(weights_path: &str, onnx_path: &str, ov: &NpuConfig) -> Result<DiamondNpu, String> {
        let (cfg, tensors) = crate::import::load(weights_path)?;
        Self::new(cfg, &tensors, onnx_path, ov)
    }

    pub fn new(
        cfg: DiamondConfig,
        tensors: &Tensors,
        onnx_path: &str,
        ov: &NpuConfig,
    ) -> Result<DiamondNpu, String> {
        let session = WmSession::load_path(
            std::path::Path::new(onnx_path),
            ov,
            cfg.img_channels as usize,
            (cfg.num_steps_conditioning * cfg.img_channels) as usize,
            cfg.h as usize,
            cfg.w as usize,
            cfg.cond_channels as usize,
        )
        .map_err(|e| e.to_string())?;
        Ok(DiamondNpu { session, cond: cond_net(&cfg, tensors), cfg })
    }

    /// The OpenVINO device the graph actually compiled for (e.g. "NPU").
    pub fn device(&self) -> &str {
        self.session.device()
    }

    /// One inner-model forward `F(c_in*x, c_noise, obs, act)` — parity hook,
    /// same contract as `DiamondUNet::forward` plus the explicit context.
    pub fn forward(
        &mut self,
        noisy_scaled: &[f32],
        c_noise: f32,
        actions: &[u32],
        obs_rescaled: &[f32],
    ) -> Result<Vec<f32>, String> {
        let cond = self.cond.cond(c_noise, actions);
        self.session.run(noisy_scaled, obs_rescaled, &cond).map_err(|e| e.to_string())
    }

    /// Full denoising of one frame, mirroring `DiamondUNet::denoise_frame`:
    /// per sigma, conditioners -> cond -> `F(c_in*x, obs, cond)` on the device
    /// -> EDM wrap + `quantize` -> Euler mix, all sampler math on the host.
    /// `sigmas` is the Karras schedule incl. trailing 0; the final Euler step
    /// (sigma_next = 0) lands exactly on the quantized `denoised`.
    pub fn denoise_frame(
        &mut self,
        x0: &[f32],
        sigmas: &[f32],
        actions: &[u32],
        obs_rescaled: &[f32],
    ) -> Result<Vec<f32>, String> {
        let (sd, so) = (self.cfg.sigma_data, self.cfg.sigma_offset_noise);
        let mut x = x0.to_vec();
        let mut denoised = vec![0.0f32; x.len()];
        for i in 0..sigmas.len() - 1 {
            let sigma = sigmas[i];
            let next = sigmas[i + 1];
            let cs = conditioners(sigma, sd, so);
            let cond = self.cond.cond(cs.c_noise, actions);
            let noisy: Vec<f32> = x.iter().map(|v| v * cs.c_in).collect();
            let f = self.session.run(&noisy, obs_rescaled, &cond).map_err(|e| e.to_string())?;
            for j in 0..x.len() {
                denoised[j] = quantize(cs.c_skip * x[j] + cs.c_out * f[j]);
            }
            // Euler: x' = (1 + dt/sigma)*x - (dt/sigma)*denoised.
            let dt = next - sigma;
            let (a, b) = (1.0 + dt / sigma, -dt / sigma);
            for j in 0..x.len() {
                x[j] = a * x[j] + b * denoised[j];
            }
        }
        Ok(denoised)
    }
}

/// DIAMOND behind [`wm_core::WorldModel`] with the UNet on the Intel NPU —
/// the context ring + Euler loop of `crate::play::DiamondWorldModel`,
/// duplicated (small) rather than refactored.
pub struct DiamondNpuWorldModel {
    npu: DiamondNpu,
    /// Context frames in [-1,1], oldest first: nsc * [ic*h*w].
    obs_ring: Vec<Vec<f32>>,
    act_ring: Vec<u32>,
    rng: NormalRng,
    num_denoise_steps: u32,
    sigmas: Vec<f32>,
    frame_len: usize,
}

impl DiamondNpuWorldModel {
    pub fn new(npu: DiamondNpu, seed: u64) -> DiamondNpuWorldModel {
        let cfg = &npu.cfg;
        let frame_len = (cfg.img_channels * cfg.h * cfg.w) as usize;
        let nsc = cfg.num_steps_conditioning as usize;
        DiamondNpuWorldModel {
            obs_ring: vec![vec![0.0; frame_len]; nsc],
            act_ring: vec![0; nsc],
            rng: NormalRng::new(seed),
            num_denoise_steps: DEFAULT_DENOISE_STEPS,
            sigmas: build_sigmas(DEFAULT_DENOISE_STEPS, SIGMA_MIN, SIGMA_MAX, RHO),
            frame_len,
            npu,
        }
    }

    /// CLI convenience: weights + exported ONNX + the default NPU device.
    pub fn load(weights_path: &str, onnx_path: &str, seed: u64) -> Result<Self, String> {
        let npu = DiamondNpu::load(weights_path, onnx_path, &NpuConfig::default())?;
        Ok(Self::new(npu, seed))
    }

    pub fn device(&self) -> &str {
        self.npu.device()
    }

    fn obs_rescaled(&self) -> Vec<f32> {
        let sd = self.npu.cfg.sigma_data;
        let mut obs = Vec::with_capacity(self.frame_len * self.obs_ring.len());
        for f in &self.obs_ring {
            obs.extend(f.iter().map(|v| v / sd));
        }
        obs
    }

    fn generate(&mut self) -> Vec<f32> {
        let obs = self.obs_rescaled();
        let x0: Vec<f32> = (0..self.frame_len).map(|_| self.rng.normal()).collect();
        self.npu
            .denoise_frame(&x0, &self.sigmas, &self.act_ring, &obs)
            .unwrap_or_else(|e| panic!("diamond npu inference failed: {e}"))
    }
}

impl WorldModel for DiamondNpuWorldModel {
    fn frame_shape(&self) -> (u32, u32, u32) {
        let c = &self.npu.cfg;
        (c.img_channels, c.h, c.w)
    }

    fn num_actions(&self) -> u32 {
        self.npu.cfg.num_actions
    }

    /// Context frames arrive in trait convention ([0,1] CHW, oldest first);
    /// missing context stays zero ([-1,1] mid-grey).
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
        for k in 0..n_act {
            self.act_ring[nsc - n_act + k] = ctx_actions[k].min(self.npu.cfg.num_actions - 1);
        }
    }

    fn step(&mut self, action: u32) -> Vec<f32> {
        let a = action.min(self.npu.cfg.num_actions - 1);
        self.act_ring.rotate_left(1);
        *self.act_ring.last_mut().unwrap() = a;
        let frame = self.generate();
        self.obs_ring.rotate_left(1);
        *self.obs_ring.last_mut().unwrap() = frame.clone();
        frame.iter().map(|v| (v + 1.0) / 2.0).collect()
    }

    /// Quality code from the display layer: 0 = default (3 denoise steps),
    /// 1 -> 2 steps, >=2 -> 1 step (same contract as `crate::play`).
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
