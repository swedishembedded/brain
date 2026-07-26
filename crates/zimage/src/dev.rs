// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device-resident Z-Image DiT forward.
//!
//! Unlike [`crate::ZImageModel`] (one device per block, a host round-trip per
//! block — the parity reference), this uploads every block's weights once and
//! records each stage as ONE graph with resident intermediates: the 30 main
//! layers (and the refiners) chain on-device with no per-block transfers. Only
//! ~3 host round-trips remain (between the image/caption/unified stages + the
//! host embedders/final-layer). Build once for a fixed size, forward many times
//! (profiling / multi-step sampling).

use gpu_core::{DeviceBuffer, Gpu, Step};

use crate::block::{build_block_steps, wf, BlockDims, BlockWeights, NormBufs, Scratch, Tensors, KERNELS};
use crate::model::{postprocess, preprocess, timestep_cond};
use crate::ZImageConfig;

/// One stage: a chain of blocks recorded into a single graph with resident
/// weights, run with one submit. Intermediates come from a single reused
/// [`Scratch`]; the residual double-buffers between two slabs (`input`/`_resb`).
struct Phase {
    input: DeviceBuffer,
    output: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    steps: Vec<Step>,
    _weights: Vec<BlockWeights>, // kept resident (referenced by steps)
    norms: Vec<NormBufs>,
    t: u32,
    _scr: Scratch,
    _resb: DeviceBuffer,
}

#[allow(clippy::too_many_arguments)]
fn build_phase(gpu: &Gpu, tensors: &Tensors, prefixes: &[String], bd: BlockDims, t: u32, modulation: bool, reg2: bool) -> Phase {
    let half = bd.head_dim / 2;
    let resa = gpu.storage((t * bd.dim) as u64);
    let resb = gpu.storage((t * bd.dim) as u64);
    let cos = gpu.storage((t * half) as u64);
    let sin = gpu.storage((t * half) as u64);
    let scr = Scratch::new(gpu, bd, t);
    let (mut weights, mut norms, mut steps) = (Vec::new(), Vec::new(), Vec::new());
    // Double-buffer the residual: block reads `cur_in`, writes `cur_out`, swap.
    let (mut cur_in, mut cur_out) = (resa.clone(), resb.clone());
    for p in prefixes {
        let w = BlockWeights::upload(gpu, tensors, p);
        let nb = NormBufs::new(gpu, tensors, p, bd.dim, modulation);
        build_block_steps(gpu, &mut steps, &w, &nb, &cur_in, &cur_out, &scr, &cos, &sin, bd, t, reg2);
        weights.push(w);
        norms.push(nb);
        std::mem::swap(&mut cur_in, &mut cur_out);
        // Flush so create_buffer_init upload staging is reclaimed rather than
        // accumulating on top of the resident weights.
        gpu.poll_wait();
    }
    Phase { input: resa, output: cur_in, cos, sin, steps, _weights: weights, norms, t, _scr: scr, _resb: resb }
}

impl Phase {
    fn run(&self, gpu: &Gpu, tokens: &[f32], c: &[f32], cos: &[f32], sin: &[f32], dim: usize, cdim: usize) -> Vec<f32> {
        for nb in &self.norms {
            nb.upload_folded(gpu, c, dim, cdim);
        }
        wf(gpu, &self.input, tokens);
        wf(gpu, &self.cos, cos);
        wf(gpu, &self.sin, sin);
        gpu.submit(&[], &self.steps);
        gpu.read(&self.output, self.t as usize * dim)
    }
}

/// A Z-Image DiT with all weights resident and stage graphs prebuilt for a fixed
/// image/caption size.
pub struct ZImageDit {
    gpu: Gpu,
    cfg: ZImageConfig,
    w: Tensors,
    f: u32,
    h: u32,
    wd: u32,
    cap_len: u32,
    noise: Phase,
    context: Phase,
    main: Phase,
}

impl ZImageDit {
    /// Build resident stage graphs for the given latent size `(f,h,wd)` and
    /// caption length. `device`: `Some("cpu")`|`Some("gpu")`|`None`.
    pub fn build(cfg: ZImageConfig, weights: Tensors, f: u32, h: u32, wd: u32, cap_len: u32, device: Option<&str>) -> ZImageDit {
        let reg2 = device != Some("cpu");
        let gpu = match device {
            Some("cpu") => Gpu::new_cpu(&KERNELS),
            Some("gpu") | Some("wgpu") => Gpu::new_wgpu(&KERNELS),
            _ => Gpu::new(&KERNELS),
        };
        let bd = cfg.block_dims();
        let (ps, pf) = (cfg.patch_size, cfg.f_patch_size);
        let n_img = (f / pf) * (h / ps) * (wd / ps);
        let ntot = n_img + cap_len;
        let np: Vec<String> = (0..cfg.n_refiner_layers).map(|l| format!("noise_refiner.{l}")).collect();
        let cp: Vec<String> = (0..cfg.n_refiner_layers).map(|l| format!("context_refiner.{l}")).collect();
        let mp: Vec<String> = (0..cfg.n_layers).map(|l| format!("layers.{l}")).collect();
        let noise = build_phase(&gpu, &weights, &np, bd, n_img, true, reg2);
        let context = build_phase(&gpu, &weights, &cp, bd, cap_len, false, reg2);
        let main = build_phase(&gpu, &weights, &mp, bd, ntot, true, reg2);
        ZImageDit { gpu, cfg, w: weights, f, h, wd, cap_len, noise, context, main }
    }

    /// One DiT forward for the built size. `latent`: `[C·F·H·W]`; `cap`:
    /// `[cap_len·cap_feat_dim]`; `t`: timestep. Returns the latent `[C·F·H·W]`.
    pub fn forward(&self, latent: &[f32], cap: &[f32], t: f32) -> Vec<f32> {
        let c = &self.cfg;
        let dim = c.dim as usize;
        let cdim = dim.min(256);
        let cvec = timestep_cond(c, &self.w, t);
        let pre = preprocess(c, &self.w, latent, self.f, self.h, self.wd, cap, self.cap_len);
        let img = self.noise.run(&self.gpu, &pre.img, &cvec, &pre.img_rope.cos, &pre.img_rope.sin, dim, cdim);
        let capt = self.context.run(&self.gpu, &pre.capt, &cvec, &pre.cap_rope.cos, &pre.cap_rope.sin, dim, cdim);
        let mut uni = img;
        uni.extend_from_slice(&capt);
        let mut uni_cos = pre.img_rope.cos.clone();
        uni_cos.extend_from_slice(&pre.cap_rope.cos);
        let mut uni_sin = pre.img_rope.sin.clone();
        uni_sin.extend_from_slice(&pre.cap_rope.sin);
        let uni_out = self.main.run(&self.gpu, &uni, &cvec, &uni_cos, &uni_sin, dim, cdim);
        postprocess(c, &self.w, &uni_out, &cvec, pre.n_img, self.f, self.h, self.wd)
    }
}

/// Z-Image DiT sharded across two GPUs so the 6B fp32 model FITS: the refiners +
/// first half of the main layers on GPU 0, the second half on GPU 1, with one
/// host-staged residual transfer at the cut (no NVLink on the P40 box). A single
/// forward runs the two cards sequentially; batch/pipeline overlap is a later
/// step — this makes the model fit and gives the per-forward latency.
pub struct ZImageDitShard {
    gpu0: Gpu,
    gpu1: Gpu,
    cfg: ZImageConfig,
    w: Tensors,
    f: u32,
    h: u32,
    wd: u32,
    cap_len: u32,
    noise: Phase,
    context: Phase,
    main0: Phase,
    main1: Phase,
}

impl ZImageDitShard {
    /// Build across GPUs 0 and 1, cutting the `n_layers` main stack in half.
    pub fn build(cfg: ZImageConfig, weights: Tensors, f: u32, h: u32, wd: u32, cap_len: u32) -> ZImageDitShard {
        let bd = cfg.block_dims();
        let (ps, pf) = (cfg.patch_size, cfg.f_patch_size);
        let n_img = (f / pf) * (h / ps) * (wd / ps);
        let ntot = n_img + cap_len;
        // Balance by BLOCK COUNT (all blocks weigh the same): card 0 also carries
        // the 2 noise + 2 context refiners, so it gets fewer main layers. cut so
        // card0 (2·n_refiner + cut) ≈ card1 (n_layers - cut). Each card ~half the
        // weights — the ~1.6× wgpu allocator overhead then fits a 24 GB card.
        let refiners = 2 * cfg.n_refiner_layers;
        let cut = (cfg.n_layers.saturating_sub(refiners) / 2) as usize;

        // One enumeration → two distinct physical cards (two separate new_wgpu
        // calls reorder and both land on card 0 on this box).
        let mut gpus = Gpu::new_wgpu_multi(&KERNELS, 2);
        let gpu1 = gpus.pop().unwrap();
        let gpu0 = gpus.pop().unwrap();

        let np: Vec<String> = (0..cfg.n_refiner_layers).map(|l| format!("noise_refiner.{l}")).collect();
        let cp: Vec<String> = (0..cfg.n_refiner_layers).map(|l| format!("context_refiner.{l}")).collect();
        let mp0: Vec<String> = (0..cut).map(|l| format!("layers.{l}")).collect();
        let mp1: Vec<String> = (cut..cfg.n_layers as usize).map(|l| format!("layers.{l}")).collect();
        let noise = build_phase(&gpu0, &weights, &np, bd, n_img, true, true);
        let context = build_phase(&gpu0, &weights, &cp, bd, cap_len, false, true);
        let main0 = build_phase(&gpu0, &weights, &mp0, bd, ntot, true, true);
        let main1 = build_phase(&gpu1, &weights, &mp1, bd, ntot, true, true);
        ZImageDitShard { gpu0, gpu1, cfg, w: weights, f, h, wd, cap_len, noise, context, main0, main1 }
    }

    pub fn forward(&self, latent: &[f32], cap: &[f32], t: f32) -> Vec<f32> {
        let c = &self.cfg;
        let dim = c.dim as usize;
        let cdim = dim.min(256);
        let cvec = timestep_cond(c, &self.w, t);
        let pre = preprocess(c, &self.w, latent, self.f, self.h, self.wd, cap, self.cap_len);
        let img = self.noise.run(&self.gpu0, &pre.img, &cvec, &pre.img_rope.cos, &pre.img_rope.sin, dim, cdim);
        let capt = self.context.run(&self.gpu0, &pre.capt, &cvec, &pre.cap_rope.cos, &pre.cap_rope.sin, dim, cdim);
        let mut uni = img;
        uni.extend_from_slice(&capt);
        let mut uni_cos = pre.img_rope.cos.clone();
        uni_cos.extend_from_slice(&pre.cap_rope.cos);
        let mut uni_sin = pre.img_rope.sin.clone();
        uni_sin.extend_from_slice(&pre.cap_rope.sin);
        // GPU0: first half; host-staged residual; GPU1: second half.
        let mid = self.main0.run(&self.gpu0, &uni, &cvec, &uni_cos, &uni_sin, dim, cdim);
        let uni_out = self.main1.run(&self.gpu1, &mid, &cvec, &uni_cos, &uni_sin, dim, cdim);
        postprocess(c, &self.w, &uni_out, &cvec, pre.n_img, self.f, self.h, self.wd)
    }
}
