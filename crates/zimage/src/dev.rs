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

use crate::block::{build_block_steps, wf, BlockDims, BlockWeights, NormBufs, Tensors, KERNELS};
use crate::model::{postprocess, preprocess, timestep_cond};
use crate::ZImageConfig;

/// One stage: a chain of blocks recorded into a single graph with resident
/// weights + intermediates, run with one submit.
struct Phase {
    input: DeviceBuffer,
    output: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    steps: Vec<Step>,
    _weights: Vec<BlockWeights>, // kept resident (referenced by steps)
    norms: Vec<NormBufs>,
    t: u32,
}

fn build_phase(gpu: &Gpu, tensors: &Tensors, prefixes: &[String], bd: BlockDims, t: u32, modulation: bool) -> Phase {
    let half = bd.head_dim / 2;
    let input = gpu.storage((t * bd.dim) as u64);
    let cos = gpu.storage((t * half) as u64);
    let sin = gpu.storage((t * half) as u64);
    let (mut weights, mut norms, mut steps) = (Vec::new(), Vec::new(), Vec::new());
    let mut x = input.clone();
    for p in prefixes {
        let w = BlockWeights::upload(gpu, tensors, p);
        let nb = NormBufs::new(gpu, tensors, p, bd.dim, modulation);
        x = build_block_steps(gpu, &mut steps, &w, &nb, &x, &cos, &sin, bd, t);
        weights.push(w);
        norms.push(nb);
    }
    Phase { input, output: x, cos, sin, steps, _weights: weights, norms, t }
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
        let noise = build_phase(&gpu, &weights, &np, bd, n_img, true);
        let context = build_phase(&gpu, &weights, &cp, bd, cap_len, false);
        let main = build_phase(&gpu, &weights, &mp, bd, ntot, true);
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
