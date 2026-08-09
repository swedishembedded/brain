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

use crate::block::{
    build_block_steps, build_block_steps_i8, read_named, wf, BlockDims, BlockWeights, Int8Scratch,
    Int8Weights, NormBufs, Scratch, Tensors, KERNELS,
};
use crate::model::{postprocess, preprocess, timestep_cond, HostLookup};
use crate::ZImageConfig;

/// The 13 tensors `preprocess`/`timestep_cond`/`postprocess` read every
/// forward — and therefore the ONLY tensors a built DiT retains in host RAM.
/// Named individually so adding one is a visible, reviewable edit, not an
/// accident: `ZImageDit`/`ZImageDitI8`/`ZImageDitShard` used to store the
/// COMPLETE source tensor map (`w: Tensors`) for their whole life, even
/// though every block's weights are already quantized/uploaded and never
/// read from that map again — on the int8 DiT alone that was ~24 GB of host
/// RAM held for nothing. See `docs/lessons.md`: "a builder that takes
/// `HashMap<String, Vec<f32>>` has already lost — the caller must
/// materialize everything, and the callee may keep it, and neither is
/// visible in the type."
pub(crate) struct HostWeights {
    xemb_w: Vec<f32>,
    xemb_b: Vec<f32>,
    cap0_w: Vec<f32>,
    cap1_w: Vec<f32>,
    cap1_b: Vec<f32>,
    t0_w: Vec<f32>,
    t0_b: Vec<f32>,
    t2_w: Vec<f32>,
    t2_b: Vec<f32>,
    fadaln_w: Vec<f32>,
    fadaln_b: Vec<f32>,
    flin_w: Vec<f32>,
    flin_b: Vec<f32>,
}

impl HostWeights {
    /// Pull exactly these 13 tensors from `src` — one at a time, via
    /// `read_named` (bounded the same way every device upload here is) —
    /// and nothing else. If `src` is a streaming `WeightReader`, the other
    /// ~2000 tensors of a real checkpoint are never materialized on the
    /// host at all.
    fn from_source(cfg: &ZImageConfig, src: &dyn checkpoint::TensorSource) -> HostWeights {
        let (ps, pf) = (cfg.patch_size, cfg.f_patch_size);
        let xk = format!("all_x_embedder.{ps}-{pf}");
        let fk = format!("all_final_layer.{ps}-{pf}");
        HostWeights {
            xemb_w: read_named(src, &format!("{xk}.weight")),
            xemb_b: read_named(src, &format!("{xk}.bias")),
            cap0_w: read_named(src, "cap_embedder.0.weight"),
            cap1_w: read_named(src, "cap_embedder.1.weight"),
            cap1_b: read_named(src, "cap_embedder.1.bias"),
            t0_w: read_named(src, "t_embedder.mlp.0.weight"),
            t0_b: read_named(src, "t_embedder.mlp.0.bias"),
            t2_w: read_named(src, "t_embedder.mlp.2.weight"),
            t2_b: read_named(src, "t_embedder.mlp.2.bias"),
            fadaln_w: read_named(src, &format!("{fk}.adaLN_modulation.1.weight")),
            fadaln_b: read_named(src, &format!("{fk}.adaLN_modulation.1.bias")),
            flin_w: read_named(src, &format!("{fk}.linear.weight")),
            flin_b: read_named(src, &format!("{fk}.linear.bias")),
        }
    }
}

impl HostLookup for HostWeights {
    fn xemb(&self, _cfg: &ZImageConfig) -> (&[f32], &[f32]) {
        (&self.xemb_w, &self.xemb_b)
    }
    fn cap_norm(&self) -> &[f32] {
        &self.cap0_w
    }
    fn cap_embed(&self) -> (&[f32], &[f32]) {
        (&self.cap1_w, &self.cap1_b)
    }
    fn t_embed(&self) -> (&[f32], &[f32], &[f32], &[f32]) {
        (&self.t0_w, &self.t0_b, &self.t2_w, &self.t2_b)
    }
    fn final_layer(&self, _cfg: &ZImageConfig) -> (&[f32], &[f32], &[f32], &[f32]) {
        (&self.fadaln_w, &self.fadaln_b, &self.flin_w, &self.flin_b)
    }
}

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
fn build_phase(gpu: &Gpu, tensors: &dyn checkpoint::TensorSource, prefixes: &[String], bd: BlockDims, t: u32, modulation: bool, reg_gemm: bool) -> Phase {
    let half = bd.head_dim / 2;
    let resa = gpu.storage((t * bd.dim) as u64);
    let resb = gpu.storage((t * bd.dim) as u64);
    let cos = gpu.storage((t * half) as u64);
    let sin = gpu.storage((t * half) as u64);
    // Flash only on the GPU (reg_gemm ⇒ GPU; the CPU JIT can't compile the barrier);
    // Scratch must match so it skips the [nh·t·t] buffers under flash.
    let scr = Scratch::new_maybe_flash(gpu, bd, t, reg_gemm && crate::block::use_flash(gpu, bd.n_heads, t));
    let (mut weights, mut norms, mut steps) = (Vec::new(), Vec::new(), Vec::new());
    // Double-buffer the residual: block reads `cur_in`, writes `cur_out`, swap.
    let (mut cur_in, mut cur_out) = (resa.clone(), resb.clone());
    for p in prefixes {
        let w = BlockWeights::upload(gpu, tensors, p);
        let nb = NormBufs::new(gpu, tensors, p, bd.dim, modulation);
        build_block_steps(gpu, &mut steps, &w, &nb, &cur_in, &cur_out, &scr, &cos, &sin, bd, t, reg_gemm);
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
    w: HostWeights,
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
    ///
    /// Takes an OWNED `Tensors` for callers that already have one in hand
    /// (small-config tests, `zimage_bench`) — but even here, only the 13
    /// wrapper tensors ([`HostWeights`]) survive past this call; `weights`
    /// itself drops when this function returns. A production loader that
    /// wants to never materialize a whole model on the host at all should
    /// call [`Self::build_from_source`] directly over a streaming
    /// `checkpoint::TensorSource` instead of constructing a `Tensors` first.
    pub fn build(cfg: ZImageConfig, weights: Tensors, f: u32, h: u32, wd: u32, cap_len: u32, device: Option<&str>) -> ZImageDit {
        Self::build_from_source(cfg, &weights, f, h, wd, cap_len, device)
    }

    /// [`Self::build`] over any streaming `checkpoint::TensorSource` — a
    /// `Tensors` (coerces, above) or an mmap'd `WeightReader`/`RemapSource`
    /// pair, which never materializes more than one tensor at a time. Peak
    /// host allocation for the DiT's weights is then one tensor (up to
    /// ~157 MB for `feed_forward.w1`), not the whole model.
    pub fn build_from_source(cfg: ZImageConfig, src: &dyn checkpoint::TensorSource, f: u32, h: u32, wd: u32, cap_len: u32, device: Option<&str>) -> ZImageDit {
        let reg_gemm = device != Some("cpu");
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
        let noise = build_phase(&gpu, src, &np, bd, n_img, true, reg_gemm);
        let context = build_phase(&gpu, src, &cp, bd, cap_len, false, reg_gemm);
        let main = build_phase(&gpu, src, &mp, bd, ntot, true, reg_gemm);
        let w = HostWeights::from_source(&cfg, src);
        ZImageDit { gpu, cfg, w, f, h, wd, cap_len, noise, context, main }
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
    w: HostWeights,
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
    /// See [`ZImageDit::build`]'s doc: only [`HostWeights`] survives past
    /// this call, whichever entry point is used.
    pub fn build(cfg: ZImageConfig, weights: Tensors, f: u32, h: u32, wd: u32, cap_len: u32) -> ZImageDitShard {
        Self::build_from_source(cfg, &weights, f, h, wd, cap_len)
    }

    /// [`Self::build`] over a streaming `checkpoint::TensorSource`.
    pub fn build_from_source(cfg: ZImageConfig, src: &dyn checkpoint::TensorSource, f: u32, h: u32, wd: u32, cap_len: u32) -> ZImageDitShard {
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
        let noise = build_phase(&gpu0, src, &np, bd, n_img, true, true);
        let context = build_phase(&gpu0, src, &cp, bd, cap_len, false, true);
        let main0 = build_phase(&gpu0, src, &mp0, bd, ntot, true, true);
        let main1 = build_phase(&gpu1, src, &mp1, bd, ntot, true, true);
        let w = HostWeights::from_source(&cfg, src);
        ZImageDitShard { gpu0, gpu1, cfg, w, f, h, wd, cap_len, noise, context, main0, main1 }
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

// ---------------- int8 (DP4A) single-GPU DiT ----------------

/// One int8 stage: like [`Phase`] but the linears run DP4A int8.
struct Int8Phase {
    input: DeviceBuffer,
    output: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    steps: Vec<Step>,
    _weights: Vec<Int8Weights>,
    _scr: Scratch,
    _i8: Int8Scratch,
    _resb: DeviceBuffer,
    norms: Vec<NormBufs>,
    t: u32,
}

fn build_phase_i8(gpu: &Gpu, tensors: &dyn checkpoint::TensorSource, prefixes: &[String], bd: BlockDims, t: u32, modulation: bool) -> Int8Phase {
    let half = bd.head_dim / 2;
    let resa = gpu.storage((t * bd.dim) as u64);
    let resb = gpu.storage((t * bd.dim) as u64);
    let cos = gpu.storage((t * half) as u64);
    let sin = gpu.storage((t * half) as u64);
    // int8 path is GPU-only, so flash follows the plain heuristic.
    let scr = Scratch::new_maybe_flash(gpu, bd, t, crate::block::use_flash(gpu, bd.n_heads, t));
    let i8 = Int8Scratch::new(gpu, bd, t);
    let (mut weights, mut norms, mut steps) = (Vec::new(), Vec::new(), Vec::new());
    let (mut cur_in, mut cur_out) = (resa.clone(), resb.clone());
    for p in prefixes {
        let w = Int8Weights::upload(gpu, tensors, p, bd);
        let nb = NormBufs::new(gpu, tensors, p, bd.dim, modulation);
        build_block_steps_i8(gpu, &mut steps, &w, &nb, &cur_in, &cur_out, &scr, &i8, &cos, &sin, bd, t);
        weights.push(w);
        norms.push(nb);
        std::mem::swap(&mut cur_in, &mut cur_out);
        gpu.poll_wait();
    }
    Int8Phase { input: resa, output: cur_in, cos, sin, steps, _weights: weights, _scr: scr, _i8: i8, _resb: resb, norms, t }
}

impl Int8Phase {
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

/// Z-Image DiT with int8 (DP4A) linears — the 6B fits ONE 24 GB P40 (~6 GB of
/// weights), no sharding. Build once, forward many times.
pub struct ZImageDitI8 {
    gpu: Gpu,
    cfg: ZImageConfig,
    w: HostWeights,
    f: u32,
    h: u32,
    wd: u32,
    cap_len: u32,
    noise: Int8Phase,
    context: Int8Phase,
    main: Int8Phase,
}

impl ZImageDitI8 {
    /// See [`ZImageDit::build`]'s doc: only [`HostWeights`] survives past
    /// this call — every one of the 32 int8-quantized blocks' fp32 weights
    /// (already uploaded to device) is dropped, not retained a second time.
    pub fn build(cfg: ZImageConfig, weights: Tensors, f: u32, h: u32, wd: u32, cap_len: u32) -> ZImageDitI8 {
        Self::build_from_source(cfg, &weights, f, h, wd, cap_len)
    }

    /// [`Self::build`] over a streaming `checkpoint::TensorSource`.
    pub fn build_from_source(cfg: ZImageConfig, src: &dyn checkpoint::TensorSource, f: u32, h: u32, wd: u32, cap_len: u32) -> ZImageDitI8 {
        let gpu = Gpu::new_wgpu(&KERNELS);
        let bd = cfg.block_dims();
        let (ps, pf) = (cfg.patch_size, cfg.f_patch_size);
        let n_img = (f / pf) * (h / ps) * (wd / ps);
        let ntot = n_img + cap_len;
        let np: Vec<String> = (0..cfg.n_refiner_layers).map(|l| format!("noise_refiner.{l}")).collect();
        let cp: Vec<String> = (0..cfg.n_refiner_layers).map(|l| format!("context_refiner.{l}")).collect();
        let mp: Vec<String> = (0..cfg.n_layers).map(|l| format!("layers.{l}")).collect();
        let noise = build_phase_i8(&gpu, src, &np, bd, n_img, true);
        let context = build_phase_i8(&gpu, src, &cp, bd, cap_len, false);
        let main = build_phase_i8(&gpu, src, &mp, bd, ntot, true);
        let w = HostWeights::from_source(&cfg, src);
        ZImageDitI8 { gpu, cfg, w, f, h, wd, cap_len, noise, context, main }
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> ZImageConfig {
        ZImageConfig {
            dim: 8,
            n_layers: 2,
            n_refiner_layers: 1,
            n_heads: 2,
            cap_feat_dim: 4,
            in_channels: 4,
            patch_size: 2,
            f_patch_size: 1,
            axes_dims: vec![2, 2, 2],
            axes_lens: vec![8, 4, 4],
            rope_theta: 256.0,
            t_scale: 1000.0,
            norm_eps: 1e-5,
        }
    }

    /// A synthetic checkpoint's block weights (deliberately far larger than
    /// the 13 wrapper tensors) must not survive past `build()` — the
    /// regression test for the retention bug: `ZImageDit`/`ZImageDitI8`/
    /// `ZImageDitShard` used to store the COMPLETE source map (`w: Tensors`)
    /// for their whole life, even though every block's weights are already
    /// uploaded to the device and never read from the host again. On the
    /// int8 DiT alone that was ~24 GB of host RAM held for nothing.
    #[test]
    fn built_dit_retains_only_the_wrapper_tensors() {
        let cfg = tiny_cfg();
        let dim = cfg.dim as usize;
        let (ps, pf) = (cfg.patch_size, cfg.f_patch_size);
        let patch_dim = (pf * ps * ps * cfg.in_channels) as usize;
        let cdim = dim.min(256);
        let mut t: Tensors = Tensors::new();
        let mut expected = 0usize;
        let mut insert = |t: &mut Tensors, name: String, n: usize| {
            t.insert(name, (vec![n], vec![0.1f32; n]));
            expected += n;
        };

        // The 13 real wrapper tensors, correctly sized for `tiny_cfg`.
        let xk = format!("all_x_embedder.{ps}-{pf}");
        let fk = format!("all_final_layer.{ps}-{pf}");
        insert(&mut t, format!("{xk}.weight"), dim * patch_dim);
        insert(&mut t, format!("{xk}.bias"), dim);
        insert(&mut t, "cap_embedder.0.weight".into(), cfg.cap_feat_dim as usize);
        insert(&mut t, "cap_embedder.1.weight".into(), dim * cfg.cap_feat_dim as usize);
        insert(&mut t, "cap_embedder.1.bias".into(), dim);
        insert(&mut t, "t_embedder.mlp.0.weight".into(), 1024 * 256);
        insert(&mut t, "t_embedder.mlp.0.bias".into(), 1024);
        insert(&mut t, "t_embedder.mlp.2.weight".into(), cdim * 1024);
        insert(&mut t, "t_embedder.mlp.2.bias".into(), cdim);
        insert(&mut t, format!("{fk}.adaLN_modulation.1.weight"), dim * cdim);
        insert(&mut t, format!("{fk}.adaLN_modulation.1.bias"), dim);
        insert(&mut t, format!("{fk}.linear.weight"), patch_dim * dim);
        insert(&mut t, format!("{fk}.linear.bias"), patch_dim);

        // Every block's weights (never read again on the host after upload) —
        // deliberately oversized relative to any wrapper tensor above.
        const HUGE: usize = 100_000;
        for prefix in ["noise_refiner.0", "context_refiner.0", "layers.0", "layers.1"] {
            for suffix in [
                "attention.to_q.weight",
                "attention.to_k.weight",
                "attention.to_v.weight",
                "attention.to_out.0.weight",
                "attention.norm_q.weight",
                "attention.norm_k.weight",
                "feed_forward.w1.weight",
                "feed_forward.w2.weight",
                "feed_forward.w3.weight",
                "attention_norm1.weight",
                "attention_norm2.weight",
                "ffn_norm1.weight",
                "ffn_norm2.weight",
                "adaLN_modulation.0.weight",
                "adaLN_modulation.0.bias",
            ] {
                t.insert(format!("{prefix}.{suffix}"), (vec![HUGE], vec![0.0f32; HUGE]));
            }
        }

        let dit = ZImageDit::build(cfg, t, 1, 4, 4, 4, Some("cpu"));
        let retained = dit.w.xemb_w.len()
            + dit.w.xemb_b.len()
            + dit.w.cap0_w.len()
            + dit.w.cap1_w.len()
            + dit.w.cap1_b.len()
            + dit.w.t0_w.len()
            + dit.w.t0_b.len()
            + dit.w.t2_w.len()
            + dit.w.t2_b.len()
            + dit.w.fadaln_w.len()
            + dit.w.fadaln_b.len()
            + dit.w.flin_w.len()
            + dit.w.flin_b.len();
        // Exact equality (not a bound) is the real proof: t_embedder's MLP
        // alone is 1024*256 = 262,144 elements regardless of `tiny_cfg`'s
        // tiny `dim` (it's a fixed-size sinusoidal-embedding MLP in the real
        // model too), so a size THRESHOLD can't distinguish "the 13 wrapper
        // tensors" from "one wrapper tensor plus a leaked block tensor" here
        // — only the precise sum can. If HostWeights ever retained even one
        // `layers.*`/`noise_refiner.*`/`context_refiner.*` key, `retained`
        // would be at least `expected + HUGE`, off by exactly one block
        // tensor's size, not off by a little.
        assert_eq!(retained, expected, "HostWeights must hold EXACTLY the 13 wrapper tensors' elements, nothing from any block");
    }
}
