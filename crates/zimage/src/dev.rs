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

// ---------------- fp32 windowed (single-GPU streaming) DiT ----------------

/// One windowed stage: a *fixed* pool of `budget` block-shaped slots, bound
/// by a [`weightset::WeightSet`] to whichever blocks are resident right now
/// — unlike [`Phase`], which uploads every block once and chains them into
/// one static graph, this re-binds and re-submits one block at a time, so
/// the device footprint is `budget` blocks' worth of weights regardless of
/// how many blocks the model actually has. This is what lets the fp32 DiT
/// (structurally too large to fit one GPU whole) run on one GPU at all —
/// the trade is `n_groups` submits per forward instead of one (see
/// docs/lessons.md and the design plan's Risk #5: unmeasured here, a later
/// concern if it turns out to dominate).
struct WindowedPhase {
    input: DeviceBuffer,
    resb: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    scr: Scratch,
    slot_w: Vec<BlockWeights>,
    slot_n: Vec<NormBufs>,
    ws: weightset::WeightSet,
    prefixes: Vec<String>,
    bd: BlockDims,
    t: u32,
    reg_gemm: bool,
}

/// Build a windowed stage and seed every *pinned* slot's data from `src`
/// (the build-time source) — `WeightSet` itself never touches device memory
/// (see its doc), so the caller must load a slot's data the moment
/// `WeightSet` assigns it a group, and the initial pin is exactly such an
/// assignment even though it never surfaces as an `advance` miss.
#[allow(clippy::too_many_arguments)]
fn build_windowed_phase(gpu: &Gpu, src: &dyn checkpoint::TensorSource, prefixes: Vec<String>, bd: BlockDims, t: u32, modulation: bool, reg_gemm: bool, window: u32) -> WindowedPhase {
    let half = bd.head_dim / 2;
    let input = gpu.storage((t * bd.dim) as u64);
    let resb = gpu.storage((t * bd.dim) as u64);
    let cos = gpu.storage((t * half) as u64);
    let sin = gpu.storage((t * half) as u64);
    let scr = Scratch::new_maybe_flash(gpu, bd, t, reg_gemm && crate::block::use_flash(gpu, bd.n_heads, t));
    let n_groups = prefixes.len() as u32;
    let budget = window.clamp(1, n_groups.max(1));
    let slot_w: Vec<BlockWeights> = (0..budget).map(|_| BlockWeights::alloc(gpu, bd)).collect();
    let mut slot_n: Vec<NormBufs> = (0..budget).map(|_| NormBufs::alloc(gpu, bd.dim, modulation)).collect();
    let sched = weightset::Schedule::cyclic(n_groups, 1);
    let ws = weightset::WeightSet::build(n_groups, budget, sched, Box::new(weightset::CyclicScan { lookahead: 1 }))
        .expect("build_windowed_phase: window is clamped to [1, n_groups] above, so build() cannot fail");
    for (i, slot) in ws.slot_contents().iter().enumerate() {
        if let Some(g) = slot {
            slot_w[i].load_into(gpu, src, &prefixes[g.0 as usize]);
            slot_n[i].reload_host(src, &prefixes[g.0 as usize]);
        }
    }
    gpu.poll_wait();
    WindowedPhase { input, resb, cos, sin, scr, slot_w, slot_n, ws, prefixes, bd, t, reg_gemm }
}

impl WindowedPhase {
    /// One forward pass over every block, in order. `src` is a *fresh* open
    /// of the checkpoint for this call — a rotating slot's miss reloads
    /// straight from it (a block's weights are streamed on demand; the pin
    /// from `build_windowed_phase` still never re-reads anything). Reused
    /// across many `forward()` calls: the pinned prefix is loaded exactly
    /// once, ever; only the unpinned tail re-streams, once per call.
    fn run(&mut self, gpu: &Gpu, src: &dyn checkpoint::TensorSource, tokens: &[f32], c: &[f32], cos: &[f32], sin: &[f32], dim: usize, cdim: usize) -> Vec<f32> {
        wf(gpu, &self.input, tokens);
        wf(gpu, &self.cos, cos);
        wf(gpu, &self.sin, sin);
        let (mut cur_in, mut cur_out) = (self.input.clone(), self.resb.clone());
        for cursor in 0..self.prefixes.len() {
            let (slot, miss) = self.ws.advance(cursor);
            let idx = slot.0 as usize;
            if miss {
                self.slot_w[idx].load_into(gpu, src, &self.prefixes[cursor]);
                self.slot_n[idx].reload_host(src, &self.prefixes[cursor]);
                gpu.poll_wait();
            }
            self.slot_n[idx].upload_folded(gpu, c, dim, cdim);
            let mut steps = Vec::new();
            build_block_steps(gpu, &mut steps, &self.slot_w[idx], &self.slot_n[idx], &cur_in, &cur_out, &self.scr, &self.cos, &self.sin, self.bd, self.t, self.reg_gemm);
            gpu.submit(&[], &steps);
            std::mem::swap(&mut cur_in, &mut cur_out);
        }
        gpu.read(&cur_in, self.t as usize * dim)
    }

    /// Reload count so far — the churn number the whole design exists to
    /// bound. Exposed for observability (`docs/models/zimage/status.md`,
    /// `braintop`), not for correctness.
    fn reloads(&self) -> u64 {
        self.ws.reloads()
    }
}

/// Z-Image DiT with the main (largest) layer stack **weight-windowed**
/// instead of fully resident — the single-GPU fp32 path. fp32's 6B params
/// (~24 GB) do not fit this crate's other single-GPU engine
/// ([`ZImageDitI8`], which fits because int8 is ~4× smaller) nor this box's
/// device budget as one blob; [`ZImageDitShard`] works around that by
/// splitting across two GPUs. This is the other way to make fp32 fit one
/// GPU: keep only `window` blocks' worth of weights resident at once and
/// stream the rest from disk per forward, per the design in
/// `docs/models/zimage/status.md`.
///
/// The noise/context refiners stay fully resident ([`Phase`], as in
/// [`ZImageDit`]) — they are a handful of layers, not the 6B-parameter
/// bottleneck; only `main` (the `n_layers`-deep stack) is windowed.
pub struct ZImageDitWindowed {
    gpu: Gpu,
    cfg: ZImageConfig,
    w: HostWeights,
    f: u32,
    h: u32,
    wd: u32,
    cap_len: u32,
    noise: Phase,
    context: Phase,
    // RefCell, not `&mut self`: WeightSet's slot bookkeeping mutates on
    // every forward, but `forward` keeps the same `&self` signature every
    // other DiT engine here has (ZImageDit/ZImageDitI8/ZImageDitShard), so
    // DitEngine::forward in pipeline.rs needs no ripple to accommodate this
    // one variant. Single-threaded per instance (one forward at a time,
    // already true of every engine here), so this can never actually
    // contend.
    main: std::cell::RefCell<WindowedPhase>,
}

impl ZImageDitWindowed {
    /// `window`: how many of `main`'s `n_layers` blocks stay resident at
    /// once. `window >= n_layers` degenerates to fully resident (every block
    /// pinned, `main.reloads()` stays `0` forever) — the same behaviour
    /// [`ZImageDit`] gives, bit-for-bit, at a higher fixed cost (a submit
    /// per block instead of one for the whole stack). `src` builds the
    /// refiners and `HostWeights` (as any other engine here does); `dit_src`
    /// is a factory the windowed stage calls to get a *fresh* streaming
    /// source on every `forward()` — a rotating slot's miss must be able to
    /// re-read the checkpoint an unbounded number of calls after `build`
    /// returns, long after `src`'s borrow here has ended.
    #[allow(clippy::too_many_arguments)]
    pub fn build_from_source(cfg: ZImageConfig, src: &dyn checkpoint::TensorSource, window: u32, f: u32, h: u32, wd: u32, cap_len: u32, device: Option<&str>) -> ZImageDitWindowed {
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
        let main = build_windowed_phase(&gpu, src, mp, bd, ntot, true, reg_gemm, window);
        let w = HostWeights::from_source(&cfg, src);
        ZImageDitWindowed { gpu, cfg, w, f, h, wd, cap_len, noise, context, main: std::cell::RefCell::new(main) }
    }

    /// `dit_src` must yield a `TensorSource` reading the SAME checkpoint
    /// `build_from_source` did — a fresh one each call is expected (e.g. a
    /// new mmap `WeightReader` open, cheap: header-only, no tensor bytes
    /// read until a miss actually needs one).
    pub fn forward(&self, dit_src: &dyn checkpoint::TensorSource, latent: &[f32], cap: &[f32], t: f32) -> Vec<f32> {
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
        let uni_out = self.main.borrow_mut().run(&self.gpu, dit_src, &uni, &cvec, &uni_cos, &uni_sin, dim, cdim);
        postprocess(c, &self.w, &uni_out, &cvec, pre.n_img, self.f, self.h, self.wd)
    }

    /// Reload count of the windowed `main` stage so far — `0` forever when
    /// `window >= n_layers`; observability for the churn claim.
    pub fn main_reloads(&self) -> u64 {
        self.main.borrow().reloads()
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

    /// Deterministic, non-trivial fill (xorshift64, range roughly [-1, 1)) —
    /// a bit-identical comparison is only a real proof if the two engines
    /// are fed genuinely varying data; an all-same-value fixture could pass
    /// by accident (e.g. a block-index bug that silently reused block 0's
    /// weights for every block would still "match" on constant inputs).
    fn filled(seed: u64, n: usize) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x % 2000) as f32 / 1000.0) - 1.0
            })
            .collect()
    }

    /// Every tensor `tiny_cfg`'s DiT needs, filled with distinct deterministic
    /// data (a different seed per tensor, derived from insertion order) —
    /// shared by the windowed-vs-resident bit-identical test below.
    fn full_tiny_tensors(cfg: &ZImageConfig) -> Tensors {
        let dim = cfg.dim as usize;
        let cdim = dim.min(256);
        let (ps, pf) = (cfg.patch_size, cfg.f_patch_size);
        let patch_dim = (pf * ps * ps * cfg.in_channels) as usize;
        let mut t: Tensors = Tensors::new();
        let mut seed = 0u64;
        let mut insert = |t: &mut Tensors, name: String, n: usize| {
            seed += 1;
            t.insert(name, (vec![n], filled(seed, n)));
        };

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

        let bd = cfg.block_dims();
        let (blockdim, hidden, head_dim) = (bd.dim as usize, bd.hidden as usize, bd.head_dim as usize);
        let prefixes: Vec<(String, bool)> = (0..cfg.n_refiner_layers)
            .map(|l| (format!("noise_refiner.{l}"), true))
            .chain((0..cfg.n_refiner_layers).map(|l| (format!("context_refiner.{l}"), false)))
            .chain((0..cfg.n_layers).map(|l| (format!("layers.{l}"), true)))
            .collect();
        for (prefix, modulation) in prefixes {
            insert(&mut t, format!("{prefix}.attention.to_q.weight"), blockdim * blockdim);
            insert(&mut t, format!("{prefix}.attention.to_k.weight"), blockdim * blockdim);
            insert(&mut t, format!("{prefix}.attention.to_v.weight"), blockdim * blockdim);
            insert(&mut t, format!("{prefix}.attention.to_out.0.weight"), blockdim * blockdim);
            insert(&mut t, format!("{prefix}.attention.norm_q.weight"), head_dim);
            insert(&mut t, format!("{prefix}.attention.norm_k.weight"), head_dim);
            insert(&mut t, format!("{prefix}.feed_forward.w1.weight"), hidden * blockdim);
            insert(&mut t, format!("{prefix}.feed_forward.w2.weight"), blockdim * hidden);
            insert(&mut t, format!("{prefix}.feed_forward.w3.weight"), hidden * blockdim);
            insert(&mut t, format!("{prefix}.attention_norm1.weight"), blockdim);
            insert(&mut t, format!("{prefix}.attention_norm2.weight"), blockdim);
            insert(&mut t, format!("{prefix}.ffn_norm1.weight"), blockdim);
            insert(&mut t, format!("{prefix}.ffn_norm2.weight"), blockdim);
            if modulation {
                insert(&mut t, format!("{prefix}.adaLN_modulation.0.weight"), 4 * blockdim * cdim);
                insert(&mut t, format!("{prefix}.adaLN_modulation.0.bias"), 4 * blockdim);
            }
        }
        t
    }

    /// The whole point of a weight *window*: residency is a pure memory
    /// placement decision, never a numerical one. A windowed DiT with
    /// `window=1` (strictly narrower than `tiny_cfg`'s 2 main layers, so
    /// this genuinely exercises eviction+reload, not just the degenerate
    /// fully-pinned case) must produce a forward output IDENTICAL — bit for
    /// bit, not merely close — to the fully-resident `ZImageDit` fed the
    /// exact same weights. `assert_eq!` on `Vec<f32>`, not a cosine/PSNR
    /// bound: any mismatch here is a real bug (wrong block loaded into a
    /// slot, a stale slot reused, a residual buffer mixed up across
    /// submits), not floating-point drift, since both paths run the
    /// identical kernels over identical data.
    #[test]
    fn windowed_dit_matches_fully_resident_dit_bit_for_bit_when_window_is_narrower_than_the_model() {
        let cfg = tiny_cfg();
        let t = full_tiny_tensors(&cfg);
        let (f, h, wd, cap_len) = (1, 4, 4, 4);
        let latent_n = (cfg.in_channels * f * h * wd) as usize;
        let latent = filled(9001, latent_n);
        let cap = filled(9002, cap_len as usize * cfg.cap_feat_dim as usize);

        let resident = ZImageDit::build(cfg.clone(), t.clone(), f, h, wd, cap_len, Some("cpu"));
        let windowed = ZImageDitWindowed::build_from_source(cfg.clone(), &t, 1, f, h, wd, cap_len, Some("cpu"));

        let want = resident.forward(&latent, &cap, 0.5);
        let got = windowed.forward(&t, &latent, &cap, 0.5);
        assert_eq!(got, want);

        // window=1 against 2 main layers must actually reload (block 1 is
        // never pinned) -- otherwise this test would pass vacuously even if
        // windowing were silently disabled.
        assert!(windowed.main_reloads() > 0, "window narrower than the model must reload at least once");
    }

    /// `window >= n_layers` must degenerate to exactly the fully-resident
    /// engine's behaviour: every block pinned at build, zero reloads ever,
    /// for any number of forward calls.
    #[test]
    fn windowed_dit_with_window_at_least_the_model_never_reloads() {
        let cfg = tiny_cfg();
        let t = full_tiny_tensors(&cfg);
        let (f, h, wd, cap_len) = (1, 4, 4, 4);
        let latent_n = (cfg.in_channels * f * h * wd) as usize;
        let latent = filled(9001, latent_n);
        let cap = filled(9002, cap_len as usize * cfg.cap_feat_dim as usize);

        let windowed = ZImageDitWindowed::build_from_source(cfg.clone(), &t, cfg.n_layers, f, h, wd, cap_len, Some("cpu"));
        for _ in 0..3 {
            windowed.forward(&t, &latent, &cap, 0.5);
        }
        assert_eq!(windowed.main_reloads(), 0);
    }
}
