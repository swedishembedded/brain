// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5's two real-weight **latent upscalers** (spatial x2, temporal x2) -
//! small conv/resblock nets operating directly in VIDEO latent space (128
//! channels), no timestep or conditioning of any kind: `forward(latent) ->
//! latent`. Ported from `ltx_core.model.upsampler.{model,res_block,
//! pixel_shuffle,spatial_rational_resampler}`
//! (`scratchpad/reference/ltxv/packages/ltx-core/src/ltx_core/model/upsampler/`),
//! real weights (`ltx-2.5-latent-{spatial,temporal}-upscaler-x2-bf16-1.0.
//! safetensors`), real parity (`crates/ltxv/tests/upsampler_parity.rs`).
//!
//! Both real checkpoints carry `dims: 3` - `initial_conv`/`res_blocks`/
//! `post_upsample_res_blocks`/`final_conv` are ALL real `nn.Conv3d` for
//! BOTH the spatial and the temporal upscaler, never `nn.Conv2d`. Only the
//! middle `upsampler` stage differs by mode - see [`UpsamplerMode`].
//!
//! # `ResBlock`'s UNUSUAL op order - verified against source, not assumed
//!
//! `res_block.py`'s `ResBlock.forward`:
//! ```text
//! residual = x
//! x = conv1(x); x = norm1(x); x = activation(x)      # SiLU BEFORE conv2
//! x = conv2(x); x = norm2(x)
//! x = activation(x + residual)                        # SiLU AFTER the add
//! ```
//! The activation after conv2/norm2 is applied to `x + residual`, not to `x`
//! alone - the residual add happens BEFORE the final SiLU, not after it (the
//! opposite of the video VAE's `resnet_block` in `vae3d.rs`, whose SiLUs both
//! precede their conv). See [`res_block`].
//!
//! # `GroupNorm(32, mid_channels)` at torch's DEFAULT eps (1e-5)
//!
//! Confirmed empirically off the built reference module
//! (`initial_norm.num_groups == 32`, `initial_norm.eps == 1e-5`, asserted by
//! `tools/goldens/ltxv_upsampler_dump_reference.py`) - `model.py` never
//! passes an explicit `eps` to `torch.nn.GroupNorm`. This is a DIFFERENT
//! call site than every other norm this crate/`vae::blocks3d` has ported so
//! far: the video VAE's `PixelNorm` (zero-parameter, eps 1e-8) and the audio
//! VAE's `PixelNorm` (zero-parameter, eps 1e-6) are both channel-RMS with no
//! learnable gain; this is a REAL, LEARNABLE, GAIN-AND-BIAS GroupNorm, the
//! same op `vae::blocks::Builder::gn`/`gn_gb` already dispatch for the 2D
//! `AutoencoderKL` family - just not previously wired into the 3D builder.
//! Reused here as three EXTRA kernel slots appended to
//! [`vae::blocks3d::KERNELS`] via [`vae::blocks3d::kernels_with`] (the same
//! `NEXT_SLOT`-based extension idiom `crates/sdxlunet`/`crates/vqgan` already
//! use for the 2D builder) - zero new WGSL, `gn_part`/`gn_stats2`/`gn_apply`
//! already exist and are exactly shape-generic enough (every GN kernel's
//! `Params` only ever uses the `H*W` PRODUCT, never `H`/`W` separately) to
//! normalise over `T*H*W` unchanged. See [`group_norm`].
//!
//! # The per-frame `Conv2d` + `PixelShuffleND(2)` IS `Builder3d`'s existing
//! # `depth_to_space` at `pt=1` - no reshape, no new kernel
//!
//! The spatial (non-rational) branch's `upsampler` is, in the reference,
//! `rearrange('b c f h w -> (b f) c h w')` -> `Conv2d(mid,4*mid,3,pad=1)` ->
//! `PixelShuffleND(2)` -> `rearrange('(b f) c h w -> b c f h w')`. Two facts
//! make this collapse to existing `vae::blocks3d::Builder3d` primitives with
//! NO batch-folding reshape at all:
//!
//! 1. A `kt=1` `Conv3d` (already [`vae::blocks3d::Conv3d::spatial`]) computes
//!    exactly the same per-frame values as the reshape+`Conv2d`+reshape-back
//!    dance - folding frames into a batch axis vs. leaving them as the `T`
//!    axis of a `kt=1` conv is a memory-LAYOUT difference, not a math one.
//! 2. `PixelShuffleND(2)`'s exact rearrange, `'b (c p1 p2) h w -> b c (h p1)
//!    (w p2)'`, is channel-outer, height-then-width - which is *literally*
//!    [`vae::blocks3d::Builder3d::depth_to_space`]'s own documented
//!    convention (`'b (c p1 p2 p3) d h w -> b c (d p1)(h p2)(w p3)'`) with
//!    `p1 = 1` (the depth/frame factor collapses to a no-op fold, leaving
//!    just `(c p2 p3)` over height/width - exactly `PixelShuffleND(2)`).
//!
//! So `upsample_spatial`'s middle stage is `conv(Conv3d::spatial(3,1,1))`
//! then `depth_to_space(x, 1, 2, 2)`, entirely inside the `[C,T,H,W]`
//! representation the rest of this model already uses. **This is why the
//! golden dumper explicitly rearranges its own `self.upsampler` tap back to
//! `[C,T,H,W]` before saving it** (`tools/goldens/
//! ltxv_upsampler_dump_reference.py`'s `Taps.watch_reshaped`) - the
//! reference computes the same tensor in `[F,C,H,W]` order internally, and
//! comparing raw would silently compare transposed data.
//!
//! `PixelShuffleND(1)` (the temporal-only branch's `upsampler`, no reshape
//! ever) is `depth_to_space(x, 2, 1, 1)` directly - `p1` on the depth axis
//! this time, `p2 = p3 = 1` no-ops on height/width. See [`upsample_temporal`].
//!
//! # `nn.Conv3d(..., padding=1)` is SYMMETRIC ZERO pad, not causal REPLICATE
//!
//! Every 3x3x3 conv in this model (`initial_conv`, both `ResBlock` convs,
//! `final_conv`, and the temporal upscaler's own `upsampler.0`) is a plain
//! `torch.nn.Conv3d(kernel_size=3, padding=1)` - ordinary SYMMETRIC pad,
//! zeros, on all three axes, the opposite of `vae3d.rs`'s `causal_conv3`
//! (one-sided, REPLICATED first/last frame). `conv3d.wgsl`'s `pt` parameter
//! is DELIBERATELY one-sided-only (see that kernel's header comment) - there
//! is no way to ask it for symmetric time padding directly - so
//! [`zero_pad_conv3`] pads the time axis by hand with
//! [`vae::blocks3d::Builder3d::zeros`] + two [`vae::blocks3d::Builder3d::time_cat`]
//! calls (one zero frame prepended, one appended) before calling `conv` with
//! `pt = 0`; height/width use the kernel's own built-in symmetric `ph =
//! pw = 1` unchanged. The per-frame spatial-upsampler conv (`Conv3d::spatial`,
//! `kt = 1`) needs none of this - a `kt=1` conv never reads a neighbouring
//! frame, so there is nothing to pad on that axis.

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use vae::blocks::Tensors;
use vae::blocks3d::{kernels_with, Builder3d, Conv3d, T3, NEXT_SLOT};

/// Extra kernel slots this module appends to [`vae::blocks3d::KERNELS`] for
/// [`group_norm`] - `gn_part`/`gn_stats2`/`gn_apply` already exist (this
/// crate's own `Cargo.toml` already depends on `brain-kernels`; every other
/// `ltxv` module reaches its WGSL sources the same way, e.g. `audio_vae.rs`'s
/// `kernels::PAD2D`) and are reused UNCHANGED, just not previously wired into
/// the 3D builder. The "no workgroup reductions" two-stage path
/// (`gn_part`+`gn_stats2`, not `gn_stats_wg`) is used unconditionally rather
/// than threading `DeviceCaps` through - this model's tensors are tiny
/// (at most 1024 channels x a few thousand spatial positions), so the
/// cooperative fast path buys nothing worth the extra plumbing.
const K_GN_PART: usize = NEXT_SLOT;
const K_GN_STATS2: usize = NEXT_SLOT + 1;
const K_GN_APPLY: usize = NEXT_SLOT + 2;
const N_KERNELS: usize = NEXT_SLOT + 3;

/// Partials per group for the two-stage GroupNorm reduction - the same value
/// `vae::blocks::Builder`'s own `gn_gb` uses (`GN_P` there, private to that
/// module, so restated here rather than imported).
const GN_P: u32 = 64;

/// This module's kernel set: [`vae::blocks3d::KERNELS`] plus the three
/// GroupNorm slots above, via [`kernels_with`] - the exact idiom
/// `crates/sdxlunet`/`crates/vqgan` already use for the 2D builder
/// (`vae::blocks::kernels_with`); `vae::blocks3d::Builder3d` supports the
/// same extension.
pub const KERNELS: [(&str, &str); N_KERNELS] = kernel_set();

const fn kernel_set() -> [(&'static str, &'static str); N_KERNELS] {
    let mut k = kernels_with::<N_KERNELS>();
    k[K_GN_PART] = ("gn_part", kernels::GN_PART);
    k[K_GN_STATS2] = ("gn_stats2", kernels::GN_STATS2);
    k[K_GN_APPLY] = ("gn_apply", kernels::GN_APPLY);
    k
}

/// `GroupNorm`'s own default eps (torch), NOT this checkpoint family's
/// `PixelNorm` eps (1e-8 video / 1e-6 audio) - see this module's doc.
pub const GROUP_NORM_EPS: f32 = 1e-5;
const GROUPS: u32 = 32;

/// Which real checkpoint's forward shape this config builds - the two real
/// LTX-2.5 upscalers share every field except which axis the `upsampler`
/// stage grows, and the mid-channel width
/// `LatentUpsamplerConfig::mid_channels` - not enough divergence to justify
/// two separate config structs the way e.g. `LtxVaeConfig`'s encoder/decoder
/// block LISTS differ block-by-block; one struct with a mode flag mirrors
/// how `LtxAvDitConfig` extends `LtxDitConfig` by field, not by type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpsamplerMode {
    /// `spatial_upsample=true, temporal_upsample=false,
    /// rational_resampler=false` - `upsampler` is `Conv3d::spatial(3,1,1)`
    /// + `depth_to_space(x,1,2,2)`.
    Spatial,
    /// `spatial_upsample=false, temporal_upsample=true` (the checkpoint's own
    /// `rational_resampler: true` is dead - the `elif temporal_upsample`
    /// branch never reads it) - `upsampler` is a full 3x3x3 `Conv3d` (manual
    /// symmetric zero pad, see [`zero_pad_conv3`]) + `depth_to_space(x,2,1,1)`,
    /// then the first output frame is dropped.
    Temporal,
}

/// Real LTX-2.5 `LatentUpsampler` config - transcribed from both checkpoints'
/// embedded `config` metadata (`_class_name: "LatentUpsampler"`), cross-checked
/// against the 72-tensor headers of both real files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatentUpsamplerConfig {
    pub in_channels: u32,
    pub mid_channels: u32,
    pub num_blocks_per_stage: u32,
    pub mode: UpsamplerMode,
}

impl LatentUpsamplerConfig {
    /// `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`'s real config.
    pub fn spatial_x2() -> LatentUpsamplerConfig {
        LatentUpsamplerConfig { in_channels: 128, mid_channels: 1024, num_blocks_per_stage: 4, mode: UpsamplerMode::Spatial }
    }

    /// `ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors`'s real config.
    pub fn temporal_x2() -> LatentUpsamplerConfig {
        LatentUpsamplerConfig { in_channels: 128, mid_channels: 512, num_blocks_per_stage: 4, mode: UpsamplerMode::Temporal }
    }

    /// Every tensor this model reads, in the checkpoint's own (bare, no
    /// prefix) name space - 72 tensors for either real config:
    /// `initial_conv` + `initial_norm` (4) + `res_blocks.{0..n}` (8 each) +
    /// `post_upsample_res_blocks.{0..n}` (8 each) + `upsampler.0` (2) +
    /// `final_conv` (2).
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let (ic, mc, n) = (self.in_channels as usize, self.mid_channels as usize, self.num_blocks_per_stage);
        let mut m: Vec<(String, Vec<usize>)> = Vec::new();
        let conv3 = |m: &mut Vec<(String, Vec<usize>)>, name: &str, cout: usize, cin: usize| {
            m.push((format!("{name}.weight"), vec![cout, cin, 3, 3, 3]));
            m.push((format!("{name}.bias"), vec![cout]));
        };
        let gn = |m: &mut Vec<(String, Vec<usize>)>, name: &str, c: usize| {
            m.push((format!("{name}.weight"), vec![c]));
            m.push((format!("{name}.bias"), vec![c]));
        };

        conv3(&mut m, "initial_conv", mc, ic);
        gn(&mut m, "initial_norm", mc);
        for i in 0..n {
            conv3(&mut m, &format!("res_blocks.{i}.conv1"), mc, mc);
            gn(&mut m, &format!("res_blocks.{i}.norm1"), mc);
            conv3(&mut m, &format!("res_blocks.{i}.conv2"), mc, mc);
            gn(&mut m, &format!("res_blocks.{i}.norm2"), mc);
        }
        match self.mode {
            // `nn.Conv2d(mid, 4*mid, 3, pad=1)` - a 2D weight `[4mid,mid,3,3]`,
            // confirmed against the real header (no depth axis).
            UpsamplerMode::Spatial => {
                m.push(("upsampler.0.weight".into(), vec![4 * mc, mc, 3, 3]));
                m.push(("upsampler.0.bias".into(), vec![4 * mc]));
            }
            // `nn.Conv3d(mid, 2*mid, 3, pad=1)`.
            UpsamplerMode::Temporal => {
                m.push(("upsampler.0.weight".into(), vec![2 * mc, mc, 3, 3, 3]));
                m.push(("upsampler.0.bias".into(), vec![2 * mc]));
            }
        }
        for i in 0..n {
            conv3(&mut m, &format!("post_upsample_res_blocks.{i}.conv1"), mc, mc);
            gn(&mut m, &format!("post_upsample_res_blocks.{i}.norm1"), mc);
            conv3(&mut m, &format!("post_upsample_res_blocks.{i}.conv2"), mc, mc);
            gn(&mut m, &format!("post_upsample_res_blocks.{i}.norm2"), mc);
        }
        conv3(&mut m, "final_conv", ic, mc);
        m
    }
}

// ---------------------------------------------------------------- blocks

/// `nn.Conv3d(cin, cout, 3, padding=1)`: SYMMETRIC zero pad on all three
/// axes. Height/width use `Conv3d`'s own built-in `ph=pw=1`; time is padded
/// by hand (one zero frame each side, via [`Builder3d::zeros`] +
/// [`Builder3d::time_cat`]) because `conv3d.wgsl`'s `pt` is one-sided-only -
/// see this module's doc.
fn zero_pad_conv3(b: &mut Builder3d, prefix: &str, cout: u32, x: &T3) -> T3 {
    let zero = b.zeros(x.c, 1, x.h, x.w);
    let a = b.time_cat(&zero, x);
    let padded = b.time_cat(&a, &zero);
    b.free(a);
    let spec = Conv3d { kt: 3, kh: 3, kw: 3, st: 1, sh: 1, sw: 1, pt: 0, ph: 1, pw: 1 };
    let y = b.conv(prefix, cout, spec, &padded);
    b.free(padded);
    y
}

/// Static affine `GroupNorm(32, C)` at [`GROUP_NORM_EPS`] - see this module's
/// doc for why the existing `gn_part`/`gn_stats2`/`gn_apply` kernels (added
/// to this module's own [`KERNELS`] table) are shape-generic enough to reuse
/// unchanged for a `[C,T,H,W]` volume (`H*W` in their `Params` is read only
/// as a product, so passing `T*H*W`/`1` works exactly like `vae::blocks::
/// Builder::gn_gb`'s own `H`/`W` would for a batch of 2D images).
fn group_norm(b: &mut Builder3d, prefix: &str, x: &T3) -> T3 {
    let c = x.c;
    let thw = x.t * x.h * x.w;
    let mut gb = b.get(&format!("{prefix}.weight")).1.clone();
    let beta = &b.get(&format!("{prefix}.bias")).1;
    assert_eq!(gb.len(), c as usize, "{prefix}.weight: {} values for {c} channels", gb.len());
    assert_eq!(beta.len(), c as usize, "{prefix}.bias: {} values for {c} channels", beta.len());
    gb.extend_from_slice(beta);
    let gbuf = b.dev_owned(&format!("{prefix}.gb"), &gb);

    let gpu = b.gpu();
    let part = b.act(2 * GROUPS as u64 * GN_P as u64);
    b.push_step(gpu.step(K_GN_PART, &[&x.buf, &part], &[1, c, thw, 1, GROUPS, GN_P], GROUPS * GN_P));
    let stats = b.act(2 * GROUPS as u64);
    b.push_step(gpu.step(K_GN_STATS2, &[&part, &stats], &[1, c, thw, 1, GROUPS, GN_P, f(GROUP_NORM_EPS)], GROUPS));
    let y_buf = b.act(x.len());
    b.push_step(gpu.step(K_GN_APPLY, &[&x.buf, &stats, &gbuf, &y_buf], &[1, c, thw, 1, GROUPS], c * thw));
    T3 { buf: y_buf, c: x.c, t: x.t, h: x.h, w: x.w }
}

/// `ResBlock.forward` - see this module's doc for the UNUSUAL op order
/// (activation AFTER the residual add).
fn res_block(b: &mut Builder3d, prefix: &str, c: u32, x: &T3) -> T3 {
    let c1 = zero_pad_conv3(b, &format!("{prefix}.conv1"), c, x);
    let n1 = group_norm(b, &format!("{prefix}.norm1"), &c1);
    b.free(c1);
    let a1 = b.silu(&n1);
    b.free(n1);
    let c2 = zero_pad_conv3(b, &format!("{prefix}.conv2"), c, &a1);
    b.free(a1);
    let n2 = group_norm(b, &format!("{prefix}.norm2"), &c2);
    b.free(c2);
    let summed = b.add(x, &n2);
    b.free(n2);
    let out = b.silu(&summed);
    b.free(summed);
    out
}

fn initial_stage(b: &mut Builder3d, cfg: &LatentUpsamplerConfig, x: &T3) -> T3 {
    let c0 = zero_pad_conv3(b, "initial_conv", cfg.mid_channels, x);
    b.tap("initial_conv", &c0);
    let n0 = group_norm(b, "initial_norm", &c0);
    b.free(c0);
    b.tap("initial_norm", &n0);
    let a0 = b.silu(&n0);
    b.free(n0);
    b.tap("initial_activation", &a0);
    a0
}

fn res_stage(b: &mut Builder3d, prefix: &str, n: u32, c: u32, x: T3) -> T3 {
    let mut cur = x;
    for i in 0..n {
        let name = format!("{prefix}.{i}");
        let next = res_block(b, &name, c, &cur);
        b.free(cur);
        b.tap(&name, &next);
        cur = next;
    }
    cur
}

/// The `upsampler` middle stage - see this module's doc for why both modes
/// reduce to a plain `Builder3d` conv + `depth_to_space`, no reshape.
fn upsampler_stage(b: &mut Builder3d, cfg: &LatentUpsamplerConfig, x: T3) -> T3 {
    let mc = cfg.mid_channels;
    match cfg.mode {
        UpsamplerMode::Spatial => {
            let conv = b.conv("upsampler.0", 4 * mc, Conv3d::spatial(3, 1, 1), &x);
            b.free(x);
            let shuffled = b.depth_to_space(&conv, 1, 2, 2);
            b.free(conv);
            b.tap("upsampler", &shuffled);
            shuffled
        }
        UpsamplerMode::Temporal => {
            let conv = zero_pad_conv3(b, "upsampler.0", 2 * mc, &x);
            b.free(x);
            let shuffled = b.depth_to_space(&conv, 2, 1, 1);
            b.free(conv);
            b.tap("upsampler", &shuffled);
            // `x = x[:, :, 1:, :, :]` - drop the first (padding) frame.
            let dropped = b.time_slice(&shuffled, 1, shuffled.t - 1);
            b.free(shuffled);
            dropped
        }
    }
}

fn new_gpu(device: Option<&str>) -> Gpu {
    match device {
        Some("cpu") => Gpu::new_cpu(&KERNELS),
        Some("gpu") | Some("wgpu") => Gpu::new_wgpu(&KERNELS),
        _ => Gpu::new(&KERNELS),
    }
}

fn read_named(gpu: &Gpu, v: &[(String, DeviceBuffer, usize)], name: &str) -> Option<Vec<f32>> {
    v.iter().find(|(n, _, _)| n == name).map(|(_, b, l)| gpu.read(b, *l))
}

/// One latent upscaler graph, built once for a fixed input shape.
pub struct LatentUpsampler {
    gpu: Gpu,
    steps: Vec<Step>,
    x_in: DeviceBuffer,
    in_len: usize,
    out: DeviceBuffer,
    out_len: usize,
    out_shape: (u32, u32, u32, u32),
    taps: Vec<(String, DeviceBuffer, usize)>,
}

impl LatentUpsampler {
    /// Build the forward graph for a `[cfg.in_channels, t, h, w]` latent.
    pub fn build(cfg: &LatentUpsamplerConfig, tensors: &Tensors, t: u32, h: u32, w: u32, device: Option<&str>) -> LatentUpsampler {
        let gpu = new_gpu(device);
        let mut b = Builder3d::new(&gpu, tensors, true);

        let x_in = gpu.storage(cfg.in_channels as u64 * t as u64 * h as u64 * w as u64);
        let input = T3 { buf: x_in.clone(), c: cfg.in_channels, t, h, w };
        b.tap("input", &input);

        let init = initial_stage(&mut b, cfg, &input);
        let pre = res_stage(&mut b, "res_blocks", cfg.num_blocks_per_stage, cfg.mid_channels, init);
        let up = upsampler_stage(&mut b, cfg, pre);
        let post = res_stage(&mut b, "post_upsample_res_blocks", cfg.num_blocks_per_stage, cfg.mid_channels, up);
        let out = zero_pad_conv3(&mut b, "final_conv", cfg.in_channels, &post);
        b.free(post);
        b.tap("final_conv", &out);
        b.tap("output", &out);

        let out_len = out.len() as usize;
        let out_shape = (out.c, out.t, out.h, out.w);
        let (steps, taps) = b.finish();
        LatentUpsampler {
            gpu,
            steps,
            x_in,
            in_len: (cfg.in_channels * t * h * w) as usize,
            out: out.buf,
            out_len,
            out_shape,
            taps,
        }
    }

    /// Run the forward: `latent` is `[cfg.in_channels, t, h, w]` row-major.
    pub fn upsample(&self, latent: &[f32]) -> Vec<f32> {
        assert_eq!(latent.len(), self.in_len, "upsample: {} values, expected {}", latent.len(), self.in_len);
        self.gpu.write_f32(&self.x_in, latent);
        self.gpu.submit(&[], &self.steps);
        self.gpu.read(&self.out, self.out_len)
    }

    /// `(c, t, h, w)` of the output this graph produces.
    pub fn out_shape(&self) -> (u32, u32, u32, u32) {
        self.out_shape
    }

    /// A named tap of the last [`LatentUpsampler::upsample`] call - `input`,
    /// `initial_conv`, `initial_norm`, `initial_activation`,
    /// `res_blocks.{i}`, `upsampler`, `post_upsample_res_blocks.{i}`,
    /// `final_conv` (== `output`).
    pub fn read_tap(&self, name: &str) -> Option<Vec<f32>> {
        read_named(&self.gpu, &self.taps, name)
    }
}

/// `ltx_core.model.upsampler.model.upsample_video`: run `ups` over a
/// DIFFUSION-space latent, un-normalizing into the VAE's own latent space on
/// the way in and re-normalizing on the way out.
///
/// # Why this wrapper is not optional
///
/// Both upscalers were trained on RAW VAE latents, not on the per-channel-
/// normalized ones the diffusion loop works in, and upstream never calls one
/// without this sandwich - `VideoUpsampler.__call__` builds a video ENCODER
/// alongside the upscaler for no other purpose than to reach its
/// `per_channel_statistics`. Handing the normalized latent straight in
/// "works" in the sense that it returns a plausible tensor of the right
/// shape and a cosine of ~1 against a correctly-normalized run, which is
/// exactly why it is easy to miss: the error is a per-channel SCALE, and
/// cosine is scale-invariant. What it costs is variance. Measured on a real
/// 25-frame 960x544 stage-1 latent through the real x2 spatial upscaler:
///
/// | per-latent-frame std | frame 0 | 1 | 2 | 3 |
/// |---|---:|---:|---:|---:|
/// | input (normalized) | 1.070 | 0.960 | 1.013 | 1.069 |
/// | out, no un-normalize (WRONG) | 0.504 | 0.524 | 0.530 | 0.465 |
///
/// A latent at half the variance the model expects decodes to a washed-out,
/// blurred clip, and a refinement pass that only has three steps does not
/// get it back.
///
/// `mean`/`std` are the VAE's `per_channel_statistics.mean-of-means` /
/// `std-of-means`, one entry per latent channel; `latent` is `[C, T, H, W]`
/// row-major, the same layout [`LatentUpsampler::upsample`] takes.
pub fn upsample_video(ups: &LatentUpsampler, mean: &[f32], std: &[f32], latent: &[f32]) -> Vec<f32> {
    let c = mean.len();
    assert_eq!(std.len(), c, "per-channel mean/std disagree: {} vs {}", mean.len(), std.len());
    assert!(c > 0 && latent.len().is_multiple_of(c), "latent of {} values does not divide into {c} channels", latent.len());
    let plane = latent.len() / c;
    // `un_normalize`: `x * std + mean`, per channel.
    let mut raw = latent.to_vec();
    for (ci, chunk) in raw.chunks_exact_mut(plane).enumerate() {
        for v in chunk {
            *v = *v * std[ci] + mean[ci];
        }
    }
    let mut out = ups.upsample(&raw);
    let (_, _, oh, ow) = ups.out_shape();
    let out_plane = out.len() / c;
    assert_eq!(out_plane * c, out.len(), "upsampler returned {} values, not a whole multiple of {c} channels", out.len());
    debug_assert_eq!(out_plane % (oh as usize * ow as usize), 0);
    // `normalize`: `(x - mean) / std`, per channel.
    for (ci, chunk) in out.chunks_exact_mut(out_plane).enumerate() {
        for v in chunk {
            *v = (*v - mean[ci]) / std[ci];
        }
    }
    out
}

/// Build+run the SPATIAL x2 upscaler on `latent` (`[128, t, h, w]`), one call.
pub fn upsample_spatial(tensors: &Tensors, t: u32, h: u32, w: u32, device: Option<&str>) -> LatentUpsampler {
    LatentUpsampler::build(&LatentUpsamplerConfig::spatial_x2(), tensors, t, h, w, device)
}

/// Build+run the TEMPORAL x2 upscaler on `latent` (`[128, t, h, w]`), one call.
pub fn upsample_temporal(tensors: &Tensors, t: u32, h: u32, w: u32, device: Option<&str>) -> LatentUpsampler {
    LatentUpsampler::build(&LatentUpsamplerConfig::temporal_x2(), tensors, t, h, w, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_counts_the_shipped_checkpoints() {
        let spatial = LatentUpsamplerConfig::spatial_x2().tensor_manifest();
        assert_eq!(spatial.len(), 72, "spatial manifest has {} tensors", spatial.len());
        let temporal = LatentUpsamplerConfig::temporal_x2().tensor_manifest();
        assert_eq!(temporal.len(), 72, "temporal manifest has {} tensors", temporal.len());

        let get = |m: &[(String, Vec<usize>)], n: &str| m.iter().find(|(k, _)| k == n).unwrap().1.clone();
        assert_eq!(get(&spatial, "initial_conv.weight"), vec![1024, 128, 3, 3, 3]);
        assert_eq!(get(&spatial, "upsampler.0.weight"), vec![4096, 1024, 3, 3]);
        assert_eq!(get(&spatial, "final_conv.weight"), vec![128, 1024, 3, 3, 3]);
        assert_eq!(get(&temporal, "initial_conv.weight"), vec![512, 128, 3, 3, 3]);
        assert_eq!(get(&temporal, "upsampler.0.weight"), vec![1024, 512, 3, 3, 3]);
        assert_eq!(get(&temporal, "final_conv.weight"), vec![128, 512, 3, 3, 3]);

        let names: std::collections::HashSet<&str> = spatial.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), spatial.len(), "duplicate tensor name in the manifest");
    }
}
