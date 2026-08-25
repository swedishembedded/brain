// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The LTX-2.5 audio VAE: a **2D** causal-conv encoder/decoder over log-mel
//! spectrograms (`[channels, time, mel_bins]`), continuous Gaussian latent.
//!
//! Ported from `ltx_core.model.audio_vae.{audio_vae,resnet,downsample,
//! upsample,causal_conv_2d}` (real weights: `ltx-2.5-audio-vae-bf16.safetensors`,
//! `config.audio_vae.model.params.ddconfig`), NOT `vae3d.rs`'s shape - the
//! audio VAE is 2D over `(time=height, freq=width)`, has **no attention
//! anywhere** at the real config (`attn_resolutions: []`,
//! `mid_block_add_attention: false` - the reference code supports an
//! attention block class, but this checkpoint never instantiates one), and
//! its conventions differ from the video VAE in several places pinned by
//! reading `causal_conv_2d.py`/`causality_axis.py`/`resnet.py`/`downsample.py`
//! /`upsample.py`/`audio_vae.py` directly (never trusting a transcription for
//! anything ambiguous):
//!
//! * **Padding is asymmetric on time, symmetric on freq** - the checkpoint's
//!   `causality_axis: "height"` and this VAE's `mel` tensor layout is
//!   `[channels, time, mel_bins]` i.e. NCHW with **H = time, W = freq**. Every
//!   kernel-3 conv pads `(top=2, bottom=0)` on H (one-sided causal, ZERO not
//!   replicate - unlike the video VAE's replicate padding) and `(left=1,
//!   right=1)` on W (symmetric). A kernel-1 conv (the `nin_shortcut`) needs no
//!   padding at all, same formula with `k=1`. Implemented as an explicit
//!   [`kernels::PAD2D`] step (asymmetric zero-pad) then a `pad=0` conv,
//!   reusing `vae::blocks`'s own `conv_bias_reg` kernel rather than a new one,
//!   since `vae::blocks::Builder`'s own `conv`/`conv_s` only expose a
//!   SYMMETRIC `pad` and cannot express this - this module dispatches the
//!   kernel directly instead of going through that `Builder`.
//! * **Downsample is a real strided (stride-2, k=3) conv**, not
//!   space-to-depth/group-mean like the video VAE - `Downsample.forward`'s own
//!   asymmetric pad for `causality_axis=HEIGHT` is `(left=0, right=1,
//!   top=2, bottom=0)` (read off `downsample.py` directly, not derived), and
//!   the conv is channel-preserving (`Conv2d(in_channels, in_channels, ...)`
//!   in the reference - never a channel multiplier on this op; only the
//!   surrounding `ResnetBlock`s change channel width).
//! * **Upsample is nearest-2x then a channel-preserving causal conv, then a
//!   crop of the FIRST row** (`x[:,:,1:,:]`) to undo the causal pad's extra
//!   length - `upsample.py`'s own comment walks the arithmetic
//!   (`[0,1,2]` -> interpolate `[0,0,1,1,2,2]` -> causal-pad-then-conv keeps
//!   length `2n` -> drop element 0, not the last, to get `2n-1`). No crop on
//!   the freq (W) axis - width doubles cleanly since its own padding is
//!   symmetric. [`kernels::CROP2D`] (pad2d's exact adjoint, already in this
//!   crate's kernel set) does the crop; [`kernels::UPSAMPLE2`] does the
//!   nearest-neighbour double (identical semantics to `F.interpolate(...,
//!   mode="nearest", scale_factor=2.0)`, already used unchanged by the video
//!   VAE).
//! * **`PixelNorm` here is `build_normalization_layer`'s call site, eps
//!   `1e-6`** (`normalization.py`) - NOT `vae3d`'s `1e-8` (`PixelNorm()`'s own
//!   default, a different call site the video VAE uses instead). Both are
//!   real, in the same checkpoint family; do not unify them. No learnable
//!   gain either way. Reuses [`kernels::L2NORM_SCALE`] with a synthesized
//!   uniform `sqrt(C)` gain and `eps_l2 = C * eps`, the exact recipe
//!   `vae::blocks3d::Builder3d::pixel_norm`'s own doc comment derives -
//!   ported here rather than imported because `blocks3d::Builder3d` is a 3D
//!   tensor type (`T3`) this module has no use for; the *kernel* is reused,
//!   not a private reimplementation of its math.
//! * **The bottleneck normalize/denormalize is genuinely per-(channel,
//!   freq-bin), not per-channel** - `AudioEncoder._normalize_latents`
//!   rearranges `b c t f -> b t (c f)` before applying
//!   `PerChannelStatistics`, whose `latent_channels=ch` (128, the encoder's
//!   BASE channel count, not `z_channels`) is deliberately sized to equal
//!   `z_channels * (mel_bins / 4)` (`8 * 16 = 128` at the real config) - the
//!   stat vectors are indexed by `c*mel_bins_bottleneck + f`, broadcast over
//!   time only. Implemented as plain host arithmetic on the read-back
//!   `[C,T,F]` array (no device kernel does a "broadcast over an outer axis,
//!   vary over an inner axis" op) - `AudioPatchifier.patchify`/`unpatchify`
//!   themselves are `patch_size=1` identity reshapes, so the whole
//!   patchify-normalize-unpatchify round trip collapses to exactly this
//!   affine and nothing else.
//! * **The decoder's general `_adjust_output_shape` crop/pad dance is
//!   PROVABLY A NO-OP for every causal config this checkpoint uses**, not
//!   merely untested: `target_frames = latent.frames * 4 - 3` (the
//!   `LATENT_DOWNSAMPLE_FACTOR=4`, causal `-3` from `AudioDecoder.
//!   _denormalize_latents`) exactly equals the two upsample stages' own
//!   `2n-1` composition (`T0 -> 2*T0-1 -> 2*(2*T0-1)-1 = 4*T0-3`), and the
//!   freq axis has no causal crop at all on either side (both halvings and
//!   both doublings are exact powers of two for any `mel_bins` a multiple of
//!   4). [`decode`] asserts this rather than reimplementing the general
//!   crop/pad, the same "prove it, then assert it" move `vae3d.rs` makes for
//!   its own out-of-scope chunking.
//!
//! Eager dispatch (one [`Gpu::submit`] per op, immediately read back where a
//! host-side reshape is cheapest), the same style `crates/mimi`'s SEANet
//! decoder uses - not the lazy `Vec<Step>`-then-one-submit graph
//! `vae::blocks`/`vae::blocks3d` build, because this VAE's per-call tensor
//! sizes are tiny (at most ~3.3M elements) and the win from batching the
//! submit is not worth a second graph-builder implementation for one small
//! model. Zero new kernels: every op here (`pad2d`, `crop2d`,
//! `conv_bias_reg`, `silu`, `add2`, `upsample2`, `nchw_nlc`/`nlc_nchw` +
//! `l2norm_scale` for `PixelNorm`) already exists in `crates/kernels` for
//! other models.

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu};
use vae::blocks::Tensors;

const K_PAD2D: usize = 0;
const K_CROP2D: usize = 1;
const K_CONV: usize = 2;
const K_SILU: usize = 3;
const K_ADD2: usize = 4;
const K_UPSAMPLE2: usize = 5;
const K_NCHW_NLC: usize = 6;
const K_NLC_NCHW: usize = 7;
const K_L2NORM_SCALE: usize = 8;

const KERNELS: [(&str, &str); 9] = [
    ("pad2d", kernels::PAD2D),
    ("crop2d", kernels::CROP2D),
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
    ("silu", kernels::SILU),
    ("add2", kernels::ADD2),
    ("upsample2", kernels::UPSAMPLE2),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("l2norm_scale", kernels::L2NORM_SCALE),
];

/// `build_normalization_layer`'s `PixelNorm` eps - the call site this audio
/// VAE uses everywhere (`norm_type: "pixel"` in the real config). NOT
/// `vae3d::PIXEL_NORM_EPS` (`1e-8`, a different call site the video VAE uses
/// instead) - both real, in the same checkpoint family, never unified.
pub const PIXEL_NORM_EPS: f32 = 1e-6;

/// Real `ltx-2.5-audio-vae-bf16.safetensors` config (`config.audio_vae.model.
/// params.ddconfig`): `ch=128, ch_mult=[1,2,4], num_res_blocks=2, z_channels=8,
/// in_channels=out_ch=2, norm_type=pixel, causality_axis=height,
/// attn_resolutions=[], mid_block_add_attention=false`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioVaeConfig {
    pub ch: u32,
    pub ch_mult: [u32; 3],
    pub num_res_blocks: u32,
    pub z_channels: u32,
    pub in_channels: u32,
    pub out_ch: u32,
    pub eps: f32,
}

impl Default for AudioVaeConfig {
    fn default() -> Self {
        Self::ltx25()
    }
}

impl AudioVaeConfig {
    pub fn ltx25() -> AudioVaeConfig {
        AudioVaeConfig { ch: 128, ch_mult: [1, 2, 4], num_res_blocks: 2, z_channels: 8, in_channels: 2, out_ch: 2, eps: PIXEL_NORM_EPS }
    }

    /// `in_ch_mult = (1, *ch_mult)` sliced to `num_resolutions` (3) entries -
    /// upstream's own derivation in `build_downsampling_path`.
    fn in_ch_mult(&self) -> [u32; 3] {
        [1, self.ch_mult[0], self.ch_mult[1]]
    }

    /// The bottleneck width (`ch * ch_mult[-1]`, 512 at the real config) -
    /// `decoder.conv_in`'s output width and every `up.2` resnet's channel
    /// count before the first upsample.
    pub fn bottleneck(&self) -> u32 {
        self.ch * self.ch_mult[2]
    }

    /// Every tensor this model reads, in the checkpoint's own (Comfy-split,
    /// bare `encoder.`/`decoder.`/`per_channel_statistics.`) name space -
    /// cross-checked leaf-by-leaf against the real header (102 tensors: 44
    /// encoder + 56 decoder + 2 shared stats). `make_conv2d(causality_axis=
    /// ...)` wraps every resnet/`conv_in`/`conv_out`/`upsample.conv` conv in a
    /// `CausalConv2d`, whose own inner `nn.Conv2d` is named `conv` - hence the
    /// doubled `....conv.{weight,bias}` leaf on those (confirmed:
    /// `encoder.down.0.block.0.conv1.conv.weight`,
    /// `decoder.up.1.upsample.conv.conv.weight`). `Downsample`'s own conv is a
    /// PLAIN `nn.Conv2d` (never wrapped), so it gets exactly one `.conv.`
    /// (`encoder.down.0.downsample.conv.weight`) - the two helper closures
    /// below (`causal`/`plain`) reproduce that difference on purpose, not by
    /// accident.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let mut m: Vec<(String, Vec<usize>)> = Vec::new();
        let causal = |m: &mut Vec<(String, Vec<usize>)>, prefix: &str, cout: u32, cin: u32, k: u32| {
            m.push((format!("{prefix}.conv.weight"), vec![cout as usize, cin as usize, k as usize, k as usize]));
            m.push((format!("{prefix}.conv.bias"), vec![cout as usize]));
        };
        let plain = |m: &mut Vec<(String, Vec<usize>)>, prefix: &str, cout: u32, cin: u32, k: u32| {
            m.push((format!("{prefix}.weight"), vec![cout as usize, cin as usize, k as usize, k as usize]));
            m.push((format!("{prefix}.bias"), vec![cout as usize]));
        };

        // ---- encoder ----
        causal(&mut m, "encoder.conv_in", self.ch, self.in_channels, 3);
        let in_mult = self.in_ch_mult();
        for (level, &im) in in_mult.iter().enumerate() {
            let block_out = self.ch * self.ch_mult[level];
            let mut cin = self.ch * im;
            for j in 0..self.num_res_blocks {
                let p = format!("encoder.down.{level}.block.{j}");
                causal(&mut m, &format!("{p}.conv1"), block_out, cin, 3);
                causal(&mut m, &format!("{p}.conv2"), block_out, block_out, 3);
                if cin != block_out {
                    causal(&mut m, &format!("{p}.nin_shortcut"), block_out, cin, 1);
                }
                cin = block_out;
            }
            if level != 2 {
                plain(&mut m, &format!("encoder.down.{level}.downsample.conv"), block_out, block_out, 3);
            }
        }
        let mid_c = self.bottleneck();
        causal(&mut m, "encoder.mid.block_1.conv1", mid_c, mid_c, 3);
        causal(&mut m, "encoder.mid.block_1.conv2", mid_c, mid_c, 3);
        causal(&mut m, "encoder.mid.block_2.conv1", mid_c, mid_c, 3);
        causal(&mut m, "encoder.mid.block_2.conv2", mid_c, mid_c, 3);
        causal(&mut m, "encoder.conv_out", 2 * self.z_channels, mid_c, 3);

        // ---- decoder ----
        let base = self.bottleneck();
        causal(&mut m, "decoder.conv_in", base, self.z_channels, 3);
        causal(&mut m, "decoder.mid.block_1.conv1", base, base, 3);
        causal(&mut m, "decoder.mid.block_1.conv2", base, base, 3);
        causal(&mut m, "decoder.mid.block_2.conv1", base, base, 3);
        causal(&mut m, "decoder.mid.block_2.conv2", base, base, 3);
        let mut block_in = base;
        for level in (0..3usize).rev() {
            let block_out = self.ch * self.ch_mult[level];
            for j in 0..(self.num_res_blocks + 1) {
                let p = format!("decoder.up.{level}.block.{j}");
                causal(&mut m, &format!("{p}.conv1"), block_out, block_in, 3);
                causal(&mut m, &format!("{p}.conv2"), block_out, block_out, 3);
                if block_in != block_out {
                    causal(&mut m, &format!("{p}.nin_shortcut"), block_out, block_in, 1);
                }
                block_in = block_out;
            }
            if level != 0 {
                causal(&mut m, &format!("decoder.up.{level}.upsample.conv"), block_out, block_out, 3);
            }
        }
        causal(&mut m, "decoder.conv_out", self.out_ch, block_in, 3);

        m.push(("per_channel_statistics.mean-of-means".into(), vec![self.ch as usize]));
        m.push(("per_channel_statistics.std-of-means".into(), vec![self.ch as usize]));
        m
    }
}

fn upload(gpu: &Gpu, t: &Tensors, name: &str) -> DeviceBuffer {
    let (_, data) = t.get(name).unwrap_or_else(|| panic!("ltxv audio_vae: missing tensor {name}"));
    gpu.storage_init(name, data)
}

fn pad2d(gpu: &Gpu, c: u32, h: u32, w: u32, l: u32, r: u32, t: u32, b: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let (hp, wp) = (h + t + b, w + l + r);
    let total = c * hp * wp;
    let y = gpu.storage(total as u64);
    gpu.submit(&[], &[gpu.step(K_PAD2D, &[x, &y], &[total, h, w, l, r, t, b], total)]);
    y
}

fn crop2d(gpu: &Gpu, c: u32, ho: u32, wo: u32, l: u32, r: u32, t: u32, b: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let total = c * ho * wo;
    let y = gpu.storage(total as u64);
    gpu.submit(&[], &[gpu.step(K_CROP2D, &[x, &y], &[total, ho, wo, l, r, t, b], total)]);
    y
}

/// Direct `conv_bias_reg` dispatch over an ALREADY-PADDED `[cin,hp,wp]`
/// input, `pad=0`. `ho`/`wo` are the caller-computed output dims.
#[allow(clippy::too_many_arguments)]
fn conv_raw(gpu: &Gpu, wgt: &DeviceBuffer, bias: &DeviceBuffer, cin: u32, cout: u32, k: u32, stride: u32, hp: u32, wp: u32, ho: u32, wo: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let y = gpu.storage((cout * ho * wo) as u64);
    let threads = cout.div_ceil(8) * (ho * wo).div_ceil(4);
    gpu.submit(&[], &[gpu.step(K_CONV, &[x, wgt, bias, &y], &[1, cin, hp, wp, cout, k, stride, 0, ho, wo], threads)]);
    y
}

/// `CausalConv2d(kernel_size=k, causality_axis=height)`: pad `(top=k-1,
/// bottom=0)` on H (one-sided ZERO pad, never replicate), `(left=(k-1)/2,
/// right=k-1-(k-1)/2)` on W (symmetric) - preserves `(h,w)`. `prefix` is the
/// OUTER module name (e.g. `encoder.down.0.block.0.conv1`); the inner
/// `.conv.{weight,bias}` leaf is appended here.
fn causal_conv(gpu: &Gpu, t: &Tensors, prefix: &str, cin: u32, cout: u32, k: u32, h: u32, w: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let pad_h = k - 1;
    let pad_w = k - 1;
    let (pl, pr) = (pad_w / 2, pad_w - pad_w / 2);
    let padded = pad2d(gpu, cin, h, w, pl, pr, pad_h, 0, x);
    let wgt = upload(gpu, t, &format!("{prefix}.conv.weight"));
    let bias = upload(gpu, t, &format!("{prefix}.conv.bias"));
    conv_raw(gpu, &wgt, &bias, cin, cout, k, 1, h + pad_h, w + pl + pr, h, w, &padded)
}

/// `Downsample.forward` (`with_conv=True`, `causality_axis=height`): pad
/// `(left=0, right=1, top=2, bottom=0)` then a stride-2, `k=3`, `pad=0`
/// PLAIN conv (channel-preserving, never wrapped in `CausalConv2d`).
fn downsample(gpu: &Gpu, t: &Tensors, prefix: &str, c: u32, h: u32, w: u32, x: &DeviceBuffer) -> (DeviceBuffer, u32, u32) {
    let padded = pad2d(gpu, c, h, w, 0, 1, 2, 0, x);
    let (hp, wp) = (h + 2, w + 1);
    let (ho, wo) = ((hp - 3) / 2 + 1, (wp - 3) / 2 + 1);
    let wgt = upload(gpu, t, &format!("{prefix}.weight"));
    let bias = upload(gpu, t, &format!("{prefix}.bias"));
    let y = conv_raw(gpu, &wgt, &bias, c, c, 3, 2, hp, wp, ho, wo, &padded);
    (y, ho, wo)
}

/// `Upsample.forward` (`with_conv=True`, `causality_axis=height`): nearest-2x,
/// then the (channel-preserving) causal conv, then drop the first H row.
fn upsample_block(gpu: &Gpu, t: &Tensors, prefix: &str, c: u32, h: u32, w: u32, x: &DeviceBuffer) -> (DeviceBuffer, u32, u32) {
    let (h2, w2) = (2 * h, 2 * w);
    let up = gpu.storage((c * h2 * w2) as u64);
    gpu.submit(&[], &[gpu.step(K_UPSAMPLE2, &[x, &up], &[1, c, h, w], c * 4 * h * w)]);
    let conv = causal_conv(gpu, t, &format!("{prefix}.conv"), c, c, 3, h2, w2, &up);
    let cropped = crop2d(gpu, c, h2 - 1, w2, 0, 0, 1, 0, &conv);
    (cropped, h2 - 1, w2)
}

/// `PixelNorm(dim=1, eps)` (`build_normalization_layer`'s call site): pure
/// channel-axis RMS-norm, no learnable gain. Reuses [`kernels::L2NORM_SCALE`]
/// with a synthesized `sqrt(C)` uniform gain and `eps_l2 = C*eps` - see this
/// module's header for the algebra.
fn pixel_norm(gpu: &Gpu, c: u32, h: u32, w: u32, eps: f32, x: &DeviceBuffer) -> DeviceBuffer {
    let hw = h * w;
    let total = c * hw;
    let gain = vec![(c as f32).sqrt(); c as usize];
    let g = gpu.storage_init("__audio_pixel_norm.gain", &gain);
    let rows = gpu.storage(total as u64);
    gpu.submit(&[], &[gpu.step(K_NCHW_NLC, &[x, &rows], &[total, c, hw], total)]);
    let normed = gpu.storage(total as u64);
    gpu.submit(&[], &[gpu.step(K_L2NORM_SCALE, &[&rows, &g, &normed], &[hw, c, f(c as f32 * eps)], total)]);
    let y = gpu.storage(total as u64);
    gpu.submit(&[], &[gpu.step(K_NLC_NCHW, &[&normed, &y], &[total, c, hw], total)]);
    y
}

fn silu(gpu: &Gpu, n: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let y = gpu.storage(n as u64);
    gpu.submit(&[], &[gpu.step(K_SILU, &[x, &y], &[n], n)]);
    y
}

fn add(gpu: &Gpu, n: u32, a: &DeviceBuffer, b: &DeviceBuffer) -> DeviceBuffer {
    let y = gpu.storage(n as u64);
    gpu.submit(&[], &[gpu.step(K_ADD2, &[a, b, &y], &[n], n)]);
    y
}

/// `ResnetBlock` (`norm_type=pixel`, `temb=None` - this VAE has no time
/// embedding, unlike a diffusion U-Net's resnet): `x + conv2(silu(norm2(
/// conv1(silu(norm1(x))))))`, `nin_shortcut` (a causal `k=1` conv - trivial
/// padding, same `causal_conv` helper) only when `cin != cout`.
#[allow(clippy::too_many_arguments)]
fn resnet_block(gpu: &Gpu, t: &Tensors, prefix: &str, cin: u32, cout: u32, h: u32, w: u32, eps: f32, x: &DeviceBuffer) -> DeviceBuffer {
    let n1 = pixel_norm(gpu, cin, h, w, eps, x);
    let s1 = silu(gpu, cin * h * w, &n1);
    let c1 = causal_conv(gpu, t, &format!("{prefix}.conv1"), cin, cout, 3, h, w, &s1);
    let n2 = pixel_norm(gpu, cout, h, w, eps, &c1);
    let s2 = silu(gpu, cout * h * w, &n2);
    let c2 = causal_conv(gpu, t, &format!("{prefix}.conv2"), cout, cout, 3, h, w, &s2);
    let shortcut = if cin != cout {
        causal_conv(gpu, t, &format!("{prefix}.nin_shortcut"), cin, cout, 1, h, w, x)
    } else {
        x.clone()
    };
    add(gpu, cout * h * w, &c2, &shortcut)
}

/// `run_mid_block`: two stacked [`resnet_block`]s, no attention
/// (`mid_block_add_attention: false` at the real config).
fn mid_block(gpu: &Gpu, t: &Tensors, prefix: &str, c: u32, h: u32, w: u32, eps: f32, x: &DeviceBuffer) -> DeviceBuffer {
    let y = resnet_block(gpu, t, &format!("{prefix}.block_1"), c, c, h, w, eps, x);
    resnet_block(gpu, t, &format!("{prefix}.block_2"), c, c, h, w, eps, &y)
}

/// Per-(channel, freq) affine over a host `[c,t,f]` array with stat vectors of
/// length `c*f` (broadcast over `t` only) - see this module's header for why
/// this is the exact reduction of `AudioPatchifier.patchify` (`patch_size=1`,
/// a no-op reshape) + `PerChannelStatistics` + `unpatchify`.
fn affine_cf(x: &mut [f32], c: usize, t: usize, fbins: usize, stat_a: &[f32], stat_b: &[f32], sub_then_div: bool) {
    assert_eq!(stat_a.len(), c * fbins, "stat vector length {} != c*f {}", stat_a.len(), c * fbins);
    assert_eq!(stat_b.len(), c * fbins);
    for ci in 0..c {
        for ti in 0..t {
            for fi in 0..fbins {
                let idx = (ci * t + ti) * fbins + fi;
                let (a, b) = (stat_a[ci * fbins + fi], stat_b[ci * fbins + fi]);
                x[idx] = if sub_then_div { (x[idx] - a) / b } else { x[idx] * a + b };
            }
        }
    }
}

/// Encode a log-mel spectrogram `[in_channels, t, f]` (row-major) into the
/// NORMALISED continuous latent `[z_channels, t/4, f/4]`. `t` must make every
/// downsample stage's height formula land on an integer (any `t >= 1` works,
/// same asymmetric-causal formula as the real model); `f` must be a multiple
/// of 4 (two clean halvings, freq padding is symmetric).
pub fn encode(cfg: &AudioVaeConfig, tensors: &Tensors, mel: &[f32], t: u32, fbins: u32, device: Option<&str>) -> Vec<f32> {
    assert_eq!(mel.len(), (cfg.in_channels * t * fbins) as usize, "encode: {} values, expected {}", mel.len(), cfg.in_channels * t * fbins);
    let gpu = Gpu::open(device, &KERNELS);
    let x_in = gpu.storage_init("audio_vae.mel", mel);

    let mut h = causal_conv(&gpu, tensors, "encoder.conv_in", cfg.in_channels, cfg.ch, 3, t, fbins, &x_in);
    let (mut cur_t, mut cur_f) = (t, fbins);
    let in_mult = cfg.in_ch_mult();
    for (level, &im) in in_mult.iter().enumerate() {
        let block_out = cfg.ch * cfg.ch_mult[level];
        let mut cin = cfg.ch * im;
        for j in 0..cfg.num_res_blocks {
            h = resnet_block(&gpu, tensors, &format!("encoder.down.{level}.block.{j}"), cin, block_out, cur_t, cur_f, cfg.eps, &h);
            cin = block_out;
        }
        if level != 2 {
            let (y, ho, wo) = downsample(&gpu, tensors, &format!("encoder.down.{level}.downsample.conv"), block_out, cur_t, cur_f, &h);
            h = y;
            cur_t = ho;
            cur_f = wo;
        }
    }
    let mid_c = cfg.bottleneck();
    h = mid_block(&gpu, tensors, "encoder.mid", mid_c, cur_t, cur_f, cfg.eps, &h);

    let normed = pixel_norm(&gpu, mid_c, cur_t, cur_f, cfg.eps, &h);
    let act = silu(&gpu, mid_c * cur_t * cur_f, &normed);
    let moments = causal_conv(&gpu, tensors, "encoder.conv_out", mid_c, 2 * cfg.z_channels, 3, cur_t, cur_f, &act);

    let moments_host = gpu.read(&moments, (2 * cfg.z_channels * cur_t * cur_f) as usize);
    // `torch.chunk(latent_output, 2, dim=1)[0]`: the first half of the
    // CHANNEL-MAJOR layout is exactly the first `z_channels*cur_t*cur_f`
    // values, a contiguous prefix - no device slice op needed.
    let mut latent = moments_host[..(cfg.z_channels * cur_t * cur_f) as usize].to_vec();

    let mean = &tensors.get("per_channel_statistics.mean-of-means").expect("stats").1;
    let std = &tensors.get("per_channel_statistics.std-of-means").expect("stats").1;
    affine_cf(&mut latent, cfg.z_channels as usize, cur_t as usize, cur_f as usize, mean, std, true);
    latent
}

/// Decode a NORMALISED latent `[z_channels, lt, lf]` into a reconstructed
/// log-mel spectrogram `[out_ch, 4*lt-3, 4*lf]`.
pub fn decode(cfg: &AudioVaeConfig, tensors: &Tensors, latent: &[f32], lt: u32, lf: u32, device: Option<&str>) -> Vec<f32> {
    assert_eq!(latent.len(), (cfg.z_channels * lt * lf) as usize, "decode: {} values, expected {}", latent.len(), cfg.z_channels * lt * lf);
    let mean = &tensors.get("per_channel_statistics.mean-of-means").expect("stats").1;
    let std = &tensors.get("per_channel_statistics.std-of-means").expect("stats").1;
    let mut z_host = latent.to_vec();
    affine_cf(&mut z_host, cfg.z_channels as usize, lt as usize, lf as usize, std, mean, false);

    let gpu = Gpu::open(device, &KERNELS);
    let z_in = gpu.storage_init("audio_vae.latent", &z_host);

    let base = cfg.bottleneck();
    let mut h = causal_conv(&gpu, tensors, "decoder.conv_in", cfg.z_channels, base, 3, lt, lf, &z_in);
    h = mid_block(&gpu, tensors, "decoder.mid", base, lt, lf, cfg.eps, &h);

    let (mut cur_t, mut cur_f) = (lt, lf);
    let mut block_in = base;
    for level in (0..3usize).rev() {
        let block_out = cfg.ch * cfg.ch_mult[level];
        for j in 0..(cfg.num_res_blocks + 1) {
            h = resnet_block(&gpu, tensors, &format!("decoder.up.{level}.block.{j}"), block_in, block_out, cur_t, cur_f, cfg.eps, &h);
            block_in = block_out;
        }
        if level != 0 {
            let (y, ho, wo) = upsample_block(&gpu, tensors, &format!("decoder.up.{level}.upsample"), block_out, cur_t, cur_f, &h);
            h = y;
            cur_t = ho;
            cur_f = wo;
        }
    }
    // `_adjust_output_shape` is a provable no-op here - see this module's
    // header. Asserted, not reimplemented.
    assert_eq!(cur_t, 4 * lt - 3, "decoder time axis {cur_t}, expected 4*lt-3 = {}", 4 * lt - 3);
    assert_eq!(cur_f, 4 * lf, "decoder freq axis {cur_f}, expected 4*lf = {}", 4 * lf);

    let normed = pixel_norm(&gpu, block_in, cur_t, cur_f, cfg.eps, &h);
    let act = silu(&gpu, block_in * cur_t * cur_f, &normed);
    let out = causal_conv(&gpu, tensors, "decoder.conv_out", block_in, cfg.out_ch, 3, cur_t, cur_f, &act);
    gpu.read(&out, (cfg.out_ch * cur_t * cur_f) as usize)
}

/// A name→(shape, data) map from a raw tensor list, for tests that build a
/// synthetic checkpoint (mirrors `vae::blocks::Tensors`'s own construction
/// pattern used by `crates/ltxv/src/import.rs`'s tests).
pub fn tensors_from(list: Vec<checkpoint::safetensors::StTensor>) -> Tensors {
    let mut m: Tensors = HashMap::new();
    for t in list {
        m.insert(t.name, (t.shape, t.data));
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_counts_the_shipped_checkpoint() {
        let m = AudioVaeConfig::ltx25().tensor_manifest();
        assert_eq!(m.len(), 102, "manifest has {} tensors", m.len());
        let names: std::collections::HashSet<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), m.len(), "duplicate tensor name in the manifest");
        assert!(names.contains("encoder.conv_in.conv.weight"));
        assert!(names.contains("encoder.down.0.downsample.conv.weight"));
        assert!(names.contains("decoder.up.1.upsample.conv.conv.weight"));
        assert!(names.contains("decoder.up.1.block.0.nin_shortcut.conv.weight"));
        assert!(names.contains("per_channel_statistics.mean-of-means"));

        let get = |n: &str| m.iter().find(|(k, _)| k == n).unwrap().1.clone();
        assert_eq!(get("encoder.conv_in.conv.weight"), vec![128, 2, 3, 3]);
        assert_eq!(get("encoder.conv_out.conv.weight"), vec![16, 512, 3, 3]);
        assert_eq!(get("encoder.down.1.block.0.nin_shortcut.conv.weight"), vec![256, 128, 1, 1]);
        assert_eq!(get("encoder.down.2.block.0.nin_shortcut.conv.weight"), vec![512, 256, 1, 1]);
        assert_eq!(get("decoder.conv_in.conv.weight"), vec![512, 8, 3, 3]);
        assert_eq!(get("decoder.conv_out.conv.weight"), vec![2, 128, 3, 3]);
        assert_eq!(get("decoder.up.2.upsample.conv.conv.weight"), vec![512, 512, 3, 3]);
        assert_eq!(get("decoder.up.1.upsample.conv.conv.weight"), vec![256, 256, 3, 3]);
        assert_eq!(get("decoder.up.0.block.0.nin_shortcut.conv.weight"), vec![128, 256, 1, 1]);
    }

    /// `bottleneck`/`in_ch_mult` land on the shapes the real checkpoint's
    /// down/up channel walk actually uses (128 -> 256 -> 512).
    #[test]
    fn channel_walk_matches_the_real_checkpoint() {
        let cfg = AudioVaeConfig::ltx25();
        assert_eq!(cfg.bottleneck(), 512);
        assert_eq!(cfg.in_ch_mult(), [1, 1, 2]);
    }
}
