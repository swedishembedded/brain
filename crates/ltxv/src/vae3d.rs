// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The LTX-2.5 causal 3D video VAE: encoder + **conv** decoder only (the
//! convolution-free "diffusion decoder", `DiffusionVideoDecoder`/NA
//! neighborhood-attention, is a completely different architecture and is not
//! implemented here).
//!
//! Built on [`vae::blocks3d::Builder3d`], the same shared 3D-VAE primitive
//! set `crates/wan`'s causal VAE uses - but several conventions genuinely
//! differ from Wan's, all settled by reading `ltx_core.model.video_vae.*`
//! (`scratchpad/reference/ltxv/packages/ltx-core/src/ltx_core/model/video_vae/`)
//! and cross-checked against the real `ltx-2.5-video-vae-conv-bf16.safetensors`
//! header (`config.vae` metadata + the 170 tensor names/shapes):
//!
//! * **Temporal padding is REPLICATE, not zero**, and it is NOT symmetric the
//!   same way on both sides of the model. The encoder is causal
//!   (`causal=True` throughout `VideoEncoder.forward`): every conv prepends
//!   TWO copies of the input's own first frame. The decoder's `causal_decoder`
//!   config is `false` for this checkpoint, so `ConvVideoDecoder.forward`
//!   passes `causal=False` to every block: convs pad SYMMETRICALLY - one
//!   replicated copy of the first frame prepended, one of the last frame
//!   appended. Composed from [`vae::blocks3d::Builder3d::time_slice`]/
//!   `::time_cat`, no new kernel (see [`causal_conv3`]).
//! - **Spatial padding is ZEROS on both encoder and decoder** for this
//!   checkpoint - empirically confirmed by instantiating both modules from
//!   the real metadata and reading `conv.padding_mode` off the built
//!   `CausalConv3d`s, because the code's own default differs (`reflect`) and
//!   only wins when the checkpoint config omits `spatial_padding_mode`; this
//!   one carries it explicitly as `"zeros"` for the whole model. Uses
//!   `Conv3d`'s existing built-in zero pad (`ph=pw=1`), same as Wan.
//! * **`PixelNorm`** (`ltx_core.model.common.normalization.PixelNorm`) is
//!   channel-axis RMS-norm with **no learnable gain**, eps **1e-8**
//!   (`PixelNorm()`'s own default - confirmed empirically off the built
//!   modules; `build_normalization_layer`'s eps=1e-6 default is a DIFFERENT
//!   call site this checkpoint's video VAE never uses, so it does not apply
//!   here). See [`vae::blocks3d::Builder3d::pixel_norm`] for how this reuses
//!   the existing L2-norm kernel with a synthesized uniform gain rather than
//!   a learned one.
//! * **Down/upsample is space-to-depth / depth-to-space with a
//!   parameter-free group-mean skip** (encoder `SpaceToDepthDownsample`) or
//!   with NO skip at all (decoder `DepthToSpaceUpsample` - `residual=False`
//!   for every block this checkpoint's `decoder_blocks` config names), never
//!   strided/transposed conv. See [`vae::blocks3d::Builder3d::space_to_depth`]
//!   / `::depth_to_space` / `::group_mean`.
//! * **No cross-chunk feature cache.** [`LtxVaeDecoder`] runs the whole clip
//!   in one graph, one submit. That is correct and exact, and it is also what
//!   stops fitting: a decode's peak VRAM tracks the OUTPUT PIXEL VOLUME
//!   (`frames x H x W`), and a 25-frame 1080p clip is a hard `wgpu` out-of-
//!   memory on a 24 GiB card. [`LtxVaeTiledDecoder`] is the overlapping-tile
//!   answer, ported from `ltx_core.tiling` + `ConvVideoDecoder.tiled_decode`
//!   - see [`crate::vae3d::LtxVaeTiledDecoder`] and [`vae::tiling3d`].
//! * **Frame rule is `F = 1 + 8k`** (stride T8xH32xW32, `patch_size=4`, 128
//!   latent channels): `patchify`/`unpatchify` (space-to-depth at the pixel
//!   boundary, `patch_size_t=1`) run on the HOST, once per encode/decode call,
//!   before upload / after readback - see [`crate::patchify`].
//! * `latent_log_var` is `"uniform"` for this checkpoint (not `"constant"` as
//!   an earlier reading of the roadmap assumed) - but both reduce to slicing
//!   the raw `conv_out` output for the mean (`means = conv_out[:, :128]`,
//!   `ltx_core`'s own docstring on `VideoEncoder.forward`), so this crate's
//!   forward (mean only, no learned variance branch) is identical either way.

use gpu_core::{DeviceBuffer, Gpu, Step};
use vae::blocks::Tensors;
use vae::blocks3d::{Builder3d, Conv3d, T3, KERNELS};

use crate::patchify;

/// `PixelNorm()`'s own default eps - the video VAE's norm everywhere
/// (`norm_layer: "pixel_norm"`), confirmed against the real checkpoint's
/// built modules (`enc.conv_norm_out.eps == 1e-8`, `resblock.norm1.eps ==
/// 1e-8`). NOT `1e-6` - that is `build_normalization_layer`'s default, a
/// different call site the video VAE never uses (the audio VAE does; not
/// this crate's concern yet).
pub const PIXEL_NORM_EPS: f32 = 1e-8;

/// `BRAIN_LTXV_VAE_TAPS=1` records every block output for parity debugging.
fn taps_enabled() -> bool {
    std::env::var("BRAIN_LTXV_VAE_TAPS").is_ok()
}

/// A `res_x` down/up-block: `n` stacked `ResnetBlock3D`s at a fixed channel
/// width (every one in this checkpoint has `in_channels == out_channels` -
/// `res_x_y`, the only block kind that would need a `conv_shortcut` /
/// `norm3`, never appears in `encoder_blocks`/`decoder_blocks`, so that path
/// is intentionally not implemented here).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResX {
    pub n: u32,
}

/// A compress/resample block's stride, shared shape for both encoder
/// (`SpaceToDepthDownsample`, contract) and decoder (`DepthToSpaceUpsample`,
/// expand) - `(temporal, height, width)`, each axis either `1` (untouched) or
/// `2` (halved/doubled).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stride {
    pub t: u32,
    pub h: u32,
    pub w: u32,
}

/// One entry of the encoder's flat `down_blocks` list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncBlock {
    Res(ResX),
    /// `SpaceToDepthDownsample`: channels multiply by `mult`, spatial/
    /// temporal extents named by `stride` divide by 2 each.
    Down { stride: Stride, mult: u32 },
}

/// One entry of the decoder's `up_blocks` list, already in EXECUTION order
/// (upstream builds `up_blocks` from `reversed(decoder_blocks)` - see
/// [`LtxVaeConfig::dec_blocks`]'s doc comment for the reversal detail).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecBlock {
    Res(ResX),
    /// `DepthToSpaceUpsample`, `residual=False` (no config in this
    /// checkpoint's `decoder_blocks` sets it): channels divide by `mult`,
    /// spatial/temporal extents named by `stride` multiply by 2 each.
    Up { stride: Stride, mult: u32 },
}

const ST_SPACE: Stride = Stride { t: 1, h: 2, w: 2 };
const ST_TIME: Stride = Stride { t: 2, h: 1, w: 1 };
const ST_ALL: Stride = Stride { t: 2, h: 2, w: 2 };

/// Architecture constants of the real `ltx-2.5-video-vae-conv-bf16.safetensors`
/// checkpoint (`config.vae` embedded metadata - `_class_name:
/// "CausalVideoAutoencoder"`), transcribed and cross-checked against the raw
/// tensor names/shapes (170 tensors: 84 encoder + 84 decoder + 2 shared
/// `per_channel_statistics`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LtxVaeConfig {
    pub patch_size: u32,
    pub latent_channels: u32,
    /// Feature width entering `conv_in` / leaving the last down-block
    /// (encoder) and leaving `conv_in` / entering the first up-block
    /// (decoder) - the bottleneck, `128 * product(down-block mults)`.
    pub bottleneck: u32,
    pub pixel_norm_eps: f32,
}

impl Default for LtxVaeConfig {
    fn default() -> Self {
        Self::conv25()
    }
}

impl LtxVaeConfig {
    /// The real LTX-2.5 conv-decoder-paired video VAE.
    pub fn conv25() -> LtxVaeConfig {
        LtxVaeConfig { patch_size: 4, latent_channels: 128, bottleneck: 1024, pixel_norm_eps: PIXEL_NORM_EPS }
    }

    /// `encoder_blocks`, transcribed from the checkpoint's `config.vae`
    /// metadata verbatim (block name, then its one integer/dict param).
    pub fn enc_blocks(&self) -> Vec<EncBlock> {
        vec![
            EncBlock::Res(ResX { n: 4 }),
            EncBlock::Down { stride: ST_SPACE, mult: 2 }, // compress_space_res
            EncBlock::Res(ResX { n: 6 }),
            EncBlock::Down { stride: ST_TIME, mult: 2 }, // compress_time_res
            EncBlock::Res(ResX { n: 4 }),
            EncBlock::Down { stride: ST_ALL, mult: 2 }, // compress_all_res
            EncBlock::Res(ResX { n: 2 }),
            EncBlock::Down { stride: ST_ALL, mult: 1 }, // compress_all_res
            EncBlock::Res(ResX { n: 2 }),
        ]
    }

    /// `decoder_blocks`, in EXECUTION order. Upstream stores `decoder_blocks`
    /// in the same encoder-mirroring order as `encoder_blocks` and builds
    /// `up_blocks` by iterating it REVERSED
    /// (`ConvVideoDecoder.__init__`: `for ... in list(reversed(decoder_blocks))`).
    /// This is that reversal, already applied - transcribed and cross-checked
    /// tensor-by-tensor against the real `decoder.up_blocks.{0..9}.*` names
    /// and shapes (e.g. `up_blocks.1.conv.weight` is `[4096,1024,...]`,
    /// matching `compress_all(mult=2)` at the 1024-wide bottleneck).
    pub fn dec_blocks(&self) -> Vec<DecBlock> {
        vec![
            DecBlock::Res(ResX { n: 2 }),
            DecBlock::Up { stride: ST_ALL, mult: 2 }, // compress_all
            DecBlock::Res(ResX { n: 2 }),
            DecBlock::Up { stride: ST_ALL, mult: 1 }, // compress_all
            DecBlock::Res(ResX { n: 4 }),
            DecBlock::Up { stride: ST_TIME, mult: 2 }, // compress_time
            DecBlock::Res(ResX { n: 6 }),
            DecBlock::Up { stride: ST_SPACE, mult: 2 }, // compress_space
            DecBlock::Res(ResX { n: 4 }),
        ]
    }

    /// Latent frames for `frames` video frames - the `1 + 8k -> 1 + k` rule.
    /// `None` when `frames` is not `1 + 8k`.
    pub fn latent_frames(&self, frames: u32) -> Option<u32> {
        ((frames >= 1) && (frames - 1).is_multiple_of(8)).then_some(1 + (frames - 1) / 8)
    }

    /// Every tensor this model reads, with its checkpoint shape, in the
    /// checkpoint's own (Comfy-split, bare `encoder.`/`decoder.`) name space.
    /// `make_conv_nd(causal=True)` wraps every conv in a `CausalConv3d`, whose
    /// own inner `nn.Conv3d` is named `conv` - hence the doubled
    /// `....conv.{weight,bias}` leaf (confirmed against the real header:
    /// `encoder.conv_in.conv.weight`, not `encoder.conv_in.weight`).
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let mut m: Vec<(String, Vec<usize>)> = Vec::new();
        let conv = |m: &mut Vec<(String, Vec<usize>)>, prefix: &str, cout: u32, cin: u32| {
            m.push((format!("{prefix}.conv.weight"), vec![cout as usize, cin as usize, 3, 3, 3]));
            m.push((format!("{prefix}.conv.bias"), vec![cout as usize]));
        };

        // ---- encoder ----
        // `conv_in` widens the patchified input to `latent_channels` (upstream:
        // `feature_channels = out_channels`, the encoder's OWN `out_channels`
        // ctor param, i.e. `latent_channels`) - the down-block walk then grows
        // that to `bottleneck` before `conv_out`. Confirmed against the real
        // header: `encoder.conv_in.conv.weight` is `[128,48,...]`, not `[1024,...]`.
        let in_ch = 3 * self.patch_size * self.patch_size;
        conv(&mut m, "encoder.conv_in", self.latent_channels, in_ch);
        let mut c = self.latent_channels;
        for (i, b) in self.enc_blocks().iter().enumerate() {
            let p = format!("encoder.down_blocks.{i}");
            match *b {
                EncBlock::Res(ResX { n }) => {
                    for j in 0..n {
                        conv(&mut m, &format!("{p}.res_blocks.{j}.conv1"), c, c);
                        conv(&mut m, &format!("{p}.res_blocks.{j}.conv2"), c, c);
                    }
                }
                EncBlock::Down { stride, mult } => {
                    let prod = stride.t * stride.h * stride.w;
                    let cout_ext = c * mult;
                    conv(&mut m, &format!("{p}.conv"), cout_ext / prod, c);
                    c = cout_ext;
                }
            }
        }
        conv(&mut m, "encoder.conv_out", self.latent_channels + 1, c);

        // ---- decoder ----
        conv(&mut m, "decoder.conv_in", self.bottleneck, self.latent_channels);
        let mut c = self.bottleneck;
        for (i, b) in self.dec_blocks().iter().enumerate() {
            let p = format!("decoder.up_blocks.{i}");
            match *b {
                DecBlock::Res(ResX { n }) => {
                    for j in 0..n {
                        conv(&mut m, &format!("{p}.res_blocks.{j}.conv1"), c, c);
                        conv(&mut m, &format!("{p}.res_blocks.{j}.conv2"), c, c);
                    }
                }
                DecBlock::Up { stride, mult } => {
                    let prod = stride.t * stride.h * stride.w;
                    conv(&mut m, &format!("{p}.conv"), (prod * c) / mult, c);
                    c /= mult;
                }
            }
        }
        conv(&mut m, "decoder.conv_out", 3 * self.patch_size * self.patch_size, c);

        m.push(("per_channel_statistics.mean-of-means".into(), vec![self.latent_channels as usize]));
        m.push(("per_channel_statistics.std-of-means".into(), vec![self.latent_channels as usize]));
        m
    }
}

// ---------------------------------------------------------------- blocks

/// `CausalConv3d(kernel_size=3)`'s temporal pad, composed from
/// [`Builder3d::time_slice`]/`::time_cat` (REPLICATE, never zero) - every conv
/// in this VAE is kernel-3, stride-1, so this is the one padding helper the
/// whole model needs. Height/width use the builder's existing zero pad
/// (`ph=pw=1`, built into [`Conv3d`]/`conv3d.wgsl`'s bounds check).
///
/// `causal=true` (the encoder, always): ONE-SIDED - two replicated copies of
/// the input's own first frame, prepended (`x[:,:,:1].repeat(1,1,2,1,1)` then
/// concat, upstream's `CausalConv3d.forward`).
/// `causal=false` (the decoder, always - `causal_decoder: false` in this
/// checkpoint): SYMMETRIC - one replicated first-frame copy prepended, one
/// replicated last-frame copy appended.
fn causal_conv3(b: &mut Builder3d, prefix: &str, cout: u32, x: &T3, causal: bool) -> T3 {
    let first = b.time_slice(x, 0, 1);
    let padded = if causal {
        let dup2 = b.time_cat(&first, &first);
        let out = b.time_cat(&dup2, x);
        b.free(dup2);
        out
    } else {
        let last = b.time_slice(x, x.t - 1, 1);
        let a = b.time_cat(&first, x);
        let out = b.time_cat(&a, &last);
        b.free(a);
        b.free(last);
        out
    };
    b.free(first);
    let spec = Conv3d { kt: 3, kh: 3, kw: 3, st: 1, sh: 1, sw: 1, pt: 0, ph: 1, pw: 1 };
    // `make_conv_nd(causal=True)` wraps every conv in a `CausalConv3d`, whose
    // own inner `nn.Conv3d` is named `conv` - the real header confirms
    // `encoder.conv_in.conv.weight`, not `encoder.conv_in.weight`. Every
    // caller passes the OUTER module name; this is the one place that adds
    // the inner `.conv` leaf, matching `LtxVaeConfig::tensor_manifest`.
    let y = b.conv(&format!("{prefix}.conv"), cout, spec, &padded);
    b.free(padded);
    y
}

/// `ResnetBlock3D` (`in_channels == out_channels`, the only case this
/// checkpoint uses): `x + conv2(silu(norm2(conv1(silu(norm1(x))))))`, both
/// convs causal per `causal`, both norms [`Builder3d::pixel_norm`].
fn resnet_block(b: &mut Builder3d, prefix: &str, c: u32, x: &T3, causal: bool, eps: f32) -> T3 {
    let n0 = b.pixel_norm(x, eps);
    let s0 = b.silu(&n0);
    b.free(n0);
    let c0 = causal_conv3(b, &format!("{prefix}.conv1"), c, &s0, causal);
    b.free(s0);
    let n1 = b.pixel_norm(&c0, eps);
    b.free(c0);
    let s1 = b.silu(&n1);
    b.free(n1);
    let c1 = causal_conv3(b, &format!("{prefix}.conv2"), c, &s1, causal);
    b.free(s1);
    let out = b.add(x, &c1);
    b.free(c1);
    out
}

/// `UNetMidBlock3D` (a `res_x` entry): `n` stacked [`resnet_block`]s.
fn res_x(b: &mut Builder3d, prefix: &str, n: u32, c: u32, x: &T3, causal: bool, eps: f32) -> T3 {
    let mut cur = x.clone();
    let mut owned = false;
    for j in 0..n {
        let next = resnet_block(b, &format!("{prefix}.res_blocks.{j}"), c, &cur, causal, eps);
        if owned {
            b.free(cur);
        }
        cur = next;
        owned = true;
    }
    cur
}

/// `SpaceToDepthDownsample`: conv branch space-to-depth'd, plus a
/// parameter-free group-mean skip of the (possibly frame0-duplicated) input
/// space-to-depth'd the same way. See this module's header for the exact
/// upstream sequence; both branches read `x2` (post frame0-duplication when
/// `stride.t == 2`), not `x`.
fn downsample(b: &mut Builder3d, prefix: &str, c_in: u32, stride: Stride, mult: u32, x: &T3) -> T3 {
    let prod = stride.t * stride.h * stride.w;
    let cout_ext = c_in * mult;
    let cout_conv = cout_ext / prod;

    // `SpaceToDepthDownsample.forward`: `x = torch.cat([x[:,:,:1,:,:], x],
    // dim=2)` - ONE duplicated copy of frame 0 prepended (making an odd frame
    // count even for the stride-2 split), NOT two. `causal_conv3`'s own pad
    // (always two replicated copies, for its kernel-3 causal receptive field)
    // is a SEPARATE, later step applied to whichever of `x`/`x2` each branch
    // feeds it - conflating the two (prepending two copies here) fed
    // `space_to_depth` an odd frame count and panicked on the divisibility
    // assert the first time this ran against real weights.
    let (x2, x2_owned) = if stride.t == 2 {
        let f0 = b.time_slice(x, 0, 1);
        let out = b.time_cat(&f0, x);
        b.free(f0);
        (out, true)
    } else {
        (x.clone(), false)
    };

    let sd = b.space_to_depth(&x2, stride.t, stride.h, stride.w);
    let group_size = (c_in * prod) / cout_ext;
    let skip = b.group_mean(&sd, group_size);
    b.free(sd);

    let conv_out = causal_conv3(b, &format!("{prefix}.conv"), cout_conv, &x2, true);
    if x2_owned {
        b.free(x2);
    }
    let folded = b.space_to_depth(&conv_out, stride.t, stride.h, stride.w);
    b.free(conv_out);

    let out = b.add(&folded, &skip);
    b.free(folded);
    b.free(skip);
    out
}

/// `DepthToSpaceUpsample` (`residual=False`, this checkpoint's only variant):
/// conv branch, depth-to-space'd, then (only when `stride.t == 2`) the first
/// upsampled frame is dropped (`x[:,:,1:,:,:]`, upstream).
fn upsample(b: &mut Builder3d, prefix: &str, c_in: u32, stride: Stride, mult: u32, x: &T3) -> T3 {
    let prod = stride.t * stride.h * stride.w;
    let cout_conv = (prod * c_in) / mult;
    let conv_out = causal_conv3(b, &format!("{prefix}.conv"), cout_conv, x, false);
    let folded = b.depth_to_space(&conv_out, stride.t, stride.h, stride.w);
    b.free(conv_out);
    if stride.t == 2 {
        let cropped = b.time_slice(&folded, 1, folded.t - 1);
        b.free(folded);
        cropped
    } else {
        folded
    }
}

fn new_gpu(device: Option<&str>) -> Gpu {
    match device {
        Some("cpu") => Gpu::new_cpu(&KERNELS),
        Some("gpu") | Some("wgpu") => Gpu::new_wgpu(&KERNELS),
        _ => Gpu::new(&KERNELS),
    }
}

/// Read one of a graph's named stage buffers.
fn read_named(gpu: &Gpu, v: &[(String, DeviceBuffer, usize)], name: &str) -> Option<Vec<f32>> {
    v.iter().find(|(n, _, _)| n == name).map(|(_, b, l)| gpu.read(b, *l))
}

fn per_channel(tensors: &Tensors, name: &str) -> Vec<f32> {
    tensors.get(name).unwrap_or_else(|| panic!("ltxv vae: missing tensor {name}")).1.clone()
}

/// The VAE's own `per_channel_statistics` as `(mean-of-means, std-of-means)`,
/// one entry per latent channel.
///
/// These are what map between the DIFFUSION latent space (what the DiT and
/// the samplers work in) and the VAE's own (what the decoder's `z*std + mean`
/// preamble consumes, and what both latent upscalers were trained on - see
/// [`crate::upsampler::upsample_video`]). Exposed because the upscalers need
/// them outside this module; the encode/decode graphs read the same two
/// tensors directly.
pub fn per_channel_statistics(tensors: &Tensors) -> (Vec<f32>, Vec<f32>) {
    (per_channel(tensors, "per_channel_statistics.mean-of-means"), per_channel(tensors, "per_channel_statistics.std-of-means"))
}

/// The encode graph for a fixed clip size, with weights resident. Whole-clip,
/// unchunked (see this module's header) - `frames` must be `1 + 8k`.
pub struct LtxVaeEncoder {
    gpu: Gpu,
    steps: Vec<Step>,
    x_in: DeviceBuffer,
    frames: u32,
    h: u32,
    w: u32,
    out: DeviceBuffer,
    out_len: usize,
    stages: Vec<(String, DeviceBuffer, usize)>,
    taps: Vec<(String, DeviceBuffer, usize)>,
    latent_frames: u32,
}

impl LtxVaeEncoder {
    /// Build the encode graph for a `[3, frames, h, w]` clip (`frames = 1 +
    /// 8k`, `h`/`w` multiples of 32).
    pub fn build(cfg: &LtxVaeConfig, tensors: &Tensors, frames: u32, h: u32, w: u32, device: Option<&str>) -> LtxVaeEncoder {
        let lat_t = cfg.latent_frames(frames).expect("frames must be 1+8k");
        assert!(h.is_multiple_of(32) && w.is_multiple_of(32), "{h}x{w} is not a multiple of 32");
        let eps = cfg.pixel_norm_eps;

        let gpu = new_gpu(device);
        let mut b = Builder3d::new(&gpu, tensors, taps_enabled());

        let p = cfg.patch_size;
        let (pc, ph_, pw_) = (3 * p * p, h / p, w / p);
        let x_in = gpu.storage(pc as u64 * frames as u64 * ph_ as u64 * pw_ as u64);
        let patched = T3 { buf: x_in.clone(), c: pc, t: frames, h: ph_, w: pw_ };

        let mut cur = causal_conv3(&mut b, "encoder.conv_in", cfg.latent_channels, &patched, true);
        b.tap("enc.conv_in", &cur);
        let mut c = cfg.latent_channels;
        for (i, blk) in cfg.enc_blocks().iter().enumerate() {
            let p = format!("encoder.down_blocks.{i}");
            let next = match *blk {
                EncBlock::Res(ResX { n }) => res_x(&mut b, &p, n, c, &cur, true, eps),
                EncBlock::Down { stride, mult } => {
                    let y = downsample(&mut b, &p, c, stride, mult, &cur);
                    c *= mult;
                    y
                }
            };
            b.free(cur);
            cur = next;
            b.tap(&format!("enc.down_blocks.{i}"), &cur);
        }
        assert_eq!(c, cfg.bottleneck, "encoder channel walk ended at {c}, expected {}", cfg.bottleneck);

        let normed = b.pixel_norm(&cur, eps);
        b.free(cur);
        b.tap("enc.conv_norm_out", &normed);
        let act = b.silu(&normed);
        b.free(normed);
        let moments = causal_conv3(&mut b, "encoder.conv_out", cfg.latent_channels + 1, &act, true);
        b.free(act);
        b.tap("enc.conv_out", &moments);

        let mean = b.chan_slice(&moments, 0, cfg.latent_channels);
        let log_var = b.chan_slice(&moments, cfg.latent_channels, 1);

        let neg_mean: Vec<f32> = per_channel(tensors, "per_channel_statistics.mean-of-means").iter().map(|v| -v).collect();
        let inv_std: Vec<f32> = per_channel(tensors, "per_channel_statistics.std-of-means").iter().map(|v| 1.0 / v).collect();
        let centred = b.add_chan("ltxv.enc.neg_mean", &neg_mean, &mean);
        let latent = b.scale_chan("ltxv.enc.inv_std", &inv_std, &centred);
        b.free(centred);

        let stages = vec![
            ("moments".to_string(), moments.buf.clone(), moments.len() as usize),
            ("mean".to_string(), mean.buf.clone(), mean.len() as usize),
            ("log_var".to_string(), log_var.buf.clone(), log_var.len() as usize),
        ];
        let out_len = latent.len() as usize;
        let (steps, taps) = b.finish();
        LtxVaeEncoder {
            gpu,
            steps,
            x_in,
            frames,
            h,
            w,
            out: latent.buf,
            out_len,
            stages,
            taps,
            latent_frames: lat_t,
        }
    }

    /// Encode raw `[3, frames, h, w]` pixels (row-major, `[-1,1]`) into the
    /// NORMALISED latent `[128, 1+k, h/32, w/32]`. `patchify` runs on the host
    /// before upload - see [`crate::patchify`].
    pub fn encode(&self, video: &[f32]) -> Vec<f32> {
        let want = (3 * self.frames * self.h * self.w) as usize;
        assert_eq!(video.len(), want, "encode: {} values, expected {want}", video.len());
        let patched = patchify::patchify(video, 3, self.frames as usize, self.h as usize, self.w as usize, 4, 4);
        self.gpu.write_f32(&self.x_in, &patched);
        self.gpu.submit(&[], &self.steps);
        self.gpu.read(&self.out, self.out_len)
    }

    /// A boundary tensor of the last [`LtxVaeEncoder::encode`]: `moments`,
    /// `mean` or `log_var`.
    pub fn read_stage(&self, name: &str) -> Option<Vec<f32>> {
        read_named(&self.gpu, &self.stages, name)
    }

    /// A per-block tap (only recorded under `BRAIN_LTXV_VAE_TAPS`).
    pub fn read_tap(&self, name: &str) -> Option<Vec<f32>> {
        read_named(&self.gpu, &self.taps, name)
    }

    /// Latent frames this graph produces.
    pub fn latent_frames(&self) -> u32 {
        self.latent_frames
    }
}

/// The decode graph for a fixed latent size, with weights resident.
/// Whole-clip, unchunked.
pub struct LtxVaeDecoder {
    gpu: Gpu,
    steps: Vec<Step>,
    z_in: DeviceBuffer,
    in_len: usize,
    out: DeviceBuffer,
    out_len: usize,
    frames: u32,
    h: u32,
    w: u32,
    stages: Vec<(String, DeviceBuffer, usize)>,
    taps: Vec<(String, DeviceBuffer, usize)>,
}

impl LtxVaeDecoder {
    /// Build the decode graph for a `[128, lat_t, lh, lw]` latent.
    pub fn build(cfg: &LtxVaeConfig, tensors: &Tensors, lat_t: u32, lh: u32, lw: u32, device: Option<&str>) -> LtxVaeDecoder {
        assert!(lat_t >= 1, "a latent needs at least one frame");
        let frames = 1 + 8 * (lat_t - 1);
        let (h, w) = (lh * 32, lw * 32);
        let eps = cfg.pixel_norm_eps;

        let gpu = new_gpu(device);
        let mut b = Builder3d::new(&gpu, tensors, taps_enabled());
        let z_in = gpu.storage(cfg.latent_channels as u64 * lat_t as u64 * lh as u64 * lw as u64);
        let z = T3 { buf: z_in.clone(), c: cfg.latent_channels, t: lat_t, h: lh, w: lw };

        let std = per_channel(tensors, "per_channel_statistics.std-of-means");
        let mean = per_channel(tensors, "per_channel_statistics.mean-of-means");
        let scaled = b.scale_chan("ltxv.dec.std", &std, &z);
        let denorm = b.add_chan("ltxv.dec.mean", &mean, &scaled);
        b.free(scaled);
        b.tap("z_denorm", &denorm);

        let mut cur = causal_conv3(&mut b, "decoder.conv_in", cfg.bottleneck, &denorm, false);
        b.tap("dec.conv_in", &cur);
        let mut c = cfg.bottleneck;
        for (i, blk) in cfg.dec_blocks().iter().enumerate() {
            let p = format!("decoder.up_blocks.{i}");
            let next = match *blk {
                DecBlock::Res(ResX { n }) => res_x(&mut b, &p, n, c, &cur, false, eps),
                DecBlock::Up { stride, mult } => {
                    let y = upsample(&mut b, &p, c, stride, mult, &cur);
                    c /= mult;
                    y
                }
            };
            b.free(cur);
            cur = next;
            b.tap(&format!("dec.up_blocks.{i}"), &cur);
        }
        assert_eq!(c, cfg.latent_channels, "decoder channel walk ended at {c}, expected {}", cfg.latent_channels);

        let normed = b.pixel_norm(&cur, eps);
        b.free(cur);
        b.tap("dec.conv_norm_out", &normed);
        let act = b.silu(&normed);
        b.free(normed);
        let p = cfg.patch_size;
        let conv_out = causal_conv3(&mut b, "decoder.conv_out", 3 * p * p, &act, false);
        b.free(act);
        b.tap("dec.conv_out", &conv_out);

        let stages = vec![("z_denorm".to_string(), denorm.buf.clone(), denorm.len() as usize)];
        let out_len = conv_out.len() as usize;
        let (steps, taps) = b.finish();
        LtxVaeDecoder {
            gpu,
            steps,
            z_in,
            in_len: (cfg.latent_channels * lat_t * lh * lw) as usize,
            out: conv_out.buf,
            out_len,
            frames,
            h,
            w,
            stages,
            taps,
        }
    }

    /// Decode a NORMALISED latent `[128, lat_t, lh, lw]` into `[3, frames,
    /// lh*32, lw*32]`. No clamp is applied - upstream clamps to `[-1,1]`
    /// outside the model. `unpatchify` runs on the host after readback.
    pub fn decode(&self, latent: &[f32]) -> Vec<f32> {
        assert_eq!(latent.len(), self.in_len, "decode: {} values, expected {}", latent.len(), self.in_len);
        self.gpu.write_f32(&self.z_in, latent);
        self.gpu.submit(&[], &self.steps);
        let raw = self.gpu.read(&self.out, self.out_len);
        let p = 4usize;
        patchify::unpatchify(&raw, 3, self.frames as usize, (self.h / 4) as usize, (self.w / 4) as usize, p, p)
    }

    /// A boundary tensor of the last [`LtxVaeDecoder::decode`]: `z_denorm`.
    pub fn read_stage(&self, name: &str) -> Option<Vec<f32>> {
        read_named(&self.gpu, &self.stages, name)
    }

    /// A per-block tap (only recorded under `BRAIN_LTXV_VAE_TAPS`).
    pub fn read_tap(&self, name: &str) -> Option<Vec<f32>> {
        read_named(&self.gpu, &self.taps, name)
    }

    /// Video frames this graph produces.
    pub fn frames(&self) -> u32 {
        self.frames
    }

    /// The recorded decode graph, e.g. for `ltxv_bench`'s per-kernel-kind
    /// profile (the `wan::WanVaeDecoder::steps` precedent) - not used by
    /// [`LtxVaeDecoder::decode`] itself, which submits `self.steps` directly.
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// The device this graph was built on.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }
}

// -------------------------------------------------------- tiled decode ---

/// The VAE's own stride, in pixel units per latent cell: `(time, height,
/// width)`. Upstream's `VIDEO_SCALE_FACTORS`.
pub const VIDEO_SCALE: (usize, usize, usize) = (8, 32, 32);

/// A decode tile layout in **pixel / frame** units - upstream's
/// `TileSizeConfig`, restricted to the three axes this decoder has.
///
/// `0` on an axis means that axis is not tiled (upstream's
/// `DimensionSizeConfig.tile_size == 0`), which is also what a tile larger
/// than the axis degenerates to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LtxVaeTiling {
    /// `(tile, overlap)` in frames.
    pub frames: (u32, u32),
    /// `(tile, overlap)` in pixels.
    pub height: (u32, u32),
    /// `(tile, overlap)` in pixels.
    pub width: (u32, u32),
}

impl LtxVaeTiling {
    /// Upstream's `_CONV_AUTO_LONG_SIDE` - the long spatial edge's tile and
    /// overlap for the CONV video decoder, in pixels
    /// (`ltx_pipelines/utils/helpers.py`). This is the reference's own
    /// number, not a guess: it is what `AUTO_TILING` resolves to for every
    /// conv-VAE pipeline upstream ships.
    pub const AUTO_LONG_SIDE: (u32, u32) = (768, 64);

    /// Upstream's `_CONV_AUTO_FRAMES` - the temporal tile and overlap, in
    /// frames. At the clip lengths this port runs (25 frames = 4 latent
    /// frames) 80 covers the whole axis, so the temporal split degenerates to
    /// one tile and only the spatial axes actually split. That is upstream's
    /// behaviour too, not a simplification.
    pub const AUTO_FRAMES: (u32, u32) = (80, 24);

    /// The reference's aspect-coupled auto layout
    /// (`TileSizeConfig.from_long_side`): the long edge gets
    /// [`Self::AUTO_LONG_SIDE`] and the short edge is scaled to it **in
    /// latent units**, not pixel units.
    ///
    /// Doing the `round` on the latent grid is upstream's own documented
    /// correction ("pixel-space `round` + ceil-snap would bias the short axis
    /// up by almost one latent, e.g. 680 -> 704 vs 672"), so it is
    /// transcribed rather than re-derived. For 1920x1088 this yields a
    /// 448x768 tile with a 64 px overlap on both axes.
    pub fn auto(height: u32, width: u32) -> LtxVaeTiling {
        let (size_px, overlap_px) = Self::AUTO_LONG_SIDE;
        let span = height.max(width);
        let axis = |axis_len: u32, factor: u32| -> u32 {
            let axis_lat = axis_len / factor;
            let long_lat = span / factor;
            let size_lat = size_px / factor;
            let overlap_lat = overlap_px / factor;
            let lower = 2.max(overlap_lat + 1);
            let tile_lat = lower.max(round_half_to_even(size_lat as f64 * axis_lat as f64 / long_lat as f64));
            let min_legal = (2 * factor).max(overlap_px + factor);
            (tile_lat * factor).max(min_legal)
        };
        LtxVaeTiling {
            frames: Self::AUTO_FRAMES,
            height: (axis(height, VIDEO_SCALE.1 as u32), overlap_px),
            width: (axis(width, VIDEO_SCALE.2 as u32), overlap_px),
        }
    }

    /// Build the latent-grid tile cover for a `[C, lat_t, lh, lw]` latent.
    ///
    /// Mirrors `TileSizeConfig.to_splitters` (pixel sizes divided by the VAE
    /// factor, floored at `max(2, overlap + 1)`) with `causal_temporal=True`,
    /// which is the default `ConvVideoDecoder._prepare_tiles` uses.
    pub fn plan(&self, lat_t: u32, lh: u32, lw: u32) -> vae::tiling3d::TilePlan3d {
        let (ft, fh, fw) = VIDEO_SCALE;
        let axis = |(tile_px, overlap_px): (u32, u32), factor: usize| -> (usize, usize) {
            let overlap = overlap_px as usize / factor;
            let size = tile_px as usize / factor;
            (2.max(overlap + 1).max(size), overlap)
        };
        let (t_tile, t_over) = axis(self.frames, ft);
        let (h_tile, h_over) = axis(self.height, fh);
        let (w_tile, w_over) = axis(self.width, fw);
        vae::tiling3d::TilePlan3d {
            t: vae::tiling3d::AxisPlan::temporal(lat_t as usize, t_tile, t_over, ft),
            h: vae::tiling3d::AxisPlan::spatial(lh as usize, h_tile, h_over, fh),
            w: vae::tiling3d::AxisPlan::spatial(lw as usize, w_tile, w_over, fw),
        }
    }
}

/// Python's `round` is round-half-to-EVEN, and `TileSizeConfig.from_long_side`
/// uses it. Ported rather than substituted with `f64::round` (which rounds
/// half away from zero) so a config that happens to land on a tie picks the
/// same tile as upstream.
fn round_half_to_even(v: f64) -> u32 {
    let f = v.floor();
    let diff = v - f;
    // Round up on a strict majority, and on a tie only when doing so lands
    // on the even neighbour - which is exactly `floor` being odd.
    let up = diff > 0.5 || (diff == 0.5 && (f as i64) % 2 != 0);
    let n = if up { f + 1.0 } else { f };
    n.max(0.0) as u32
}

/// Peak decode VRAM tracks the OUTPUT PIXEL VOLUME, `frames x H x W`, and is
/// close to independent of how that volume splits between the temporal and
/// spatial axes. Measured on one Tesla P40 (24576 MiB), real
/// `ltx-2.5-video-vae-conv-bf16.safetensors`, whole (un-tiled) path, peak
/// sampled at 200 ms:
///
/// | shape | Mpx | peak MiB |
/// |---|---:|---:|
/// | 9f @1920x1088 | 18.80 | 15186 |
/// | 25f @1280x704 | 22.53 | 16564 |
/// | 9f @2560x1440 | 33.18 | 23785 |
/// | 33f @1408x768 | 35.68 | 23813 |
/// | 17f @1920x1088 | 35.51 | 24264 |
/// | 25f @1920x1088 | 52.22 | **out of memory** |
///
/// Three shapes near 35 Mpx with very different frame/area splits all land
/// within 500 MiB of each other, which is what "the product, not either
/// axis" means numerically. The fit is roughly `3.9 GiB + 598 MiB/Mpx`, so a
/// 24 GiB card runs out just past 35 Mpx.
///
/// This is the whole path's ceiling in pixels, set below the measured cliff
/// with real margin - `24_000_000` keeps every shape this port ships today
/// (9 frames at 1080p = 18.8 Mpx, 25 at 720p = 22.5 Mpx) on the exact,
/// un-tiled path and sends 25 frames at 1080p to the tiled one. It is a
/// P40-sized constant and says so; `BRAIN_LTXV_VAE_TILE` overrides the
/// decision either way.
pub const WHOLE_DECODE_MAX_PIXELS: u64 = 24_000_000;

/// Whether a decode of this output shape should take the tiled path.
///
/// `BRAIN_LTXV_VAE_TILE=1`/`0` forces tiling on/off; anything else (or unset)
/// is the measured [`WHOLE_DECODE_MAX_PIXELS`] policy. Forcing it ON at a
/// shape that already fits is the supported way to compare the two paths.
pub fn should_tile(frames: u32, h: u32, w: u32) -> bool {
    match std::env::var("BRAIN_LTXV_VAE_TILE").ok().as_deref() {
        Some("1") | Some("on") | Some("true") => return true,
        Some("0") | Some("off") | Some("false") => return false,
        _ => {}
    }
    (frames as u64) * (h as u64) * (w as u64) > WHOLE_DECODE_MAX_PIXELS
}

/// Decodes a `[128, lat_t, lh, lw]` latent as a cover of **overlapping
/// tiles**, one device-resident graph at a time, blending the decoded pixel
/// tiles with the reference's trapezoidal masks.
///
/// Swedish Embedded AB implements memory-bounded video-model inference for
/// its clients. If your team needs expertise in running large diffusion and
/// autoencoder models inside a fixed VRAM budget, you can procure our
/// services by sending an email to info@swedishembedded.com.
///
/// # What this is and is not
///
/// It is **not** an exact factorisation of [`LtxVaeDecoder`]. This decoder's
/// spatial receptive field is roughly 15 latent cells wide (summing every
/// kernel-3 conv at the resolution it runs at: ~6 cells at the latent grid,
/// ~2.5 at 2x, ~5.5 at 4x, ~1.1 at 8x), while a 1080p latent is only 34 cells
/// tall - so no overlap that still saves memory can cover it, and a
/// halo-and-crop scheme like `imaging::tiling`'s cannot be exact either.
/// Blending is what upstream ships for precisely this reason. A tile's
/// interior is exact; the seam is a weighted average of two tiles that saw
/// different context, and the trapezoidal ramp makes that a gradient rather
/// than an edge.
///
/// What IS exact, and gated as such:
///
/// * a plan that yields ONE tile decodes bit-identically to
///   [`LtxVaeDecoder`] (`crates/ltxv/tests/vae_tiling.rs`);
/// * the blend reconstructs a known volume to `< 1e-5`
///   (`vae::tiling3d`'s own `the_blend_reconstructs_a_known_volume_exactly`),
///   so mask/slice/divisor bugs cannot hide behind the receptive-field
///   approximation.
///
/// # One graph per tile SHAPE, not per tile
///
/// A tile's graph is built, used for every tile of that shape, and dropped
/// before the next shape's is built - the same "fresh resources per unit of
/// work" pattern `RealDit::forward_q_streamed` uses, and the reason peak VRAM
/// is one tile's, not the clip's. Tiles are grouped by shape rather than
/// rebuilt per tile because a `split_by_size` cover has at most four distinct
/// spatial shapes (interior, short last row, short last column, and their
/// corner) however many tiles it has.
pub struct LtxVaeTiledDecoder<'a> {
    cfg: LtxVaeConfig,
    /// BORROWED, not owned: the whole-clip decode above needs the weights
    /// only until its single graph is recorded, but this path needs them
    /// across one graph build per distinct tile shape - and a caller that
    /// decodes several clips against the same weights (`pipeline::upscale`'s
    /// per-segment decode) must not pay a ~3 GB host copy per clip to do it.
    tensors: &'a Tensors,
    lat_t: u32,
    lh: u32,
    lw: u32,
    device: Option<String>,
    plan: vae::tiling3d::TilePlan3d,
}

impl<'a> LtxVaeTiledDecoder<'a> {
    /// Build a tiled decoder for a `[128, lat_t, lh, lw]` latent under
    /// `tiling`. Weights are retained on the HOST (no device graph exists
    /// until [`LtxVaeTiledDecoder::decode`] runs), so constructing this costs
    /// no VRAM at all.
    pub fn new(cfg: &LtxVaeConfig, tensors: &'a Tensors, lat_t: u32, lh: u32, lw: u32, device: Option<&str>, tiling: LtxVaeTiling) -> LtxVaeTiledDecoder<'a> {
        assert!(lat_t >= 1, "a latent needs at least one frame");
        let plan = tiling.plan(lat_t, lh, lw);
        LtxVaeTiledDecoder { cfg: *cfg, tensors, lat_t, lh, lw, device: device.map(str::to_string), plan }
    }

    /// The same, with upstream's aspect-coupled auto layout for the OUTPUT
    /// pixel size this latent decodes to.
    pub fn auto(cfg: &LtxVaeConfig, tensors: &'a Tensors, lat_t: u32, lh: u32, lw: u32, device: Option<&str>) -> LtxVaeTiledDecoder<'a> {
        let tiling = LtxVaeTiling::auto(lh * VIDEO_SCALE.1 as u32, lw * VIDEO_SCALE.2 as u32);
        Self::new(cfg, tensors, lat_t, lh, lw, device, tiling)
    }

    /// The tile cover this decoder will run.
    pub fn plan(&self) -> &vae::tiling3d::TilePlan3d {
        &self.plan
    }

    /// Video frames this decoder produces - the same `1 + 8(lat_t - 1)` rule
    /// [`LtxVaeDecoder::frames`] follows.
    pub fn frames(&self) -> u32 {
        1 + 8 * (self.lat_t - 1)
    }

    /// Decode a NORMALISED latent `[128, lat_t, lh, lw]` into `[3, frames,
    /// lh*32, lw*32]`, one tile at a time.
    ///
    /// `on_tile(done, total)` is called after each tile so a caller can
    /// report progress on what is, at 1080p, a multi-minute stage.
    pub fn decode_with(&self, latent: &[f32], mut on_tile: impl FnMut(usize, usize)) -> Vec<f32> {
        let c = self.cfg.latent_channels as usize;
        let want = c * self.lat_t as usize * self.lh as usize * self.lw as usize;
        assert_eq!(latent.len(), want, "tiled decode: {} values, expected {want}", latent.len());

        let tiles = self.plan.tiles();
        let total = tiles.len();
        let mut blender = vae::tiling3d::Blender::new(&self.plan, 3);

        // Group by latent tile SHAPE so one graph serves every tile of that
        // shape; `BTreeMap` keeps the order deterministic run to run, which a
        // `HashMap` would not, and a decode that reorders its own float
        // accumulation is a decode whose output is not reproducible.
        let mut by_shape: std::collections::BTreeMap<(usize, usize, usize), Vec<usize>> = std::collections::BTreeMap::new();
        for (i, t) in tiles.iter().enumerate() {
            by_shape.entry((t.t.src_len(), t.h.src_len(), t.w.src_len())).or_default().push(i);
        }

        let mut done = 0usize;
        for ((st, sh, sw), idxs) in by_shape {
            let dec = LtxVaeDecoder::build(&self.cfg, self.tensors, st as u32, sh as u32, sw as u32, self.device.as_deref());
            for i in idxs {
                let tile = tiles[i];
                let sub = self.slice_latent(latent, tile);
                let pixels = dec.decode(&sub);
                blender.add(tile, &pixels);
                done += 1;
                on_tile(done, total);
            }
            // Explicit: the whole point is that this tile shape's device
            // buffers are gone before the next shape's are allocated.
            drop(dec);
        }
        blender.finish()
    }

    /// [`LtxVaeTiledDecoder::decode_with`] with no progress callback.
    pub fn decode(&self, latent: &[f32]) -> Vec<f32> {
        self.decode_with(latent, |_, _| {})
    }

    /// Cut one tile's `[C, t, h, w]` sub-volume out of the full latent.
    fn slice_latent(&self, latent: &[f32], tile: vae::tiling3d::Tile3d<'_>) -> Vec<f32> {
        let c = self.cfg.latent_channels as usize;
        let (lt, lh, lw) = (self.lat_t as usize, self.lh as usize, self.lw as usize);
        let (t0, t1) = tile.t.src;
        let (h0, h1) = tile.h.src;
        let (w0, w1) = tile.w.src;
        let (tt, th, tw) = (t1 - t0, h1 - h0, w1 - w0);
        let mut out = vec![0.0f32; c * tt * th * tw];
        for ci in 0..c {
            for ti in 0..tt {
                for hi in 0..th {
                    let src = (((ci * lt + t0 + ti) * lh) + h0 + hi) * lw + w0;
                    let dst = (((ci * tt + ti) * th) + hi) * tw;
                    out[dst..dst + tw].copy_from_slice(&latent[src..src + tw]);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_matches_the_real_checkpoint_config() {
        let cfg = LtxVaeConfig::conv25();
        let enc = cfg.enc_blocks();
        assert_eq!(enc.len(), 9);
        assert_eq!(enc[0], EncBlock::Res(ResX { n: 4 }));
        assert_eq!(enc[1], EncBlock::Down { stride: ST_SPACE, mult: 2 });
        assert_eq!(enc[3], EncBlock::Down { stride: ST_TIME, mult: 2 });
        assert_eq!(enc[7], EncBlock::Down { stride: ST_ALL, mult: 1 });

        let dec = cfg.dec_blocks();
        assert_eq!(dec.len(), 9);
        assert_eq!(dec[0], DecBlock::Res(ResX { n: 2 }));
        assert_eq!(dec[1], DecBlock::Up { stride: ST_ALL, mult: 2 });
        assert_eq!(dec[3], DecBlock::Up { stride: ST_ALL, mult: 1 });
        assert_eq!(dec[7], DecBlock::Up { stride: ST_SPACE, mult: 2 });
    }

    /// 170 is the real checkpoint's tensor count (84 encoder + 84 decoder + 2
    /// shared `per_channel_statistics`); a schedule that drifts by one block
    /// changes this number.
    #[test]
    fn manifest_counts_the_shipped_checkpoint() {
        let m = LtxVaeConfig::conv25().tensor_manifest();
        assert_eq!(m.len(), 170, "manifest has {} tensors", m.len());
        let names: std::collections::HashSet<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), m.len(), "duplicate tensor name in the manifest");
        assert!(names.contains("encoder.conv_in.conv.weight"));
        assert!(names.contains("encoder.down_blocks.1.conv.conv.weight"));
        assert!(names.contains("decoder.up_blocks.1.conv.conv.weight"));
        assert!(names.contains("per_channel_statistics.mean-of-means"));

        // Shapes read off the real header, pinned so a channel-walk bug is
        // caught by a plain assert rather than only by a real-weight run.
        let get = |n: &str| m.iter().find(|(k, _)| k == n).unwrap().1.clone();
        assert_eq!(get("encoder.conv_in.conv.weight"), vec![128, 48, 3, 3, 3]);
        assert_eq!(get("encoder.conv_out.conv.weight"), vec![129, 1024, 3, 3, 3]);
        assert_eq!(get("encoder.down_blocks.1.conv.conv.weight"), vec![64, 128, 3, 3, 3]);
        assert_eq!(get("encoder.down_blocks.7.conv.conv.weight"), vec![128, 1024, 3, 3, 3]);
        assert_eq!(get("decoder.conv_in.conv.weight"), vec![1024, 128, 3, 3, 3]);
        assert_eq!(get("decoder.conv_out.conv.weight"), vec![48, 128, 3, 3, 3]);
        assert_eq!(get("decoder.up_blocks.1.conv.conv.weight"), vec![4096, 1024, 3, 3, 3]);
        assert_eq!(get("decoder.up_blocks.3.conv.conv.weight"), vec![4096, 512, 3, 3, 3]);
        assert_eq!(get("decoder.up_blocks.5.conv.conv.weight"), vec![512, 512, 3, 3, 3]);
        assert_eq!(get("decoder.up_blocks.7.conv.conv.weight"), vec![512, 256, 3, 3, 3]);
    }

    // ------------------------------------------------------------ tiling

    /// `TileSizeConfig.from_long_side` with upstream's own conv-VAE
    /// constants, evaluated by hand for the two shapes this port runs:
    /// 1080p (span 1920, so width takes the full 768 and height scales to
    /// `round(24 * 34/60) = 14` latent cells = 448 px) and 720p (span 1280,
    /// width 768, height `round(24 * 22/40) = 13` = 416 px).
    #[test]
    fn the_auto_tiling_matches_the_reference_long_side_layout() {
        let a = LtxVaeTiling::auto(1088, 1920);
        assert_eq!(a.frames, (80, 24));
        assert_eq!(a.height, (448, 64));
        assert_eq!(a.width, (768, 64));

        let b = LtxVaeTiling::auto(704, 1280);
        assert_eq!(b.height, (416, 64));
        assert_eq!(b.width, (768, 64));

        // A square never scales the short edge, because there isn't one.
        let s = LtxVaeTiling::auto(1024, 1024);
        assert_eq!(s.height, s.width);
    }

    /// Python's `round` is half-to-even and upstream's tile sizing calls it,
    /// so a tie must pick the even neighbour, not the larger one.
    #[test]
    fn tile_sizing_rounds_half_to_even_like_python() {
        assert_eq!(round_half_to_even(2.5), 2);
        assert_eq!(round_half_to_even(3.5), 4);
        assert_eq!(round_half_to_even(13.6), 14);
        assert_eq!(round_half_to_even(13.2), 13);
    }

    /// At 25 frames the temporal axis is 4 latent frames, well under the
    /// reference's 80-frame (10-latent) temporal tile, so only the spatial
    /// axes split - 3 x 3 tiles. The pixel cover must be exact and the blend
    /// masks must partition unity, or the stitched clip has seams.
    #[test]
    fn the_1080p_plan_splits_space_only_and_covers_the_clip() {
        let plan = LtxVaeTiling::auto(1088, 1920).plan(4, 34, 60);
        assert_eq!(plan.t.len(), 1, "25 frames must not need a temporal split");
        assert_eq!((plan.h.len(), plan.w.len()), (3, 3));
        assert_eq!(plan.tiles().len(), 9);
        assert_eq!(plan.out_shape(), (25, 1088, 1920));
        assert!(plan.masks_are_complementary(), "unity error t/h/w = {}/{}/{}", plan.t.unity_error(), plan.h.unity_error(), plan.w.unity_error());
        let waste = plan.overlap_waste();
        assert!((1.19..1.20).contains(&waste), "overlap waste {waste}");
    }

    /// A long clip DOES split temporally, and the causal split plus the
    /// `1 + 8k` frame mapping must still cover exactly - this is the case the
    /// 25-frame plan above never exercises.
    #[test]
    fn a_long_clip_also_splits_the_temporal_axis_and_still_covers_it() {
        // 121 frames = 16 latent frames, over the 10-latent temporal tile.
        let plan = LtxVaeTiling::auto(704, 1280).plan(16, 22, 40);
        assert!(plan.t.len() > 1, "16 latent frames must split under a 10-frame tile");
        assert_eq!(plan.out_shape().0, 121);
        assert!(plan.masks_are_complementary(), "temporal unity error {}", plan.t.unity_error());
    }

    /// The policy that decides which path a real generation takes. Pinned
    /// against the measured table on [`WHOLE_DECODE_MAX_PIXELS`]: everything
    /// this port ships today stays on the exact path, and the shape that
    /// out-of-memories does not.
    #[test]
    fn the_tiling_policy_keeps_todays_shapes_on_the_exact_path() {
        if std::env::var("BRAIN_LTXV_VAE_TILE").is_ok() {
            return; // an explicit override is being tested elsewhere
        }
        assert!(!should_tile(9, 1088, 1920), "9 frames at 1080p fits whole (measured 15186 MiB)");
        assert!(!should_tile(25, 704, 1280), "25 frames at 720p fits whole (measured 16564 MiB)");
        assert!(should_tile(25, 1088, 1920), "25 frames at 1080p is a measured out-of-memory");
        assert!(should_tile(49, 704, 1280), "49 frames at 720p is 44.2 Mpx, past the measured cliff");
    }

    #[test]
    fn latent_frame_rule() {
        let cfg = LtxVaeConfig::conv25();
        assert_eq!(cfg.latent_frames(9), Some(2));
        assert_eq!(cfg.latent_frames(17), Some(3));
        assert_eq!(cfg.latent_frames(1), Some(1));
        assert_eq!(cfg.latent_frames(8), None);
    }
}
