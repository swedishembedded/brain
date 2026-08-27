// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `AutoencoderKL` decoder as a pre-recorded brain kernel graph.
//!
//! Mirrors the diffusers `Decoder`: [1×1 `post_quant_conv` when the config
//! enables it — FLUX.2] → `conv_in` → mid block (resnet, self-attn, resnet) →
//! up blocks (`layers_per_block+1` resnets each, nearest-neighbour doubling + conv on
//! all but the last block) → `conv_norm_out` → SiLU → `conv_out`.
//! The graph is built once for a fixed input `[latent_ch, h, w]`; `decode`
//! uploads the latent, submits, and reads the image `[out_ch, H, W]`.
//!
//! Reuse: identical conv/GroupNorm/SiLU/add/upsample/attention kernels as the
//! DIAMOND UNet (`crates/wm-diamond/src/model.rs`), which validates them. VAE
//! specifics vs DIAMOND: static (non-conditioned) affine GroupNorm with **32
//! groups** and **eps 1e-6**; the mid-block attention is **single-head**
//! (`head_dim = C`, scale `1/√C`) with the residual added to the **pre-norm**
//! input; and `to_q/to_k/to_v` are fused into one 1×1 qkv conv at build time so
//! the exact DIAMOND attention path (`nchw_nlc` → bidir scores/softmax/apply →
//! `nlc_nchw`) applies unchanged.

use gpu_core::{DeviceBuffer, Gpu, Step};

use crate::blocks::{BlockNames, Builder};
use crate::config::VaeConfig;

pub use crate::blocks::{Tensors, KERNELS};

/// `BRAIN_VAE_TAPS=1` records every block output for parity debugging (pins
/// buffers, so it disables the activation pool).
fn taps_enabled() -> bool {
    std::env::var("BRAIN_VAE_TAPS").is_ok()
}

/// A decode graph for a fixed input latent size, with weights resident.
pub struct VaeDecoder {
    gpu: Gpu,
    cfg: VaeConfig,
    steps: Vec<Step>,
    z_in: DeviceBuffer,
    out: DeviceBuffer,
    out_len: usize,
    taps: Vec<(String, DeviceBuffer, usize)>,
    device_bytes: u64,
}

impl VaeDecoder {
    /// Build the decode graph for an input latent `[latent_ch, h, w]` and upload
    /// all decoder weights. `device`: `Some("cpu")` | `Some("gpu")` | `None`.
    pub fn from_diffusers(cfg: VaeConfig, tensors: &Tensors, h: u32, w: u32, device: Option<&str>) -> VaeDecoder {
        VaeDecoder::build(Gpu::open(device, &KERNELS), cfg, tensors, h, w)
    }

    /// [`VaeDecoder::from_diffusers`] on an EXISTING device: a second handle
    /// onto `gpu` (same adapter, queue and already-compiled pipelines) rather
    /// than a fresh one.
    ///
    /// A pipeline that encodes several reference images and then decodes a
    /// result builds several of these graphs in one generation. Each
    /// `from_diffusers` stands up its own device and recompiles every kernel;
    /// on a two-card box it also re-resolves the ambient selection, which may
    /// have moved. Sharing is explicit here for the reason AGENTS.md gives:
    /// the number of real devices a process holds stays answerable by reading
    /// the code.
    pub fn from_diffusers_on(gpu: &Gpu, cfg: VaeConfig, tensors: &Tensors, h: u32, w: u32) -> VaeDecoder {
        VaeDecoder::build(gpu.share_or_new(&KERNELS), cfg, tensors, h, w)
    }

    fn build(gpu: Gpu, cfg: VaeConfig, tensors: &Tensors, h: u32, w: u32) -> VaeDecoder {
        let mut b =
            Builder::new(&gpu, tensors, cfg.norm_eps, cfg.norm_num_groups, BlockNames::diffusers(), taps_enabled());

        let z_in = gpu.storage((cfg.latent_channels * h * w) as u64);
        let rc = cfg.reversed_channels();
        let mid_c = *cfg.block_out_channels.last().unwrap();

        // FLUX.2 post_quant_conv: 1×1 latent→latent ahead of conv_in. The shipped
        // diffusers checkpoint stores it top-level (`post_quant_conv.weight`); the
        // BFL-native layout nests it (`decoder.post_quant_conv.*`). Accept both;
        // a checkpoint with neither fails loudly in `Builder::get`.
        let zc = cfg.latent_channels;
        let pq = cfg.use_post_quant_conv.then(|| {
            let p = if tensors.contains_key("post_quant_conv.weight") {
                "post_quant_conv"
            } else {
                "decoder.post_quant_conv"
            };
            b.conv(p, zc, zc, 1, 0, h, w, &z_in)
        });

        // conv_in: latent → highest block channel. `z_in` is the persistent input,
        // never freed; `xlen` tracks the current `x` length so the previous `x` can
        // be returned to the pool once the next op has consumed it.
        let mut x = b.conv("decoder.conv_in", zc, mid_c, 3, 1, h, w, pq.as_ref().unwrap_or(&z_in));
        if let Some(p) = pq {
            b.free((zc * h * w) as u64, p); // last read was conv_in above
        }
        let mut xlen = (mid_c * h * w) as u64;
        b.tap("conv_in".into(), &x, mid_c * h * w);

        // Mid block: resnet, (optional) attention, resnet.
        let nx = b.resnet("decoder.mid_block.resnets.0", mid_c, mid_c, h, w, &x);
        b.free(xlen, x);
        x = nx;
        if cfg.mid_block_add_attention {
            let nx = b.attn("decoder.mid_block.attentions.0", mid_c, h, w, &x);
            b.free(xlen, x);
            x = nx;
            b.tap("mid_attn".into(), &x, mid_c * h * w);
        }
        let nx = b.resnet("decoder.mid_block.resnets.1", mid_c, mid_c, h, w, &x);
        b.free(xlen, x);
        x = nx;
        b.tap("mid".into(), &x, mid_c * h * w);

        // Up blocks: reversed channel schedule; upsample on all but the last.
        let n_blocks = rc.len();
        let n_res = cfg.layers_per_block + 1;
        let (mut cur_h, mut cur_w) = (h, w);
        let mut prev = mid_c;
        for (i, &out_c) in rc.iter().enumerate() {
            for r in 0..n_res {
                let cin = if r == 0 { prev } else { out_c };
                let nx = b.resnet(&format!("decoder.up_blocks.{i}.resnets.{r}"), cin, out_c, cur_h, cur_w, &x);
                b.free(xlen, x);
                x = nx;
                xlen = (out_c * cur_h * cur_w) as u64;
            }
            if i < n_blocks - 1 {
                let nx = b.upsample(out_c, cur_h, cur_w, &x);
                b.free(xlen, x);
                x = nx;
                cur_h *= 2;
                cur_w *= 2;
                xlen = (out_c * cur_h * cur_w) as u64;
                let nx = b.conv(&format!("decoder.up_blocks.{i}.upsamplers.0.conv"), out_c, out_c, 3, 1, cur_h, cur_w, &x);
                b.free(xlen, x);
                x = nx;
            }
            prev = out_c;
            b.tap(format!("up_block.{i}"), &x, out_c * cur_h * cur_w);
        }

        // Head: GroupNorm → SiLU → conv_out.
        let hn = b.gn("decoder.conv_norm_out", prev, cur_h, cur_w, &x);
        b.free(xlen, x);
        let hlen = (prev * cur_h * cur_w) as u64;
        let hs = b.silu(prev * cur_h * cur_w, &hn);
        b.free(hlen, hn);
        let out = b.conv("decoder.conv_out", prev, cfg.out_channels, 3, 1, cur_h, cur_w, &hs);
        b.free(hlen, hs);
        let out_len = (cfg.out_channels * cur_h * cur_w) as usize;

        // `z_in` is allocated here rather than through the builder, so it is
        // added explicitly: the total has to be what the GRAPH costs, not what
        // one of its two allocators happened to see.
        let device_bytes = b.allocated_bytes() + (cfg.latent_channels * h * w) as u64 * 4;
        let (steps, taps) = b.finish();
        VaeDecoder { gpu, cfg, steps, z_in, out, out_len, taps, device_bytes }
    }

    /// Decode a latent `[latent_ch·h·w]` (row-major NCHW, batch 1) into the
    /// image `[out_ch·H·W]`. Raw decode — no scaling/shift is applied here.
    pub fn decode(&self, latent: &[f32]) -> Vec<f32> {
        let bits: Vec<u32> = latent.iter().map(|v| v.to_bits()).collect();
        self.gpu.write(&self.z_in, &bits);
        self.gpu.submit(&[], &self.steps);
        self.gpu.read(&self.out, self.out_len)
    }

    /// Read a named intermediate tap after a `decode` (parity debugging).
    pub fn read_tap(&self, name: &str) -> Option<Vec<f32>> {
        self.taps.iter().find(|(n, _, _)| n == name).map(|(_, buf, len)| self.gpu.read(buf, *len))
    }

    pub fn config(&self) -> &VaeConfig {
        &self.cfg
    }

    /// The device the graph was built on (profiling / benches).
    /// Device bytes this graph holds: weights plus its resident activation
    /// set. The ground truth [`crate::decoder_device_bytes`] is gated against.
    pub fn device_bytes(&self) -> u64 {
        self.device_bytes
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The pre-recorded decode dispatch sequence (profiling / benches). Each
    /// [`Step`]'s `meta()` names its slot in [`KERNELS`].
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }
}

// ===================== encoder =====================

/// An `AutoencoderKL` **encoder** graph for a fixed image size, weights resident.
/// Mirrors [`VaeDecoder`]: `image[3,H,W] → moments[2·latent, H/8, W/8]` (mean ‖
/// logvar). `conv_out` yields the moments directly unless the config enables the
/// FLUX.2 1×1 `quant_conv`, which then maps them in place. Reuses the shared
/// [`Builder`] blocks (conv/resnet/gn/silu/attn) and the strided
/// [`Builder::conv_down`] for the diffusers `Downsample2D`.
pub struct VaeEncoder {
    gpu: Gpu,
    cfg: VaeConfig,
    steps: Vec<Step>,
    img_in: DeviceBuffer,
    out: DeviceBuffer,
    out_len: usize,
    taps: Vec<(String, DeviceBuffer, usize)>,
    device_bytes: u64,
}

impl VaeEncoder {
    /// Build the encode graph for an input image `[in_channels, h, w]` (full-res,
    /// NOT latent size) and upload all encoder weights. `device`: `Some("cpu")` |
    /// `Some("gpu")` | `None`.
    pub fn from_diffusers(cfg: VaeConfig, tensors: &Tensors, h: u32, w: u32, device: Option<&str>) -> VaeEncoder {
        VaeEncoder::build(Gpu::open(device, &KERNELS), cfg, tensors, h, w)
    }

    /// [`VaeEncoder::from_diffusers`] on an EXISTING device - see
    /// [`VaeDecoder::from_diffusers_on`] for why sharing matters here.
    pub fn from_diffusers_on(gpu: &Gpu, cfg: VaeConfig, tensors: &Tensors, h: u32, w: u32) -> VaeEncoder {
        VaeEncoder::build(gpu.share_or_new(&KERNELS), cfg, tensors, h, w)
    }

    fn build(gpu: Gpu, cfg: VaeConfig, tensors: &Tensors, h: u32, w: u32) -> VaeEncoder {
        let mut b =
            Builder::new(&gpu, tensors, cfg.norm_eps, cfg.norm_num_groups, BlockNames::diffusers(), taps_enabled());

        let img_in = gpu.storage((cfg.in_channels * h * w) as u64);
        let ch = &cfg.block_out_channels;

        // conv_in: image → block_out[0].
        let mut x = b.conv("encoder.conv_in", cfg.in_channels, ch[0], 3, 1, h, w, &img_in);
        b.tap("conv_in".into(), &x, ch[0] * h * w);

        // Down blocks: `layers_per_block` resnets each; downsample on all but last.
        let n_blocks = ch.len();
        let n_res = cfg.layers_per_block;
        let (mut cur_h, mut cur_w) = (h, w);
        let mut prev = ch[0];
        for (i, &out_c) in ch.iter().enumerate() {
            for r in 0..n_res {
                let cin = if r == 0 { prev } else { out_c };
                x = b.resnet(&format!("encoder.down_blocks.{i}.resnets.{r}"), cin, out_c, cur_h, cur_w, &x);
            }
            prev = out_c;
            if i < n_blocks - 1 {
                x = b.conv_down(&format!("encoder.down_blocks.{i}.downsamplers.0.conv"), out_c, cur_h, cur_w, &x);
                cur_h /= 2;
                cur_w /= 2;
            }
            b.tap(format!("down_block.{i}"), &x, out_c * cur_h * cur_w);
        }

        // Mid block: resnet, (optional) attention, resnet.
        let mid_c = *ch.last().unwrap();
        x = b.resnet("encoder.mid_block.resnets.0", mid_c, mid_c, cur_h, cur_w, &x);
        if cfg.mid_block_add_attention {
            x = b.attn("encoder.mid_block.attentions.0", mid_c, cur_h, cur_w, &x);
        }
        x = b.resnet("encoder.mid_block.resnets.1", mid_c, mid_c, cur_h, cur_w, &x);
        b.tap("mid".into(), &x, mid_c * cur_h * cur_w);

        // Head: GroupNorm → SiLU → conv_out → moments (2·latent channels).
        let hn = b.gn("encoder.conv_norm_out", mid_c, cur_h, cur_w, &x);
        let hs = b.silu(mid_c * cur_h * cur_w, &hn);
        let moments = 2 * cfg.latent_channels;
        let out = b.conv("encoder.conv_out", mid_c, moments, 3, 1, cur_h, cur_w, &hs);
        // FLUX.2 quant_conv: 1×1 moments→moments after conv_out. Top-level
        // (`quant_conv.weight`, diffusers export) or nested (`encoder.quant_conv.*`,
        // BFL-native); a checkpoint with neither fails loudly in `Builder::get`.
        let out = if cfg.use_quant_conv {
            let p = if tensors.contains_key("quant_conv.weight") {
                "quant_conv"
            } else {
                "encoder.quant_conv"
            };
            b.conv(p, moments, moments, 1, 0, cur_h, cur_w, &out)
        } else {
            out
        };
        let out_len = (moments * cur_h * cur_w) as usize;

        // `img_in` is allocated outside the builder - see the decoder's note.
        let device_bytes = b.allocated_bytes() + (cfg.in_channels * h * w) as u64 * 4;
        let (steps, taps) = b.finish();
        VaeEncoder { gpu, cfg, steps, img_in, out, out_len, taps, device_bytes }
    }

    /// Encode an image `[in_channels·H·W]` (row-major NCHW, batch 1) into the
    /// moments `[2·latent·h·w]` — the first `latent` channels are the posterior
    /// mean, the next `latent` the log-variance. Raw: no scaling/shift applied.
    pub fn encode(&self, image: &[f32]) -> Vec<f32> {
        let bits: Vec<u32> = image.iter().map(|v| v.to_bits()).collect();
        self.gpu.write(&self.img_in, &bits);
        self.gpu.submit(&[], &self.steps);
        self.gpu.read(&self.out, self.out_len)
    }

    /// Encode and return only the posterior **mean** `[latent·h·w]` (the deterministic
    /// latent used for image-conditioned generation).
    pub fn encode_mean(&self, image: &[f32], lh: u32, lw: u32) -> Vec<f32> {
        let m = self.encode(image);
        let plane = (lh * lw) as usize;
        m[..self.cfg.latent_channels as usize * plane].to_vec()
    }

    /// Device bytes this graph holds - see [`VaeDecoder::device_bytes`].
    pub fn device_bytes(&self) -> u64 {
        self.device_bytes
    }

    pub fn read_tap(&self, name: &str) -> Option<Vec<f32>> {
        self.taps.iter().find(|(n, _, _)| n == name).map(|(_, buf, len)| self.gpu.read(buf, *len))
    }
}

// ---- placement footprint ---------------------------------------------------
//
// What a VAE graph costs on the device, answerable BEFORE it is built - which
// is the only time a placement decision can use it.
//
// This exists because a flat guess is not good enough. A FLUX.2 decode at a
// real output size is dominated by its activations, not its weights, and an
// estimate that tracks only the checkpoint is short by multiples exactly when
// it matters: every denoise step completes and then the last stage asks the
// driver for memory the plan never reserved.
//
// The shape is derived; the two constants are CALIBRATED against
// [`VaeDecoder::device_bytes`]/[`VaeEncoder::device_bytes`] - the builder's own
// account of what it allocated - and gated in `tests/footprint.rs` at sizes
// spanning a thumbnail to a full frame, in both directions. A change to the
// block schedule that moves the real allocation fails that gate rather than
// silently making every placement decision wrong.

/// Live activation buffers per resolution level, the calibrated multiplier on
/// [`level_bytes_per_pixel`]. Separate per direction because the two graphs
/// are not mirror images: the decoder's up-blocks carry one more resnet each
/// (`layers_per_block + 1`) and its skip-free chain keeps a different number
/// of buffers alive than the encoder's down path.
///
/// The activation pool reuses same-length buffers, so a graph's resident set is
/// a small multiple of "one buffer at each level" rather than the sum of every
/// intermediate. That multiple is what these are, rounded UP: an estimate that
/// is high reserves a little too much, an estimate that is low is a driver
/// out-of-memory.
const DECODE_LIVE_BUFFERS_PER_LEVEL: u64 = 11;
const ENCODE_LIVE_BUFFERS_PER_LEVEL: u64 = 8;

/// The size-independent floor: the chunked-GEMM `col` scratch (capped, see
/// `blocks::COL_BUDGET_MIB`), the staging a large weight upload leaves
/// resident, and the driver's own per-context allocation. Flat by nature - it
/// is what stops the per-pixel term from being asked to explain a constant.
const FIXED_SCRATCH: u64 = 768 << 20;

/// Bytes of activation per OUTPUT pixel for one live buffer at each level of
/// `cfg`'s channel schedule.
///
/// Level `j` runs at `1/4^j` of the output pixels with `block_out_channels[j]`
/// channels (the encoder walks it high-res-first, the decoder low-res-first;
/// the sum is the same either way), so a buffer there costs
/// `channels * 4 / 4^j` bytes for every output pixel. Reading it off the config
/// is what makes this follow a different VAE instead of going stale beside one.
pub fn level_bytes_per_pixel(cfg: &VaeConfig) -> u64 {
    cfg.block_out_channels
        .iter()
        .enumerate()
        .map(|(j, &ch)| (ch as u64 * 4) / 4u64.pow(j as u32))
        .sum()
}

/// Device bytes of DECODER weights, summed from the same tensor schedule
/// `VaeDecoder::from_diffusers` uploads. Exact, not estimated.
pub fn decoder_weight_bytes(cfg: &VaeConfig) -> u64 {
    let (zc, rc) = (cfg.latent_channels as u64, cfg.reversed_channels());
    let mid_c = *cfg.block_out_channels.last().expect("block_out_channels") as u64;
    let mut n = 0u64;
    if cfg.use_post_quant_conv {
        n += conv_params(zc, zc, 1);
    }
    n += conv_params(zc, mid_c, 3);
    n += resnet_params(mid_c, mid_c);
    if cfg.mid_block_add_attention {
        n += attn_params(mid_c);
    }
    n += resnet_params(mid_c, mid_c);
    let mut prev = mid_c;
    for (i, &out_c) in rc.iter().enumerate() {
        let out_c = out_c as u64;
        for r in 0..=cfg.layers_per_block as u64 {
            n += resnet_params(if r == 0 { prev } else { out_c }, out_c);
        }
        if i + 1 < rc.len() {
            n += conv_params(out_c, out_c, 3);
        }
        prev = out_c;
    }
    n += 2 * prev; // conv_norm_out
    n += conv_params(prev, cfg.out_channels as u64, 3);
    n * 4
}

/// Device bytes of ENCODER weights - see [`decoder_weight_bytes`].
pub fn encoder_weight_bytes(cfg: &VaeConfig) -> u64 {
    let ch: Vec<u64> = cfg.block_out_channels.iter().map(|&c| c as u64).collect();
    let mut n = conv_params(cfg.in_channels as u64, ch[0], 3);
    let mut prev = ch[0];
    for (i, &out_c) in ch.iter().enumerate() {
        for r in 0..cfg.layers_per_block as u64 {
            n += resnet_params(if r == 0 { prev } else { out_c }, out_c);
        }
        prev = out_c;
        if i + 1 < ch.len() {
            n += conv_params(out_c, out_c, 3);
        }
    }
    let mid_c = *ch.last().expect("block_out_channels");
    n += resnet_params(mid_c, mid_c);
    if cfg.mid_block_add_attention {
        n += attn_params(mid_c);
    }
    n += resnet_params(mid_c, mid_c);
    n += 2 * mid_c; // conv_norm_out
    let moments = 2 * cfg.latent_channels as u64;
    n += conv_params(mid_c, moments, 3);
    if cfg.use_quant_conv {
        n += conv_params(moments, moments, 1);
    }
    n * 4
}

fn conv_params(cin: u64, cout: u64, k: u64) -> u64 {
    cout * cin * k * k + cout
}

fn resnet_params(cin: u64, cout: u64) -> u64 {
    let mut n = 2 * cin + conv_params(cin, cout, 3) + 2 * cout + conv_params(cout, cout, 3);
    if cin != cout {
        n += conv_params(cin, cout, 1);
    }
    n
}

/// diffusers stores each projection as a 1x1 conv, plus the group norm.
fn attn_params(c: u64) -> u64 {
    2 * c + 4 * conv_params(c, c, 1)
}

/// Device bytes a decode graph will hold, for a latent of `lh x lw` (i.e. an
/// output image of `8*lh x 8*lw`). What a placement decision reserves.
pub fn decoder_device_bytes(cfg: &VaeConfig, lh: u32, lw: u32) -> u64 {
    decoder_device_bytes_for_pixels(cfg, (lh as u64 * 8) * (lw as u64 * 8))
}

/// [`decoder_device_bytes`] for a caller that knows the OUTPUT pixel count but
/// not yet its shape - a pipeline sizing itself for "at most N image tokens"
/// before any request has said what aspect ratio those are.
pub fn decoder_device_bytes_for_pixels(cfg: &VaeConfig, px: u64) -> u64 {
    decoder_weight_bytes(cfg) + DECODE_LIVE_BUFFERS_PER_LEVEL * level_bytes_per_pixel(cfg) * px + FIXED_SCRATCH
}

/// Device bytes an encode graph will hold, for an image of `h x w`.
pub fn encoder_device_bytes(cfg: &VaeConfig, h: u32, w: u32) -> u64 {
    encoder_device_bytes_for_pixels(cfg, h as u64 * w as u64)
}

/// [`encoder_device_bytes`] by pixel count - see
/// [`decoder_device_bytes_for_pixels`].
pub fn encoder_device_bytes_for_pixels(cfg: &VaeConfig, px: u64) -> u64 {
    encoder_weight_bytes(cfg) + ENCODE_LIVE_BUFFERS_PER_LEVEL * level_bytes_per_pixel(cfg) * px + FIXED_SCRATCH
}

/// A device carrying the VAE kernel set, for a caller that wants ONE of them
/// to share across every graph it builds (see
/// [`VaeDecoder::from_diffusers_on`]). `device`: `Some("cpu")` | `Some("gpu")`
/// | `Some("gpu<i>")` | `None` for the ambient selection.
pub fn device(device: Option<&str>) -> Gpu {
    Gpu::open(device, &KERNELS)
}
