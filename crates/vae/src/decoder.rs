// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `AutoencoderKL` decoder as a pre-recorded brain kernel graph.
//!
//! Mirrors the diffusers `Decoder`: [1×1 `post_quant_conv` when the config
//! enables it — FLUX.2] → `conv_in` → mid block (resnet, self-attn, resnet) →
//! up blocks (`layers_per_block+1` resnets each, nearest-2× upsample + conv on
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
}

impl VaeDecoder {
    /// Build the decode graph for an input latent `[latent_ch, h, w]` and upload
    /// all decoder weights. `device`: `Some("cpu")` | `Some("gpu")` | `None`.
    pub fn from_diffusers(cfg: VaeConfig, tensors: &Tensors, h: u32, w: u32, device: Option<&str>) -> VaeDecoder {
        let gpu = match device {
            Some("cpu") => Gpu::new_cpu(&KERNELS),
            Some("gpu") | Some("wgpu") => Gpu::new_wgpu(&KERNELS),
            _ => Gpu::new(&KERNELS),
        };
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

        let (steps, taps) = b.finish();
        VaeDecoder { gpu, cfg, steps, z_in, out, out_len, taps }
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
}

impl VaeEncoder {
    /// Build the encode graph for an input image `[in_channels, h, w]` (full-res,
    /// NOT latent size) and upload all encoder weights. `device`: `Some("cpu")` |
    /// `Some("gpu")` | `None`.
    pub fn from_diffusers(cfg: VaeConfig, tensors: &Tensors, h: u32, w: u32, device: Option<&str>) -> VaeEncoder {
        let gpu = match device {
            Some("cpu") => Gpu::new_cpu(&KERNELS),
            Some("gpu") | Some("wgpu") => Gpu::new_wgpu(&KERNELS),
            _ => Gpu::new(&KERNELS),
        };
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

        let (steps, taps) = b.finish();
        VaeEncoder { gpu, cfg, steps, img_in, out, out_len, taps }
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

    pub fn read_tap(&self, name: &str) -> Option<Vec<f32>> {
        self.taps.iter().find(|(n, _, _)| n == name).map(|(_, buf, len)| self.gpu.read(buf, *len))
    }
}
