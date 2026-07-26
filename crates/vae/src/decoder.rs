// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `AutoencoderKL` decoder as a pre-recorded brain kernel graph.
//!
//! Mirrors the diffusers `Decoder`: `conv_in` → mid block (resnet, self-attn,
//! resnet) → up blocks (`layers_per_block+1` resnets each, nearest-2× upsample
//! + conv on all but the last block) → `conv_norm_out` → SiLU → `conv_out`.
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

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use std::collections::HashMap;

use crate::config::VaeConfig;

// Kernel-table indices (order matches KERNELS).
const K_CONV: usize = 0;
const K_GN_STATS: usize = 1;
const K_GN_APPLY: usize = 2;
const K_SILU: usize = 3;
const K_ADD2: usize = 4;
const K_UPSAMPLE2: usize = 5;
const K_NCHW_NLC: usize = 6;
const K_NLC_NCHW: usize = 7;
const K_ATTN_SCORES: usize = 8;
const K_ATTN_SOFTMAX: usize = 9;
const K_ATTN_APPLY: usize = 10;

const KERNELS: [(&str, &str); 11] = [
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
    ("gn_stats", kernels::GN_STATS),
    ("gn_apply", kernels::GN_APPLY),
    ("silu", kernels::SILU),
    ("add2", kernels::ADD2),
    ("upsample2", kernels::UPSAMPLE2),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
];

/// Host tensors by name (diffusers key, e.g. `decoder.conv_in.weight`) →
/// `(shape, row-major f32 data)`.
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

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

/// Graph-construction state (borrows the device + host tensors).
struct Builder<'a> {
    gpu: &'a Gpu,
    t: &'a Tensors,
    eps: f32,
    groups: u32,
    steps: Vec<Step>,
    taps: Vec<(String, DeviceBuffer, usize)>,
}

impl<'a> Builder<'a> {
    fn get(&self, name: &str) -> &(Vec<usize>, Vec<f32>) {
        self.t.get(name).unwrap_or_else(|| panic!("vae: missing tensor {name}"))
    }
    fn dev(&self, name: &str) -> DeviceBuffer {
        self.gpu.storage_init(name, &self.get(name).1)
    }
    fn act(&self, len: u64) -> DeviceBuffer {
        self.gpu.storage(len)
    }
    fn tap(&mut self, name: String, buf: &DeviceBuffer, len: u32) {
        self.taps.push((name, buf.clone(), len as usize));
    }

    /// Conv (+bias) `prefix.{weight,bias}`: `x[cin,h,w] → y[cout,ho,wo]`.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        prefix: &str,
        cin: u32,
        cout: u32,
        k: u32,
        pad: u32,
        h: u32,
        w: u32,
        x: &DeviceBuffer,
    ) -> DeviceBuffer {
        let ho = (h + 2 * pad - k) + 1;
        let wo = (w + 2 * pad - k) + 1;
        let wgt = self.dev(&format!("{prefix}.weight"));
        let bias = self.dev(&format!("{prefix}.bias"));
        let y = self.act((cout * ho * wo) as u64);
        let threads = cout.div_ceil(8) * (ho * wo).div_ceil(4);
        let step = self.gpu.step(
            K_CONV,
            &[x, &wgt, &bias, &y],
            &[1, cin, h, w, cout, k, 1, pad, ho, wo],
            threads,
        );
        self.steps.push(step);
        y
    }

    /// Static affine GroupNorm from `prefix.{weight,bias}` (32 groups, eps
    /// 1e-6): `y = gamma·(x-μ)/σ + beta` per group. `gb = [gamma‖beta]`.
    fn gn(&mut self, prefix: &str, c: u32, h: u32, w: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let (_, gamma) = self.get(&format!("{prefix}.weight"));
        let (_, beta) = self.get(&format!("{prefix}.bias"));
        let mut gbv = gamma.clone();
        gbv.extend_from_slice(beta);
        let gb = self.gpu.storage_init(&format!("{prefix}.gb"), &gbv);
        let g = self.groups;
        let stats = self.act(2 * g as u64);
        let y = self.act((c * h * w) as u64);
        self.steps.push(self.gpu.step(
            K_GN_STATS,
            &[x, &stats],
            &[1, c, h, w, g, f(self.eps)],
            g,
        ));
        self.steps.push(self.gpu.step(
            K_GN_APPLY,
            &[x, &stats, &gb, &y],
            &[1, c, h, w, g],
            c * h * w,
        ));
        y
    }

    fn silu(&mut self, n: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let y = self.act(n as u64);
        self.steps.push(self.gpu.step(K_SILU, &[x, &y], &[n], n));
        y
    }

    fn add(&mut self, n: u32, a: &DeviceBuffer, b: &DeviceBuffer) -> DeviceBuffer {
        let y = self.act(n as u64);
        self.steps.push(self.gpu.step(K_ADD2, &[a, b, &y], &[n], n));
        y
    }

    /// Nearest-neighbour 2× upsample: `[c,h,w] → [c,2h,2w]`.
    fn upsample(&mut self, c: u32, h: u32, w: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let y = self.act((c * 2 * h * 2 * w) as u64);
        self.steps.push(self.gpu.step(K_UPSAMPLE2, &[x, &y], &[1, c, h, w], c * 4 * h * w));
        y
    }

    /// One diffusers `ResnetBlock2D` (no temb): `x → conv2(silu(norm2(conv1(
    /// silu(norm1(x)))))) + shortcut(x)`, shortcut = 1×1 conv when `cin≠cout`.
    fn resnet(
        &mut self,
        prefix: &str,
        cin: u32,
        cout: u32,
        h: u32,
        w: u32,
        x: &DeviceBuffer,
    ) -> DeviceBuffer {
        let r = if cin != cout {
            self.conv(&format!("{prefix}.conv_shortcut"), cin, cout, 1, 0, h, w, x)
        } else {
            x.clone()
        };
        let n1 = self.gn(&format!("{prefix}.norm1"), cin, h, w, x);
        let s1 = self.silu(cin * h * w, &n1);
        let c1 = self.conv(&format!("{prefix}.conv1"), cin, cout, 3, 1, h, w, &s1);
        let n2 = self.gn(&format!("{prefix}.norm2"), cout, h, w, &c1);
        let s2 = self.silu(cout * h * w, &n2);
        let c2 = self.conv(&format!("{prefix}.conv2"), cout, cout, 3, 1, h, w, &s2);
        let out = self.add(cout * h * w, &c2, &r);
        self.tap(prefix.to_string(), &out, cout * h * w);
        out
    }

    /// Mid-block single-head self-attention (diffusers `Attention`,
    /// `residual_connection=True`): `x + to_out(attn(to_qkv(group_norm(x))))`.
    /// `to_q/k/v` are fused into a single 1×1 qkv conv so the bidir attention
    /// trio applies unchanged; residual is added to the **pre-norm** input.
    fn attn(&mut self, prefix: &str, c: u32, h: u32, w: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let t = h * w;
        let normed = self.gn(&format!("{prefix}.group_norm"), c, h, w, x);

        // Fuse to_q/to_k/to_v (each [C,C] linear = [C,C,1,1] conv) into one
        // [3C,C,1,1] qkv conv weight + [3C] bias.
        let (_, qw) = self.get(&format!("{prefix}.to_q.weight"));
        let (_, kw) = self.get(&format!("{prefix}.to_k.weight"));
        let (_, vw) = self.get(&format!("{prefix}.to_v.weight"));
        let mut qkv_w = Vec::with_capacity(qw.len() * 3);
        qkv_w.extend_from_slice(qw);
        qkv_w.extend_from_slice(kw);
        qkv_w.extend_from_slice(vw);
        let (_, qb) = self.get(&format!("{prefix}.to_q.bias"));
        let (_, kb) = self.get(&format!("{prefix}.to_k.bias"));
        let (_, vb) = self.get(&format!("{prefix}.to_v.bias"));
        let mut qkv_b = Vec::with_capacity(qb.len() * 3);
        qkv_b.extend_from_slice(qb);
        qkv_b.extend_from_slice(kb);
        qkv_b.extend_from_slice(vb);
        let qkv_wd = self.gpu.storage_init(&format!("{prefix}.qkv.w"), &qkv_w);
        let qkv_bd = self.gpu.storage_init(&format!("{prefix}.qkv.b"), &qkv_b);

        // qkv 1×1 conv: [C,h,w] → [3C,h,w].
        let qkv_chw = self.act((3 * c * t) as u64);
        self.steps.push(self.gpu.step(
            K_CONV,
            &[&normed, &qkv_wd, &qkv_bd, &qkv_chw],
            &[1, c, h, w, 3 * c, 1, 1, 0, h, w],
            (3 * c).div_ceil(8) * t.div_ceil(4),
        ));
        // NCHW [3C,h,w] → NLC rows [T, 3C].
        let qkv = self.act((3 * c * t) as u64);
        self.steps.push(self.gpu.step(K_NCHW_NLC, &[&qkv_chw, &qkv], &[3 * c * t, 3 * c, t], 3 * c * t));

        // Single head, head_dim = C, scale 1/√C (applied in the kernel).
        let scores = self.act((t * t) as u64);
        self.steps.push(self.gpu.step(
            K_ATTN_SCORES,
            &[&qkv, &scores],
            &[1, 1, t, c, 3 * c, 0, c],
            t * t,
        ));
        let probs = self.act((t * t) as u64);
        self.steps.push(self.gpu.step(K_ATTN_SOFTMAX, &[&scores, &probs], &[1, 1, t], t));
        let attn_rows = self.act((t * c) as u64);
        self.steps.push(self.gpu.step(
            K_ATTN_APPLY,
            &[&probs, &qkv, &attn_rows],
            &[1, 1, t, c, 3 * c, 2 * c, c],
            t * c,
        ));
        // NLC rows [T, C] → NCHW [C,h,w].
        let attn_chw = self.act((c * t) as u64);
        self.steps.push(self.gpu.step(K_NLC_NCHW, &[&attn_rows, &attn_chw], &[c * t, c, t], c * t));

        let proj = self.conv(&format!("{prefix}.to_out.0"), c, c, 1, 0, h, w, &attn_chw);
        self.add(c * h * w, x, &proj)
    }
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
        let mut b = Builder {
            gpu: &gpu,
            t: tensors,
            eps: cfg.norm_eps,
            groups: cfg.norm_num_groups,
            steps: vec![],
            taps: vec![],
        };

        let z_in = gpu.storage((cfg.latent_channels * h * w) as u64);
        let rc = cfg.reversed_channels();
        let mid_c = *cfg.block_out_channels.last().unwrap();

        // conv_in: latent → highest block channel.
        let mut x = b.conv("decoder.conv_in", cfg.latent_channels, mid_c, 3, 1, h, w, &z_in);
        b.tap("conv_in".into(), &x, mid_c * h * w);

        // Mid block: resnet, (optional) attention, resnet.
        x = b.resnet("decoder.mid_block.resnets.0", mid_c, mid_c, h, w, &x);
        if cfg.mid_block_add_attention {
            x = b.attn("decoder.mid_block.attentions.0", mid_c, h, w, &x);
            b.tap("mid_attn".into(), &x, mid_c * h * w);
        }
        x = b.resnet("decoder.mid_block.resnets.1", mid_c, mid_c, h, w, &x);
        b.tap("mid".into(), &x, mid_c * h * w);

        // Up blocks: reversed channel schedule; upsample on all but the last.
        let n_blocks = rc.len();
        let n_res = cfg.layers_per_block + 1;
        let (mut cur_h, mut cur_w) = (h, w);
        let mut prev = mid_c;
        for (i, &out_c) in rc.iter().enumerate() {
            for r in 0..n_res {
                let cin = if r == 0 { prev } else { out_c };
                x = b.resnet(
                    &format!("decoder.up_blocks.{i}.resnets.{r}"),
                    cin,
                    out_c,
                    cur_h,
                    cur_w,
                    &x,
                );
            }
            if i < n_blocks - 1 {
                x = b.upsample(out_c, cur_h, cur_w, &x);
                cur_h *= 2;
                cur_w *= 2;
                x = b.conv(
                    &format!("decoder.up_blocks.{i}.upsamplers.0.conv"),
                    out_c,
                    out_c,
                    3,
                    1,
                    cur_h,
                    cur_w,
                    &x,
                );
            }
            prev = out_c;
            b.tap(format!("up_block.{i}"), &x, out_c * cur_h * cur_w);
        }

        // Head: GroupNorm → SiLU → conv_out.
        let hn = b.gn("decoder.conv_norm_out", prev, cur_h, cur_w, &x);
        let hs = b.silu(prev * cur_h * cur_w, &hn);
        let out = b.conv("decoder.conv_out", prev, cfg.out_channels, 3, 1, cur_h, cur_w, &hs);
        let out_len = (cfg.out_channels * cur_h * cur_w) as usize;

        let Builder { steps, taps, .. } = b;
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
}
