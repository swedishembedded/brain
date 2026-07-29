// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Nemotron FastConformer encoder (offline / non-streaming). This file lands the
//! stages incrementally, parity-gated against dumped HF activations:
//!   1. depthwise-separable causal subsampling (×8) + linear   ← this pass
//!   2. macaron Conformer blocks (rel-pos MHA + conv module)    (next)
//!   3. prompt + encoder projectors
//!
//! The causal Conv2d used by NeMo pads `(kernel-1, stride-1)` on BOTH the time and
//! frequency axes (asymmetric), which brain's symmetric-pad conv2d kernel can't
//! express directly — so the padding is done host-side and the conv runs with
//! `pad=0`. Padding is glue (no weights); the conv/linear math runs on device.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu, Step};

use crate::config::NemotronConfig;

/// Kernels the encoder dispatches.
pub fn encoder_pipelines() -> &'static [(&'static str, &'static str)] {
    &[
        ("conv2d", kernels::CONV2D),                 // 0
        ("conv2d_gd", kernels::CONV2D_GD),           // 1
        ("add_chan_bcast", kernels::ADD_CHAN_BCAST), // 2
        ("relu_inplace", kernels::RELU_INPLACE),     // 3
        ("matmul", kernels::MATMUL),                 // 4
        ("bias_add", kernels::BIAS_ADD),             // 5
    ]
}
const K_CONV2D: usize = 0;
const K_CONV2D_GD: usize = 1;
const K_ADD_CHAN: usize = 2;
const K_RELU: usize = 3;
const K_MATMUL: usize = 4;
const K_BIAS_ADD: usize = 5;

/// Pad an NCHW buffer with `(top, bottom, left, right)` zeros. Host-side glue.
fn pad_nchw(x: &[f32], n: u32, c: u32, h: u32, w: u32, top: u32, bot: u32, left: u32, right: u32) -> (Vec<f32>, u32, u32) {
    let (hp, wp) = (h + top + bot, w + left + right);
    let mut out = vec![0.0f32; (n * c * hp * wp) as usize];
    for nn in 0..n {
        for cc in 0..c {
            for hh in 0..h {
                let src = ((nn * c + cc) * h + hh) * w;
                let dst = ((nn * c + cc) * hp + (hh + top)) * wp + left;
                out[dst as usize..(dst + w) as usize].copy_from_slice(&x[src as usize..(src + w) as usize]);
            }
        }
    }
    (out, hp, wp)
}

pub struct Encoder<'g> {
    g: &'g Gpu,
    cfg: NemotronConfig,
    w: HashMap<String, DeviceBuffer>,
    raw: HashMap<String, Vec<f32>>,
}

impl<'g> Encoder<'g> {
    pub fn new(g: &'g Gpu, cfg: NemotronConfig, weights: &HashMap<String, Vec<f32>>) -> Encoder<'g> {
        let w = weights.iter().map(|(k, v)| (k.clone(), g.storage_init(k, v))).collect();
        Encoder { g, cfg, w, raw: weights.clone() }
    }

    fn wb(&self, name: &str) -> &DeviceBuffer {
        self.w.get(name).unwrap_or_else(|| panic!("nemotron weight missing: {name}"))
    }

    /// A single strided causal Conv2d (dense or depthwise) with `(k-1,s-1)` causal
    /// pad on both axes, per-channel bias, applied to NCHW `x`. Returns `(y, Ho, Wo)`.
    #[allow(clippy::too_many_arguments)]
    fn causal_conv(&self, x: &[f32], cin: u32, h: u32, w: u32, cout: u32, wname: &str, bname: &str, groups: u32) -> (Vec<f32>, u32, u32) {
        let (k, s) = (self.cfg.subsampling_kernel, self.cfg.subsampling_stride);
        let (pad, hp, wp) = pad_nchw(x, 1, cin, h, w, k - 1, s - 1, k - 1, s - 1);
        let ho = (hp - k) / s + 1;
        let wo = (wp - k) / s + 1;
        let xin = self.g.storage_init("nem.conv.x", &pad);
        let conv = self.g.storage((cout * ho * wo) as u64);
        let out = self.g.storage((cout * ho * wo) as u64);
        let mut steps = Vec::new();
        if groups == 1 {
            steps.push(self.g.step(K_CONV2D, &[&xin, self.wb(wname), &conv], &[1, cin, hp, wp, cout, k, s, 0, ho, wo], cout * ho * wo));
        } else {
            steps.push(self.g.step(K_CONV2D_GD, &[&xin, self.wb(wname), &conv], &[1, cin, hp, wp, cout, k, s, 0, 1, groups, ho, wo], cout * ho * wo));
        }
        steps.push(self.g.step(K_ADD_CHAN, &[&conv, self.wb(bname), &out], &[1, cout, ho * wo], cout * ho * wo));
        self.g.submit(&[], &steps);
        (self.g.read(&out, (cout * ho * wo) as usize), ho, wo)
    }

    /// 1×1 pointwise Conv2d (dense, stride 1) + bias.
    fn pointwise(&self, x: &[f32], cin: u32, h: u32, w: u32, cout: u32, wname: &str, bname: &str) -> Vec<f32> {
        let xin = self.g.storage_init("nem.pw.x", x);
        let conv = self.g.storage((cout * h * w) as u64);
        let out = self.g.storage((cout * h * w) as u64);
        let steps = vec![
            self.g.step(K_CONV2D, &[&xin, self.wb(wname), &conv], &[1, cin, h, w, cout, 1, 1, 0, h, w], cout * h * w),
            self.g.step(K_ADD_CHAN, &[&conv, self.wb(bname), &out], &[1, cout, h * w], cout * h * w),
        ];
        self.g.submit(&[], &steps);
        self.g.read(&out, (cout * h * w) as usize)
    }

    fn relu(&self, x: &mut [f32]) {
        let b = self.g.storage_init("nem.relu", x);
        self.g.submit(&[], &[self.g.step(K_RELU, &[&b], &[x.len() as u32], x.len() as u32)]);
        x.copy_from_slice(&self.g.read(&b, x.len()));
    }

    /// Subsampled valid length after one stride-2 causal stage.
    fn stage_len(&self, len: u32) -> u32 {
        let (k, s) = (self.cfg.subsampling_kernel, self.cfg.subsampling_stride);
        (len + (k - 1) + (s - 1) - k) / s + 1
    }

    /// Zero time frames `>= valid` in an NCHW `[1, C, T, F]` buffer (matches
    /// NeMo `_mask_subsampled_frames`, stopping masked padding leaking into the
    /// next conv / the linear bias).
    fn mask_time(x: &mut [f32], c: u32, t: u32, f: u32, valid: u32) {
        for cc in 0..c as usize {
            for tt in valid as usize..t as usize {
                let base = (cc * t as usize + tt) * f as usize;
                for v in &mut x[base..base + f as usize] {
                    *v = 0.0;
                }
            }
        }
    }

    /// Depthwise-separable causal subsampling (×8) + linear projection.
    /// Input mel `[T, num_mel]` (row-major), `valid` real mel frames; output `[T', hidden]`.
    pub fn subsampling(&self, mel: &[f32], t: u32, valid: u32) -> (Vec<f32>, u32) {
        let cfg = &self.cfg;
        let ch = cfg.subsampling_channels;
        // [T, mel] -> NCHW [1, 1, T, mel]
        let (mut cur, mut h, mut w, mut cin) = (mel.to_vec(), t, cfg.num_mel_bins, 1u32);
        let mut vlen = valid;

        // stem: conv_in(1->ch), +bias, mask, relu
        let (y, ho, wo) = self.causal_conv(&cur, cin, h, w, ch, "encoder.subsampling.conv_in.weight", "encoder.subsampling.conv_in.bias", 1);
        cur = y;
        (h, w, cin) = (ho, wo, ch);
        vlen = self.stage_len(vlen);
        Self::mask_time(&mut cur, ch, h, w, vlen);
        self.relu(&mut cur);

        // depthwise-separable stages
        for i in 0..cfg.subsampling_stages() - 1 {
            let (y, ho, wo) = self.causal_conv(
                &cur, cin, h, w, ch,
                &format!("encoder.subsampling.layers.{i}.depthwise_conv.weight"),
                &format!("encoder.subsampling.layers.{i}.depthwise_conv.bias"),
                ch,
            );
            let mut pw = self.pointwise(&y, ch, ho, wo, ch, &format!("encoder.subsampling.layers.{i}.pointwise_conv.weight"), &format!("encoder.subsampling.layers.{i}.pointwise_conv.bias"));
            (h, w) = (ho, wo);
            vlen = self.stage_len(vlen);
            Self::mask_time(&mut pw, ch, h, w, vlen);
            cur = pw;
            self.relu(&mut cur);
        }

        // reshape [1, ch, T', F'] -> [T', ch*F'] then linear -> [T', hidden]
        let (tt, ff) = (h, w);
        let flat = ch * ff;
        let mut perm = vec![0.0f32; (tt * flat) as usize];
        for c in 0..ch as usize {
            for tpos in 0..tt as usize {
                for f in 0..ff as usize {
                    perm[tpos * flat as usize + c * ff as usize + f] = cur[(c * tt as usize + tpos) * ff as usize + f];
                }
            }
        }
        let pin = self.g.storage_init("nem.sub.perm", &perm);
        let lin = self.g.storage((tt * cfg.hidden) as u64);
        let steps = vec![
            self.g.step(K_MATMUL, &[&pin, self.wb("encoder.subsampling.linear.weight"), &lin], &[tt, flat, cfg.hidden], tt * cfg.hidden),
            self.g.step(K_BIAS_ADD, &[&lin, self.wb("encoder.subsampling.linear.bias")], &[tt, cfg.hidden], tt * cfg.hidden),
        ];
        self.g.submit(&[], &steps);
        let _ = &self.raw;
        (self.g.read(&lin, (tt * cfg.hidden) as usize), tt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::path::Path;

    const GOLD: &str = "/data/workspace/resources/asr/golden/nemotron";
    const CKPT: &str = "/data/workspace/resources/asr/nemotron/hf";

    fn read_f32(p: &str) -> Vec<f32> {
        let mut f = std::fs::File::open(p).unwrap_or_else(|_| panic!("missing {p}"));
        let mut b = Vec::new();
        f.read_to_end(&mut b).unwrap();
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    #[test]
    fn subsampling_matches_reference() {
        if !Path::new(&format!("{GOLD}/subsampling.f32")).exists() || !Path::new(&format!("{CKPT}/model.safetensors")).exists() {
            eprintln!("skipping: goldens/checkpoint absent");
            return;
        }
        let cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        let mel = read_f32(&format!("{GOLD}/input_features.f32")); // [T, 128]
        let nmel = cfg.num_mel_bins as usize;
        let t = (mel.len() / nmel) as u32;
        // valid frames = frames not zeroed by the frontend mask (masked frames are exactly 0)
        let valid = (0..t as usize).filter(|&i| mel[i * nmel..(i + 1) * nmel].iter().any(|&v| v != 0.0)).count() as u32;
        let refsub = read_f32(&format!("{GOLD}/subsampling.f32")); // [T', 1024]

        let weights = crate::import::load_tensors(Path::new(CKPT)).expect("load");
        let g = Gpu::new_cpu(encoder_pipelines());
        let enc = Encoder::new(&g, cfg, &weights);
        let (sub, tt) = enc.subsampling(&mel, t, valid);
        eprintln!("subsampling out [{tt}, {}] vs golden {}", sub.len() / tt as usize, refsub.len());
        assert_eq!(sub.len(), refsub.len(), "shape mismatch");
        let d = sub.iter().zip(&refsub).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        eprintln!("subsampling maxdiff {d}");
        assert!(d < 2e-3, "subsampling maxdiff {d}");
    }
}
