// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ECAPA-TDNN speaker encoder forward (Qwen3-TTS `speaker_encoder`).
//!
//! Mirrors `Qwen3TTSSpeakerEncoder.forward`: log-mel `[T,128]` is transposed to
//! channel-major NCL `[128,T]`, run through the initial TDNN block, three
//! SE-Res2Net blocks, multi-layer feature aggregation (MFA), attentive
//! statistics pooling (ASP) and a final 1×1 conv, yielding a 1024-dim embedding.
//!
//! Building blocks (no BatchNorm in this variant — `TimeDelayNetBlock` is
//! Conv1d + ReLU only):
//!   * every conv is `audio::conv::conv1d` over the shared WGSL engine, in NCL.
//!     PyTorch `padding="same", padding_mode="reflect"` is reproduced exactly by
//!     host reflect-padding the input by `dilation*(K-1)/2` before a pad-0 conv
//!     (k=1 convs need no padding). Reflect (not zero) matters for parity.
//!   * ReLU = `leaky_relu` slope 0; Tanh = `tanh_act`; per-channel SE gate =
//!     `scale_chan`. Per-channel conv bias is broadcast + `add2` (NCL).
//!   * the pooling reductions (uniform + attention-weighted mean/std and the
//!     **time-axis softmax**) are computed on the host — they are tiny vector
//!     reductions over the time axis and no time-softmax kernel exists; the two
//!     ASP projections and the final fc remain on-device convs.
//!
//! Execution is eager (each step submits immediately), matching the codec
//! decoder; the encoder runs once per voice-clone so a single fused graph buys
//! nothing.

use std::collections::HashMap;

use bytemuck::cast_slice;
use gpu_core::{f, BufUsage, DeviceBuffer, Gpu};
use paramstore::{ParamStore, Role};

use crate::config::SpeakerConfig;

// ---- kernel indices (order matches PIPELINES) ----
const CONV1D: usize = 0;
const LEAKY_RELU: usize = 1;
const TANH_ACT: usize = 2;
const SCALE_CHAN: usize = 3;
const ADD2: usize = 4;

pub const PIPELINES: &[(&str, &str)] = &[
    ("conv1d", kernels::CONV1D),
    ("leaky_relu", kernels::LEAKY_RELU),
    ("tanh_act", kernels::TANH_ACT),
    ("scale_chan", kernels::SCALE_CHAN),
    ("add2", kernels::ADD2),
];

/// ASP variance floor (`AttentiveStatisticsPooling.eps`).
const ASP_EPS: f32 = 1e-12;

/// Inference-only ECAPA speaker encoder: frozen weights on device + a full host
/// mirror (conv biases for NCL broadcasts, SE 1×1 weights, and the ASP host
/// reductions all read from it).
pub struct SpeakerEncoder {
    gpu: Gpu,
    cfg: SpeakerConfig,
    ps: ParamStore,
    host: HashMap<String, Vec<f32>>,
}

impl SpeakerEncoder {
    /// Load an inference-only encoder from a brain checkpoint produced by
    /// [`crate::import::import`].
    pub fn load_inference(weights_path: &str) -> SpeakerEncoder {
        Self::load_inference_on(Gpu::new(PIPELINES), weights_path)
    }

    /// Build on an existing device handle (see `gpu_core::Gpu::share`) so a
    /// process holds ONE device however many components it loads.
    pub fn load_inference_on(gpu: Gpu, weights_path: &str) -> SpeakerEncoder {
        let c = checkpoint::load(weights_path);
        let cfg = SpeakerConfig::from_json(&c.header["config"]);
        let init = c.by_role("");
        SpeakerEncoder::from_weights_on(gpu, cfg, init)
    }

    /// Build from an in-memory weight map (tests + [`load_inference`]).
    pub fn from_weights(cfg: SpeakerConfig, init: HashMap<String, Vec<f32>>) -> SpeakerEncoder {
        Self::from_weights_on(Gpu::new(PIPELINES), cfg, init)
    }

    pub(crate) fn from_weights_on(gpu: Gpu, cfg: SpeakerConfig, init: HashMap<String, Vec<f32>>) -> SpeakerEncoder {
        let host = init.clone();
        let roles: Vec<(String, usize, Role)> =
            init.iter().map(|(n, v)| (n.clone(), v.len(), Role::Frozen)).collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, &init);
        SpeakerEncoder { gpu, cfg, ps, host }
    }

    pub fn config(&self) -> &SpeakerConfig {
        &self.cfg
    }

    // -- eager helpers (each submits immediately) --
    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }
    fn st(&self, n: usize) -> DeviceBuffer {
        self.gpu.storage(n as u64)
    }
    fn run(&self, step: gpu_core::Step) {
        self.gpu.submit(&[], &[step]);
    }
    fn upload(&self, data: &[f32]) -> DeviceBuffer {
        let b = self.gpu.buffer("act", (data.len().max(1) * 4) as u64, BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC);
        self.gpu.write(&b, cast_slice(data));
        b
    }
    fn add2(&self, a: &DeviceBuffer, b: &DeviceBuffer, total: u32) -> DeviceBuffer {
        let out = self.st(total as usize);
        self.run(self.gpu.step(ADD2, &[a, b, &out], &[total], total));
        out
    }
    /// ReLU = leaky_relu with slope 0, in a fresh buffer.
    fn relu(&self, x: &DeviceBuffer, total: u32) -> DeviceBuffer {
        let out = self.st(total as usize);
        self.run(self.gpu.step(LEAKY_RELU, &[x, &out], &[total, f(0.0)], total));
        out
    }
    fn tanh(&self, x: &DeviceBuffer, total: u32) -> DeviceBuffer {
        let out = self.st(total as usize);
        self.run(self.gpu.step(TANH_ACT, &[x, &out], &[total], total));
        out
    }
    /// Per-channel gate over NCL `[C, L]`: `y = x * scale[c]`.
    fn scale_chan(&self, x: &DeviceBuffer, scale: &DeviceBuffer, c: u32, l: u32) -> DeviceBuffer {
        let total = c * l;
        let out = self.st(total as usize);
        self.run(self.gpu.step(SCALE_CHAN, &[x, scale, &out], &[total, c, l], total));
        out
    }
    /// Add a per-channel bias to NCL `[C, L]` (channel = idx / L): broadcast the
    /// host bias to `[C, L]` then `add2` (the engine `bias_add` indexes the inner
    /// axis, which is the time axis here).
    fn add_ncl_bias(&self, x: &DeviceBuffer, bias_name: &str, c: u32, l: u32) -> DeviceBuffer {
        let bias = &self.host[bias_name];
        let mut bcast = vec![0.0f32; (c * l) as usize];
        for ch in 0..c as usize {
            let v = bias[ch];
            for t in 0..l as usize {
                bcast[ch * l as usize + t] = v;
            }
        }
        let bbuf = self.upload(&bcast);
        self.add2(x, &bbuf, c * l)
    }

    /// `Conv1d` with PyTorch `padding="same", padding_mode="reflect"` in NCL,
    /// stride 1, groups 1, then per-channel bias. `prefix` -> `{prefix}.weight`,
    /// `{prefix}.bias`. Returns `[cout, l]`.
    fn same_conv(&self, x: &DeviceBuffer, prefix: &str, cin: u32, cout: u32, l: u32, k: u32, dilation: u32) -> DeviceBuffer {
        let kref = audio::conv::ConvKernels { fwd: CONV1D, dx: 0, dw: 0 };
        let wname = format!("{prefix}.weight");
        let out = self.st((cout * l) as usize);
        let p = dilation * (k - 1) / 2;
        if p == 0 {
            let c = audio::conv::Conv1d { n: 1, cin, l, cout, k, stride: 1, pad: 0, dilation, groups: 1, lo: l };
            self.run(audio::conv::conv1d_fwd(&self.gpu, &kref, &c, x, self.w(&wname), &out));
        } else {
            let hv = self.gpu.read(x, (cin * l) as usize);
            let padded = reflect_pad(&hv, cin as usize, l as usize, p as usize);
            let xin = self.upload(&padded);
            let lin = l + 2 * p;
            let c = audio::conv::Conv1d { n: 1, cin, l: lin, cout, k, stride: 1, pad: 0, dilation, groups: 1, lo: l };
            self.run(audio::conv::conv1d_fwd(&self.gpu, &kref, &c, &xin, self.w(&wname), &out));
        }
        self.add_ncl_bias(&out, &format!("{prefix}.bias"), cout, l)
    }

    /// `TimeDelayNetBlock` = ReLU(same-conv + bias).
    fn tdnn(&self, x: &DeviceBuffer, prefix: &str, cin: u32, cout: u32, l: u32, k: u32, dilation: u32) -> DeviceBuffer {
        let y = self.same_conv(x, prefix, cin, cout, l, k, dilation);
        self.relu(&y, cout * l)
    }

    /// `Res2NetBlock` (scale 8): split `[512,L]` into 8 channel chunks of 64,
    /// `out[0]=chunk0`, `out[1]=tdnn(chunk1)`, `out[i]=tdnn(chunk_i + out_{i-1})`.
    fn res2net(&self, h: &DeviceBuffer, prefix: &str, l: u32, dilation: u32) -> DeviceBuffer {
        let scale = self.cfg.enc_res2net_scale as usize; // 8
        let l = l as usize;
        let sub = 512 / scale; // 64 channels per chunk
        let hv = self.gpu.read(h, 512 * l);
        let chunk = |i: usize| hv[i * sub * l..(i + 1) * sub * l].to_vec();

        let mut outs: Vec<Vec<f32>> = Vec::with_capacity(scale);
        outs.push(chunk(0));
        for i in 1..scale {
            let inp = if i == 1 {
                chunk(i)
            } else {
                let c = chunk(i);
                let prev = &outs[i - 1];
                c.iter().zip(prev).map(|(a, b)| a + b).collect()
            };
            let buf = self.upload(&inp);
            let y = self.tdnn(&buf, &format!("{prefix}.res2net_block.blocks.{}.conv", i - 1), sub as u32, sub as u32, l as u32, 3, dilation);
            outs.push(self.gpu.read(&y, sub * l));
        }
        // concat along channels -> [512, L]
        let mut cat = vec![0.0f32; 512 * l];
        for (i, o) in outs.iter().enumerate() {
            cat[i * sub * l..(i + 1) * sub * l].copy_from_slice(o);
        }
        self.upload(&cat)
    }

    /// `SqueezeExcitationBlock`: gate `x` by a sigmoid of two 1×1 convs over the
    /// time-mean. conv1: 512->se, ReLU; conv2: se->512, Sigmoid. Host-computed
    /// (tiny `[se]`/`[512]` vectors), applied on-device via `scale_chan`.
    fn se_block(&self, x: &DeviceBuffer, prefix: &str, l: u32) -> DeviceBuffer {
        let c = 512usize;
        let se = self.cfg.enc_se_channels as usize; // 128
        let l = l as usize;
        let xv = self.gpu.read(x, c * l);
        // time-mean per channel
        let mut mean = vec![0.0f32; c];
        for ch in 0..c {
            let mut s = 0.0f32;
            for t in 0..l {
                s += xv[ch * l + t];
            }
            mean[ch] = s / l as f32;
        }
        // conv1 (se, 512, 1) + bias, ReLU
        let w1 = &self.host[&format!("{prefix}.se_block.conv1.weight")];
        let b1 = &self.host[&format!("{prefix}.se_block.conv1.bias")];
        let mut m1 = vec![0.0f32; se];
        for s in 0..se {
            let mut acc = b1[s];
            for ch in 0..c {
                acc += w1[s * c + ch] * mean[ch];
            }
            m1[s] = acc.max(0.0);
        }
        // conv2 (512, se, 1) + bias, Sigmoid
        let w2 = &self.host[&format!("{prefix}.se_block.conv2.weight")];
        let b2 = &self.host[&format!("{prefix}.se_block.conv2.bias")];
        let mut gate = vec![0.0f32; c];
        for ch in 0..c {
            let mut acc = b2[ch];
            for s in 0..se {
                acc += w2[ch * se + s] * m1[s];
            }
            gate[ch] = 1.0 / (1.0 + (-acc).exp());
        }
        let gbuf = self.upload(&gate);
        self.scale_chan(x, &gbuf, c as u32, l as u32)
    }

    /// One `SqueezeExcitationRes2NetBlock`: tdnn1 -> res2net -> tdnn2 -> se, plus
    /// the input residual.
    fn se_res2net(&self, x: &DeviceBuffer, prefix: &str, l: u32, dilation: u32) -> DeviceBuffer {
        let h = self.tdnn(x, &format!("{prefix}.tdnn1.conv"), 512, 512, l, 1, 1);
        let h = self.res2net(&h, prefix, l, dilation);
        let h = self.tdnn(&h, &format!("{prefix}.tdnn2.conv"), 512, 512, l, 1, 1);
        let h = self.se_block(&h, prefix, l);
        self.add2(x, &h, 512 * l)
    }

    /// `AttentiveStatisticsPooling` over `[C=1536, L]` -> `[3072]`.
    fn asp(&self, x: &DeviceBuffer, l: u32) -> Vec<f32> {
        let cch = (self.cfg.enc_channels[4]) as usize; // 1536
        let l = l as usize;
        let xv = self.gpu.read(x, cch * l);

        // uniform (mask/total = 1/L) mean & std over time, per channel
        let (umean, ustd) = stats(&xv, cch, l, &vec![1.0 / l as f32; l], true);

        // attn_in = concat([x, mean.broadcast, std.broadcast]) -> [3*1536, L]
        let mut attn_in = vec![0.0f32; 3 * cch * l];
        attn_in[..cch * l].copy_from_slice(&xv);
        for ch in 0..cch {
            for t in 0..l {
                attn_in[(cch + ch) * l + t] = umean[ch];
                attn_in[(2 * cch + ch) * l + t] = ustd[ch];
            }
        }
        let attn_buf = self.upload(&attn_in);

        // tdnn (4608->128, k1) + ReLU ; tanh ; conv (128->1536, k1)
        let att_ch = self.cfg.enc_attention_channels; // 128
        let h = self.tdnn(&attn_buf, "asp.tdnn.conv", (3 * cch) as u32, att_ch, l as u32, 1, 1);
        let h = self.tanh(&h, att_ch * l as u32);
        let attn = self.same_conv(&h, "asp.conv", att_ch, cch as u32, l as u32, 1, 1);
        let av = self.gpu.read(&attn, cch * l);

        // time-axis softmax per channel
        let mut weights = vec![0.0f32; cch * l];
        for ch in 0..cch {
            let row = &av[ch * l..(ch + 1) * l];
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for (t, &v) in row.iter().enumerate() {
                let e = (v - mx).exp();
                weights[ch * l + t] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for t in 0..l {
                weights[ch * l + t] *= inv;
            }
        }

        // attention-weighted mean & std, concat -> [3072]
        let mut pooled = vec![0.0f32; 2 * cch];
        for ch in 0..cch {
            let w = &weights[ch * l..(ch + 1) * l];
            let mut mean = 0.0f32;
            for t in 0..l {
                mean += w[t] * xv[ch * l + t];
            }
            let mut var = 0.0f32;
            for t in 0..l {
                let d = xv[ch * l + t] - mean;
                var += w[t] * d * d;
            }
            pooled[ch] = mean;
            pooled[cch + ch] = var.max(ASP_EPS).sqrt();
        }
        pooled
    }

    /// Encode a log-mel `[T, 128]` (row-major, time-major) into the `enc_dim`
    /// speaker embedding.
    pub fn embed(&self, mel: &[f32]) -> Vec<f32> {
        let md = self.cfg.mel_dim as usize; // 128
        assert_eq!(mel.len() % md, 0, "mel length not a multiple of {md}");
        let t = (mel.len() / md) as u32;
        assert!(t > 0, "empty mel");

        // transpose [T,128] -> [128,T] NCL
        let mut nclv = vec![0.0f32; (t as usize) * md];
        for ti in 0..t as usize {
            for ch in 0..md {
                nclv[ch * t as usize + ti] = mel[ti * md + ch];
            }
        }
        let x = self.upload(&nclv);

        // initial TDNN block (k5)
        let c0 = self.cfg.enc_channels[0]; // 512
        let mut h = self.tdnn(&x, "blocks.0.conv", md as u32, c0, t, self.cfg.enc_kernel_sizes[0], self.cfg.enc_dilations[0]);

        // 3 SE-Res2Net blocks -> collect their outputs for MFA
        let mut feats: Vec<DeviceBuffer> = Vec::new();
        for i in 1..4usize {
            h = self.se_res2net(&h, &format!("blocks.{i}"), t, self.cfg.enc_dilations[i]);
            feats.push(clone_buf(self, &h, c0 * t));
        }

        // MFA: concat the 3 SE-Res2Net outputs -> [1536, L], conv1x1 + ReLU
        let cch = self.cfg.enc_channels[4] as usize; // 1536
        let mut cat = vec![0.0f32; cch * t as usize];
        for (i, fb) in feats.iter().enumerate() {
            let v = self.gpu.read(fb, (c0 * t) as usize);
            cat[i * (c0 * t) as usize..(i + 1) * (c0 * t) as usize].copy_from_slice(&v);
        }
        let catb = self.upload(&cat);
        let mfa = self.tdnn(&catb, "mfa.conv", cch as u32, cch as u32, t, 1, 1);

        // ASP -> [3072]
        let pooled = self.asp(&mfa, t);

        // fc (3072 -> enc_dim, k1) on [3072, 1]
        let pbuf = self.upload(&pooled);
        let out = self.same_conv(&pbuf, "fc", (2 * cch) as u32, self.cfg.enc_dim, 1, 1, 1);
        self.gpu.read(&out, self.cfg.enc_dim as usize)
    }

    /// Front-end + encoder: resample `samples` to 24 kHz, compute the reference
    /// log-mel, then [`embed`](Self::embed).
    pub fn embed_wav(&self, samples: &[f32], sr: u32) -> Vec<f32> {
        let wav = audio::resample_linear(samples, sr, 24000);
        let (mel, _t) = crate::mel::log_mel(&wav);
        self.embed(&mel)
    }
}

fn clone_buf(c: &SpeakerEncoder, x: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let v = c.gpu.read(x, n as usize);
    c.upload(&v)
}

/// Reflect-pad an NCL `[c, l]` host buffer by `p` on each side of the time axis
/// (PyTorch `mode="reflect"`, no border repeat). Requires `p < l`.
fn reflect_pad(x: &[f32], c: usize, l: usize, p: usize) -> Vec<f32> {
    let lp = l + 2 * p;
    let mut o = vec![0.0f32; c * lp];
    for ch in 0..c {
        for j in 0..lp {
            let mut idx = j as isize - p as isize;
            if idx < 0 {
                idx = -idx;
            }
            if idx as usize >= l {
                idx = 2 * (l as isize - 1) - idx;
            }
            o[ch * lp + j] = x[ch * l + idx as usize];
        }
    }
    o
}

/// Per-channel weighted mean & std over time for NCL `[c, l]` with time weights
/// `w[t]` (same for every channel). `clamp` floors the variance at `ASP_EPS`.
fn stats(x: &[f32], c: usize, l: usize, w: &[f32], clamp: bool) -> (Vec<f32>, Vec<f32>) {
    let mut mean = vec![0.0f32; c];
    let mut std = vec![0.0f32; c];
    for ch in 0..c {
        let mut m = 0.0f32;
        for t in 0..l {
            m += w[t] * x[ch * l + t];
        }
        let mut v = 0.0f32;
        for t in 0..l {
            let d = x[ch * l + t] - m;
            v += w[t] * d * d;
        }
        mean[ch] = m;
        std[ch] = if clamp { v.max(ASP_EPS).sqrt() } else { v.sqrt() };
    }
    (mean, std)
}
