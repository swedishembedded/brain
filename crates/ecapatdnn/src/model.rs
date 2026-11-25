// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speaker-embedding pipelines that run whole
// speech corpora through a speaker encoder on GPUs for its clients. If your
// team needs expertise in batched speaker verification models then you can
// procure our services by sending an email to info@swedishembedded.com.

//! ECAPA-TDNN speaker encoder forward (Qwen3-TTS `speaker_encoder`).
//!
//! Mirrors `Qwen3TTSSpeakerEncoder.forward`: log-mel `[T,128]` is transposed to
//! channel-major NCL `[128,T]`, run through the initial TDNN block, three
//! SE-Res2Net blocks, multi-layer feature aggregation (MFA), attentive
//! statistics pooling (ASP) and a final 1×1 conv, yielding a 1024-dim embedding.
//!
//! Building blocks (no BatchNorm in this variant — `TimeDelayNetBlock` is
//! Conv1d + ReLU only): ReLU = `leaky_relu` slope 0, Tanh = `tanh_act`, the
//! squeeze-excitation gate = `scale_chan`; PyTorch `padding="same",
//! padding_mode="reflect"` is a reflect pad by `dilation*(K-1)/2` before the
//! conv (reflect, not zero, matters for parity).
//!
//! The forward is ONE recorded device pass over a batch of utterances
//! ([`SpeakerEncoder::embed_batch`]); only the `[n, enc_dim]` embeddings read
//! back. The batch is packed along time: every activation is `[C, n, S]`, row
//! `(c, b)` holding utterance `b` on `[P, P + T_b)` with `P` the widest reflect
//! pad, so each conv runs once over the flattened `C x (n*S)` sequence. Before
//! a "same" conv the margins are rewritten with each utterance's own
//! reflection (`seg_reflect`), so every valid output reads exactly what the
//! unbatched conv reads; the time pooling (the SE mean and the attentive
//! statistics) reads each utterance's own span only (`seg_pool_stats`). The
//! concatenations the reference makes (`[x, mean, std]` before the attention
//! TDNN, `[mean, std]` before the final fc) are split weights instead: the
//! per-utterance statistics enter those layers through their own column blocks
//! of the weight, as a per-(utterance, channel) term.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use audio::conv::{ConvGemmKernels, ConvKernels, ConvScratch};
use backend_cpu::par;
use bytemuck::cast_slice;
use gpu_core::{f, BufUsage, DeviceBuffer, Gpu, Step};
use paramstore::{ParamStore, Role};

use crate::config::SpeakerConfig;

// ---- kernel indices (order matches PIPELINES) ----
const CONV1D: usize = 0;
const LEAKY_RELU: usize = 1;
const TANH_ACT: usize = 2;
const SCALE_CHAN: usize = 3;
const ADD2: usize = 4;
const ADD_CHAN_INPLACE: usize = 5;
const IM2COL1D_AT: usize = 6;
const MATMUL_REG3: usize = 7;
const MATMUL_DX_REG: usize = 8;
const MATMUL_DW_REG_SPLITK: usize = 9;
const NLC_BIAS_NCHW: usize = 10;
const COL2IM1D_BIAS: usize = 11;
const MATMUL: usize = 12;
const BIAS_ADD: usize = 13;
const SIGMOID: usize = 14;
const NCHW_NLC: usize = 15;
const REGION_COPY: usize = 16;
const SEG_REFLECT: usize = 17;
const SEG_POOL_STATS: usize = 18;

pub const PIPELINES: &[(&str, &str)] = &[
    ("conv1d", kernels::CONV1D),
    ("leaky_relu", kernels::LEAKY_RELU),
    ("tanh_act", kernels::TANH_ACT),
    ("scale_chan", kernels::SCALE_CHAN),
    ("add2", kernels::ADD2),
    ("add_chan_inplace", kernels::ADD_CHAN_INPLACE),
    ("im2col1d_at", kernels::IM2COL1D_AT),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg_splitk", kernels::MATMUL_DW_REG_SPLITK),
    ("nlc_bias_nchw", kernels::NLC_BIAS_NCHW),
    ("col2im1d_bias", kernels::COL2IM1D_BIAS),
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("sigmoid", kernels::SIGMOID),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("region_copy", kernels::REGION_COPY),
    ("seg_reflect", kernels::SEG_REFLECT),
    ("seg_pool_stats", kernels::SEG_POOL_STATS),
];

/// The convs' kernel set: the GEMM lowering where the device selector picks
/// it, the direct `conv1d` otherwise, the bias fused either way.
const CONV: ConvGemmKernels = ConvGemmKernels {
    direct: ConvKernels { fwd: CONV1D, dx: 0, dw: 0 },
    bias: ADD_CHAN_INPLACE,
    im2col: IM2COL1D_AT,
    matmul: MATMUL_REG3,
    matmul_nn: MATMUL_DX_REG,
    matmul_tn: MATMUL_DW_REG_SPLITK,
    nlc_bias: NLC_BIAS_NCHW,
    col2im: COL2IM1D_BIAS,
};

/// ASP variance floor (`AttentiveStatisticsPooling.eps`).
const ASP_EPS: f32 = 1e-12;

/// Mel frames one pass packs at most (summed over its utterances, margins
/// included); a larger batch is split into several passes, utterances grouped
/// by length. The widest activations are the MFA's 1536 channels, a few of
/// them alive at once: about 0.8 GB each at 2^17 frames (23 minutes of audio).
pub const EMBED_PASS_FRAMES: usize = 1 << 17;

/// Words a sub-range binding's offset must be a multiple of
/// (`backend_api::STORAGE_BINDING_OFFSET_ALIGN_BYTES`).
const SLICE_ALIGN_WORDS: usize = (gpu_core::STORAGE_BINDING_OFFSET_ALIGN_BYTES / 4) as usize;

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Fresh activation bytes a pass lets pile up before it submits what it has
/// recorded, so a long batch's intermediates go back to the pool as it runs.
const FLUSH_BYTES: u64 = 1 << 30;

/// The derived weights the split concatenations read: the attention TDNN's
/// `[att, 3*C]` weight as its `x`, mean and std column blocks, the final fc's
/// `[enc_dim, 2*C]` as its mean and std blocks, and a zero bias for the conv
/// half of the attention TDNN (its real bias joins the per-utterance term).
const ASP_X: &str = "asp.tdnn.conv.weight:x";
const ASP_MEAN: &str = "asp.tdnn.conv.weight:mean";
const ASP_STD: &str = "asp.tdnn.conv.weight:std";
const ASP_ZERO_BIAS: &str = "asp.tdnn.conv.bias:zero";
const FC_MEAN: &str = "fc.weight:mean";
const FC_STD: &str = "fc.weight:std";

/// Inference-only ECAPA speaker encoder: frozen weights on the device.
pub struct SpeakerEncoder {
    gpu: Gpu,
    cfg: SpeakerConfig,
    ps: ParamStore,
}

impl SpeakerEncoder {
    /// Load an inference-only encoder ([`crate::import::open`]: a brain
    /// checkpoint or the HF checkpoint dir).
    pub fn load_inference(weights_path: &str) -> SpeakerEncoder {
        Self::load_inference_on(Gpu::new(PIPELINES), weights_path)
    }

    /// Build on an existing device handle (see `gpu_core::Gpu::share`) so a
    /// process holds ONE device however many components it loads.
    pub fn load_inference_on(gpu: Gpu, weights_path: &str) -> SpeakerEncoder {
        let c = crate::import::load(weights_path);
        let cfg = SpeakerConfig::from_json(&c.header["config"]);
        let init = c.by_role("");
        SpeakerEncoder::from_weights_on(gpu, cfg, init)
    }

    /// Build from an in-memory weight map (tests + [`load_inference`]).
    pub fn from_weights(cfg: SpeakerConfig, init: HashMap<String, Vec<f32>>) -> SpeakerEncoder {
        Self::from_weights_on(Gpu::new(PIPELINES), cfg, init)
    }

    pub(crate) fn from_weights_on(gpu: Gpu, cfg: SpeakerConfig, mut init: HashMap<String, Vec<f32>>) -> SpeakerEncoder {
        let cch = cfg.enc_channels[4] as usize;
        let blocks = |name: &str, rows: usize, parts: usize| -> Vec<Vec<f32>> {
            let w = &init[name];
            let width = parts * cch;
            assert_eq!(w.len(), rows * width, "{name}: expected [{rows}, {width}]");
            (0..parts).map(|p| w.chunks_exact(width).flat_map(|row| row[p * cch..(p + 1) * cch].iter().copied()).collect()).collect()
        };
        let asp = blocks("asp.tdnn.conv.weight", cfg.enc_attention_channels as usize, 3);
        let fc = blocks("fc.weight", cfg.enc_dim as usize, 2);
        for (name, w) in [ASP_X, ASP_MEAN, ASP_STD].into_iter().zip(asp).chain([FC_MEAN, FC_STD].into_iter().zip(fc)) {
            init.insert(name.to_string(), w);
        }
        init.insert(ASP_ZERO_BIAS.to_string(), vec![0.0; cfg.enc_attention_channels as usize]);
        let roles: Vec<(String, usize, Role)> = init.iter().map(|(n, v)| (n.clone(), v.len(), Role::Frozen)).collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, &init);
        SpeakerEncoder { gpu, cfg, ps }
    }

    pub fn config(&self) -> &SpeakerConfig {
        &self.cfg
    }

    /// Encode a log-mel `[T, 128]` (row-major, time-major) into the `enc_dim`
    /// speaker embedding.
    pub fn embed(&self, mel: &[f32]) -> Vec<f32> {
        self.embed_batch(&[mel]).pop().expect("one mel in, one embedding out")
    }

    /// [`Self::embed`] of every log-mel, in as few device passes as
    /// [`EMBED_PASS_FRAMES`] allows; the embeddings in input order. Every mel
    /// needs more frames than the widest reflect pad
    /// ([`SpeakerConfig::max_pad_frames`]).
    pub fn embed_batch(&self, mels: &[&[f32]]) -> Vec<Vec<f32>> {
        let md = self.cfg.mel_dim as usize;
        let pad = self.cfg.max_pad_frames();
        let frames: Vec<usize> = mels
            .iter()
            .map(|m| {
                assert!(m.len() % md == 0, "a mel of {} floats is not [T, {md}]", m.len());
                assert!(m.len() / md > pad, "a mel of {} frames is not longer than the widest reflect pad ({pad})", m.len() / md);
                m.len() / md
            })
            .collect();
        let mut order: Vec<usize> = (0..mels.len()).collect();
        order.sort_by_key(|&i| frames[i]);
        let mut out = vec![Vec::new(); mels.len()];
        let mut group: Vec<usize> = Vec::new();
        let flush = |group: &mut Vec<usize>, out: &mut Vec<Vec<f32>>| {
            if group.is_empty() {
                return;
            }
            let batch: Vec<&[f32]> = group.iter().map(|&i| mels[i]).collect();
            for (i, e) in group.drain(..).zip(Pass::new(self).embed(&batch)) {
                out[i] = e;
            }
        };
        for i in order {
            // Sorted ascending: this utterance sets the group's segment length.
            if !group.is_empty() && (group.len() + 1) * (frames[i] + 2 * pad) > EMBED_PASS_FRAMES {
                flush(&mut group, &mut out);
            }
            group.push(i);
        }
        flush(&mut group, &mut out);
        out
    }

    /// Front-end + encoder: resample `samples` to 24 kHz, compute the reference
    /// log-mel, then [`embed`](Self::embed). A clip too short for the encoder's
    /// reflect padding ([`SpeakerConfig::min_samples`]) is zero-padded up to it, so a
    /// sub-50 ms upload gives a (meaningless but finite) vector, not a panic.
    pub fn embed_wav(&self, samples: &[f32], sr: u32) -> Vec<f32> {
        self.embed_wav_batch(&[(samples, sr)]).pop().expect("one clip in, one embedding out")
    }

    /// [`Self::embed_wav`] of every `(samples, rate)` clip: the front end per
    /// clip on the host (in parallel), the encoder as one batch on the device.
    pub fn embed_wav_batch(&self, clips: &[(&[f32], u32)]) -> Vec<Vec<f32>> {
        let mels: Vec<Vec<f32>> = par::map(clips.len(), |i| self.log_mel(clips[i].0, clips[i].1));
        self.embed_batch(&mels.iter().map(Vec::as_slice).collect::<Vec<_>>())
    }

    /// The reference log-mel of a clip at `sr`, padded up to
    /// [`SpeakerConfig::min_samples`] at 24 kHz.
    fn log_mel(&self, samples: &[f32], sr: u32) -> Vec<f32> {
        let mut wav = audio::resample::rational(samples, 24000, sr);
        wav.resize(wav.len().max(self.cfg.min_samples()), 0.0);
        crate::mel::log_mel(&wav).0
    }
}

/// The device ops of one batch: every step is recorded and submitted in a few
/// batches ([`FLUSH_BYTES`]); only the embeddings read back.
struct Pass<'a> {
    enc: &'a SpeakerEncoder,
    steps: RefCell<Vec<Step>>,
    pending_bytes: Cell<u64>,
    /// Shared by every lowered conv of the pass (they run in submit order).
    scratch: RefCell<ConvScratch>,
}

/// A `[c, n, S]` segment-packed activation.
struct Seg {
    buf: DeviceBuffer,
    c: u32,
}

/// The batch's packing: `n` utterances, segments of `s` frames, utterance `b`
/// valid on `[off, off + lens[b])` of its segment.
struct Layout {
    n: u32,
    s: u32,
    off: u32,
    lens: DeviceBuffer,
}

impl Layout {
    /// Floats in a `[c, n, s]` activation.
    fn numel(&self, c: u32) -> u32 {
        c * self.n * self.s
    }
}

impl<'a> Pass<'a> {
    fn new(enc: &'a SpeakerEncoder) -> Pass<'a> {
        Pass { enc, steps: RefCell::new(Vec::new()), pending_bytes: Cell::new(0), scratch: RefCell::new(ConvScratch::new()) }
    }
    fn gpu(&self) -> &Gpu {
        &self.enc.gpu
    }
    fn w(&self, name: &str) -> &DeviceBuffer {
        self.enc.ps.w(name)
    }
    fn st(&self, n: u32) -> DeviceBuffer {
        self.pending_bytes.set(self.pending_bytes.get() + u64::from(n) * 4);
        self.gpu().storage(u64::from(n.max(1)))
    }
    fn run(&self, step: Step) {
        self.steps.borrow_mut().push(step);
        if self.pending_bytes.get() >= FLUSH_BYTES {
            self.flush();
        }
    }
    fn run_all(&self, steps: Vec<Step>) {
        steps.into_iter().for_each(|s| self.run(s));
    }
    fn flush(&self) {
        let steps = std::mem::take(&mut *self.steps.borrow_mut());
        if !steps.is_empty() {
            self.gpu().submit(&[], &steps);
        }
        self.pending_bytes.set(0);
    }
    fn upload(&self, data: &[f32]) -> DeviceBuffer {
        let b = self.gpu().buffer("act", (data.len().max(1) * 4) as u64, BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC);
        self.gpu().write(&b, cast_slice(data));
        b
    }

    /// The embeddings of `mels` (each `[T_b, mel_dim]`), in order.
    fn embed(&self, mels: &[&[f32]]) -> Vec<Vec<f32>> {
        let cfg = &self.enc.cfg;
        let md = cfg.mel_dim as usize;
        let off = cfg.max_pad_frames();
        let lens: Vec<u32> = mels.iter().map(|m| (m.len() / md) as u32).collect();
        let tmax = lens.iter().copied().max().expect("a non-empty batch") as usize;
        // Segments long enough for both margins, and rounded so that `n * S`
        // is a multiple of [`SLICE_ALIGN_WORDS`]: every channel range (the
        // Res2Net chunks, the MFA's concatenation) then starts on a bindable
        // offset. The extra right margin is never read as valid data.
        let n = mels.len();
        let step = SLICE_ALIGN_WORDS / gcd(n, SLICE_ALIGN_WORDS);
        let s = (tmax + 2 * off).div_ceil(step) * step;
        // [mel_dim, n, S]: utterance b's frame t of channel ch at (ch*n + b)*S + off + t.
        let mut packed = vec![0.0f32; md * n * s];
        for (b, mel) in mels.iter().enumerate() {
            for (t, frame) in mel.chunks_exact(md).enumerate() {
                for (ch, &v) in frame.iter().enumerate() {
                    packed[(ch * n + b) * s + off + t] = v;
                }
            }
        }
        let lens_buf = self.gpu().buffer("lens", (lens.len() * 4) as u64, BufUsage::STORAGE | BufUsage::COPY_DST);
        self.gpu().write(&lens_buf, &lens);
        let lay = Layout { n: n as u32, s: s as u32, off: off as u32, lens: lens_buf };
        let x = Seg { buf: self.upload(&packed), c: cfg.mel_dim };

        let c0 = cfg.enc_channels[0];
        let mut h = self.tdnn(&lay, (&x.buf, 0), x.c, "blocks.0.conv", c0, cfg.enc_kernel_sizes[0], cfg.enc_dilations[0]);
        // The three SE-Res2Net outputs, concatenated for the MFA as they are made.
        let cch = cfg.enc_channels[4];
        let cat = self.st(lay.numel(cch));
        for i in 1..4u32 {
            h = self.se_res2net(&lay, &h, &format!("blocks.{i}"), cfg.enc_dilations[i as usize]);
            self.copy(&h.buf, 0, &cat, (i - 1) * lay.numel(c0), lay.numel(c0));
        }
        let mfa = self.tdnn(&lay, (&cat, 0), cch, "mfa.conv", cch, 1, 1);
        let (mean, std) = self.asp(&lay, &mfa);
        // The fc over [mean, std]: its two column blocks.
        let dim = cfg.enc_dim;
        let out = self.linear(&mean, FC_MEAN, lay.n, cch, dim);
        let out_std = self.linear(&std, FC_STD, lay.n, cch, dim);
        let out = self.add2(&out, &out_std, lay.n * dim);
        self.bias_add(&out, "fc.bias", lay.n, dim);
        self.flush();
        let v = self.gpu().read(&out, (lay.n * dim) as usize);
        v.chunks_exact(dim as usize).map(<[f32]>::to_vec).collect()
    }

    /// `TimeDelayNetBlock`: ReLU of the "same" (reflect-padded) conv `prefix`
    /// with its bias, over the `cin`-channel activation at word offset `x.1` of
    /// `x.0`.
    #[allow(clippy::too_many_arguments)]
    fn tdnn(&self, lay: &Layout, x: (&DeviceBuffer, u32), cin: u32, prefix: &str, cout: u32, k: u32, dilation: u32) -> Seg {
        let pad = dilation * (k - 1) / 2;
        let words = lay.numel(cin);
        let input = if pad > 0 {
            // Every segment's margins rewritten as its utterance's own reflection.
            let y = self.st(words);
            let src = (u64::from(x.1), u64::from(words));
            self.run(self.gpu().step_sliced(SEG_REFLECT, &[x.0, &y, &lay.lens], &[src, (0, 0), (0, 0)], &[words, lay.n, lay.s, lay.off], words));
            y
        } else if x.1 == 0 {
            x.0.clone()
        } else {
            let y = self.st(words);
            self.copy(x.0, x.1, &y, 0, words);
            y
        };
        let l = lay.n * lay.s;
        let c = audio::conv::Conv1d { n: 1, cin, l, cout, k, stride: 1, pad, dilation, groups: 1, lo: l };
        Seg { buf: self.relu(&self.conv(&c, &input, &format!("{prefix}.weight"), &format!("{prefix}.bias")), lay.numel(cout)), c: cout }
    }

    /// `conv1d` of `x` by `wname` with the bias `bname` fused, into a fresh buffer.
    fn conv(&self, c: &audio::conv::Conv1d, x: &DeviceBuffer, wname: &str, bname: &str) -> DeviceBuffer {
        let y = self.st(c.n * c.cout * c.lo);
        self.run_all(audio::conv::conv1d_bias_fwd(self.gpu(), &CONV, c, x, self.w(wname), self.w(bname), &y, &mut self.scratch.borrow_mut()));
        y
    }
    fn relu(&self, x: &DeviceBuffer, total: u32) -> DeviceBuffer {
        let out = self.st(total);
        self.run(self.gpu().step(LEAKY_RELU, &[x, &out], &[total, f(0.0)], total));
        out
    }
    fn add2(&self, a: &DeviceBuffer, b: &DeviceBuffer, total: u32) -> DeviceBuffer {
        let out = self.st(total);
        self.run(self.gpu().step(ADD2, &[a, b, &out], &[total], total));
        out
    }
    /// In-place per-feature bias `bname` on a token-major `[m, n]` buffer.
    fn bias_add(&self, x: &DeviceBuffer, bname: &str, m: u32, n: u32) {
        self.run(self.gpu().step(BIAS_ADD, &[x, self.w(bname)], &[m, n], m * n));
    }
    /// `words` words of `src` from word `from` into `dst` at word `to`.
    fn copy(&self, src: &DeviceBuffer, from: u32, dst: &DeviceBuffer, to: u32, words: u32) {
        let range = |at: u32| (u64::from(at), u64::from(words));
        self.run(self.gpu().step_sliced(REGION_COPY, &[src, dst], &[range(from), range(to)], &[1, words, words, 0], words));
    }
    /// `x[m, k] @ W[n, k]ᵀ` for the weight `wname`.
    fn linear(&self, x: &DeviceBuffer, wname: &str, m: u32, k: u32, n: u32) -> DeviceBuffer {
        let out = self.st(m * n);
        self.run(self.gpu().step(MATMUL, &[x, self.w(wname), &out], &[m, k, n], m * n));
        out
    }
    /// A per-utterance `[n, c]` vector laid out `[c, n]`, the per-row term of
    /// a `[c, n, S]` activation.
    fn per_row(&self, v: &DeviceBuffer, n: u32, c: u32) -> DeviceBuffer {
        let out = self.st(n * c);
        // `[n, c]` read as NCHW `[1, n, c]`, written NLC `[1, c, n]`.
        self.run(self.gpu().step(NCHW_NLC, &[v, &out], &[n * c, n, c], n * c));
        out
    }
    /// Per-row mean and std over each utterance's span: uniform, or weighted
    /// by the softmax of `logits` over it. Both `[n, c]`.
    fn pool(&self, lay: &Layout, x: &Seg, logits: Option<&DeviceBuffer>) -> (DeviceBuffer, DeviceBuffer) {
        let (mean, std) = (self.st(lay.n * x.c), self.st(lay.n * x.c));
        let a = logits.unwrap_or(&x.buf);
        let params = [x.c, lay.n, lay.s, lay.off, f(ASP_EPS), u32::from(logits.is_some())];
        self.run(self.gpu().step(SEG_POOL_STATS, &[&x.buf, a, &lay.lens, &mean, &std], &params, x.c * lay.n));
        (mean, std)
    }

    /// `Res2NetBlock` (scale 8) over `[C, n, S]`: chunk 0 passes through,
    /// `out[1] = tdnn(chunk1)`, `out[i] = tdnn(chunk_i + out[i-1])`, the
    /// outputs concatenated along channels (each chunk a contiguous range).
    fn res2net(&self, lay: &Layout, h: &Seg, prefix: &str, dilation: u32) -> Seg {
        let scale = self.enc.cfg.enc_res2net_scale;
        let sub = h.c / scale;
        let chunk = lay.numel(sub);
        let out = self.st(lay.numel(h.c));
        self.copy(&h.buf, 0, &out, 0, chunk);
        for i in 1..scale {
            let name = format!("{prefix}.res2net_block.blocks.{}.conv", i - 1);
            let y = if i == 1 {
                self.tdnn(lay, (&h.buf, chunk), sub, &name, sub, 3, dilation)
            } else {
                let sum = self.st(chunk);
                let range = |at: u32| (u64::from(at), u64::from(chunk));
                self.run(self.gpu().step_sliced(ADD2, &[&h.buf, &out, &sum], &[range(i * chunk), range((i - 1) * chunk), (0, 0)], &[chunk], chunk));
                self.tdnn(lay, (&sum, 0), sub, &name, sub, 3, dilation)
            };
            self.copy(&y.buf, 0, &out, i * chunk, chunk);
        }
        Seg { buf: out, c: h.c }
    }

    /// `SqueezeExcitationBlock`: gate each channel of `x` by a sigmoid of two
    /// 1x1 convs over the utterance's time mean.
    fn se_block(&self, lay: &Layout, x: &Seg, prefix: &str) -> Seg {
        let se = self.enc.cfg.enc_se_channels;
        let n = lay.n;
        let (mean, _) = self.pool(lay, x, None);
        let g = self.linear(&mean, &format!("{prefix}.se_block.conv1.weight"), n, x.c, se);
        self.bias_add(&g, &format!("{prefix}.se_block.conv1.bias"), n, se);
        let g = self.relu(&g, n * se);
        let g = self.linear(&g, &format!("{prefix}.se_block.conv2.weight"), n, se, x.c);
        self.bias_add(&g, &format!("{prefix}.se_block.conv2.bias"), n, x.c);
        let gate = self.st(n * x.c);
        self.run(self.gpu().step(SIGMOID, &[&g, &gate], &[n * x.c], n * x.c));
        let gate = self.per_row(&gate, n, x.c);
        let total = lay.numel(x.c);
        let out = self.st(total);
        self.run(self.gpu().step(SCALE_CHAN, &[&x.buf, &gate, &out], &[total, x.c * n, lay.s], total));
        Seg { buf: out, c: x.c }
    }

    /// One `SqueezeExcitationRes2NetBlock`: tdnn1 -> res2net -> tdnn2 -> se, plus
    /// the input residual.
    fn se_res2net(&self, lay: &Layout, x: &Seg, prefix: &str, dilation: u32) -> Seg {
        let c = x.c;
        let h = self.tdnn(lay, (&x.buf, 0), c, &format!("{prefix}.tdnn1.conv"), c, 1, 1);
        let h = self.res2net(lay, &h, prefix, dilation);
        let h = self.tdnn(lay, (&h.buf, 0), c, &format!("{prefix}.tdnn2.conv"), c, 1, 1);
        let h = self.se_block(lay, &h, prefix);
        Seg { buf: self.add2(&x.buf, &h.buf, lay.numel(c)), c }
    }

    /// `AttentiveStatisticsPooling` over `[C, n, S]`: the attention-weighted
    /// mean and std of every channel, each `[n, C]`.
    fn asp(&self, lay: &Layout, x: &Seg) -> (DeviceBuffer, DeviceBuffer) {
        let att = self.enc.cfg.enc_attention_channels;
        let n = lay.n;
        // The TDNN over [x, mean, std]: its x block as a conv, its mean and
        // std blocks (and its bias) as a per-(utterance, channel) term.
        let (umean, ustd) = self.pool(lay, x, None);
        let term = self.linear(&umean, ASP_MEAN, n, x.c, att);
        let term_std = self.linear(&ustd, ASP_STD, n, x.c, att);
        let term = self.add2(&term, &term_std, n * att);
        self.bias_add(&term, "asp.tdnn.conv.bias", n, att);
        let term = self.per_row(&term, n, att);
        let l = n * lay.s;
        let pointwise = |cin: u32, cout: u32| audio::conv::Conv1d { n: 1, cin, l, cout, k: 1, stride: 1, pad: 0, dilation: 1, groups: 1, lo: l };
        let h = self.conv(&pointwise(x.c, att), &x.buf, ASP_X, ASP_ZERO_BIAS);
        let total = lay.numel(att);
        self.run(self.gpu().step(ADD_CHAN_INPLACE, &[&h, &term], &[total, att * n, lay.s], total));
        let h = self.relu(&h, total);
        let t = self.st(total);
        self.run(self.gpu().step(TANH_ACT, &[&h, &t], &[total], total));
        // conv (att -> C, k1) + bias: the attention logits.
        let logits = self.conv(&pointwise(att, x.c), &t, "asp.conv.weight", "asp.conv.bias");
        self.pool(lay, x, Some(&logits))
    }
}
