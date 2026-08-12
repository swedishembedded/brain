// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-ASR audio encoder - Whisper/Qwen-omni style, on the shared WGSL engine.
//!
//! Pipeline (mel `[num_mel, T]`, T a multiple of `chunk_len = 2*n_window = 100`):
//!   chunk into `[num_chunks, 1, 128, 100]` → 3× `conv2d(k3,s2,p1)+bias` + erf-GELU
//!   → conv_out `Linear(480*16 → 1024)` (bias-free) → + sinusoidal abs pos
//!   → pack the valid (non-padding) post-CNN positions into a flat `[n_audio, 1024]`
//!   → `n_layers`× pre-LN transformer block (block-diagonal window attention over
//!      `cu_seqlens`, no RoPE, no QK-norm) via the shared `model::vit` builder
//!   → final LayerNorm (`ln_post`).
//! The multi-modal projector (`Linear→GELU→Linear`, 1024→1024→2048) turns the
//! encoder output into decoder-space audio embeddings.
//!
//! Parity-gated against `Qwen3ASREncoder` - see the test at the bottom.

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::vit::{vit_block_fwd, VitBlockWeights, VitKernelIds, VitScratch, VitShape};

use crate::config::AudioEncoderConfig;

/// Kernel pipeline for the audio encoder. Indices 0..=11 mirror the `model::vit`
/// forward kernels (so `vit_ids()` lines up); index 12 is the conv stem.
pub fn audio_pipelines() -> &'static [(&'static str, &'static str)] {
    &[
        ("layernorm", kernels::LAYERNORM),                 // 0
        ("matmul", kernels::MATMUL),                       // 1
        ("matmul_rows", kernels::MATMUL_ROWS),             // 2
        ("bias_add", kernels::BIAS_ADD),                   // 3
        ("gelu_erf", kernels::GELU_ERF),                   // 4  (torch F.gelu)
        ("scale_chan", kernels::SCALE_CHAN),               // 5  (unused: no LayerScale)
        ("add2", kernels::ADD2),                           // 6
        ("attn_scores_cross", kernels::ATTN_SCORES_CROSS), // 7
        ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS), // 8
        ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),   // 9
        ("ln_head", kernels::LN_HEAD),                     // 10 (unused: no QK-norm)
        ("rope2d", kernels::ROPE2D),                       // 11 (unused: no RoPE)
        ("conv_bias", kernels::CONV_BIAS),                 // 12
    ]
}

const K_GELU: usize = 4;
const K_MATMUL: usize = 1;
const K_BIAS_ADD: usize = 3;
const K_LAYERNORM: usize = 0;
const K_CONV_BIAS: usize = 12;

fn vit_ids() -> VitKernelIds {
    VitKernelIds {
        layernorm: 0,
        matmul: 1,
        matmul_rows: 2,
        bias_add: 3,
        mlp_act: 4,
        scale_chan: 5,
        add2: 6,
        attn_scores_cross: 7,
        attn_softmax_cross: 8,
        attn_apply_cross: 9,
        ln_head: 10,
        rope2d: 11,
    }
}

/// Per-block weight leaves (post-import; q/k/v fused into `qkv`).
pub const BLOCK_LEAVES: &[&str] = &[
    "norm1.weight", "norm1.bias", "qkv.weight", "qkv.bias", "proj.weight", "proj.bias", "norm2.weight", "norm2.bias",
    "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
];

pub struct AudioEncoder<'g> {
    gpu: &'g Gpu,
    cfg: AudioEncoderConfig,
    w: HashMap<String, DeviceBuffer>,
    /// Sinusoidal absolute positional table `[max_pos, d_model]` (host).
    pos: Vec<f32>,
}

impl<'g> AudioEncoder<'g> {
    /// Required weight keys: `conv2d{1,2,3}.{weight,bias}`, `conv_out.weight`,
    /// per block `blocks.{b}.<leaf>` (see [`BLOCK_LEAVES`]), `ln_post.{weight,bias}`,
    /// and the projector `multi_modal_projector.linear_{1,2}.{weight,bias}`.
    pub fn new(gpu: &'g Gpu, cfg: AudioEncoderConfig, weights: &HashMap<String, Vec<f32>>) -> AudioEncoder<'g> {
        let w = weights.iter().map(|(k, v)| (k.clone(), gpu.storage_init(k, v))).collect();
        let pos = sinusoids(cfg.max_pos, cfg.d_model, 10000.0);
        AudioEncoder { gpu, cfg, w, pos }
    }

    fn wb(&self, name: &str) -> &DeviceBuffer {
        self.w.get(name).unwrap_or_else(|| panic!("audio encoder weight missing: {name}"))
    }

    /// Encode a `[num_mel, T]` (channels-first) log-mel spectrogram whose first
    /// `valid_frames` columns are real audio (the rest zero padding). Returns the
    /// encoder output `[n_audio, d_model]` and the projected audio embeddings
    /// `[n_audio, output_dim]`.
    pub fn encode(&self, mel: &[f32], valid_frames: u32) -> (Vec<f32>, Vec<f32>) {
        let (packed, n_audio, spans) = self.prepare_packed(mel, valid_frames);
        self.encode_packed(&packed, n_audio, &spans)
    }

    /// [`encode`](Self::encode) with the windowed-transformer HEAD supplied by a
    /// closure - the seam the NPU resident uses to run the audio-encoder ONNX head
    /// (`packed[n_audio·d], n_audio, spans → (encoder_out, audio_embeds)`) on the
    /// Intel NPU, while the conv stem + valid-position packing stay host-side.
    /// Bit-identical to `encode` when the closure reproduces `encode_packed` (the
    /// ONNX head is parity-gated to cosine 1.0 vs the device head).
    pub fn encode_with_head<F>(&self, mel: &[f32], valid_frames: u32, head: F) -> (Vec<f32>, Vec<f32>)
    where
        F: FnOnce(&[f32], u32, &[(u32, u32)]) -> (Vec<f32>, Vec<f32>),
    {
        let (packed, n_audio, spans) = self.prepare_packed(mel, valid_frames);
        head(&packed, n_audio, &spans)
    }

    /// Conv stem + sinusoidal-pos add + valid-position packing → the packed
    /// post-CNN tokens `[n_audio, d_model]`, `n_audio`, and the block-diagonal
    /// `cu_seqlens` attention spans. The host part shared by `encode` /
    /// `encode_with_head`.
    fn prepare_packed(&self, mel: &[f32], valid_frames: u32) -> (Vec<f32>, u32, Vec<(u32, u32)>) {
        let c = &self.cfg;
        let (nm, chunk_len) = (c.num_mel_bins as usize, c.chunk_len() as usize);
        let t = mel.len() / nm;
        assert!(t.is_multiple_of(chunk_len), "T ({t}) must be a multiple of chunk_len ({chunk_len})");
        let num_chunks = (t / chunk_len) as u32;
        let hidden = c.d_model;

        let conv_tokens = self.conv_stem(mel, num_chunks); // [num_chunks*13, hidden]
        let tpc = c.post_cnn_len(c.chunk_len()); // time-positions per full chunk (13)

        let mut packed: Vec<f32> = Vec::new();
        let mut chunk_valid = Vec::new();
        for ch in 0..num_chunks {
            let vf = valid_frames.saturating_sub(ch * c.chunk_len()).min(c.chunk_len());
            let vp = c.post_cnn_len(vf); // valid post-CNN positions in this chunk
            chunk_valid.push(vp);
            for tpos in 0..vp {
                let row = (ch * tpc + tpos) as usize;
                for d in 0..hidden as usize {
                    packed.push(conv_tokens[row * hidden as usize + d] + self.pos[tpos as usize * hidden as usize + d]);
                }
            }
        }
        let n_audio = (packed.len() / hidden as usize) as u32;
        let spans = self.cu_seqlens(valid_frames, &chunk_valid);
        (packed, n_audio, spans)
    }

    /// The windowed transformer + `ln_post` + projector over already-packed,
    /// pos-added tokens `[n_audio, d_model]` with block-diagonal `spans`
    /// (`(row0, len)`). Returns `(encoder_out [n_audio, d_model], audio_embeds
    /// [n_audio, output_dim])`. This is the NPU-portable head - `crates/npu` builds
    /// an ONNX graph parity-gated against it; `encode` = conv stem + pack + this.
    pub fn encode_packed(&self, packed: &[f32], n_audio: u32, spans: &[(u32, u32)]) -> (Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let hidden = c.d_model;
        if n_audio == 0 {
            return (Vec::new(), Vec::new());
        }

        // ---- windowed transformer blocks (device) ----
        let max_span = spans.iter().map(|&(_, l)| l).max().unwrap_or(n_audio);
        let sh = VitShape { dim: hidden, heads: c.n_heads, mlp: c.ffn_dim, eps: c.eps };
        let ids = vit_ids();
        let scr = VitScratch::new(self.gpu, &sh, n_audio, max_span, max_span);

        let x = self.gpu.storage_init("asr.x", packed);
        let mut steps: Vec<Step> = Vec::new();
        for b in 0..c.n_layers {
            let p = |leaf: &str| self.wb(&format!("blocks.{b}.{leaf}"));
            let bw = VitBlockWeights {
                norm1_w: p("norm1.weight"),
                norm1_b: p("norm1.bias"),
                qkv_w: p("qkv.weight"),
                qkv_b: p("qkv.bias"),
                qk_norm: None,
                rope: None,
                proj_w: p("proj.weight"),
                proj_b: p("proj.bias"),
                ls1: None,
                norm2_w: p("norm2.weight"),
                norm2_b: p("norm2.bias"),
                fc1_w: p("fc1.weight"),
                fc1_b: p("fc1.bias"),
                fc2_w: p("fc2.weight"),
                fc2_b: p("fc2.bias"),
                ls2: None,
            };
            vit_block_fwd(self.gpu, &ids, &sh, &bw, &x, n_audio, spans, max_span, &scr, &mut steps);
        }
        // ln_post
        let enc = self.gpu.storage((n_audio * hidden) as u64);
        steps.push(self.gpu.step(
            K_LAYERNORM,
            &[&x, self.wb("ln_post.weight"), self.wb("ln_post.bias"), &enc],
            &[hidden, n_audio, f(c.eps)],
            n_audio,
        ));
        self.gpu.submit(&[], &steps);
        let encoder_out = self.gpu.read(&enc, (n_audio * hidden) as usize);

        // ---- projector ----
        let audio_embeds = self.project(&enc, n_audio);
        (encoder_out, audio_embeds)
    }

    /// 3× conv2d(k3,s2,p1)+bias + erf-GELU, then conv_out Linear (bias-free) with
    /// the NCHW→(time, chan*freq) permute. Returns `[num_chunks*tpc, d_model]`.
    fn conv_stem(&self, mel: &[f32], num_chunks: u32) -> Vec<f32> {
        let g = self.gpu;
        let c = &self.cfg;
        let (nm, w0) = (c.num_mel_bins, c.chunk_len());
        let nc = num_chunks;
        let ch = c.downsample_hidden;

        // host reshape mel[nm, T] -> chunked [nc, 1, nm, w0]
        let t = (nc * w0) as usize;
        let mut chunked = vec![0.0f32; (nc * nm * w0) as usize];
        for cc in 0..nc as usize {
            for fq in 0..nm as usize {
                for j in 0..w0 as usize {
                    chunked[(cc * nm as usize + fq) * w0 as usize + j] = mel[fq * t + cc * w0 as usize + j];
                }
            }
        }

        let conv = |g: &Gpu, x: &DeviceBuffer, wname: &str, bname: &str, cin: u32, hh: u32, ww: u32, cout: u32| {
            let ho = (hh + 2 - 3) / 2 + 1;
            let wo = (ww + 2 - 3) / 2 + 1;
            let out = g.storage((nc * cout * ho * wo) as u64);
            let act = g.storage((nc * cout * ho * wo) as u64);
            let steps = vec![
                g.step(K_CONV_BIAS, &[x, self.wb(wname), self.wb(bname), &out], &[nc, cin, hh, ww, cout, 3, 2, 1, ho, wo], nc * cout * ho * wo),
                g.step(K_GELU, &[&out, &act], &[nc * cout * ho * wo], nc * cout * ho * wo),
            ];
            g.submit(&[], &steps);
            (act, ho, wo)
        };

        let x0 = g.storage_init("asr.mel", &chunked);
        let (a1, h1, w1) = conv(g, &x0, "conv2d1.weight", "conv2d1.bias", 1, nm, w0, ch);
        let (a2, h2, w2) = conv(g, &a1, "conv2d2.weight", "conv2d2.bias", ch, h1, w1, ch);
        let (a3, h3, w3) = conv(g, &a2, "conv2d3.weight", "conv2d3.bias", ch, h2, w2, ch);
        // a3: [nc, ch, h3, w3] NCHW.  Permute to conv_out input [nc*w3, ch*h3].
        let a3h = g.read(&a3, (nc * ch * h3 * w3) as usize);
        let rows = nc * w3;
        let cols = ch * h3; // conv_out_in = 480*16 = 7680
        let mut perm = vec![0.0f32; (rows * cols) as usize];
        for cc in 0..nc as usize {
            for chan in 0..ch as usize {
                for fr in 0..h3 as usize {
                    for tt in 0..w3 as usize {
                        let src = ((cc * ch as usize + chan) * h3 as usize + fr) * w3 as usize + tt;
                        let dst = (cc * w3 as usize + tt) * cols as usize + (chan * h3 as usize + fr);
                        perm[dst] = a3h[src];
                    }
                }
            }
        }
        // conv_out: [rows, cols] · conv_out.weight[hidden, cols]^T  (bias-free)
        let hidden = c.d_model;
        let pin = g.storage_init("asr.convperm", &perm);
        let pout = g.storage((rows * hidden) as u64);
        g.submit(&[], &[g.step(K_MATMUL, &[&pin, self.wb("conv_out.weight"), &pout], &[rows, cols, hidden], rows * hidden)]);
        g.read(&pout, (rows * hidden) as usize)
    }

    /// Multi-modal projector: Linear(d→d)+bias → erf-GELU → Linear(d→out)+bias.
    fn project(&self, enc: &DeviceBuffer, n_audio: u32) -> Vec<f32> {
        let g = self.gpu;
        let (d, out) = (self.cfg.d_model, self.cfg.output_dim);
        let h = g.storage((n_audio * d) as u64);
        let h2 = g.storage((n_audio * d) as u64);
        let o = g.storage((n_audio * out) as u64);
        let steps = vec![
            g.step(K_MATMUL, &[enc, self.wb("multi_modal_projector.linear_1.weight"), &h], &[n_audio, d, d], n_audio * d),
            g.step(K_BIAS_ADD, &[&h, self.wb("multi_modal_projector.linear_1.bias")], &[n_audio, d], n_audio * d),
            g.step(K_GELU, &[&h, &h2], &[n_audio * d], n_audio * d),
            g.step(K_MATMUL, &[&h2, self.wb("multi_modal_projector.linear_2.weight"), &o], &[n_audio, d, out], n_audio * out),
            g.step(K_BIAS_ADD, &[&o, self.wb("multi_modal_projector.linear_2.bias")], &[n_audio, out], n_audio * out),
        ];
        g.submit(&[], &steps);
        g.read(&o, (n_audio * out) as usize)
    }

    /// Block-diagonal attention windows over the packed post-CNN tokens, matching
    /// `get_audio_cu_seqlens`. Returns `(row0, len)` spans.
    fn cu_seqlens(&self, valid_frames: u32, chunk_valid: &[u32]) -> Vec<(u32, u32)> {
        let c = &self.cfg;
        let aftercnn: u32 = chunk_valid.iter().copied().sum(); // == _get_feat_extract_output_lengths(valid_frames)
        let _ = valid_frames;
        let max_len_after_cnn = c.post_cnn_len(c.chunk_len());
        let ratio = c.n_window_infer / c.chunk_len(); // 8
        let window = max_len_after_cnn * ratio; // 104
        let mut spans = Vec::new();
        let mut start = 0u32;
        let full = aftercnn / window;
        for _ in 0..full {
            spans.push((start, window));
            start += window;
        }
        let rem = aftercnn % window;
        if rem != 0 {
            spans.push((start, rem));
        }
        spans
    }
}

/// Whisper-style sinusoidal positional table `[length, channels]`:
/// `[sin(t*inv), cos(t*inv)]` with `inv = exp(-ln(max_ts)/(c/2-1) * arange(c/2))`.
fn sinusoids(length: u32, channels: u32, max_timescale: f32) -> Vec<f32> {
    let half = (channels / 2) as usize;
    let log_inc = max_timescale.ln() / (half as f32 - 1.0);
    let inv: Vec<f32> = (0..half).map(|i| (-log_inc * i as f32).exp()).collect();
    let mut pe = vec![0.0f32; (length * channels) as usize];
    for t in 0..length as usize {
        for i in 0..half {
            let s = t as f32 * inv[i];
            pe[t * channels as usize + i] = s.sin();
            pe[t * channels as usize + half + i] = s.cos();
        }
    }
    pe
}
