// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS 12 Hz codec — decode path (codes `[T,16]` -> 24 kHz waveform).
//!
//! Forward only; pure fp32 over the shared WGSL engine. The graph follows
//! `Qwen3TTSTokenizerV2Decoder.forward`:
//!
//! 1. `quantizer.decode` — SplitResidualVectorQuantizer: gather the semantic
//!    codebook (q0) and the 15 acoustic codebooks (q1..15) via `embed`, sum each
//!    group, run each group's `output_proj` (1×1 conv == matmul), and add the two
//!    -> latent `[512, T]`.
//! 2. `pre_conv` — causal `conv1d` 512->1024, k3.
//! 3. `pre_transformer` — 8-layer causal MHA (head_dim 64, 16 heads, half-split
//!    RoPE θ=1e4, RMSNorm, SiLU MLP, per-channel LayerScale on each residual),
//!    with `input_proj`/`output_proj` around it.
//! 4. `upsample.{0,1}` — causal transposed conv (×2) + ConvNeXt block.
//! 5. `decoder.{0..6}` — SEANet: head conv, 4 `DecoderBlock`s (SnakeBeta + causal
//!    transposed conv + 3 dilated residual units), tail SnakeBeta + conv to 1ch.
//! 6. clamp to [-1, 1].
//!
//! Layout convention: conv stages run in channel-major NCL `[C, L]`; the
//! transformer and the pointwise linears in ConvNeXt run token-major `[L, C]`.
//! The few layout flips are explicit host transposes (no transpose kernel
//! exists). ConvNeXt's `LayerNorm` (eps 1e-6) and exact-erf `GELU` are computed
//! on the host for bit-faithfulness (the device kernels use eps 1e-5 / the tanh
//! GELU approximation); everything else is on-device.

use std::collections::HashMap;

use bytemuck::cast_slice;
use gpu_core::{f, BufUsage, DeviceBuffer, Gpu};
use backend_cpu::par;
use model::block::{self, Gqa, KernelIds};
use paramstore::{ParamStore, Role};

use crate::config::CodecConfig;

// ---- kernel indices (order matches PIPELINES) ----
const EMBED: usize = 0;
const MATMUL: usize = 1;
const RMSNORM: usize = 2;
const ROPE: usize = 3;
const GQA_SCORES: usize = 4;
const ATTN_SOFTMAX: usize = 5;
const GQA_APPLY: usize = 6;
const SILU_MUL: usize = 7;
const ADD2: usize = 8;
const BIAS_ADD: usize = 9;
const CONV1D: usize = 10;
const CONVTR1D: usize = 11;
const SNAKE: usize = 12;
const SCALE_CHAN: usize = 13;
const AXPY: usize = 14;
const GQA_SCORES_WIN: usize = 15;

const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("matmul", kernels::MATMUL),
    ("rmsnorm", kernels::RMSNORM),
    ("rope_base", kernels::ROPE_BASE),
    ("gqa_scores", kernels::GQA_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("silu_mul", kernels::SILU_MUL),
    ("add2", kernels::ADD2),
    ("bias_add", kernels::BIAS_ADD),
    ("conv1d", kernels::CONV1D),
    ("convtr1d", kernels::CONVTR1D),
    ("snake_beta", kernels::SNAKE_BETA),
    ("scale_chan", kernels::SCALE_CHAN),
    ("axpy", kernels::AXPY),
    ("gqa_scores_win", kernels::GQA_SCORES_WIN),
];

/// SnakeBeta's `no_div_by_zero` (the reference's fixed epsilon).
const SNAKE_EPS: f32 = 1e-9;
/// ConvNeXt LayerNorm epsilon.
const LN_EPS: f32 = 1e-6;
/// Element-count threshold below which host loops stay single-threaded (rayon
/// fan-out costs more than it saves on tiny buffers). The decode path's hot host
/// loops (transposes, ConvNeXt LayerNorm + exact-erf GELU, bias broadcast, final
/// clamp) cross this once the sequence grows, so they spread across all cores.
const PAR_MIN: usize = 1 << 14;

/// The decode model: frozen weights on device + small (≤1-D) tensors mirrored on
/// the host for per-channel NCL bias broadcasts and the host LayerNorm/GELU.
pub struct Codec {
    pub gpu: Gpu,
    pub cfg: CodecConfig,
    ps: ParamStore,
    host: HashMap<String, Vec<f32>>,
}

impl Codec {
    /// Load an inference-only decoder from a brain checkpoint produced by
    /// [`crate::import::import`]. Every parameter is frozen (weights only).
    pub fn load_inference(weights_path: &str) -> Codec {
        let c = checkpoint::load(weights_path);
        let cfg = CodecConfig::from_json(&c.header["config"]);
        let init = c.by_role("");
        Codec::from_weights(cfg, init)
    }

    /// Build from an in-memory weight map (used by tests and [`load_inference`]).
    pub fn from_weights(cfg: CodecConfig, init: HashMap<String, Vec<f32>>) -> Codec {
        let gpu = Gpu::new_cpu(PIPELINES);
        // Mirror small tensors (biases, norms, scales, alphas/betas/gammas) on the
        // host: NCL conv-bias broadcasts and ConvNeXt's host LayerNorm read these.
        let host: HashMap<String, Vec<f32>> = init
            .iter()
            .filter(|(_, v)| v.len() <= 8192)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let roles: Vec<(String, usize, Role)> =
            init.iter().map(|(n, v)| (n.clone(), v.len(), Role::Frozen)).collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, &init);
        Codec { gpu, cfg, ps, host }
    }

    // -- tiny eager helpers (each submits immediately on the CPU backend) --

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
        let b = self.gpu.buffer("act", (data.len() * 4) as u64, BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC);
        self.gpu.write(&b, cast_slice(data));
        b
    }

    /// `out = x[M,K] @ W[N,K]ᵀ` (PyTorch `nn.Linear`, no bias).
    fn matmul(&self, x: &DeviceBuffer, wname: &str, m: u32, k: u32, n: u32) -> DeviceBuffer {
        let out = self.st((m * n) as usize);
        self.run(self.gpu.step(MATMUL, &[x, self.w(wname), &out], &[m, k, n], m * n));
        out
    }
    /// In-place per-output-feature bias for a token-major `[M,N]` buffer.
    fn bias_add(&self, buf: &DeviceBuffer, bias_name: &str, m: u32, n: u32) {
        self.run(self.gpu.step(BIAS_ADD, &[buf, self.w(bias_name)], &[m, n], m * n));
    }
    fn add2(&self, a: &DeviceBuffer, b: &DeviceBuffer, total: u32) -> DeviceBuffer {
        let out = self.st(total as usize);
        self.run(self.gpu.step(ADD2, &[a, b, &out], &[total], total));
        out
    }
    /// SnakeBeta over NCL `[C, L]`: `y = x + (1/(exp(β)+ε))·sin(exp(α)·x)²`.
    fn snake(&self, x: &DeviceBuffer, prefix: &str, c: u32, l: u32) -> DeviceBuffer {
        let out = self.st((c * l) as usize);
        let (a, b) = (format!("{prefix}.alpha"), format!("{prefix}.beta"));
        let total = c * l;
        self.run(self.gpu.step(SNAKE, &[x, self.w(&a), self.w(&b), &out], &[total, c, l, f(SNAKE_EPS)], total));
        out
    }
    /// Per-channel scale over a `[rows, c, inner]` layout: `y = x·scale[c]` where
    /// channel `= (idx / inner) % c` (LayerScale / ConvNeXt γ). `total` is the
    /// full element count (`rows·c·inner`).
    fn scale_chan(&self, x: &DeviceBuffer, scale_name: &str, total: u32, c: u32, inner: u32) -> DeviceBuffer {
        let out = self.st(total as usize);
        self.run(self.gpu.step(SCALE_CHAN, &[x, self.w(scale_name), &out], &[total, c, inner], total));
        out
    }
    /// Transpose a row-major `[a, b]` buffer to `[b, a]` on the host. Output rows
    /// (`b` of them, each gathering one column of the input) are independent, so
    /// they fan out across cores once the buffer is large.
    fn transpose(&self, x: &DeviceBuffer, a: u32, b: u32) -> DeviceBuffer {
        let v = self.gpu.read(x, (a * b) as usize);
        let (a, b) = (a as usize, b as usize);
        let mut o = vec![0.0f32; a * b];
        let fill = |j: usize, row: &mut [f32]| {
            for (i, dst) in row.iter_mut().enumerate() {
                *dst = v[i * b + j];
            }
        };
        if a * b >= PAR_MIN {
            par::rows_mut(&mut o, a, fill);
        } else {
            o.chunks_mut(a).enumerate().for_each(|(j, row)| fill(j, row));
        }
        self.upload(&o)
    }

    /// Causal `conv1d` in NCL: left pad `dilation*(K-1)`, stride 1, `Lo == L`,
    /// then add the per-channel bias (`{prefix}.conv.bias`). Returns `[Cout, L]`.
    fn causal_conv(&self, x: &DeviceBuffer, prefix: &str, cin: u32, cout: u32, l: u32, k: u32, dilation: u32, groups: u32) -> DeviceBuffer {
        let pad = dilation * (k - 1);
        let c = audio::conv::Conv1d { n: 1, cin, l, cout, k, stride: 1, pad, dilation, groups, lo: l };
        let out = self.st((cout * l) as usize);
        self.run(audio::conv::conv1d_fwd(
            &self.gpu,
            &audio::conv::ConvKernels { fwd: CONV1D, dx: 0, dw: 0 },
            &c,
            x,
            self.w(&format!("{prefix}.conv.weight")),
            &out,
        ));
        self.add_ncl_bias(&out, &format!("{prefix}.conv.bias"), cout, l)
    }

    /// Causal transposed conv in NCL (upsampling by `stride`). The reference crops
    /// `K-stride` samples off the right; cropping the right == keeping the first
    /// `L·stride` outputs, so we just request `Lo = L·stride` (pad 0). Returns
    /// `[Cout, L·stride]`.
    fn causal_convtr(&self, x: &DeviceBuffer, prefix: &str, cin: u32, cout: u32, l: u32, k: u32, stride: u32) -> DeviceBuffer {
        let lo = l * stride;
        let c = audio::conv::Conv1d { n: 1, cin, l, cout, k, stride, pad: 0, dilation: 1, groups: 1, lo };
        let out = self.st((cout * lo) as usize);
        self.run(audio::conv::convtr1d_fwd(
            &self.gpu,
            &audio::conv::ConvKernels { fwd: CONVTR1D, dx: 0, dw: 0 },
            &c,
            x,
            self.w(&format!("{prefix}.conv.weight")),
            &out,
        ));
        self.add_ncl_bias(&out, &format!("{prefix}.conv.bias"), cout, lo)
    }

    /// Add a per-channel bias to an NCL `[C, L]` buffer (channel = idx / L). The
    /// engine's `bias_add` indexes the *inner* axis, so we broadcast the host bias
    /// to `[C, L]` and use `add2`.
    fn add_ncl_bias(&self, x: &DeviceBuffer, bias_name: &str, c: u32, l: u32) -> DeviceBuffer {
        let bias = &self.host[bias_name];
        let (c, l) = (c as usize, l as usize);
        let mut bcast = vec![0.0f32; c * l];
        let fill = |ch: usize, row: &mut [f32]| row.fill(bias[ch]);
        if c * l >= PAR_MIN {
            par::rows_mut(&mut bcast, l, fill);
        } else {
            bcast.chunks_mut(l).enumerate().for_each(|(ch, row)| fill(ch, row));
        }
        let bbuf = self.upload(&bcast);
        self.add2(x, &bbuf, (c * l) as u32)
    }

    // ------------------------------------------------------------------
    // decode
    // ------------------------------------------------------------------

    /// Decode `codes` (`[T,16]` row-major, q0 semantic + q1..15 acoustic) into a
    /// mono 24 kHz waveform (`≈ T·1920` samples).
    pub fn decode(&self, codes: &[u32]) -> Vec<f32> {
        let nq = self.cfg.num_quantizers as usize;
        assert_eq!(codes.len() % nq, 0, "codes length not a multiple of {nq}");
        let t = (codes.len() / nq) as u32;
        assert!(t > 0, "empty codes");
        let dim = self.cfg.codebook_dim / 2; // per-codebook dim (256)
        let lat = self.cfg.codebook_dim; // 512 (quantizer output)
        let hidden = self.cfg.hidden_size; // 512
        let latent = self.cfg.latent_dim; // 1024

        // --- 1. quantizer.decode ---
        let gather = |table: &str, col: usize| -> DeviceBuffer {
            let col_codes: Vec<u32> = (0..t as usize).map(|ti| codes[ti * nq + col]).collect();
            let codes_buf = {
                let b = self.gpu.buffer("codes", (t as u64) * 4, BufUsage::STORAGE | BufUsage::COPY_DST);
                self.gpu.write(&b, &col_codes);
                b
            };
            let out = self.st((t * dim) as usize);
            self.run(self.gpu.step(EMBED, &[&codes_buf, self.w(table), &out], &[dim, t], t * dim));
            out
        };
        // semantic group (rvq_first): 1 codebook.
        let sem = gather("quantizer.rvq_first.vq.layers.0.table", 0);
        let first = self.matmul(&sem, "quantizer.rvq_first.output_proj.weight", t, dim, lat);
        // acoustic group (rvq_rest): codebooks q1..15 summed, then projected.
        let mut acc = gather("quantizer.rvq_rest.vq.layers.0.table", 1);
        for i in 1..(nq - 1) {
            let g = gather(&format!("quantizer.rvq_rest.vq.layers.{i}.table"), i + 1);
            acc = self.add2(&acc, &g, t * dim);
        }
        let rest = self.matmul(&acc, "quantizer.rvq_rest.output_proj.weight", t, dim, lat);
        let quant_tm = self.add2(&first, &rest, t * lat); // [T, 512] token-major
        let quant = self.transpose(&quant_tm, t, lat); // [512, T] NCL

        // --- 2. pre_conv (512 -> 1024, k3 causal) ---
        let pre = self.causal_conv(&quant, "pre_conv", lat, latent, t, 3, 1, 1); // [1024, T]
        let mut x = self.transpose(&pre, latent, t); // [T, 1024] token-major

        // --- 3. pre_transformer ---
        x = self.matmul(&x, "pre_transformer.input_proj.weight", t, latent, hidden);
        self.bias_add(&x, "pre_transformer.input_proj.bias", t, hidden);
        x = self.transformer(&x, t);
        x = self.matmul(&x, "pre_transformer.output_proj.weight", t, hidden, latent);
        self.bias_add(&x, "pre_transformer.output_proj.bias", t, latent);
        let mut h = self.transpose(&x, t, latent); // [1024, L] NCL
        let mut l = t;

        // --- 4. upsample stages ---
        for (u, &factor) in self.cfg.upsampling_ratios.clone().iter().enumerate() {
            h = self.causal_convtr(&h, &format!("upsample.{u}.0"), latent, latent, l, factor, factor);
            l *= factor;
            h = self.convnext(&h, &format!("upsample.{u}.1"), latent, l);
        }

        // --- 5. SEANet decoder ---
        let dec_dim = self.cfg.decoder_dim; // 1536
        h = self.causal_conv(&h, "decoder.0", latent, dec_dim, l, 7, 1, 1); // [1536, L]
        let rates = self.cfg.upsample_rates.clone();
        for (i, &rate) in rates.iter().enumerate() {
            let in_dim = dec_dim >> i;
            let out_dim = dec_dim >> (i + 1);
            let bp = format!("decoder.{}", i + 1);
            // block.0 SnakeBeta(in_dim)
            h = self.snake(&h, &format!("{bp}.block.0"), in_dim, l);
            // block.1 causal transposed conv (in->out, k=2·rate, stride=rate)
            h = self.causal_convtr(&h, &format!("{bp}.block.1"), in_dim, out_dim, l, 2 * rate, rate);
            l *= rate;
            // block.2/3/4 dilated residual units (dilation 1, 3, 9)
            for (j, dil) in [(2u32, 1u32), (3, 3), (4, 9)] {
                h = self.residual_unit(&h, &format!("{bp}.block.{j}"), out_dim, l, dil);
            }
        }
        let out_dim = dec_dim >> rates.len(); // 96
        h = self.snake(&h, "decoder.5", out_dim, l); // tail SnakeBeta
        h = self.causal_conv(&h, "decoder.6", out_dim, 1, l, 7, 1, 1); // -> [1, L]

        // --- 6. clamp ---
        let mut wav = self.gpu.read(&h, l as usize);
        if wav.len() >= PAR_MIN {
            par::each_mut(&mut wav, |_, s| *s = s.clamp(-1.0, 1.0));
        } else {
            for s in &mut wav {
                *s = s.clamp(-1.0, 1.0);
            }
        }
        wav
    }

    /// Qwen3-Omni's Code2Wav decode: same SEANet decoder/upsample/pre-transformer
    /// SHAPE as [`Self::decode`] (config-compatible: `hidden_size 1024`,
    /// `intermediate_size 3072` on this `CodecConfig`, matching the real
    /// `code2wav_config` -- reused unchanged, not re-derived), but the input path
    /// is genuinely different, not a config difference:
    /// `Qwen3OmniMoeCode2Wav.forward`: `hidden = code_embedding(codes +
    /// code_offset).mean(1)` -- ONE combined `[num_quantizers * codebook_size,
    /// hidden_size]` embedding table (`code_offset[q] = q * codebook_size`,
    /// not persisted as a weight), MEANED over the 16 quantizers per frame,
    /// straight to `hidden_size` -- replacing [`Self::decode`]'s
    /// `quantizer.decode` (per-group RVQ dequant + `output_proj`) AND
    /// `pre_conv` entirely; `pre_transformer` has no `input_proj`/`output_proj`
    /// either (`hidden_size` IS the working width throughout, no separate
    /// `latent_dim`).
    ///
    /// The other genuine (non-config) difference: `Qwen3OmniMoeCausalTransConvNet`
    /// crops `pad = K - stride` off BOTH sides of the native `ConvTranspose1d`
    /// output (`Lo = (L-1)*stride - 2*pad + K`), where [`Self::causal_convtr`]
    /// crops only the right (`pad = 0`, `Lo = L*stride`) per its own doc comment
    /// -- correct for the standalone Qwen3-TTS codec, but NOT what Omni's
    /// reference does. For the upsample stages (`K == stride`, so `pad = 0`)
    /// the two conventions agree and [`Self::causal_convtr`] is reused unchanged;
    /// the SEANet decoder's own transposed convs (`K = 2*stride`, `pad =
    /// stride`) need the symmetric crop, so this function calls the
    /// `audio::conv` primitives directly for those with an explicit `pad`
    /// (`convtr1d.wgsl`'s `pad` param already matches PyTorch `ConvTranspose1d`'s
    /// native symmetric `padding` semantics -- confirmed by reading the kernel;
    /// no new device math). Verified against a real golden
    /// (`tools/goldens/qwen3omnimoe_dump_reference.py`'s `code2wav`): the naive
    /// `Lo = L*stride` assumption produces the wrong waveform LENGTH (15360 vs
    /// the golden's 14805 samples for `T=8`), which is what surfaced this.
    ///
    /// `codes` is `[T, num_quantizers]` row-major (same convention as
    /// [`Self::decode`]).
    pub fn decode_omni(&self, codes: &[u32]) -> Vec<f32> {
        let nq = self.cfg.num_quantizers as usize;
        assert_eq!(codes.len() % nq, 0, "codes length not a multiple of {nq}");
        let t = (codes.len() / nq) as u32;
        assert!(t > 0, "empty codes");
        let hidden = self.cfg.hidden_size; // 1024

        // --- 1. code_embedding(codes + code_offset).mean(1) ---
        let x = self.code_embedding_mean(codes, t, nq, hidden);

        // --- 2. pre_transformer (no input_proj/output_proj: hidden_size IS the
        //        working width already) ---
        let x = self.transformer(&x, t);
        let mut h = self.transpose(&x, t, hidden); // [hidden, T] NCL
        let mut l = t;

        // --- 3. upsample stages (K == stride here, so the crop conventions
        //        agree -- causal_convtr reused unchanged) ---
        for (u, &factor) in self.cfg.upsampling_ratios.clone().iter().enumerate() {
            h = self.causal_convtr(&h, &format!("upsample.{u}.0"), hidden, hidden, l, factor, factor);
            l *= factor;
            h = self.convnext(&h, &format!("upsample.{u}.1"), hidden, l);
        }

        // --- 4. SEANet decoder (symmetric-crop transposed conv, see doc) ---
        let dec_dim = self.cfg.decoder_dim; // 1536
        h = self.causal_conv(&h, "decoder.0", hidden, dec_dim, l, 7, 1, 1);
        let rates = self.cfg.upsample_rates.clone();
        for (i, &rate) in rates.iter().enumerate() {
            let in_dim = dec_dim >> i;
            let out_dim = dec_dim >> (i + 1);
            let bp = format!("decoder.{}", i + 1);
            h = self.snake(&h, &format!("{bp}.block.0"), in_dim, l);
            let new_l = (l - 1) * rate; // symmetric crop: Lo = (L-1)*stride, see doc
            h = self.causal_convtr_sym(&h, &format!("{bp}.block.1"), in_dim, out_dim, l, 2 * rate, rate, rate, new_l);
            l = new_l;
            for (j, dil) in [(2u32, 1u32), (3, 3), (4, 9)] {
                h = self.residual_unit(&h, &format!("{bp}.block.{j}"), out_dim, l, dil);
            }
        }
        let out_dim = dec_dim >> rates.len();
        h = self.snake(&h, "decoder.5", out_dim, l);
        h = self.causal_conv(&h, "decoder.6", out_dim, 1, l, 7, 1, 1);

        // --- 5. clamp ---
        let mut wav = self.gpu.read(&h, l as usize);
        if wav.len() >= PAR_MIN {
            par::each_mut(&mut wav, |_, s| *s = s.clamp(-1.0, 1.0));
        } else {
            for s in &mut wav {
                *s = s.clamp(-1.0, 1.0);
            }
        }
        wav
    }

    /// Chunked variant of [`Self::decode_omni`]: yields audio incrementally
    /// via `on_chunk` instead of building one `Vec` -- bounded per-chunk
    /// memory for the (expensive) upsample+SEANet stage instead of holding
    /// the whole `~T*1920`-sample buffer, and a real streaming decode a
    /// caller can start emitting from before the whole waveform is ready.
    ///
    /// The pre-transformer front runs ONCE over the full `codes` sequence,
    /// same as `decode_omni` -- there is no O(chunks²) cost to chunking
    /// here, since only the back (upsample + SEANet) actually needs
    /// persistent cross-chunk state (`decode_stream::Back::new_omni`, the
    /// symmetric-crop-aware streaming primitives added alongside this).
    /// Bit-exact vs `decode_omni` on the same input regardless of
    /// `chunk_frames` -- proven in `crates/codec/tests/`, not just asserted
    /// here.
    ///
    /// `chunk_frames` is measured in CODE frames (12.5 Hz), not output
    /// samples -- matching the reference's own `chunked_decode(chunk_size=
    /// 300, ...)` convention (though this implementation's front/back split
    /// is real streaming state, not the reference's re-decode-and-discard
    /// `left_context_size` shape). `chunk_frames == 0` decodes the whole
    /// thing in one `Back::step` call.
    pub fn decode_omni_chunked(&self, codes: &[u32], chunk_frames: usize, mut on_chunk: impl FnMut(&[f32])) {
        let nq = self.cfg.num_quantizers as usize;
        assert_eq!(codes.len() % nq, 0, "codes length not a multiple of {nq}");
        let t = (codes.len() / nq) as u32;
        assert!(t > 0, "empty codes");
        let hidden = self.cfg.hidden_size;

        // --- front: identical to decode_omni's own (correctly
        // sliding-window-attended), run ONCE over the whole
        // sequence -- see this fn's own doc for why that's not a chunking
        // cost. ---
        let x = self.code_embedding_mean(codes, t, nq, hidden);
        let x = self.transformer(&x, t);
        let h = self.transpose(&x, t, hidden); // [hidden, T] NCL
        let latent: Vec<f32> = self.gpu.read(&h, (hidden * t) as usize);

        // --- back: streamed, chunk_frames code-frames at a time. ---
        let w = self.back_weights();
        let mut back = crate::decode_stream::Back::new_omni(&w, &self.cfg);
        let chunk = if chunk_frames == 0 { t as usize } else { chunk_frames.max(1) };
        let mut a = 0usize;
        while a < t as usize {
            let b = (a + chunk).min(t as usize);
            let l_new = b - a;
            let mut slab = vec![0.0f32; hidden as usize * l_new];
            for c in 0..hidden as usize {
                slab[c * l_new..c * l_new + l_new].copy_from_slice(&latent[c * t as usize + a..c * t as usize + b]);
            }
            let out = back.step(&slab, l_new);
            if !out.is_empty() {
                on_chunk(&out);
            }
            a = b;
        }
    }

    /// This Code2Wav's `upsample.*`/`decoder.*` weights as a plain host map
    /// -- what the pure-CPU streaming back (`decode_stream::Back`) needs.
    /// Read once via `ParamStore::read_weight` (not per-chunk): `Back`'s own
    /// persistent state is what amortizes the cost across chunks; this is
    /// just getting those weights off the `ParamStore` once at session
    /// start, mirroring `codec_bridge::codec_weights`'s existing "strip a
    /// prefix, read every match" shape for a live `Codec` instead of a raw
    /// checkpoint reader.
    fn back_weights(&self) -> HashMap<String, Vec<f32>> {
        self.ps
            .params
            .iter()
            .filter(|(name, _)| name.starts_with("upsample.") || name.starts_with("decoder."))
            .map(|(name, _)| (name.clone(), self.ps.read_weight(&self.gpu, name)))
            .collect()
    }

    /// `hidden[t] = mean_q( code_embedding[ codes[t,q] + q*codebook_size ] )` --
    /// `codes` is `[T, nq]` row-major; returns token-major `[T, hidden]`.
    /// `codebook_size` is deliberately the SAME stride for every quantizer
    /// (including the semantic one, whose true vocab is larger --
    /// `semantic_codebook_size` -- matching `Qwen3OmniMoeCode2Wav`'s own
    /// `code_offset = arange(num_quantizers) * config.codebook_size` exactly;
    /// not a brain-side simplification).
    fn code_embedding_mean(&self, codes: &[u32], t: u32, nq: usize, hidden: u32) -> DeviceBuffer {
        let cb = self.cfg.codebook_size;
        let acc = self.st((t * hidden) as usize);
        for q in 0..nq {
            let offset_codes: Vec<u32> = (0..t as usize).map(|ti| codes[ti * nq + q] + q as u32 * cb).collect();
            let codes_buf = {
                let b = self.gpu.buffer("codes", (t as u64) * 4, BufUsage::STORAGE | BufUsage::COPY_DST);
                self.gpu.write(&b, &offset_codes);
                b
            };
            let gathered = self.st((t * hidden) as usize);
            self.run(self.gpu.step(EMBED, &[&codes_buf, self.w("code_embedding.weight"), &gathered], &[hidden, t], t * hidden));
            self.run(self.gpu.step(AXPY, &[&acc, &gathered], &[t * hidden, f(1.0 / nq as f32)], t * hidden));
        }
        acc
    }

    /// Causal transposed conv (NCL) with an EXPLICIT symmetric crop `pad` and
    /// output length `lo` -- unlike [`Self::causal_convtr`] (which always
    /// assumes `pad = 0`, `lo = l*stride`), this is `Qwen3OmniMoeCausalTransConvNet`'s
    /// own convention: PyTorch `ConvTranspose1d(padding=pad)` crops `pad`
    /// samples off BOTH sides of the native output
    /// (`convtr1d.wgsl`'s `pad` param already matches this -- see
    /// [`Self::decode_omni`]'s doc for why a new helper, not a change to
    /// [`Self::causal_convtr`], which stays correct for the standalone
    /// Qwen3-TTS codec's own (right-only-crop) convention).
    #[allow(clippy::too_many_arguments)]
    fn causal_convtr_sym(&self, x: &DeviceBuffer, prefix: &str, cin: u32, cout: u32, l: u32, k: u32, stride: u32, pad: u32, lo: u32) -> DeviceBuffer {
        let c = audio::conv::Conv1d { n: 1, cin, l, cout, k, stride, pad, dilation: 1, groups: 1, lo };
        let out = self.st((cout * lo) as usize);
        self.run(audio::conv::convtr1d_fwd(
            &self.gpu,
            &audio::conv::ConvKernels { fwd: CONVTR1D, dx: 0, dw: 0 },
            &c,
            x,
            self.w(&format!("{prefix}.conv.weight")),
            &out,
        ));
        self.add_ncl_bias(&out, &format!("{prefix}.conv.bias"), cout, lo)
    }

    /// One SEANet residual unit: `x + conv2(snake(conv1(snake(x))))`, conv1 is a
    /// dilated causal k7 conv, conv2 a causal k1 conv.
    fn residual_unit(&self, x: &DeviceBuffer, prefix: &str, c: u32, l: u32, dil: u32) -> DeviceBuffer {
        let y = self.snake(x, &format!("{prefix}.act1"), c, l);
        let y = self.causal_conv(&y, &format!("{prefix}.conv1"), c, c, l, 7, dil, 1);
        let y = self.snake(&y, &format!("{prefix}.act2"), c, l);
        let y = self.causal_conv(&y, &format!("{prefix}.conv2"), c, c, l, 1, 1, 1);
        self.add2(x, &y, c * l)
    }

    /// ConvNeXt block (NCL in/out): depthwise causal conv -> LayerNorm -> pwconv1
    /// -> GELU -> pwconv2 -> γ scale -> residual. LayerNorm (eps 1e-6) and GELU
    /// (exact erf) are host-computed to match `nn.LayerNorm`/`nn.GELU`.
    fn convnext(&self, x: &DeviceBuffer, prefix: &str, c: u32, l: u32) -> DeviceBuffer {
        // depthwise causal conv (groups = C, k7)
        let dw = self.causal_conv(x, &format!("{prefix}.dwconv"), c, c, l, 7, 1, c); // [C, L]
        let mut tm = self.gpu.read(&self.transpose(&dw, c, l), (l * c) as usize); // host [L, C]
        // host LayerNorm over C
        host_layernorm(&mut tm, l as usize, c as usize, &self.host[&format!("{prefix}.norm.weight")], &self.host[&format!("{prefix}.norm.bias")], LN_EPS);
        let tmb = self.upload(&tm);
        // pwconv1 (C -> 4C) + bias, exact GELU on host
        let hid = c * 4;
        let g = self.matmul(&tmb, &format!("{prefix}.pwconv1.weight"), l, c, hid);
        self.bias_add(&g, &format!("{prefix}.pwconv1.bias"), l, hid);
        let mut gv = self.gpu.read(&g, (l * hid) as usize);
        par_gelu(&mut gv);
        let gbuf = self.upload(&gv);
        // pwconv2 (4C -> C) + bias
        let o = self.matmul(&gbuf, &format!("{prefix}.pwconv2.weight"), l, hid, c);
        self.bias_add(&o, &format!("{prefix}.pwconv2.bias"), l, c);
        // γ per-channel scale over token-major [L,C]: channel = idx % C (inner=1).
        let o = self.scale_chan(&o, &format!("{prefix}.gamma"), l * c, c, 1);
        let o_ncl = self.transpose(&o, l, c); // [C, L]
        self.add2(x, &o_ncl, c * l)
    }

    /// The 8-layer causal transformer over a token-major `[T, hidden]` buffer.
    fn transformer(&self, x0: &DeviceBuffer, t: u32) -> DeviceBuffer {
        let c = &self.cfg;
        let d = c.hidden_size;
        let ff = c.intermediate_size;
        let hd = c.head_dim;
        let nh = c.num_attention_heads;
        let nkv = c.num_key_value_heads;
        let hq = nh * hd;
        let hkv = nkv * hd;
        let theta = c.rope_theta;
        let ids = ids();
        let ga = Gqa { b: 1, t, n_heads: nh, n_kv_heads: nkv, head_dim: hd };

        let mut x = clone_buf(self, x0, t * d);
        for layer in 0..c.num_hidden_layers as usize {
            let p = |leaf: &str| format!("pre_transformer.layers.{layer}.{leaf}");
            // --- attention ---
            let xn = self.st((t * d) as usize);
            self.run(block::rmsnorm_fwd(&self.gpu, &ids, &x, self.w(&p("input_layernorm.weight")), &xn, d, t));
            let q = self.matmul(&xn, &p("self_attn.q_proj.weight"), t, d, hq);
            let k = self.matmul(&xn, &p("self_attn.k_proj.weight"), t, d, hkv);
            let v = self.matmul(&xn, &p("self_attn.v_proj.weight"), t, d, hkv);
            self.run(block::rope_fwd(&self.gpu, &ids, &q, t, nh, hd, hq, t, theta));
            self.run(block::rope_fwd(&self.gpu, &ids, &k, t, nkv, hd, hkv, t, theta));
            let scores = self.st((nh * t * t) as usize);
            let probs = self.st((nh * t * t) as usize);
            let ctx = self.st((t * hq) as usize);
            // Sliding-window causal, not plain causal: the real reference
            // (`Qwen3OmniMoeCode2WavAttention`/`MimiAttention`, both built on
            // `create_sliding_window_causal_mask(sliding_window=config.
            // sliding_window)`) masks key `j` out once `i-j >= sliding_window`
            // on EVERY forward call, not only a chunked/streaming one -- this
            // pre_transformer is Mimi-derived and both `decode` (standalone
            // Qwen3-TTS) and `decode_omni` (Qwen3-Omni's Code2Wav) share this
            // one `transformer()` body and this one `sliding_window` config
            // field (`CodecConfig::sliding_window`, real value 72 for both
            // released checkpoints). `gqa_fwd_win` degenerates to `gqa_fwd`'s
            // plain causal mask exactly when `sliding_window >= t`, so this is
            // the correct dispatch unconditionally, not a `decode_omni`-only
            // special case. NOTE: `enc_transformer` below has the identical
            // gap (its own `EncoderConfig::sliding_window`, 250, is likewise
            // parsed and never applied) -- left unfixed here, out of this
            // change's scope (the encode path is not on Qwen3-Omni's call
            // path at all).
            self.gpu.submit(&[], &block::gqa_fwd_win(&self.gpu, GQA_SCORES_WIN, &ids, &ga, self.cfg.sliding_window, &q, &k, &v, &scores, &probs, &ctx));
            let attn = self.matmul(&ctx, &p("self_attn.o_proj.weight"), t, hq, d);
            let attn = self.scale_chan(&attn, &p("self_attn_layer_scale.scale"), t * d, d, 1);
            x = self.add2(&x, &attn, t * d);
            // --- MLP ---
            let xn = self.st((t * d) as usize);
            self.run(block::rmsnorm_fwd(&self.gpu, &ids, &x, self.w(&p("post_attention_layernorm.weight")), &xn, d, t));
            let gate = self.matmul(&xn, &p("mlp.gate_proj.weight"), t, d, ff);
            let up = self.matmul(&xn, &p("mlp.up_proj.weight"), t, d, ff);
            let hmid = self.st((t * ff) as usize);
            self.run(block::swiglu_fwd(&self.gpu, &ids, &gate, &up, &hmid, t * ff));
            let mlp = self.matmul(&hmid, &p("mlp.down_proj.weight"), t, ff, d);
            let mlp = self.scale_chan(&mlp, &p("mlp_layer_scale.scale"), t * d, d, 1);
            x = self.add2(&x, &mlp, t * d);
        }
        let out = self.st((t * d) as usize);
        self.run(block::rmsnorm_fwd(&self.gpu, &ids, &x, self.w("pre_transformer.norm.weight"), &out, d, t));
        out
    }
}

// ======================================================================
// ENCODE: wav -> codes [T,16]  (HuggingFace Mimi encoder, forward only)
// ======================================================================
//
// Mirror of `MimiModel._encode_frame`:
//   1. SEANet conv encoder (`encoder.encoder.layers.*`) -> [512, L']
//   2. encoder transformer (`encoder.encoder_transformer.layers.*`, pre-LN,
//      LayerNorm+bias, gelu MLP, LayerScale) -> [512, L']
//   3. frame-rate-match downsample (replicate-padded causal conv, stride 2)
//      -> [512, T]
//   4. split-RVQ encode: project 512->256 (`input_proj`), then nearest-codebook
//      argmin + residual subtraction across 1 semantic + 15 acoustic codebooks.
//
// Causal MimiConv1d uses LEFT pad = `effective_kernel - stride` with output
// length `ceil(L / stride)` (zero pad on the right for `pad_mode="constant"`,
// replicate for the downsample). The argmin/residual loop runs on the HOST (no
// kernel exists; same policy as the speaker-pooling host code).
impl Codec {
    /// Encode a mono 24 kHz `wav` into `[T,16]` codec codes (row-major, q0
    /// semantic + q1..15 acoustic) — the inverse of [`Codec::decode`].
    pub fn encode(&self, wav: &[f32]) -> Vec<u32> {
        assert!(!wav.is_empty(), "empty wav");
        let e = self.cfg.enc.clone();
        let l0 = wav.len() as u32;

        // --- 1. SEANet conv encoder ---
        // head conv: 1 -> num_filters, k=kernel_size, stride 1.
        let x = self.upload(wav); // [1, L0] NCL (cin = 1)
        let (mut h, mut l) = self.enc_conv(&x, "encoder.encoder.layers.0", 1, e.num_filters, l0, e.kernel_size, 1, 1, 1);
        let mut dim = e.num_filters;
        // downsample stages: reversed(upsampling_ratios).
        let ratios: Vec<u32> = e.upsampling_ratios.iter().rev().copied().collect();
        let mut li = 1usize;
        for &r in &ratios {
            // residual block: ELU, conv(dim->dim/compress,k=res_k), ELU, conv(->dim,k1).
            let hidden = dim / e.compress;
            let base = format!("encoder.encoder.layers.{li}");
            let y = self.elu(&h, dim * l);
            let (y, _) = self.enc_conv(&y, &format!("{base}.block.1"), dim, hidden, l, e.residual_kernel_size, 1, 1, 1);
            let y = self.elu(&y, hidden * l);
            let (y, _) = self.enc_conv(&y, &format!("{base}.block.3"), hidden, dim, l, 1, 1, 1, 1);
            h = self.add2(&h, &y, dim * l);
            li += 1; // resnet
            li += 1; // ELU (no params)
            // downsample conv: dim -> 2*dim, k=2r, stride=r.
            let y = self.elu(&h, dim * l);
            let (hd, lo) = self.enc_conv(&y, &format!("encoder.encoder.layers.{li}"), dim, dim * 2, l, 2 * r, r, 1, 1);
            h = hd;
            l = lo;
            dim *= 2;
            li += 1; // downsample conv
        }
        // final: ELU, conv(dim -> hidden_size, k=last_kernel_size, stride 1).
        li += 1; // final ELU (no params)
        let y = self.elu(&h, dim * l);
        let (hd, lo) = self.enc_conv(&y, &format!("encoder.encoder.layers.{li}"), dim, e.hidden_size, l, e.last_kernel_size, 1, 1, 1);
        h = hd;
        l = lo;

        // --- 2. encoder transformer (token-major [L, hidden]) ---
        let mut tm = self.transpose(&h, e.hidden_size, l); // [L, 512]
        tm = self.enc_transformer(&tm, l);
        h = self.transpose(&tm, l, e.hidden_size); // [512, L]

        // --- 3. frame-rate-match downsample ---
        let (h, t) = self.enc_downsample(&h, e.hidden_size, l);

        // --- 4. split-RVQ encode ---
        self.rvq_encode(&h, t)
    }

    /// Causal Mimi `conv1d` (NCL): left pad `effective_kernel - stride`, output
    /// length `ceil(L / stride)`, then add the per-channel `{prefix}.conv.bias`.
    /// Returns `([Cout, lo], lo)`.
    fn enc_conv(&self, x: &DeviceBuffer, prefix: &str, cin: u32, cout: u32, l: u32, k: u32, stride: u32, dilation: u32, groups: u32) -> (DeviceBuffer, u32) {
        let keff = (k - 1) * dilation + 1;
        let pad = keff - stride;
        let lo = l.div_ceil(stride);
        let c = audio::conv::Conv1d { n: 1, cin, l, cout, k, stride, pad, dilation, groups, lo };
        let out = self.st((cout * lo) as usize);
        self.run(audio::conv::conv1d_fwd(
            &self.gpu,
            &audio::conv::ConvKernels { fwd: CONV1D, dx: 0, dw: 0 },
            &c,
            x,
            self.w(&format!("{prefix}.conv.weight")),
            &out,
        ));
        let out = self.add_ncl_bias(&out, &format!("{prefix}.conv.bias"), cout, lo);
        (out, lo)
    }

    /// The frame-rate-match downsample: a causal conv with **replicate** padding
    /// (`hidden -> hidden`, kernel `downsample_kernel`, stride `downsample_stride`,
    /// no bias). Left pad `k - stride` and right pad `extra = ceil(L/s)*s - L` are
    /// filled by edge replication on the host, then a `pad=0` conv. Returns
    /// `([C, T], T)`.
    fn enc_downsample(&self, x: &DeviceBuffer, c: u32, l: u32) -> (DeviceBuffer, u32) {
        let k = self.cfg.enc.downsample_kernel;
        let stride = self.cfg.enc.downsample_stride;
        let pad_left = k - stride;
        let lo = l.div_ceil(stride);
        let pad_right = lo * stride - l;
        let xv = self.gpu.read(x, (c * l) as usize);
        let lp = (pad_left + l + pad_right) as usize;
        let (lu, plu) = (l as usize, pad_left as usize);
        let mut padded = vec![0.0f32; c as usize * lp];
        for ch in 0..c as usize {
            let src = &xv[ch * lu..(ch + 1) * lu];
            let dst = &mut padded[ch * lp..(ch + 1) * lp];
            for d in dst.iter_mut().take(plu) {
                *d = src[0];
            }
            dst[plu..plu + lu].copy_from_slice(src);
            for d in dst[plu + lu..].iter_mut() {
                *d = src[lu - 1];
            }
        }
        let pbuf = self.upload(&padded);
        let conv = audio::conv::Conv1d { n: 1, cin: c, l: lp as u32, cout: c, k, stride, pad: 0, dilation: 1, groups: 1, lo };
        let out = self.st((c * lo) as usize);
        self.run(audio::conv::conv1d_fwd(
            &self.gpu,
            &audio::conv::ConvKernels { fwd: CONV1D, dx: 0, dw: 0 },
            &conv,
            &pbuf,
            self.w("encoder.downsample.conv.weight"),
            &out,
        ));
        (out, lo)
    }

    /// Mimi encoder transformer over token-major `[T, hidden]`: 8 pre-LN layers
    /// (LayerNorm **with bias**, RoPE GQA attention with scaling `1/√head_dim`,
    /// gelu MLP `fc2(gelu(fc1))`, per-channel LayerScale on each residual). No
    /// input/output projection and no final norm (unlike the decoder).
    fn enc_transformer(&self, x0: &DeviceBuffer, t: u32) -> DeviceBuffer {
        let e = self.cfg.enc.clone();
        let d = e.hidden_size;
        let hd = e.head_dim;
        let nh = e.num_attention_heads;
        let nkv = e.num_key_value_heads;
        let hq = nh * hd;
        let hkv = nkv * hd;
        let ff = e.intermediate_size;
        let theta = e.rope_theta;
        let ids = ids();
        let ga = Gqa { b: 1, t, n_heads: nh, n_kv_heads: nkv, head_dim: hd };

        let mut x = clone_buf(self, x0, t * d);
        for layer in 0..e.num_hidden_layers as usize {
            let p = |leaf: &str| format!("encoder.encoder_transformer.layers.{layer}.{leaf}");
            // --- attention ---
            let xn = self.host_ln(&x, t, d, &p("input_layernorm"), e.norm_eps);
            let q = self.matmul(&xn, &p("self_attn.q_proj.weight"), t, d, hq);
            let k = self.matmul(&xn, &p("self_attn.k_proj.weight"), t, d, hkv);
            let v = self.matmul(&xn, &p("self_attn.v_proj.weight"), t, d, hkv);
            self.run(block::rope_fwd(&self.gpu, &ids, &q, t, nh, hd, hq, t, theta));
            self.run(block::rope_fwd(&self.gpu, &ids, &k, t, nkv, hd, hkv, t, theta));
            let scores = self.st((nh * t * t) as usize);
            let probs = self.st((nh * t * t) as usize);
            let ctx = self.st((t * hq) as usize);
            self.gpu.submit(&[], &block::gqa_fwd(&self.gpu, &ids, &ga, &q, &k, &v, &scores, &probs, &ctx));
            let attn = self.matmul(&ctx, &p("self_attn.o_proj.weight"), t, hq, d);
            let attn = self.scale_chan(&attn, &p("self_attn_layer_scale.scale"), t * d, d, 1);
            x = self.add2(&x, &attn, t * d);
            // --- MLP: fc2(gelu(fc1(x))) ---
            let xn = self.host_ln(&x, t, d, &p("post_attention_layernorm"), e.norm_eps);
            let g = self.matmul(&xn, &p("mlp.fc1.weight"), t, d, ff);
            let mut gv = self.gpu.read(&g, (t * ff) as usize);
            par_gelu(&mut gv);
            let gbuf = self.upload(&gv);
            let mlp = self.matmul(&gbuf, &p("mlp.fc2.weight"), t, ff, d);
            let mlp = self.scale_chan(&mlp, &p("mlp_layer_scale.scale"), t * d, d, 1);
            x = self.add2(&x, &mlp, t * d);
        }
        x
    }

    /// Host `nn.LayerNorm` (affine γ/β, given eps) over the last axis of a
    /// token-major `[t, d]` device buffer; returns a fresh device buffer.
    fn host_ln(&self, x: &DeviceBuffer, t: u32, d: u32, prefix: &str, eps: f32) -> DeviceBuffer {
        let mut v = self.gpu.read(x, (t * d) as usize);
        host_layernorm(&mut v, t as usize, d as usize, &self.host[&format!("{prefix}.weight")], &self.host[&format!("{prefix}.bias")], eps);
        self.upload(&v)
    }

    /// Host ELU activation (α = 1): `x if x>0 else exp(x)-1`. `n` = element count.
    fn elu(&self, x: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let mut v = self.gpu.read(x, n as usize);
        for val in &mut v {
            if *val < 0.0 {
                *val = val.exp() - 1.0;
            }
        }
        self.upload(&v)
    }

    /// Split residual-vector-quantizer **encode**: project the `[hidden, T]`
    /// embeddings to `codebook_dim` (semantic & acoustic `input_proj`), then run
    /// the nearest-codebook argmin + residual loop on the host. Returns `[T,16]`
    /// row-major codes (q0 semantic, q1..15 acoustic).
    fn rvq_encode(&self, emb_ncl: &DeviceBuffer, t: u32) -> Vec<u32> {
        let e = self.cfg.enc.clone();
        let (h, cd) = (e.hidden_size, e.codebook_dim);
        let cs = e.codebook_size as usize;
        let n_sem = e.num_semantic_quantizers as usize;
        let nq = self.cfg.num_quantizers as usize;
        let n_aco = nq - n_sem;
        let emb_tm = self.transpose(emb_ncl, h, t); // [T, hidden]

        let sem_p = self.matmul(&emb_tm, "encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight", t, h, cd);
        let aco_p = self.matmul(&emb_tm, "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight", t, h, cd);
        let mut sem = self.gpu.read(&sem_p, (t * cd) as usize);
        let mut aco = self.gpu.read(&aco_p, (t * cd) as usize);

        let (cd, tt) = (cd as usize, t as usize);
        let mut codes = vec![0u32; tt * nq];
        let mut encode_group = |buf: &mut [f32], group: &str, layers: usize, col0: usize| {
            for q in 0..layers {
                let table = self.gpu.read(self.w(&format!("encoder.quantizer.{group}_residual_vector_quantizer.layers.{q}.table")), cs * cd);
                for ti in 0..tt {
                    let v = &buf[ti * cd..(ti + 1) * cd];
                    let idx = nearest(v, &table, cs, cd);
                    codes[ti * nq + col0 + q] = idx as u32;
                    let row = &table[idx * cd..(idx + 1) * cd];
                    for c in 0..cd {
                        buf[ti * cd + c] -= row[c];
                    }
                }
            }
        };
        encode_group(&mut sem, "semantic", n_sem, 0);
        encode_group(&mut aco, "acoustic", n_aco, n_sem);
        codes
    }
}

/// Nearest codebook row (argmin squared-Euclidean) to `v` over a `[bins, dim]`
/// flat table — `MimiEuclideanCodebook.quantize` (cdist p=2 + argmin).
fn nearest(v: &[f32], table: &[f32], bins: usize, dim: usize) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for b in 0..bins {
        let row = &table[b * dim..(b + 1) * dim];
        let mut d = 0.0f32;
        for c in 0..dim {
            let diff = v[c] - row[c];
            d += diff * diff;
        }
        if d < best_d {
            best_d = d;
            best = b;
        }
    }
    best
}

fn clone_buf(c: &Codec, x: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let v = c.gpu.read(x, n as usize);
    c.upload(&v)
}

/// Kernel-id map for the shared `model::block` builders (only forward ids used).
fn ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: 0,
        rmsnorm_dx: 0,
        rmsnorm_dw: 0,
        rope: ROPE,
        rope_bwd: 0,
        gqa_scores: GQA_SCORES,
        gqa_apply: GQA_APPLY,
        attn_softmax: ATTN_SOFTMAX,
        gqa_dscores: 0,
        gqa_dv: 0,
        gqa_dq: 0,
        gqa_dk: 0,
        silu_mul: SILU_MUL,
        silu_da: 0,
        silu_db: 0,
    }
}

/// `nn.LayerNorm` over the last axis (C) of a row-major `[rows, C]` buffer,
/// population variance, eps 1e-6, with affine γ/β. In place. Rows are
/// independent, so they fan out across cores for large buffers (the per-row sum
/// order is unchanged — bit-identical to the serial path).
fn host_layernorm(x: &mut [f32], rows: usize, c: usize, gamma: &[f32], beta: &[f32], eps: f32) {
    let ln_row = |row: &mut [f32]| {
        let mean = row.iter().sum::<f32>() / c as f32;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..c {
            row[j] = (row[j] - mean) * inv * gamma[j] + beta[j];
        }
    };
    if rows * c >= PAR_MIN {
        par::rows_mut(x, c, |_, row| ln_row(row));
    } else {
        x.chunks_mut(c).for_each(ln_row);
    }
}

/// Exact (erf) GELU over a host buffer, in place; element-independent, so it
/// parallelizes for large buffers (the ConvNeXt pwconv1 activation, `[L, 4C]`).
fn par_gelu(buf: &mut [f32]) {
    if buf.len() >= PAR_MIN {
        par::each_mut(buf, |_, v| *v = gelu_exact(*v));
    } else {
        for v in buf.iter_mut() {
            *v = gelu_exact(*v);
        }
    }
}

/// Exact (erf) GELU, matching `nn.GELU()` (default `approximate='none'`).
fn gelu_exact(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// erf via the Numerical-Recipes `erfc` rational approximation (|err| < 1.2e-7).
fn erf(x: f32) -> f32 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let erfc = t
        * (-z * z - 1.265_512_2
            + t * (1.000_023_7
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.135_204
                                    + t * (1.488_515_9 + t * (-0.82215223 + t * 0.17087277)))))))))
        .exp();
    let e = 1.0 - erfc;
    if x >= 0.0 {
        e
    } else {
        -e
    }
}
