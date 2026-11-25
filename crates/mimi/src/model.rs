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
//! The layout flips are device permutes (`nchw_nlc` / `nlc_nchw`).
//!
//! The whole decode is a device pass ([`Pass`]): the steps are recorded and
//! submitted in a few batches, the convs take the GEMM lowering with the bias
//! fused (`audio::conv::conv1d_bias_fwd` / `convtr1d_bias_fwd`), ConvNeXt's
//! `LayerNorm` (eps 1e-6) and exact-erf `GELU` are device kernels, and the
//! only host transfer is the final waveform read (plus the clamp to [-1, 1]
//! on it). The encoder ([`Codec::encode_batch`]) is a device pass over a batch
//! of clips the same way, reading back only the codes.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use audio::conv::{ConvGemmKernels, ConvKernels, ConvScratch};
use bytemuck::cast_slice;
use gpu_core::select::{DefaultSelector, Dtype, KernelSelector, KernelVariant, Op, OpShape, VQ_TILE_ROWS};
use gpu_core::{f, BufUsage, DeviceBuffer, Gpu, Step};
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
// The GEMM conv lowering (`audio::conv::ConvGemmKernels`) and the layout/norm
// kernels the device decode pass uses in place of host round trips.
const ADD_CHAN_INPLACE: usize = 16;
const IM2COL1D_AT: usize = 17;
const MATMUL_REG3: usize = 18;
const MATMUL_DX_REG: usize = 19;
const MATMUL_DW_REG_SPLITK: usize = 20;
const NLC_BIAS_NCHW: usize = 21;
const COL2IM1D_BIAS: usize = 22;
const NCHW_NLC: usize = 23;
const NLC_NCHW: usize = 24;
const LAYERNORM: usize = 25;
const GELU_ERF: usize = 26;
// 27 is `nchw_nlc_at`, resolved by name by `audio::conv`.
// The batched encoder pass: ELU, the per-clip length mask and replicate pad,
// and the residual quantizer on the device.
const ELU: usize = 28;
const NCL_LEN_MASK: usize = 29;
const PAD1D_EDGE_LENS: usize = 30;
const VQ_ARGMIN: usize = 31;
const VQ_RESIDUAL: usize = 32;
const VQ_ARGMIN_TILED: usize = 33;

/// The kernel set this decoder dispatches. `pub` so a caller building the
/// codec on its own device handle (`Codec::load_inference_on`) can construct
/// that handle with the right pipelines - the indices above are this module's
/// contract with it.
pub const PIPELINES: &[(&str, &str)] = &[
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
    ("add_chan_inplace", kernels::ADD_CHAN_INPLACE),
    ("im2col1d_at", kernels::IM2COL1D_AT),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg_splitk", kernels::MATMUL_DW_REG_SPLITK),
    ("nlc_bias_nchw", kernels::NLC_BIAS_NCHW),
    ("col2im1d_bias", kernels::COL2IM1D_BIAS),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("layernorm", kernels::LAYERNORM),
    ("gelu_erf", kernels::GELU_ERF),
    // The tap-sum conv lowering places its input channels-last behind its padding (`audio::conv`).
    ("nchw_nlc_at", kernels::NCHW_NLC_AT),
    // The batched encoder pass.
    ("elu", kernels::ELU),
    ("ncl_len_mask", kernels::NCL_LEN_MASK),
    ("pad1d_edge_lens", kernels::PAD1D_EDGE_LENS),
    ("vq_argmin", kernels::VQ_ARGMIN),
    ("vq_residual", kernels::VQ_RESIDUAL),
    ("vq_argmin_tiled", kernels::VQ_ARGMIN_TILED),
];

/// SnakeBeta's `no_div_by_zero` (the reference's fixed epsilon).
const SNAKE_EPS: f32 = 1e-9;
/// ConvNeXt LayerNorm epsilon.
const LN_EPS: f32 = 1e-6;
/// Clamp decoded samples to the waveform's [-1, 1] on the calling thread. A clip is a few hundred thousand samples,
/// well under a millisecond of memory-bound work; handed to the shared worker pool instead, a served decode waited
/// about 14 ms a clip behind the other requests' host work for the pool to take it.
fn clamp_samples(wav: &mut [f32]) {
    for s in wav {
        *s = s.clamp(-1.0, 1.0);
    }
}

/// The codec model: frozen weights on the device.
pub struct Codec {
    pub gpu: Gpu,
    pub cfg: CodecConfig,
    ps: ParamStore,
    /// The vendor GEMM the codec's products take where it is active, in binary32 (the codec is stored and gated in
    /// fp32).
    gemm: gpu_core::provider::library::LibraryGemmProvider,
}

impl Codec {
    /// Load an inference-only decoder from a brain checkpoint produced by
    /// [`crate::import::import`]. Every parameter is frozen (weights only).
    ///
    /// Builds on the **CPU backend**, which is this constructor's whole
    /// contract, not an accident - see [`Self::from_weights`]. A caller that
    /// wants the codec on the ambient `--device` selection asks for it
    /// explicitly through [`Self::load_inference_on`].
    pub fn load_inference(weights_path: &str) -> Codec {
        Self::load_inference_on(Gpu::new_cpu(PIPELINES), weights_path)
    }

    /// [`Self::load_inference`] on an existing device handle (see
    /// `gpu_core::Gpu::share`), so a process holds ONE device however many
    /// components it loads, and so a caller can put the codec on the same
    /// device the rest of its pipeline runs on.
    pub fn load_inference_on(gpu: Gpu, weights_path: &str) -> Codec {
        let c = crate::import::load(weights_path);
        let cfg = CodecConfig::from_json(&c.header["config"]);
        let init = c.by_role("");
        Codec::from_weights_on(gpu, cfg, init)
    }

    /// Build from an in-memory weight map (used by tests and
    /// [`load_inference`](Self::load_inference)), **on the CPU backend**.
    ///
    /// This constructor is CPU-pinned deliberately and every existing caller
    /// depends on that: it is the one every real-weight parity test in this
    /// crate and in `qwen3omnimoe` is gated against. [`Self::from_weights_on`]
    /// is the device-taking form.
    pub fn from_weights(cfg: CodecConfig, init: HashMap<String, Vec<f32>>) -> Codec {
        Self::from_weights_on(Gpu::new_cpu(PIPELINES), cfg, init)
    }

    /// [`Self::from_weights`] on an existing device handle.
    pub fn from_weights_on(gpu: Gpu, cfg: CodecConfig, mut init: HashMap<String, Vec<f32>>) -> Codec {
        // Each encoder codebook also transposed (`[dim, bins]`), for the
        // row-tiled assignment (`vq_argmin_tiled`) that reads it coalesced.
        let dim = cfg.enc.codebook_dim as usize;
        let tables: Vec<String> = init.keys().filter(|n| n.starts_with("encoder.quantizer.") && n.ends_with(".table")).cloned().collect();
        for name in tables {
            let t = &init[&name];
            let bins = t.len() / dim;
            let transposed: Vec<f32> = (0..dim * bins).map(|j| t[(j % bins) * dim + j / bins]).collect();
            init.insert(transposed_name(&name), transposed);
        }
        let roles: Vec<(String, usize, Role)> =
            init.iter().map(|(n, v)| (n.clone(), v.len(), Role::Frozen)).collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, &init);
        let gemm = gpu_core::provider::library::LibraryGemmProvider::for_gpu(&gpu, gpu_core::provider::library::GemmPrecision::Auto);
        Codec { gpu, cfg, ps, gemm }
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
        let p = Pass::new(self);
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
            let out = p.st((t * dim) as usize);
            p.run(self.gpu.step(EMBED, &[&codes_buf, p.w(table), &out], &[dim, t], t * dim));
            out
        };
        // semantic group (rvq_first): 1 codebook.
        let sem = gather("quantizer.rvq_first.vq.layers.0.table", 0);
        let first = p.matmul(&sem, "quantizer.rvq_first.output_proj.weight", t, dim, lat);
        // acoustic group (rvq_rest): codebooks q1..15 summed, then projected.
        let mut acc = gather("quantizer.rvq_rest.vq.layers.0.table", 1);
        for i in 1..(nq - 1) {
            let g = gather(&format!("quantizer.rvq_rest.vq.layers.{i}.table"), i + 1);
            acc = p.add2(&acc, &g, t * dim);
        }
        let rest = p.matmul(&acc, "quantizer.rvq_rest.output_proj.weight", t, dim, lat);
        let quant_tm = p.add2(&first, &rest, t * lat); // [T, 512] token-major
        let quant = p.to_channel_major(&quant_tm, 1, t, lat); // [512, T] NCL

        // --- 2. pre_conv (512 -> 1024, k3 causal) ---
        let pre = p.causal_conv(&quant, "pre_conv", lat, latent, t, 3, 1, 1); // [1024, T]
        let mut x = p.to_token_major(&pre, 1, latent, t); // [T, 1024] token-major

        // --- 3. pre_transformer ---
        x = p.matmul(&x, "pre_transformer.input_proj.weight", t, latent, hidden);
        p.bias_add(&x, "pre_transformer.input_proj.bias", t, hidden);
        x = p.transformer(&x, t);
        x = p.matmul(&x, "pre_transformer.output_proj.weight", t, hidden, latent);
        p.bias_add(&x, "pre_transformer.output_proj.bias", t, latent);
        let mut h = p.to_channel_major(&x, 1, t, latent); // [1024, L] NCL
        let mut l = t;

        // --- 4. upsample stages ---
        for (u, &factor) in self.cfg.upsampling_ratios.clone().iter().enumerate() {
            h = p.causal_convtr(&h, &format!("upsample.{u}.0"), latent, latent, l, factor, factor);
            l *= factor;
            h = p.convnext(&h, &format!("upsample.{u}.1"), latent, l);
        }

        // --- 5. SEANet decoder ---
        let dec_dim = self.cfg.decoder_dim; // 1536
        h = p.causal_conv(&h, "decoder.0", latent, dec_dim, l, 7, 1, 1); // [1536, L]
        let rates = self.cfg.upsample_rates.clone();
        for (i, &rate) in rates.iter().enumerate() {
            let in_dim = dec_dim >> i;
            let out_dim = dec_dim >> (i + 1);
            let bp = format!("decoder.{}", i + 1);
            // block.0 SnakeBeta(in_dim)
            h = p.snake(&h, &format!("{bp}.block.0"), in_dim, l);
            // block.1 causal transposed conv (in->out, k=2·rate, stride=rate)
            h = p.causal_convtr(&h, &format!("{bp}.block.1"), in_dim, out_dim, l, 2 * rate, rate);
            l *= rate;
            // block.2/3/4 dilated residual units (dilation 1, 3, 9)
            for (j, dil) in [(2u32, 1u32), (3, 3), (4, 9)] {
                h = p.residual_unit(&h, &format!("{bp}.block.{j}"), out_dim, l, dil);
            }
        }
        let out_dim = dec_dim >> rates.len(); // 96
        h = p.snake(&h, "decoder.5", out_dim, l); // tail SnakeBeta
        h = p.causal_conv(&h, "decoder.6", out_dim, 1, l, 7, 1, 1); // -> [1, L]

        // --- 6. clamp ---
        let mut wav = p.read(&h, l as usize);
        clamp_samples(&mut wav);
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
        let p = Pass::new(self);
        let hidden = self.cfg.hidden_size; // 1024

        // --- 1. code_embedding(codes + code_offset).mean(1) ---
        let x = p.code_embedding_mean(codes, t, nq, hidden);

        // --- 2. pre_transformer (no input_proj/output_proj: hidden_size IS the
        //        working width already) ---
        let x = p.transformer(&x, t);
        let mut h = p.to_channel_major(&x, 1, t, hidden); // [hidden, T] NCL
        let mut l = t;

        // --- 3. upsample stages (K == stride here, so the crop conventions
        //        agree -- causal_convtr reused unchanged) ---
        for (u, &factor) in self.cfg.upsampling_ratios.clone().iter().enumerate() {
            h = p.causal_convtr(&h, &format!("upsample.{u}.0"), hidden, hidden, l, factor, factor);
            l *= factor;
            h = p.convnext(&h, &format!("upsample.{u}.1"), hidden, l);
        }

        // --- 4. SEANet decoder (symmetric-crop transposed conv, see doc) ---
        let dec_dim = self.cfg.decoder_dim; // 1536
        h = p.causal_conv(&h, "decoder.0", hidden, dec_dim, l, 7, 1, 1);
        let rates = self.cfg.upsample_rates.clone();
        for (i, &rate) in rates.iter().enumerate() {
            let in_dim = dec_dim >> i;
            let out_dim = dec_dim >> (i + 1);
            let bp = format!("decoder.{}", i + 1);
            h = p.snake(&h, &format!("{bp}.block.0"), in_dim, l);
            let new_l = (l - 1) * rate; // symmetric crop: Lo = (L-1)*stride, see doc
            h = p.causal_convtr_sym(&h, &format!("{bp}.block.1"), in_dim, out_dim, l, 2 * rate, rate, rate, new_l);
            l = new_l;
            for (j, dil) in [(2u32, 1u32), (3, 3), (4, 9)] {
                h = p.residual_unit(&h, &format!("{bp}.block.{j}"), out_dim, l, dil);
            }
        }
        let out_dim = dec_dim >> rates.len();
        h = p.snake(&h, "decoder.5", out_dim, l);
        h = p.causal_conv(&h, "decoder.6", out_dim, 1, l, 7, 1, 1);

        // --- 5. clamp ---
        let mut wav = p.read(&h, l as usize);
        clamp_samples(&mut wav);
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
        self.decode_omni_chunked_until(codes, chunk_frames, |chunk| {
            on_chunk(chunk);
            true
        });
    }

    /// [`Self::decode_omni_chunked`], where `on_chunk` returns `false` to stop
    /// decoding (the caller was cancelled): no further chunk is computed.
    /// Returns whether every chunk was decoded.
    pub fn decode_omni_chunked_until(&self, codes: &[u32], chunk_frames: usize, mut on_chunk: impl FnMut(&[f32]) -> bool) -> bool {
        let nq = self.cfg.num_quantizers as usize;
        assert_eq!(codes.len() % nq, 0, "codes length not a multiple of {nq}");
        let t = (codes.len() / nq) as u32;
        assert!(t > 0, "empty codes");
        let p = Pass::new(self);
        let hidden = self.cfg.hidden_size;

        // --- front: identical to decode_omni's own (correctly
        // sliding-window-attended), run ONCE over the whole
        // sequence -- see this fn's own doc for why that's not a chunking
        // cost. ---
        let x = p.code_embedding_mean(codes, t, nq, hidden);
        let x = p.transformer(&x, t);
        let h = p.to_channel_major(&x, 1, t, hidden); // [hidden, T] NCL
        let latent: Vec<f32> = p.read(&h, (hidden * t) as usize);

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
            if !out.is_empty() && !on_chunk(&out) {
                return false;
            }
            a = b;
        }
        true
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

}

// ======================================================================
// THE DEVICE PASS: every graph builder runs on a `Pass`
// ======================================================================

/// Fresh activation bytes a recorded pass lets pile up before it submits what
/// it has recorded so far.
///
/// Recording everything and submitting once would keep every intermediate of
/// the whole graph alive until the end: at 600 frames the SEANet stages alone
/// are tens of GB. Submitting whenever this much has been allocated keeps a
/// long clip's peak bounded - the buffers a submitted batch no longer needs go
/// back to the backend's held-block pool for the next stage - while a typical
/// clip (about 100 frames, 8 s) still goes out in a handful of submissions
/// rather than one per op.
const FLUSH_BYTES: u64 = 1 << 30;

/// Freed device blocks the decode pass asks the backend to keep for reuse, so
/// that a repeated decode does not return them to the driver and take them
/// again: a driver free costs milliseconds per block.
const RETAIN_BYTES: u64 = 4 << 30;

/// The device ops of one decode (or encode) call: every helper that builds a
/// graph step goes through [`Pass::run`], which records it for a later batched
/// submit.
///
/// Dereferences to the [`Codec`] it runs on, so the weights, config and device
/// handle read the same as on the codec itself.
struct Pass<'a> {
    codec: &'a Codec,
    steps: RefCell<Vec<Step>>,
    /// Bytes allocated for activations since the last submit.
    pending_bytes: Cell<u64>,
    /// Shared by every lowered conv of the pass: the convs run in submit order,
    /// so one `im2col`/GEMM scratch serves all of them (see [`ConvScratch`]).
    scratch: RefCell<ConvScratch>,
}

impl std::ops::Deref for Pass<'_> {
    type Target = Codec;
    fn deref(&self) -> &Codec {
        self.codec
    }
}

/// The conv kernel set for a conv whose direct fallback is `direct_fwd`
/// ([`CONV1D`] or [`CONVTR1D`]); the lowering's own pipelines are the same
/// for both.
fn conv_kernels(direct_fwd: usize) -> ConvGemmKernels {
    ConvGemmKernels {
        direct: ConvKernels { fwd: direct_fwd, dx: 0, dw: 0 },
        bias: ADD_CHAN_INPLACE,
        im2col: IM2COL1D_AT,
        matmul: MATMUL_REG3,
        matmul_nn: MATMUL_DX_REG,
        matmul_tn: MATMUL_DW_REG_SPLITK,
        nlc_bias: NLC_BIAS_NCHW,
        col2im: COL2IM1D_BIAS,
    }
}

impl<'a> Pass<'a> {
    /// A pass over `codec`: its products take the vendor GEMM where it is active (the decode's and the encoder's
    /// alike; the encoder's codes are pinned against the CPU backend on it).
    fn new(codec: &'a Codec) -> Pass<'a> {
        codec.gpu.hold_freed_blocks(true);
        codec.gpu.retain_freed_blocks(RETAIN_BYTES);
        let scratch = ConvScratch::new().with_gemm(codec.gemm.clone());
        Pass { codec, steps: RefCell::new(Vec::new()), pending_bytes: Cell::new(0), scratch: RefCell::new(scratch) }
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }
    /// A zeroed device buffer for `n` floats.
    fn st(&self, n: usize) -> DeviceBuffer {
        self.pending_bytes.set(self.pending_bytes.get() + n as u64 * 4);
        self.gpu.storage(n as u64)
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
    /// Submit everything recorded so far, in order.
    fn flush(&self) {
        let steps = std::mem::take(&mut *self.steps.borrow_mut());
        if !steps.is_empty() {
            self.gpu.submit(&[], &steps);
        }
        self.pending_bytes.set(0);
    }
    /// Read `n` floats of `buf` back to the host, after everything recorded.
    fn read(&self, buf: &DeviceBuffer, n: usize) -> Vec<f32> {
        self.flush();
        self.gpu.read(buf, n)
    }
    fn upload(&self, data: &[f32]) -> DeviceBuffer {
        let b = self.gpu.buffer("act", (data.len() * 4) as u64, BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC);
        self.gpu.write(&b, cast_slice(data));
        b
    }

    /// `out = x[M,K] @ W[N,K]ᵀ` (PyTorch `nn.Linear`, no bias).
    fn matmul(&self, x: &DeviceBuffer, wname: &str, m: u32, k: u32, n: u32) -> DeviceBuffer {
        use gpu_core::provider::library::Operand;
        let out = self.st((m * n) as usize);
        let mut steps = Vec::new();
        let on_library = self.codec.gemm.forward_f32(&self.gpu, &mut steps, &Operand { buf: x, range: (0, 0) }, &Operand { buf: self.w(wname), range: (0, 0) }, &Operand { buf: &out, range: (0, 0) }, m, n, k);
        if on_library {
            self.run_all(steps);
        } else {
            self.run(self.gpu.step(MATMUL, &[x, self.w(wname), &out], &[m, k, n], m * n));
        }
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
    /// Channel-major `[n, c, l]` (NCL) to token-major `[n, l, c]`, on the device.
    fn to_token_major(&self, x: &DeviceBuffer, n: u32, c: u32, l: u32) -> DeviceBuffer {
        let out = self.st((n * c * l) as usize);
        self.run(self.gpu.step(NCHW_NLC, &[x, &out], &[n * c * l, c, l], n * c * l));
        out
    }
    /// Token-major `[n, l, c]` to channel-major `[n, c, l]` (NCL), on the device.
    fn to_channel_major(&self, x: &DeviceBuffer, n: u32, l: u32, c: u32) -> DeviceBuffer {
        let out = self.st((n * c * l) as usize);
        self.run(self.gpu.step(NLC_NCHW, &[x, &out], &[n * c * l, c, l], n * c * l));
        out
    }
    /// `nn.LayerNorm` (affine, `eps`) over the last axis of a token-major
    /// `[rows, d]` buffer, with the `{prefix}.weight` / `{prefix}.bias` tensors.
    fn layernorm(&self, x: &DeviceBuffer, prefix: &str, rows: u32, d: u32, eps: f32) -> DeviceBuffer {
        let out = self.st((rows * d) as usize);
        let (kind, grid) = block::ln_variant(&self.gpu, LAYERNORM, None, rows, d);
        let (gamma, beta) = (format!("{prefix}.weight"), format!("{prefix}.bias"));
        self.run(self.gpu.dispatch(kind, &[x, self.w(&gamma), self.w(&beta), &out], &[d, rows, f(eps)], grid));
        out
    }
    /// Exact (erf) GELU, `nn.GELU()`, over `n` floats.
    fn gelu(&self, x: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let out = self.st(n as usize);
        self.run(self.gpu.step(GELU_ERF, &[x, &out], &[n], n));
        out
    }

    /// Causal `conv1d` in NCL: left pad `dilation*(K-1)`, stride 1, `Lo == L`,
    /// bias `{prefix}.conv.bias` fused. Returns `[Cout, L]`.
    #[allow(clippy::too_many_arguments)]
    fn causal_conv(&self, x: &DeviceBuffer, prefix: &str, cin: u32, cout: u32, l: u32, k: u32, dilation: u32, groups: u32) -> DeviceBuffer {
        let pad = dilation * (k - 1);
        let c = audio::conv::Conv1d { n: 1, cin, l, cout, k, stride: 1, pad, dilation, groups, lo: l };
        self.conv(&c, x, prefix)
    }

    /// `conv1d` of `x` by `{prefix}.conv.{weight,bias}` into a fresh
    /// `[N, Cout, Lo]` buffer: the GEMM lowering where the device selector picks
    /// it, the direct kernel otherwise, the bias fused either way.
    fn conv(&self, c: &audio::conv::Conv1d, x: &DeviceBuffer, prefix: &str) -> DeviceBuffer {
        let out = self.st((c.n * c.cout * c.lo) as usize);
        let steps = audio::conv::conv1d_bias_fwd(
            &self.gpu,
            &conv_kernels(CONV1D),
            c,
            x,
            self.w(&format!("{prefix}.conv.weight")),
            self.w(&format!("{prefix}.conv.bias")),
            &out,
            &mut self.scratch.borrow_mut(),
        );
        self.run_all(steps);
        out
    }

    /// Causal transposed conv in NCL (upsampling by `stride`). The reference crops
    /// `K-stride` samples off the right; cropping the right == keeping the first
    /// `L·stride` outputs, so we just request `Lo = L·stride` (pad 0). Returns
    /// `[Cout, L·stride]`.
    #[allow(clippy::too_many_arguments)]
    fn causal_convtr(&self, x: &DeviceBuffer, prefix: &str, cin: u32, cout: u32, l: u32, k: u32, stride: u32) -> DeviceBuffer {
        self.causal_convtr_sym(x, prefix, cin, cout, l, k, stride, 0, l * stride)
    }

    /// Causal transposed conv (NCL) with an EXPLICIT symmetric crop `pad` and
    /// output length `lo` -- unlike [`Self::causal_convtr`] (which always
    /// assumes `pad = 0`, `lo = l*stride`), this is `Qwen3OmniMoeCausalTransConvNet`'s
    /// own convention: PyTorch `ConvTranspose1d(padding=pad)` crops `pad`
    /// samples off BOTH sides of the native output
    /// (`convtr1d.wgsl`'s `pad` param already matches this -- see
    /// [`Codec::decode_omni`]'s doc for why a separate entry point, not a change
    /// to [`Self::causal_convtr`], which stays correct for the standalone
    /// Qwen3-TTS codec's own (right-only-crop) convention).
    #[allow(clippy::too_many_arguments)]
    fn causal_convtr_sym(&self, x: &DeviceBuffer, prefix: &str, cin: u32, cout: u32, l: u32, k: u32, stride: u32, pad: u32, lo: u32) -> DeviceBuffer {
        let c = audio::conv::Conv1d { n: 1, cin, l, cout, k, stride, pad, dilation: 1, groups: 1, lo };
        let out = self.st((cout * lo) as usize);
        let steps = audio::conv::convtr1d_bias_fwd(
            &self.gpu,
            &conv_kernels(CONVTR1D),
            &c,
            x,
            self.w(&format!("{prefix}.conv.weight")),
            self.w(&format!("{prefix}.conv.bias")),
            &out,
            &mut self.scratch.borrow_mut(),
        );
        self.run_all(steps);
        out
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
    /// -> GELU -> pwconv2 -> γ scale -> residual, all on the device. The LayerNorm
    /// takes the reference's eps (1e-6, not the kernel's torch-default 1e-5) as
    /// its parameter and the GELU is the exact-erf kernel (`nn.GELU()`, not the
    /// tanh approximation).
    fn convnext(&self, x: &DeviceBuffer, prefix: &str, c: u32, l: u32) -> DeviceBuffer {
        // depthwise causal conv (groups = C, k7)
        let dw = self.causal_conv(x, &format!("{prefix}.dwconv"), c, c, l, 7, 1, c); // [C, L]
        let tm = self.to_token_major(&dw, 1, c, l); // [L, C]
        let normed = self.layernorm(&tm, &format!("{prefix}.norm"), l, c, LN_EPS);
        // pwconv1 (C -> 4C) + bias, GELU
        let hid = c * 4;
        let g = self.matmul(&normed, &format!("{prefix}.pwconv1.weight"), l, c, hid);
        self.bias_add(&g, &format!("{prefix}.pwconv1.bias"), l, hid);
        let g = self.gelu(&g, l * hid);
        // pwconv2 (4C -> C) + bias
        let o = self.matmul(&g, &format!("{prefix}.pwconv2.weight"), l, hid, c);
        self.bias_add(&o, &format!("{prefix}.pwconv2.bias"), l, c);
        // γ per-channel scale over token-major [L,C]: channel = idx % C (inner=1).
        let o = self.scale_chan(&o, &format!("{prefix}.gamma"), l * c, c, 1);
        let o_ncl = self.to_channel_major(&o, 1, l, c); // [C, L]
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

        let mut x = x0.clone();
        for layer in 0..c.num_hidden_layers as usize {
            let p = |leaf: &str| format!("pre_transformer.layers.{layer}.{leaf}");
            // --- attention ---
            let xn = self.st((t * d) as usize);
            self.run(block::rmsnorm_fwd(&self.gpu, &ids, &x, self.w(&p("input_layernorm.weight")), &xn, d, t, c.rms_norm_eps));
            let q = self.matmul(&xn, &p("self_attn.q_proj.weight"), t, d, hq);
            let k = self.matmul(&xn, &p("self_attn.k_proj.weight"), t, d, hkv);
            let v = self.matmul(&xn, &p("self_attn.v_proj.weight"), t, d, hkv);
            self.run(block::rope_fwd(&self.gpu, &ids, &q, t, nh, hd, hq, t, theta));
            self.run(block::rope_fwd(&self.gpu, &ids, &k, t, nkv, hd, hkv, t, theta));
            let scores = self.st((nh * t * t) as usize);
            let probs = self.st((nh * t * t) as usize);
            let ctx = self.st((t * hq) as usize);
            // Sliding-window causal attention, as the reference's
            // `create_sliding_window_causal_mask(sliding_window=config.
            // sliding_window)`: key `j` is masked out of query `i` once
            // `i - j >= sliding_window`, on every forward call (value 72 for
            // both released checkpoints). `decode` and `decode_omni` share this
            // body; `gqa_fwd_win` equals plain causal attention when the window
            // is at least `t`.
            //
            // `0` means "unset" (a hand-built `CodecConfig::default()`; the
            // parser defaults the field to 72) and is normalized to `t`
            // (unbounded): a literal 0 would mask every key and turn the
            // softmax uniform over the whole row, future keys included.
            // `decode_stream::front` applies the same normalization.
            let window = if self.cfg.sliding_window == 0 { t } else { self.cfg.sliding_window };
            self.run_all(block::gqa_fwd_win(&self.gpu, GQA_SCORES_WIN, &ids, &ga, window, &q, &k, &v, &scores, &probs, &ctx));
            let attn = self.matmul(&ctx, &p("self_attn.o_proj.weight"), t, hq, d);
            let attn = self.scale_chan(&attn, &p("self_attn_layer_scale.scale"), t * d, d, 1);
            x = self.add2(&x, &attn, t * d);
            // --- MLP ---
            let xn = self.st((t * d) as usize);
            self.run(block::rmsnorm_fwd(&self.gpu, &ids, &x, self.w(&p("post_attention_layernorm.weight")), &xn, d, t, c.rms_norm_eps));
            let gate = self.matmul(&xn, &p("mlp.gate_proj.weight"), t, d, ff);
            let up = self.matmul(&xn, &p("mlp.up_proj.weight"), t, d, ff);
            let hmid = self.st((t * ff) as usize);
            self.run(block::swiglu_fwd(&self.gpu, &ids, &gate, &up, &hmid, t * ff));
            let mlp = self.matmul(&hmid, &p("mlp.down_proj.weight"), t, ff, d);
            let mlp = self.scale_chan(&mlp, &p("mlp_layer_scale.scale"), t * d, d, 1);
            x = self.add2(&x, &mlp, t * d);
        }
        let out = self.st((t * d) as usize);
        self.run(block::rmsnorm_fwd(&self.gpu, &ids, &x, self.w("pre_transformer.norm.weight"), &out, d, t, c.rms_norm_eps));
        out
    }
}

// ======================================================================
// ENCODE: wav -> codes [T,16]  (HuggingFace Mimi encoder, forward only)
// ======================================================================
//
// Mirror of `MimiModel._encode_frame`, over a BATCH of clips in one recorded
// device pass:
//   1. SEANet conv encoder (`encoder.encoder.layers.*`) -> [n, 512, L']
//   2. encoder transformer (`encoder.encoder_transformer.layers.*`, pre-LN,
//      LayerNorm+bias, gelu MLP, LayerScale), plain causal -> [n, 512, L']
//   3. frame-rate-match downsample (replicate-padded causal conv, stride 2)
//      -> [n, 512, T]
//   4. split-RVQ encode: project 512->256 (`input_proj`), then nearest-codebook
//      argmin + residual subtraction across 1 semantic + 15 acoustic codebooks,
//      on the device (`vq_argmin` + `vq_residual`); only the codes read back.
//
// Causal MimiConv1d uses LEFT pad = `effective_kernel - stride` with output
// length `ceil(L / stride)` (zero pad on the right for `pad_mode="constant"`,
// replicate for the downsample).
//
// The clips are padded to one length, a multiple of the encoder's total stride
// (one code frame of samples), and carry their own lengths through every
// stage. Every conv is causal, so a padded position never reaches an earlier
// valid one, except where a strided conv's last window reads past the clip's
// end: the input of each strided conv is zeroed past the clip's own length
// first (`ncl_len_mask`), which is the zero padding the unbatched conv reads,
// and the downsample replicates each clip's own last frame (`pad1d_edge_lens`).
// So each clip's codes are the codes it encodes to alone.

/// Padded input samples one encode pass takes at most; a larger batch is
/// split into several passes (clips grouped by length, so little is padded).
///
/// The SEANet head runs `num_filters` (64) channels at the input rate, so a
/// pass's first activations are `64 x` this many floats (2 GiB at 2^23) with a
/// few of them alive at once: a bound on device memory, and it keeps every
/// dispatch's element count far inside `u32`.
pub const ENCODE_PASS_SAMPLES: usize = 1 << 23;

impl Codec {
    /// Encode a mono 24 kHz `wav` into `[T,16]` codec codes (row-major, q0
    /// semantic + q1..15 acoustic) - the inverse of [`Codec::decode`].
    pub fn encode(&self, wav: &[f32]) -> Vec<u32> {
        self.encode_batch(&[wav]).pop().expect("one clip in, one code sequence out")
    }

    /// [`Self::encode`] of every clip, as few recorded device passes as
    /// [`ENCODE_PASS_SAMPLES`] allows: each clip's codes equal its own
    /// [`Self::encode`]. Returns the codes in input order.
    pub fn encode_batch(&self, wavs: &[&[f32]]) -> Vec<Vec<u32>> {
        assert!(wavs.iter().all(|w| !w.is_empty()), "empty wav");
        let hop = self.encode_hop() as usize;
        let mut order: Vec<usize> = (0..wavs.len()).collect();
        order.sort_by_key(|&i| wavs[i].len());
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); wavs.len()];
        let mut group: Vec<usize> = Vec::new();
        let flush = |group: &mut Vec<usize>, out: &mut Vec<Vec<u32>>| {
            if group.is_empty() {
                return;
            }
            let clips: Vec<&[f32]> = group.iter().map(|&i| wavs[i]).collect();
            for (i, codes) in group.drain(..).zip(Pass::new(self).encode(&clips)) {
                out[i] = codes;
            }
        };
        for i in order {
            // Sorted ascending, so this clip sets the group's padded length.
            let padded = wavs[i].len().div_ceil(hop) * hop;
            if !group.is_empty() && (group.len() + 1) * padded > ENCODE_PASS_SAMPLES {
                flush(&mut group, &mut out);
            }
            group.push(i);
        }
        flush(&mut group, &mut out);
        out
    }

    /// Input samples per code frame: the SEANet strides times the
    /// frame-rate-match downsample.
    fn encode_hop(&self) -> u32 {
        let e = &self.cfg.enc;
        e.upsampling_ratios.iter().product::<u32>() * e.downsample_stride
    }
}

impl Pass<'_> {
    /// One device pass over `wavs` (non-empty clips): their codes, each
    /// `[T_b, 16]` row-major for clip `b`'s own `T_b`.
    fn encode(&self, wavs: &[&[f32]]) -> Vec<Vec<u32>> {
        let e = self.cfg.enc.clone();
        let n = wavs.len() as u32;
        let hop = self.encode_hop();
        let lmax = (wavs.iter().map(|w| w.len()).max().unwrap_or(0) as u32).div_ceil(hop) * hop;
        let mut lens: Vec<u32> = wavs.iter().map(|w| w.len() as u32).collect();

        // --- 1. SEANet conv encoder ---
        let mut padded = vec![0.0f32; (n * lmax) as usize];
        for (b, w) in wavs.iter().enumerate() {
            padded[b * lmax as usize..b * lmax as usize + w.len()].copy_from_slice(w);
        }
        let x = self.upload(&padded); // [n, 1, L0] NCL (cin = 1)
        let (mut h, mut l) = self.enc_conv(&x, "encoder.encoder.layers.0", n, 1, e.num_filters, lmax, e.kernel_size, 1);
        let mut dim = e.num_filters;
        // downsample stages: reversed(upsampling_ratios).
        let ratios: Vec<u32> = e.upsampling_ratios.iter().rev().copied().collect();
        let mut li = 1usize;
        for &r in &ratios {
            // residual block: ELU, conv(dim->dim/compress,k=res_k), ELU, conv(->dim,k1).
            let hidden = dim / e.compress;
            let base = format!("encoder.encoder.layers.{li}");
            let y = self.elu(&h, n * dim * l);
            let (y, _) = self.enc_conv(&y, &format!("{base}.block.1"), n, dim, hidden, l, e.residual_kernel_size, 1);
            let y = self.elu(&y, n * hidden * l);
            let (y, _) = self.enc_conv(&y, &format!("{base}.block.3"), n, hidden, dim, l, 1, 1);
            h = self.add2(&h, &y, n * dim * l);
            li += 1; // resnet
            li += 1; // ELU (no params)
            // downsample conv: dim -> 2*dim, k=2r, stride=r. Its last window
            // reads past a clip's end: zero there, as the unpadded conv reads.
            let y = self.elu(&h, n * dim * l);
            self.mask_past(&y, n, dim, l, &lens);
            let (hd, lo) = self.enc_conv(&y, &format!("encoder.encoder.layers.{li}"), n, dim, dim * 2, l, 2 * r, r);
            h = hd;
            l = lo;
            lens.iter_mut().for_each(|v| *v = v.div_ceil(r));
            dim *= 2;
            li += 1; // downsample conv
        }
        // final: ELU, conv(dim -> hidden_size, k=last_kernel_size, stride 1).
        li += 1; // final ELU (no params)
        let y = self.elu(&h, n * dim * l);
        let (hd, lo) = self.enc_conv(&y, &format!("encoder.encoder.layers.{li}"), n, dim, e.hidden_size, l, e.last_kernel_size, 1);
        h = hd;
        l = lo;

        // --- 2. encoder transformer (token-major [n*L, hidden]) ---
        let mut tm = self.to_token_major(&h, n, e.hidden_size, l); // [n*L, 512]
        tm = self.enc_transformer(&tm, n, l);
        h = self.to_channel_major(&tm, n, l, e.hidden_size); // [n, 512, L]

        // --- 3. frame-rate-match downsample ---
        let (h, t) = self.enc_downsample(&h, n, e.hidden_size, l, &lens);
        let frames: Vec<u32> = lens.iter().map(|v| v.div_ceil(e.downsample_stride)).collect();

        // --- 4. split-RVQ encode ---
        let nq = self.cfg.num_quantizers as usize;
        let codes = self.rvq_encode(&h, n, t);
        let rows = t as usize * nq;
        frames.iter().enumerate().map(|(b, &tb)| codes[b * rows..b * rows + tb as usize * nq].iter().map(|&c| c as u32).collect()).collect()
    }

    /// Causal Mimi `conv1d` over a batch (NCL `[n, cin, l]`): left pad
    /// `effective_kernel - stride`, output length `ceil(L / stride)`, the
    /// per-channel `{prefix}.conv.bias` fused. Returns `([n, Cout, lo], lo)`.
    #[allow(clippy::too_many_arguments)]
    fn enc_conv(&self, x: &DeviceBuffer, prefix: &str, n: u32, cin: u32, cout: u32, l: u32, k: u32, stride: u32) -> (DeviceBuffer, u32) {
        let pad = k - stride;
        let lo = l.div_ceil(stride);
        let c = audio::conv::Conv1d { n, cin, l, cout, k, stride, pad, dilation: 1, groups: 1, lo };
        (self.conv(&c, x, prefix), lo)
    }

    /// Clip lengths as a device `u32` buffer, for the length-aware kernels.
    fn lens(&self, lens: &[u32]) -> DeviceBuffer {
        let b = self.gpu.buffer("lens", (lens.len() * 4) as u64, BufUsage::STORAGE | BufUsage::COPY_DST);
        self.gpu.write(&b, lens);
        b
    }

    /// Zero every position of `[n, c, l]` at or past its clip's own length.
    fn mask_past(&self, x: &DeviceBuffer, n: u32, c: u32, l: u32, lens: &[u32]) {
        let lb = self.lens(lens);
        self.run(self.gpu.step(NCL_LEN_MASK, &[x, &lb], &[n * c * l, c, l], n * c * l));
    }

    /// ELU (alpha 1) over `n` floats, on the device.
    fn elu(&self, x: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let out = self.st(n as usize);
        self.run(self.gpu.step(ELU, &[x, &out], &[n, f(1.0)], n));
        out
    }

    /// The frame-rate-match downsample: a causal conv with **replicate** padding
    /// (`hidden -> hidden`, kernel `downsample_kernel`, stride `downsample_stride`,
    /// no bias). Left pad `k - stride`, and each clip's right edge replicated at
    /// its own length `lens[b]` (`pad1d_edge_lens`), then a `pad=0` conv.
    /// Returns `([n, C, T], T)` for the padded length.
    fn enc_downsample(&self, x: &DeviceBuffer, n: u32, c: u32, l: u32, lens: &[u32]) -> (DeviceBuffer, u32) {
        let k = self.cfg.enc.downsample_kernel;
        let stride = self.cfg.enc.downsample_stride;
        let left = k - stride;
        let lo = l.div_ceil(stride);
        let lp = left + lo * stride;
        let padded = self.st((n * c * lp) as usize);
        let lb = self.lens(lens);
        self.run(self.gpu.step(PAD1D_EDGE_LENS, &[x, &padded, &lb], &[n * c * lp, c, l, left, lp], n * c * lp));
        let conv = audio::conv::Conv1d { n, cin: c, l: lp, cout: c, k, stride, pad: 0, dilation: 1, groups: 1, lo };
        let out = self.st((n * c * lo) as usize);
        self.run(audio::conv::conv1d_fwd(&self.gpu, &audio::conv::ConvKernels { fwd: CONV1D, dx: 0, dw: 0 }, &conv, &padded, self.w("encoder.downsample.conv.weight"), &out));
        (out, lo)
    }

    /// Mimi encoder transformer over token-major `[n*T, hidden]` (`n` clips of
    /// `T` frames): 8 pre-LN layers (LayerNorm **with bias**, RoPE GQA
    /// attention with scaling `1/√head_dim`, exact-erf gelu MLP
    /// `fc2(gelu(fc1))`, per-channel LayerScale on each residual), plain causal
    /// attention. No input/output projection and no final norm (unlike the
    /// decoder).
    fn enc_transformer(&self, x0: &DeviceBuffer, n: u32, t: u32) -> DeviceBuffer {
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
        let ga = Gqa { b: n, t, n_heads: nh, n_kv_heads: nkv, head_dim: hd };
        let rows = n * t;

        let mut x = x0.clone();
        for layer in 0..e.num_hidden_layers as usize {
            let p = |leaf: &str| format!("encoder.encoder_transformer.layers.{layer}.{leaf}");
            // --- attention ---
            let xn = self.layernorm(&x, &p("input_layernorm"), rows, d, e.norm_eps);
            let q = self.matmul(&xn, &p("self_attn.q_proj.weight"), rows, d, hq);
            let k = self.matmul(&xn, &p("self_attn.k_proj.weight"), rows, d, hkv);
            let v = self.matmul(&xn, &p("self_attn.v_proj.weight"), rows, d, hkv);
            self.run(block::rope_fwd(&self.gpu, &ids, &q, rows, nh, hd, hq, t, theta));
            self.run(block::rope_fwd(&self.gpu, &ids, &k, rows, nkv, hd, hkv, t, theta));
            let scores = self.st((n * nh * t * t) as usize);
            let probs = self.st((n * nh * t * t) as usize);
            let ctx = self.st((rows * hq) as usize);
            // Plain causal attention over the whole clip. The released config
            // carries `encoder_config.sliding_window` (250), but the reference
            // encode (transformers' Mimi encoder, which `qwen-tts` runs) builds
            // its mask with `create_causal_mask` and never applies that window
            // in a single-shot encode: windowing here diverged from the
            // reference's codes from encoder frame 250 (10 s) on (knowledge
            // #582). A window of `t` is unbounded.
            let window = t;
            self.run_all(block::gqa_fwd_win(&self.gpu, GQA_SCORES_WIN, &ids, &ga, window, &q, &k, &v, &scores, &probs, &ctx));
            let attn = self.matmul(&ctx, &p("self_attn.o_proj.weight"), rows, hq, d);
            let attn = self.scale_chan(&attn, &p("self_attn_layer_scale.scale"), rows * d, d, 1);
            x = self.add2(&x, &attn, rows * d);
            // --- MLP: fc2(gelu(fc1(x))) ---
            let xn = self.layernorm(&x, &p("post_attention_layernorm"), rows, d, e.norm_eps);
            let g = self.matmul(&xn, &p("mlp.fc1.weight"), rows, d, ff);
            let g = self.gelu(&g, rows * ff);
            let mlp = self.matmul(&g, &p("mlp.fc2.weight"), rows, ff, d);
            let mlp = self.scale_chan(&mlp, &p("mlp_layer_scale.scale"), rows * d, d, 1);
            x = self.add2(&x, &mlp, rows * d);
        }
        x
    }

    /// Split residual-vector-quantizer **encode** over `n` clips of `t`
    /// frames: project the `[n, hidden, t]` embeddings to `codebook_dim`
    /// (semantic & acoustic `input_proj`), then per codebook the nearest-row
    /// assignment (`vq_argmin`, squared Euclidean, lowest index on ties - the
    /// host loop this replaced, bit for bit) and the residual subtraction
    /// (`vq_residual`), all on the device. Returns `[n*t, 16]` row-major codes
    /// (q0 semantic, q1..15 acoustic) as exact f32, the pass's one readback.
    fn rvq_encode(&self, emb_ncl: &DeviceBuffer, n: u32, t: u32) -> Vec<f32> {
        let e = self.cfg.enc.clone();
        let (h, cd) = (e.hidden_size, e.codebook_dim);
        let cs = e.codebook_size;
        let n_sem = e.num_semantic_quantizers;
        let nq = self.cfg.num_quantizers;
        let m = n * t;
        let emb_tm = self.to_token_major(emb_ncl, n, h, t); // [n*t, hidden]

        let codes = self.st((m * nq) as usize);
        let mut col = 0u32;
        for (group, layers) in [("semantic", n_sem), ("acoustic", nq - n_sem)] {
            let prefix = format!("encoder.quantizer.{group}_residual_vector_quantizer");
            let residual = self.matmul(&emb_tm, &format!("{prefix}.input_proj.weight"), m, h, cd);
            for q in 0..layers {
                let name = format!("{prefix}.layers.{q}.table");
                let table = self.w(&name);
                let assign = self.st(2 * m as usize);
                let shape = OpShape { m, n: cs, k: cd, dtype: Dtype::F32 };
                if DefaultSelector.select(Op::VqAssign, shape, &self.gpu.caps()) == KernelVariant::RowTiled {
                    let tiles = m.div_ceil(VQ_TILE_ROWS);
                    self.run(self.gpu.dispatch(VQ_ARGMIN_TILED, &[&residual, self.w(&transposed_name(&name)), &assign], &[m, cs, cd], gpu_core::Dispatch::Workgroups(tiles)));
                } else {
                    self.run(self.gpu.step(VQ_ARGMIN, &[&residual, table, &assign], &[m, cs, cd], m));
                }
                self.run(self.gpu.step(VQ_RESIDUAL, &[&assign, table, &residual, &codes], &[m, cd, nq, col], m * cd));
                col += 1;
            }
        }
        self.read(&codes, (m * nq) as usize)
    }
}

/// The name of an encoder codebook's transposed copy.
fn transposed_name(table: &str) -> String {
    format!("{table}:t")
}

/// Kernel-id map for the shared `model::block` builders (only forward ids used).
///
/// Every backward slot - and the RMSNorm
/// gain-grad trio - is [`block::UNREGISTERED`], never `0`: index 0 here is
/// `embed`, a kernel this model really does dispatch, so a `0` placeholder is a
/// misroute rather than a fail-fast (see `model::block::UNREGISTERED`).
fn ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: block::UNREGISTERED,
        rmsnorm_dx: block::UNREGISTERED,
        rmsnorm_dx_rows: block::UNREGISTERED,
        rmsnorm_dw: block::UNREGISTERED,
        rope: ROPE,
        rope_bwd: block::UNREGISTERED,
        gqa_scores: GQA_SCORES,
        gqa_apply: GQA_APPLY,
        attn_softmax: ATTN_SOFTMAX,
        gqa_dscores: block::UNREGISTERED,
        gqa_dv: block::UNREGISTERED,
        gqa_dq: block::UNREGISTERED,
        gqa_dk: block::UNREGISTERED,
        silu_mul: SILU_MUL,
        silu_da: block::UNREGISTERED,
        silu_db: block::UNREGISTERED,
        rmsnorm_rows: block::UNREGISTERED,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EncoderConfig;
    use data::rng::Lcg;

    /// One-layer tiny encoder transformer with random weights; every weight
    /// `enc_transformer` reads.
    fn tiny_encoder() -> Codec {
        let (d, heads, hd, ff) = (8u32, 2u32, 4u32, 16u32);
        let enc = EncoderConfig {
            hidden_size: d,
            num_hidden_layers: 1,
            num_attention_heads: heads,
            num_key_value_heads: heads,
            head_dim: hd,
            intermediate_size: ff,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            ..Default::default()
        };
        let cfg = CodecConfig { enc, ..Default::default() };
        let mut rng = Lcg::new(0x5eed_0001);
        let (du, hq, ffu) = (d as usize, (heads * hd) as usize, ff as usize);
        let p = |leaf: &str| format!("encoder.encoder_transformer.layers.0.{leaf}");
        let mut w: HashMap<String, Vec<f32>> = HashMap::new();
        for (name, n) in [
            ("input_layernorm.weight", du),
            ("input_layernorm.bias", du),
            ("post_attention_layernorm.weight", du),
            ("post_attention_layernorm.bias", du),
            ("self_attn.q_proj.weight", hq * du),
            ("self_attn.k_proj.weight", hq * du),
            ("self_attn.v_proj.weight", hq * du),
            ("self_attn.o_proj.weight", du * hq),
            ("mlp.fc1.weight", ffu * du),
            ("mlp.fc2.weight", du * ffu),
        ] {
            w.insert(p(name), rng.vec(n));
        }
        // Unit layer scales so a far frame's leak is not damped below visibility.
        w.insert(p("self_attn_layer_scale.scale"), vec![1.0; du]);
        w.insert(p("mlp_layer_scale.scale"), vec![1.0; du]);
        Codec::from_weights(cfg, w)
    }

    /// Encoder transformer output for `x` (`[t, hidden]` token-major).
    fn run_enc(codec: &Codec, x: &[f32], t: u32) -> Vec<f32> {
        let d = codec.cfg.enc.hidden_size as usize;
        let pass = Pass::new(codec);
        let out = pass.enc_transformer(&pass.upload(x), 1, t);
        pass.read(&out, t as usize * d)
    }

    /// The reference encode attends every earlier frame of the clip (plain
    /// causal, no window: see `enc_transformer`): a perturbed frame however far
    /// back reaches the last output frame, a later one never reaches an earlier.
    #[test]
    fn encoder_attention_is_causal_over_the_whole_clip() {
        let (t, d) = (40u32, 8usize);
        let codec = tiny_encoder();
        let x = Lcg::new(7).vec(t as usize * d);
        let base = run_enc(&codec, &x, t);
        let row = |v: &[f32], i: usize| v[i * d..(i + 1) * d].to_vec();
        let perturbed = |j: usize| {
            let mut y = x.clone();
            y[j * d] += 1.0;
            run_enc(&codec, &y, t)
        };
        let last = t as usize - 1;
        let first = perturbed(0);
        assert_ne!(row(&first, last), row(&base, last), "frame 0 did not reach frame {last}");
        let late = perturbed(last);
        for i in 0..last {
            assert_eq!(row(&late, i), row(&base, i), "frame {last} leaked into earlier frame {i}");
        }
    }

    /// The host loop the device quantizer replaced (`MimiEuclideanCodebook.quantize`: squared Euclidean distance
    /// summed over the dimension in order, the lowest index on ties, then the residual), as the test's oracle.
    fn host_rvq(residual: &mut [f32], tables: &[Vec<f32>], dim: usize) -> Vec<u32> {
        tables
            .iter()
            .map(|table| {
                let bins = table.len() / dim;
                let (mut best, mut best_d) = (0usize, f32::INFINITY);
                for b in 0..bins {
                    let mut d = 0.0f32;
                    for c in 0..dim {
                        let diff = residual[c] - table[b * dim + c];
                        d += diff * diff;
                    }
                    if d < best_d {
                        best_d = d;
                        best = b;
                    }
                }
                for c in 0..dim {
                    residual[c] -= table[best * dim + c];
                }
                best as u32
            })
            .collect()
    }

    /// The device quantizer gives the host loop's codes bit for bit, on the codec's own shapes (a 512-wide
    /// embedding projected to 256, 1 semantic + 15 acoustic codebooks of 2048 rows), over enough frames that a
    /// distance rounded differently would show as a flipped near-tie.
    #[test]
    fn the_device_quantizer_codes_as_the_host_loop() {
        let (h, cd, cs, nq) = (512u32, 256u32, 2048u32, 16u32);
        let enc = EncoderConfig { hidden_size: h, codebook_dim: cd, codebook_size: cs, num_semantic_quantizers: 1, ..Default::default() };
        let cfg = CodecConfig { num_quantizers: nq, enc, ..Default::default() };
        let mut rng = Lcg::new(data::rng::random_seed());
        let mut w: HashMap<String, Vec<f32>> = HashMap::new();
        let mut tables: Vec<(String, Vec<f32>)> = Vec::new();
        for (group, layers) in [("semantic", 1), ("acoustic", nq - 1)] {
            let prefix = format!("encoder.quantizer.{group}_residual_vector_quantizer");
            w.insert(format!("{prefix}.input_proj.weight"), rng.vec_scaled((cd * h) as usize, 1.0 / (h as f32).sqrt()));
            for q in 0..layers {
                // Codebooks that shrink stage by stage, as a trained residual quantizer's do.
                let t = rng.vec_scaled((cs * cd) as usize, 0.8f32.powi(q as i32 + 1));
                tables.push((group.to_string(), t.clone()));
                w.insert(format!("{prefix}.layers.{q}.table"), t);
            }
        }
        let codec = Codec::from_weights_on(gpu_core::testgpu::dev(PIPELINES), cfg, w);
        let (n, t) = (3u32, 400u32);
        let pass = Pass::new(&codec);
        let emb = pass.upload(&rng.vec_scaled((n * h * t) as usize, 1.0));
        let device = pass.rvq_encode(&emb, n, t);
        let tm = pass.to_token_major(&emb, n, h, t);
        let proj = |g: &str| pass.read(&pass.matmul(&tm, &format!("encoder.quantizer.{g}_residual_vector_quantizer.input_proj.weight"), n * t, h, cd), (n * t * cd) as usize);
        let (sem, aco) = (proj("semantic"), proj("acoustic"));
        let (sem_t, aco_t): (Vec<_>, Vec<_>) = tables.into_iter().partition(|(g, _)| g == "semantic");
        let sem_t: Vec<Vec<f32>> = sem_t.into_iter().map(|(_, t)| t).collect();
        let aco_t: Vec<Vec<f32>> = aco_t.into_iter().map(|(_, t)| t).collect();
        let cd = cd as usize;
        let mut flipped = 0usize;
        for f in 0..(n * t) as usize {
            let mut want = host_rvq(&mut sem[f * cd..(f + 1) * cd].to_vec(), &sem_t, cd);
            want.extend(host_rvq(&mut aco[f * cd..(f + 1) * cd].to_vec(), &aco_t, cd));
            let got: Vec<u32> = device[f * nq as usize..(f + 1) * nq as usize].iter().map(|&c| c as u32).collect();
            flipped += usize::from(got != want);
        }
        assert_eq!(flipped, 0, "{flipped} of {} frames quantise differently on {}", n * t, gpu_core::backend_name());
    }

}
