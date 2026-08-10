// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! NPU (OpenVINO) Qwen3-TTS generation — the autoregressive **Talker** runs as a
//! compiled fixed-length hidden-state graph on the Intel NPU (one infer per
//! frame, the growing input-embedding context padded to the compiled length;
//! causal masking makes the last real position independent of the padding,
//! exactly like [`npu::qwen_decode`]). The codebook-0 head + sampling and the MTP
//! residual fill stay on the host (CPU); the generated codes are decoded to a
//! 24 kHz waveform on the NPU **codec** graph.
//!
//! This path deliberately avoids instantiating any `gpu_core` decoder:
//! [`TalkerTables`] loads only the CPU-side text/codec tables straight from the
//! checkpoint, so `--device npu` never uploads the 3 GB Talker decoder to the host
//! backend (which, under `--device npu`, stays at its wgpu/GL default). The two
//! big graphs (Talker hidden + codec) are exported once and cached as ONNX (+ an
//! external-data sidecar), then compiled by OpenVINO — reused across runs via the
//! cache dir, mirroring the proven `brain qwen … --device npu` flow.

use std::path::{Path, PathBuf};

use data::rng::Rng;
use npu::openvino::{
    BackStreamSession, CodecSession, EmbedSession, FusedMtpSession, KvSession, NpuConfig, NpuDevice, PrefillSession,
};

use crate::config::TalkerConfig;
use crate::pipeline::{sample_cb0, GenOpts};
use crate::prompt::{Prompt, TalkerHost, TtsSpecials};
use crate::talker::TextProjection;

/// Human-readable summary of the resolved Talker hardware path, printed at startup
/// so it is always clear which device + weight precision is ACTUALLY used. Crucially
/// distinguishes a *native* INT4 device from one that only lists INT8: on the latter
/// (the Intel NPU today) an INT4 graph still runs, but as weight-**compression** (the
/// 4-bit weights are decompressed to a native type for the MAC) — a memory-bandwidth
/// win, not native 4-bit arithmetic. Queries the device via OpenVINO.
pub fn describe_talker_path(device: NpuDevice, allow_fallback: bool, quant: bool, int4: bool) -> String {
    let want = if int4 { "INT4" } else if quant { "INT8" } else { "fp32" };
    match npu::openvino::device_info(device, allow_fallback) {
        Ok(info) => {
            let precision = if int4 {
                if info.supports("INT4") {
                    "INT4 (native 4-bit compute)".to_string()
                } else {
                    let native_max = if info.supports("INT8") { "INT8" } else { "FP16" };
                    format!(
                        "INT4 weight-compression (device native max={native_max}; 4-bit weights \
                         decompressed at runtime — a bandwidth win, NOT native 4-bit compute)"
                    )
                }
            } else {
                want.to_string()
            };
            format!(
                "tts: Talker path => device={} ({}) | weights={} | device OPTIMIZATION_CAPABILITIES=[{}]",
                info.device,
                info.full_name,
                precision,
                info.capabilities.join(", ")
            )
        }
        Err(e) => format!(
            "tts: Talker path => device={} | weights={want} (device capability query failed: {e})",
            device.ov_str()
        ),
    }
}

/// Elementwise `a += b` over `[d]` slices (host feedback-embedding accumulation).
fn add_into(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// Talker graph context-length bucket and codec code-length bucket. Rounding the
/// compiled shapes to these multiples means a small, fixed set of graphs/blobs is
/// reused across prompts of varying length (bounded cache + amortised compile).
const CAP_BUCKET: usize = 64;
const CODEC_BUCKET: usize = 32;

fn round_up(n: usize, m: usize) -> usize {
    n.div_ceil(m) * m
}

/// CPU-only Talker tables for the NPU host path: `d_model`, the text-projection
/// front-end, the codebook-0 input-embedding table (`tok.weight`) and the codec
/// head (`lm_head.weight`). Loaded straight from the brain Talker checkpoint with
/// no `gpu_core` decoder — the decoder itself runs on the NPU via [`TalkerNpu`].
pub struct TalkerTables {
    pub cfg: TalkerConfig,
    text: TextProjection,
    codec_embedding: Vec<f32>, // [vocab, d] (= tok.weight)
    codec_head: Vec<f32>,      // [vocab, d] (= lm_head.weight)
}

impl TalkerTables {
    /// Load the CPU tables from the brain Talker checkpoint (the same container
    /// [`crate::gen::TalkerGen::load`] reads, minus the decoder upload).
    pub fn load(path: &str) -> TalkerTables {
        let c = checkpoint::load(path);
        let qcfg = qwen3::QwenConfig::from_json(&c.header["config"]);
        let mut cfg = TalkerConfig::from_qwen(&qcfg);
        let take = |name: &str| {
            c.find(name, "")
                .cloned()
                .unwrap_or_else(|| panic!("TalkerTables::load missing tensor {name}"))
        };
        let codec_embedding = take("tok.weight");
        let codec_head = take("lm_head.weight");
        let fc1_w = take("text_projection.fc1.weight");
        let fc1_b = take("text_projection.fc1.bias");
        let fc2_w = take("text_projection.fc2.weight");
        let fc2_b = take("text_projection.fc2.bias");
        let text_embedding = c.find("text_embedding.weight", "").cloned();
        let inter = fc1_b.len();
        let in_dim = fc1_w.len() / inter;
        let out = fc2_b.len();
        let text_vocab = text_embedding.as_ref().map(|e| e.len() / in_dim).unwrap_or(0);
        cfg.text_hidden_size = in_dim as u32;
        if text_vocab > 0 {
            cfg.text_vocab_size = text_vocab as u32;
        }
        let text = TextProjection {
            text_embedding,
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
            in_dim,
            inter,
            out,
            text_vocab,
        };
        TalkerTables {
            cfg,
            text,
            codec_embedding,
            codec_head,
        }
    }

    /// d_model.
    pub fn d(&self) -> usize {
        self.cfg.d_model as usize
    }

    /// Codebook-0 logits (`[vocab]`) for a final-norm hidden row — the same host
    /// head as [`crate::gen::TalkerGen::codec_head_logits`].
    pub fn codec_head_logits(&self, hidden_row: &[f32]) -> Vec<f32> {
        let d = self.d();
        let v = self.cfg.vocab as usize;
        assert_eq!(hidden_row.len(), d);
        // `y = x·Wᵀ`, W=codec_head[v,d] row-major — the shared AVX2+rayon matvec
        // (was a scalar single-thread loop, ~15ms/frame; parallel ~2ms).
        model::hostmath::matvec(&self.codec_head, hidden_row, v, d)
    }
}

impl TalkerHost for TalkerTables {
    fn d(&self) -> usize {
        self.cfg.d_model as usize
    }
    fn text(&self) -> &TextProjection {
        &self.text
    }
    fn codec_embed(&self, id: u32) -> &[f32] {
        let d = self.d();
        let s = id as usize * d;
        &self.codec_embedding[s..s + d]
    }
}

/// The NPU-resident Talker decoder: a compiled fixed-length (`cap`) hidden-state
/// graph (`inputs_embeds:[1,cap,d] -> hidden:[1,cap,d]`) plus the growing input-
/// embedding context. Each [`feed`](Self::feed) appends positions, zero-pads the
/// context to `cap`, runs one NPU inference, and returns the final-norm hidden row
/// at the new last position — the cache-free prefill the static graph requires.
pub struct TalkerNpu {
    sess: EmbedSession,
    cap: usize,
    d: usize,
    ctx: Vec<f32>,
    max_pos: usize,
    device: String,
}

impl TalkerNpu {
    /// Export (or reuse cached) the Talker hidden graph for `cap` positions and
    /// compile it on the OpenVINO `device`. `max_pos` is the model's
    /// `max_position_embeddings` (a hard ceiling on the context).
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        talker_path: &str,
        cap: usize,
        device: NpuDevice,
        allow_fallback: bool,
        cache_dir: Option<&Path>,
        max_pos: usize,
        quant: bool,
    ) -> Result<TalkerNpu, String> {
        let onnx = prepare_talker_onnx(talker_path, cap, cache_dir, quant)?;
        let cfg = npu_config(device, allow_fallback, cache_dir);
        let sess = EmbedSession::load_path(&onnx, &cfg).map_err(|e| e.to_string())?;
        let d = sess.d_in();
        let device = sess.device().to_string();
        Ok(TalkerNpu {
            sess,
            cap,
            d,
            ctx: Vec::new(),
            max_pos,
            device,
        })
    }

    pub fn device(&self) -> &str {
        &self.device
    }
    pub fn d(&self) -> usize {
        self.d
    }
    pub fn pos(&self) -> usize {
        self.ctx.len() / self.d
    }
    pub fn cap(&self) -> usize {
        self.cap
    }
    pub fn max_pos(&self) -> usize {
        self.max_pos
    }

    /// Clear the context (start a new utterance).
    pub fn reset(&mut self) {
        self.ctx.clear();
    }

    /// Append `embeds` (`[k*d]`, k≥1 positions) to the context, run one NPU
    /// inference over the zero-padded `[1,cap,d]` context, and return the final-
    /// norm hidden row at the new last position (`[d]`).
    pub fn feed(&mut self, embeds: &[f32]) -> Result<Vec<f32>, String> {
        assert!(!embeds.is_empty() && embeds.len().is_multiple_of(self.d), "feed must be a whole number of [d] rows");
        self.ctx.extend_from_slice(embeds);
        let len = self.ctx.len() / self.d;
        if len > self.cap {
            return Err(format!("Talker context {len} exceeds compiled cap {}", self.cap));
        }
        let mut buf = vec![0.0f32; self.cap * self.d];
        buf[..self.ctx.len()].copy_from_slice(&self.ctx);
        let hidden = self.sess.run_embeds(&buf).map_err(|e| e.to_string())?;
        Ok(hidden[(len - 1) * self.d..len * self.d].to_vec())
    }
}

/// Autoregressively generate codec codes `[n_frames*16]` with the Talker on the
/// NPU. Identical sampling + MTP feedback logic to
/// [`crate::pipeline::generate_codes_cached`]; only the Talker decoder differs
/// (compiled NPU graph vs. the CPU KV-cache mirror). The prefix is streamed in a
/// single NPU inference; each subsequent frame is one inference over the (padded)
/// growing context.
pub fn generate_codes_npu(
    talker: &mut TalkerNpu,
    tables: &TalkerTables,
    mtp: &mut dyn MtpEngine,
    sp: &TtsSpecials,
    prompt: &Prompt,
    opts: &GenOpts,
) -> Result<Vec<u32>, String> {
    let d = tables.d();
    let n_trailing = prompt.trailing.len() / d;
    let mut rng = Rng::new(opts.seed);
    let profile = std::env::var("TTS_PROFILE").is_ok();
    let (mut t_step, mut t_mtp) = (0.0f64, 0.0f64);
    use std::time::Instant;

    talker.reset();
    let t_pref0 = Instant::now();
    let mut past_hidden = talker.feed(&prompt.embeds)?;
    let t_prefix = t_pref0.elapsed().as_secs_f64() * 1e3;
    let mut cb0 = sample_cb0(
        tables.codec_head_logits(&past_hidden),
        sp.codec_eos,
        opts.min_new == 0,
        opts.temperature,
        opts.top_k,
        &mut rng,
    );

    let mut frames: Vec<u32> = Vec::new();
    let mut s = 0usize;
    loop {
        if (cb0 == sp.codec_eos && s >= opts.min_new) || s >= opts.max_frames {
            break;
        }
        let cb0_embed = tables.codec_embed(cb0).to_vec();
        let tm = Instant::now();
        let (residuals, res_sum) = mtp.generate_residuals(&past_hidden, &cb0_embed);
        t_mtp += tm.elapsed().as_secs_f64() * 1e3;
        frames.push(cb0);
        frames.extend_from_slice(&residuals);

        let mut feed = cb0_embed;
        add_into(&mut feed, &res_sum);
        if s < n_trailing {
            add_into(&mut feed, &prompt.trailing[s * d..(s + 1) * d]);
        } else {
            add_into(&mut feed, &prompt.tts_pad);
        }
        s += 1;
        // Stop before exceeding the model context or the compiled graph capacity.
        if talker.pos() >= talker.max_pos() || talker.pos() >= talker.cap() {
            break;
        }
        let ts = Instant::now();
        past_hidden = talker.feed(&feed)?;
        t_step += ts.elapsed().as_secs_f64() * 1e3;
        cb0 = sample_cb0(
            tables.codec_head_logits(&past_hidden),
            sp.codec_eos,
            s >= opts.min_new,
            opts.temperature,
            opts.top_k,
            &mut rng,
        );
    }
    if profile {
        let nf = s.max(1) as f64;
        eprintln!(
            "[tts-npu-profile] prefix-feed={t_prefix:.0}ms | talker-step total={t_step:.0}ms \
             ({:.0}ms/frame) | mtp total={t_mtp:.0}ms ({:.0}ms/frame) | frames={s}",
            t_step / nf,
            t_mtp / nf,
        );
    }
    Ok(frames)
}

/// One-shot: size + compile the Talker hidden graph for this prompt, then run the
/// NPU generation loop. `cap` is `prefix + max_frames + slack`, clamped to the
/// model's `max_position_embeddings`.
#[allow(clippy::too_many_arguments)]
pub fn generate_npu(
    talker_path: &str,
    tables: &TalkerTables,
    mtp: &mut dyn MtpEngine,
    sp: &TtsSpecials,
    prompt: &Prompt,
    opts: &GenOpts,
    device: NpuDevice,
    allow_fallback: bool,
    cache_dir: Option<&Path>,
    quant: bool,
) -> Result<Vec<u32>, String> {
    let d = tables.d();
    let n_prefix = prompt.embeds.len() / d;
    let max_pos = tables.cfg.max_position_embeddings as usize;
    // Bucket the compiled context length to a multiple of CAP_BUCKET so prompts of
    // similar size share ONE compiled graph — amortising the (slow) first compile
    // across runs and bounding the NPU cache to a couple of graphs instead of one
    // per exact length.
    let need = n_prefix + opts.max_frames + 2;
    let cap = round_up(need, CAP_BUCKET).min(max_pos);
    eprintln!(
        "tts npu: compiling Talker hidden graph ({} | prefix={n_prefix} + max_frames={} -> cap={cap}); \
         first compile per cap is slow, cached after…",
        if quant { "INT8" } else { "fp32" },
        opts.max_frames
    );
    let mut talker = TalkerNpu::load(talker_path, cap, device, allow_fallback, cache_dir, max_pos, quant)?;
    eprintln!("tts npu: Talker decoder running on {}", talker.device());
    generate_codes_npu(&mut talker, tables, mtp, sp, prompt, opts)
}

/// The NPU **KV-cache decode** Talker: a compiled decode-step graph plus the
/// host-side per-layer key/value cache. Each [`feed1`](Self::feed1) processes ONE
/// new token — O(1) projections + O(t) attention over the cache — instead of
/// re-running the whole context like [`TalkerNpu`]. The new token's k/v (a graph
/// output) are written back into the cache at the current slot for the next step.
pub struct KvTalker {
    sess: KvSession,
    /// One-shot prefill graph — `None` skips compiling it (short prefixes seed the
    /// cache token-by-token instead, avoiding a second ~1.4 GB graph compile).
    prefill: Option<PrefillSession>,
    cap: usize,
    d: usize,
    nkv: usize,
    hd: usize,
    half: usize,
    n_layers: usize,
    theta: f32,
    max_pos: usize,
    past_k: Vec<Vec<f32>>, // [n_layers][nkv*cap*hd]
    past_v: Vec<Vec<f32>>,
    pos: usize,
    device: String,
}

impl KvTalker {
    /// Export (or reuse) the decode-step graph for `cap` cache slots and compile it.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        talker_path: &str,
        cap: usize,
        device: NpuDevice,
        allow_fallback: bool,
        cache_dir: Option<&Path>,
        cfg: &TalkerConfig,
        quant: bool,
        int4: bool,
        with_prefill: bool,
    ) -> Result<KvTalker, String> {
        let onnx = prepare_decode_onnx(talker_path, cap, cache_dir, quant, int4)?;
        let ncfg = npu_config(device, allow_fallback, cache_dir);
        let nkv = cfg.n_kv_heads as usize;
        let hd = cfg.head_dim as usize;
        let nl = cfg.n_layers as usize;
        let d = cfg.d_model as usize;
        let sess = KvSession::load_path(&onnx, &ncfg, nl, d, nkv, hd, cap).map_err(|e| e.to_string())?;
        // Companion prefill graph (full context -> hidden + K/V) seeds the cache in
        // one inference — only worth its (large) compile for long prefixes (clone).
        let prefill = if with_prefill {
            let ponnx = prepare_prefill_onnx(talker_path, cap, cache_dir, quant, int4)?;
            Some(PrefillSession::load_path(&ponnx, &ncfg, nl, d, nkv, hd, cap).map_err(|e| e.to_string())?)
        } else {
            None
        };
        let device = sess.device().to_string();
        Ok(KvTalker {
            sess,
            prefill,
            cap,
            d,
            nkv,
            hd,
            half: hd / 2,
            n_layers: nl,
            theta: cfg.rope_theta,
            max_pos: cfg.max_position_embeddings as usize,
            past_k: vec![vec![0.0f32; nkv * cap * hd]; nl],
            past_v: vec![vec![0.0f32; nkv * cap * hd]; nl],
            pos: 0,
            device,
        })
    }

    pub fn device(&self) -> &str {
        &self.device
    }
    pub fn d(&self) -> usize {
        self.d
    }
    pub fn pos(&self) -> usize {
        self.pos
    }
    pub fn cap(&self) -> usize {
        self.cap
    }
    pub fn max_pos(&self) -> usize {
        self.max_pos
    }

    pub fn reset(&mut self) {
        for b in self.past_k.iter_mut().chain(self.past_v.iter_mut()) {
            b.iter_mut().for_each(|x| *x = 0.0);
        }
        self.pos = 0;
    }

    /// Seed the cache with the whole prompt prefix in ONE prefill inference (vs
    /// streaming it token-by-token). Returns the final-norm hidden at the last
    /// prefix position and leaves `pos = n_prefix`.
    pub fn prefill_prompt(&mut self, embeds: &[f32]) -> Result<Vec<f32>, String> {
        let d = self.d;
        let n = embeds.len() / d;
        if n == 0 || n > self.cap {
            return Err(format!("prefix {n} positions exceeds cap {}", self.cap));
        }
        // Fast path: one prefill inference seeds the whole cache.
        if self.prefill.is_some() {
            let mut buf = vec![0.0f32; self.cap * d];
            buf[..embeds.len()].copy_from_slice(embeds);
            let (hidden, k, v) = self.prefill.as_mut().unwrap().run(&buf).map_err(|e| e.to_string())?;
            for l in 0..self.n_layers {
                self.past_k[l].copy_from_slice(&k[l]);
                self.past_v[l].copy_from_slice(&v[l]);
            }
            self.pos = n;
            return Ok(hidden[(n - 1) * d..n * d].to_vec());
        }
        // Fallback (no prefill graph compiled): seed token-by-token. Cheap for the
        // short prefixes (design/synth) this mode is used for.
        let mut last = vec![0.0f32; d];
        for i in 0..n {
            last = self.feed1(&embeds[i * d..(i + 1) * d])?;
        }
        Ok(last)
    }

    /// Decode one token (`embed:[d]`), returning its final-norm hidden state `[d]`.
    pub fn feed1(&mut self, embed: &[f32]) -> Result<Vec<f32>, String> {
        assert_eq!(embed.len(), self.d);
        if self.pos >= self.cap {
            return Err(format!("KV cache full ({} slots)", self.cap));
        }
        let (hd, half, cap) = (self.hd, self.half, self.cap);
        // RoPE tables for the current absolute position (half-split / NeoX).
        let mut cos = vec![0.0f32; hd];
        let mut sin = vec![0.0f32; hd];
        for j in 0..hd {
            let m = (j % half) as f32;
            let ang = self.pos as f32 * self.theta.powf(-2.0 * m / hd as f32);
            cos[j] = ang.cos();
            sin[j] = ang.sin();
        }
        // Additive mask: the new token attends to the already-filled slots [0,pos).
        let mut mask = vec![f32::NEG_INFINITY; cap];
        for m in mask.iter_mut().take(self.pos) {
            *m = 0.0;
        }
        let (hidden, nk, nv) = self
            .sess
            .run_step(embed, &cos, &sin, &mask, &self.past_k, &self.past_v)
            .map_err(|e| e.to_string())?;
        // Write this token's k/v into the cache at row `pos` (per head).
        let pos = self.pos;
        for l in 0..self.n_layers {
            for h in 0..self.nkv {
                let dst = h * cap * hd + pos * hd;
                let src = h * hd;
                self.past_k[l][dst..dst + hd].copy_from_slice(&nk[l][src..src + hd]);
                self.past_v[l][dst..dst + hd].copy_from_slice(&nv[l][src..src + hd]);
            }
        }
        self.pos += 1;
        Ok(hidden)
    }
}

/// The per-frame residual (MTP) code predictor — implemented on the CPU
/// ([`crate::gen_kv_mtp::CpuMtp`]) or on the NPU ([`KvMtp`]). Lets the generation
/// loop pick the MTP backend without being generic.
pub trait MtpEngine {
    fn generate_residuals(&mut self, talker_hidden: &[f32], cb0_embed: &[f32]) -> (Vec<u32>, Vec<f32>);
}

impl MtpEngine for crate::gen_kv_mtp::CpuMtp {
    fn generate_residuals(&mut self, talker_hidden: &[f32], cb0_embed: &[f32]) -> (Vec<u32>, Vec<f32>) {
        crate::gen_kv_mtp::CpuMtp::generate_residuals(self, talker_hidden, cb0_embed)
    }
}

/// MTP code predictor with its 5-layer decoder running on the NPU via the resident
/// KV-cache decode graph (the same block as the Talker, reused). The 15 residual
/// substeps each decode one token on the NPU; the `small_to_mtp_projection`,
/// per-residual `codec_embedding` and `lm_head` stay on the host.
pub struct KvMtp {
    sess: KvSession,
    cap: usize,    // num_code_groups (MTP sequence length)
    d: usize,      // MTP decoder hidden
    emb: usize,    // embedding_dim (Talker hidden)
    nkv: usize,
    hd: usize,
    half: usize,
    n_layers: usize,
    theta: f32,
    vocab: usize,
    n_res: usize,
    proj: Option<(Vec<f32>, Vec<f32>)>, // small_to_mtp_projection (d x emb) + bias
    codec_embedding: Vec<Vec<f32>>,     // [n_res][vocab*emb]
    lm_head: Vec<Vec<f32>>,             // [n_res][vocab*d]
    past_k: Vec<Vec<f32>>,
    past_v: Vec<Vec<f32>>,
    pos: usize,
    device: String,
}

impl KvMtp {
    pub fn load(
        mtp_path: &str,
        device: NpuDevice,
        allow_fallback: bool,
        cache_dir: Option<&Path>,
        quant: bool,
    ) -> Result<KvMtp, String> {
        let c = checkpoint::load(mtp_path);
        let cfg = crate::config::MtpConfig::from_brain_json(&c.header["config"]);
        let cap = cfg.num_code_groups as usize;
        let d = cfg.d_model as usize;
        let emb = cfg.embedding_dim as usize;
        let nkv = cfg.n_kv_heads as usize;
        let hd = cfg.head_dim as usize;
        let nl = cfg.n_layers as usize;
        let vocab = cfg.vocab as usize;
        let n_res = cfg.n_residual() as usize;
        let take = |n: &str| c.find(n, "").cloned().unwrap_or_else(|| panic!("KvMtp: missing {n}"));
        let codec_embedding = (0..n_res).map(|i| take(&format!("codec_embedding.{i}.weight"))).collect();
        let lm_head = (0..n_res).map(|i| take(&format!("lm_head.{i}.weight"))).collect();
        let proj = if emb != d {
            Some((take("small_to_mtp_projection.weight"), take("small_to_mtp_projection.bias")))
        } else {
            None
        };
        let onnx = prepare_mtp_decode_onnx(mtp_path, cap, cache_dir, quant)?;
        let ncfg = npu_config(device, allow_fallback, cache_dir);
        let sess = KvSession::load_path(&onnx, &ncfg, nl, d, nkv, hd, cap).map_err(|e| e.to_string())?;
        let device = sess.device().to_string();
        Ok(KvMtp {
            sess,
            cap,
            d,
            emb,
            nkv,
            hd,
            half: hd / 2,
            n_layers: nl,
            theta: cfg.rope_theta,
            vocab,
            n_res,
            proj,
            codec_embedding,
            lm_head,
            past_k: vec![vec![0.0f32; nkv * cap * hd]; nl],
            past_v: vec![vec![0.0f32; nkv * cap * hd]; nl],
            pos: 0,
            device,
        })
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    /// Project a Talker-width embedding to the MTP decoder width (Identity on 0.6B).
    fn project(&self, emb: &[f32]) -> Vec<f32> {
        match &self.proj {
            Some((w, b)) => {
                let mut y = model::hostmath::matvec(w, emb, self.d, self.emb);
                for (yi, bi) in y.iter_mut().zip(b) {
                    *yi += bi;
                }
                y
            }
            None => emb.to_vec(),
        }
    }

    /// Decode one MTP token (already projected, `[d]`) on the NPU; returns the
    /// final-norm hidden `[d]`.
    fn feed1(&mut self, x: &[f32]) -> Result<Vec<f32>, String> {
        let (hd, half, cap) = (self.hd, self.half, self.cap);
        let mut cos = vec![0.0f32; hd];
        let mut sin = vec![0.0f32; hd];
        for j in 0..hd {
            let m = (j % half) as f32;
            let ang = self.pos as f32 * self.theta.powf(-2.0 * m / hd as f32);
            cos[j] = ang.cos();
            sin[j] = ang.sin();
        }
        let mut mask = vec![f32::NEG_INFINITY; cap];
        for mm in mask.iter_mut().take(self.pos) {
            *mm = 0.0;
        }
        let (hidden, nk, nv) = self
            .sess
            .run_step(x, &cos, &sin, &mask, &self.past_k, &self.past_v)
            .map_err(|e| e.to_string())?;
        let pos = self.pos;
        for l in 0..self.n_layers {
            for h in 0..self.nkv {
                let dst = h * cap * hd + pos * hd;
                let src = h * hd;
                self.past_k[l][dst..dst + hd].copy_from_slice(&nk[l][src..src + hd]);
                self.past_v[l][dst..dst + hd].copy_from_slice(&nv[l][src..src + hd]);
            }
        }
        self.pos += 1;
        Ok(hidden)
    }

    fn reset(&mut self) {
        for b in self.past_k.iter_mut().chain(self.past_v.iter_mut()) {
            b.iter_mut().for_each(|x| *x = 0.0);
        }
        self.pos = 0;
    }

    fn argmax(row: &[f32]) -> usize {
        let mut best = 0usize;
        for j in 1..row.len() {
            if row[j] > row[best] {
                best = j;
            }
        }
        best
    }
}

impl MtpEngine for KvMtp {
    /// Mirror of [`crate::gen_kv_mtp::CpuMtp::generate_residuals`] but the 5-layer
    /// decoder runs on the NPU. Greedy over the 15 residual codebooks.
    fn generate_residuals(&mut self, talker_hidden: &[f32], cb0_embed: &[f32]) -> (Vec<u32>, Vec<f32>) {
        let (emb, d, vocab) = (self.emb, self.d, self.vocab);
        let nres = self.n_res;
        self.reset();
        // pos 0: the Talker hidden (projected); no head reads it.
        let p0 = self.project(talker_hidden);
        let _ = self.feed1(&p0).expect("KvMtp feed1");
        let mut codes = vec![0u32; nres];
        let mut res_sum = vec![0.0f32; emb];
        let mut input_raw = cb0_embed.to_vec();
        for k in 1..=nres {
            let pin = self.project(&input_raw);
            let hidden = self.feed1(&pin).expect("KvMtp feed1");
            let logits = model::hostmath::matvec(&self.lm_head[k - 1], &hidden, vocab, d);
            let best = Self::argmax(&logits);
            codes[k - 1] = best as u32;
            let r = self.codec_embedding[k - 1][best * emb..(best + 1) * emb].to_vec();
            for j in 0..emb {
                res_sum[j] += r[j];
            }
            if k < nres {
                input_raw = r;
            }
        }
        (codes, res_sum)
    }
}

/// The **fused single-infer MTP**: the whole per-frame residual prediction runs in
/// ONE NPU inference (vs [`KvMtp`]'s 15 tiny per-substep infers, which are dispatch-
/// bound). The `small_to_mtp_projection`, per-residual `lm_head`/`codec_embedding`,
/// argmax and gather all live inside the graph ([`build_mtp_fused_graph`]) — the host
/// just feeds `talker_hidden`+`cb0_embed` and reads back `codes`+`res_sum`.
///
/// **Correctness vs precision (measured):** the topology is EXACT — on the OV-CPU
/// device (fp32) it is bit-identical to [`crate::gen_kv_mtp::CpuMtp`] (codes match,
/// res_sum max-abs 0.0; see `examples/fused_parity.rs`). It is also faster on the NPU
/// (~203ms/frame hot vs KvMtp's ~232-267ms — one big infer beats 15 tiny ones). BUT
/// the Intel NPU is **fp16-only** (`OPTIMIZATION_CAPABILITIES=[FP16,INT8]`), and doing
/// the greedy **argmax in-graph in fp16** flips near-ties in the 2048-entry codebook,
/// which then **cascades** through the autoregressive residual feedback → degraded
/// audio (spk-cos to reference 0.84 vs KvMtp's 0.98). KvMtp sidesteps this by keeping
/// lm_head+argmax on the **fp32 host**. So `fused` is opt-in and best on a device that
/// can run the head/argmax in fp32 (or with sampling, which is fp16-robust); the
/// default stays KvMtp for greedy decoding on this fp16 NPU.
pub struct FusedMtp {
    sess: FusedMtpSession,
    device: String,
}

impl FusedMtp {
    pub fn load(
        mtp_path: &str,
        device: NpuDevice,
        allow_fallback: bool,
        cache_dir: Option<&Path>,
    ) -> Result<FusedMtp, String> {
        let (emb, nres) = {
            let c = checkpoint::load(mtp_path);
            let cfg = crate::config::MtpConfig::from_brain_json(&c.header["config"]);
            (cfg.embedding_dim as usize, cfg.n_residual() as usize)
        };
        let onnx = prepare_mtp_fused_onnx(mtp_path, cache_dir)?;
        let ncfg = npu_config(device, allow_fallback, cache_dir);
        let sess = FusedMtpSession::load_path(&onnx, &ncfg, emb, nres).map_err(|e| e.to_string())?;
        let device = sess.device().to_string();
        Ok(FusedMtp { sess, device })
    }

    pub fn device(&self) -> &str {
        &self.device
    }
}

impl MtpEngine for FusedMtp {
    fn generate_residuals(&mut self, talker_hidden: &[f32], cb0_embed: &[f32]) -> (Vec<u32>, Vec<f32>) {
        self.sess.run(talker_hidden, cb0_embed).expect("FusedMtp inference")
    }
}

fn prepare_mtp_fused_onnx(mtp_path: &str, cache_dir: Option<&Path>) -> Result<PathBuf, String> {
    prepare_graph(mtp_path, cache_dir, "mtp-fused.onnx", "brain_tts_npu_mtp_fused", |out| {
        npu::qwen_export::export_mtp_fused(mtp_path, out)
    })
}

/// Diagnostic parity gate: max-abs difference between the **NPU** Talker and the
/// **CPU** KV-cache Talker ([`crate::gen_kv::CpuTalker`]) for the final-norm
/// hidden state at the last prefix position. Deterministic (no sampling, no
/// codec), and prefix-only so it returns in seconds. Loads both decoders, so this
/// is for explicit validation, not the hot path.
#[allow(clippy::too_many_arguments)]
pub fn talker_prefix_parity(
    talker_path: &str,
    tables: &TalkerTables,
    prompt: &Prompt,
    cache_dir: Option<&Path>,
    device: NpuDevice,
    quant: bool,
) -> Result<f32, String> {
    let d = tables.d();
    let n_prefix = prompt.embeds.len() / d;
    if n_prefix == 0 {
        return Err("empty prompt prefix".to_string());
    }
    // CPU reference (pure-scalar KV-cache mirror; no gpu_core upload).
    let mut cpu = crate::gen_kv::CpuTalker::load(talker_path);
    cpu.reset();
    let mut cpu_h = vec![0.0f32; d];
    for i in 0..n_prefix {
        cpu_h = cpu.step(&prompt.embeds[i * d..(i + 1) * d]);
    }
    // NPU: compile a prefix-sized graph and stream the prefix in one inference.
    let max_pos = tables.cfg.max_position_embeddings as usize;
    let mut npu = TalkerNpu::load(talker_path, n_prefix, device, true, cache_dir, max_pos, quant)?;
    let npu_h = npu.feed(&prompt.embeds)?;
    let maxabs = cpu_h
        .iter()
        .zip(&npu_h)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    Ok(maxabs)
}

/// Decode `[T,16]` codes (row-major, codebooks 0..15 per frame) to a 24 kHz
/// waveform on the NPU **codec** graph. brain codes are reordered to the codec
/// graph's codebook-major `[nq,T]` int64 input. Returns `(waveform, device)`.
pub fn decode_codes_npu(
    codec_path: &str,
    codes: &[u32],
    device: NpuDevice,
    allow_fallback: bool,
    cache_dir: Option<&Path>,
) -> Result<(Vec<f32>, String), String> {
    let tb = codec_bucket(codes.len() / 16);
    eprintln!("tts npu: compiling codec decoder graph (code_len={tb})…");
    let mut sess = open_codec_session(codec_path, tb, device, allow_fallback, cache_dir)?;
    let dev = sess.device().to_string();
    let wav = decode_with_session(&mut sess, codes)?;
    Ok((wav, dev))
}

/// Codec code-length bucket for `frames` (rounds up to [`CODEC_BUCKET`]). One
/// compiled graph per bucket serves a range of frame counts.
pub fn codec_bucket(frames: usize) -> usize {
    round_up(frames.max(1), CODEC_BUCKET)
}

/// Compile (or reuse, via the blob cache) a codec decoder session for exactly
/// `code_len` frames. The server caches these per bucket so they stay resident.
pub fn open_codec_session(
    codec_path: &str,
    code_len: usize,
    device: NpuDevice,
    allow_fallback: bool,
    cache_dir: Option<&Path>,
) -> Result<CodecSession, String> {
    let onnx = prepare_codec_onnx(codec_path, code_len, cache_dir)?;
    CodecSession::load_path(&onnx, &npu_config(device, allow_fallback, cache_dir)).map_err(|e| e.to_string())
}

/// Decode `[T,16]` row-major codes on a resident codec session whose compiled
/// `code_len` (`>= T`) the caller picked via [`codec_bucket`]. Pads to the
/// session's length and trims the (causal) output back to `T` frames.
pub fn decode_with_session(sess: &mut CodecSession, codes: &[u32]) -> Result<Vec<f32>, String> {
    if codes.is_empty() {
        return Err("no codec frames were generated".to_string());
    }
    let ncode = 16usize;
    if !codes.len().is_multiple_of(ncode) {
        return Err(format!("codes len {} is not a multiple of 16", codes.len()));
    }
    let t = codes.len() / ncode;
    let tb = sess.code_len();
    if t > tb {
        return Err(format!("{t} frames exceed codec session code_len {tb}"));
    }
    let nq = sess.nq();
    if nq != ncode {
        return Err(format!("codec graph expects {nq} codebooks, frames have {ncode}"));
    }
    // [T,16] row-major u32 -> [nq,tb] codebook-major i64 (real frames, zero-padded).
    let mut cm = vec![0i64; nq * tb];
    for f in 0..t {
        for q in 0..ncode {
            cm[q * tb + f] = codes[f * ncode + q] as i64;
        }
    }
    let wav_full = sess.run_codes(&cm).map_err(|e| e.to_string())?;
    let spf = if tb > 0 { sess.out_len() / tb } else { 0 };
    let real = (t * spf).min(wav_full.len());
    Ok(if real > 0 { wav_full[..real].to_vec() } else { wav_full })
}

/// OpenVINO config: target `device`, optional CPU/GPU fallback, and the cache dir
/// (reuses both the ONNX file and OpenVINO's compiled-blob cache).
fn npu_config(device: NpuDevice, allow_fallback: bool, cache_dir: Option<&Path>) -> NpuConfig {
    // `BRAIN_NPU_TURBO=1` requests the NPU's turbo clock (NPU_TURBO=YES). Off by
    // default (it raises power/heat, which can throttle sooner under sustained
    // load) — A/B it against the thermal envelope before making it the default.
    let turbo = std::env::var("BRAIN_NPU_TURBO").map(|v| v == "1" || v == "yes").unwrap_or(false);
    NpuConfig {
        device,
        allow_fallback,
        cache_dir: cache_dir.map(|p| p.to_path_buf()),
        turbo,
        ..Default::default()
    }
}

fn mtime(p: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

/// Cache-aware ONNX export: write `<dir>/<file>` (+ `.data` sidecar) via `export`
/// unless an up-to-date copy (newer than `weights_path`) already exists. Without a
/// `cache_dir` a per-process temp dir is used (kept until process exit so the
/// external-data sidecar stays resolvable while the session is alive).
fn prepare_graph(
    weights_path: &str,
    cache_dir: Option<&Path>,
    file: &str,
    tmp_tag: &str,
    export: impl FnOnce(&str) -> std::io::Result<()>,
) -> Result<PathBuf, String> {
    let map = |e: std::io::Error| e.to_string();
    let dir = match cache_dir {
        Some(cd) => cd.to_path_buf(),
        None => std::env::temp_dir().join(format!("{tmp_tag}_{}", std::process::id())),
    };
    std::fs::create_dir_all(&dir).map_err(map)?;
    let onnx = dir.join(file);
    let data = dir.join(format!("{file}.data"));
    let fresh = onnx.exists()
        && data.exists()
        && match (mtime(&onnx), mtime(Path::new(weights_path))) {
            (Some(o), Some(w)) => o >= w,
            _ => false,
        };
    if !fresh {
        export(onnx.to_str().ok_or("non-utf8 cache path")?).map_err(map)?;
    }
    Ok(onnx)
}

fn prepare_talker_onnx(weights_path: &str, cap: usize, cache_dir: Option<&Path>, quant: bool) -> Result<PathBuf, String> {
    let file = if quant {
        format!("talker-hidden-int8-seq{cap}.onnx")
    } else {
        format!("talker-hidden-seq{cap}.onnx")
    };
    let tag = if quant { "brain_tts_npu_talker_int8" } else { "brain_tts_npu_talker" };
    prepare_graph(weights_path, cache_dir, &file, tag, |out| {
        if quant {
            npu::qwen_export::export_talker_hidden_int8(weights_path, out, cap)
        } else {
            npu::qwen_export::export_talker_hidden_fp32(weights_path, out, cap)
        }
    })
}

fn prepare_prefill_onnx(weights_path: &str, cap: usize, cache_dir: Option<&Path>, quant: bool, int4: bool) -> Result<PathBuf, String> {
    let (file, tag) = if int4 {
        (format!("talker-prefill-int4-seq{cap}.onnx"), "brain_tts_npu_prefill_int4")
    } else if quant {
        (format!("talker-prefill-int8-seq{cap}.onnx"), "brain_tts_npu_prefill_int8")
    } else {
        (format!("talker-prefill-seq{cap}.onnx"), "brain_tts_npu_prefill")
    };
    prepare_graph(weights_path, cache_dir, &file, tag, |out| {
        if int4 {
            npu::qwen_export::export_talker_prefill_int4(weights_path, out, cap)
        } else if quant {
            npu::qwen_export::export_talker_prefill_int8(weights_path, out, cap)
        } else {
            npu::qwen_export::export_talker_prefill_fp32(weights_path, out, cap)
        }
    })
}

fn prepare_mtp_decode_onnx(mtp_path: &str, cap: usize, cache_dir: Option<&Path>, quant: bool) -> Result<PathBuf, String> {
    let file = if quant {
        format!("mtp-decode-int8-seq{cap}.onnx")
    } else {
        format!("mtp-decode-seq{cap}.onnx")
    };
    let tag = if quant { "brain_tts_npu_mtp_int8" } else { "brain_tts_npu_mtp" };
    prepare_graph(mtp_path, cache_dir, &file, tag, |out| {
        if quant {
            npu::qwen_export::export_mtp_decode_int8(mtp_path, out, cap)
        } else {
            npu::qwen_export::export_mtp_decode_fp32(mtp_path, out, cap)
        }
    })
}

fn prepare_decode_onnx(weights_path: &str, cap: usize, cache_dir: Option<&Path>, quant: bool, int4: bool) -> Result<PathBuf, String> {
    let (file, tag) = if int4 {
        (format!("talker-decode-int4-seq{cap}.onnx"), "brain_tts_npu_decode_int4")
    } else if quant {
        (format!("talker-decode-int8-seq{cap}.onnx"), "brain_tts_npu_decode_int8")
    } else {
        (format!("talker-decode-seq{cap}.onnx"), "brain_tts_npu_decode")
    };
    prepare_graph(weights_path, cache_dir, &file, tag, |out| {
        if int4 {
            npu::qwen_export::export_talker_decode_int4(weights_path, out, cap)
        } else if quant {
            npu::qwen_export::export_talker_decode_int8(weights_path, out, cap)
        } else {
            npu::qwen_export::export_talker_decode_fp32(weights_path, out, cap)
        }
    })
}

/// Autoregressive generation with the **KV-cache decode** Talker: the prefix is
/// streamed token-by-token into the cache, then each frame decodes one token.
/// Same sampling + MTP feedback as [`generate_codes_npu`]; only the Talker decoder
/// differs (resident cache vs cache-free recompute).
pub fn generate_codes_kv(
    kv: &mut KvTalker,
    tables: &TalkerTables,
    mtp: &mut dyn MtpEngine,
    sp: &TtsSpecials,
    prompt: &Prompt,
    opts: &GenOpts,
) -> Result<Vec<u32>, String> {
    use std::time::Instant;
    let d = tables.d();
    let n_trailing = prompt.trailing.len() / d;
    let n_prefix = prompt.embeds.len() / d;
    let mut rng = Rng::new(opts.seed);
    let profile = std::env::var("TTS_PROFILE").is_ok();

    kv.reset();
    let tp0 = Instant::now();
    // Seed the cache for the whole prefix in one prefill inference.
    let mut past_hidden = kv.prefill_prompt(&prompt.embeds)?;
    let t_prefix = tp0.elapsed().as_secs_f64() * 1e3;
    let mut cb0 = sample_cb0(
        tables.codec_head_logits(&past_hidden),
        sp.codec_eos,
        opts.min_new == 0,
        opts.temperature,
        opts.top_k,
        &mut rng,
    );

    let mut frames: Vec<u32> = Vec::new();
    let mut s = 0usize;
    let (mut t_step, mut t_mtp) = (0.0f64, 0.0f64);
    loop {
        if (cb0 == sp.codec_eos && s >= opts.min_new) || s >= opts.max_frames {
            break;
        }
        let cb0_embed = tables.codec_embed(cb0).to_vec();
        let tm = Instant::now();
        let (residuals, res_sum) = mtp.generate_residuals(&past_hidden, &cb0_embed);
        t_mtp += tm.elapsed().as_secs_f64() * 1e3;
        frames.push(cb0);
        frames.extend_from_slice(&residuals);

        let mut feed = cb0_embed;
        add_into(&mut feed, &res_sum);
        if s < n_trailing {
            add_into(&mut feed, &prompt.trailing[s * d..(s + 1) * d]);
        } else {
            add_into(&mut feed, &prompt.tts_pad);
        }
        s += 1;
        if kv.pos() >= kv.max_pos() || kv.pos() >= kv.cap() {
            break;
        }
        let ts = Instant::now();
        past_hidden = kv.feed1(&feed)?;
        t_step += ts.elapsed().as_secs_f64() * 1e3;
        cb0 = sample_cb0(
            tables.codec_head_logits(&past_hidden),
            sp.codec_eos,
            s >= opts.min_new,
            opts.temperature,
            opts.top_k,
            &mut rng,
        );
    }
    if profile {
        let nf = s.max(1) as f64;
        eprintln!(
            "[tts-npu-profile] KV: prefix({n_prefix} tok)={t_prefix:.0}ms | talker-step total={t_step:.0}ms \
             ({:.0}ms/frame) | mtp total={t_mtp:.0}ms ({:.0}ms/frame) | frames={s}",
            t_step / nf,
            t_mtp / nf,
        );
    }
    Ok(frames)
}

/// Streaming variant of [`generate_codes_kv`]: identical generation + sampling,
/// but invokes `on_chunk(all_codes_so_far)` every `chunk` frames (and once more
/// for the final remainder) so the caller can decode and emit audio progressively
/// instead of waiting for the whole clip. Returns the full codes.
pub fn generate_codes_kv_streaming(
    kv: &mut KvTalker,
    tables: &TalkerTables,
    mtp: &mut dyn MtpEngine,
    sp: &TtsSpecials,
    prompt: &Prompt,
    opts: &GenOpts,
    chunk: usize,
    on_chunk: &mut dyn FnMut(&[u32]),
) -> Result<Vec<u32>, String> {
    let d = tables.d();
    let n_trailing = prompt.trailing.len() / d;
    let mut rng = Rng::new(opts.seed);
    let chunk = chunk.max(1);

    kv.reset();
    let mut past_hidden = kv.prefill_prompt(&prompt.embeds)?;
    let mut cb0 = sample_cb0(
        tables.codec_head_logits(&past_hidden),
        sp.codec_eos,
        opts.min_new == 0,
        opts.temperature,
        opts.top_k,
        &mut rng,
    );

    let mut frames: Vec<u32> = Vec::new();
    let mut s = 0usize;
    let mut emitted = 0usize;
    loop {
        if (cb0 == sp.codec_eos && s >= opts.min_new) || s >= opts.max_frames {
            break;
        }
        let cb0_embed = tables.codec_embed(cb0).to_vec();
        let (residuals, res_sum) = mtp.generate_residuals(&past_hidden, &cb0_embed);
        frames.push(cb0);
        frames.extend_from_slice(&residuals);
        let mut feed = cb0_embed;
        add_into(&mut feed, &res_sum);
        if s < n_trailing {
            add_into(&mut feed, &prompt.trailing[s * d..(s + 1) * d]);
        } else {
            add_into(&mut feed, &prompt.tts_pad);
        }
        s += 1;
        if s - emitted >= chunk {
            on_chunk(&frames);
            emitted = s;
        }
        if kv.pos() >= kv.max_pos() || kv.pos() >= kv.cap() {
            break;
        }
        past_hidden = kv.feed1(&feed)?;
        cb0 = sample_cb0(
            tables.codec_head_logits(&past_hidden),
            sp.codec_eos,
            s >= opts.min_new,
            opts.temperature,
            opts.top_k,
            &mut rng,
        );
    }
    if s > emitted {
        on_chunk(&frames);
    }
    Ok(frames)
}

/// One-shot: size + compile the decode-step graph for this prompt, then run the
/// KV-cache generation loop.
#[allow(clippy::too_many_arguments)]
pub fn generate_kv(
    talker_path: &str,
    tables: &TalkerTables,
    mtp: &mut dyn MtpEngine,
    sp: &TtsSpecials,
    prompt: &Prompt,
    opts: &GenOpts,
    device: NpuDevice,
    allow_fallback: bool,
    cache_dir: Option<&Path>,
    quant: bool,
    int4: bool,
) -> Result<Vec<u32>, String> {
    let d = tables.d();
    let n_prefix = prompt.embeds.len() / d;
    let max_pos = tables.cfg.max_position_embeddings as usize;
    let cap = round_up(n_prefix + opts.max_frames + 2, CAP_BUCKET).min(max_pos);
    eprintln!(
        "tts npu: compiling KV-cache decode graph ({} | cap={cap}); first compile per cap is slow, cached after…",
        if int4 { "INT4" } else if quant { "INT8" } else { "fp32" }
    );
    // INT4: skip the companion prefill graph — compiling BOTH i4 graphs peaks ~15GB
    // and OOMs on a memory-pressured box. With only the decode graph resident the
    // compile fits; the prefix seeds token-by-token (slower once, same result).
    let with_prefill = !int4;
    let mut kv = KvTalker::load(talker_path, cap, device, allow_fallback, cache_dir, &tables.cfg, quant, int4, with_prefill)?;
    eprintln!("tts npu: Talker (KV-cache decode) running on {}", kv.device());
    generate_codes_kv(&mut kv, tables, mtp, sp, prompt, opts)
}

/// Diagnostic: max-abs difference between the **KV-cache decode** Talker and the
/// CPU KV-cache reference ([`crate::gen_kv::CpuTalker`]) for the final-norm hidden
/// at the last prefix position. Validates the resident-cache graph's correctness
/// independent of the device (use `device = Cpu` for an exact fp32 check).
pub fn kv_prefix_parity(
    talker_path: &str,
    tables: &TalkerTables,
    prompt: &Prompt,
    cache_dir: Option<&Path>,
    device: NpuDevice,
    quant: bool,
) -> Result<f32, String> {
    let d = tables.d();
    let n_prefix = prompt.embeds.len() / d;
    if n_prefix == 0 {
        return Err("empty prompt prefix".to_string());
    }
    let mut cpu = crate::gen_kv::CpuTalker::load(talker_path);
    cpu.reset();
    let mut cpu_h = vec![0.0f32; d];
    for i in 0..n_prefix {
        cpu_h = cpu.step(&prompt.embeds[i * d..(i + 1) * d]);
    }
    let max_pos = tables.cfg.max_position_embeddings as usize;
    let cap = round_up(n_prefix + 2, CAP_BUCKET).min(max_pos);
    let mut kv = KvTalker::load(talker_path, cap, device, true, cache_dir, &tables.cfg, quant, false, true)?;
    // Validate the one-shot prefill path (what generation uses) against the CPU.
    let kv_h = kv.prefill_prompt(&prompt.embeds)?;
    let maxabs = cpu_h
        .iter()
        .zip(&kv_h)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    Ok(maxabs)
}

fn prepare_codec_onnx(weights_path: &str, code_len: usize, cache_dir: Option<&Path>) -> Result<PathBuf, String> {
    prepare_graph(
        weights_path,
        cache_dir,
        &format!("codec-clen{code_len}.onnx"),
        "brain_tts_npu_codec",
        |out| npu::codec_export::export_codec_fp32(weights_path, out, code_len),
    )
}

fn prepare_codec_front_onnx(weights_path: &str, t: usize, cache_dir: Option<&Path>) -> Result<PathBuf, String> {
    prepare_graph(
        weights_path,
        cache_dir,
        &format!("codec-front-t{t}.onnx"),
        "brain_tts_npu_codec_front",
        |out| npu::codec_export::export_codec_front_fp32(weights_path, out, t).map(|_| ()),
    )
}

/// The stateful streaming codec on the NPU: the causal **front** graph
/// (`codes -> latent`, run once, reusing [`CodecSession`]) plus the
/// **streaming-back** graph driven chunk-by-chunk with the per-conv state
/// carried on the host. Each chunk decodes only its new frames — no warmup
/// re-decode (cf. the windowed [`decode_codes_npu`]).
pub struct NpuStreamCodec {
    front: CodecSession,
    back: BackStreamSession,
    bufs: Vec<Vec<f32>>,
    latent_dim: usize,
    front_t: usize,
    chunk: usize,
    nq: usize,
    device: String,
}

impl NpuStreamCodec {
    pub fn load(
        codec_path: &str,
        front_t: usize,
        chunk: usize,
        device: NpuDevice,
        allow_fallback: bool,
        cache_dir: Option<&Path>,
    ) -> Result<NpuStreamCodec, String> {
        let ncfg = npu_config(device, allow_fallback, cache_dir);
        let fonnx = prepare_codec_front_onnx(codec_path, front_t, cache_dir)?;
        let front = CodecSession::load_path(&fonnx, &ncfg).map_err(|e| e.to_string())?;
        let nq = front.nq();
        let latent_dim = front.out_len() / front_t;
        // The streaming-back graph: always (re)export to recover the buffer specs;
        // the per-bucket file lives in the cache and OpenVINO blob-caches the compile.
        let dir = cache_dir.map(|p| p.to_path_buf()).unwrap_or_else(std::env::temp_dir);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let bpath = dir.join(format!("codec-back-stream-chunk{chunk}.onnx"));
        let (_, specs) = npu::codec_export::export_codec_back_stream_fp32(codec_path, bpath.to_str().ok_or("path")?, chunk)
            .map_err(|e| e.to_string())?;
        let back = BackStreamSession::load_path(&bpath, &ncfg, specs, latent_dim, chunk).map_err(|e| e.to_string())?;
        let bufs = back.zero_buffers();
        let device = back.device().to_string();
        Ok(NpuStreamCodec { front, back, bufs, latent_dim, front_t, chunk, nq, device })
    }

    pub fn device(&self) -> &str {
        &self.device
    }
    pub fn front_t(&self) -> usize {
        self.front_t
    }

    /// Decode `[T,16]` row-major codes, streaming each chunk's audio via
    /// `on_audio(samples, seq)`. `T` must be `<= front_t` (the compiled front).
    pub fn decode(&mut self, codes_rowmajor: &[u32], on_audio: &mut dyn FnMut(&[f32], u32)) -> Result<usize, String> {
        let nq = self.nq;
        let t = codes_rowmajor.len() / nq;
        if t == 0 {
            return Err("no codec frames".into());
        }
        if t > self.front_t {
            return Err(format!("{t} frames exceed front graph length {}", self.front_t));
        }
        // Front (one infer): codes [T,nq] row-major -> [nq,front_t] i64 (zero-padded).
        let mut cm = vec![0i64; nq * self.front_t];
        for f in 0..t {
            for q in 0..nq {
                cm[q * self.front_t + f] = codes_rowmajor[f * nq + q] as i64;
            }
        }
        let latent = self.front.run_codes(&cm).map_err(|e| e.to_string())?; // [latent_dim, front_t] NCL
        for b in self.bufs.iter_mut() {
            b.iter_mut().for_each(|x| *x = 0.0);
        }
        let (ld, ft, ch) = (self.latent_dim, self.front_t, self.chunk);
        let mut seq = 0u32;
        let mut total = 0usize;
        let mut a = 0usize;
        while a < t {
            let b = (a + ch).min(t);
            let l_new = b - a;
            // Slice latent cols [a,b) per channel into a chunk-wide slab (zero-pad).
            let mut slab = vec![0.0f32; ld * ch];
            for c in 0..ld {
                slab[c * ch..c * ch + l_new].copy_from_slice(&latent[c * ft + a..c * ft + b]);
            }
            let (wav, nb) = self.back.run(&slab, &self.bufs).map_err(|e| e.to_string())?;
            self.bufs = nb;
            let spf = wav.len() / ch.max(1);
            let take = l_new * spf;
            on_audio(&wav[..take.min(wav.len())], seq);
            seq += 1;
            total += take;
            a = b;
        }
        Ok(total)
    }
}

#[cfg(test)]
mod stream_codec_tests {
    use super::*;

    /// NPU stateful streaming codec vs the bit-exact CPU reference. Run (CPU dev):
    ///   BRAIN_CODEC_WEIGHTS=.../codec.safetensors BRAIN_TTS_NPU_DEVICE=cpu \
    ///   cargo test --release -p brain-tts npu_stream_matches_cpu -- --ignored --nocapture
    #[test]
    #[ignore]
    fn npu_stream_matches_cpu() {
        let path = std::env::var("BRAIN_CODEC_WEIGHTS").expect("set BRAIN_CODEC_WEIGHTS");
        let device = std::env::var("BRAIN_TTS_NPU_DEVICE")
            .ok()
            .and_then(|s| NpuDevice::parse(&s))
            .unwrap_or(NpuDevice::Cpu);
        let cache = std::path::Path::new("out/tts-1b7/npu-cache");
        let (front_t, chunk) = (32usize, 16usize);
        let mut npu = NpuStreamCodec::load(&path, front_t, chunk, device, true, Some(cache)).unwrap();

        let nq = 16usize;
        let t = 24usize;
        // The unified deterministic LCG (audit F39/F40).
        let mut lcg = data::rng::Lcg::new(11);
        let codes: Vec<u32> = (0..t * nq).map(|_| lcg.next_u32() % 64).collect();

        let mut npu_wav = Vec::new();
        npu.decode(&codes, &mut |pcm, _| npu_wav.extend_from_slice(pcm)).unwrap();

        let cpu = codec::decode_stream::StreamingCodecDecoder::load(&path);
        let cpu_wav = cpu.decode_streaming(&codes, chunk);

        let n = npu_wav.len().min(cpu_wav.len());
        let maxd = npu_wav[..n].iter().zip(&cpu_wav[..n]).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("npu-stream vs cpu-stream: len {} vs {}, max-abs {maxd:.3e}", npu_wav.len(), cpu_wav.len());
        assert_eq!(npu_wav.len(), cpu_wav.len(), "length mismatch");
        assert!(maxd < 5e-3, "npu stream codec differs: {maxd}");
    }
}
