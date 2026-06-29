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
use npu::openvino::{CodecSession, EmbedSession, NpuConfig, NpuDevice};

use crate::config::TalkerConfig;
use crate::pipeline::{sample_cb0, GenOpts};
use crate::prompt::{Prompt, TalkerHost, TtsSpecials};
use crate::talker::TextProjection;

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
        let qcfg = qwen::QwenConfig::from_json(&c.header["config"]);
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
        let mut out = vec![0.0f32; v];
        for (o, dst) in out.iter_mut().enumerate() {
            let wrow = &self.codec_head[o * d..(o + 1) * d];
            *dst = wrow.iter().zip(hidden_row).map(|(a, b)| a * b).sum();
        }
        out
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
        assert!(!embeds.is_empty() && embeds.len() % self.d == 0, "feed must be a whole number of [d] rows");
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
    mtp: &mut crate::gen_kv_mtp::CpuMtp,
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
    mtp: &mut crate::gen_kv_mtp::CpuMtp,
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
    if codes.is_empty() {
        return Err("no codec frames were generated".to_string());
    }
    let ncode = 16usize;
    if codes.len() % ncode != 0 {
        return Err(format!("codes len {} is not a multiple of 16", codes.len()));
    }
    let t = codes.len() / ncode;
    // Bucket the codec code length so one compiled graph serves a range of frame
    // counts. The codec is causal, so decoding `tb >= t` frames (zero-padded) and
    // trimming back to `t` frames is exact for the real region.
    let tb = round_up(t, CODEC_BUCKET);
    let onnx = prepare_codec_onnx(codec_path, tb, cache_dir)?;
    let cfg = npu_config(device, allow_fallback, cache_dir);
    eprintln!("tts npu: compiling codec decoder graph (code_len={tb} for {t} frames)…");
    let mut sess = CodecSession::load_path(&onnx, &cfg).map_err(|e| e.to_string())?;
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
    // Trim to the real frames (samples-per-frame = total upsampled length / tb).
    let spf = if tb > 0 { sess.out_len() / tb } else { 0 };
    let real = (t * spf).min(wav_full.len());
    let wav = if real > 0 { wav_full[..real].to_vec() } else { wav_full };
    Ok((wav, sess.device().to_string()))
}

/// OpenVINO config: target `device`, optional CPU/GPU fallback, and the cache dir
/// (reuses both the ONNX file and OpenVINO's compiled-blob cache).
fn npu_config(device: NpuDevice, allow_fallback: bool, cache_dir: Option<&Path>) -> NpuConfig {
    NpuConfig {
        device,
        allow_fallback,
        cache_dir: cache_dir.map(|p| p.to_path_buf()),
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

fn prepare_codec_onnx(weights_path: &str, code_len: usize, cache_dir: Option<&Path>) -> Result<PathBuf, String> {
    prepare_graph(
        weights_path,
        cache_dir,
        &format!("codec-clen{code_len}.onnx"),
        "brain_tts_npu_codec",
        |out| npu::codec_export::export_codec_fp32(weights_path, out, code_len),
    )
}
