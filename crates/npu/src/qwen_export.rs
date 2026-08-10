// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export a brain Qwen `.safetensors` checkpoint to an ONNX decoder graph (fixed
//! sequence length) for OpenVINO whole-graph compilation. Pure Rust — no NPU
//! needed to produce the file.

use onnx::builder::GraphBuilder;
use qwen3::config::QwenConfig;

/// Build the fp32 ONNX decoder for `seq_len` and return `(bytes, config)`.
pub fn build_qwen_fp32_bytes(weights_path: &str, seq_len: usize) -> std::io::Result<(Vec<u8>, QwenConfig)> {
    let reader = checkpoint::weightio::WeightReader::open(weights_path)?;
    let cfg = QwenConfig::from_json(&reader.config());
    let mut g = GraphBuilder::new("qwen_decoder");
    crate::qwen_topology::build_qwen_graph(&cfg, &reader, seq_len, &mut g);
    Ok((g.finish(), cfg))
}

/// Build the fp32 ONNX **Talker** decoder for `seq_len` and return
/// `(bytes, config)`. The Qwen3-TTS Talker is byte-for-byte a Qwen3 decoder with
/// an *untied* codec head (`tie_embeddings = false`, so a separate
/// `lm_head.weight`), exported by the same [`build_qwen_fp32_bytes`] path — the
/// `text_projection`/`text_embedding` tensors that ride along in the Talker
/// container are simply unused by the decoder graph. Provided as a named entry
/// point so callers reach for it by intent; the input is `input_ids` (codec
/// token ids) and the output is the codebook-0 `logits`.
pub fn build_talker_fp32_bytes(weights_path: &str, seq_len: usize) -> std::io::Result<(Vec<u8>, QwenConfig)> {
    build_qwen_fp32_bytes(weights_path, seq_len)
}

/// Export the fp32 ONNX Talker decoder to `out_path` (+ sidecar).
pub fn export_talker_fp32(weights_path: &str, out_path: &str, seq_len: usize) -> std::io::Result<()> {
    export_qwen_fp32(weights_path, out_path, seq_len)
}

/// Build the fp32 ONNX **Talker hidden-state** graph for `seq_len` and return
/// `(bytes, config)`. Unlike [`build_talker_fp32_bytes`] (token-id → logits),
/// this is the input-embedding-driven graph the autoregressive Talker loop needs:
/// input `inputs_embeds:[1,seq_len,d]` (f32), output `hidden:[1,seq_len,d]` (f32,
/// post-final-norm). The codebook-0 head and MTP residual fill stay on the host.
/// See [`crate::qwen_topology::build_talker_hidden_graph`].
pub fn build_talker_hidden_fp32_bytes(weights_path: &str, seq_len: usize) -> std::io::Result<(Vec<u8>, QwenConfig)> {
    talker_hidden_bytes(weights_path, seq_len, false)
}

/// As [`build_talker_hidden_fp32_bytes`] but weight-only **INT8** (per-output-
/// channel symmetric, `DequantizeLinear` -> MatMul): ~4x smaller, so the 1.7B
/// Talker fits the NPU and compiles faster.
pub fn build_talker_hidden_int8_bytes(weights_path: &str, seq_len: usize) -> std::io::Result<(Vec<u8>, QwenConfig)> {
    talker_hidden_bytes(weights_path, seq_len, true)
}

fn talker_hidden_bytes(weights_path: &str, seq_len: usize, quant: bool) -> std::io::Result<(Vec<u8>, QwenConfig)> {
    let reader = checkpoint::weightio::WeightReader::open(weights_path)?;
    let cfg = QwenConfig::from_json(&reader.config());
    let mut g = GraphBuilder::new("qwen_talker_hidden");
    crate::qwen_topology::build_talker_hidden_graph(&cfg, &reader, seq_len, quant, &mut g);
    Ok((g.finish(), cfg))
}

/// Export the fp32 ONNX Talker hidden-state graph to `out_path` (+ a
/// `<out_path>.data` sidecar for the large decoder weights).
pub fn export_talker_hidden_fp32(weights_path: &str, out_path: &str, seq_len: usize) -> std::io::Result<()> {
    export_talker_hidden(weights_path, out_path, seq_len, false)
}

/// Export the weight-only **INT8** Talker hidden-state graph to `out_path`.
pub fn export_talker_hidden_int8(weights_path: &str, out_path: &str, seq_len: usize) -> std::io::Result<()> {
    export_talker_hidden(weights_path, out_path, seq_len, true)
}

fn export_talker_hidden(weights_path: &str, out_path: &str, seq_len: usize, quant: bool) -> std::io::Result<()> {
    let reader = checkpoint::weightio::WeightReader::open(weights_path)?;
    let cfg = QwenConfig::from_json(&reader.config());
    let mut g = GraphBuilder::new("qwen_talker_hidden");
    crate::qwen_topology::build_talker_hidden_graph(&cfg, &reader, seq_len, quant, &mut g);
    g.finish_external(out_path, EXTERNAL_THRESHOLD)
}

/// Export the **KV-cache decode-step** Talker graph (one token + per-layer
/// past/present K/V) for `cap` cache slots; `quant` selects weight-only INT8.
pub fn export_talker_decode_fp32(weights_path: &str, out_path: &str, cap: usize) -> std::io::Result<()> {
    export_talker_decode(weights_path, out_path, cap, crate::qwen_topology::Quant::F32)
}

/// INT8 weight-only variant of [`export_talker_decode_fp32`].
pub fn export_talker_decode_int8(weights_path: &str, out_path: &str, cap: usize) -> std::io::Result<()> {
    export_talker_decode(weights_path, out_path, cap, crate::qwen_topology::Quant::Int8)
}

/// INT4 weight-only variant (~8x smaller than fp32; weight-bandwidth-bound decode
/// runs faster and RAM roughly halves vs INT8). Lossier than INT8 — validate quality.
pub fn export_talker_decode_int4(weights_path: &str, out_path: &str, cap: usize) -> std::io::Result<()> {
    export_talker_decode(weights_path, out_path, cap, crate::qwen_topology::Quant::Int4)
}

fn export_talker_decode(weights_path: &str, out_path: &str, cap: usize, quant: crate::qwen_topology::Quant) -> std::io::Result<()> {
    let reader = checkpoint::weightio::WeightReader::open(weights_path)?;
    let cfg = QwenConfig::from_json(&reader.config());
    let mut g = GraphBuilder::new("qwen_talker_decode");
    crate::qwen_topology::build_talker_decode_graph(&cfg, &reader, cap, quant, &mut g);
    finish_quant(&g, out_path, quant)
}

/// Serialize `g`: INT4 needs ONNX opset 21 / IR 10 (the version that introduced the
/// 4-bit tensor types + `DequantizeLinear`-int4); INT8/fp32 use the default (13/8).
fn finish_quant(g: &GraphBuilder, out_path: &str, quant: crate::qwen_topology::Quant) -> std::io::Result<()> {
    if quant == crate::qwen_topology::Quant::Int4 {
        g.finish_external_with(out_path, EXTERNAL_THRESHOLD, 21, 10)
    } else {
        g.finish_external(out_path, EXTERNAL_THRESHOLD)
    }
}

/// Export the **MTP decode-step** graph: the MTP's 5-layer Qwen3 decoder is the
/// same block as the Talker, so it reuses [`crate::qwen_topology::build_talker_decode_graph`]
/// driven by the MTP's dims (from the brain MTP checkpoint header). Input `x` is
/// the already-projected residual embedding `[1,1,d_mtp]`; the host keeps the
/// `small_to_mtp_projection`, per-residual `codec_embedding` and `lm_head` tables.
/// `cap` is `num_code_groups` (the MTP's 16-position sequence). `quant` => INT8.
pub fn export_mtp_decode_fp32(mtp_path: &str, out_path: &str, cap: usize) -> std::io::Result<()> {
    export_mtp_decode(mtp_path, out_path, cap, false)
}

/// INT8 weight-only variant of [`export_mtp_decode_fp32`].
pub fn export_mtp_decode_int8(mtp_path: &str, out_path: &str, cap: usize) -> std::io::Result<()> {
    export_mtp_decode(mtp_path, out_path, cap, true)
}

fn export_mtp_decode(mtp_path: &str, out_path: &str, cap: usize, quant: bool) -> std::io::Result<()> {
    let reader = checkpoint::weightio::WeightReader::open(mtp_path)?;
    let h = reader.config();
    let gu = |k: &str, d: u32| h[k].as_u64().map(|x| x as u32).unwrap_or(d);
    let gf = |k: &str, d: f32| h[k].as_f64().map(|x| x as f32).unwrap_or(d);
    // The MTP header (MtpConfig::to_json) keys map to the decoder QwenConfig.
    let cfg = QwenConfig {
        vocab: 0,
        block_size: 0,
        n_layers: gu("n_layers", 5),
        d_model: gu("d_model", 1024),
        n_heads: gu("n_heads", 16),
        n_kv_heads: gu("n_kv_heads", 8),
        head_dim: gu("head_dim", 128),
        d_ff: gu("d_ff", 3072),
        rope_theta: gf("rope_theta", 1_000_000.0),
        rms_eps: gf("rms_norm_eps", 1e-6),
        max_position_embeddings: 0,
        tie_embeddings: false,
        qk_norm: true,
        attn_bias: false,
        lora: None,
    };
    let mut g = GraphBuilder::new("qwen_mtp_decode");
    crate::qwen_topology::build_talker_decode_graph(&cfg, &reader, cap, crate::qwen_topology::Quant::from_bool(quant), &mut g);
    g.finish_external(out_path, EXTERNAL_THRESHOLD)
}

/// Export the **fused single-infer MTP** graph (see
/// [`crate::qwen_topology::build_mtp_fused_graph`]): the whole per-frame residual
/// prediction (16 substeps) in ONE inference. Inputs `talker_hidden` + `cb0_embed`,
/// outputs `codes` (f32, host rounds) + `res_sum`. fp32 weights.
pub fn export_mtp_fused(mtp_path: &str, out_path: &str) -> std::io::Result<()> {
    let reader = checkpoint::weightio::WeightReader::open(mtp_path)?;
    let h = reader.config();
    let gu = |k: &str, d: u32| h[k].as_u64().map(|x| x as u32).unwrap_or(d);
    let gf = |k: &str, d: f32| h[k].as_f64().map(|x| x as f32).unwrap_or(d);
    let cfg = QwenConfig {
        vocab: 0,
        block_size: 0,
        n_layers: gu("n_layers", 5),
        d_model: gu("d_model", 1024),
        n_heads: gu("n_heads", 16),
        n_kv_heads: gu("n_kv_heads", 8),
        head_dim: gu("head_dim", 128),
        d_ff: gu("d_ff", 3072),
        rope_theta: gf("rope_theta", 1_000_000.0),
        rms_eps: gf("rms_norm_eps", 1e-6),
        max_position_embeddings: 0,
        tie_embeddings: false,
        qk_norm: true,
        attn_bias: false,
        lora: None,
    };
    let emb = gu("embedding_dim", gu("d_model", 1024)) as usize;
    let vocab = gu("vocab_size", 2048) as usize;
    let n_groups = gu("num_code_groups", 16) as usize;
    let mut g = GraphBuilder::new("qwen_mtp_fused");
    crate::qwen_topology::build_mtp_fused_graph(&cfg, emb, vocab, n_groups, &reader, &mut g);
    g.finish_external(out_path, EXTERNAL_THRESHOLD)
}

/// Export the **prefill** Talker graph (full context -> hidden + per-layer K/V) to
/// seed the decode KV cache in one inference. `quant` selects weight-only INT8.
pub fn export_talker_prefill_fp32(weights_path: &str, out_path: &str, seq_len: usize) -> std::io::Result<()> {
    export_talker_prefill(weights_path, out_path, seq_len, crate::qwen_topology::Quant::F32)
}

/// INT8 weight-only variant of [`export_talker_prefill_fp32`].
pub fn export_talker_prefill_int8(weights_path: &str, out_path: &str, seq_len: usize) -> std::io::Result<()> {
    export_talker_prefill(weights_path, out_path, seq_len, crate::qwen_topology::Quant::Int8)
}

/// INT4 weight-only variant of [`export_talker_prefill_fp32`] (pairs with the INT4
/// decode graph so the prefill-seeded cache and the decode steps use matching weights).
pub fn export_talker_prefill_int4(weights_path: &str, out_path: &str, seq_len: usize) -> std::io::Result<()> {
    export_talker_prefill(weights_path, out_path, seq_len, crate::qwen_topology::Quant::Int4)
}

fn export_talker_prefill(weights_path: &str, out_path: &str, seq_len: usize, quant: crate::qwen_topology::Quant) -> std::io::Result<()> {
    let reader = checkpoint::weightio::WeightReader::open(weights_path)?;
    let cfg = QwenConfig::from_json(&reader.config());
    let mut g = GraphBuilder::new("qwen_talker_prefill");
    crate::qwen_topology::build_talker_prefill_graph(&cfg, &reader, seq_len, quant, &mut g);
    finish_quant(&g, out_path, quant)
}

/// Bytes larger than this go to the ONNX external-data sidecar (keeps the proto
/// under protobuf's 2GB parse limit while inlining the small tensors).
const EXTERNAL_THRESHOLD: usize = 1 << 20; // 1 MiB

/// Export the fp32 ONNX decoder to `out_path` (+ a `<out_path>.data` sidecar for
/// large weights). The pair is read back with a file-based OpenVINO loader.
pub fn export_qwen_fp32(weights_path: &str, out_path: &str, seq_len: usize) -> std::io::Result<()> {
    let reader = checkpoint::weightio::WeightReader::open(weights_path)?;
    let cfg = QwenConfig::from_json(&reader.config());
    let mut g = GraphBuilder::new("qwen_decoder");
    crate::qwen_topology::build_qwen_graph(&cfg, &reader, seq_len, &mut g);
    g.finish_external(out_path, EXTERNAL_THRESHOLD)
}
