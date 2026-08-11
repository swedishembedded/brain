// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ONNX export of omni's three NPU-eligible pieces — audio tower, vision
//! tower (single-merger path), Code2Wav vocoder — as OpenVINO-compilable fp32
//! graphs. Pure Rust; no NPU/OpenVINO needed to *produce* a graph (mirrors
//! every other `*_export.rs` in `crates/npu`). Validation stops at CPU-side
//! structural checks + (where an OpenVINO CPU device is available) numeric
//! parity against the real host forward — there is no NPU device on this box
//! to run against.
//!
//! Each export builds its weight map with the SAME loader `crate::mm`/
//! `crate::codec_bridge` already use for the real GPU/host forward
//! (`mm::audio_weights`, `mm::vision_weights`, `codec_bridge::codec_weights`,
//! `codec_bridge::codec_config_from`) — not a second parallel loader — so a
//! naming-convention drift between the host path and the NPU path cannot
//! silently happen.
//!
//! **Not exported here**: the Thinker MoE decoder. 30B of mostly-expert
//! parameters is not an NPU target on this generation of hardware (the
//! top-level plan's M15 note); only the three encoder/vocoder pieces are.
//! Vision export is also scoped to the single (main) `PatchMerger` — see
//! `crates/npu/src/qwenvl_topology.rs`'s module doc for why DeepStack is out
//! of scope.

use checkpoint::weightio::WeightReader;
use npu::{QwenAsrTopo, VitTopo};
use onnx::GraphBuilder;
use qwen_asr::config::AudioEncoderConfig;
use qwenvl::config::VisionConfig;

use crate::config::Code2WavConfig;

/// Bytes larger than this go to the ONNX external-data sidecar (matches every
/// other `*_export.rs` in `crates/npu`).
const EXTERNAL_THRESHOLD: usize = 1 << 20; // 1 MiB

/// Export the audio-tower (AuT) head — `n_layers`× windowed ViT block +
/// `ln_post` + multi-modal projector — for a fixed `n_audio` packed-token
/// count, one attention window (`spans = [(0, n_audio)]`; a real multi-second
/// clip windows within `n_window_infer`-sized chunks, but a single span
/// covering the whole export length is what a short clip actually produces,
/// and is sufficient to validate the graph structurally / numerically).
/// Writes `out_path` (+ a `.data` sidecar for weights over 1 MiB).
pub fn export_audio_onnx(reader: &WeightReader, out_path: &str, n_audio: u32) -> std::io::Result<()> {
    let cfg = AudioEncoderConfig::qwen3_omni();
    let weights = crate::mm::audio_weights(reader).map_err(std::io::Error::other)?;
    let topo = QwenAsrTopo {
        d_model: cfg.d_model,
        n_heads: cfg.n_heads,
        head_dim: cfg.d_model / cfg.n_heads,
        ffn_dim: cfg.ffn_dim,
        n_layers: cfg.n_layers,
        output_dim: cfg.output_dim,
        eps: cfg.eps,
    };
    let mut g = GraphBuilder::new("omni_audio_tower");
    g.input_f32("packed_tokens", &[n_audio as i64, topo.d_model as i64]);
    g.output_f32("audio_embeds", &[n_audio as i64, topo.output_dim as i64]);
    npu::build_qwen_asr_head(&mut g, &topo, &weights, n_audio, &[(0, n_audio)], "packed_tokens", "audio_embeds");
    g.finish_external(out_path, EXTERNAL_THRESHOLD)
}

/// Export the vision-tower head — patch-embed + `depth`× ViT block + main
/// `PatchMerger` — for a fixed `grid_h × grid_w` patch grid (must be
/// multiples of `spatial_merge_size`, e.g. 16×16 for a modest single-image
/// export). Writes `out_path` (+ `.data` sidecar).
pub fn export_vision_onnx(reader: &WeightReader, out_path: &str, grid_h: u32, grid_w: u32) -> std::io::Result<()> {
    let cfg = VisionConfig::qwen3_omni();
    let (encoder_w, merger_w) = crate::mm::vision_weights(reader).map_err(std::io::Error::other)?;
    let mut weights = encoder_w;
    for (k, v) in merger_w {
        weights.insert(format!("merger.{k}"), v);
    }
    let topo = VitTopo {
        depth: cfg.depth,
        hidden: cfg.hidden,
        num_heads: cfg.num_heads,
        intermediate: cfg.intermediate,
        out_hidden: cfg.out_hidden_size,
        merge: cfg.spatial_merge_size,
        eps: 1e-6, // `qwenvl::encoder::VISION_EPS` — not itself a config field
        rope_theta: 10000.0, // `qwenvl::encoder::VISION_ROPE_THETA`
    };
    let n = grid_h * grid_w;
    let mrows = n / (cfg.spatial_merge_size * cfg.spatial_merge_size);
    let mut g = GraphBuilder::new("omni_vision_tower");
    g.input_f32("pixels", &[n as i64, cfg.patch_vec_dim() as i64]);
    g.output_f32("visual_embeds", &[mrows as i64, cfg.out_hidden_size as i64]);
    npu::build_vit_head(&mut g, &topo, &weights, grid_h, grid_w, cfg.pos_grid(), cfg.patch_vec_dim(), "pixels", "visual_embeds");
    g.finish_external(out_path, EXTERNAL_THRESHOLD)
}

/// Export the Code2Wav vocoder decoder for `code_len` frames — the exact same
/// graph shape `crate::codec_bridge::load_codec`'s `codec::Codec` decodes,
/// built from `oc` (`OmniConfig::code2wav`) + `reader` directly (no
/// checkpoint-file round trip). Writes `out_path` (+ `.data` sidecar).
pub fn export_codec_onnx(reader: &WeightReader, oc: &Code2WavConfig, out_path: &str, code_len: usize) -> std::io::Result<()> {
    let cfg = crate::codec_bridge::codec_config_from(oc);
    let weights = crate::codec_bridge::codec_weights(reader).map_err(std::io::Error::other)?;
    let mut g = GraphBuilder::new("omni_codec_decoder");
    npu::codec_topology::build_codec_graph(&cfg, &weights, code_len, &mut g);
    g.finish_external(out_path, EXTERNAL_THRESHOLD)
}

#[cfg(test)]
mod tests {
    /// Structural check against a real checkpoint: all three graphs build and
    /// emit valid ONNX. The vision-tower MATH itself is numerically verified
    /// against the host reference in `crates/npu/tests/qwenvl_onnx.rs`
    /// (synthetic weights, OpenVINO CPU); this test proves the omni-specific
    /// wiring (weight loading + naming, config mapping) holds against a real
    /// checkpoint's actual tensor shapes/names. Run:
    ///   BRAIN_OMNI_HF_DIR=/path/to/hf/dir \
    ///   cargo test -p brain-omni npu_export_builds_real_graphs -- --ignored --nocapture
    #[test]
    #[ignore]
    fn npu_export_builds_real_graphs() {
        let dir = std::env::var("BRAIN_OMNI_HF_DIR").expect("set BRAIN_OMNI_HF_DIR");
        let reader = checkpoint::weightio::WeightReader::open_hf_dir(std::path::Path::new(&dir)).expect("open checkpoint");
        let config_json = std::fs::read_to_string(std::path::Path::new(&dir).join("config.json")).expect("read config.json");
        let root: serde_json::Value = serde_json::from_str(&config_json).expect("parse config.json");
        let cfg = crate::config::OmniConfig::from_json(&root);

        let out = std::env::temp_dir();
        let audio_path = out.join("omni_audio_tower.onnx");
        super::export_audio_onnx(&reader, audio_path.to_str().unwrap(), 100).expect("export audio tower");
        assert!(audio_path.exists());

        let vision_path = out.join("omni_vision_tower.onnx");
        super::export_vision_onnx(&reader, vision_path.to_str().unwrap(), 16, 16).expect("export vision tower");
        assert!(vision_path.exists());

        let codec_path = out.join("omni_codec_decoder.onnx");
        super::export_codec_onnx(&reader, &cfg.code2wav, codec_path.to_str().unwrap(), 32).expect("export codec decoder");
        assert!(codec_path.exists());

        for p in [&audio_path, &vision_path, &codec_path] {
            let bytes = std::fs::read(p).unwrap();
            let m = onnx::decode_model(&bytes).unwrap_or_else(|e| panic!("{p:?}: malformed ONNX: {e}"));
            let g = m.graph.unwrap_or_default();
            eprintln!("{p:?}: {} nodes, {} initializers", g.node.len(), g.initializer.len());
            assert!(!g.node.is_empty() && !g.initializer.is_empty());
        }
    }
}
