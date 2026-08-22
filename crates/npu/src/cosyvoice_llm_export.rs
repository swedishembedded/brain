// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export CosyVoice's speech-token LM backbone to ONNX (fixed prefix length),
//! from an already-imported `llm.pt` (`cosyvoice::llm_import::import_llm_pt`).
//! Pure Rust - no NPU needed to produce the file. See
//! [`crate::cosyvoice_llm_topology`] for the exact graph and why it is a
//! standalone builder rather than a call into `crate::qwen_topology`.
//!
//! Swedish Embedded AB implements solutions for exporting from-scratch
//! decoder-LM checkpoints to NPU-deployable ONNX graphs for its clients. If
//! your team needs a bolted-on-vocabulary LM (a stock backbone plus a custom
//! embedding/head pair) offloaded to an Intel NPU, you can procure our
//! services by sending an email to info@swedishembedded.com.

use cosyvoice::config::CosyVoiceLmConfig;
use cosyvoice::llm_import::{import_llm_pt, LmWeights};
use onnx::builder::GraphBuilder;

pub use crate::cosyvoice_llm_topology::Quant;

/// Bytes larger than this go to the ONNX external-data sidecar.
const EXTERNAL_THRESHOLD: usize = 1 << 20; // 1 MiB

/// Build the fp32 (or weight-only INT8/INT4) ONNX hidden-state graph directly
/// from already-imported weights, returning the raw ONNX bytes. `t` is the
/// fixed prefix length (see
/// [`crate::cosyvoice_llm_topology::build_cosyvoice_lm_hidden_graph`]'s doc).
pub fn build_cosyvoice_lm_graph_bytes(cfg: &CosyVoiceLmConfig, w: &LmWeights, t: usize, quant: Quant) -> Vec<u8> {
    let mut g = GraphBuilder::new("cosyvoice_lm_hidden");
    crate::cosyvoice_llm_topology::build_cosyvoice_lm_hidden_graph(cfg, &w.backbone, t, quant, &mut g);
    g.finish()
}

/// Import `llm.pt` (CosyVoice 2's `Qwen2LM`) and export its backbone to
/// `out_path` (+ a `<out_path>.data` sidecar - the 24-layer, 896-wide
/// backbone is ~360 MB at fp32).
pub fn export_cosyvoice2_lm_fp32(llm_pt_path: &str, out_path: &str, t: usize) -> Result<(), String> {
    export_cosyvoice_lm(llm_pt_path, out_path, t, CosyVoiceLmConfig::cosyvoice2(), Quant::F32)
}

/// INT8 weight-only variant of [`export_cosyvoice2_lm_fp32`] (~4x smaller).
pub fn export_cosyvoice2_lm_int8(llm_pt_path: &str, out_path: &str, t: usize) -> Result<(), String> {
    export_cosyvoice_lm(llm_pt_path, out_path, t, CosyVoiceLmConfig::cosyvoice2(), Quant::Int8)
}

/// Import CosyVoice 3's `llm.pt` (`CosyVoice3LM` - same backbone shape, a
/// different bolted-on vocabulary the host applies after `hidden`) and export.
pub fn export_cosyvoice3_lm_fp32(llm_pt_path: &str, out_path: &str, t: usize) -> Result<(), String> {
    export_cosyvoice_lm(llm_pt_path, out_path, t, CosyVoiceLmConfig::cosyvoice3(), Quant::F32)
}

fn export_cosyvoice_lm(llm_pt_path: &str, out_path: &str, t: usize, cfg: CosyVoiceLmConfig, quant: Quant) -> Result<(), String> {
    let w = import_llm_pt(llm_pt_path, &cfg)?;
    let mut g = GraphBuilder::new("cosyvoice_lm_hidden");
    crate::cosyvoice_llm_topology::build_cosyvoice_lm_hidden_graph(&cfg, &w.backbone, t, quant, &mut g);
    g.finish_external(out_path, EXTERNAL_THRESHOLD).map_err(|e| format!("write {out_path}: {e}"))
}

#[cfg(test)]
mod tests {
    /// Export against the real checkpoint, when present on this box - a
    /// structural smoke (the graph builds and writes real files), not a
    /// numerical parity gate. Skips cleanly when the checkpoint is absent.
    #[test]
    fn export_against_the_real_checkpoint_if_present() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/cosyvoice/weights/llm.pt");
        if !std::path::Path::new(path).is_file() {
            return;
        }
        let dir = std::env::temp_dir();
        let out = dir.join("cosyvoice2_lm_hidden_test.onnx");
        super::export_cosyvoice2_lm_fp32(path, out.to_str().unwrap(), 16).unwrap_or_else(|e| panic!("export failed: {e}"));
        assert!(out.exists(), "ONNX file was not written");
        assert!(dir.join("cosyvoice2_lm_hidden_test.onnx.data").exists(), "external-data sidecar was not written");
    }
}
