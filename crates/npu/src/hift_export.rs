// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export CosyVoice 2's HiFT vocoder conv trunk to ONNX (fixed mel/STFT
//! length), for OpenVINO whole-graph compilation. Pure Rust - no NPU needed to
//! produce the file. Mirrors [`crate::codec_export`] for the conv-heavy HiFT
//! trunk; see [`crate::hift_topology`] for the exact graph and the deliberate
//! host/device split (STFT/ISTFT and the NSF source stay host-side).

use cosyvoice::hift_config::HiftConfig;
use cosyvoice::hift_import::{import_hift_pt, HiftWeights};
use onnx::builder::GraphBuilder;

/// Bytes larger than this go to the ONNX external-data sidecar.
const EXTERNAL_THRESHOLD: usize = 1 << 20; // 1 MiB

/// Build the fp32 ONNX HiFT decode-trunk graph directly from already-imported
/// weights (no checkpoint file to reopen), returning the raw ONNX bytes plus
/// the trunk's output length `L`. `t_mel` is the fixed mel frame count;
/// `n_frames_s` is the fixed frame count of the excitation STFT the caller
/// will feed as `s_stft` at inference time - see
/// [`crate::hift_topology::build_hift_decode_graph`]'s doc for why that
/// tensor is a graph input.
pub fn build_hift_decode_graph_bytes(cfg: &HiftConfig, w: &HiftWeights, t_mel: usize, n_frames_s: usize) -> (Vec<u8>, usize) {
    let mut g = GraphBuilder::new("cosyvoice2_hift_decode");
    let l = crate::hift_topology::build_hift_decode_graph(cfg, w, t_mel, n_frames_s, &mut g);
    (g.finish(), l)
}

/// Import `hift.pt` and export its conv trunk to `out_path` (+ a
/// `<out_path>.data` sidecar for the resblock/upsample weights). Returns the
/// trunk's output length `L` (same as [`build_hift_decode_graph_bytes`]).
pub fn export_hift_decode_fp32(weights_path: &str, out_path: &str, t_mel: usize, n_frames_s: usize) -> Result<usize, String> {
    let cfg = HiftConfig::cosyvoice2();
    let w = import_hift_pt(weights_path, &cfg)?;
    let mut g = GraphBuilder::new("cosyvoice2_hift_decode");
    let l = crate::hift_topology::build_hift_decode_graph(&cfg, &w, t_mel, n_frames_s, &mut g);
    g.finish_external(out_path, EXTERNAL_THRESHOLD).map_err(|e| format!("write {out_path}: {e}"))?;
    Ok(l)
}

#[cfg(test)]
mod tests {
    /// Export against the real checkpoint, when present on this box
    /// (`resources/cosyvoice/weights/hift.pt`) - a structural smoke, not a
    /// numerical parity gate (see `hift_topology`'s own OpenVINO-gated test
    /// for that). Skips cleanly when the checkpoint is absent.
    #[test]
    fn export_against_the_real_checkpoint_if_present() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/cosyvoice/weights/hift.pt");
        if !std::path::Path::new(path).is_file() {
            return;
        }
        let cfg = cosyvoice::hift_config::HiftConfig::cosyvoice2();
        let t_mel = 40usize;
        let n_frames_s = t_mel * cfg.upsample_rates.iter().product::<u32>() as usize / cfg.hop_len as usize + 4;
        let dir = std::env::temp_dir();
        let out = dir.join("cosyvoice2_hift_decode_test.onnx");
        let l = super::export_hift_decode_fp32(path, out.to_str().unwrap(), t_mel, n_frames_s).unwrap_or_else(|e| panic!("export failed: {e}"));
        assert!(l > t_mel);
        assert!(out.exists(), "ONNX file was not written");
        let data_sidecar = dir.join("cosyvoice2_hift_decode_test.onnx.data");
        assert!(data_sidecar.exists(), "external-data sidecar was not written");
    }
}
