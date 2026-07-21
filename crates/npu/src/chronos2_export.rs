// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export a loaded Chronos-2 checkpoint to an ONNX graph (transformer core) for
//! OpenVINO / NPU compilation. The host does the scaler/patch/embed/REG and the
//! head rearrange/denorm around it (see `chronos2_topology`).

use chronos2::Chronos2Config;
use onnx::builder::GraphBuilder;

use crate::chronos2_topology::build_chronos2_graph_quant;
use crate::qwen_topology::Quant;

/// Build the Chronos-2 core ONNX bytes from a brain `.weights` container, for a
/// fixed sequence length `s` and forecast-patch count `n_out`.
pub fn export_onnx(weights_path: &str, s: usize, n_out: usize, quant: Quant) -> Result<Vec<u8>, String> {
    let c = checkpoint::load(weights_path);
    let cfg = Chronos2Config::from_hf(&c.header["config"])?;
    let w = c.by_role("");
    let mut g = GraphBuilder::new("chronos2");
    build_chronos2_graph_quant(&cfg, &w, s, n_out, &mut g, quant);
    Ok(g.finish())
}

/// Export directly to a file.
pub fn export_file(weights_path: &str, s: usize, n_out: usize, quant: Quant, out: &str) -> Result<(), String> {
    let bytes = export_onnx(weights_path, s, n_out, quant)?;
    std::fs::write(out, &bytes).map_err(|e| format!("write {out}: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: usize = 35;
    const N_OUT: usize = 2;

    fn write_f32(path: &str, a: &[f32]) {
        let mut b = Vec::with_capacity(a.len() * 4);
        for &v in a {
            b.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, &b).unwrap();
    }

    /// Env-gated: export the REAL Chronos-2 to ONNX so an external OpenVINO probe
    /// can compile it for the NPU. Writes `$CHRONOS2_ONNX_OUT` (or /tmp default).
    #[test]
    fn export_real_checkpoint_to_onnx() {
        let Ok(weights) = std::env::var("CHRONOS2_WEIGHTS") else {
            eprintln!("CHRONOS2_WEIGHTS unset; skipping Chronos-2 ONNX export");
            return;
        };
        let out = std::env::var("CHRONOS2_ONNX_OUT")
            .unwrap_or_else(|_| std::env::temp_dir().join("chronos2_core.onnx").to_string_lossy().into_owned());
        export_file(&weights, S, N_OUT, Quant::F32, &out).expect("export chronos2 onnx");
        let sz = std::fs::metadata(&out).unwrap().len();
        eprintln!("wrote {out} ({} bytes)", sz);
        assert!(sz > 1000);
    }

    /// Env-gated: dump brain's WGSL transformer-core output for a FIXED emb+kmask
    /// (S=35, n_out=2) so an external probe can verify the ONNX/NPU graph matches
    /// brain's own forward — closing the parity chain reference→WGSL→ONNX→NPU.
    #[test]
    fn dump_core_reference_for_onnx_parity() {
        let Ok(weights) = std::env::var("CHRONOS2_WEIGHTS") else {
            eprintln!("CHRONOS2_WEIGHTS unset; skipping core-reference dump");
            return;
        };
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let model = chronos2::Chronos2::load(&weights).unwrap();
        let d = model.config().d_model;
        // deterministic emb + all-attend mask (matches the probe's fixed input)
        let emb: Vec<f32> = (0..S * d).map(|i| ((i as f32) * 0.017).sin() * 0.1).collect();
        let kmask = vec![0.0f32; S];
        let qhead = model.core_forward(&emb, &kmask, N_OUT);
        let dir = std::env::var("CHRONOS2_PARITY_DIR")
            .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
        write_f32(&format!("{dir}/c2_emb.f32"), &emb);
        write_f32(&format!("{dir}/c2_kmask.f32"), &kmask);
        write_f32(&format!("{dir}/c2_qhead_brain.f32"), &qhead);
        eprintln!("dumped brain core: emb[{}] kmask[{}] qhead[{}]", emb.len(), kmask.len(), qhead.len());
    }
}
