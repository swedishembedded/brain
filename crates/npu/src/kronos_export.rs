// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export a Kronos AR-decoder checkpoint to an ONNX graph (`decode_s1` core) for
//! OpenVINO / NPU compilation. The host does the (s1,s2) embedding gather +
//! fusion_proj + calendar sum around it, and `decode_s2` / sampling after it
//! (see `kronos_decoder_topology`).

use onnx::builder::GraphBuilder;

use crate::kronos_topology::{build_kronos_decoder_graph_quant, build_kronos_dep_graph_quant};
use crate::qwen_topology::Quant;

/// Build the Kronos `decode_s1` decoder-core ONNX bytes from an HF checkpoint
/// dir, for a fixed context length `t`.
pub fn export_onnx(decoder_dir: &str, t: usize, quant: Quant) -> Result<Vec<u8>, String> {
    let (cfg, w) = kronos::import::load_decoder(decoder_dir)?;
    let mut g = GraphBuilder::new("kronos_decoder");
    build_kronos_decoder_graph_quant(&cfg, &w, t, &mut g, quant);
    Ok(g.finish())
}

/// Build the Kronos `decode_s2` dependency-layer ONNX bytes.
pub fn export_dep_onnx(decoder_dir: &str, t: usize, quant: Quant) -> Result<Vec<u8>, String> {
    let (cfg, w) = kronos::import::load_decoder(decoder_dir)?;
    let mut g = GraphBuilder::new("kronos_dep");
    build_kronos_dep_graph_quant(&cfg, &w, t, &mut g, quant);
    Ok(g.finish())
}

/// Export the `decode_s1` core directly to a file.
pub fn export_file(decoder_dir: &str, t: usize, quant: Quant, out: &str) -> Result<(), String> {
    let bytes = export_onnx(decoder_dir, t, quant)?;
    std::fs::write(out, &bytes).map_err(|e| format!("write {out}: {e}"))?;
    Ok(())
}

/// Export the `decode_s2` dependency layer directly to a file.
pub fn export_dep_file(decoder_dir: &str, t: usize, quant: Quant, out: &str) -> Result<(), String> {
    let bytes = export_dep_onnx(decoder_dir, t, quant)?;
    std::fs::write(out, &bytes).map_err(|e| format!("write {out}: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: usize = 16;

    fn write_f32(path: &str, a: &[f32]) {
        let mut b = Vec::with_capacity(a.len() * 4);
        for &v in a {
            b.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, &b).unwrap();
    }

    /// Env-gated: export the REAL Kronos decoder to ONNX so an external OpenVINO
    /// probe can compile it for the NPU. Writes `$KRONOS_ONNX_OUT` (or /tmp).
    #[test]
    fn export_real_checkpoint_to_onnx() {
        let Ok(dir) = std::env::var("KRONOS_DECODER_DIR") else {
            eprintln!("KRONOS_DECODER_DIR unset; skipping Kronos ONNX export");
            return;
        };
        let out = std::env::var("KRONOS_ONNX_OUT")
            .unwrap_or_else(|_| std::env::temp_dir().join("kronos_decoder.onnx").to_string_lossy().into_owned());
        let t: usize = std::env::var("KRONOS_ONNX_T").ok().and_then(|v| v.parse().ok()).unwrap_or(T);
        export_file(&dir, t, Quant::F32, &out).expect("export kronos decoder onnx");
        let sz = std::fs::metadata(&out).unwrap().len();
        eprintln!("wrote {out} ({} bytes)", sz);
        assert!(sz > 1000);
    }

    /// Env-gated: dump brain's WGSL `decode_s1` core output for a FIXED input
    /// embedding `x` (T=16, `x[i]=sin(i*0.017)*0.1`) so an external probe can
    /// verify the ONNX/NPU graph matches brain's own forward — closing the parity
    /// chain reference→WGSL→ONNX→NPU for Kronos.
    #[test]
    fn dump_core_reference_for_onnx_parity() {
        let Ok(dir) = std::env::var("KRONOS_DECODER_DIR") else {
            eprintln!("KRONOS_DECODER_DIR unset; skipping Kronos core-reference dump");
            return;
        };
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let (cfg, w) = kronos::import::load_decoder(&dir).expect("load kronos decoder");
        let d = cfg.d_model;
        let s1v = cfg.s1_vocab();
        let dec = kronos::KronosDecoder::from_weights(cfg, &w).expect("build decoder");
        // deterministic embedding (matches the probe's fixed input)
        let x: Vec<f32> = (0..T * d).map(|i| ((i as f32) * 0.017).sin() * 0.1).collect();
        let (s1_logits, ctx) = dec.core_forward_s1(&x, T);
        assert_eq!(s1_logits.len(), T * s1v);
        assert_eq!(ctx.len(), T * d);
        let odir = std::env::var("KRONOS_PARITY_DIR")
            .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
        write_f32(&format!("{odir}/k_x.f32"), &x);
        write_f32(&format!("{odir}/k_s1_logits_brain.f32"), &s1_logits);
        write_f32(&format!("{odir}/k_ctx_brain.f32"), &ctx);
        eprintln!(
            "dumped kronos core: x[{}] s1_logits[{}] ctx[{}]",
            x.len(),
            s1_logits.len(),
            ctx.len()
        );
    }

    /// Env-gated: export the REAL Kronos `decode_s2` dependency layer to ONNX.
    #[test]
    fn export_real_dep_to_onnx() {
        let Ok(dir) = std::env::var("KRONOS_DECODER_DIR") else {
            eprintln!("KRONOS_DECODER_DIR unset; skipping Kronos dep ONNX export");
            return;
        };
        let out = std::env::var("KRONOS_DEP_ONNX_OUT")
            .unwrap_or_else(|_| std::env::temp_dir().join("kronos_dep.onnx").to_string_lossy().into_owned());
        let t: usize = std::env::var("KRONOS_ONNX_T").ok().and_then(|v| v.parse().ok()).unwrap_or(T);
        export_dep_file(&dir, t, Quant::F32, &out).expect("export kronos dep onnx");
        let sz = std::fs::metadata(&out).unwrap().len();
        eprintln!("wrote {out} ({} bytes)", sz);
        assert!(sz > 1000);
    }

    /// Env-gated: dump brain's WGSL `decode_s2` (dependency layer) output for a
    /// FIXED `(ctx, sib)` input so the probe can verify the dep ONNX/NPU graph.
    #[test]
    fn dump_dep_reference_for_onnx_parity() {
        let Ok(dir) = std::env::var("KRONOS_DECODER_DIR") else {
            eprintln!("KRONOS_DECODER_DIR unset; skipping Kronos dep reference dump");
            return;
        };
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let (cfg, w) = kronos::import::load_decoder(&dir).expect("load kronos decoder");
        let d = cfg.d_model;
        let s2v = cfg.s2_vocab();
        let dec = kronos::KronosDecoder::from_weights(cfg, &w).expect("build decoder");
        let ctx: Vec<f32> = (0..T * d).map(|i| ((i as f32) * 0.013).sin() * 0.1).collect();
        let sib: Vec<f32> = (0..T * d).map(|i| ((i as f32) * 0.011).cos() * 0.1).collect();
        let s2_logits = dec.core_forward_s2(&ctx, &sib, T);
        assert_eq!(s2_logits.len(), T * s2v);
        let odir = std::env::var("KRONOS_PARITY_DIR")
            .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
        write_f32(&format!("{odir}/k_dep_ctx.f32"), &ctx);
        write_f32(&format!("{odir}/k_dep_sib.f32"), &sib);
        write_f32(&format!("{odir}/k_s2_logits_brain.f32"), &s2_logits);
        eprintln!("dumped kronos dep: ctx[{}] sib[{}] s2_logits[{}]", ctx.len(), sib.len(), s2_logits.len());
    }
}
