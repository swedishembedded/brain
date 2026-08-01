// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export a loaded Chronos-2 checkpoint to an ONNX graph (transformer core) for
//! OpenVINO / NPU compilation. The host does the scaler/patch/embed/REG and the
//! head rearrange/denorm around it (see `chronos2_topology`).

use chronos2::Chronos2Config;
use onnx::builder::GraphBuilder;

use crate::chronos2_topology::build_chronos2_graph_quant;
use crate::qwen_topology::Quant;

/// Build the Chronos-2 core ONNX bytes from a brain `.safetensors` container, for a
/// fixed sequence length `s` and forecast-patch count `n_out`.
pub fn export_onnx(weights_path: &str, s: usize, n_out: usize, quant: Quant) -> Result<Vec<u8>, String> {
    let reader = checkpoint::weightio::WeightReader::open(weights_path).map_err(|e| format!("open {weights_path}: {e}"))?;
    let cfg = Chronos2Config::from_hf(&reader.config())?;
    let mut g = GraphBuilder::new("chronos2");
    build_chronos2_graph_quant(&cfg, &reader, s, n_out, &mut g, quant);
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

    /// End-to-end: a tiny Chronos-2's exported core, compiled + run through
    /// [`Chronos2Session`], must reproduce the device `core_forward`. Runs on the
    /// NPU when present, else CPU-OpenVINO via `allow_fallback`; skips only if no
    /// OpenVINO runtime is available at all. Guards the two-input session wiring
    /// (`emb` + `kmask` → `qhead`) that `brain npu chronos2` depends on.
    #[test]
    fn chronos2_session_matches_core_forward() {
        use crate::openvino::{Chronos2Session, NpuConfig, NpuDevice};
        use chronos2::model::Chronos2;
        use chronos2::Chronos2Config;
        use std::collections::HashMap;

        let cfg = Chronos2Config::tiny();
        let d = cfg.d_model;
        let weights: HashMap<String, Vec<f32>> = cfg
            .param_list()
            .into_iter()
            .map(|(k, s)| {
                let n: usize = s.iter().product();
                let seed = k.len();
                (k, (0..n).map(|i| (((i + seed) as f32) * 0.1).sin() * 0.05).collect())
            })
            .collect();
        let model = Chronos2::from_weights(cfg.clone(), &weights).unwrap();

        let (s, n_out) = (6usize, 2usize);
        let emb: Vec<f32> = (0..s * d).map(|i| ((i as f32) * 0.017).sin() * 0.1).collect();
        let kmask = vec![0.0f32; s];
        let reference = model.core_forward(&emb, &kmask, n_out);

        // save tiny weights, export the core ONNX for this (s, n_out).
        let tmp = std::env::temp_dir().join(format!("chronos2_tiny_{}.safetensors", std::process::id()));
        let tmp = tmp.to_str().unwrap();
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .param_list()
            .into_iter()
            .map(|(k, shp)| (k.clone(), shp.iter().map(|&x| x as u64).collect(), weights[&k].clone()))
            .collect();
        checkpoint::save(tmp, cfg.to_json(), &tensors);
        let bytes = export_onnx(tmp, s, n_out, Quant::F32).expect("export tiny chronos2");
        let _ = std::fs::remove_file(tmp);

        // compile on NPU (fallback CPU-OpenVINO); skip if no runtime.
        let npu_cfg = NpuConfig { device: NpuDevice::Npu, allow_fallback: true, ..Default::default() };
        let mut sess = match Chronos2Session::load_bytes(&bytes, &npu_cfg) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("no OpenVINO runtime, skipping session parity: {e:?}");
                return;
            }
        };
        let out = sess.run(&emb, &kmask).expect("chronos2 session infer");
        assert_eq!(out.len(), reference.len());

        let dot: f32 = out.iter().zip(&reference).map(|(a, b)| a * b).sum();
        let na = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb = reference.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cos = dot / (na * nb + 1e-9);
        eprintln!("chronos2 session on {}: cosine {cos:.6}", sess.device());
        assert!(cos > 0.99, "NPU core vs device core cosine {cos} too low");
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
