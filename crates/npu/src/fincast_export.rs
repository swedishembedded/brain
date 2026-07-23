// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export a loaded FinCast checkpoint to an ONNX graph (transformer core) for
//! OpenVINO / NPU compilation. The host does the patch-embed/freq around it and
//! the head rearrange/denorm (see `fincast_topology`).

use fincast::FincastConfig;
use onnx::builder::GraphBuilder;

use crate::fincast_topology::build_fincast_graph_quant;
use crate::qwen_topology::Quant;

/// Build the FinCast core ONNX bytes from a brain `.weights` container, for a
/// fixed sequence length `s` (= number of patch tokens).
pub fn export_onnx(weights_path: &str, s: usize, quant: Quant) -> Result<Vec<u8>, String> {
    let c = checkpoint::load(weights_path);
    let cfg = FincastConfig::from_json(&c.header["config"])?;
    let w = c.by_role("");
    let mut g = GraphBuilder::new("fincast");
    build_fincast_graph_quant(&cfg, &w, s, &mut g, quant);
    Ok(g.finish())
}

/// Export directly to a file.
pub fn export_file(weights_path: &str, s: usize, quant: Quant, out: &str) -> Result<(), String> {
    let bytes = export_onnx(weights_path, s, quant)?;
    std::fs::write(out, &bytes).map_err(|e| format!("write {out}: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end: a tiny FinCast's exported core, compiled + run through
    /// [`crate::openvino::FincastSession`], must reproduce the device
    /// `core_forward`. Runs on the NPU when present, else CPU-OpenVINO via
    /// `allow_fallback`; skips only if no OpenVINO runtime is available at all.
    #[test]
    fn fincast_session_matches_core_forward() {
        use crate::openvino::{FincastSession, NpuConfig, NpuDevice};
        use fincast::model::Fincast;
        use std::collections::HashMap;

        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return; // building the device model needs a backend
        }
        let cfg = FincastConfig::tiny();
        let d = cfg.hidden_size;
        let weights: HashMap<String, Vec<f32>> = cfg
            .param_list()
            .into_iter()
            .map(|(k, s)| {
                let n: usize = s.iter().product();
                let seed = k.len();
                (k, (0..n).map(|i| (((i + seed) as f32) * 0.1).sin() * 0.05).collect())
            })
            .collect();
        let model = Fincast::from_weights(cfg.clone(), &weights).unwrap();

        let s = 6usize;
        let emb: Vec<f32> = (0..s * d).map(|i| ((i as f32) * 0.017).sin() * 0.1).collect();
        let padmask = vec![0.0f32; s]; // all valid -> pure causal
        let reference = model.core_forward(&emb, &padmask);

        // build the additive [S,S] causal mask the graph expects.
        let mut amask = vec![0.0f32; s * s];
        for i in 0..s {
            for j in 0..s {
                if j > i {
                    amask[i * s + j] = -1.0e9;
                }
            }
        }

        // save tiny weights, export the core ONNX for this s.
        let tmp = std::env::temp_dir().join(format!("fincast_tiny_{}.weights", std::process::id()));
        let tmp = tmp.to_str().unwrap();
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .param_list()
            .into_iter()
            .map(|(k, shp)| (k.clone(), shp.iter().map(|&x| x as u64).collect(), weights[&k].clone()))
            .collect();
        checkpoint::save(tmp, cfg.to_json(), &tensors);
        let bytes = export_onnx(tmp, s, Quant::F32).expect("export tiny fincast");
        let _ = std::fs::remove_file(tmp);

        let cosine = |sess: &mut FincastSession| -> f32 {
            let out = sess.run(&emb, &amask).expect("fincast session infer");
            assert_eq!(out.len(), reference.len());
            let dot: f32 = out.iter().zip(&reference).map(|(a, b)| a * b).sum();
            let na = out.iter().map(|v| v * v).sum::<f32>().sqrt();
            let nb = reference.iter().map(|v| v * v).sum::<f32>().sqrt();
            dot / (na * nb + 1e-9)
        };

        // Graph-correctness gate: fp32 CPU-OpenVINO must reproduce core_forward
        // (this is the "runs without an NPU" path). Skips only if no runtime.
        let cpu_cfg = NpuConfig { device: NpuDevice::Cpu, allow_fallback: true, ..Default::default() };
        let mut cpu = match FincastSession::load_bytes(&bytes, &cpu_cfg) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("no OpenVINO runtime, skipping session parity: {e:?}");
                return;
            }
        };
        let cos_cpu = cosine(&mut cpu);
        eprintln!("fincast session on {} (fp32): cosine {cos_cpu:.6}", cpu.device());
        assert!(cos_cpu > 0.99, "fp32 OpenVINO core vs device core cosine {cos_cpu} too low");

        // Also exercise the real NPU when present (informational): the deterministic
        // top-2 MoE routing is sensitive to NPU fp16 rounding — a near-tie gate can
        // flip an expert, so a small cosine deviation there is a hardware precision
        // artifact, not a graph bug. Documented in docs/models/fincast/status.md.
        let npu_cfg = NpuConfig { device: NpuDevice::Npu, allow_fallback: false, ..Default::default() };
        if let Ok(mut npu) = FincastSession::load_bytes(&bytes, &npu_cfg) {
            let cos_npu = cosine(&mut npu);
            eprintln!("fincast session on {} (fp16): cosine {cos_npu:.6}", npu.device());
        }
    }
}
