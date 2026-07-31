// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export a Kronos AR-decoder checkpoint to an ONNX graph (`decode_s1` core) for
//! OpenVINO / NPU compilation. The host does the (s1,s2) embedding gather +
//! fusion_proj + calendar sum around it, and `decode_s2` / sampling after it
//! (see `kronos_decoder_topology`).

use onnx::builder::GraphBuilder;

use crate::kronos_topology::{
    build_kronos_decoder_graph_quant, build_kronos_dep_decode_graph_quant, build_kronos_dep_graph_quant,
    build_kronos_dep_prefill_graph_quant, build_kronos_s1_decode_graph_quant, build_kronos_s1_prefill_graph_quant,
};
use crate::qwen_topology::Quant;

/// Build the four KV-cached Kronos graphs from one checkpoint load, for a fixed
/// context length `t` and cache capacity `cap` (>= `t + horizon`). Returns their
/// ONNX bytes as `(s1_prefill @T=t, s1_decode @cap, dep_prefill @T=t-1,
/// dep_decode @cap)` — the s1/dep prefill seed the cache, the decode graphs append
/// one token per step (the NPU's KV-cache path; see `KronosModel::
/// forecast_cached_with_cores`). `dep_prefill` fills context positions `0..t-1`,
/// leaving the last (`t-1`) for the first `dep_decode` self-projection.
#[allow(clippy::type_complexity)]
pub fn export_cached_onnx(decoder_dir: &str, t: usize, cap: usize, quant: Quant) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>), String> {
    let (cfg, w) = kronos::import::load_decoder(decoder_dir)?;
    let s1_prefill = {
        let mut g = GraphBuilder::new("kronos_s1_prefill");
        build_kronos_s1_prefill_graph_quant(&cfg, &w, t, &mut g, quant);
        g.finish()
    };
    let s1_decode = {
        let mut g = GraphBuilder::new("kronos_s1_decode");
        build_kronos_s1_decode_graph_quant(&cfg, &w, cap, &mut g, quant);
        g.finish()
    };
    let dep_prefill = {
        let mut g = GraphBuilder::new("kronos_dep_prefill");
        build_kronos_dep_prefill_graph_quant(&cfg, &w, (t - 1).max(1), &mut g, quant);
        g.finish()
    };
    let dep_decode = {
        let mut g = GraphBuilder::new("kronos_dep_decode");
        build_kronos_dep_decode_graph_quant(&cfg, &w, cap, &mut g, quant);
        g.finish()
    };
    Ok((s1_prefill, s1_decode, dep_prefill, dep_decode))
}

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

    /// End-to-end: a tiny Kronos decoder's exported `decode_s1`/`decode_s2` graphs,
    /// compiled + run through [`KronosS1Session`]/[`KronosS2Session`], must
    /// reproduce the device `core_forward_s1`/`core_forward_s2`. Runs on the NPU
    /// when present, else CPU-OpenVINO; skips only with no OpenVINO runtime. Guards
    /// the multi-IO session wiring `brain npu kronos` depends on (s1: 1 in → ctx +
    /// s1_logits; s2: ctx + sib → s2_logits).
    #[test]
    fn kronos_sessions_match_core_forward() {
        use crate::openvino::{KronosS1Session, KronosS2Session, NpuConfig, NpuDevice};
        use kronos::config::KronosConfig;
        use kronos::decoder::KronosDecoder;
        use std::collections::HashMap;

        let cfg = KronosConfig::tiny();
        let t = 8usize;
        let (vs1, vs2) = (cfg.s1_vocab(), cfg.s2_vocab());
        let mut seed = 0x0DE_u64;
        let mut rnd = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1
                })
                .collect()
        };
        let weights: HashMap<String, Vec<f32>> =
            cfg.param_list().into_iter().map(|(k, s)| (k, rnd(s.iter().product()))).collect();
        let dec = KronosDecoder::from_weights(cfg.clone(), &weights).unwrap();

        // deterministic in-vocab tokens.
        let s1: Vec<u32> = (0..t).map(|i| (i as u32 * 3 + 1) % vs1 as u32).collect();
        let s2: Vec<u32> = (0..t).map(|i| (i as u32 * 5 + 2) % vs2 as u32).collect();
        let x = dec.embed_tokens(&s1, &s2, &[]);
        let (s1_ref, ctx_ref) = dec.core_forward_s1(&x, t);
        let sib = dec.sib_embed(&s1);
        let s2_ref = dec.core_forward_s2(&ctx_ref, &sib, t);

        // save tiny decoder, export both graphs at T=t.
        let tmp = std::env::temp_dir().join(format!("kronos_tiny_{}.safetensors", std::process::id()));
        let tmp = tmp.to_str().unwrap();
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .param_list()
            .into_iter()
            .map(|(k, shp)| (k.clone(), shp.iter().map(|&x| x as u64).collect(), weights[&k].clone()))
            .collect();
        checkpoint::save(tmp, cfg.to_json(), &tensors);
        let s1_bytes = export_onnx(tmp, t, Quant::F32).expect("export decode_s1");
        let s2_bytes = export_dep_onnx(tmp, t, Quant::F32).expect("export decode_s2");
        let _ = std::fs::remove_file(tmp);

        let npu_cfg = NpuConfig { device: NpuDevice::Npu, allow_fallback: true, ..Default::default() };
        let mut s1_sess = match KronosS1Session::load_bytes(&s1_bytes, &npu_cfg) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("no OpenVINO runtime, skipping: {e:?}");
                return;
            }
        };
        let mut s2_sess = KronosS2Session::load_bytes(&s2_bytes, &npu_cfg).expect("compile decode_s2");

        let (ctx_npu, s1_npu) = s1_sess.run(&x).expect("s1 infer");
        let s2_npu = s2_sess.run(&ctx_npu, &sib).expect("s2 infer");

        let cos = |a: &[f32], b: &[f32]| -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na = a.iter().map(|v| v * v).sum::<f32>().sqrt();
            let nb = b.iter().map(|v| v * v).sum::<f32>().sqrt();
            dot / (na * nb + 1e-9)
        };
        let dev = s1_sess.device().to_string();
        eprintln!(
            "kronos sessions on {dev}: ctx cos {:.6}, s1 cos {:.6}, s2 cos {:.6}",
            cos(&ctx_npu, &ctx_ref),
            cos(&s1_npu, &s1_ref),
            cos(&s2_npu, &s2_ref)
        );
        assert!(cos(&ctx_npu, &ctx_ref) > 0.99, "ctx parity");
        assert!(cos(&s1_npu, &s1_ref) > 0.99, "s1_logits parity");
        assert!(cos(&s2_npu, &s2_ref) > 0.99, "s2_logits parity");
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
