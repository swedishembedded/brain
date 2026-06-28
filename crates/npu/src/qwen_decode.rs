// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Autoregressive Qwen generation on OpenVINO (NPU / GPU / CPU). Compiles a
//! fixed-length prefill graph once; each step fills the real context (rest
//! padded) and reads the logits at the last real position — causal masking makes
//! that position independent of the padding, so one compile serves the whole
//! greedy decode. Greedy (argmax) only.
//!
//! Caching (avoid the per-run export + compile wait): pass a `cache_dir`. The
//! ONNX (`qwen-seq{N}.onnx` + `.data`) is written there once and reused while it
//! is newer than the weights, and OpenVINO's `CACHE_DIR` blob cache is pointed at
//! the same dir so the compiled NPU graph is reused too. Use a fixed `seq` (and
//! `brain qwen precompile`) so one cached graph serves every prompt up to that
//! length.

use std::path::{Path, PathBuf};

use crate::openvino::{DecoderSession, NpuConfig, NpuDevice, NpuError};
use crate::qwen_export::export_qwen_fp32;

fn argmax(s: &[f32]) -> usize {
    let mut bi = 0;
    for i in 1..s.len() {
        if s[i] > s[bi] {
            bi = i;
        }
    }
    bi
}

/// Timing + result of an OpenVINO decode run.
pub struct NpuRun {
    pub tokens: Vec<u32>,
    pub device: String,
    /// ONNX export (if any) + OpenVINO load/compile, in ms. Near-zero export and
    /// much smaller compile on a cache hit.
    pub load_ms: f64,
    /// Token generation loop in ms.
    pub gen_ms: f64,
    /// Whether a cached ONNX was reused (export skipped).
    pub onnx_cached: bool,
}

fn mtime(p: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

/// Resolve (and, if needed, build) the ONNX decoder for `cap` tokens. With a
/// `cache_dir`, the file is persisted as `qwen-seq{cap}.onnx` and reused while it
/// is at least as new as `weights_path`; without one it goes to a fresh temp dir
/// (returned as the second element to delete after use). Returns
/// `(onnx_path, temp_dir_to_clean, reused)`.
fn prepare_onnx(weights_path: &str, cap: usize, cache_dir: Option<&Path>) -> Result<(PathBuf, Option<PathBuf>, bool), NpuError> {
    let map = |e: std::io::Error| NpuError::Other(format!("export: {e}"));
    match cache_dir {
        Some(cd) => {
            std::fs::create_dir_all(cd).map_err(map)?;
            let onnx = cd.join(format!("qwen-seq{cap}.onnx"));
            let data = cd.join(format!("qwen-seq{cap}.onnx.data"));
            let fresh = onnx.exists()
                && data.exists()
                && match (mtime(&onnx), mtime(Path::new(weights_path))) {
                    (Some(o), Some(w)) => o >= w,
                    _ => true,
                };
            if !fresh {
                export_qwen_fp32(weights_path, onnx.to_str().unwrap(), cap).map_err(map)?;
            }
            Ok((onnx, None, fresh))
        }
        None => {
            let dir = std::env::temp_dir().join(format!("brain_qwen_npu_{}", std::process::id()));
            std::fs::create_dir_all(&dir).map_err(map)?;
            let onnx = dir.join("qwen.onnx");
            export_qwen_fp32(weights_path, onnx.to_str().unwrap(), cap).map_err(map)?;
            Ok((onnx, Some(dir), false))
        }
    }
}

fn config(device: NpuDevice, allow_fallback: bool, cache_dir: Option<&Path>) -> NpuConfig {
    NpuConfig {
        device,
        allow_fallback,
        // OpenVINO CACHE_DIR: caches the compiled blob so re-compiles are skipped.
        cache_dir: cache_dir.map(|p| p.to_path_buf()),
        ..Default::default()
    }
}

/// Export (if needed) + compile the decoder for `seq` tokens into `cache_dir`,
/// without generating — warms both caches so later `generate` calls are fast.
/// Returns `(device_used, elapsed_ms)`.
pub fn precompile(
    weights_path: &str,
    seq: usize,
    device: NpuDevice,
    allow_fallback: bool,
    cache_dir: &Path,
) -> Result<(String, f64), NpuError> {
    let t = std::time::Instant::now();
    let (onnx, _tmp, _reused) = prepare_onnx(weights_path, seq, Some(cache_dir))?;
    let sess = DecoderSession::load_path(&onnx, &config(device, allow_fallback, Some(cache_dir)))?;
    Ok((sess.device().to_string(), t.elapsed().as_secs_f64() * 1e3))
}

/// Greedily generate up to `max_new` tokens continuing `prompt_ids` on the
/// OpenVINO `device`. `seq` pins the compiled context length (so one cached graph
/// serves any prompt with `prompt + max_new <= seq`); when `None` it is exactly
/// `prompt + max_new`. `cache_dir` enables ONNX + compiled-blob reuse.
pub fn generate(
    weights_path: &str,
    prompt_ids: &[u32],
    max_new: usize,
    device: NpuDevice,
    allow_fallback: bool,
    cache_dir: Option<&Path>,
    seq: Option<usize>,
) -> Result<NpuRun, NpuError> {
    let need = prompt_ids.len() + max_new;
    let cap = seq.unwrap_or(need);
    if need > cap {
        return Err(NpuError::Other(format!("prompt+max_new = {need} exceeds --seq {cap}")));
    }

    let t_load = std::time::Instant::now();
    let (onnx, tmp, reused) = prepare_onnx(weights_path, cap, cache_dir)?;
    let sess_result = DecoderSession::load_path(&onnx, &config(device, allow_fallback, cache_dir));
    let mut sess = match sess_result {
        Ok(s) => s,
        Err(e) => {
            if let Some(d) = &tmp {
                std::fs::remove_dir_all(d).ok();
            }
            return Err(e);
        }
    };
    let vocab = sess.vocab();
    let used = sess.device().to_string();
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    let t_gen = std::time::Instant::now();
    let mut ctx: Vec<i64> = prompt_ids.iter().map(|&x| x as i64).collect();
    let mut out = Vec::with_capacity(max_new);
    for _ in 0..max_new {
        let len = ctx.len();
        if len > cap {
            break;
        }
        let mut ids = vec![0i64; cap];
        ids[..len].copy_from_slice(&ctx);
        let logits = sess.run_ids(&ids)?;
        let last = &logits[(len - 1) * vocab..len * vocab];
        let next = argmax(last) as u32;
        ctx.push(next as i64);
        out.push(next);
    }
    let gen_ms = t_gen.elapsed().as_secs_f64() * 1e3;
    if let Some(d) = &tmp {
        std::fs::remove_dir_all(d).ok();
    }
    Ok(NpuRun { tokens: out, device: used, load_ms, gen_ms, onnx_cached: reused })
}
