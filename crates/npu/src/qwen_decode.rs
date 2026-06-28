// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Autoregressive Qwen generation on OpenVINO (NPU / GPU / CPU). Compiles a
//! fixed-length (`prompt + max_new`) prefill graph once; each step fills the
//! real context (rest padded) and reads the logits at the last real position —
//! causal masking makes that position independent of the padding, so one compile
//! serves the whole greedy decode. Greedy (argmax) only.

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
    /// ONNX export + OpenVINO compile (one-time) in ms.
    pub load_ms: f64,
    /// Token generation loop in ms.
    pub gen_ms: f64,
}

/// Greedily generate up to `max_new` tokens continuing `prompt_ids`, running the
/// decoder on the OpenVINO `device` (falling back to CPU/GPU if `allow_fallback`
/// and the device is absent). Returns the tokens, device, and load/gen timing.
pub fn generate(
    weights_path: &str,
    prompt_ids: &[u32],
    max_new: usize,
    device: NpuDevice,
    allow_fallback: bool,
) -> Result<NpuRun, NpuError> {
    let t_load = std::time::Instant::now();
    let cap = prompt_ids.len() + max_new;
    // Export the fixed-length decoder (weights as ONNX external data so the proto
    // stays under protobuf's 2GB limit) to a temp dir, then compile from the path.
    let dir = std::env::temp_dir().join(format!("brain_qwen_npu_{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| NpuError::Other(format!("tmp dir: {e}")))?;
    let onnx = dir.join("qwen.onnx");
    export_qwen_fp32(weights_path, onnx.to_str().unwrap(), cap)
        .map_err(|e| NpuError::Other(format!("export: {e}")))?;
    let cfg = NpuConfig { device, allow_fallback, ..Default::default() };
    let sess_result = DecoderSession::load_path(&onnx, &cfg);
    let mut sess = match sess_result {
        Ok(s) => s,
        Err(e) => {
            std::fs::remove_dir_all(&dir).ok();
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
    std::fs::remove_dir_all(&dir).ok();
    Ok(NpuRun { tokens: out, device: used, load_ms, gen_ms })
}
