// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GLM autoregressive decode via OpenVINO (NPU/CPU/GPU): export the fixed-seq
//! ONNX decoder once, compile it, and greedily fill tokens (cache-free — re-runs
//! the graph over the padded context each step, mirroring brain's own sampler).
//! See `docs/glm/NPU.md`.

use std::time::Instant;

use crate::openvino::{DecoderSession, NpuConfig, NpuDevice, NpuError};

/// Result of an NPU decode run.
pub struct GlmNpuRun {
    pub tokens: Vec<u32>,
    pub device: String,
    pub load_ms: f64,
    pub gen_ms: f64,
    pub int8: bool,
}

fn argmax(s: &[f32]) -> usize {
    (0..s.len()).max_by(|&a, &b| s[a].partial_cmp(&s[b]).unwrap()).unwrap()
}

/// Greedily generate `max_new` tokens continuing `prompt_ids` on OpenVINO
/// `device`. `seq` pins the compiled context length (`prompt+max_new <= seq`);
/// `None` uses exactly `prompt+max_new`. `int8` selects the INT8 weight-only graph.
pub fn generate(
    weights_path: &str,
    prompt_ids: &[u32],
    max_new: usize,
    device: NpuDevice,
    allow_fallback: bool,
    seq: Option<usize>,
    int8: bool,
) -> Result<GlmNpuRun, NpuError> {
    let need = prompt_ids.len() + max_new;
    let cap = seq.unwrap_or(need);
    if need > cap {
        return Err(NpuError::Other(format!("prompt+max_new = {need} exceeds --seq {cap}")));
    }

    let t_load = Instant::now();
    let (bytes, _cfg) = if int8 {
        crate::glm_export::build_glm_int8_bytes(weights_path, cap)
    } else {
        crate::glm_export::build_glm_fp32_bytes(weights_path, cap)
    }
    .map_err(|e| NpuError::Other(format!("build GLM onnx: {e}")))?;

    let mut sess = DecoderSession::load_bytes(
        &bytes,
        &NpuConfig { device, allow_fallback, ..Default::default() },
    )?;
    let vocab = sess.vocab();
    let used = sess.device().to_string();
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    let t_gen = Instant::now();
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
    Ok(GlmNpuRun { tokens: out, device: used, load_ms, gen_ms, int8 })
}
