// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import the official Qwen3-TTS `speaker_encoder.*` weights into a brain
//! `.safetensors` container.
//!
//! Pure 1:1 name remap: every `speaker_encoder.*` tensor is kept verbatim (just
//! the `speaker_encoder.` prefix stripped) with its PyTorch layout untouched —
//! conv weights stay `[Cout, Cin/G, K]` (what `audio::conv::conv1d` expects) and
//! all biases stay 1-D. There is **no BatchNorm** in this ECAPA variant (the
//! `TimeDelayNetBlock` is Conv1d + ReLU only), so nothing is folded; the encoder
//! consumes the conv weight/bias pairs directly. The original `config.json` is
//! stored so [`crate::SpeakerConfig::from_json`] can recover `enc_dim` /
//! `sample_rate`. Fails loudly if any `speaker_encoder.*` tensor is unaccounted.

use std::collections::HashMap;
use std::path::Path;

/// Import `<ckpt_dir>/config.json` + `<ckpt_dir>/model.safetensors` into the
/// brain checkpoint `out_path`.
pub fn import(ckpt_dir: &str, out_path: &str) -> Result<(), String> {
    let dir = Path::new(ckpt_dir);
    let cfg_json = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("read config.json: {e}"))?;
    let config: serde_json::Value =
        serde_json::from_str(&cfg_json).map_err(|e| format!("parse config.json: {e}"))?;

    let st_path = dir.join("model.safetensors");
    let tensors =
        checkpoint::safetensors::read(st_path.to_str().ok_or("non-utf8 checkpoint path")?)?;

    let mut out: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    let mut seen = 0usize;
    for t in tensors {
        let Some(name) = t.name.strip_prefix("speaker_encoder.") else {
            continue;
        };
        seen += 1;
        if out.insert(name.to_string(), (t.shape, t.data)).is_some() {
            return Err(format!("duplicate speaker_encoder tensor {name}"));
        }
    }
    if seen == 0 {
        return Err("no speaker_encoder.* tensors found in checkpoint".to_string());
    }

    let mut names: Vec<String> = out.keys().cloned().collect();
    names.sort();
    let saved: Vec<(String, Vec<u64>, Vec<f32>)> = names
        .into_iter()
        .map(|n| {
            let (shape, data) = out.remove(&n).unwrap();
            (n, shape.iter().map(|&x| x as u64).collect(), data)
        })
        .collect();

    checkpoint::save(out_path, config, &saved);
    eprintln!("speaker import: {seen} speaker_encoder tensors -> {} params in {out_path}", saved.len());
    Ok(())
}
