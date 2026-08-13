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

use std::path::Path;

/// Import `<ckpt_dir>/config.json` + `<ckpt_dir>/model.safetensors` into the
/// brain checkpoint `out_path`. Streams one source tensor at a time — the
/// output plan (names + shapes) is built first from the source header alone
/// (no tensor data touched), then the actual tensor bytes stream straight
/// through to `out_path` without ever holding the whole checkpoint in RAM.
pub fn import(ckpt_dir: &str, out_path: &str) -> Result<(), String> {
    let dir = Path::new(ckpt_dir);
    let cfg_json = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("read config.json: {e}"))?;
    let config: serde_json::Value =
        serde_json::from_str(&cfg_json).map_err(|e| format!("parse config.json: {e}"))?;

    let st_path = dir.join("model.safetensors");
    let reader = checkpoint::weightio::WeightReader::open(st_path.to_str().ok_or("non-utf8 checkpoint path")?)
        .map_err(|e| format!("import: opening checkpoint: {e}"))?;

    // Header-only pass: every `speaker_encoder.*` tensor's stripped name +
    // shape, sorted (source names are already unique, so no dedup needed).
    let mut plan: Vec<(String, Vec<u64>)> = Vec::new();
    for name in reader.names() {
        let Some(out_name) = name.strip_prefix("speaker_encoder.") else {
            continue;
        };
        let shape = reader.shape(name).ok_or_else(|| format!("import: missing shape for {name}"))?;
        plan.push((out_name.to_string(), shape.to_vec()));
    }
    if plan.is_empty() {
        return Err("no speaker_encoder.* tensors found in checkpoint".to_string());
    }
    plan.sort_by(|a, b| a.0.cmp(&b.0));
    let seen = plan.len();

    let mut writer = checkpoint::weightio::StWriter::create(out_path, &plan, &config, None)
        .map_err(|e| format!("import: creating output: {e}"))?;
    let mut err: Option<String> = None;
    reader.for_each(|name, _shape, data| {
        if err.is_some() {
            return;
        }
        let Some(out_name) = name.strip_prefix("speaker_encoder.") else {
            return;
        };
        if let Err(e) = writer.write(out_name, &data) {
            err = Some(format!("import: {e}"));
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    writer.finish().map_err(|e| format!("import: {e}"))?;
    eprintln!("speaker import: {seen} speaker_encoder tensors -> {} params in {out_path}", seen);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_import_strips_prefix_and_drops_other_tensors() {
        // A synthetic HF checkpoint dir: config.json + a model.safetensors with
        // two speaker_encoder.* tensors and one unrelated tensor (e.g. talker.*)
        // that must be silently dropped (only the seen/plan count reflects it).
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("speaker-import-src-{pid}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), br#"{"enc_dim": 4, "sample_rate": 16000}"#).unwrap();

        let plan = vec![
            ("speaker_encoder.tdnn.0.conv.weight".to_string(), vec![2u64, 1, 3]),
            ("speaker_encoder.tdnn.0.conv.bias".to_string(), vec![2u64]),
            ("talker.model.norm.weight".to_string(), vec![4u64]),
        ];
        let mut w = checkpoint::weightio::StWriter::create(
            dir.join("model.safetensors").to_str().unwrap(),
            &plan,
            &serde_json::Value::Null,
            None,
        )
        .unwrap();
        w.write("speaker_encoder.tdnn.0.conv.weight", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        w.write("speaker_encoder.tdnn.0.conv.bias", &[0.5, -0.5]).unwrap();
        w.write("talker.model.norm.weight", &[9.0, 9.0, 9.0, 9.0]).unwrap();
        w.finish().unwrap();

        let out = std::env::temp_dir().join(format!("speaker-import-out-{pid}.safetensors"));
        import(dir.to_str().unwrap(), out.to_str().unwrap()).unwrap();

        let reader = checkpoint::weightio::WeightReader::open(out.to_str().unwrap()).unwrap();
        assert_eq!(reader.names().count(), 2, "only speaker_encoder.* tensors, prefix stripped");
        assert_eq!(reader.tensor("tdnn.0.conv.weight").unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(reader.tensor("tdnn.0.conv.bias").unwrap(), vec![0.5, -0.5]);
        assert!(reader.tensor("talker.model.norm.weight").is_none());
        assert_eq!(reader.config()["enc_dim"], 4);

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_file(&out).ok();
    }
}
