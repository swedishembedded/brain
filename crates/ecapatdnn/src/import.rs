// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Read the official Qwen3-TTS `speaker_encoder.*` weights as brain's speaker
//! encoder, as they are downloaded ([`view`]), or write them out as a brain
//! `.safetensors` container ([`import`]).
//!
//! Pure 1:1 name remap: every `speaker_encoder.*` tensor is kept verbatim (just
//! the `speaker_encoder.` prefix stripped) with its PyTorch layout untouched —
//! conv weights stay `[Cout, Cin/G, K]` (what `audio::conv::conv1d` expects) and
//! all biases stay 1-D. There is **no BatchNorm** in this ECAPA variant (the
//! `TimeDelayNetBlock` is Conv1d + ReLU only), so nothing is folded; the encoder
//! consumes the conv weight/bias pairs directly. The view's config is the
//! checkpoint's own `config.json`, so [`crate::SpeakerConfig::from_json`] can
//! recover `enc_dim` / `sample_rate`.

use std::path::Path;

use checkpoint::weightio::{Renamed, WeightReader};

/// The speaker encoder of `<ckpt_dir>/model.safetensors`, under brain's names,
/// with `<ckpt_dir>/config.json` as its config. Fails when the checkpoint has
/// no `speaker_encoder.*` tensor (a CustomVoice/VoiceDesign checkpoint).
pub fn view(ckpt_dir: &Path) -> Result<WeightReader, String> {
    let cfg_json = std::fs::read_to_string(ckpt_dir.join("config.json"))
        .map_err(|e| format!("read config.json: {e}"))?;
    let config: serde_json::Value =
        serde_json::from_str(&cfg_json).map_err(|e| format!("parse config.json: {e}"))?;
    let st_path = ckpt_dir.join("model.safetensors");
    let src = WeightReader::open(st_path.to_str().ok_or("non-utf8 checkpoint path")?)
        .map_err(|e| format!("import: opening checkpoint: {e}"))?;
    let mut map: Vec<(String, String)> = src
        .names()
        .filter_map(|n| n.strip_prefix("speaker_encoder.").map(|out| (out.to_string(), n.to_string())))
        .collect();
    if map.is_empty() {
        return Err("no speaker_encoder.* tensors found in checkpoint".to_string());
    }
    map.sort();
    Ok(WeightReader::derived(Box::new(Renamed::new(src, map, config)?)))
}

/// A speaker-encoder checkpoint as the loader takes it: a brain file (from
/// [`import`]) as it is, or the HF checkpoint dir through [`view`].
pub fn open(path: &str) -> Result<WeightReader, String> {
    let r = if Path::new(path).is_dir() { view(Path::new(path)) } else { WeightReader::open(path).map_err(|e| e.to_string()) };
    r.map_err(|e| format!("{path}: {e}"))
}

/// [`open`], every tensor read as f32, for the eager loader. Panics naming
/// `path` when it cannot be read, as `checkpoint::load` does.
pub fn load(path: &str) -> checkpoint::Container {
    open(path).and_then(|r| checkpoint::load_reader(&r)).unwrap_or_else(|e| panic!("{e}"))
}

/// Write the speaker encoder of `<ckpt_dir>` to the brain checkpoint
/// `out_path` - [`view`], one tensor at a time.
pub fn import(ckpt_dir: &str, out_path: &str) -> Result<(), String> {
    let v = view(Path::new(ckpt_dir))?;
    v.save(out_path, None).map_err(|e| format!("import: {e}"))?;
    eprintln!("speaker import: {} speaker_encoder tensors -> {out_path}", v.names().count());
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

        // Served as downloaded: the view reads what the import wrote.
        let viewed = load(dir.to_str().unwrap());
        let written = checkpoint::load(out.to_str().unwrap());
        assert_eq!(viewed.header, written.header);
        assert_eq!(viewed.by_role(""), written.by_role(""));

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_file(&out).ok();
    }
}
