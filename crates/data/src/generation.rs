// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which token ids end a generation, read from the checkpoint rather than
//! guessed from one vocabulary's conventions.
//!
//! A checkpoint states its end-of-sequence ids in `generation_config.json`
//! (`eos_token_id`, a single id or a list: R1-Distill-Qwen names two), and
//! otherwise names the token in `tokenizer_config.json` (`eos_token`). A chat
//! format adds its own end-of-turn marker on top: a model answering in
//! ChatML closes its turn with `<|im_end|>` whatever its eos is.
//!
//! Swedish Embedded AB implements model serving for its clients. If your team
//! needs generation that stops where the model was trained to stop, you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::path::Path;

use serde_json::Value;

use crate::qwen_tokenizer::QwenBpe;

/// The fields of a `generation_config.json` that decide where generation
/// stops and what it starts from.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationConfig {
    /// `eos_token_id`, as a list whether the file gives one id or several.
    pub eos_token_ids: Vec<u32>,
    pub bos_token_id: Option<u32>,
}

impl GenerationConfig {
    /// Read `<dir>/generation_config.json`; `None` when the file is absent.
    pub fn read(dir: &Path) -> Result<Option<GenerationConfig>, String> {
        let path = dir.join("generation_config.json");
        let Ok(text) = std::fs::read_to_string(&path) else { return Ok(None) };
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let id = |x: &Value| x.as_u64().map(|n| n as u32).ok_or_else(|| format!("{}: token id {x} is not an integer", path.display()));
        let eos_token_ids = match &v["eos_token_id"] {
            Value::Null => Vec::new(),
            Value::Array(a) => a.iter().map(id).collect::<Result<_, _>>()?,
            x => vec![id(x)?],
        };
        let bos_token_id = match &v["bos_token_id"] {
            Value::Null => None,
            x => Some(id(x)?),
        };
        Ok(Some(GenerationConfig { eos_token_ids, bos_token_id }))
    }
}

/// The sampling settings a checkpoint declares as its own defaults: its `generation_config.json`
/// (`temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`) or, for a GGUF, the
/// `general.sampling.*` KV keys. What a request that does not name a setting should sample with;
/// a setting the checkpoint does not declare is `None` and the serving action's own constant stays.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SamplingDefaults {
    pub temp: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

impl SamplingDefaults {
    /// Whether the checkpoint declares nothing.
    pub fn is_empty(&self) -> bool {
        *self == SamplingDefaults::default()
    }

    /// Read the settings from `<dir>/generation_config.json`; empty when the file is absent. A file that declares
    /// `do_sample: false` asks for greedy decoding, which is what a zero temperature is. A malformed value is an
    /// error naming the file and the key, never a silent fallback.
    pub fn from_generation_config(dir: &Path) -> Result<SamplingDefaults, String> {
        let path = dir.join("generation_config.json");
        let Ok(text) = std::fs::read_to_string(&path) else { return Ok(SamplingDefaults::default()) };
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let number = |key: &str| -> Result<Option<f64>, String> {
            match &v[key] {
                Value::Null => Ok(None),
                x => x.as_f64().map(Some).ok_or_else(|| format!("{}: {key} {x} is not a number", path.display())),
            }
        };
        let mut d = SamplingDefaults::from_values(number("temperature")?, number("top_k")?, number("top_p")?, number("min_p")?, number("repetition_penalty")?)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if v["do_sample"] == Value::Bool(false) {
            d.temp = Some(0.0);
        }
        Ok(d)
    }

    /// Read the `general.sampling.*` keys of a GGUF through `get` (a key to its numeric value, `None` when absent).
    pub fn from_gguf(get: impl Fn(&str) -> Option<f64>) -> Result<SamplingDefaults, String> {
        SamplingDefaults::from_values(get("general.sampling.temp"), get("general.sampling.top_k"), get("general.sampling.top_p"), get("general.sampling.min_p"), get("general.sampling.penalty_repeat"))
            .map_err(|e| format!("general.sampling: {e}"))
    }

    fn from_values(temp: Option<f64>, top_k: Option<f64>, top_p: Option<f64>, min_p: Option<f64>, repetition_penalty: Option<f64>) -> Result<SamplingDefaults, String> {
        let within = |name: &str, x: Option<f64>, lo: f64, hi: f64| match x {
            Some(x) if !(lo..=hi).contains(&x) => Err(format!("{name} {x} is outside {lo}..={hi}")),
            x => Ok(x),
        };
        Ok(SamplingDefaults {
            temp: within("temperature", temp, 0.0, 100.0)?.map(|x| x as f32),
            top_k: within("top_k", top_k, 0.0, u32::MAX as f64)?.map(|x| x as u32),
            top_p: within("top_p", top_p, 0.0, 1.0)?.map(|x| x as f32),
            min_p: within("min_p", min_p, 0.0, 1.0)?.map(|x| x as f32),
            repetition_penalty: within("repetition_penalty", repetition_penalty, 0.0, 100.0)?.map(|x| x as f32),
        })
    }
}

/// `tokenizer_config.json`'s `eos_token` (a string, or `{"content": ...}`).
fn tokenizer_config_eos(dir: &Path) -> Option<String> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json")).ok()?).ok()?;
    let eos = &v["eos_token"];
    eos.as_str().or_else(|| eos["content"].as_str()).map(str::to_string)
}

/// The ids that end a generation from the checkpoint in `dir`: its
/// `generation_config.json` eos ids, else its `tokenizer_config.json` eos
/// token, plus `end_of_turn` (the rendered chat format's marker) when the
/// vocabulary has it. `dir` is `None` for a checkpoint with no sibling files
/// (a GGUF), whose own declared eos is `gguf_eos`.
pub fn stop_ids(dir: Option<&Path>, tok: &QwenBpe, gguf_eos: Option<u32>, end_of_turn: Option<&str>) -> Result<Vec<u32>, String> {
    let mut ids: Vec<u32> = gguf_eos.into_iter().collect();
    if let Some(dir) = dir {
        match GenerationConfig::read(dir)? {
            Some(g) if !g.eos_token_ids.is_empty() => ids.extend(g.eos_token_ids),
            _ => ids.extend(tokenizer_config_eos(dir).and_then(|t| tok.special_id(&t))),
        }
    }
    ids.extend(end_of_turn.and_then(|t| tok.special_id(t)));
    let mut seen = std::collections::HashSet::new();
    ids.retain(|id| seen.insert(*id));
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("brain-generation-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn tok() -> QwenBpe {
        let json = serde_json::json!({
            "model": {"vocab": {"a": 0, "b": 1}, "merges": []},
            "added_tokens": [{"content": "<|endoftext|>", "id": 2}, {"content": "<|im_end|>", "id": 3}, {"content": "<|EOT|>", "id": 4}],
        });
        QwenBpe::from_json_bytes(json.to_string().as_bytes()).unwrap()
    }

    #[test]
    fn sampling_defaults_come_from_the_generation_config() {
        let d = scratch("sampling");
        std::fs::write(d.join("generation_config.json"), r#"{"do_sample": true, "temperature": 0.6, "top_k": 20, "top_p": 0.95, "repetition_penalty": 1.05}"#).unwrap();
        let got = SamplingDefaults::from_generation_config(&d).unwrap();
        assert_eq!(got, SamplingDefaults { temp: Some(0.6), top_k: Some(20), top_p: Some(0.95), min_p: None, repetition_penalty: Some(1.05) });
        std::fs::write(d.join("generation_config.json"), r#"{"do_sample": false, "top_p": 0.9}"#).unwrap();
        assert_eq!(SamplingDefaults::from_generation_config(&d).unwrap().temp, Some(0.0), "do_sample false is greedy");
        std::fs::write(d.join("generation_config.json"), r#"{"eos_token_id": 2}"#).unwrap();
        assert!(SamplingDefaults::from_generation_config(&d).unwrap().is_empty(), "a file that declares nothing leaves every constant alone");
        std::fs::write(d.join("generation_config.json"), r#"{"top_p": 7}"#).unwrap();
        assert!(SamplingDefaults::from_generation_config(&d).unwrap_err().contains("top_p"), "an out-of-range value is refused by name");
        std::fs::write(d.join("generation_config.json"), r#"{"top_k": "twenty"}"#).unwrap();
        assert!(SamplingDefaults::from_generation_config(&d).unwrap_err().contains("top_k"));
        assert!(SamplingDefaults::from_generation_config(&d.join("missing")).unwrap().is_empty());
    }

    #[test]
    fn sampling_defaults_come_from_the_gguf_sampling_keys() {
        let kv = |key: &str| match key {
            "general.sampling.temp" => Some(1.0),
            "general.sampling.top_k" => Some(20.0),
            "general.sampling.top_p" => Some(0.95),
            _ => None,
        };
        assert_eq!(SamplingDefaults::from_gguf(kv).unwrap(), SamplingDefaults { temp: Some(1.0), top_k: Some(20), top_p: Some(0.95), min_p: None, repetition_penalty: None });
        assert!(SamplingDefaults::from_gguf(|_| None).unwrap().is_empty());
        assert!(SamplingDefaults::from_gguf(|k| (k == "general.sampling.top_p").then_some(-1.0)).unwrap_err().contains("top_p"));
    }

    #[test]
    fn eos_ids_come_from_the_generation_config_as_one_id_or_a_list() {
        let d = scratch("list");
        std::fs::write(d.join("generation_config.json"), r#"{"bos_token_id": 1, "eos_token_id": [2, 3]}"#).unwrap();
        assert_eq!(GenerationConfig::read(&d).unwrap().unwrap(), GenerationConfig { eos_token_ids: vec![2, 3], bos_token_id: Some(1) });
        std::fs::write(d.join("generation_config.json"), r#"{"eos_token_id": 4}"#).unwrap();
        assert_eq!(stop_ids(Some(&d), &tok(), None, None).unwrap(), vec![4]);
        assert!(GenerationConfig::read(&scratch("absent")).unwrap().is_none());
    }

    /// Without a generation config the tokenizer config names the token; a
    /// chat format's end-of-turn marker joins whatever the checkpoint gives.
    #[test]
    fn the_tokenizer_config_and_the_chat_format_fill_in() {
        let d = scratch("tokcfg");
        std::fs::write(d.join("tokenizer_config.json"), r#"{"eos_token": {"content": "<|EOT|>"}}"#).unwrap();
        assert_eq!(stop_ids(Some(&d), &tok(), None, Some("<|im_end|>")).unwrap(), vec![4, 3]);
        assert_eq!(stop_ids(None, &tok(), Some(2), Some("<|im_end|>")).unwrap(), vec![2, 3]);
        assert_eq!(stop_ids(None, &tok(), None, Some("<|missing|>")).unwrap(), Vec::<u32>::new());
    }

    /// The real checkpoints, when present: coder-instruct ends on `<|EOT|>`,
    /// R1-Distill-Llama on `<｜end▁of▁sentence｜>`.
    #[test]
    fn deepseek_checkpoints_declare_their_own_stops() {
        for (repo, want) in [("deepseek-coder-1.3b-instruct", 32021u32), ("DeepSeek-R1-Distill-Llama-8B", 128001)] {
            let Some(dir) = brain_testutil::model_dir(&format!("deepseek-ai/{repo}")) else { continue };
            let dir = Path::new(&dir);
            let Ok(t) = QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()) else { continue };
            assert!(stop_ids(Some(dir), &t, None, None).unwrap().contains(&want), "{repo}");
        }
    }
}
