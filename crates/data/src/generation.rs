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
