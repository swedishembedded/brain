// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Janus-Pro's `config.json`: the understanding tower and aligner, the
//! generation aligner, head and image tokenizer, and the decoder. A required
//! key that is missing is an error naming it.

use deepseekvl::AlignerConfig;
use serde_json::Value;

/// The understanding tower.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionConfig {
    /// `siglip_large_patch16_384`.
    pub model_name: String,
    pub image_size: u32,
}

/// The generation head: decoder hidden state to image-token logits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenHeadConfig {
    pub n_embed: u32,
    pub image_token_embed: u32,
    /// The VQ codebook size the logits range over.
    pub image_token_size: u32,
}

/// The image tokenizer generation predicts codes of.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenVisionConfig {
    /// `VQ-16`.
    pub cls: String,
    pub image_token_size: u32,
    /// The codebook entry width.
    pub n_embed: u32,
}

/// The whole checkpoint configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct JanusProConfig {
    pub vision: VisionConfig,
    pub aligner: AlignerConfig,
    pub gen_aligner: AlignerConfig,
    pub gen_head: GenHeadConfig,
    pub gen_vision: GenVisionConfig,
    /// `language_config` as the checkpoint writes it: a partial Llama config
    /// the reference completes with `transformers.LlamaConfig`'s defaults.
    pub language: Value,
}

fn at<'a>(v: &'a Value, path: &str) -> Result<&'a Value, String> {
    path.split('.').try_fold(v, |v, key| v.get(key).ok_or_else(|| format!("config.json: missing '{path}'")))
}

fn u32_at(v: &Value, path: &str) -> Result<u32, String> {
    at(v, path)?.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| format!("config.json: '{path}' is not an unsigned integer"))
}

fn str_at(v: &Value, path: &str) -> Result<String, String> {
    at(v, path)?.as_str().map(str::to_string).ok_or_else(|| format!("config.json: '{path}' is not a string"))
}

impl JanusProConfig {
    pub fn from_json(v: &Value) -> Result<JanusProConfig, String> {
        Ok(JanusProConfig {
            vision: VisionConfig { model_name: str_at(v, "vision_config.params.model_name")?, image_size: u32_at(v, "vision_config.params.image_size")? },
            aligner: AlignerConfig::parse(v, "aligner_config")?,
            gen_aligner: AlignerConfig::parse(v, "gen_aligner_config")?,
            gen_head: GenHeadConfig {
                n_embed: u32_at(v, "gen_head_config.params.n_embed")?,
                image_token_embed: u32_at(v, "gen_head_config.params.image_token_embed")?,
                image_token_size: u32_at(v, "gen_head_config.params.image_token_size")?,
            },
            gen_vision: GenVisionConfig {
                cls: str_at(v, "gen_vision_config.cls")?,
                image_token_size: u32_at(v, "gen_vision_config.params.image_token_size")?,
                n_embed: u32_at(v, "gen_vision_config.params.n_embed")?,
            },
            language: at(v, "language_config")?.clone(),
        })
    }

    /// The `config.json` in a checkpoint directory.
    pub fn from_dir(dir: &std::path::Path) -> Result<JanusProConfig, String> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        JanusProConfig::from_json(&v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_released_checkpoint_config_parses() {
        let Some(dir) = brain_testutil::model_dir("deepseek-ai/Janus-Pro-7B") else {
            brain_testutil::skip("Janus-Pro-7B not downloaded");
            return;
        };
        let cfg = JanusProConfig::from_dir(std::path::Path::new(&dir)).unwrap();
        assert_eq!(cfg.vision, VisionConfig { model_name: "siglip_large_patch16_384".into(), image_size: 384 });
        assert_eq!((cfg.aligner.projector_type.as_str(), cfg.aligner.input_dim, cfg.aligner.n_embed), ("mlp_gelu", 1024, 4096));
        assert_eq!((cfg.gen_aligner.input_dim, cfg.gen_aligner.n_embed), (8, 4096));
        assert_eq!(cfg.gen_head, GenHeadConfig { n_embed: 4096, image_token_embed: 4096, image_token_size: 16384 });
        assert_eq!(cfg.gen_vision, GenVisionConfig { cls: "VQ-16".into(), image_token_size: 16384, n_embed: 8 });
    }

    #[test]
    fn a_config_without_generation_heads_is_not_janus_pro() {
        let err = JanusProConfig::from_json(&serde_json::json!({
            "vision_config": {"params": {"model_name": "siglip_large_patch16_384", "image_size": 384}},
            "aligner_config": {"params": {"projector_type": "mlp_gelu", "depth": 2, "input_dim": 1024, "n_embed": 4096}},
            "language_config": {}
        }))
        .unwrap_err();
        assert!(err.contains("gen_aligner_config"), "{err}");
    }
}
