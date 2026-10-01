// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-VL's `config.json`: the hybrid tower's two branches, the aligner
//! and the decoder. A required key that is missing is an error naming it,
//! never a default standing in for the checkpoint's own value.

use serde_json::Value;

/// One branch of the hybrid vision tower.
#[derive(Clone, Debug, PartialEq)]
pub struct TowerBranch {
    /// The upstream model name: `sam_b_downsample` (high) or
    /// `siglip_large_patch16_384` (low).
    pub model_name: String,
    /// The square side the branch sees, in pixels.
    pub image_size: u32,
    /// The feature width it hands the aligner.
    pub output_dim: u32,
    pub pixel_mean: [f32; 3],
    pub pixel_std: [f32; 3],
    /// Which layer's output is taken (`-1`: the last).
    pub select_layer: i32,
}

/// The `MlpProjector` from vision features to the decoder's width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlignerConfig {
    /// `mlp_gelu`, or `low_high_hybrid_split_mlp_gelu` for the hybrid tower's
    /// two feature streams.
    pub projector_type: String,
    pub depth: u32,
    pub input_dim: u32,
    pub n_embed: u32,
}

/// The whole checkpoint configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct DeepseekVlConfig {
    pub high: TowerBranch,
    pub low: TowerBranch,
    pub aligner: AlignerConfig,
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

fn rgb_at(v: &Value, path: &str) -> Result<[f32; 3], String> {
    let a = at(v, path)?.as_array().filter(|a| a.len() == 3).ok_or_else(|| format!("config.json: '{path}' is not three numbers"))?;
    let n = |i: usize| a[i].as_f64().map(|x| x as f32).ok_or_else(|| format!("config.json: '{path}' is not three numbers"));
    Ok([n(0)?, n(1)?, n(2)?])
}

impl TowerBranch {
    fn parse(v: &Value, path: &str) -> Result<TowerBranch, String> {
        let select_layer = at(v, &format!("{path}.select_layer"))?.as_i64().and_then(|n| i32::try_from(n).ok());
        Ok(TowerBranch {
            model_name: str_at(v, &format!("{path}.model_name"))?,
            image_size: u32_at(v, &format!("{path}.image_size"))?,
            output_dim: u32_at(v, &format!("{path}.output_dim"))?,
            pixel_mean: rgb_at(v, &format!("{path}.pixel_mean"))?,
            pixel_std: rgb_at(v, &format!("{path}.pixel_std"))?,
            select_layer: select_layer.ok_or_else(|| format!("config.json: '{path}.select_layer' is not an integer"))?,
        })
    }
}

impl AlignerConfig {
    /// The aligner under `key` (`aligner_config`, or Janus-Pro's
    /// `gen_aligner_config`).
    pub fn parse(v: &Value, key: &str) -> Result<AlignerConfig, String> {
        Ok(AlignerConfig {
            projector_type: str_at(v, &format!("{key}.params.projector_type"))?,
            depth: u32_at(v, &format!("{key}.params.depth"))?,
            input_dim: u32_at(v, &format!("{key}.params.input_dim"))?,
            n_embed: u32_at(v, &format!("{key}.params.n_embed"))?,
        })
    }
}

impl DeepseekVlConfig {
    pub fn from_json(v: &Value) -> Result<DeepseekVlConfig, String> {
        let tower = str_at(v, "vision_config.cls")?;
        if tower != "HybridVisionTower" {
            return Err(format!("config.json: vision_config.cls is {tower:?}, not DeepSeek-VL's HybridVisionTower"));
        }
        Ok(DeepseekVlConfig {
            high: TowerBranch::parse(v, "vision_config.params.high_res_cfg")?,
            low: TowerBranch::parse(v, "vision_config.params.low_res_cfg")?,
            aligner: AlignerConfig::parse(v, "aligner_config")?,
            language: at(v, "language_config")?.clone(),
        })
    }

    /// The `config.json` in a checkpoint directory.
    pub fn from_dir(dir: &std::path::Path) -> Result<DeepseekVlConfig, String> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        DeepseekVlConfig::from_json(&v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_released_checkpoint_config_parses() {
        let Some(dir) = brain_testutil::model_dir("deepseek-ai/deepseek-vl-7b-chat").filter(|d| std::path::Path::new(d).is_dir()) else {
            brain_testutil::skip("deepseek-vl-7b-chat not downloaded");
            return;
        };
        let cfg = DeepseekVlConfig::from_dir(std::path::Path::new(&dir)).unwrap();
        assert_eq!((cfg.high.model_name.as_str(), cfg.high.image_size), ("sam_b_downsample", 1024));
        assert_eq!((cfg.low.model_name.as_str(), cfg.low.image_size), ("siglip_large_patch16_384", 384));
        assert_eq!(cfg.low.pixel_mean, [0.5; 3]);
        assert_eq!(cfg.aligner, AlignerConfig { projector_type: "low_high_hybrid_split_mlp_gelu".into(), depth: 2, input_dim: 1024, n_embed: 4096 });
        assert_eq!(cfg.language["num_hidden_layers"], 30);
    }

    #[test]
    fn a_missing_key_is_named() {
        let mut v: Value = serde_json::json!({
            "vision_config": {"cls": "HybridVisionTower", "params": {}},
            "aligner_config": {}, "language_config": {}
        });
        assert!(DeepseekVlConfig::from_json(&v).unwrap_err().contains("high_res_cfg"));
        v["vision_config"]["cls"] = "CLIPVisionTower".into();
        assert!(DeepseekVlConfig::from_json(&v).unwrap_err().contains("HybridVisionTower"));
    }
}
