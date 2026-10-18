// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniCPM-V 4.6's `config.json`: the vision tower, the two mergers and the
//! Qwen3.5 decoder. A required key that is missing is an error naming it.
//! The only defaults taken are the reference configuration class's own
//! (`MiniCPMV4_6Config`/`MiniCPMV4_6VisionConfig`), for keys the released
//! file leaves out: `merge_kernel_size`, `merger_times`,
//! `window_kernel_size` and `downsample_mode`.

use std::path::Path;

use qwen35::config::Qwen35Config;
use serde_json::Value;

/// The NaViT SigLIP tower and its in-tower window merger.
#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    pub n_layers: u32,
    pub hidden: u32,
    pub heads: u32,
    pub mlp: u32,
    pub patch: u32,
    /// Side of the learned position table in patches (`image_size / patch`):
    /// 70 for the released 980-pixel table.
    pub pos_side: u32,
    pub eps: f32,
    /// Encoder layer (0-based) after which the window merger runs in `16x`
    /// mode.
    pub insert_layer_id: u32,
    /// Side of the window merger's square window, in patches.
    pub window: u32,
}

impl VisionConfig {
    pub fn head_dim(&self) -> u32 {
        self.hidden / self.heads
    }

    /// Width of the window merger's concatenated window (`hidden * window^2`).
    pub fn window_hidden(&self) -> u32 {
        self.hidden * self.window * self.window
    }

    /// The window merger's MLP width (`mlp * window^2`).
    pub fn window_mlp(&self) -> u32 {
        self.mlp * self.window * self.window
    }
}

/// How many visual tokens each view keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Downsample {
    /// The in-tower window merger and the merger: one token per 4x4 patches.
    X16,
    /// The merger only: one token per 2x2 patches.
    X4,
}

impl Downsample {
    pub fn parse(s: &str) -> Result<Downsample, String> {
        match s {
            "16x" => Ok(Downsample::X16),
            "4x" => Ok(Downsample::X4),
            other => Err(format!("downsample mode {other:?}: expected \"16x\" or \"4x\"")),
        }
    }

    /// Patches per visual token.
    pub fn divisor(self) -> u32 {
        match self {
            Downsample::X16 => 16,
            Downsample::X4 => 4,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Downsample::X16 => "16x",
            Downsample::X4 => "4x",
        }
    }
}

/// The whole checkpoint configuration.
#[derive(Clone, Debug)]
pub struct MiniCpmVConfig {
    pub vision: VisionConfig,
    pub text: Qwen35Config,
    /// Side of the merger's square merge kernel, in patches.
    pub merge: u32,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub downsample: Downsample,
}

fn at<'a>(v: &'a Value, path: &str) -> Result<&'a Value, String> {
    path.split('.').try_fold(v, |v, key| v.get(key).filter(|x| !x.is_null()).ok_or_else(|| format!("config.json: missing '{path}'")))
}

fn u32_at(v: &Value, path: &str) -> Result<u32, String> {
    at(v, path)?.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| format!("config.json: '{path}' is not an unsigned integer"))
}

/// A square `[k, k]` kernel field, or the reference default when absent.
fn square_or(v: &Value, key: &str, default: u32) -> Result<u32, String> {
    match v.get(key).filter(|x| !x.is_null()) {
        None => Ok(default),
        Some(a) => {
            let k: Vec<u64> = a.as_array().map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
            match k[..] {
                [h, w] if h == w && h > 0 => Ok(h as u32),
                _ => Err(format!("config.json: '{key}' must be a square [k, k] kernel, got {a}")),
            }
        }
    }
}

impl MiniCpmVConfig {
    /// Read `dir/config.json`.
    pub fn from_dir(dir: &Path) -> Result<MiniCpmVConfig, String> {
        let path = dir.join("config.json");
        let top: Value = serde_json::from_slice(&std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let text = qwen35::fp8_source::hf_config(dir)?;
        MiniCpmVConfig::from_json(&top, text)
    }

    /// The configuration from the parsed `config.json` and its decoder.
    pub fn from_json(top: &Value, text: Qwen35Config) -> Result<MiniCpmVConfig, String> {
        let model_type = at(top, "model_type")?.as_str().unwrap_or("");
        if model_type != "minicpmv4_6" {
            return Err(format!("config.json: model_type {model_type:?} is not \"minicpmv4_6\""));
        }
        let act = at(top, "vision_config.hidden_act")?.as_str().unwrap_or("");
        if act != "gelu_pytorch_tanh" {
            return Err(format!("config.json: vision_config.hidden_act {act:?} is not the tanh GELU this tower implements"));
        }
        let merger_times = top.get("merger_times").and_then(Value::as_u64).unwrap_or(1);
        if merger_times != 1 {
            return Err(format!("config.json: merger_times {merger_times} - only the single merge round of the released checkpoint is implemented"));
        }
        let patch = u32_at(top, "vision_config.patch_size")?;
        let image_size = u32_at(top, "vision_config.image_size")?;
        if patch == 0 || image_size % patch != 0 {
            return Err(format!("config.json: vision image_size {image_size} is not a multiple of patch_size {patch}"));
        }
        let vision_cfg = &top["vision_config"];
        let vision = VisionConfig {
            n_layers: u32_at(top, "vision_config.num_hidden_layers")?,
            hidden: u32_at(top, "vision_config.hidden_size")?,
            heads: u32_at(top, "vision_config.num_attention_heads")?,
            mlp: u32_at(top, "vision_config.intermediate_size")?,
            patch,
            pos_side: image_size / patch,
            eps: at(top, "vision_config.layer_norm_eps")?.as_f64().ok_or("config.json: 'vision_config.layer_norm_eps' is not a number")? as f32,
            insert_layer_id: u32_at(top, "insert_layer_id")?,
            window: square_or(vision_cfg, "window_kernel_size", 2)?,
        };
        if !vision.hidden.is_multiple_of(vision.heads) {
            return Err(format!("config.json: vision hidden {} is not divisible by {} heads", vision.hidden, vision.heads));
        }
        if vision.insert_layer_id >= vision.n_layers {
            return Err(format!("config.json: insert_layer_id {} is past the tower's {} layers", vision.insert_layer_id, vision.n_layers));
        }
        let downsample = Downsample::parse(top.get("downsample_mode").and_then(Value::as_str).unwrap_or("16x"))?;
        Ok(MiniCpmVConfig {
            vision,
            merge: square_or(top, "merge_kernel_size", 2)?,
            image_token_id: u32_at(top, "image_token_id")?,
            video_token_id: u32_at(top, "video_token_id")?,
            downsample,
            text,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn released() -> Value {
        serde_json::json!({
            "model_type": "minicpmv4_6", "insert_layer_id": 6, "image_token_id": 248056, "video_token_id": 248057,
            "vision_config": {"hidden_act": "gelu_pytorch_tanh", "hidden_size": 1152, "image_size": 980,
                "intermediate_size": 4304, "layer_norm_eps": 1e-6, "num_attention_heads": 16,
                "num_hidden_layers": 27, "patch_size": 14},
        })
    }

    #[test]
    fn the_released_config_reads_with_the_reference_defaults() {
        let c = MiniCpmVConfig::from_json(&released(), Qwen35Config::tiny()).unwrap();
        assert_eq!(c.vision.pos_side, 70);
        assert_eq!(c.vision.head_dim(), 72);
        assert_eq!((c.vision.window, c.merge, c.downsample), (2, 2, Downsample::X16));
        assert_eq!((c.vision.window_hidden(), c.vision.window_mlp()), (4608, 17216));
    }

    #[test]
    fn a_missing_key_or_an_unimplemented_variant_is_refused_by_name() {
        let mut v = released();
        v["vision_config"].as_object_mut().unwrap().remove("num_hidden_layers");
        assert!(MiniCpmVConfig::from_json(&v, Qwen35Config::tiny()).unwrap_err().contains("vision_config.num_hidden_layers"));
        let mut v = released();
        v["merger_times"] = serde_json::json!(2);
        assert!(MiniCpmVConfig::from_json(&v, Qwen35Config::tiny()).unwrap_err().contains("merger_times"));
        let mut v = released();
        v["downsample_mode"] = serde_json::json!("8x");
        assert!(MiniCpmVConfig::from_json(&v, Qwen35Config::tiny()).unwrap_err().contains("8x"));
    }
}
