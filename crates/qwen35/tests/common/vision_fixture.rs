// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Random weights for a tiny vision tower and its merger, shared by the tests that splice one into a tiny decoder.

#![allow(dead_code)]

use std::collections::HashMap;

use data::rng::Rng;
use qwen3vl::config::VisionConfig;

fn rand_map(mut rng: Rng, specs: &[(&str, usize, bool)]) -> HashMap<String, Vec<f32>> {
    let mut m = HashMap::new();
    for &(name, n, ones) in specs {
        let v = if ones { vec![1.0; n] } else { (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect() };
        m.insert(name.to_string(), v);
    }
    m
}

pub fn vision_weights(vcfg: &VisionConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let (c, pv, mlp) = (vcfg.hidden as usize, vcfg.patch_vec_dim() as usize, vcfg.intermediate as usize);
    let mut specs: Vec<(String, usize, bool)> = vec![
        ("patch_embed.weight".into(), c * pv, false),
        ("patch_embed.bias".into(), c, false),
        ("pos_embed".into(), vcfg.num_position_embeddings as usize * c, false),
    ];
    for b in 0..vcfg.depth {
        specs.extend([
            (format!("blocks.{b}.norm1.weight"), c, true),
            (format!("blocks.{b}.norm1.bias"), c, false),
            (format!("blocks.{b}.qkv.weight"), 3 * c * c, false),
            (format!("blocks.{b}.qkv.bias"), 3 * c, false),
            (format!("blocks.{b}.proj.weight"), c * c, false),
            (format!("blocks.{b}.proj.bias"), c, false),
            (format!("blocks.{b}.norm2.weight"), c, true),
            (format!("blocks.{b}.norm2.bias"), c, false),
            (format!("blocks.{b}.fc1.weight"), mlp * c, false),
            (format!("blocks.{b}.fc1.bias"), mlp, false),
            (format!("blocks.{b}.fc2.weight"), c * mlp, false),
            (format!("blocks.{b}.fc2.bias"), c, false),
        ]);
    }
    let refs: Vec<(&str, usize, bool)> = specs.iter().map(|(n, s, o)| (n.as_str(), *s, *o)).collect();
    rand_map(Rng::new(seed), &refs)
}

pub fn merger_weights(in_dim: u32, merge: u32, out_dim: u32, seed: u64) -> HashMap<String, Vec<f32>> {
    let merged = in_dim * merge * merge;
    rand_map(
        Rng::new(seed),
        &[
            ("ln.weight", in_dim as usize, true),
            ("ln.bias", in_dim as usize, false),
            ("fc1.weight", (merged * merged) as usize, false),
            ("fc1.bias", merged as usize, false),
            ("fc2.weight", (out_dim * merged) as usize, false),
            ("fc2.bias", out_dim as usize, false),
        ],
    )
}


// ---- a tiny checkpoint that reads images ------------------------------------------------------------------------

use std::path::Path;

use imaging::pixels::Rgb8;
use qwen3vl::import::TowerWeights;
use qwen3vl::preprocess::ImageGeometry;
use qwen35::vision_tower::{VisionSpec, VisionTokens};

/// The placeholder tokens of the tiny vocabulary (`Qwen35Config::tiny().vocab` is 29).
pub const TOKENS: VisionTokens = VisionTokens { start: 26, image: 27, end: 28 };

/// A tower whose merger output width matches `Qwen35Config::tiny().d_model`, reading RGB patches.
pub fn tiny_image_vcfg() -> VisionConfig {
    VisionConfig {
        depth: 2,
        hidden: 16,
        num_heads: 2,
        intermediate: 32,
        patch_size: 2,
        temporal_patch_size: 1,
        spatial_merge_size: 2,
        num_position_embeddings: 16,
        out_hidden_size: 96,
        in_channels: 3,
        tokens_per_second: 2,
        deepstack_indexes: vec![],
    }
}

/// The preprocessing geometry of the tiny checkpoint: an 8 x 8 picture is a 4 x 4 patch grid, 2 x 2 = 4 decoder rows.
pub fn tiny_geometry() -> ImageGeometry {
    ImageGeometry { patch_size: 2, merge_size: 2, temporal_patch_size: 1, min_pixels: 16, max_pixels: 4096, mean: [0.5; 3], std: [0.25; 3] }
}

pub fn tiny_spec() -> VisionSpec {
    VisionSpec { config: tiny_image_vcfg(), geometry: tiny_geometry(), tokens: TOKENS }
}

pub fn tiny_tower_weights(seed: u64) -> TowerWeights {
    let cfg = tiny_image_vcfg();
    TowerWeights { vision: vision_weights(&cfg, seed), merger: merger_weights(cfg.hidden, cfg.spatial_merge_size, cfg.out_hidden_size, seed ^ 0x55) }
}

/// A picture of noise, `seed` deciding which.
pub fn picture(w: u32, h: u32, seed: u64) -> Rgb8 {
    let mut rng = Rng::new(seed);
    Rgb8::new(w, h, (0..w * h * 3).map(|_| (rng.next_f32() * 255.0) as u8).collect()).unwrap()
}

/// `img` as the bytes of a binary PPM file, a format the image decoder reads.
pub fn ppm(img: &Rgb8) -> Vec<u8> {
    let mut bytes = format!("P6\n{} {}\n255\n", img.w, img.h).into_bytes();
    bytes.extend_from_slice(&img.px);
    bytes
}

/// The HF name of the tower tensor the encoder calls `key`.
fn hf_name(key: &str, merger: bool) -> String {
    if merger {
        let leaf = match key {
            "ln.weight" => "norm.weight",
            "ln.bias" => "norm.bias",
            "fc1.weight" => "linear_fc1.weight",
            "fc1.bias" => "linear_fc1.bias",
            "fc2.weight" => "linear_fc2.weight",
            "fc2.bias" => "linear_fc2.bias",
            other => panic!("no merger tensor {other}"),
        };
        return format!("model.visual.merger.{leaf}");
    }
    match key {
        "patch_embed.weight" => return "model.visual.patch_embed.proj.weight".into(),
        "patch_embed.bias" => return "model.visual.patch_embed.proj.bias".into(),
        "pos_embed" => return "model.visual.pos_embed.weight".into(),
        _ => {}
    }
    let (n, leaf) = key.strip_prefix("blocks.").and_then(|s| s.split_once('.')).expect("a block tensor");
    let hf = match leaf {
        "qkv.weight" | "qkv.bias" => format!("attn.{leaf}"),
        "proj.weight" | "proj.bias" => format!("attn.{leaf}"),
        "fc1.weight" | "fc1.bias" | "fc2.weight" | "fc2.bias" => format!("mlp.linear_{leaf}"),
        other => other.to_string(),
    };
    format!("model.visual.blocks.{n}.{hf}")
}

/// Write the vision side of a release into `dir`: `config.json` (the vision config and placeholder token ids),
/// `preprocessor_config.json` and the `model.visual.*` tensors in `vision.safetensors`, as the released checkpoint
/// names them, listed in the shard index. A `config.json` already there keeps its other keys, and so does an index:
/// the decoder files of a release (or a brain-native file beside the vision files) are left as they are.
pub fn add_vision_release(dir: &Path, seed: u64) {
    std::fs::create_dir_all(dir).unwrap();
    let (cfg, g) = (tiny_image_vcfg(), tiny_geometry());
    let vision_config = serde_json::json!({
        "depth": cfg.depth, "hidden_size": cfg.hidden, "num_heads": cfg.num_heads, "intermediate_size": cfg.intermediate,
        "patch_size": cfg.patch_size, "temporal_patch_size": cfg.temporal_patch_size, "spatial_merge_size": cfg.spatial_merge_size,
        "num_position_embeddings": cfg.num_position_embeddings, "out_hidden_size": cfg.out_hidden_size, "in_channels": cfg.in_channels,
    });
    let read_json = |name: &str, empty: serde_json::Value| std::fs::read(dir.join(name)).ok().map(|b| serde_json::from_slice(&b).unwrap()).unwrap_or(empty);
    let mut config = read_json("config.json", serde_json::json!({}));
    config["vision_config"] = vision_config;
    config["image_token_id"] = TOKENS.image.into();
    config["vision_start_token_id"] = TOKENS.start.into();
    config["vision_end_token_id"] = TOKENS.end.into();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let pre = serde_json::json!({
        "patch_size": g.patch_size, "merge_size": g.merge_size, "temporal_patch_size": g.temporal_patch_size,
        "size": {"shortest_edge": g.min_pixels, "longest_edge": g.max_pixels}, "image_mean": g.mean, "image_std": g.std,
    });
    std::fs::write(dir.join("preprocessor_config.json"), pre.to_string()).unwrap();
    let w = tiny_tower_weights(seed);
    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    for (key, data) in &w.vision {
        tensors.push((hf_name(key, false), vec![data.len() as u64], data.clone()));
    }
    for (key, data) in &w.merger {
        tensors.push((hf_name(key, true), vec![data.len() as u64], data.clone()));
    }
    checkpoint::st::save_safetensors(dir.join("vision.safetensors").to_str().unwrap(), &tensors, &serde_json::json!({}), None).unwrap();
    let mut index = read_json("model.safetensors.index.json", serde_json::json!({"weight_map": {}}));
    for (name, _, _) in &tensors {
        index["weight_map"][name] = "vision.safetensors".into();
    }
    std::fs::write(dir.join("model.safetensors.index.json"), index.to_string()).unwrap();
}
