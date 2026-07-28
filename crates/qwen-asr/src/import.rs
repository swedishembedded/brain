// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Load a Qwen3-ASR checkpoint's audio encoder + projector into the weight map
//! [`crate::encoder::AudioEncoder`] consumes. HF names are `model.audio_tower.*`
//! and `model.multi_modal_projector.*`. The per-layer `self_attn.{q,k,v}_proj`
//! are fused into a single `blocks.{b}.qkv` (weights stacked `[3·d, d]`, biases
//! `[3·d]`) to match the `model::vit` block layout. No transposes (brain `matmul`
//! is `x·Wᵀ` with HF's `[out, in]` layout). The Qwen3 decoder half is loaded
//! separately via `qwen`-style mapping under `model.language_model.`.

use std::collections::HashMap;
use std::path::Path;

use crate::config::AudioEncoderConfig;

/// Read `dir` (single or sharded safetensors) and return the audio-encoder weight
/// map keyed as `AudioEncoder::new` expects.
pub fn load_audio_encoder(dir: &Path, cfg: &AudioEncoderConfig) -> Result<HashMap<String, Vec<f32>>, String> {
    let tensors = checkpoint::safetensors::read_model_dir(dir)?;
    let mut src: HashMap<String, Vec<f32>> = HashMap::new();
    for t in tensors {
        src.insert(t.name, t.data);
    }
    Ok(map_audio_encoder(&src, cfg))
}

/// Pure name/shape remap (testable without a checkpoint on disk).
pub fn map_audio_encoder(src: &HashMap<String, Vec<f32>>, cfg: &AudioEncoderConfig) -> HashMap<String, Vec<f32>> {
    let get = |name: &str| -> Vec<f32> { src.get(name).unwrap_or_else(|| panic!("Qwen3-ASR tensor missing: {name}")).clone() };
    let mut w = HashMap::new();

    // conv stem + conv_out
    for i in 1..=3 {
        w.insert(format!("conv2d{i}.weight"), get(&format!("model.audio_tower.conv2d{i}.weight")));
        w.insert(format!("conv2d{i}.bias"), get(&format!("model.audio_tower.conv2d{i}.bias")));
    }
    w.insert("conv_out.weight".into(), get("model.audio_tower.conv_out.weight"));

    // transformer blocks (fuse q/k/v)
    for b in 0..cfg.n_layers {
        let a = |leaf: &str| get(&format!("model.audio_tower.layers.{b}.{leaf}"));
        let mut qkv_w = a("self_attn.q_proj.weight");
        qkv_w.extend(a("self_attn.k_proj.weight"));
        qkv_w.extend(a("self_attn.v_proj.weight"));
        let mut qkv_b = a("self_attn.q_proj.bias");
        qkv_b.extend(a("self_attn.k_proj.bias"));
        qkv_b.extend(a("self_attn.v_proj.bias"));
        w.insert(format!("blocks.{b}.qkv.weight"), qkv_w);
        w.insert(format!("blocks.{b}.qkv.bias"), qkv_b);
        w.insert(format!("blocks.{b}.proj.weight"), a("self_attn.out_proj.weight"));
        w.insert(format!("blocks.{b}.proj.bias"), a("self_attn.out_proj.bias"));
        w.insert(format!("blocks.{b}.norm1.weight"), a("self_attn_layer_norm.weight"));
        w.insert(format!("blocks.{b}.norm1.bias"), a("self_attn_layer_norm.bias"));
        w.insert(format!("blocks.{b}.norm2.weight"), a("final_layer_norm.weight"));
        w.insert(format!("blocks.{b}.norm2.bias"), a("final_layer_norm.bias"));
        w.insert(format!("blocks.{b}.fc1.weight"), a("fc1.weight"));
        w.insert(format!("blocks.{b}.fc1.bias"), a("fc1.bias"));
        w.insert(format!("blocks.{b}.fc2.weight"), a("fc2.weight"));
        w.insert(format!("blocks.{b}.fc2.bias"), a("fc2.bias"));
    }

    // final norm + projector
    w.insert("ln_post.weight".into(), get("model.audio_tower.ln_post.weight"));
    w.insert("ln_post.bias".into(), get("model.audio_tower.ln_post.bias"));
    for i in 1..=2 {
        w.insert(format!("multi_modal_projector.linear_{i}.weight"), get(&format!("model.multi_modal_projector.linear_{i}.weight")));
        w.insert(format!("multi_modal_projector.linear_{i}.bias"), get(&format!("model.multi_modal_projector.linear_{i}.bias")));
    }
    w
}
