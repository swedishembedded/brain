// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import a HuggingFace LFM2.5-Encoder checkpoint (`config.json` +
//! `model.safetensors`) into a brain `.weights` container.
//!
//! Convention match (same as qwen): brain's `matmul.wgsl` is `out = x @ Wᵀ`
//! with `W:[out,in]` row-major — exactly HF `nn.Linear.weight`; the embedding
//! table is `[vocab, hidden]` in both; the depthwise conv weight `[d,1,k]` is
//! stored flat as `[d,k]`. **No tensor is transposed.** The head is tied
//! (`lm_head.weight` never ships in the checkpoint; the model reuses
//! `tok.weight`).

use std::collections::HashMap;
use std::path::Path;

use crate::config::{adjust_ff_dim, LayerType, LfmConfig};

/// Map an HF LFM2.5 tensor name to its brain parameter name, or `None` to drop
/// it (nothing is expected to drop for these checkpoints; unknown names fail
/// coverage instead of being silently ignored).
fn hf_to_brain(name: &str) -> Option<String> {
    if name == "lfm2.embed_tokens.weight" {
        return Some("tok.weight".to_string());
    }
    if name == "lfm2.embedding_norm.weight" {
        return Some("norm.weight".to_string());
    }
    if name == "lm_head.weight" {
        return None; // tied: model reuses tok.weight
    }
    let rest = name.strip_prefix("lfm2.layers.")?;
    let (n, rest) = rest.split_once('.')?;
    let leaf = match rest {
        "operator_norm.weight" => "ln1.weight",
        "ffn_norm.weight" => "ln2.weight",
        // Gated short-conv mixer.
        "conv.in_proj.weight" => "conv.in_proj.weight",
        "conv.conv.weight" => "conv.conv.weight",
        "conv.out_proj.weight" => "conv.out_proj.weight",
        // Bidirectional GQA attention.
        "self_attn.q_proj.weight" => "attn.wq.weight",
        "self_attn.k_proj.weight" => "attn.wk.weight",
        "self_attn.v_proj.weight" => "attn.wv.weight",
        "self_attn.out_proj.weight" => "attn.wo.weight",
        "self_attn.q_layernorm.weight" => "attn.q_norm.weight",
        "self_attn.k_layernorm.weight" => "attn.k_norm.weight",
        // SwiGLU: w1 = gate (SiLU side), w3 = up, w2 = down.
        "feed_forward.w1.weight" => "mlp.gate.weight",
        "feed_forward.w3.weight" => "mlp.up.weight",
        "feed_forward.w2.weight" => "mlp.down.weight",
        _ => return None,
    };
    Some(format!("blocks.{n}.{leaf}"))
}

/// Read an HF LFM2.5 `config.json` into an [`LfmConfig`]. The FFN width is
/// resolved through the `block_auto_adjust_ff_dim` rule so the stored config
/// always carries the *effective* `d_ff`. `block_size` defaults to the model's
/// usable context (8192); the runtime sequence length is chosen at load time.
pub fn config_from_hf(json: &str) -> Result<LfmConfig, String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let g = |k: &str| v[k].as_u64().map(|x| x as u32);
    let layer_types: Vec<LayerType> = v["layer_types"]
        .as_array()
        .ok_or("config: layer_types")?
        .iter()
        .map(|t| match t.as_str() {
            Some("full_attention") => Ok(LayerType::Attention),
            Some("conv") => Ok(LayerType::Conv),
            other => Err(format!("config: unknown layer_type {other:?}")),
        })
        .collect::<Result<_, _>>()?;
    let n_layers = g("num_hidden_layers").ok_or("config: num_hidden_layers")?;
    if layer_types.len() != n_layers as usize {
        return Err(format!(
            "config: layer_types len {} != num_hidden_layers {n_layers}",
            layer_types.len()
        ));
    }
    let d_model = g("hidden_size").ok_or("config: hidden_size")?;
    let n_heads = g("num_attention_heads").ok_or("config: num_attention_heads")?;
    let d_ff = adjust_ff_dim(
        g("intermediate_size").ok_or("config: intermediate_size")?,
        v["block_auto_adjust_ff_dim"].as_bool().unwrap_or(false),
        v["block_ffn_dim_multiplier"].as_f64().unwrap_or(1.0),
        v["block_multiple_of"].as_u64().unwrap_or(256) as u32,
    );
    Ok(LfmConfig {
        vocab: g("vocab_size").ok_or("config: vocab_size")?,
        block_size: 8192,
        d_model,
        n_heads,
        n_kv_heads: g("num_key_value_heads").ok_or("config: num_key_value_heads")?,
        head_dim: g("head_dim").unwrap_or(d_model / n_heads),
        d_ff,
        conv_k: g("conv_L_cache").ok_or("config: conv_L_cache")?,
        rope_theta: v["rope_parameters"]["rope_theta"]
            .as_f64()
            .or_else(|| v["rope_theta"].as_f64())
            .unwrap_or(1.0e6) as f32,
        norm_eps: v["norm_eps"].as_f64().unwrap_or(1e-5) as f32,
        tie_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(true),
        layer_types,
    })
}

/// Remap HF LFM2.5 safetensors into brain's `name → f32 data` init map,
/// validating full coverage against `cfg.param_list()` (every brain parameter
/// produced exactly once with the right element count) and that no mapped HF
/// tensor is left over. Fails loudly.
pub fn brain_init_from_hf(
    tensors: Vec<checkpoint::safetensors::StTensor>,
    cfg: &LfmConfig,
) -> Result<HashMap<String, Vec<f32>>, String> {
    let mut brain: HashMap<String, Vec<f32>> = HashMap::new();
    let mut unmapped: Vec<String> = Vec::new();
    for t in tensors {
        match hf_to_brain(&t.name) {
            Some(bn) => {
                if brain.insert(bn.clone(), t.data).is_some() {
                    return Err(format!("duplicate mapping to {bn}"));
                }
            }
            None if t.name == "lm_head.weight" => {} // tied, deliberately dropped
            None => unmapped.push(t.name),
        }
    }
    if !unmapped.is_empty() {
        return Err(format!("import: {} unmapped HF tensors: {unmapped:?}", unmapped.len()));
    }
    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    for (name, numel) in cfg.param_list() {
        let data = brain
            .remove(&name)
            .ok_or_else(|| format!("import: missing tensor for brain param {name}"))?;
        if data.len() != numel {
            return Err(format!("import: {name} element count {} != expected {numel}", data.len()));
        }
        init.insert(name, data);
    }
    if !brain.is_empty() {
        let extra: Vec<&String> = brain.keys().collect();
        return Err(format!("import: {} mapped HF tensors unused: {extra:?}", brain.len()));
    }
    Ok(init)
}

/// Import `<hf_dir>/config.json` + `model.safetensors` into the brain
/// checkpoint `out_path`. Never writes a partial checkpoint.
pub fn import(hf_dir: &str, out_path: &str) -> Result<(), String> {
    let dir = Path::new(hf_dir);
    let cfg_json = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("read config.json: {e}"))?;
    let cfg = config_from_hf(&cfg_json)?;

    let tensors = checkpoint::safetensors::read_model_dir(dir)?;
    let init = brain_init_from_hf(tensors, &cfg)?;

    let mut out: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    for (name, numel) in cfg.param_list() {
        let data = init.get(&name).expect("coverage validated").clone();
        out.push((name, vec![numel as u64], data));
    }
    checkpoint::save(out_path, cfg.to_json(), &out);
    eprintln!("imported {} tensors -> {out_path}", out.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_mapping() {
        assert_eq!(hf_to_brain("lfm2.embed_tokens.weight").unwrap(), "tok.weight");
        assert_eq!(hf_to_brain("lfm2.embedding_norm.weight").unwrap(), "norm.weight");
        assert_eq!(hf_to_brain("lm_head.weight"), None);
        assert_eq!(
            hf_to_brain("lfm2.layers.0.conv.conv.weight").unwrap(),
            "blocks.0.conv.conv.weight"
        );
        assert_eq!(
            hf_to_brain("lfm2.layers.2.self_attn.q_layernorm.weight").unwrap(),
            "blocks.2.attn.q_norm.weight"
        );
        assert_eq!(
            hf_to_brain("lfm2.layers.13.feed_forward.w1.weight").unwrap(),
            "blocks.13.mlp.gate.weight"
        );
        assert_eq!(hf_to_brain("lfm2.layers.1.operator_norm.weight").unwrap(), "blocks.1.ln1.weight");
    }

    #[test]
    fn parse_lfm_config_230m_shape() {
        let json = r#"{"vocab_size":65536,"hidden_size":1024,"num_hidden_layers":3,
            "num_attention_heads":16,"num_key_value_heads":8,"intermediate_size":2560,
            "block_auto_adjust_ff_dim":false,"block_multiple_of":256,
            "conv_L_cache":3,"norm_eps":1e-5,
            "rope_parameters":{"rope_theta":1000000.0,"rope_type":"default"},
            "tie_word_embeddings":true,
            "layer_types":["conv","conv","full_attention"]}"#;
        let cfg = config_from_hf(json).unwrap();
        assert_eq!(cfg.d_model, 1024);
        assert_eq!(cfg.head_dim, 64); // derived: 1024/16
        assert_eq!(cfg.d_ff, 2560);
        assert_eq!(cfg.rope_theta, 1.0e6);
        assert_eq!(cfg.layer_types, vec![LayerType::Conv, LayerType::Conv, LayerType::Attention]);
    }

    #[test]
    fn parse_lfm_config_350m_ff_adjust() {
        let json = r#"{"vocab_size":65536,"hidden_size":1024,"num_hidden_layers":1,
            "num_attention_heads":16,"num_key_value_heads":8,"intermediate_size":6656,
            "block_auto_adjust_ff_dim":true,"block_ffn_dim_multiplier":1.0,
            "block_multiple_of":256,"conv_L_cache":3,
            "layer_types":["conv"]}"#;
        let cfg = config_from_hf(json).unwrap();
        assert_eq!(cfg.d_ff, 4608); // matches the real 350M checkpoint shapes
    }
}
