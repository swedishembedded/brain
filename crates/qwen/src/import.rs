// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import a HuggingFace Qwen3 checkpoint (`config.json` + `model.safetensors`)
//! into a brain `.weights` container.
//!
//! Convention match (verified): brain's `matmul.wgsl` is `out = x @ Wᵀ` with
//! `W:[out,in]` row-major — exactly HF `nn.Linear.weight`. The embedding table
//! is `[vocab, hidden]` row-major in both. So **no tensor is transposed**; the
//! import is a pure 1:1 name remap + bf16→f32 dequant. Tied embeddings: the
//! `lm_head.weight` tensor (if present) is dropped — the model reuses
//! `tok.weight` for the head.

use std::collections::HashMap;
use std::path::Path;

use crate::config::QwenConfig;

/// Map an HF Qwen3 tensor name to its brain parameter name, or `None` to drop it
/// (e.g. a tied `lm_head.weight`, handled by reusing `tok.weight`).
fn hf_to_brain(name: &str, tie: bool) -> Option<String> {
    if name == "model.embed_tokens.weight" {
        return Some("tok.weight".to_string());
    }
    if name == "model.norm.weight" {
        return Some("norm.weight".to_string());
    }
    if name == "lm_head.weight" {
        return if tie { None } else { Some("lm_head.weight".to_string()) };
    }
    // Per-layer: model.layers.{N}.<rest>
    let rest = name.strip_prefix("model.layers.")?;
    let (n, rest) = rest.split_once('.')?;
    let leaf = match rest {
        "input_layernorm.weight" => "ln1.weight".to_string(),
        "post_attention_layernorm.weight" => "ln2.weight".to_string(),
        "self_attn.q_proj.weight" => "attn.wq.weight".to_string(),
        "self_attn.k_proj.weight" => "attn.wk.weight".to_string(),
        "self_attn.v_proj.weight" => "attn.wv.weight".to_string(),
        "self_attn.o_proj.weight" => "attn.wo.weight".to_string(),
        "self_attn.q_norm.weight" => "attn.q_norm.weight".to_string(),
        "self_attn.k_norm.weight" => "attn.k_norm.weight".to_string(),
        "mlp.gate_proj.weight" => "mlp.gate.weight".to_string(),
        "mlp.up_proj.weight" => "mlp.up.weight".to_string(),
        "mlp.down_proj.weight" => "mlp.down.weight".to_string(),
        _ => return None, // unknown per-layer tensor (e.g. a bias Qwen3 doesn't have)
    };
    Some(format!("blocks.{n}.{leaf}"))
}

/// Read an HF `config.json` into a [`QwenConfig`]. `block_size` defaults to 2048
/// (the actual inference/training sequence length is chosen at load time, not
/// from `max_position_embeddings`, which would size buffers absurdly).
pub fn config_from_hf(json: &str) -> Result<QwenConfig, String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let g = |k: &str| v[k].as_u64().map(|x| x as u32);
    let cfg = QwenConfig {
        vocab: g("vocab_size").ok_or("config: vocab_size")?,
        block_size: 2048,
        n_layers: g("num_hidden_layers").ok_or("config: num_hidden_layers")?,
        d_model: g("hidden_size").ok_or("config: hidden_size")?,
        n_heads: g("num_attention_heads").ok_or("config: num_attention_heads")?,
        n_kv_heads: g("num_key_value_heads").ok_or("config: num_key_value_heads")?,
        head_dim: g("head_dim").unwrap_or(0), // 0 -> derived in with_defaults
        d_ff: g("intermediate_size").ok_or("config: intermediate_size")?,
        rope_theta: v["rope_theta"].as_f64().unwrap_or(1.0e6) as f32,
        rms_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
        tie_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(true),
        lora: None,
    }
    .with_defaults();
    Ok(cfg)
}

/// Import `<hf_dir>/config.json` + `<hf_dir>/model.safetensors` into the brain
/// checkpoint `out_path`. Validates that every brain parameter is produced
/// exactly once with the right element count; fails loudly otherwise (never
/// writes a partial checkpoint).
pub fn import(hf_dir: &str, out_path: &str) -> Result<(), String> {
    import_with_block(hf_dir, out_path, None)
}

/// Like [`import`] but overrides the checkpoint's `block_size` (max context the
/// model is built with). For RoPE the value is not a hard positional limit —
/// inference sizes context via `load_inference(.., t)` — so a smaller value is a
/// cheaper fine-tuning window (attention is O(T²)); `None` keeps the HF default.
pub fn import_with_block(hf_dir: &str, out_path: &str, block_size: Option<u32>) -> Result<(), String> {
    let dir = Path::new(hf_dir);
    let cfg_json = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("read config.json: {e}"))?;
    let mut cfg = config_from_hf(&cfg_json)?;
    if let Some(b) = block_size {
        cfg.block_size = b;
    }

    let st_path = dir.join("model.safetensors");
    if !st_path.exists() {
        return Err(format!(
            "missing {}: sharded checkpoints (model.safetensors.index.json) are not yet supported",
            st_path.display()
        ));
    }
    let tensors = checkpoint::safetensors::read(st_path.to_str().unwrap())?;

    // Remap into a name -> (shape, data) table.
    let mut brain: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    let mut dropped = 0usize;
    for t in tensors {
        match hf_to_brain(&t.name, cfg.tie_embeddings) {
            Some(bn) => {
                if brain.insert(bn.clone(), (t.shape, t.data)).is_some() {
                    return Err(format!("duplicate mapping to {bn}"));
                }
            }
            None => dropped += 1,
        }
    }

    // Validate coverage against the model's parameter list and build the ordered
    // tensor list for the checkpoint.
    let mut out: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    for (name, numel) in cfg.param_list() {
        let (_, data) = brain
            .remove(&name)
            .ok_or_else(|| format!("import: missing tensor for brain param {name}"))?;
        if data.len() != numel {
            return Err(format!(
                "import: {name} element count {} != expected {numel}",
                data.len()
            ));
        }
        out.push((name, vec![numel as u64], data));
    }
    if !brain.is_empty() {
        let extra: Vec<&String> = brain.keys().collect();
        return Err(format!("import: {} mapped HF tensors unused: {extra:?}", brain.len()));
    }

    checkpoint::save(out_path, cfg.to_json(), &out);
    eprintln!(
        "imported {} tensors -> {out_path} ({} HF tensors dropped: tied lm_head/etc.)",
        out.len(),
        dropped
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_mapping() {
        assert_eq!(hf_to_brain("model.embed_tokens.weight", true).unwrap(), "tok.weight");
        assert_eq!(hf_to_brain("model.norm.weight", true).unwrap(), "norm.weight");
        assert_eq!(hf_to_brain("lm_head.weight", true), None); // tied -> dropped
        assert_eq!(hf_to_brain("lm_head.weight", false).unwrap(), "lm_head.weight");
        assert_eq!(
            hf_to_brain("model.layers.5.self_attn.q_proj.weight", true).unwrap(),
            "blocks.5.attn.wq.weight"
        );
        assert_eq!(
            hf_to_brain("model.layers.0.self_attn.k_norm.weight", true).unwrap(),
            "blocks.0.attn.k_norm.weight"
        );
        assert_eq!(
            hf_to_brain("model.layers.27.mlp.down_proj.weight", true).unwrap(),
            "blocks.27.mlp.down.weight"
        );
    }

    #[test]
    fn parse_qwen3_config() {
        let json = r#"{"vocab_size":151936,"hidden_size":1024,"num_hidden_layers":28,
            "num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,
            "intermediate_size":3072,"rope_theta":1000000,"rms_norm_eps":1e-6,
            "tie_word_embeddings":true}"#;
        let cfg = config_from_hf(json).unwrap();
        assert_eq!(cfg.d_model, 1024);
        assert_eq!(cfg.n_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.d_ff, 3072);
        assert!(cfg.tie_embeddings);
    }
}
