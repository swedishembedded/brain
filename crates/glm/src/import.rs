// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import a HuggingFace GLM-5.2 (`glm_moe_dsa`) checkpoint (`config.json` +
//! single or sharded `model.safetensors`) into a brain `.safetensors` container.
//!
//! Convention: brain's `matmul` is `out = x·Wᵀ` with `W:[out,in]` row-major —
//! exactly HF `nn.Linear.weight` — so linears are **not transposed**. Two HF
//! structures need reshaping into brain's split-projection layout (see
//! `config.rs`):
//!   * **Row de-interleave**: HF `q_b_proj` `[H*(nope+rope), q_lora]`,
//!     `kv_b_proj` `[H*(nope+v), kv_lora]`, and `kv_a_proj_with_mqa`
//!     `[kv_lora+rope, d]` are split per-head (or by prefix) into brain's
//!     contiguous `q_b_nope`/`q_b_rope`, `kv_b_nope`/`kv_b_v`, `kv_a_c`/`kv_a_rope`.
//!   * **Packed experts**: HF stores routed experts as 3D `experts.gate_up_proj`
//!     `[E, 2*moe_ff, d]` (gate‖up fused) and `experts.down_proj` `[E, d, moe_ff]`;
//!     brain uses per-expert `gate`/`up`/`down`.
//! Phase-2 tensors (the DSA `indexer.*`) and any MTP (`layers.{n_layers}.*`) are
//! dropped — the Phase-1 model does not carry them.

use std::collections::HashMap;
use std::path::Path;

use crate::config::GlmConfig;

/// Read an HF `config.json` into a [`GlmConfig`]. `block_size` defaults to 4096
/// (the real sequence length is chosen at load time, not from
/// `max_position_embeddings`, which would size buffers absurdly).
pub fn config_from_hf(json: &str) -> Result<GlmConfig, String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let g = |k: &str| v[k].as_u64().map(|x| x as u32);
    let mut cfg = GlmConfig::glm5_2();
    cfg.block_size = 4096;
    cfg.vocab = g("vocab_size").ok_or("config: vocab_size")?;
    cfg.n_layers = g("num_hidden_layers").ok_or("config: num_hidden_layers")?;
    cfg.d_model = g("hidden_size").ok_or("config: hidden_size")?;
    cfg.n_heads = g("num_attention_heads").ok_or("config: num_attention_heads")?;
    cfg.q_lora_rank = g("q_lora_rank").ok_or("config: q_lora_rank")?;
    cfg.kv_lora_rank = g("kv_lora_rank").ok_or("config: kv_lora_rank")?;
    cfg.qk_nope_head_dim = g("qk_nope_head_dim").ok_or("config: qk_nope_head_dim")?;
    cfg.qk_rope_head_dim = g("qk_rope_head_dim").ok_or("config: qk_rope_head_dim")?;
    cfg.v_head_dim = g("v_head_dim").ok_or("config: v_head_dim")?;
    cfg.n_routed_experts = g("n_routed_experts").ok_or("config: n_routed_experts")?;
    cfg.n_shared_experts = g("n_shared_experts").unwrap_or(1);
    cfg.num_experts_per_tok = g("num_experts_per_tok").ok_or("config: num_experts_per_tok")?;
    cfg.moe_intermediate_size = g("moe_intermediate_size").ok_or("config: moe_intermediate_size")?;
    cfg.intermediate_size = g("intermediate_size").ok_or("config: intermediate_size")?;
    cfg.first_k_dense_replace = g("first_k_dense_replace").unwrap_or(3);
    cfg.n_group = g("n_group").unwrap_or(1);
    cfg.topk_group = g("topk_group").unwrap_or(1);
    cfg.routed_scaling_factor = v["routed_scaling_factor"].as_f64().unwrap_or(2.5) as f32;
    cfg.norm_topk_prob = v["norm_topk_prob"].as_bool().unwrap_or(true);
    cfg.rms_eps = v["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32;
    cfg.tie_embeddings = v["tie_word_embeddings"].as_bool().unwrap_or(false);
    // brain's MTP head is a simplified position-wise block, not the reference's
    // full decoder layer, so the HF MTP weights are not imported (the MTP layer
    // tensors at index `num_hidden_layers` are dropped); the main model imports
    // normally and MTP, if wanted, is trained from scratch.
    cfg.mtp = false;
    if let Some(rp) = v["rope_parameters"]["rope_theta"].as_f64().or_else(|| v["rope_theta"].as_f64()) {
        cfg.rope_theta = rp as f32;
    }
    cfg.index_topk = g("index_topk").unwrap_or(2048);
    cfg.index_n_heads = g("index_n_heads").unwrap_or(32);
    cfg.index_head_dim = g("index_head_dim").unwrap_or(128);
    // Per-layer indexer schedule: "full" runs its own indexer, "shared" reuses
    // the previous full layer's top-k (IndexShare). If absent, derive from the
    // freq/offset schedule (index_topk_freq / index_skip_topk_offset).
    cfg.indexer_full = if let Some(types) = v["indexer_types"].as_array() {
        types.iter().map(|x| x.as_str() == Some("full")).collect()
    } else {
        let freq = g("index_topk_freq").unwrap_or(1).max(1);
        let offset = g("index_skip_topk_offset").unwrap_or(2);
        (0..cfg.n_layers).map(|i| (i.saturating_sub(offset) + 1) % freq == 0).collect()
    };
    Ok(cfg)
}

/// De-interleave the per-head rows of a `[H*(a+bdim), in]` row-major matrix into
/// two contiguous `[H*a, in]` and `[H*bdim, in]` matrices (head h contributes its
/// first `a` output rows to the first, its next `bdim` to the second).
fn split_heads(src: &[f32], h: usize, a: usize, bdim: usize, inw: usize) -> (Vec<f32>, Vec<f32>) {
    let mut first = vec![0.0f32; h * a * inw];
    let mut second = vec![0.0f32; h * bdim * inw];
    for head in 0..h {
        let base = head * (a + bdim);
        for r in 0..a {
            let s = (base + r) * inw;
            let d = (head * a + r) * inw;
            first[d..d + inw].copy_from_slice(&src[s..s + inw]);
        }
        for r in 0..bdim {
            let s = (base + a + r) * inw;
            let d = (head * bdim + r) * inw;
            second[d..d + inw].copy_from_slice(&src[s..s + inw]);
        }
    }
    (first, second)
}

/// Import `<hf_dir>` (config.json + single/sharded safetensors) into `out_path`.
/// Validates full coverage of the model's parameter list; fails loudly (never
/// writes a partial checkpoint).
pub fn import(hf_dir: &str, out_path: &str) -> Result<(), String> {
    let dir = Path::new(hf_dir);
    let cfg_json = std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("read config.json: {e}"))?;
    let cfg = config_from_hf(&cfg_json)?;
    let tensors = checkpoint::safetensors::read_model_dir(dir)?;

    let d = cfg.d_model as usize;
    let h = cfg.n_heads as usize;
    let nope = cfg.qk_nope_head_dim as usize;
    let rope = cfg.qk_rope_head_dim as usize;
    let vhd = cfg.v_head_dim as usize;
    let kvl = cfg.kv_lora_rank as usize;
    let moe_ff = cfg.moe_intermediate_size as usize;

    let mut brain: HashMap<String, Vec<f32>> = HashMap::new();
    let mut put = |name: String, data: Vec<f32>| -> Result<(), String> {
        if brain.insert(name.clone(), data).is_some() {
            return Err(format!("duplicate mapping to {name}"));
        }
        Ok(())
    };
    let mut dropped = 0usize;

    for t in &tensors {
        let n = t.name.as_str();
        if n == "model.embed_tokens.weight" {
            put("tok.weight".into(), t.data.clone())?;
            continue;
        }
        if n == "model.norm.weight" {
            put("norm.weight".into(), t.data.clone())?;
            continue;
        }
        if n == "lm_head.weight" {
            if cfg.tie_embeddings {
                dropped += 1;
            } else {
                put("lm_head.weight".into(), t.data.clone())?;
            }
            continue;
        }
        let Some(rest) = n.strip_prefix("model.layers.") else {
            dropped += 1;
            continue;
        };
        let Some((li, leaf)) = rest.split_once('.') else {
            dropped += 1;
            continue;
        };
        let layer: u32 = li.parse().map_err(|_| format!("bad layer index in {n}"))?;
        if layer >= cfg.n_layers {
            dropped += 1; // MTP / extra head layers
            continue;
        }
        let bp = |s: &str| format!("blocks.{layer}.{s}");
        match leaf {
            "input_layernorm.weight" => put(bp("input_ln.weight"), t.data.clone())?,
            "post_attention_layernorm.weight" => put(bp("post_ln.weight"), t.data.clone())?,
            "self_attn.q_a_proj.weight" => put(bp("attn.q_a.weight"), t.data.clone())?,
            "self_attn.q_a_layernorm.weight" => put(bp("attn.q_a_norm.weight"), t.data.clone())?,
            "self_attn.kv_a_layernorm.weight" => put(bp("attn.kv_a_norm.weight"), t.data.clone())?,
            "self_attn.o_proj.weight" => put(bp("attn.o.weight"), t.data.clone())?,
            // DSA indexer (only present on "full" layers in HF)
            "self_attn.indexer.wq_b.weight" => put(bp("idx.wq_b.weight"), t.data.clone())?,
            "self_attn.indexer.wk.weight" => put(bp("idx.wk.weight"), t.data.clone())?,
            "self_attn.indexer.k_norm.weight" => put(bp("idx.k_norm.weight"), t.data.clone())?,
            "self_attn.indexer.k_norm.bias" => put(bp("idx.k_norm.bias"), t.data.clone())?,
            "self_attn.indexer.weights_proj.weight" => put(bp("idx.weights_proj.weight"), t.data.clone())?,
            "self_attn.q_b_proj.weight" => {
                let (nope_w, rope_w) = split_heads(&t.data, h, nope, rope, cfg.q_lora_rank as usize);
                put(bp("attn.q_b_nope.weight"), nope_w)?;
                put(bp("attn.q_b_rope.weight"), rope_w)?;
            }
            "self_attn.kv_b_proj.weight" => {
                let (nope_w, v_w) = split_heads(&t.data, h, nope, vhd, kvl);
                put(bp("attn.kv_b_nope.weight"), nope_w)?;
                put(bp("attn.kv_b_v.weight"), v_w)?;
            }
            "self_attn.kv_a_proj_with_mqa.weight" => {
                // [(kv_lora+rope), d] -> kv_a_c [kv_lora,d] (prefix) + kv_a_rope [rope,d] (suffix)
                let (c_w, rope_w) = split_heads(&t.data, 1, kvl, rope, d);
                put(bp("attn.kv_a_c.weight"), c_w)?;
                put(bp("attn.kv_a_rope.weight"), rope_w)?;
            }
            // dense MLP (first_k_dense layers)
            "mlp.gate_proj.weight" => put(bp("mlp.gate.weight"), t.data.clone())?,
            "mlp.up_proj.weight" => put(bp("mlp.up.weight"), t.data.clone())?,
            "mlp.down_proj.weight" => put(bp("mlp.down.weight"), t.data.clone())?,
            // MoE router + shared expert
            "mlp.gate.weight" => put(bp("moe.router.weight"), t.data.clone())?,
            "mlp.gate.e_score_correction_bias" => put(bp("moe.router.bias"), t.data.clone())?,
            "mlp.shared_experts.gate_proj.weight" => put(bp("moe.shared.gate.weight"), t.data.clone())?,
            "mlp.shared_experts.up_proj.weight" => put(bp("moe.shared.up.weight"), t.data.clone())?,
            "mlp.shared_experts.down_proj.weight" => put(bp("moe.shared.down.weight"), t.data.clone())?,
            // packed routed experts: gate_up_proj [E, 2*moe_ff, d], down_proj [E, d, moe_ff]
            "mlp.experts.gate_up_proj" => {
                let e = cfg.n_routed_experts as usize;
                let per = 2 * moe_ff * d;
                for ei in 0..e {
                    let slab = &t.data[ei * per..(ei + 1) * per];
                    let (gate, up) = (slab[..moe_ff * d].to_vec(), slab[moe_ff * d..].to_vec());
                    put(format!("blocks.{layer}.moe.experts.{ei}.gate.weight"), gate)?;
                    put(format!("blocks.{layer}.moe.experts.{ei}.up.weight"), up)?;
                }
            }
            "mlp.experts.down_proj" => {
                let e = cfg.n_routed_experts as usize;
                let per = d * moe_ff;
                for ei in 0..e {
                    put(format!("blocks.{layer}.moe.experts.{ei}.down.weight"), t.data[ei * per..(ei + 1) * per].to_vec())?;
                }
            }
            _ => dropped += 1, // indexer.*, biases GLM-5.2 doesn't have, etc.
        }
    }

    // Validate coverage against the model's parameter list, build ordered output.
    let mut out: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    for (name, numel) in cfg.param_list() {
        let data = brain.remove(&name).ok_or_else(|| format!("import: missing tensor for brain param {name}"))?;
        if data.len() != numel {
            return Err(format!("import: {name} element count {} != expected {numel}", data.len()));
        }
        out.push((name, vec![numel as u64], data));
    }
    if !brain.is_empty() {
        let extra: Vec<&String> = brain.keys().collect();
        return Err(format!("import: {} mapped HF tensors unused: {extra:?}", brain.len()));
    }
    // A card so this file auto-serves from the global model directory (P2) with
    // no BRAIN_GLM_WEIGHTS env var — id defaults to the output filename stem,
    // matching how the model dir keys catalog entries.
    let param_count: u64 = out.iter().map(|(_, shape, _)| shape.iter().product::<u64>()).sum();
    let id = Path::new(out_path).file_stem().and_then(|s| s.to_str()).unwrap_or("glm");
    let mut card = checkpoint::st::ModelCard::new(id, "glm");
    card.context_length = Some(cfg.block_size as u64);
    card.param_count = Some(param_count);
    checkpoint::st::save_safetensors(out_path, &out, &cfg.to_json(), Some(&card))
        .map_err(|e| format!("write {out_path}: {e}"))?;
    eprintln!("imported {} tensors -> {out_path} ({dropped} HF tensors dropped: indexer/MTP/tied)", out.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_heads_deinterleaves_per_head() {
        // H=2, a=2, bdim=1, inw=1 -> src rows per head = [n0, n1, r0]. Values chosen
        // so head h row r = 100*h + 10*kind + r  (kind: 0=nope, 1=rope).
        let src = vec![
            0.0, 1.0, // head0 nope rows 0,1
            10.0, // head0 rope row 0
            100.0, 101.0, // head1 nope rows 0,1
            110.0, // head1 rope row 0
        ];
        let (nope, rope) = split_heads(&src, 2, 2, 1, 1);
        assert_eq!(nope, vec![0.0, 1.0, 100.0, 101.0]); // [h0n0,h0n1,h1n0,h1n1]
        assert_eq!(rope, vec![10.0, 110.0]); // [h0r0, h1r0]
    }

    #[test]
    fn config_from_hf_parses_glm52_shape() {
        let json = r#"{"vocab_size":154880,"hidden_size":6144,"num_hidden_layers":78,
            "num_attention_heads":64,"q_lora_rank":2048,"kv_lora_rank":512,
            "qk_nope_head_dim":192,"qk_rope_head_dim":64,"v_head_dim":256,
            "n_routed_experts":256,"num_experts_per_tok":8,"moe_intermediate_size":2048,
            "intermediate_size":12288,"first_k_dense_replace":3,"norm_topk_prob":true,
            "routed_scaling_factor":2.5,"rms_norm_eps":1e-5,"tie_word_embeddings":false}"#;
        let cfg = config_from_hf(json).unwrap();
        assert_eq!(cfg.d_model, 6144);
        assert_eq!(cfg.qk_nope_head_dim, 192);
        assert_eq!(cfg.v_head_dim, 256);
        assert_eq!(cfg.n_routed_experts, 256);
        assert_eq!(cfg.first_k_dense_replace, 3);
        assert!(!cfg.tie_embeddings);
    }
}
