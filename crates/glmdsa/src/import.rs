// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import a HuggingFace GLM-5.2 (`glm_moe_dsa`) checkpoint (`config.json` +
//! single or sharded `model.safetensors`) into a brain `.safetensors` container.
//!
//! Convention: brain's `matmul` is `out = x·Wᵀ` with `W:[out,in]` row-major -
//! exactly HF `nn.Linear.weight` - so linears are **not transposed**. Two HF
//! structures need reshaping into brain's split-projection layout (see
//! `config.rs`):
//!   * **Row de-interleave**: HF `q_b_proj` `[H*(nope+rope), q_lora]`,
//!     `kv_b_proj` `[H*(nope+v), kv_lora]`, and `kv_a_proj_with_mqa`
//!     `[kv_lora+rope, d]` are split per-head (or by prefix) into brain's
//!     contiguous `q_b_nope`/`q_b_rope`, `kv_b_nope`/`kv_b_v`, `kv_a_c`/`kv_a_rope`.
//!   * **Packed experts**: HF stores routed experts as 3D `experts.gate_up_proj`
//!     `[E, 2*moe_ff, d]` (gate‖up fused) and `experts.down_proj` `[E, d, moe_ff]`;
//!     brain uses per-expert `gate`/`up`/`down`.
//!
//! Phase-2 tensors (the DSA `indexer.*`) and any MTP (`layers.{n_layers}.*`) are
//! dropped - the Phase-1 model does not carry them.

use std::collections::HashMap;
use std::path::Path;

use checkpoint::remap::{Fetch, RemapSource};
use checkpoint::TensorSource;

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

/// The rows of a `[H*(a+b), in]` row-major matrix that belong to each head's
/// first `a` rows (`second == false`) or its following `b` rows, as the fetch
/// that concatenates them head by head: HF interleaves them per head, brain
/// keeps each part contiguous.
fn head_rows(name: &str, h: usize, a: usize, b: usize, inw: usize, second: bool) -> Fetch {
    let (off, rows) = if second { (a, b) } else { (0, a) };
    Fetch::Concat((0..h).map(|head| Fetch::Slice { name: name.to_string(), start: (head * (a + b) + off) * inw, len: rows * inw }).collect())
}

/// Where each brain parameter of `cfg` lives in a Hugging Face GLM checkpoint
/// whose tensors are `names`: the plan [`source`] reads the checkpoint
/// through, as downloaded.
///
/// HF keeps two structures brain splits (see the module doc): the per-head
/// interleaved `q_b_proj`/`kv_b_proj` rows and the `kv_a_proj_with_mqa`
/// prefix/suffix become contiguous slices, and the packed routed experts
/// become one slice per expert. MTP layers (index `>= n_layers`) and a tied
/// head have no brain parameter.
pub fn plan<'n>(names: impl IntoIterator<Item = &'n str>, cfg: &GlmConfig) -> Result<HashMap<String, Fetch>, String> {
    let d = cfg.d_model as usize;
    let h = cfg.n_heads as usize;
    let nope = cfg.qk_nope_head_dim as usize;
    let rope = cfg.qk_rope_head_dim as usize;
    let vhd = cfg.v_head_dim as usize;
    let kvl = cfg.kv_lora_rank as usize;
    let ql = cfg.q_lora_rank as usize;
    let moe_ff = cfg.moe_intermediate_size as usize;
    let mut plan = HashMap::new();
    let whole = |n: &str| Fetch::Whole(n.to_string());
    for n in names {
        match n {
            "model.embed_tokens.weight" => {
                plan.insert("tok.weight".to_string(), whole(n));
                continue;
            }
            "model.norm.weight" => {
                plan.insert("norm.weight".to_string(), whole(n));
                continue;
            }
            "lm_head.weight" => {
                if !cfg.tie_embeddings {
                    plan.insert("lm_head.weight".to_string(), whole(n));
                }
                continue;
            }
            _ => {}
        }
        let Some((li, leaf)) = n.strip_prefix("model.layers.").and_then(|r| r.split_once('.')) else { continue };
        let layer: u32 = li.parse().map_err(|_| format!("bad layer index in {n}"))?;
        if layer >= cfg.n_layers {
            continue; // MTP layers: brain's MTP head is its own, not imported
        }
        let bp = |s: &str| format!("blocks.{layer}.{s}");
        let one = |brain: &str| vec![(bp(brain), whole(n))];
        let fetches = match leaf {
            "input_layernorm.weight" => one("input_ln.weight"),
            "post_attention_layernorm.weight" => one("post_ln.weight"),
            "self_attn.q_a_proj.weight" => one("attn.q_a.weight"),
            "self_attn.q_a_layernorm.weight" => one("attn.q_a_norm.weight"),
            "self_attn.kv_a_layernorm.weight" => one("attn.kv_a_norm.weight"),
            "self_attn.o_proj.weight" => one("attn.o.weight"),
            // DSA indexer (only present on "full" layers in HF)
            "self_attn.indexer.wq_b.weight" => one("idx.wq_b.weight"),
            "self_attn.indexer.wk.weight" => one("idx.wk.weight"),
            "self_attn.indexer.k_norm.weight" => one("idx.k_norm.weight"),
            "self_attn.indexer.k_norm.bias" => one("idx.k_norm.bias"),
            "self_attn.indexer.weights_proj.weight" => one("idx.weights_proj.weight"),
            "self_attn.q_b_proj.weight" => vec![(bp("attn.q_b_nope.weight"), head_rows(n, h, nope, rope, ql, false)), (bp("attn.q_b_rope.weight"), head_rows(n, h, nope, rope, ql, true))],
            "self_attn.kv_b_proj.weight" => vec![(bp("attn.kv_b_nope.weight"), head_rows(n, h, nope, vhd, kvl, false)), (bp("attn.kv_b_v.weight"), head_rows(n, h, nope, vhd, kvl, true))],
            // [(kv_lora+rope), d] -> kv_a_c [kv_lora,d] (prefix) + kv_a_rope [rope,d] (suffix)
            "self_attn.kv_a_proj_with_mqa.weight" => vec![
                (bp("attn.kv_a_c.weight"), Fetch::Slice { name: n.to_string(), start: 0, len: kvl * d }),
                (bp("attn.kv_a_rope.weight"), Fetch::Slice { name: n.to_string(), start: kvl * d, len: rope * d }),
            ],
            // dense MLP (first_k_dense layers)
            "mlp.gate_proj.weight" => one("mlp.gate.weight"),
            "mlp.up_proj.weight" => one("mlp.up.weight"),
            "mlp.down_proj.weight" => one("mlp.down.weight"),
            // MoE router + shared expert
            "mlp.gate.weight" => one("moe.router.weight"),
            "mlp.gate.e_score_correction_bias" => one("moe.router.bias"),
            "mlp.shared_experts.gate_proj.weight" => one("moe.shared.gate.weight"),
            "mlp.shared_experts.up_proj.weight" => one("moe.shared.up.weight"),
            "mlp.shared_experts.down_proj.weight" => one("moe.shared.down.weight"),
            // packed routed experts: gate_up_proj [E, 2*moe_ff, d] (gate || up per
            // expert), down_proj [E, d, moe_ff]
            "mlp.experts.gate_up_proj" => {
                let per = 2 * moe_ff * d;
                (0..cfg.n_routed_experts as usize)
                    .flat_map(|ei| {
                        [
                            (bp(&format!("moe.experts.{ei}.gate.weight")), Fetch::Slice { name: n.to_string(), start: ei * per, len: moe_ff * d }),
                            (bp(&format!("moe.experts.{ei}.up.weight")), Fetch::Slice { name: n.to_string(), start: ei * per + moe_ff * d, len: moe_ff * d }),
                        ]
                    })
                    .collect()
            }
            "mlp.experts.down_proj" => (0..cfg.n_routed_experts as usize)
                .map(|ei| (bp(&format!("moe.experts.{ei}.down.weight")), Fetch::Slice { name: n.to_string(), start: ei * d * moe_ff, len: d * moe_ff }))
                .collect(),
            _ => Vec::new(), // an HF tensor the Phase-1 model does not carry
        };
        for (brain, fetch) in fetches {
            if plan.insert(brain.clone(), fetch).is_some() {
                return Err(format!("glm import: two checkpoint tensors map to {brain}"));
            }
        }
    }
    Ok(plan)
}

/// A Hugging Face GLM checkpoint read under brain's names, as downloaded:
/// every parameter of `cfg` present at its size, checked before a byte of
/// weight data is read.
pub fn source(reader: checkpoint::weightio::WeightReader, cfg: &GlmConfig) -> Result<RemapSource<'static>, String> {
    let plan = plan(reader.names(), cfg)?;
    let src = RemapSource::owning(Box::new(reader), plan);
    src.validate(&cfg.param_list())?;
    Ok(src)
}

/// The config and the brain-named source of the Hugging Face GLM checkpoint
/// directory `dir`.
pub fn open_dir(dir: &Path) -> Result<(GlmConfig, RemapSource<'static>), String> {
    let cfg = config_from_hf(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("{}: {e}", dir.join("config.json").display()))?)?;
    let reader = checkpoint::weightio::WeightReader::open_hf_dir(dir).map_err(|e| format!("open {}: {e}", dir.display()))?;
    Ok((cfg.clone(), source(reader, &cfg)?))
}

/// Import `<hf_dir>` (config.json + single/sharded safetensors) into `out_path`.
/// Validates full coverage of the model's parameter list; fails loudly (never
/// writes a partial checkpoint). Streams one brain parameter at a time through
/// [`source`].
pub fn import(hf_dir: &str, out_path: &str) -> Result<(), String> {
    import_as(hf_dir, out_path, None)
}

/// Like [`import`] but overrides the card's `id` (defaults to the output
/// filename stem).
pub fn import_as(hf_dir: &str, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
    let (cfg, src) = open_dir(Path::new(hf_dir))?;
    let plan: Vec<(String, Vec<u64>)> = cfg.param_list().into_iter().map(|(name, numel)| (name, vec![numel as u64])).collect();
    let param_count: u64 = plan.iter().map(|(_, s)| s.iter().product::<u64>()).sum();
    let id = id_override.unwrap_or_else(|| Path::new(out_path).file_stem().and_then(|s| s.to_str()).unwrap_or("glm"));
    let mut card = checkpoint::st::ModelCard::new(id, "glm");
    card.context_length = Some(cfg.block_size as u64);
    card.param_count = Some(param_count);
    let mut writer = checkpoint::weightio::StWriter::create(out_path, &plan, &cfg.to_json(), Some(&card)).map_err(|e| format!("create {out_path}: {e}"))?;
    for (name, _) in &plan {
        let mut result = Ok(());
        if !src.with_tensor(name, &mut |t| result = writer.write(name, t).map_err(|e| e.to_string())) {
            return Err(format!("glm import: {name} has no source"));
        }
        result?;
    }
    writer.finish().map_err(|e| e.to_string())?;
    eprintln!("imported {} tensors -> {out_path}", plan.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // ---- streaming import() parity: synthetic tiny HF checkpoint, 2 routed experts ----

    fn seq(base: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| base + i as f32).collect()
    }

    /// Build a tiny 2-layer (1 dense + 1 MoE, 2 routed experts) synthetic HF-shaped
    /// checkpoint dir: config.json + a single model.safetensors. Layer 1's packed
    /// `gate_up_proj`/`down_proj` are hand-crafted (not `seq`) so expert 0 vs 1's
    /// gate/up/down values are distinguishable and can't pass a swapped/aliased test.
    fn build_tiny_hf_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-glm-import-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let json = r#"{"vocab_size":5,"hidden_size":8,"num_hidden_layers":2,
            "num_attention_heads":2,"q_lora_rank":4,"kv_lora_rank":4,
            "qk_nope_head_dim":2,"qk_rope_head_dim":2,"v_head_dim":2,
            "n_routed_experts":2,"n_shared_experts":1,"num_experts_per_tok":1,
            "moe_intermediate_size":4,"intermediate_size":6,"first_k_dense_replace":1,
            "n_group":1,"topk_group":1,"routed_scaling_factor":2.5,"norm_topk_prob":true,
            "rms_norm_eps":1e-5,"tie_word_embeddings":false,
            "index_topk":99999,"index_n_heads":2,"index_head_dim":2,
            "indexer_types":["shared","shared"]}"#;
        std::fs::write(dir.join("config.json"), json).unwrap();

        // moe_ff=4, d=8 -> per-expert gate_up slab = 2*4*8 = 64; gate/up = 32 each.
        let mut gate_up = Vec::new();
        gate_up.extend(seq(0.0, 32)); // expert0 gate
        gate_up.extend(seq(100.0, 32)); // expert0 up
        gate_up.extend(seq(10000.0, 32)); // expert1 gate
        gate_up.extend(seq(10100.0, 32)); // expert1 up
        // per-expert down slab = d*moe_ff = 32.
        let mut down = Vec::new();
        down.extend(seq(200.0, 32)); // expert0 down
        down.extend(seq(10200.0, 32)); // expert1 down

        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
            ("model.embed_tokens.weight".into(), vec![40], seq(1_000_000.0, 40)),
            ("model.norm.weight".into(), vec![8], seq(2_000_000.0, 8)),
            ("lm_head.weight".into(), vec![40], seq(3_000_000.0, 40)),
            // layer 0: dense
            ("model.layers.0.input_layernorm.weight".into(), vec![8], seq(10.0, 8)),
            ("model.layers.0.self_attn.q_a_proj.weight".into(), vec![32], seq(20.0, 32)),
            ("model.layers.0.self_attn.q_a_layernorm.weight".into(), vec![4], seq(30.0, 4)),
            ("model.layers.0.self_attn.q_b_proj.weight".into(), vec![32], seq(40.0, 32)),
            ("model.layers.0.self_attn.kv_a_proj_with_mqa.weight".into(), vec![48], seq(50.0, 48)),
            ("model.layers.0.self_attn.kv_a_layernorm.weight".into(), vec![4], seq(60.0, 4)),
            ("model.layers.0.self_attn.kv_b_proj.weight".into(), vec![32], seq(70.0, 32)),
            ("model.layers.0.self_attn.o_proj.weight".into(), vec![32], seq(80.0, 32)),
            ("model.layers.0.post_attention_layernorm.weight".into(), vec![8], seq(90.0, 8)),
            ("model.layers.0.mlp.gate_proj.weight".into(), vec![48], seq(100.0, 48)),
            ("model.layers.0.mlp.up_proj.weight".into(), vec![48], seq(150.0, 48)),
            ("model.layers.0.mlp.down_proj.weight".into(), vec![48], seq(200.0, 48)),
            // layer 1: MoE
            ("model.layers.1.input_layernorm.weight".into(), vec![8], seq(300.0, 8)),
            ("model.layers.1.self_attn.q_a_proj.weight".into(), vec![32], seq(310.0, 32)),
            ("model.layers.1.self_attn.q_a_layernorm.weight".into(), vec![4], seq(320.0, 4)),
            ("model.layers.1.self_attn.q_b_proj.weight".into(), vec![32], seq(330.0, 32)),
            ("model.layers.1.self_attn.kv_a_proj_with_mqa.weight".into(), vec![48], seq(340.0, 48)),
            ("model.layers.1.self_attn.kv_a_layernorm.weight".into(), vec![4], seq(350.0, 4)),
            ("model.layers.1.self_attn.kv_b_proj.weight".into(), vec![32], seq(360.0, 32)),
            ("model.layers.1.self_attn.o_proj.weight".into(), vec![32], seq(370.0, 32)),
            ("model.layers.1.post_attention_layernorm.weight".into(), vec![8], seq(380.0, 8)),
            ("model.layers.1.mlp.gate.weight".into(), vec![16], seq(390.0, 16)),
            ("model.layers.1.mlp.gate.e_score_correction_bias".into(), vec![2], seq(400.0, 2)),
            ("model.layers.1.mlp.experts.gate_up_proj".into(), vec![128], gate_up),
            ("model.layers.1.mlp.experts.down_proj".into(), vec![64], down),
            ("model.layers.1.mlp.shared_experts.gate_proj.weight".into(), vec![32], seq(410.0, 32)),
            ("model.layers.1.mlp.shared_experts.up_proj.weight".into(), vec![32], seq(420.0, 32)),
            ("model.layers.1.mlp.shared_experts.down_proj.weight".into(), vec![32], seq(430.0, 32)),
        ];
        checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &tensors, &serde_json::Value::Null, None)
            .unwrap();
        dir
    }

    #[test]
    fn import_streams_and_matches_param_list_with_expert_fan_out_not_swapped() {
        let dir = build_tiny_hf_dir("streaming");
        let out = std::env::temp_dir().join(format!("brain-glm-import-streaming-out-{}.st", std::process::id()));
        let out_str = out.to_str().unwrap();

        import(dir.to_str().unwrap(), out_str).expect("streaming import");

        let cfg = config_from_hf(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        let m = checkpoint::st::load_safetensors(out_str).unwrap();

        // Full coverage: every param_list name present, right length, no extras.
        let expected = cfg.param_list();
        assert_eq!(m.tensors.len(), expected.len());
        for (name, numel) in &expected {
            let data = m.tensors.get(name).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(data.len(), *numel, "{name}");
        }

        // Spot check plain 1:1 tensors (untied here, so lm_head.weight is written too).
        assert_eq!(m.tensors["tok.weight"], seq(1_000_000.0, 40));
        assert_eq!(m.tensors["lm_head.weight"], seq(3_000_000.0, 40));

        // Packed-expert fan-out: expert 0 and expert 1 must land un-swapped, un-aliased.
        assert_eq!(m.tensors["blocks.1.moe.experts.0.gate.weight"], seq(0.0, 32));
        assert_eq!(m.tensors["blocks.1.moe.experts.0.up.weight"], seq(100.0, 32));
        assert_eq!(m.tensors["blocks.1.moe.experts.1.gate.weight"], seq(10000.0, 32));
        assert_eq!(m.tensors["blocks.1.moe.experts.1.up.weight"], seq(10100.0, 32));
        assert_eq!(m.tensors["blocks.1.moe.experts.0.down.weight"], seq(200.0, 32));
        assert_eq!(m.tensors["blocks.1.moe.experts.1.down.weight"], seq(10200.0, 32));

        // Per-head de-interleave: q_b_proj rows are [h0 nope | h0 rope | h1 nope | h1 rope].
        let q_b = seq(40.0, 32);
        let ql = cfg.q_lora_rank as usize;
        let (nope, rope) = (cfg.qk_nope_head_dim as usize, cfg.qk_rope_head_dim as usize);
        let rows = |head: usize, off: usize, n: usize| q_b[(head * (nope + rope) + off) * ql..(head * (nope + rope) + off + n) * ql].to_vec();
        assert_eq!(m.tensors["blocks.0.attn.q_b_nope.weight"], [rows(0, 0, nope), rows(1, 0, nope)].concat());
        assert_eq!(m.tensors["blocks.0.attn.q_b_rope.weight"], [rows(0, nope, rope), rows(1, nope, rope)].concat());

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_file(&out).ok();
    }

    /// Serving reads the downloaded directory itself: a model loaded from it
    /// computes exactly what one loaded from the imported file computes, and
    /// nothing is written beside the download.
    #[test]
    fn the_downloaded_directory_serves_as_its_import_does() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let dir = build_tiny_hf_dir("served");
        let out = std::env::temp_dir().join(format!("brain-glm-import-served-{}.st", std::process::id()));
        import(dir.to_str().unwrap(), out.to_str().unwrap()).unwrap();
        let before: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        let tokens = [1u32, 3, 2];
        let bits = |m: crate::model::Glm| m.logits_all(&tokens).into_iter().map(f32::to_bits).collect::<Vec<_>>();
        let from_dir = bits(crate::model::Glm::load_inference(dir.to_str().unwrap(), 1, 4));
        let from_file = bits(crate::model::Glm::load_inference(out.to_str().unwrap(), 1, 4));
        assert_eq!(from_dir, from_file);
        let after: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(after, before);
        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_file(&out).ok();
    }
}
