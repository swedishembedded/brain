// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import a HuggingFace Qwen3 checkpoint (`config.json` + `model.safetensors`)
//! into a brain `.safetensors` container.
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
    let block_size = 2048;
    let cfg = QwenConfig {
        vocab: g("vocab_size").ok_or("config: vocab_size")?,
        block_size,
        n_layers: g("num_hidden_layers").ok_or("config: num_hidden_layers")?,
        d_model: g("hidden_size").ok_or("config: hidden_size")?,
        n_heads: g("num_attention_heads").ok_or("config: num_attention_heads")?,
        n_kv_heads: g("num_key_value_heads").ok_or("config: num_key_value_heads")?,
        head_dim: g("head_dim").unwrap_or(0), // 0 -> derived in with_defaults
        d_ff: g("intermediate_size").ok_or("config: intermediate_size")?,
        rope_theta: v["rope_theta"].as_f64().unwrap_or(1.0e6) as f32,
        rms_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
        // The HF trained RoPE extent, carried through for reference (`block_size`
        // is what actually sizes buffers — see `QwenConfig::max_position_embeddings`).
        // Older `config.json`s lacking the key, and pre-existing brain checkpoints,
        // fall back to `block_size` for backward compatibility.
        max_position_embeddings: g("max_position_embeddings").unwrap_or(block_size),
        tie_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(true),
        qk_norm: true,
        attn_bias: false,
        lora: None,
    }
    .with_defaults();
    Ok(cfg)
}

/// Remap a set of HF Qwen3 safetensors into brain's `name → f32 data` init map,
/// validating full coverage against `cfg.param_list()` (every brain parameter
/// produced exactly once with the right element count) and that no mapped HF
/// tensor is left unused. Fails loudly. Shared by the checkpoint [`import`] path
/// and by in-memory loaders (e.g. wiring the frozen text encoder directly).
pub fn brain_init_from_hf(
    tensors: Vec<checkpoint::safetensors::StTensor>,
    cfg: &QwenConfig,
) -> Result<HashMap<String, Vec<f32>>, String> {
    let mut brain: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    for t in tensors {
        if let Some(bn) = hf_to_brain(&t.name, cfg.tie_embeddings) {
            if brain.insert(bn.clone(), (t.shape, t.data)).is_some() {
                return Err(format!("duplicate mapping to {bn}"));
            }
        }
    }
    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    for (name, numel) in cfg.param_list() {
        let (_, data) = brain
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

/// The streaming sibling of [`brain_init_from_hf`]: a
/// `checkpoint::remap::RemapSource` over `r` that resolves every brain
/// parameter name to its HF tensor via [`hf_to_brain`]'s same map, validated
/// the same way (every brain param produced exactly once, right element
/// count; every mapped HF tensor recognized) — but reading no tensor data.
/// `Qwen::new_shard`/`new_shard_i8` accept the result directly, so an
/// encoder built from this never materializes the whole checkpoint on the
/// host: peak allocation is one tensor, at upload time.
pub fn hf_source<'a>(r: &'a checkpoint::weightio::WeightReader, cfg: &QwenConfig) -> Result<checkpoint::remap::RemapSource<'a>, String> {
    let want = cfg.param_list();
    let want_names: std::collections::HashSet<&str> = want.iter().map(|(n, _)| n.as_str()).collect();
    let mut plan: HashMap<String, checkpoint::remap::Fetch> = HashMap::new();
    for name in r.names() {
        let Some(bn) = hf_to_brain(name, cfg.tie_embeddings) else { continue };
        if !want_names.contains(bn.as_str()) {
            return Err(format!("import: '{name}' maps to unexpected brain param '{bn}'"));
        }
        if plan.insert(bn.clone(), checkpoint::remap::Fetch::Whole(name.to_string())).is_some() {
            return Err(format!("duplicate mapping to {bn}"));
        }
    }
    let src = checkpoint::remap::RemapSource::new(r, plan);
    src.validate(&want)?;
    Ok(src)
}

/// Import `<hf_dir>/config.json` + `model.safetensors` (single **or** sharded via
/// `model.safetensors.index.json`) into the brain checkpoint `out_path`.
/// Validates that every brain parameter is produced exactly once with the right
/// element count; fails loudly otherwise (never writes a partial checkpoint).
pub fn import(hf_dir: &str, out_path: &str) -> Result<(), String> {
    import_with_block(hf_dir, out_path, None)
}

/// Like [`import`] but overrides the checkpoint's `block_size` (max context the
/// model is built with). For RoPE the value is not a hard positional limit —
/// inference sizes context via `load_inference(.., t)` — so a smaller value is a
/// cheaper fine-tuning window (attention is O(T²)); `None` keeps the HF default.
pub fn import_with_block(hf_dir: &str, out_path: &str, block_size: Option<u32>) -> Result<(), String> {
    import_as(hf_dir, out_path, block_size, None)
}

/// Like [`import_with_block`] but overrides the card's `id` (defaults to the
/// output filename stem). Used by the model-store auto-fetch dispatcher, which
/// needs the id to be the fully-qualified `vendor/repo` reference rather than a
/// filesystem-derived name.
pub fn import_as(hf_dir: &str, out_path: &str, block_size: Option<u32>, id_override: Option<&str>) -> Result<(), String> {
    let dir = Path::new(hf_dir);
    let cfg_json = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("read config.json: {e}"))?;
    let mut cfg = config_from_hf(&cfg_json)?;
    if let Some(b) = block_size {
        cfg.block_size = b;
    }

    let plan: Vec<(String, Vec<u64>)> =
        cfg.param_list().into_iter().map(|(name, numel)| (name, vec![numel as u64])).collect();
    let param_count: u64 = plan.iter().map(|(_, s)| s.iter().product::<u64>()).sum();
    // A card so this file auto-serves from the global model directory (P2) with
    // no BRAIN_QWEN_WEIGHTS env var — id defaults to the output filename stem,
    // matching how the model dir keys catalog entries, unless the caller
    // overrides it (the auto-fetch dispatcher needs the vendor/repo ref).
    let id = id_override.unwrap_or_else(|| Path::new(out_path).file_stem().and_then(|s| s.to_str()).unwrap_or("qwen"));
    let mut card = checkpoint::st::ModelCard::new(id, "qwen");
    card.context_length = Some(cfg.block_size as u64);
    card.param_count = Some(param_count);

    let mut writer = checkpoint::weightio::StWriter::create(out_path, &plan, &cfg.to_json(), Some(&card))
        .map_err(|e| format!("create {out_path}: {e}"))?;
    // Single `model.safetensors` or sharded `model.safetensors.index.json`, streamed one tensor at a time.
    let reader = checkpoint::weightio::WeightReader::open_hf_dir(dir).map_err(|e| format!("open {hf_dir}: {e}"))?;

    let mut err: Option<String> = None;
    let mut n_written = 0usize;
    reader.for_each(|name, _shape, data| {
        if err.is_some() {
            return;
        }
        if let Some(bn) = hf_to_brain(name, cfg.tie_embeddings) {
            n_written += 1;
            if let Err(e) = writer.write(&bn, &data) {
                err = Some(e.to_string());
            }
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    writer.finish().map_err(|e| e.to_string())?;
    eprintln!("imported {n_written} tensors -> {out_path}");
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

    // ---- streaming import() parity: synthetic tiny HF checkpoint ----

    fn seq(base: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| base + i as f32).collect()
    }

    /// Tiny 1-layer tied-embedding checkpoint dir. Ships a redundant
    /// `lm_head.weight` (as real tied Qwen3 checkpoints sometimes do) to exercise
    /// the "tied -> drop" branch of `hf_to_brain` under streaming.
    ///
    /// Every call gets its OWN directory (pid + a monotonic counter, not pid
    /// alone) — multiple tests in this file call this concurrently, and a
    /// pid-only path let one test's `remove_dir_all` cleanup delete the
    /// directory out from under another still-running test.
    fn build_tiny_hf_dir() -> std::path::PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("brain-qwen-import-streaming-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let json = r#"{"vocab_size":5,"hidden_size":6,"num_hidden_layers":1,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":8,"rope_theta":1000000,"rms_norm_eps":1e-6,
            "tie_word_embeddings":true}"#;
        std::fs::write(dir.join("config.json"), json).unwrap();

        // hq = 2*4 = 8, hkv = 1*4 = 4, d = 6, ff = 8.
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
            ("model.embed_tokens.weight".into(), vec![30], seq(1_000_000.0, 30)),
            ("model.norm.weight".into(), vec![6], seq(2_000_000.0, 6)),
            ("lm_head.weight".into(), vec![30], seq(3_000_000.0, 30)), // tied -> must be dropped
            ("model.layers.0.input_layernorm.weight".into(), vec![6], seq(10.0, 6)),
            ("model.layers.0.self_attn.q_proj.weight".into(), vec![48], seq(20.0, 48)),
            ("model.layers.0.self_attn.k_proj.weight".into(), vec![24], seq(70.0, 24)),
            ("model.layers.0.self_attn.v_proj.weight".into(), vec![24], seq(100.0, 24)),
            ("model.layers.0.self_attn.q_norm.weight".into(), vec![4], seq(130.0, 4)),
            ("model.layers.0.self_attn.k_norm.weight".into(), vec![4], seq(140.0, 4)),
            ("model.layers.0.self_attn.o_proj.weight".into(), vec![48], seq(150.0, 48)),
            ("model.layers.0.post_attention_layernorm.weight".into(), vec![6], seq(200.0, 6)),
            ("model.layers.0.mlp.gate_proj.weight".into(), vec![48], seq(210.0, 48)),
            ("model.layers.0.mlp.up_proj.weight".into(), vec![48], seq(260.0, 48)),
            ("model.layers.0.mlp.down_proj.weight".into(), vec![48], seq(310.0, 48)),
        ];
        checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &tensors, &serde_json::Value::Null, None)
            .unwrap();
        dir
    }

    #[test]
    fn import_streams_and_matches_param_list_tied_head_dropped() {
        let dir = build_tiny_hf_dir();
        let out = std::env::temp_dir().join(format!("brain-qwen-import-streaming-out-{}.st", std::process::id()));
        let out_str = out.to_str().unwrap();

        import(dir.to_str().unwrap(), out_str).expect("streaming import");

        let cfg = config_from_hf(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        let m = checkpoint::st::load_safetensors(out_str).unwrap();

        let expected = cfg.param_list();
        assert_eq!(m.tensors.len(), expected.len());
        for (name, numel) in &expected {
            let data = m.tensors.get(name).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(data.len(), *numel, "{name}");
        }
        assert!(!m.tensors.contains_key("lm_head.weight")); // tied source tensor dropped, never written

        assert_eq!(m.tensors["tok.weight"], seq(1_000_000.0, 30));
        assert_eq!(m.tensors["norm.weight"], seq(2_000_000.0, 6));
        assert_eq!(m.tensors["blocks.0.attn.wq.weight"], seq(20.0, 48));
        assert_eq!(m.tensors["blocks.0.mlp.down.weight"], seq(310.0, 48));

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_file(&out).ok();
    }

    /// [`hf_source`] must be byte-for-byte identical to the eager
    /// [`brain_init_from_hf`] for every brain parameter — the numeric-parity
    /// guarantee a streaming-loader switch relies on (equal weights in ⇒
    /// identical device weights ⇒ identical numerics).
    #[test]
    fn hf_source_streaming_matches_eager_brain_init_from_hf() {
        use checkpoint::TensorSource;

        let dir = build_tiny_hf_dir();
        let cfg = config_from_hf(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();

        let eager_tensors = checkpoint::safetensors::read_model_dir(&dir).unwrap();
        let eager = brain_init_from_hf(eager_tensors, &cfg).unwrap();

        let reader = checkpoint::weightio::WeightReader::open_hf_dir(&dir).unwrap();
        let src = hf_source(&reader, &cfg).unwrap();

        for (name, numel) in cfg.param_list() {
            let mut got = None;
            assert!(src.with_tensor(&name, &mut |d| got = Some(d.to_vec())), "missing {name}");
            let got = got.unwrap();
            assert_eq!(got.len(), numel, "{name}");
            assert_eq!(&got, &eager[&name], "{name}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A checkpoint missing a required tensor must be refused by `validate`
    /// (called inside `hf_source`) before any data is read, not discovered
    /// partway through a multi-GB build.
    #[test]
    fn hf_source_refuses_a_checkpoint_missing_a_required_tensor() {
        let dir = build_tiny_hf_dir();
        // Drop a required tensor by rewriting the checkpoint without it.
        let full = checkpoint::safetensors::read_model_dir(&dir).unwrap();
        let trimmed: Vec<_> = full.into_iter().filter(|t| t.name != "model.layers.0.self_attn.q_proj.weight").collect();
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = trimmed.into_iter().map(|t| (t.name, t.shape.iter().map(|&s| s as u64).collect(), t.data)).collect();
        std::fs::remove_file(dir.join("model.safetensors")).unwrap();
        checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &tensors, &serde_json::Value::Null, None).unwrap();

        let cfg = config_from_hf(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        let reader = checkpoint::weightio::WeightReader::open_hf_dir(&dir).unwrap();
        let err = match hf_source(&reader, &cfg) {
            Ok(_) => panic!("a checkpoint missing a required tensor must be refused"),
            Err(e) => e,
        };
        assert!(err.contains("blocks.0.attn.wq.weight"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
