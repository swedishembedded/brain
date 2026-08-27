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

/// [`hf_source`] for a caller that will build a **partial** [`crate::Shard`]:
/// the required-tensor set is the shard's own
/// ([`crate::shard_param_list`]), not the whole `cfg.param_list()`.
///
/// This is the difference between "reads the checkpoint efficiently" and
/// "does not need the whole checkpoint at all". FLUX.2 conditions on hidden
/// states from a mid-stack tap, so its encoder shard is `start: 0, end:
/// <deepest tap>, embed: true, head: false` - the layers past the tap, the
/// final norm and the LM head are never read by any buffer the build
/// allocates. Validating against the full list forces those tensors to exist
/// (and, on a checkpoint being fetched, to be downloaded) purely to be
/// discarded.
///
/// The coverage check is **narrowed, never weakened**, and stays exact on
/// three counts:
///
/// - Every tensor the shard genuinely needs must be present with the right
///   element count, checked before a byte of weight data is read.
/// - Every tensor that IS present and maps into the full `param_list()` is
///   *also* element-count checked, even when this shard will not read it - so
///   a config/checkpoint dimension mismatch is still caught on a tensor
///   outside the tap range.
/// - A source tensor mapping to a brain parameter outside the full
///   `param_list()` - a 36-layer checkpoint against a 28-layer config, say -
///   is still a hard error. The *allowed* set stays the full list; only the
///   *required* set shrinks.
pub fn hf_shard_source<'a>(
    r: &'a checkpoint::weightio::WeightReader,
    cfg: &QwenConfig,
    shard: &crate::Shard,
) -> Result<checkpoint::remap::RemapSource<'a>, String> {
    let full = cfg.param_list();
    let allowed: std::collections::HashSet<&str> = full.iter().map(|(n, _)| n.as_str()).collect();
    let mut plan: HashMap<String, checkpoint::remap::Fetch> = HashMap::new();
    for name in r.names() {
        let Some(bn) = hf_to_brain(name, cfg.tie_embeddings) else { continue };
        if !allowed.contains(bn.as_str()) {
            return Err(format!("import: '{name}' maps to unexpected brain param '{bn}'"));
        }
        if plan.insert(bn.clone(), checkpoint::remap::Fetch::Whole(name.to_string())).is_some() {
            return Err(format!("duplicate mapping to {bn}"));
        }
    }
    let src = checkpoint::remap::RemapSource::new(r, plan.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
    // Shape-check everything the checkpoint actually offers, required or not.
    let present: Vec<(String, usize)> =
        full.into_iter().filter(|(n, _)| plan.contains_key(n)).collect();
    src.validate(&present)?;
    // Then require the shard's own set - the half that can fail on a MISSING
    // tensor, and the reason this function exists.
    src.validate(&crate::shard_param_list(cfg, shard))?;
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
        build_hf_dir(1, true)
    }

    /// [`build_tiny_hf_dir`] generalised over depth and head tying, so a
    /// shard-aware test can cut a checkpoint that has layers on both sides of
    /// the cut. `tied == false` keeps a REAL `lm_head.weight` parameter (the
    /// FLUX.2 text encoder's shape: `tie_embeddings: false`, so the head is a
    /// distinct tensor a truncated shard must not need).
    fn build_hf_dir(n_layers: usize, tied: bool) -> std::path::PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("brain-qwen-import-streaming-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let json = format!(
            r#"{{"vocab_size":5,"hidden_size":6,"num_hidden_layers":{n_layers},
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":8,"rope_theta":1000000,"rms_norm_eps":1e-6,
            "tie_word_embeddings":{tied}}}"#
        );
        std::fs::write(dir.join("config.json"), &json).unwrap();

        // hq = 2*4 = 8, hkv = 1*4 = 4, d = 6, ff = 8.
        let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
            ("model.embed_tokens.weight".into(), vec![30], seq(1_000_000.0, 30)),
            ("model.norm.weight".into(), vec![6], seq(2_000_000.0, 6)),
            // Tied: a redundant head tensor real Qwen3 checkpoints sometimes
            // ship, which `hf_to_brain` must drop. Untied: the genuine head.
            ("lm_head.weight".into(), vec![30], seq(3_000_000.0, 30)),
        ];
        for l in 0..n_layers {
            // Per-layer base so no two layers share a value: a shard that read
            // the wrong layer's weights would show up as a value mismatch.
            let b = 1000.0 * l as f32;
            let p = |s: &str| format!("model.layers.{l}.{s}");
            tensors.extend([
                (p("input_layernorm.weight"), vec![6], seq(b + 10.0, 6)),
                (p("self_attn.q_proj.weight"), vec![48], seq(b + 20.0, 48)),
                (p("self_attn.k_proj.weight"), vec![24], seq(b + 70.0, 24)),
                (p("self_attn.v_proj.weight"), vec![24], seq(b + 100.0, 24)),
                (p("self_attn.q_norm.weight"), vec![4], seq(b + 130.0, 4)),
                (p("self_attn.k_norm.weight"), vec![4], seq(b + 140.0, 4)),
                (p("self_attn.o_proj.weight"), vec![48], seq(b + 150.0, 48)),
                (p("post_attention_layernorm.weight"), vec![6], seq(b + 200.0, 6)),
                (p("mlp.gate_proj.weight"), vec![48], seq(b + 210.0, 48)),
                (p("mlp.up_proj.weight"), vec![48], seq(b + 260.0, 48)),
                (p("mlp.down_proj.weight"), vec![48], seq(b + 310.0, 48)),
            ]);
        }
        checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &tensors, &serde_json::Value::Null, None)
            .unwrap();
        dir
    }

    /// Rewrite a fixture checkpoint keeping only the tensors `keep` accepts.
    fn retain_tensors(dir: &std::path::Path, keep: impl Fn(&str) -> bool) {
        let full = checkpoint::safetensors::read_model_dir(dir).unwrap();
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = full
            .into_iter()
            .filter(|t| keep(&t.name))
            .map(|t| (t.name, t.shape.iter().map(|&s| s as u64).collect(), t.data))
            .collect();
        std::fs::remove_file(dir.join("model.safetensors")).unwrap();
        checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &tensors, &serde_json::Value::Null, None).unwrap();
    }

    /// The FLUX.2 text-encoder shape: embedding + layers `[0, end)`, no head.
    /// `Shard::owns` is `l < end`, so `end` is the count of layers kept - a tap
    /// at depth `end` reads the residual stream that has passed through them.
    fn tap_shard(end: usize) -> crate::Shard {
        crate::Shard { start: 0, end, embed: true, head: false, gpu_index: crate::Shard::ANY_GPU }
    }

    /// A truncated shard must load from a checkpoint that HAS the layers past
    /// its tap - and every value it reads must be bit-identical to what the
    /// eager whole-checkpoint import produces for the same parameter. This is
    /// the parity guarantee the pipeline switch rides on.
    #[test]
    fn hf_shard_source_matches_eager_over_the_shard_it_will_build() {
        use checkpoint::TensorSource;

        let dir = build_hf_dir(4, false);
        let cfg = config_from_hf(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        let eager = brain_init_from_hf(checkpoint::safetensors::read_model_dir(&dir).unwrap(), &cfg).unwrap();

        let shard = tap_shard(2);
        let reader = checkpoint::weightio::WeightReader::open_hf_dir(&dir).unwrap();
        let src = hf_shard_source(&reader, &cfg, &shard).unwrap();

        let want = crate::shard_param_list(&cfg, &shard);
        // The shard must be a REAL truncation, or this test proves nothing.
        assert!(want.len() < cfg.param_list().len(), "fixture shard must be partial");
        assert!(!want.iter().any(|(n, _)| n == "lm_head.weight" || n == "norm.weight"), "a headless shard holds neither");
        assert!(!want.iter().any(|(n, _)| n.starts_with("blocks.2.") || n.starts_with("blocks.3.")), "layers past the tap");

        for (name, numel) in &want {
            let mut got = None;
            assert!(src.with_tensor(name, &mut |d| got = Some(d.to_vec())), "missing {name}");
            let got = got.unwrap();
            assert_eq!(got.len(), *numel, "{name}");
            // Bits, not a tolerance: both routes read the same source tensor.
            assert_eq!(&got, &eager[name], "{name}: streamed and eager must be identical");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The point of the shard-aware source: a checkpoint that does NOT contain
    /// the layers past the tap (nor the head) is a perfectly good source for a
    /// shard that never reads them. The full-list [`hf_source`] must still
    /// refuse the very same checkpoint - that contrast is the whole change.
    #[test]
    fn hf_shard_source_accepts_a_checkpoint_the_full_list_refuses() {
        let dir = build_hf_dir(4, false);
        let cfg = config_from_hf(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        retain_tensors(&dir, |n| {
            !(n == "lm_head.weight" || n == "model.norm.weight" || n.starts_with("model.layers.2.") || n.starts_with("model.layers.3."))
        });

        let reader = checkpoint::weightio::WeightReader::open_hf_dir(&dir).unwrap();
        hf_shard_source(&reader, &cfg, &tap_shard(2)).expect("a truncated shard must not need what it never reads");

        let err = match hf_source(&reader, &cfg) {
            Ok(_) => panic!("the full list must still demand every tensor"),
            Err(e) => e,
        };
        assert!(err.contains("blocks.2."), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Narrowed, not weakened (1/2): a tensor the shard DOES read, missing,
    /// is still a hard failure before any data is touched.
    #[test]
    fn hf_shard_source_still_refuses_a_tensor_the_shard_needs() {
        let dir = build_hf_dir(4, false);
        let cfg = config_from_hf(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        retain_tensors(&dir, |n| n != "model.layers.1.self_attn.q_proj.weight");

        let reader = checkpoint::weightio::WeightReader::open_hf_dir(&dir).unwrap();
        let err = match hf_shard_source(&reader, &cfg, &tap_shard(2)) {
            Ok(_) => panic!("a tensor inside the tap range is required"),
            Err(e) => e,
        };
        assert!(err.contains("blocks.1.attn.wq.weight"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Narrowed, not weakened (2/2): the ALLOWED set stays the full
    /// `param_list()`. A deeper checkpoint than the config describes is the
    /// classic wrong-checkpoint mistake and must still be caught - even though
    /// the extra layers are outside the shard and would never be read.
    #[test]
    fn hf_shard_source_still_refuses_a_checkpoint_deeper_than_the_config() {
        let dir = build_hf_dir(4, false);
        // Same tensors, a config claiming two fewer layers.
        let mut cfg = config_from_hf(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        cfg.n_layers = 2;

        let reader = checkpoint::weightio::WeightReader::open_hf_dir(&dir).unwrap();
        let err = match hf_shard_source(&reader, &cfg, &tap_shard(2)) {
            Ok(_) => panic!("a 4-layer checkpoint is not a 2-layer model"),
            Err(e) => e,
        };
        assert!(err.contains("blocks.2.") || err.contains("blocks.3."), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// And a present-but-unread tensor is still element-count checked, so a
    /// config/checkpoint dimension mismatch is caught on a tensor outside the
    /// tap range rather than silently accepted.
    #[test]
    fn hf_shard_source_shape_checks_tensors_it_will_never_read() {
        let dir = build_hf_dir(4, false);
        let cfg = config_from_hf(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        // Corrupt the ELEMENT COUNT of a layer past the tap.
        let full = checkpoint::safetensors::read_model_dir(&dir).unwrap();
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = full
            .into_iter()
            .map(|t| {
                if t.name == "model.layers.3.mlp.up_proj.weight" {
                    return (t.name, vec![24], vec![0.0f32; 24]);
                }
                (t.name, t.shape.iter().map(|&s| s as u64).collect(), t.data)
            })
            .collect();
        std::fs::remove_file(dir.join("model.safetensors")).unwrap();
        checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &tensors, &serde_json::Value::Null, None).unwrap();

        let reader = checkpoint::weightio::WeightReader::open_hf_dir(&dir).unwrap();
        let err = match hf_shard_source(&reader, &cfg, &tap_shard(2)) {
            Ok(_) => panic!("a wrong element count is a wrong checkpoint, read or not"),
            Err(e) => e,
        };
        assert!(err.contains("blocks.3.mlp.up.weight"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
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
