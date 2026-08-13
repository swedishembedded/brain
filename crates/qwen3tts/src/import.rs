// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import a HuggingFace `Qwen3-TTS` checkpoint (`config.json` +
//! `model.safetensors`) into brain `.safetensors` containers — one for the Talker
//! decoder, one for the MTP code-predictor.
//!
//! Convention match (identical to `crate::qwen3::import`): brain's `matmul.wgsl`
//! is `out = x @ Wᵀ` with `W:[out,in]` row-major — exactly HF `nn.Linear.weight`;
//! the embedding tables are `[vocab, hidden]` row-major in both. So **no tensor
//! is transposed**; the import is a pure 1:1 name remap + bf16→f32 dequant.
//!
//! The Talker decoder is loaded by [`crate::qwen3::Qwen`] with `tie_embeddings =
//! false`: `talker.model.codec_embedding → tok.weight`, `talker.codec_head →
//! lm_head.weight`. The text-conditioning tensors (`talker.model.text_embedding`,
//! `talker.text_projection.*`) ride along in the same container under
//! `text_embedding.weight` / `text_projection.*` names (ignored by the Qwen
//! loader, picked up by [`crate::talker::TalkerModel`]).

use std::path::Path;

use serde_json::Value;

use crate::config::{MtpConfig, TalkerConfig};

/// Map a per-layer Qwen3 decoder leaf (`self_attn.q_proj.weight`, …) to its brain
/// name (`attn.wq.weight`, …). Returns `None` for an unknown leaf.
fn layer_leaf(rest: &str) -> Option<&'static str> {
    Some(match rest {
        "input_layernorm.weight" => "ln1.weight",
        "post_attention_layernorm.weight" => "ln2.weight",
        "self_attn.q_proj.weight" => "attn.wq.weight",
        "self_attn.k_proj.weight" => "attn.wk.weight",
        "self_attn.v_proj.weight" => "attn.wv.weight",
        "self_attn.o_proj.weight" => "attn.wo.weight",
        "self_attn.q_norm.weight" => "attn.q_norm.weight",
        "self_attn.k_norm.weight" => "attn.k_norm.weight",
        "mlp.gate_proj.weight" => "mlp.gate.weight",
        "mlp.up_proj.weight" => "mlp.up.weight",
        "mlp.down_proj.weight" => "mlp.down.weight",
        _ => return None,
    })
}

/// Map an HF `talker.*` tensor name to its brain Talker-container name, or `None`
/// to drop it (a non-Talker tensor: `code_predictor.*`, `speaker_encoder.*`).
pub fn talker_hf_to_brain(name: &str) -> Option<String> {
    if name == "talker.model.codec_embedding.weight" {
        return Some("tok.weight".to_string());
    }
    if name == "talker.model.norm.weight" {
        return Some("norm.weight".to_string());
    }
    if name == "talker.model.text_embedding.weight" {
        return Some("text_embedding.weight".to_string());
    }
    if name == "talker.codec_head.weight" {
        return Some("lm_head.weight".to_string());
    }
    if let Some(rest) = name.strip_prefix("talker.text_projection.") {
        let leaf = match rest {
            "linear_fc1.weight" => "text_projection.fc1.weight",
            "linear_fc1.bias" => "text_projection.fc1.bias",
            "linear_fc2.weight" => "text_projection.fc2.weight",
            "linear_fc2.bias" => "text_projection.fc2.bias",
            _ => return None,
        };
        return Some(leaf.to_string());
    }
    let rest = name.strip_prefix("talker.model.layers.")?;
    let (n, rest) = rest.split_once('.')?;
    Some(format!("blocks.{n}.{}", layer_leaf(rest)?))
}

/// Map an HF `talker.code_predictor.*` tensor name to its brain MTP-container
/// name, or `None` to drop it.
pub fn mtp_hf_to_brain(name: &str) -> Option<String> {
    if name == "talker.code_predictor.model.norm.weight" {
        return Some("norm.weight".to_string());
    }
    if name == "talker.code_predictor.small_to_mtp_projection.weight" {
        return Some("small_to_mtp_projection.weight".to_string());
    }
    if name == "talker.code_predictor.small_to_mtp_projection.bias" {
        return Some("small_to_mtp_projection.bias".to_string());
    }
    if let Some(i) = name
        .strip_prefix("talker.code_predictor.model.codec_embedding.")
        .and_then(|r| r.strip_suffix(".weight"))
    {
        return Some(format!("codec_embedding.{i}.weight"));
    }
    if let Some(i) = name
        .strip_prefix("talker.code_predictor.lm_head.")
        .and_then(|r| r.strip_suffix(".weight"))
    {
        return Some(format!("lm_head.{i}.weight"));
    }
    let rest = name.strip_prefix("talker.code_predictor.model.layers.")?;
    let (n, rest) = rest.split_once('.')?;
    Some(format!("blocks.{n}.{}", layer_leaf(rest)?))
}

fn read_config(dir: &Path) -> Result<Value, String> {
    let s = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("read config.json: {e}"))?;
    serde_json::from_str(&s).map_err(|e| format!("parse config.json: {e}"))
}

fn open_source(dir: &Path) -> Result<checkpoint::weightio::WeightReader, String> {
    let st = dir.join("model.safetensors");
    if !st.exists() {
        return Err(format!(
            "missing {}: sharded checkpoints are not supported",
            st.display()
        ));
    }
    checkpoint::weightio::WeightReader::open(st.to_str().unwrap())
        .map_err(|e| format!("import: opening checkpoint: {e}"))
}

/// The Talker decoder parameter list (Qwen3, untied) plus the text-conditioning
/// extras, with the element counts the import must match.
fn talker_param_specs(cfg: &TalkerConfig) -> Vec<(String, usize)> {
    let mut out = cfg.to_qwen(0).param_list(); // tok/blocks/norm/lm_head (untied)
    let th = cfg.text_hidden_size as usize;
    let d = cfg.d_model as usize;
    out.push((
        "text_embedding.weight".to_string(),
        cfg.text_vocab_size as usize * th,
    ));
    out.push(("text_projection.fc1.weight".to_string(), th * th));
    out.push(("text_projection.fc1.bias".to_string(), th));
    out.push(("text_projection.fc2.weight".to_string(), d * th));
    out.push(("text_projection.fc2.bias".to_string(), d));
    out
}

/// MTP parameter list: the 5-layer decoder (blocks + norm) plus the 15 input
/// codec-embedding tables and 15 output lm_head tables.
fn mtp_param_specs(cfg: &MtpConfig) -> Vec<(String, usize)> {
    let d = cfg.d_model as usize;
    let ff = cfg.d_ff as usize;
    let hq = cfg.q_dim() as usize;
    let hkv = cfg.kv_dim() as usize;
    let hd = cfg.head_dim as usize;
    let v = cfg.vocab as usize;
    let mut out = Vec::new();
    for l in 0..cfg.n_layers {
        let p = |s: &str| format!("blocks.{l}.{s}");
        out.push((p("ln1.weight"), d));
        out.push((p("attn.wq.weight"), hq * d));
        out.push((p("attn.wk.weight"), hkv * d));
        out.push((p("attn.wv.weight"), hkv * d));
        out.push((p("attn.q_norm.weight"), hd));
        out.push((p("attn.k_norm.weight"), hd));
        out.push((p("attn.wo.weight"), d * hq));
        out.push((p("ln2.weight"), d));
        out.push((p("mlp.gate.weight"), ff * d));
        out.push((p("mlp.up.weight"), ff * d));
        out.push((p("mlp.down.weight"), d * ff));
    }
    out.push(("norm.weight".to_string(), d));
    let emb = cfg.embedding_dim as usize;
    for i in 0..cfg.n_residual() {
        // codec_embedding rows are in the Talker hidden width (`embedding_dim`);
        // lm_head reads the MTP decoder hidden width (`d_model`).
        out.push((format!("codec_embedding.{i}.weight"), v * emb));
        out.push((format!("lm_head.{i}.weight"), v * d));
    }
    // small_to_mtp_projection (embedding_dim -> d_model) exists only when the two
    // widths differ (the 1.7B); the 0.6B has no such tensor (Identity).
    if emb != d {
        out.push(("small_to_mtp_projection.weight".to_string(), d * emb));
        out.push(("small_to_mtp_projection.bias".to_string(), d));
    }
    out
}

/// Stream `reader`'s tensors through `map`, writing each mapped tensor straight
/// to `writer` (one at a time — never the whole checkpoint in memory). Fails
/// loudly on a missing tensor, a shape mismatch, or a duplicate mapping — all
/// caught by `StWriter::write`/`finish` (an unplanned or already-written name,
/// or a wrong element count), so no separate coverage pass is needed here.
fn stream_container(
    reader: &checkpoint::weightio::WeightReader,
    map: impl Fn(&str) -> Option<String>,
    writer: &mut checkpoint::weightio::StWriter,
) -> Result<(usize, usize), String> {
    let mut dropped = 0usize;
    let mut mapped = 0usize;
    let mut err: Option<String> = None;
    reader.for_each(|name, _shape, data| {
        if err.is_some() {
            return;
        }
        match map(name) {
            Some(bn) => {
                mapped += 1;
                if let Err(e) = writer.write(&bn, &data) {
                    err = Some(format!("import: {e}"));
                }
            }
            None => dropped += 1,
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    Ok((mapped, dropped))
}

/// Import the Talker decoder (+ text-conditioning tensors) from `<hf_dir>` into a
/// brain checkpoint at `out_path`. Loadable by [`crate::talker::TalkerModel`].
/// Streams one source tensor at a time (never the whole checkpoint in memory).
pub fn import_talker(hf_dir: &str, out_path: &str) -> Result<(), String> {
    let dir = Path::new(hf_dir);
    let root = read_config(dir)?;
    let cfg = TalkerConfig::from_json(&root);
    let specs = talker_param_specs(&cfg);
    let plan: Vec<(String, Vec<u64>)> = specs.iter().map(|(n, numel)| (n.clone(), vec![*numel as u64])).collect();
    // The container's config is the Qwen3 decoder config (untied) so the shared
    // loader parses it directly; the talker's M-RoPE/code-group metadata is not
    // needed for the decoder forward.
    let cfg_json = cfg.to_qwen(2048).to_json();
    let mut writer = checkpoint::weightio::StWriter::create(out_path, &plan, &cfg_json, None)
        .map_err(|e| format!("import: creating output: {e}"))?;
    let reader = open_source(dir)?;
    let (mapped, dropped) = stream_container(&reader, talker_hf_to_brain, &mut writer)?;
    writer.finish().map_err(|e| format!("import: {e}"))?;
    eprintln!(
        "imported Talker: {} tensors -> {out_path} ({mapped} HF talker.* tensors mapped, {dropped} dropped)",
        plan.len(),
    );
    Ok(())
}

/// Import the MTP code-predictor from `<hf_dir>` into a brain checkpoint at
/// `out_path`. Loadable by [`crate::mtp::MtpModel`].
/// Streams one source tensor at a time (never the whole checkpoint in memory).
pub fn import_mtp(hf_dir: &str, out_path: &str) -> Result<(), String> {
    let dir = Path::new(hf_dir);
    let root = read_config(dir)?;
    let cfg = MtpConfig::from_json(&root);
    let specs = mtp_param_specs(&cfg);
    let plan: Vec<(String, Vec<u64>)> = specs.iter().map(|(n, numel)| (n.clone(), vec![*numel as u64])).collect();
    let mut writer = checkpoint::weightio::StWriter::create(out_path, &plan, &cfg.to_json(), None)
        .map_err(|e| format!("import: creating output: {e}"))?;
    let reader = open_source(dir)?;
    let (mapped, dropped) = stream_container(&reader, mtp_hf_to_brain, &mut writer)?;
    writer.finish().map_err(|e| format!("import: {e}"))?;
    eprintln!(
        "imported MTP: {} tensors -> {out_path} ({mapped} HF code_predictor.* tensors mapped, {dropped} dropped)",
        plan.len(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn talker_name_mapping() {
        assert_eq!(
            talker_hf_to_brain("talker.model.codec_embedding.weight").unwrap(),
            "tok.weight"
        );
        assert_eq!(
            talker_hf_to_brain("talker.codec_head.weight").unwrap(),
            "lm_head.weight"
        );
        assert_eq!(
            talker_hf_to_brain("talker.model.norm.weight").unwrap(),
            "norm.weight"
        );
        assert_eq!(
            talker_hf_to_brain("talker.model.text_embedding.weight").unwrap(),
            "text_embedding.weight"
        );
        assert_eq!(
            talker_hf_to_brain("talker.text_projection.linear_fc1.weight").unwrap(),
            "text_projection.fc1.weight"
        );
        assert_eq!(
            talker_hf_to_brain("talker.model.layers.5.self_attn.q_proj.weight").unwrap(),
            "blocks.5.attn.wq.weight"
        );
        assert_eq!(
            talker_hf_to_brain("talker.model.layers.27.mlp.down_proj.weight").unwrap(),
            "blocks.27.mlp.down.weight"
        );
        // Non-talker tensors are dropped.
        assert_eq!(
            talker_hf_to_brain("talker.code_predictor.model.norm.weight"),
            None
        );
        assert_eq!(talker_hf_to_brain("speaker_encoder.foo"), None);
    }

    #[test]
    fn mtp_name_mapping() {
        assert_eq!(
            mtp_hf_to_brain("talker.code_predictor.model.norm.weight").unwrap(),
            "norm.weight"
        );
        assert_eq!(
            mtp_hf_to_brain("talker.code_predictor.model.codec_embedding.0.weight").unwrap(),
            "codec_embedding.0.weight"
        );
        assert_eq!(
            mtp_hf_to_brain("talker.code_predictor.lm_head.14.weight").unwrap(),
            "lm_head.14.weight"
        );
        assert_eq!(
            mtp_hf_to_brain("talker.code_predictor.model.layers.3.self_attn.k_norm.weight")
                .unwrap(),
            "blocks.3.attn.k_norm.weight"
        );
        assert_eq!(mtp_hf_to_brain("talker.model.norm.weight"), None);
    }

    #[test]
    fn param_spec_counts() {
        // Real Talker: 28 layers × 11 + tok + norm + lm_head + text_embedding + 4
        // text_projection.
        let tc = TalkerConfig::from_json(&serde_json::json!({"talker_config": {}}));
        let ts = talker_param_specs(&tc);
        assert_eq!(ts.len(), 28 * 11 + 2 + 1 + 1 + 4);
        // Real MTP: 5 layers × 11 + norm + 15 codec_embedding + 15 lm_head.
        let mc = MtpConfig {
            n_layers: 5,
            num_code_groups: 16,
            ..MtpConfig::tiny()
        };
        let ms = mtp_param_specs(&mc);
        assert_eq!(ms.len(), 5 * 11 + 1 + 15 + 15);
    }

    /// Inverse of `layer_leaf`: the HF decoder-layer leaf name for a brain leaf.
    fn hf_layer_leaf(brain_leaf: &str) -> &'static str {
        match brain_leaf {
            "ln1.weight" => "input_layernorm.weight",
            "ln2.weight" => "post_attention_layernorm.weight",
            "attn.wq.weight" => "self_attn.q_proj.weight",
            "attn.wk.weight" => "self_attn.k_proj.weight",
            "attn.wv.weight" => "self_attn.v_proj.weight",
            "attn.wo.weight" => "self_attn.o_proj.weight",
            "attn.q_norm.weight" => "self_attn.q_norm.weight",
            "attn.k_norm.weight" => "self_attn.k_norm.weight",
            "mlp.gate.weight" => "mlp.gate_proj.weight",
            "mlp.up.weight" => "mlp.up_proj.weight",
            "mlp.down.weight" => "mlp.down_proj.weight",
            other => panic!("unknown brain leaf {other}"),
        }
    }

    /// Inverse of `talker_hf_to_brain`: the HF tensor name a given brain Talker
    /// param name came from.
    fn hf_name_for_talker(brain_name: &str) -> String {
        match brain_name {
            "tok.weight" => "talker.model.codec_embedding.weight".to_string(),
            "norm.weight" => "talker.model.norm.weight".to_string(),
            "lm_head.weight" => "talker.codec_head.weight".to_string(),
            "text_embedding.weight" => "talker.model.text_embedding.weight".to_string(),
            "text_projection.fc1.weight" => "talker.text_projection.linear_fc1.weight".to_string(),
            "text_projection.fc1.bias" => "talker.text_projection.linear_fc1.bias".to_string(),
            "text_projection.fc2.weight" => "talker.text_projection.linear_fc2.weight".to_string(),
            "text_projection.fc2.bias" => "talker.text_projection.linear_fc2.bias".to_string(),
            other => {
                let rest = other.strip_prefix("blocks.").unwrap();
                let (n, leaf) = rest.split_once('.').unwrap();
                format!("talker.model.layers.{n}.{}", hf_layer_leaf(leaf))
            }
        }
    }

    /// Inverse of `mtp_hf_to_brain`.
    fn hf_name_for_mtp(brain_name: &str) -> String {
        match brain_name {
            "norm.weight" => "talker.code_predictor.model.norm.weight".to_string(),
            "small_to_mtp_projection.weight" => {
                "talker.code_predictor.small_to_mtp_projection.weight".to_string()
            }
            "small_to_mtp_projection.bias" => "talker.code_predictor.small_to_mtp_projection.bias".to_string(),
            other if other.starts_with("codec_embedding.") => {
                let i = other.strip_prefix("codec_embedding.").unwrap().strip_suffix(".weight").unwrap();
                format!("talker.code_predictor.model.codec_embedding.{i}.weight")
            }
            other if other.starts_with("lm_head.") => {
                let i = other.strip_prefix("lm_head.").unwrap().strip_suffix(".weight").unwrap();
                format!("talker.code_predictor.lm_head.{i}.weight")
            }
            other => {
                let rest = other.strip_prefix("blocks.").unwrap();
                let (n, leaf) = rest.split_once('.').unwrap();
                format!("talker.code_predictor.model.layers.{n}.{}", hf_layer_leaf(leaf))
            }
        }
    }

    #[test]
    fn streaming_import_talker_and_mtp_from_one_shared_checkpoint() {
        // A synthetic HF checkpoint dir holding BOTH talker.* and
        // talker.code_predictor.* tensors in one model.safetensors (mirrors the
        // real Qwen3-TTS layout) — each import must pick out only its own
        // tensors, streaming, without cross-contamination.
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("tts-import-src-{pid}"));
        std::fs::create_dir_all(&dir).unwrap();
        let config = serde_json::json!({
            "talker_config": {
                "num_hidden_layers": 2, "hidden_size": 16, "head_dim": 8,
                "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 32,
                "vocab_size": 23, "text_hidden_size": 20, "text_vocab_size": 29,
                "code_predictor_config": {
                    "num_hidden_layers": 2, "hidden_size": 16, "head_dim": 8,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 32,
                    "vocab_size": 23, "num_code_groups": 4,
                },
            },
        });
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();

        let root = &config;
        let tc = TalkerConfig::from_json(root);
        let mc = MtpConfig::from_json(root);
        let talker_specs = talker_param_specs(&tc);
        let mtp_specs = mtp_param_specs(&mc);

        // Every tensor both imports need, keyed by its brain name (namespaced so
        // the two never collide, matching how the real checkpoint co-resides).
        let mut plan: Vec<(String, Vec<u64>)> = Vec::new();
        let mut expect_talker: HashMap<String, Vec<f32>> = HashMap::new();
        let mut expect_mtp: HashMap<String, Vec<f32>> = HashMap::new();
        for (i, (name, numel)) in talker_specs.iter().enumerate() {
            let hf = hf_name_for_talker(name);
            let data: Vec<f32> = (0..*numel).map(|j| (i * 10_000 + j) as f32 * 0.001).collect();
            plan.push((hf.clone(), vec![*numel as u64]));
            expect_talker.insert(name.clone(), data);
        }
        for (i, (name, numel)) in mtp_specs.iter().enumerate() {
            let hf = hf_name_for_mtp(name);
            let data: Vec<f32> = (0..*numel).map(|j| (i * 20_000 + j) as f32 * 0.001 + 5.0).collect();
            plan.push((hf.clone(), vec![*numel as u64]));
            expect_mtp.insert(name.clone(), data);
        }

        let mut w = checkpoint::weightio::StWriter::create(
            dir.join("model.safetensors").to_str().unwrap(),
            &plan,
            &serde_json::Value::Null,
            None,
        )
        .unwrap();
        for (i, (name, numel)) in talker_specs.iter().enumerate() {
            let hf = hf_name_for_talker(name);
            let data: Vec<f32> = (0..*numel).map(|j| (i * 10_000 + j) as f32 * 0.001).collect();
            w.write(&hf, &data).unwrap();
        }
        for (i, (name, numel)) in mtp_specs.iter().enumerate() {
            let hf = hf_name_for_mtp(name);
            let data: Vec<f32> = (0..*numel).map(|j| (i * 20_000 + j) as f32 * 0.001 + 5.0).collect();
            w.write(&hf, &data).unwrap();
        }
        w.finish().unwrap();

        let talker_out = std::env::temp_dir().join(format!("tts-import-talker-{pid}.safetensors"));
        let mtp_out = std::env::temp_dir().join(format!("tts-import-mtp-{pid}.safetensors"));
        import_talker(dir.to_str().unwrap(), talker_out.to_str().unwrap()).unwrap();
        import_mtp(dir.to_str().unwrap(), mtp_out.to_str().unwrap()).unwrap();

        let tr = checkpoint::weightio::WeightReader::open(talker_out.to_str().unwrap()).unwrap();
        assert_eq!(tr.names().count(), expect_talker.len());
        for (name, data) in &expect_talker {
            assert_eq!(tr.tensor(name).unwrap(), *data, "talker {name}");
        }

        let mr = checkpoint::weightio::WeightReader::open(mtp_out.to_str().unwrap()).unwrap();
        assert_eq!(mr.names().count(), expect_mtp.len());
        for (name, data) in &expect_mtp {
            assert_eq!(mr.tensor(name).unwrap(), *data, "mtp {name}");
        }

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_file(&talker_out).ok();
        std::fs::remove_file(&mtp_out).ok();
    }
}
