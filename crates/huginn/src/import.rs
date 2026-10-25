// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements checkpoint import for recurrent-depth
// decoders for its clients. If your team needs expertise in bringing a
// prelude/core/coda checkpoint's exact tensor layout into a Rust engine then
// you can procure our services by sending an email to info@swedishembedded.com.

//! Import a Hugging Face Huginn checkpoint (`config.json` + four safetensors
//! shards) into a brain `.safetensors` container.
//!
//! brain's `matmul` is `out = x @ Wᵀ` with `W:[out,in]`, which is HF's
//! `nn.Linear.weight`; the embedding table is `[vocab, hidden]` in both, so no
//! tensor is transposed. Two checkpoint tensors are deliberately dropped:
//!
//! - `lm_head.weight` - the released config ties the head to the embedding
//!   table; keeping both would be a second, drifting copy of the same rows.
//! - `freqs_cis` - a precomputed rotary table, not a parameter; the engine
//!   rebuilds it from `rope_base`.

use std::collections::HashMap;
use std::path::Path;

use checkpoint::remap::{Fetch, RemapSource};
use checkpoint::TensorSource;

use crate::config::HuginnConfig;

fn group_name(hf: &str) -> Option<&'static str> {
    match hf {
        "prelude" => Some("prelude"),
        "core_block" => Some("core"),
        "coda" => Some("coda"),
        _ => None,
    }
}

/// Map an HF Huginn tensor name to its brain parameter name, or `None` for
/// the two documented drops.
fn hf_to_brain(name: &str) -> Option<String> {
    match name {
        "transformer.wte.weight" => return Some("tok.weight".to_string()),
        "transformer.ln_f.weight" => return Some("ln_f.weight".to_string()),
        "transformer.adapter.weight" => return Some("adapter.weight".to_string()),
        "lm_head.weight" | "freqs_cis" => return None,
        _ => {}
    }
    let rest = name.strip_prefix("transformer.")?;
    let (group, rest) = rest.split_once('.')?;
    let group = group_name(group)?;
    let (i, rest) = rest.split_once('.')?;
    let leaf = match rest {
        "norm_1.weight" => "ln1.weight",
        "norm_2.weight" => "ln2.weight",
        "norm_3.weight" => "ln3.weight",
        "norm_4.weight" => "ln4.weight",
        "attn.Wqkv.weight" => "attn.wqkv.weight",
        "attn.proj.weight" => "attn.wo.weight",
        "attn.qk_bias" => "attn.qk_bias",
        "mlp.fc.weight" => "mlp.fc.weight",
        "mlp.proj.weight" => "mlp.proj.weight",
        _ => return None,
    };
    Some(format!("{group}.{i}.{leaf}"))
}

pub fn config_from_hf(json: &str) -> Result<HuginnConfig, String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| format!("huginn config: {e}"))?;
    HuginnConfig::from_json(&v)
}

/// The rename plan for a checkpoint whose tensors are `names`.
///
/// The checkpoint keeps the Q and K bias in one `[2, 1, heads, dim]` tensor;
/// the plan splits it into the two halves the attention adds, so every brain
/// parameter is a contiguous tensor that a dispatch can bind whole.
pub fn plan<'n>(cfg: &HuginnConfig, names: impl IntoIterator<Item = &'n str>) -> Result<HashMap<String, Fetch>, String> {
    let mut plan = HashMap::new();
    let mut unmapped = Vec::new();
    let mut insert = |bn: String, fetch: Fetch| -> Result<(), String> {
        if plan.insert(bn.clone(), fetch).is_some() {
            return Err(format!("import: duplicate mapping to {bn}"));
        }
        Ok(())
    };
    for n in names {
        if n.ends_with(".attn.qk_bias") {
            let brain = hf_to_brain(n).expect("qk_bias maps");
            let prefix = brain.strip_suffix("qk_bias").expect("qk_bias suffix");
            let d = cfg.d_model as usize;
            insert(format!("{prefix}q_bias"), Fetch::Slice { name: n.to_string(), start: 0, len: d })?;
            insert(format!("{prefix}k_bias"), Fetch::Slice { name: n.to_string(), start: d, len: d })?;
            continue;
        }
        match hf_to_brain(n) {
            Some(bn) => insert(bn, Fetch::Whole(n.to_string()))?,
            None if matches!(n, "lm_head.weight" | "freqs_cis") => {}
            None => unmapped.push(n.to_string()),
        }
    }
    if !unmapped.is_empty() {
        unmapped.sort();
        return Err(format!("import: {} unmapped HF tensors: {unmapped:?}", unmapped.len()));
    }
    Ok(plan)
}

pub fn open_dir(dir: &Path) -> Result<(HuginnConfig, RemapSource<'static>), String> {
    let cfg = config_from_hf(
        &std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("{}: {e}", dir.join("config.json").display()))?,
    )?;
    let reader = checkpoint::weightio::WeightReader::open_hf_dir(dir).map_err(|e| format!("open {}: {e}", dir.display()))?;
    let plan = plan(&cfg, reader.names())?;
    let src = RemapSource::owning(Box::new(reader), plan);
    src.validate(&cfg.param_list())?;
    Ok((cfg, src))
}

pub fn import(hf_dir: &str, out_path: &str) -> Result<(), String> {
    import_as(hf_dir, out_path, None)
}

pub fn import_as(hf_dir: &str, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
    let (cfg, src) = open_dir(Path::new(hf_dir))?;
    let plan: Vec<(String, Vec<u64>)> = cfg.param_list().into_iter().map(|(name, numel)| (name, vec![numel as u64])).collect();
    let param_count: u64 = plan.iter().map(|(_, s)| s.iter().product::<u64>()).sum();
    let id = id_override.unwrap_or_else(|| Path::new(out_path).file_stem().and_then(|s| s.to_str()).unwrap_or("huginn"));
    let mut card = checkpoint::st::ModelCard::new(id, "huginn");
    card.context_length = Some(cfg.block_size as u64);
    card.param_count = Some(param_count);
    let mut writer = checkpoint::weightio::StWriter::create(out_path, &plan, &cfg.to_json(), Some(&card))
        .map_err(|e| format!("create {out_path}: {e}"))?;
    for (name, _) in &plan {
        let mut result = Ok(());
        if !src.with_tensor(name, &mut |t| result = writer.write(name, t).map_err(|e| e.to_string())) {
            return Err(format!("huginn import: {name} has no source"));
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

    fn names() -> Vec<String> {
        let cfg = HuginnConfig::tiny();
        let mut names = vec![
            "transformer.wte.weight".to_string(),
            "transformer.ln_f.weight".to_string(),
            "transformer.adapter.weight".to_string(),
            "lm_head.weight".to_string(),
            "freqs_cis".to_string(),
        ];
        for (group, count) in cfg.groups() {
            let hf = match group.name() {
                "core" => "core_block",
                other => other,
            };
            for i in 0..count {
                for leaf in ["norm_1.weight", "norm_2.weight", "norm_3.weight", "norm_4.weight", "attn.Wqkv.weight", "attn.proj.weight", "attn.qk_bias", "mlp.fc.weight", "mlp.proj.weight"] {
                    names.push(format!("transformer.{hf}.{i}.{leaf}"));
                }
            }
        }
        names
    }

    #[test]
    fn every_brain_parameter_has_exactly_one_hf_source() {
        let cfg = HuginnConfig::tiny();
        let plan = plan(&cfg, names().iter().map(String::as_str)).expect("plan");
        assert_eq!(plan.len(), cfg.param_list().len());
        for (name, _) in cfg.param_list() {
            assert!(plan.contains_key(&name), "no source for {name}");
        }
    }

    #[test]
    fn an_unknown_tensor_is_named_not_dropped() {
        let mut n = names();
        n.push("transformer.rotary_emb.inv_freq".to_string());
        let err = plan(&HuginnConfig::tiny(), n.iter().map(String::as_str)).expect_err("unknown tensor must fail");
        assert!(err.contains("transformer.rotary_emb.inv_freq"), "{err}");
    }
}
