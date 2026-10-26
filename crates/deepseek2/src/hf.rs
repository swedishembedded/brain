// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The DeepSeek-V2-family decoder as the upstream `transformers` releases
//! ship it (`model.embed_tokens`, `model.layers.N.*`, `model.norm`,
//! `lm_head`): the one map from those tensor names onto this crate's
//! parameter list, and the one derivation of the decoder's shape from the
//! checkpoint's own tensor shapes.
//!
//! DeepSeek-OCR and DeepSeek-OCR-2 ship byte-identical decoders under the same
//! names, so each model's importer calls this module for its decoder half and
//! only classifies its own vision tensors.
//!
//! Swedish Embedded AB implements checkpoint importers that derive a model's
//! shape from its weights and refuse anything unrecognized. If your team needs
//! a published language model brought onto its own inference stack with every
//! tensor accounted for, you can procure our services by emailing
//! info@swedishembedded.com.

use std::collections::{BTreeMap, HashMap};

use checkpoint::remap::Fetch;
use gguf::deepseek_ocr::DeepseekOcrConfig as DecoderShape;
use serde_json::Value;

use crate::config::DeepseekV2Config;

/// Where one upstream decoder tensor lands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decoded {
    /// A whole tensor under this brain-side name.
    Whole(String),
    /// ONE expert's slice of a MoE expert bank. Upstream ships each routed
    /// expert as its own `[out, in]` tensor; this decoder reads one
    /// `blocks.{l}.mlp.experts.{gate,up,down}.weight` bank per projection, so
    /// `expert` says where in the bank this tensor belongs.
    Expert { bank: String, expert: u32 },
}

/// Split `"{index}.{leaf}"`, bounds-checking the index, so a checkpoint that
/// grew a block (or an expert) stops the import instead of landing silently
/// outside the parameter list.
pub fn split_indexed<'a>(rest: &'a str, full: &str, bound: u32, what: &str) -> Result<(u32, &'a str), String> {
    let (idx, leaf) = rest.split_once('.').ok_or_else(|| format!("malformed {what} name {full:?}"))?;
    let i: u32 = idx.parse().map_err(|_| format!("malformed {what} index in {full:?}"))?;
    if i >= bound {
        return Err(format!("{full}: {what} index {i} beyond {bound}"));
    }
    Ok((i, leaf))
}

/// The brain-side home of `name` if it is a decoder tensor, `None` if it
/// belongs to something else (a vision tower), `Err` if it is a decoder name
/// this crate does not understand.
pub fn classify(name: &str, shape: &DecoderShape) -> Result<Option<Decoded>, String> {
    match name {
        "model.embed_tokens.weight" => return Ok(Some(Decoded::Whole("tok.weight".to_string()))),
        "model.norm.weight" => return Ok(Some(Decoded::Whole("norm.weight".to_string()))),
        "lm_head.weight" => return Ok(Some(Decoded::Whole("lm_head.weight".to_string()))),
        _ => {}
    }
    match name.strip_prefix("model.layers.") {
        Some(rest) => layer_leaf(rest, name, shape).map(Some),
        None => Ok(None),
    }
}

fn layer_leaf(rest: &str, full: &str, shape: &DecoderShape) -> Result<Decoded, String> {
    let unknown = || format!("unrecognized decoder tensor {full:?}");
    let (l, leaf) = split_indexed(rest, full, shape.n_layers, "decoder layer")?;
    let p = |s: &str| Ok(Decoded::Whole(format!("blocks.{l}.{s}")));
    match leaf {
        "input_layernorm.weight" => return p("ln1.weight"),
        "post_attention_layernorm.weight" => return p("ln2.weight"),
        "self_attn.q_proj.weight" | "self_attn.k_proj.weight" | "self_attn.v_proj.weight" | "self_attn.o_proj.weight" => return p(leaf),
        _ => {}
    }
    let mlp = leaf.strip_prefix("mlp.").ok_or_else(unknown)?;
    // Which MLP a layer carries is a fact of the config (`first_k_dense_replace`),
    // so a router tensor on a dense layer is an error rather than a name this
    // decoder happens not to have.
    if !shape.is_moe_layer(l) {
        return match mlp {
            "gate_proj.weight" => p("mlp.gate.weight"),
            "up_proj.weight" => p("mlp.up.weight"),
            "down_proj.weight" => p("mlp.down.weight"),
            _ => Err(unknown()),
        };
    }
    match mlp {
        // NOT `mlp.gate_proj.weight`: on a MoE layer `mlp.gate` is the router.
        "gate.weight" => return p("mlp.router.weight"),
        "shared_experts.gate_proj.weight" => return p("mlp.shared.gate.weight"),
        "shared_experts.up_proj.weight" => return p("mlp.shared.up.weight"),
        "shared_experts.down_proj.weight" => return p("mlp.shared.down.weight"),
        _ => {}
    }
    let rest = mlp.strip_prefix("experts.").ok_or_else(unknown)?;
    let (e, eleaf) = split_indexed(rest, full, shape.n_experts, "expert")?;
    let which = match eleaf {
        "gate_proj.weight" => "gate",
        "up_proj.weight" => "up",
        "down_proj.weight" => "down",
        _ => return Err(unknown()),
    };
    Ok(Decoded::Expert { bank: format!("blocks.{l}.mlp.experts.{which}.weight"), expert: e })
}

/// The decoder half's fetch plan: brain-side name to how it is read from the
/// checkpoint, for every decoder tensor in `names` (others are ignored).
///
/// `Fetch::Whole` for everything except the expert banks, which are a
/// `Fetch::Concat` of the experts in index order. The plan is checked against
/// the decoder's parameter list in both directions: an unrecognized decoder
/// tensor is an error, and so is a parameter no tensor produced.
pub fn plan(names: &[String], cfg: &DeepseekV2Config) -> Result<HashMap<String, Fetch>, String> {
    let mut out = HashMap::new();
    // bank name -> expert index -> source tensor. `BTreeMap` on the inner key
    // makes the concatenation expert-ordered by construction.
    let mut banks: BTreeMap<String, BTreeMap<u32, String>> = BTreeMap::new();
    for src in names {
        match classify(src, &cfg.shape)? {
            None => {}
            Some(Decoded::Whole(brain)) => {
                if out.insert(brain.clone(), Fetch::Whole(src.clone())).is_some() {
                    return Err(format!("two source tensors map to {brain}"));
                }
            }
            Some(Decoded::Expert { bank, expert }) => {
                if let Some(prev) = banks.entry(bank.clone()).or_default().insert(expert, src.clone()) {
                    return Err(format!("two source tensors ({prev}, {src}) are expert {expert} of {bank}"));
                }
            }
        }
    }
    let n_experts = cfg.shape.n_experts;
    for (bank, parts) in banks {
        let missing: Vec<u32> = (0..n_experts).filter(|e| !parts.contains_key(e)).collect();
        if !missing.is_empty() {
            // Name each absent expert with its own per-expert leaf
            // (`...experts.{e}.gate.weight`), so the message points at the one
            // source tensor that is missing rather than only at the bank.
            let (head, tail) = bank.split_once("experts.").unwrap_or((bank.as_str(), ""));
            let named: Vec<String> = missing.iter().map(|e| format!("{head}experts.{e}.{tail}")).collect();
            return Err(format!("{bank}: this checkpoint produces no tensor for {} of {n_experts} experts: {named:?}", missing.len()));
        }
        let fetch = Fetch::Concat(parts.into_values().map(Fetch::Whole).collect());
        if out.insert(bank.clone(), fetch).is_some() {
            return Err(format!("two source tensors map to {bank}"));
        }
    }
    let missing: Vec<String> = cfg.param_list().into_iter().map(|(n, _)| n).filter(|n| !out.contains_key(n)).collect();
    if !missing.is_empty() {
        return Err(format!("this checkpoint produces no decoder tensor for {} parameter(s): {missing:?}", missing.len()));
    }
    Ok(out)
}

/// Every tensor's shape, header-only.
pub fn shapes(reader: &checkpoint::weightio::WeightReader) -> BTreeMap<String, Vec<usize>> {
    reader
        .names()
        .map(str::to_string)
        .collect::<Vec<_>>()
        .into_iter()
        .filter_map(|n| reader.shape(&n).map(|s| (n.clone(), s.iter().map(|d| *d as usize).collect())))
        .collect()
}

/// One tensor's extent along `axis`, or a message naming the tensor.
pub fn dim(shapes: &BTreeMap<String, Vec<usize>>, name: &str, axis: usize) -> Result<u32, String> {
    let s = shapes.get(name).ok_or_else(|| format!("missing {name}"))?;
    let d = s.get(axis).ok_or_else(|| format!("{name} has no axis {axis} (shape {s:?})"))?;
    Ok(*d as u32)
}

/// How many `{prefix}{i}.` stacks the checkpoint declares - one past the
/// highest index present, so a gap is caught by the coverage check rather
/// than silently shortening the stack.
pub fn stack_len(shapes: &BTreeMap<String, Vec<usize>>, prefix: &str) -> u32 {
    shapes
        .keys()
        .filter_map(|k| k.strip_prefix(prefix))
        .filter_map(|rest| rest.split('.').next())
        .filter_map(|idx| idx.parse::<u32>().ok())
        .max()
        .map_or(0, |m| m + 1)
}

/// A `config.json` integer reached by a path of object keys.
pub fn cfg_u32(config: &Value, path: &[&str]) -> Result<u32, String> {
    let mut v = config;
    for k in path {
        v = v.get(k).ok_or_else(|| format!("config.json has no {}", path.join(".")))?;
    }
    v.as_u64().map(|n| n as u32).ok_or_else(|| format!("config.json {} is not an integer", path.join(".")))
}

/// Same, for a float with a documented upstream default when absent.
pub fn cfg_f32(config: &Value, path: &[&str], default: f32) -> f32 {
    let mut v = config;
    for k in path {
        match v.get(k) {
            Some(next) => v = next,
            None => return default,
        }
    }
    v.as_f64().map_or(default, |n| n as f32)
}

/// Refuse when a number the checkpoint states twice disagrees with itself.
pub fn agree(what: &str, derived: u32, declared: u32) -> Result<(), String> {
    if derived != declared {
        return Err(format!("{what}: the tensor shapes say {derived}, config.json says {declared}"));
    }
    Ok(())
}

/// The decoder's shape. Everything the tensors can carry is read off them;
/// the head counts, the router policy numbers and the two RoPE/norm constants
/// come from `config.json` (or, where the repo omits them, from the documented
/// `DeepseekV2Config` defaults its own `configuration_deepseek_v2.py` declares).
pub fn decoder_from_shapes(shapes: &BTreeMap<String, Vec<usize>>, config: &Value, block_size: u32) -> Result<DeepseekV2Config, String> {
    let lang = if config.get("language_config").is_some() { &["language_config"][..] } else { &[][..] };
    let path = |k: &'static str| -> Vec<&str> { lang.iter().copied().chain(std::iter::once(k)).collect() };
    let get = |k: &'static str| cfg_u32(config, &path(k));

    let vocab = dim(shapes, "model.embed_tokens.weight", 0)?;
    let d_model = dim(shapes, "model.embed_tokens.weight", 1)?;
    let n_layers = stack_len(shapes, "model.layers.");
    agree("decoder width", d_model, get("hidden_size")?)?;
    agree("vocabulary", vocab, get("vocab_size")?)?;
    agree("decoder depth", n_layers, get("num_hidden_layers")?)?;

    let n_heads = get("num_attention_heads")?;
    let n_kv_heads = get("num_key_value_heads")?;
    let q_dim = dim(shapes, "model.layers.0.self_attn.q_proj.weight", 0)?;
    if n_heads == 0 || q_dim % n_heads != 0 {
        return Err(format!("q projection is {q_dim} wide, which is not a multiple of {n_heads} heads"));
    }
    let head_dim = q_dim / n_heads;
    let kv_dim = dim(shapes, "model.layers.0.self_attn.k_proj.weight", 0)?;
    agree("key/value projection width", kv_dim, n_kv_heads * head_dim)?;

    // The dense-then-MoE schedule is visible in the tensors: a MoE layer has a
    // router, a dense one has a plain `gate_proj`.
    let n_dense_layers = (0..n_layers).take_while(|l| !shapes.contains_key(&format!("model.layers.{l}.mlp.gate.weight"))).count() as u32;
    agree("dense-layer prefix", n_dense_layers, get("first_k_dense_replace")?)?;
    if n_dense_layers >= n_layers {
        return Err("no layer carries a router; this is not a MoE decoder".to_string());
    }
    let first_moe = n_dense_layers;
    let moe_intermediate_size = dim(shapes, &format!("model.layers.{first_moe}.mlp.experts.0.gate_proj.weight"), 0)?;
    let shared_width = dim(shapes, &format!("model.layers.{first_moe}.mlp.shared_experts.gate_proj.weight"), 0)?;
    if moe_intermediate_size == 0 || shared_width % moe_intermediate_size != 0 {
        return Err(format!("the fused shared expert is {shared_width} wide, not a multiple of one expert's {moe_intermediate_size}"));
    }

    let shape = DecoderShape {
        vocab,
        n_layers,
        d_model,
        max_position_embeddings: get("max_position_embeddings")?,
        // Absent from the OCR repos' config.json; both are `DeepseekV2Config`'s
        // own documented defaults.
        rms_eps: cfg_f32(config, &path("rms_norm_eps"), 1e-6),
        tie_embeddings: !shapes.contains_key("lm_head.weight"),
        n_heads,
        n_kv_heads,
        head_dim,
        rope_theta: cfg_f32(config, &path("rope_theta"), 10_000.0),
        // These checkpoints name no partial-rotary width, so the whole head
        // rotates - the same resolution `gguf::deepseek_ocr::config_from_gguf`
        // applies to the GGUF's `rope.dimension_count = 0`.
        rotary_dim: head_dim,
        n_dense_layers,
        ffn_hidden: dim(shapes, "model.layers.0.mlp.gate_proj.weight", 0)?,
        n_experts: dim(shapes, &format!("model.layers.{first_moe}.mlp.gate.weight"), 0)?,
        top_k: get("num_experts_per_tok")?,
        moe_intermediate_size,
        n_shared_experts: shared_width / moe_intermediate_size,
        n_expert_groups: get("n_group")?,
        n_expert_groups_used: get("topk_group")?,
    };
    agree("routed-expert count", shape.n_experts, get("n_routed_experts")?)?;
    agree("shared-expert count", shape.n_shared_experts, get("n_shared_experts")?)?;
    agree("expert width", shape.moe_intermediate_size, get("moe_intermediate_size")?)?;
    agree("dense feed-forward width", shape.ffn_hidden, get("intermediate_size")?)?;
    Ok(DeepseekV2Config::from_shape(shape, block_size))
}
