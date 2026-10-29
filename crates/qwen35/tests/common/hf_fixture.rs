// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded loading and fine-tuning of
// quantized checkpoints for its clients. If your team needs expertise in
// testing loaders against real checkpoint layouts then you can procure our
// services by sending an email to info@swedishembedded.com.

//! A synthetic Qwen3.5/3.8 HuggingFace checkpoint with the real layout: FP8 e4m3 block-scaled linears with
//! bf16 `weight_scale_inv`, bf16 norms and embeddings, f32 `A_log`, sharded `layers-N.safetensors` plus
//! `outside.safetensors` and `mtp.safetensors` and an index.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use qwen35::config::Qwen35Config;

/// The FP8 `weight_block_size` of the fixture.
pub const BLOCK: usize = 32;

pub struct Lcg(pub u64);
impl Lcg {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    pub fn unit(&mut self) -> f32 {
        (self.next() % 2001) as f32 / 1000.0 - 1.0
    }
}

pub fn bf16(v: f32) -> [u8; 2] {
    ((v.to_bits() >> 16) as u16).to_le_bytes()
}

pub struct Raw {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

pub fn write_st(path: &Path, tensors: &[Raw]) {
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for t in tensors {
        header.insert(t.name.clone(), serde_json::json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [off, off + t.bytes.len()]}));
        off += t.bytes.len();
    }
    let h = serde_json::to_vec(&header).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    for t in tensors {
        out.extend(&t.bytes);
    }
    std::fs::write(path, out).unwrap();
}

/// brain-canonical suffix -> HF leaf (inverse of `import::classify`'s renames).
pub fn hf_leaf(suffix: &str) -> String {
    match suffix {
        "ln1.weight" => "input_layernorm.weight".into(),
        "ln2.weight" => "post_attention_layernorm.weight".into(),
        s => s.replace("mlp.gate.", "mlp.gate_proj.").replace("mlp.up.", "mlp.up_proj.").replace("mlp.down.", "mlp.down_proj."),
    }
}

/// How a checkpoint names the decoder's tensors: the multimodal release nests
/// it under `model.language_model.` (with a vision tower and an MTP head
/// beside it); a text-only release (EdgeRock) has `model.layers.*` and a
/// top-level `config.json`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    Multimodal,
    TextOnly,
}

/// A fresh scratch dir holding the checkpoint; `skip` drops one canonical
/// tensor (to test the refusal path).
pub fn write_checkpoint(cfg: &Qwen35Config, tag: &str, skip: Option<&str>) -> PathBuf {
    write_checkpoint_from(cfg, tag, skip, None, Layout::Multimodal, BLOCK)
}

/// [`write_checkpoint`] with the FP8 scale grid of `block` x `block` (128 in the real release, which the
/// resident tier holds as stored; [`BLOCK`] elsewhere).
pub fn write_checkpoint_blocked(cfg: &Qwen35Config, tag: &str, block: usize) -> PathBuf {
    write_checkpoint_from(cfg, tag, None, None, Layout::Multimodal, block)
}

/// [`write_checkpoint`] holding `values` (brain-canonical, as `qwen35::init::init_weights` makes them) in the
/// same layout, every tensor bf16 (the recurrence's `A_log` f32) instead of random FP8 bytes: a model whose
/// weights are sane enough to train, in a file the packed loader reads like the real release.
pub fn write_dense_checkpoint(cfg: &Qwen35Config, tag: &str, values: &HashMap<String, Vec<f32>>) -> PathBuf {
    write_checkpoint_from(cfg, tag, None, Some(values), Layout::Multimodal, BLOCK)
}

/// [`write_dense_checkpoint`] in a text-only release's layout (EdgeRock): no
/// vision tower or MTP shard, `model.layers.*` names, the anchor-memory
/// tensors of `cfg`, and a tied head when `cfg` ties it.
pub fn write_text_only_checkpoint(cfg: &Qwen35Config, tag: &str, values: &HashMap<String, Vec<f32>>) -> PathBuf {
    write_checkpoint_from(cfg, tag, None, Some(values), Layout::TextOnly, BLOCK)
}

/// A plain RMSNorm gain is stored as `w` and applied as `1 + w` (`qwen35::import`).
fn stored_as_offset_from_one(canon: &str) -> bool {
    canon == "norm.weight"
        || canon.ends_with(".ln1.weight")
        || canon.ends_with(".ln2.weight")
        || canon.ends_with(".self_attn.q_norm.weight")
        || canon.ends_with(".self_attn.k_norm.weight")
        || canon.ends_with(".memory.input_norm.weight")
        || canon.ends_with(".memory.q_norm.weight")
        || canon.ends_with(".memory.k_norm.weight")
}

fn write_checkpoint_from(cfg: &Qwen35Config, tag: &str, skip: Option<&str>, dense: Option<&HashMap<String, Vec<f32>>>, layout: Layout, block: usize) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qwen35_fp8_source_{tag}_{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    // seed: any fixed bytes do; equality is against an independent reader of the same files
    let mut rng = Lcg(0x5eed);
    let d = cfg.d_model as usize;

    let mut linear_shapes: HashMap<String, (usize, usize)> = HashMap::new();
    for (l, ty) in cfg.layer_types().iter().enumerate() {
        for (suffix, n, k) in cfg.layer_leaves(*ty) {
            linear_shapes.insert(format!("blocks.{l}.{suffix}"), (n as usize, k as usize));
        }
    }
    let mut by_file: HashMap<String, Vec<Raw>> = HashMap::new();
    let mut weight_map = serde_json::Map::new();
    for (canon, numel) in cfg.param_list() {
        if skip == Some(canon.as_str()) {
            continue;
        }
        let (file, hf, shape) = if let Some(rest) = canon.strip_prefix("blocks.") {
            let (l, suffix) = rest.split_once('.').unwrap();
            let hf = format!("{}layers.{l}.{}", decoder_prefix(layout), hf_leaf(suffix));
            let shape = match linear_shapes.get(&canon) {
                Some(&(n, k)) => vec![n, k],
                None if suffix == "linear_attn.conv1d.weight" => vec![cfg.linear_conv_dim() as usize, 1, cfg.linear_conv_kernel_dim as usize],
                None => vec![numel],
            };
            (format!("layers-{l}.safetensors"), hf, shape)
        } else {
            let (hf, shape) = match canon.as_str() {
                "tok.weight" => (format!("{}embed_tokens.weight", decoder_prefix(layout)), vec![cfg.vocab as usize, d]),
                "lm_head.weight" => ("lm_head.weight".to_string(), vec![cfg.vocab as usize, d]),
                "norm.weight" => (format!("{}norm.weight", decoder_prefix(layout)), vec![d]),
                other => panic!("fixture does not know {other}"),
            };
            ("outside.safetensors".to_string(), hf, shape)
        };
        // FP8-quantized in the real release: the big linears, except the GDN
        // in_proj_a/b gates (kept bf16, which also covers a mixed layer).
        let fp8 = dense.is_none() && linear_shapes.contains_key(&canon) && !canon.contains("in_proj_a") && !canon.contains("in_proj_b");
        let list = by_file.entry(file.clone()).or_default();
        weight_map.insert(hf.clone(), file.clone().into());
        if fp8 {
            let bytes: Vec<u8> = (0..numel).map(|_| match (rng.next() & 0xff) as u8 {
                b if b & 0x7f == 0x7f => 0x38, // e4m3fn NaN
                b => b,
            }).collect();
            let (rows, cols) = (shape[0], shape[1]);
            let (rb, cb) = model::fp8::scale_shape(rows, cols, block);
            let scales: Vec<u8> = (0..rb * cb).flat_map(|_| bf16(0.001 + (rng.next() % 1000) as f32 / 100_000.0)).collect();
            list.push(Raw { name: hf.clone(), dtype: "F8_E4M3", shape, bytes });
            let sname = format!("{hf}_scale_inv");
            weight_map.insert(sname.clone(), file.clone().into());
            list.push(Raw { name: sname, dtype: "BF16", shape: vec![rb, cb], bytes: scales });
        } else if let Some(values) = dense {
            let offset = if stored_as_offset_from_one(&canon) { 1.0 } else { 0.0 };
            let v = &values[&canon];
            if canon.ends_with("A_log") {
                list.push(Raw { name: hf, dtype: "F32", shape, bytes: v.iter().flat_map(|x| x.to_le_bytes()).collect() });
            } else {
                list.push(Raw { name: hf, dtype: "BF16", shape, bytes: v.iter().flat_map(|x| bf16(x - offset)).collect() });
            }
        } else if canon.ends_with("A_log") {
            let bytes = (0..numel).flat_map(|_| rng.unit().to_le_bytes()).collect();
            list.push(Raw { name: hf, dtype: "F32", shape, bytes });
        } else {
            // bf16, including the norms (which the (1+w) fold must touch)
            let bytes = (0..numel).flat_map(|_| bf16(rng.unit() * 0.5)).collect();
            list.push(Raw { name: hf, dtype: "BF16", shape, bytes });
        }
    }
    if layout == Layout::Multimodal {
        // Real, present, out-of-scope tensors in their own shard.
        by_file.entry("mtp.safetensors".into()).or_default().push(Raw { name: "mtp.fc.weight".into(), dtype: "BF16", shape: vec![d, 2 * d], bytes: vec![0; d * 2 * d * 2] });
        weight_map.insert("mtp.fc.weight".into(), "mtp.safetensors".into());
        by_file.get_mut("outside.safetensors").unwrap().push(Raw { name: "model.visual.patch_embed.weight".into(), dtype: "BF16", shape: vec![4], bytes: vec![0; 8] });
        weight_map.insert("model.visual.patch_embed.weight".into(), "outside.safetensors".into());
    }

    for (file, tensors) in &by_file {
        write_st(&dir.join(file), tensors);
    }
    let config = match layout {
        Layout::Multimodal => hf_config_json(cfg, block),
        Layout::TextOnly => hf_text_config_json(cfg),
    };
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
    std::fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap()).unwrap();
    dir
}

/// The tensor-name prefix of the decoder in `layout`.
fn decoder_prefix(layout: Layout) -> &'static str {
    match layout {
        Layout::Multimodal => "model.language_model.",
        Layout::TextOnly => "model.",
    }
}

/// The `config.json` of a text-only release for `cfg`: the text config at the
/// top level, with EdgeRock's `fixed_memory_*` and `independent_memory` fields
/// when `cfg` has a window and an anchor memory.
pub fn hf_text_config_json(cfg: &Qwen35Config) -> serde_json::Value {
    let mut c = hf_config_json(cfg, BLOCK)["text_config"].clone();
    c["architectures"] = serde_json::json!(["EdgeRockForCausalLM"]);
    if let (Some(window), Some(mem)) = (cfg.swa_window, cfg.anchor_memory) {
        c["fixed_memory_revision"] = qwen35::config::EDGEROCK_MEMORY_REVISION.into();
        c["fixed_memory_swa_window"] = window.into();
        let full: Vec<usize> = cfg.layer_types().iter().enumerate().filter(|(_, ty)| **ty == qwen35::config::LayerType::Full).map(|(l, _)| l).collect();
        c["fixed_memory_swa_layers"] = serde_json::json!(full);
        c["independent_memory"] = serde_json::json!({
            "capacity": mem.capacity, "max_commit_chunk": mem.commit_chunk,
            "admission_rank": mem.admission_rank, "writer_intermediate_size": mem.writer_intermediate,
        });
    }
    c
}

/// The `config.json` of the multimodal release for `cfg`: the text model
/// nested under `text_config`, with HF's key names, plus the FP8 block size.
pub fn hf_config_json(cfg: &Qwen35Config, block: usize) -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3_5",
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [block, block]},
        "text_config": {
            "vocab_size": cfg.vocab, "num_hidden_layers": cfg.n_layers, "hidden_size": cfg.d_model,
            "num_attention_heads": cfg.n_heads, "num_key_value_heads": cfg.n_kv_heads, "head_dim": cfg.head_dim,
            "rms_norm_eps": cfg.rms_eps, "max_position_embeddings": cfg.max_position_embeddings,
            "attention_bias": cfg.attn_bias, "full_attention_interval": cfg.full_attention_interval,
            "intermediate_size": cfg.intermediate_size, "linear_num_key_heads": cfg.linear_num_key_heads,
            "linear_num_value_heads": cfg.linear_num_value_heads, "linear_key_head_dim": cfg.linear_key_head_dim,
            "linear_value_head_dim": cfg.linear_value_head_dim, "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
            "tie_word_embeddings": cfg.tie_embeddings,
            "rope_parameters": {"mrope_section": cfg.mrope_section, "partial_rotary_factor": cfg.partial_rotary_factor, "rope_theta": cfg.rope_theta, "rope_type": "default"}
        }
    })
}

