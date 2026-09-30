// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export a checkpoint of this decoder as a Hugging Face `transformers`
//! directory: sharded safetensors under the HF tensor names, a `config.json`
//! naming the class the configuration is (`Qwen3ForCausalLM` with QK-norm,
//! `Qwen2ForCausalLM` with q/k/v bias, `LlamaForCausalLM` with neither), and
//! the tokenizer files beside it. What brain trains, any `transformers`
//! stack loads.
//!
//! Tensors stream one at a time from any [`TensorSource`] - a brain file, an
//! HF directory, a GGUF - so an export's peak memory is one tensor, not the
//! model.
//!
//! Swedish Embedded AB implements model training and conversion pipelines
//! like this for its clients. If your team needs expertise in moving models
//! between training stacks, you can procure our services by sending an email
//! to info@swedishembedded.com.

use std::io::Write;
use std::path::Path;

use checkpoint::TensorSource;
use serde_json::{json, Value};

use crate::config::QwenConfig;
use crate::hf::HfNames;

/// The element type an export writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HfDtype {
    F32,
    Bf16,
    F16,
}

impl HfDtype {
    fn bytes(self) -> u64 {
        match self {
            HfDtype::F32 => 4,
            HfDtype::Bf16 | HfDtype::F16 => 2,
        }
    }
    fn safetensors(self) -> &'static str {
        match self {
            HfDtype::F32 => "F32",
            HfDtype::Bf16 => "BF16",
            HfDtype::F16 => "F16",
        }
    }
    fn torch(self) -> &'static str {
        match self {
            HfDtype::F32 => "float32",
            HfDtype::Bf16 => "bfloat16",
            HfDtype::F16 => "float16",
        }
    }
    fn encode(self, x: &[f32], out: &mut Vec<u8>) {
        match self {
            HfDtype::F32 => x.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
            HfDtype::Bf16 => x.iter().for_each(|v| out.extend_from_slice(&half::bf16::from_f32(*v).to_le_bytes())),
            HfDtype::F16 => x.iter().for_each(|v| out.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes())),
        }
    }
}

/// How [`export_hf`] writes.
#[derive(Clone, Debug)]
pub struct HfExport<'a> {
    pub dtype: HfDtype,
    /// The most tensor bytes one shard holds; `transformers` itself shards
    /// at 5 GB.
    pub shard_bytes: u64,
    /// A directory holding the checkpoint's tokenizer files
    /// (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
    /// `generation_config.json`), copied beside the weights when present.
    pub tokenizer_dir: Option<&'a Path>,
}

impl Default for HfExport<'_> {
    fn default() -> Self {
        HfExport { dtype: HfDtype::Bf16, shard_bytes: 5_000_000_000, tokenizer_dir: None }
    }
}

/// The HF architecture a configuration is, as `(class, model_type)`.
fn hf_class(cfg: &QwenConfig) -> Result<(&'static str, &'static str), String> {
    match (cfg.qk_norm, cfg.attn_bias) {
        (true, false) => Ok(("Qwen3ForCausalLM", "qwen3")),
        (false, true) => Ok(("Qwen2ForCausalLM", "qwen2")),
        (false, false) => Ok(("LlamaForCausalLM", "llama")),
        (true, true) => Err("a decoder with both QK-norm and q/k/v bias is no transformers architecture".to_string()),
    }
}

/// The `config.json` `transformers` reads back as `cfg` (every field
/// written, none left to a class default).
pub fn hf_config(cfg: &QwenConfig, dtype: HfDtype) -> Result<Value, String> {
    if cfg.lora.is_some() {
        return Err("export: a LoRA training configuration is not a checkpoint; export the base, and the adapter on its own".to_string());
    }
    let (class, model_type) = hf_class(cfg)?;
    let mut v = json!({
        "architectures": [class],
        "model_type": model_type,
        "vocab_size": cfg.vocab,
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.d_ff,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "hidden_act": "silu",
        "rms_norm_eps": cfg.rms_eps,
        "rope_theta": cfg.rope_theta,
        "max_position_embeddings": cfg.max_position_embeddings,
        "tie_word_embeddings": cfg.tie_embeddings,
        "rope_scaling": cfg.rope_scaling.as_ref().map(|s| s.to_config()),
        "torch_dtype": dtype.torch(),
    });
    if model_type == "llama" {
        // transformers' LlamaAttention biases o_proj too when this is set;
        // brain's decoder has no Llama bias to export.
        v["attention_bias"] = json!(false);
    }
    Ok(v)
}

/// The shape a brain parameter has in the HF checkpoint (`[out, in]` for a
/// linear, `[n]` for a norm or bias).
fn hf_shape(name: &str, cfg: &QwenConfig) -> Result<Vec<u64>, String> {
    let (v, d, ff) = (cfg.vocab as u64, cfg.d_model as u64, cfg.d_ff as u64);
    let (hq, hkv, hd) = (cfg.q_dim() as u64, cfg.kv_dim() as u64, cfg.head_dim as u64);
    let leaf = name.strip_prefix("blocks.").and_then(|r| r.split_once('.')).map_or(name, |(_, leaf)| leaf);
    Ok(match leaf {
        "tok.weight" | "lm_head.weight" => vec![v, d],
        "norm.weight" | "ln1.weight" | "ln2.weight" => vec![d],
        "attn.wq.weight" => vec![hq, d],
        "attn.wk.weight" | "attn.wv.weight" => vec![hkv, d],
        "attn.wo.weight" => vec![d, hq],
        "attn.q_norm.weight" | "attn.k_norm.weight" => vec![hd],
        "attn.wq.bias" => vec![hq],
        "attn.wk.bias" | "attn.wv.bias" => vec![hkv],
        "mlp.gate.weight" | "mlp.up.weight" => vec![ff, d],
        "mlp.down.weight" => vec![d, ff],
        other => return Err(format!("export: no HF shape for {other:?}")),
    })
}

/// One tensor of the export: its brain and HF names, shape and byte size.
struct Planned {
    brain: String,
    hf: String,
    shape: Vec<u64>,
    bytes: u64,
}

/// Write `cfg`'s checkpoint, read from `src`, to `out` as a `transformers`
/// directory (see the module doc). Shards are named
/// `model-0000i-of-0000n.safetensors` with a `model.safetensors.index.json`,
/// or `model.safetensors` alone when one holds everything.
pub fn export_hf(src: &dyn TensorSource, cfg: &QwenConfig, out: &Path, opts: &HfExport) -> Result<(), String> {
    let config = hf_config(cfg, opts.dtype)?;
    let names = HfNames::CAUSAL_LM;
    let plan: Vec<Planned> = cfg
        .param_list()
        .into_iter()
        .map(|(brain, numel)| {
            let hf = names.from_brain(&brain).ok_or_else(|| format!("export: no HF name for {brain:?}"))?;
            let shape = hf_shape(&brain, cfg)?;
            assert_eq!(shape.iter().product::<u64>(), numel as u64, "{brain}: shape and element count disagree");
            Ok(Planned { brain, hf, shape, bytes: numel as u64 * opts.dtype.bytes() })
        })
        .collect::<Result<_, String>>()?;

    // Greedy shards in parameter order, each at most `shard_bytes` unless one
    // tensor alone is larger.
    let mut shards: Vec<Vec<&Planned>> = vec![Vec::new()];
    let mut filled = 0u64;
    for p in &plan {
        if filled > 0 && filled + p.bytes > opts.shard_bytes {
            shards.push(Vec::new());
            filled = 0;
        }
        shards.last_mut().expect("one shard").push(p);
        filled += p.bytes;
    }

    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let n = shards.len();
    let mut weight_map = serde_json::Map::new();
    for (i, shard) in shards.iter().enumerate() {
        let file = if n == 1 { "model.safetensors".to_string() } else { format!("model-{:05}-of-{n:05}.safetensors", i + 1) };
        write_shard(src, shard, opts.dtype, &out.join(&file))?;
        for p in shard {
            weight_map.insert(p.hf.clone(), json!(file));
        }
    }
    if n > 1 {
        let total: u64 = plan.iter().map(|p| p.bytes).sum();
        let index = json!({ "metadata": { "total_size": total }, "weight_map": weight_map });
        write_json(&out.join("model.safetensors.index.json"), &index)?;
    }
    write_json(&out.join("config.json"), &config)?;
    if let Some(dir) = opts.tokenizer_dir {
        for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"] {
            let from = dir.join(f);
            if from.is_file() {
                std::fs::copy(&from, out.join(f)).map_err(|e| format!("{}: {e}", from.display()))?;
            }
        }
    }
    Ok(())
}

fn write_json(path: &Path, v: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(v).map_err(|e| e.to_string())?;
    std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
}

/// One safetensors file: the header, then each tensor as it is read -
/// written to a temporary name and renamed into place, so an interrupted
/// export never leaves a truncated shard under its final name.
fn write_shard(src: &dyn TensorSource, shard: &[&Planned], dtype: HfDtype, path: &Path) -> Result<(), String> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_string(), json!({ "format": "pt" }));
    let mut offset = 0u64;
    for p in shard {
        header.insert(p.hf.clone(), json!({ "dtype": dtype.safetensors(), "shape": p.shape, "data_offsets": [offset, offset + p.bytes] }));
        offset += p.bytes;
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).map_err(|e| e.to_string())?;
    // The data section starts 8-byte aligned, as the format asks.
    header.resize(header.len().div_ceil(8) * 8, b' ');

    let tmp = path.with_extension("safetensors.tmp");
    let io = |e: std::io::Error| format!("{}: {e}", tmp.display());
    let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp).map_err(io)?);
    w.write_all(&(header.len() as u64).to_le_bytes()).map_err(io)?;
    w.write_all(&header).map_err(io)?;
    let mut buf = Vec::new();
    for p in shard {
        buf.clear();
        let found = src.with_tensor(&p.brain, &mut |x| dtype.encode(x, &mut buf));
        if !found {
            return Err(format!("export: the checkpoint has no tensor {:?}", p.brain));
        }
        if buf.len() as u64 != p.bytes {
            return Err(format!("export: {} holds {} bytes, expected {}", p.brain, buf.len(), p.bytes));
        }
        w.write_all(&buf).map_err(io)?;
    }
    w.into_inner().map_err(|e| io(e.into_error()))?.sync_all().map_err(io)?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}
