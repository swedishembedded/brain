// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export a checkpoint of this decoder as a Hugging Face `transformers`
//! directory ([`export_hf`]) or a llama.cpp GGUF ([`export_gguf`]).
//!
//! The `transformers` directory holds sharded safetensors under the HF tensor
//! names, a `config.json` naming the class the configuration is (`Qwen3ForCausalLM` with QK-norm,
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

use checkpoint::gguf::GgufValue;
use checkpoint::TensorSource;
use model::rope_scaling::RopeScaling;
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

/// The element type of a GGUF export's matrices (norms and biases are
/// always F32, as llama.cpp writes them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufDtype {
    F32,
    F16,
}

/// How [`export_gguf`] writes.
#[derive(Clone, Debug)]
pub struct GgufExport<'a> {
    pub dtype: GgufDtype,
    /// The checkpoint's tokenizer directory (`tokenizer.json`,
    /// `tokenizer_config.json`), embedded as the file's `tokenizer.ggml.*`
    /// keys: llama.cpp loads no GGUF without its vocabulary.
    pub tokenizer_dir: &'a Path,
    /// `general.name`.
    pub name: &'a str,
}

/// The pre-tokenizer of every tokenizer family this exporter embeds, by the
/// SHA-256 of its `tokenizer.json` `pre_tokenizer` (compact, keys sorted),
/// and the `tokenizer.ggml.pre` llama.cpp runs that pre-tokenizer as.
const PRE_TOKENIZERS: [(&str, &str); 4] = [
    // Qwen2/2.5/3 and the R1 Qwen distills.
    ("78d576476d401e55be4850cbdeeb3183809b5bb60da2d2e20872c6d43a2db99e", "qwen2"),
    // Llama 3 and the R1 Llama distill.
    ("65e1fe6fbe22e0df7a3877257a5e19e7b0336324a8678a75de61b6fc0df0e147", "llama-bpe"),
    // deepseek-coder 1.3b/6.7b.
    ("421985d88074b163d9dcc3f29f9b34dcac5cfd5a43e9eeae965ede66f7ac82cb", "deepseek-coder"),
    // deepseek-llm, deepseek-math and deepseek-coder v1.5.
    ("3fbf36c22d6c95d88cb3fdec944559ffbb3b7ed6df84913bd6bd16db03fc878d", "deepseek-llm"),
];

/// The `tokenizer.ggml.pre` for a `tokenizer.json`, or an error naming the
/// unknown pre-tokenizer: llama.cpp would split text differently from the
/// model's training, so an unrecognized one is refused, not guessed.
pub fn gguf_pre_tokenizer(tokenizer_json: &Value) -> Result<&'static str, String> {
    let pre = tokenizer_json.get("pre_tokenizer").ok_or("tokenizer.json has no pre_tokenizer")?;
    let hash = brain_modelstore::fetch::bytes_digest(serde_json::to_string(pre).map_err(|e| e.to_string())?.as_bytes());
    PRE_TOKENIZERS
        .iter()
        .find(|(h, _)| *h == hash)
        .map(|(_, name)| *name)
        .ok_or_else(|| format!("tokenizer.json: pre-tokenizer {hash} is not one this exporter knows the llama.cpp name of"))
}

/// A control token by its look, as llama.cpp's converter decides it for an
/// added token not flagged special (`does_token_look_special`:
/// deepseek-coder's `<pad>` is one).
fn looks_special(t: &str) -> bool {
    matches!(t, "<pad>" | "<mask>" | "<2mass>" | "[@BOS@]")
        || (t.starts_with("<|") && t.ends_with("|>"))
        || (t.starts_with("<｜") && t.ends_with("｜>"))
        || (t.starts_with("<unused") && t.ends_with('>'))
}

/// Whether encoding adds a BOS and an EOS, as llama.cpp's converter reads it
/// (`gguf.SpecialVocab`): the post-processor decides - `ByteLevel` adds
/// neither, a `TemplateProcessing` adds each its single-sequence template
/// opens or closes with - and `tokenizer_config.json`'s `add_*_token` only
/// where the post-processor says nothing.
fn adds_bos_eos(tj: &Value, tc: &Value) -> (Option<bool>, Option<bool>) {
    let (mut bos, mut eos) = (None, None);
    let post = &tj["post_processor"];
    let processors: Vec<&Value> = match post["processors"].as_array() {
        Some(list) => list.iter().collect(),
        None if !post.is_null() => vec![post],
        None => Vec::new(),
    };
    for p in processors {
        match p["type"].as_str() {
            Some("ByteLevel") => {
                bos.get_or_insert(false);
                eos.get_or_insert(false);
            }
            Some("TemplateProcessing") => {
                let single = p["single"].as_array().map(Vec::as_slice).unwrap_or_default();
                let special = |v: Option<&Value>| v.is_some_and(|v| v.get("SpecialToken").is_some());
                if single.len() > 1 {
                    bos = Some(special(single.first()));
                    eos = Some(special(single.last()));
                }
            }
            _ => {}
        }
    }
    (bos.or(tc["add_bos_token"].as_bool()), eos.or(tc["add_eos_token"].as_bool()))
}

/// The `tokenizer.ggml.*` keys (and chat template) of a BPE `tokenizer.json`
/// for a model with `rows` embedding rows: every id's token (`[PADn]` for a
/// row no token uses), its type (normal 1, control 3, user-defined 4,
/// unused 5), the merges, the pre-tokenizer and the special ids.
fn tokenizer_kv(dir: &Path, rows: u32) -> Result<Vec<(String, GgufValue)>, String> {
    let read = |f: &str| -> Result<Value, String> {
        let p = dir.join(f);
        serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?).map_err(|e| format!("{}: {e}", p.display()))
    };
    let tj = read("tokenizer.json")?;
    if tj["model"]["type"] != "BPE" {
        return Err(format!("tokenizer.json: model type {} - only BPE tokenizers are exported", tj["model"]["type"]));
    }
    let pre = gguf_pre_tokenizer(&tj)?;
    let mut tokens: Vec<Option<(String, i32)>> = vec![None; rows as usize];
    let mut put = |id: u64, text: &str, ty: i32| -> Result<(), String> {
        let slot = tokens.get_mut(id as usize).ok_or_else(|| format!("tokenizer.json: token {text:?} has id {id}, past the model's {rows} rows"))?;
        *slot = Some((text.to_string(), ty));
        Ok(())
    };
    for (text, id) in tj["model"]["vocab"].as_object().ok_or("tokenizer.json: model.vocab is not an object")? {
        put(id.as_u64().ok_or("tokenizer.json: non-integer vocab id")?, text, 1)?;
    }
    for t in tj["added_tokens"].as_array().into_iter().flatten() {
        let (Some(id), Some(text)) = (t["id"].as_u64(), t["content"].as_str()) else { continue };
        if t["special"].as_bool() == Some(true) || looks_special(text) {
            put(id, text, 3)?;
        } else {
            put(id, &text.replace('▁', " "), 4)?;
        }
    }
    let (texts, types): (Vec<GgufValue>, Vec<GgufValue>) = tokens
        .into_iter()
        .enumerate()
        .map(|(i, t)| {
            let (text, ty) = t.unwrap_or_else(|| (format!("[PAD{i}]"), 5));
            (GgufValue::String(text), GgufValue::I32(ty))
        })
        .unzip();
    let merges: Vec<GgufValue> = tj["model"]["merges"]
        .as_array()
        .ok_or("tokenizer.json: model.merges is not an array")?
        .iter()
        .map(|m| match m {
            Value::String(s) => Ok(GgufValue::String(s.clone())),
            Value::Array(p) if p.len() == 2 => Ok(GgufValue::String(format!("{} {}", p[0].as_str().unwrap_or(""), p[1].as_str().unwrap_or("")))),
            other => Err(format!("tokenizer.json: merge {other} is neither \"a b\" nor [a, b]")),
        })
        .collect::<Result<_, String>>()?;

    let mut kv = vec![
        ("tokenizer.ggml.model".to_string(), GgufValue::String("gpt2".into())),
        ("tokenizer.ggml.pre".to_string(), GgufValue::String(pre.into())),
        ("tokenizer.ggml.tokens".to_string(), GgufValue::Array(texts)),
        ("tokenizer.ggml.token_type".to_string(), GgufValue::Array(types)),
        ("tokenizer.ggml.merges".to_string(), GgufValue::Array(merges)),
    ];
    // Special ids and the chat template, from tokenizer_config.json.
    if let Ok(tc) = read("tokenizer_config.json") {
        let content = |v: &Value| v.as_str().map(str::to_string).or_else(|| v["content"].as_str().map(str::to_string));
        let id_of = |text: &str| -> Option<u32> {
            tj["added_tokens"].as_array().into_iter().flatten().find(|t| t["content"] == text).and_then(|t| t["id"].as_u64()).map(|i| i as u32).or_else(|| tj["model"]["vocab"][text].as_u64().map(|i| i as u32))
        };
        for (key, field) in [("bos_token_id", "bos_token"), ("eos_token_id", "eos_token"), ("unknown_token_id", "unk_token"), ("padding_token_id", "pad_token")] {
            if let Some(id) = content(&tc[field]).and_then(|t| id_of(&t)) {
                kv.push((format!("tokenizer.ggml.{key}"), GgufValue::U32(id)));
            }
        }
        let (bos, eos) = adds_bos_eos(&tj, &tc);
        for (key, add) in [("add_bos_token", bos), ("add_eos_token", eos)] {
            if let Some(add) = add {
                kv.push((format!("tokenizer.ggml.{key}"), GgufValue::Bool(add)));
            }
        }
        if let Some(t) = tc["chat_template"].as_str() {
            kv.push(("tokenizer.chat_template".to_string(), GgufValue::String(t.to_string())));
        }
    }
    Ok(kv)
}

/// Write `cfg`'s checkpoint, read from `src`, as one llama.cpp GGUF at
/// `path`, under the architecture the configuration is (`qwen3`, `qwen2`,
/// `llama`), with its tokenizer embedded. A llama checkpoint's q/k rows are
/// permuted into the interleaved order llama.cpp's RoPE expects (the inverse
/// of what the importer undoes), and a llama3 or per-frequency RoPE scaling
/// is written as the `rope_freqs.weight` divisors llama.cpp reads.
pub fn export_gguf(src: &dyn TensorSource, cfg: &QwenConfig, path: &Path, opts: &GgufExport) -> Result<(), String> {
    use checkpoint::gguf::GgmlType;
    let (_, arch) = hf_class(cfg)?;
    if cfg.lora.is_some() {
        return Err("export: a LoRA training configuration is not a checkpoint".to_string());
    }
    let a = |k: &str| format!("{arch}.{k}");
    let mut kv = vec![
        ("general.architecture".to_string(), GgufValue::String(arch.into())),
        ("general.name".to_string(), GgufValue::String(opts.name.into())),
        ("general.file_type".to_string(), GgufValue::U32(if opts.dtype == GgufDtype::F16 { 1 } else { 0 })),
        ("general.quantization_version".to_string(), GgufValue::U32(2)),
        (a("vocab_size"), GgufValue::U32(cfg.vocab)),
        (a("context_length"), GgufValue::U32(cfg.max_position_embeddings)),
        (a("embedding_length"), GgufValue::U32(cfg.d_model)),
        (a("block_count"), GgufValue::U32(cfg.n_layers)),
        (a("feed_forward_length"), GgufValue::U32(cfg.d_ff)),
        (a("attention.head_count"), GgufValue::U32(cfg.n_heads)),
        (a("attention.head_count_kv"), GgufValue::U32(cfg.n_kv_heads)),
        (a("attention.key_length"), GgufValue::U32(cfg.head_dim)),
        (a("attention.value_length"), GgufValue::U32(cfg.head_dim)),
        (a("rope.dimension_count"), GgufValue::U32(cfg.head_dim)),
        (a("rope.freq_base"), GgufValue::F32(cfg.rope_theta)),
        (a("attention.layer_norm_rms_epsilon"), GgufValue::F32(cfg.rms_eps)),
    ];
    let mut rope_freqs: Option<Vec<f32>> = None;
    match &cfg.rope_scaling {
        None => {}
        Some(RopeScaling::Linear { factor }) => {
            kv.push((a("rope.scaling.type"), GgufValue::String("linear".into())));
            kv.push((a("rope.scaling.factor"), GgufValue::F32(*factor)));
        }
        Some(RopeScaling::Yarn(y)) => {
            kv.push((a("rope.scaling.type"), GgufValue::String("yarn".into())));
            kv.push((a("rope.scaling.factor"), GgufValue::F32(y.factor)));
            kv.push((a("rope.scaling.original_context_length"), GgufValue::U32(y.original_max_position_embeddings)));
            kv.push((a("rope.scaling.yarn_beta_fast"), GgufValue::F32(y.beta_fast)));
            kv.push((a("rope.scaling.yarn_beta_slow"), GgufValue::F32(y.beta_slow)));
            if let Some(f) = y.attention_factor {
                kv.push((a("rope.scaling.yarn_attn_factor"), GgufValue::F32(f)));
            }
        }
        Some(s @ (RopeScaling::Llama3 { .. } | RopeScaling::Factors(_))) => {
            // llama.cpp divides the base inverse frequencies by these.
            let base = model::rope_scaling::base_inv_freq(cfg.head_dim, cfg.rope_theta);
            let (scaled, _) = s.inv_freq(cfg.head_dim, cfg.rope_theta);
            rope_freqs = Some(match s {
                RopeScaling::Factors(f) => f.clone(),
                _ => base.iter().zip(&scaled).map(|(b, s)| b / s).collect(),
            });
        }
    }
    kv.extend(tokenizer_kv(opts.tokenizer_dir, cfg.vocab)?);

    // Every tensor's name, shape, type and permutation, planned up front: the
    // header carries each one's offset before any bytes are written.
    let permute_heads = |name: &str| -> Option<u32> {
        (arch == "llama").then_some(()).and_then(|_| {
            if name.ends_with("attn.wq.weight") {
                Some(cfg.n_heads)
            } else if name.ends_with("attn.wk.weight") {
                Some(cfg.n_kv_heads)
            } else {
                None
            }
        })
    };
    struct Out {
        brain: Option<String>,
        gguf: String,
        shape: Vec<usize>,
        ty: GgmlType,
        permute: Option<u32>,
    }
    let mut outs: Vec<Out> = Vec::new();
    for (brain, _) in cfg.param_list() {
        let gguf = crate::gguf_import::brain_to_gguf(&brain).ok_or_else(|| format!("export: no GGUF name for {brain:?}"))?;
        let shape: Vec<usize> = hf_shape(&brain, cfg)?.into_iter().map(|d| d as usize).collect();
        let ty = if shape.len() == 1 || opts.dtype == GgufDtype::F32 { GgmlType::F32 } else { GgmlType::F16 };
        outs.push(Out { permute: permute_heads(&brain), brain: Some(brain), gguf, shape, ty });
    }
    if let Some(f) = &rope_freqs {
        outs.push(Out { brain: None, gguf: "rope_freqs.weight".into(), shape: vec![f.len()], ty: GgmlType::F32, permute: None });
    }
    let bytes = |o: &Out| o.shape.iter().product::<usize>() * if o.ty == GgmlType::F16 { 2 } else { 4 };
    let plan = outs.iter().map(|o| checkpoint::gguf_write::TensorPlan { name: o.gguf.clone(), shape: o.shape.clone(), ty: o.ty.id(), nbytes: bytes(o) }).collect();
    let mut w = checkpoint::gguf_write::Writer::create(&path.to_string_lossy(), &kv, plan, 32).map_err(|e| format!("{}: {e}", path.display()))?;
    for o in &outs {
        let mut data = match &o.brain {
            None => rope_freqs.clone().expect("planned with it"),
            Some(brain) => {
                let mut v = Vec::new();
                if !src.with_tensor(brain, &mut |x| v.extend_from_slice(x)) {
                    return Err(format!("export: the checkpoint has no tensor {brain:?}"));
                }
                v
            }
        };
        if let Some(n) = o.permute {
            // The importer reads brain row r from stored row order[r]; the
            // export stores brain row r at order[r].
            let cols = o.shape[1];
            let order = crate::gguf_import::llama_unpermute_order(n as usize, cfg.head_dim as usize);
            let mut stored = vec![0f32; data.len()];
            for (r, &from) in order.iter().enumerate() {
                stored[from as usize * cols..(from as usize + 1) * cols].copy_from_slice(&data[r * cols..(r + 1) * cols]);
            }
            data = stored;
        }
        let encoded: Vec<u8> = match o.ty {
            GgmlType::F16 => data.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            _ => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        w.write_tensor(&o.gguf, &encoded).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    w.finish().map_err(|e| format!("{}: {e}", path.display()))
}

/// Write a LoRA adapter brain trained for this decoder (`adapter_path`, as
/// `qwen3::lora::save_adapter` writes it) as a PEFT adapter directory:
/// `adapter_model.safetensors` under PEFT's names
/// (`base_model.model.<HF module>.lora_A.weight` `[r, in]`, `.lora_B.weight`
/// `[out, r]`) and an `adapter_config.json` carrying the rank, `lora_alpha`
/// and target modules, so `peft.PeftModel.from_pretrained` applies it to the
/// HF checkpoint exactly as brain applies it to its own: `(alpha / r) · B·A`.
/// `base_model` names the HF checkpoint it applies to; `None` takes the base
/// the adapter's own card records.
pub fn export_peft(adapter_path: &str, out: &Path, base_model: Option<&str>) -> Result<(), String> {
    let st = checkpoint::st::load_safetensors(adapter_path).map_err(|e| format!("{adapter_path}: {e}"))?;
    let card = st.card().ok_or_else(|| format!("{adapter_path}: no ModelCard"))?;
    let adapter = card.adapter.as_ref().ok_or_else(|| format!("{adapter_path}: its card describes no adapter"))?;
    if adapter.kind != "lora" {
        return Err(format!("{adapter_path}: a {:?} adapter has no PEFT LoRA form", adapter.kind));
    }
    let r = adapter.rank.filter(|&r| r > 0).ok_or_else(|| format!("{adapter_path}: the adapter has no rank"))? as usize;
    let alpha = adapter.alpha.unwrap_or(r as f32);
    let base_model = base_model.map(str::to_string).or_else(|| adapter.base.clone()).ok_or_else(|| format!("{adapter_path}: no base model given and none on its card"))?;

    let names = HfNames::CAUSAL_LM;
    let mut bases: Vec<&str> = st.tensors.keys().filter_map(|n| n.strip_suffix(".lora_a")).collect();
    bases.sort();
    if bases.is_empty() {
        return Err(format!("{adapter_path}: no .lora_a/.lora_b pairs"));
    }
    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    let mut modules: Vec<String> = Vec::new();
    for base in bases {
        let a = &st.tensors[&format!("{base}.lora_a")];
        let b = st.tensors.get(&format!("{base}.lora_b")).ok_or_else(|| format!("{adapter_path}: {base}.lora_a has no .lora_b"))?;
        if a.len() % r != 0 || b.len() % r != 0 {
            return Err(format!("{adapter_path}: {base}'s factors are not rank-{r} matrices"));
        }
        let hf = names.from_brain(base).ok_or_else(|| format!("{adapter_path}: no HF name for {base}"))?;
        let stem = hf.strip_suffix(".weight").ok_or_else(|| format!("{adapter_path}: {base} is not a weight"))?;
        let module = stem.rsplit('.').next().unwrap_or(stem).to_string();
        if !modules.contains(&module) {
            modules.push(module);
        }
        tensors.push((format!("base_model.model.{stem}.lora_A.weight"), vec![r as u64, (a.len() / r) as u64], a.clone()));
        tensors.push((format!("base_model.model.{stem}.lora_B.weight"), vec![(b.len() / r) as u64, r as u64], b.clone()));
    }
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    checkpoint::st::save_safetensors(&out.join("adapter_model.safetensors").to_string_lossy(), &tensors, &Value::Null, None).map_err(|e| format!("{}: {e}", out.display()))?;
    let config = json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "r": r,
        "lora_alpha": alpha,
        "target_modules": modules,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": false,
        "inference_mode": true,
    });
    write_json(&out.join("adapter_config.json"), &config)
}
