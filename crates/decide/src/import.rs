// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import a HuggingFace BERT-family encoder (`config.json` +
//! `model.safetensors`) into a brain `.safetensors` container.
//!
//! Convention match: brain's `matmul.wgsl` is `out = x @ Wᵀ` with `W:[out,in]`
//! row-major, which is exactly HF `nn.Linear.weight`, and both embedding tables
//! are `[rows, hidden]`. **No tensor is transposed.**
//!
//! One tensor is RESHAPED rather than copied: query, key and value arrive as
//! three `[H, H]` matrices and are concatenated into one `[3H, H]` in that
//! order, because the attention kernels read a fused `[rows, 3H]` activation
//! with `q_off`/`k_off`/`v_off`. Concatenating the weights is what makes the
//! forward one GEMM per layer instead of three; the order q, k, v is the one
//! `model::block`'s offsets assume, and getting it wrong is the failure the
//! `l0.attn_ctx` rung of the parity ladder exists to catch.
//!
//! The write is streamed and out of order: `StWriter` seeks to a slot by name,
//! so q/k/v are held (three `[H, H]` blocks, under 2 MB for this family) only
//! until a layer's third arrives.
//!
//! Two tensors are deliberately dropped, and nothing else may be:
//! `embeddings.position_ids` is a non-parameter index buffer, and `pooler.*` is
//! BERT's tanh CLS head, which sentence-transformers does not use and this
//! model does not either (it mean-pools).

use crate::Tensors;
use std::collections::HashMap;
use std::path::Path;

use crate::config::EncoderConfig;

/// Where one HF tensor goes.
enum Dest {
    /// Straight to a brain parameter of the same shape.
    Param(String),
    /// Into a third of layer `layer`'s fused QKV: `part` is 0=q, 1=k, 2=v.
    Qkv { layer: u32, part: usize, bias: bool },
    /// Not a parameter of this model - see the module docs for the only two.
    Drop,
}

fn hf_to_brain(name: &str) -> Option<Dest> {
    // Some releases nest the encoder under `bert.`; MiniLM's does not.
    let name = name.strip_prefix("bert.").unwrap_or(name);
    let p = |s: &str| Some(Dest::Param(s.to_string()));
    match name {
        "embeddings.word_embeddings.weight" => return p("tok.weight"),
        "embeddings.position_embeddings.weight" => return p("pos.weight"),
        "embeddings.token_type_embeddings.weight" => return p("type.weight"),
        "embeddings.LayerNorm.weight" => return p("emb_ln.weight"),
        "embeddings.LayerNorm.bias" => return p("emb_ln.bias"),
        // A registered index buffer, not a weight.
        "embeddings.position_ids" | "embeddings.token_type_ids" => return Some(Dest::Drop),
        // BERT's tanh CLS head. sentence-transformers mean-pools instead and
        // never loads it; neither does this model.
        "pooler.dense.weight" | "pooler.dense.bias" => return Some(Dest::Drop),
        _ => {}
    }
    let rest = name.strip_prefix("encoder.layer.")?;
    let (n, rest) = rest.split_once('.')?;
    let layer: u32 = n.parse().ok()?;
    let qkv = |part: usize, bias: bool| Some(Dest::Qkv { layer, part, bias });
    match rest {
        "attention.self.query.weight" => qkv(0, false),
        "attention.self.key.weight" => qkv(1, false),
        "attention.self.value.weight" => qkv(2, false),
        "attention.self.query.bias" => qkv(0, true),
        "attention.self.key.bias" => qkv(1, true),
        "attention.self.value.bias" => qkv(2, true),
        "attention.output.dense.weight" => p(&format!("blocks.{layer}.proj.weight")),
        "attention.output.dense.bias" => p(&format!("blocks.{layer}.proj.bias")),
        "attention.output.LayerNorm.weight" => p(&format!("blocks.{layer}.ln1.weight")),
        "attention.output.LayerNorm.bias" => p(&format!("blocks.{layer}.ln1.bias")),
        "intermediate.dense.weight" => p(&format!("blocks.{layer}.fc1.weight")),
        "intermediate.dense.bias" => p(&format!("blocks.{layer}.fc1.bias")),
        "output.dense.weight" => p(&format!("blocks.{layer}.fc2.weight")),
        "output.dense.bias" => p(&format!("blocks.{layer}.fc2.bias")),
        "output.LayerNorm.weight" => p(&format!("blocks.{layer}.ln2.weight")),
        "output.LayerNorm.bias" => p(&format!("blocks.{layer}.ln2.bias")),
        _ => None,
    }
}

/// Read an HF BERT `config.json`.
///
/// Every architectural choice this encoder does NOT implement is refused by
/// name rather than ignored: a relative-position checkpoint or a GELU-variant
/// checkpoint would load cleanly and be quietly wrong, which is the failure
/// mode a coverage check alone cannot see.
pub fn config_from_hf(json: &str) -> Result<EncoderConfig, String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| format!("config.json: {e}"))?;
    let u = |k: &str| v.get(k).and_then(serde_json::Value::as_u64).map(|x| x as u32);
    let s = |k: &str| v.get(k).and_then(serde_json::Value::as_str);
    if let Some(mt) = s("model_type") {
        if mt != "bert" {
            return Err(format!("config: model_type {mt:?}, expected \"bert\""));
        }
    }
    match s("position_embedding_type") {
        None | Some("absolute") => {}
        Some(o) => return Err(format!("config: position_embedding_type {o:?} is not implemented (absolute only)")),
    }
    match s("hidden_act") {
        // HF's "gelu" is the exact erf form, which is `kernels::GELU_ERF`. The
        // tanh approximation is a DIFFERENT function and ships under its own
        // name, so accepting it here would silently change every activation.
        None | Some("gelu") => {}
        Some(o) => return Err(format!("config: hidden_act {o:?} is not implemented (exact gelu only)")),
    }
    Ok(EncoderConfig {
        vocab: u("vocab_size").ok_or("config: vocab_size")?,
        max_positions: u("max_position_embeddings").ok_or("config: max_position_embeddings")?,
        d_model: u("hidden_size").ok_or("config: hidden_size")?,
        n_layers: u("num_hidden_layers").ok_or("config: num_hidden_layers")?,
        n_heads: u("num_attention_heads").ok_or("config: num_attention_heads")?,
        d_ff: u("intermediate_size").ok_or("config: intermediate_size")?,
        type_vocab: u("type_vocab_size").unwrap_or(2),
        eps: v.get("layer_norm_eps").and_then(serde_json::Value::as_f64).unwrap_or(1e-12) as f32,
    })
}

/// Collects the three parts of one layer's QKV until all have arrived.
struct QkvJoin {
    /// `[part][..]` - empty until that part is seen.
    parts: [Vec<f32>; 3],
    seen: usize,
}

impl QkvJoin {
    fn new() -> QkvJoin {
        QkvJoin { parts: [Vec::new(), Vec::new(), Vec::new()], seen: 0 }
    }

    /// Record one part; returns the concatenation once the third arrives.
    fn put(&mut self, part: usize, data: Vec<f32>) -> Option<Vec<f32>> {
        if !self.parts[part].is_empty() {
            return None; // duplicate - the caller's coverage check reports it
        }
        self.parts[part] = data;
        self.seen += 1;
        if self.seen < 3 {
            return None;
        }
        let mut out = Vec::with_capacity(self.parts.iter().map(Vec::len).sum());
        for p in &mut self.parts {
            out.append(p);
        }
        Some(out)
    }
}

/// Which naming convention a checkpoint's tensors are written in.
///
/// A caller passes a PATH, not a format: which convention a file uses is a
/// fact about the file, and asking the caller to declare it is asking them
/// to be right about something the file already says. The same reasoning
/// `qwen3::import::Naming` follows for GGUF-vs-safetensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Naming {
    /// As released by `sentence-transformers` / Hugging Face:
    /// `encoder.layer.0.attention.self.query.weight`, q/k/v separate.
    Hf,
    /// As this crate names its own parameters: `blocks.0.qkv.weight`, q/k/v
    /// fused. What [`crate::decide::Decide::save_model`] writes.
    Brain,
}

/// Read the convention off the tensor names themselves.
///
/// Errs rather than guessing when the names match neither, because a
/// defaulted guess here is a whole model silently loaded from the wrong
/// mapping - the tensors are shape-compatible often enough for that to
/// produce a model that runs.
pub fn sniff_naming(names: &[String], cfg: &EncoderConfig) -> Result<Naming, String> {
    let brain: std::collections::HashSet<String> =
        cfg.tensor_manifest().into_iter().map(|(n, _)| n).collect();
    if names.iter().any(|n| brain.contains(n)) {
        return Ok(Naming::Brain);
    }
    if names.iter().any(|n| hf_to_brain(n).is_some()) {
        return Ok(Naming::Hf);
    }
    let sample: Vec<&String> = names.iter().take(4).collect();
    Err(format!(
        "import: {} tensors match neither this crate's own names nor the Hugging Face \
         encoder layout (first few: {sample:?})",
        names.len()
    ))
}

/// Take an already-read tensor list written in THIS crate's own names and
/// check it against the encoder manifest.
///
/// The same two-way coverage [`brain_init_from_hf`] applies to an imported
/// checkpoint: every encoder parameter present exactly once at the right
/// element count. Tensors outside the encoder manifest are returned
/// separately rather than rejected - a saved decision model carries its head
/// in the same file, and the head's manifest belongs to `crate::head`, not
/// here.
pub fn split_brain_tensors(
    tensors: Vec<checkpoint::safetensors::StTensor>,
    cfg: &EncoderConfig,
) -> Result<(Tensors, Tensors), String> {
    let mut seen: Tensors = HashMap::new();
    for t in tensors {
        if seen.insert(t.name.clone(), t.data).is_some() {
            return Err(format!("import: duplicate tensor {}", t.name));
        }
    }
    let mut enc: Tensors = HashMap::new();
    for (name, shape) in cfg.tensor_manifest() {
        let numel: usize = shape.iter().product();
        let data = seen
            .remove(&name)
            .ok_or_else(|| format!("import: missing tensor for brain param {name}"))?;
        if data.len() != numel {
            return Err(format!("import: {name} element count {} != expected {numel}", data.len()));
        }
        enc.insert(name, data);
    }
    Ok((enc, seen))
}

/// Remap an already-read HF tensor list into brain's `name -> f32` init map,
/// with two-way coverage: every brain parameter produced exactly once with the
/// right element count, and no mapped HF tensor left over.
pub fn brain_init_from_hf(
    tensors: Vec<checkpoint::safetensors::StTensor>,
    cfg: &EncoderConfig,
) -> Result<Tensors, String> {
    let mut brain: Tensors = HashMap::new();
    let mut joins: HashMap<(u32, bool), QkvJoin> = HashMap::new();
    let mut unmapped: Vec<String> = Vec::new();
    for t in tensors {
        match hf_to_brain(&t.name) {
            Some(Dest::Param(bn)) => {
                if brain.insert(bn.clone(), t.data).is_some() {
                    return Err(format!("import: duplicate mapping to {bn}"));
                }
            }
            Some(Dest::Qkv { layer, part, bias }) => {
                let j = joins.entry((layer, bias)).or_insert_with(QkvJoin::new);
                if let Some(fused) = j.put(part, t.data) {
                    let leaf = if bias { "bias" } else { "weight" };
                    brain.insert(format!("blocks.{layer}.qkv.{leaf}"), fused);
                }
            }
            Some(Dest::Drop) => {}
            None => unmapped.push(t.name),
        }
    }
    if !unmapped.is_empty() {
        return Err(format!("import: {} unmapped HF tensors: {unmapped:?}", unmapped.len()));
    }
    let mut init: Tensors = HashMap::new();
    for (name, shape) in cfg.tensor_manifest() {
        let numel: usize = shape.iter().product();
        let data = brain.remove(&name).ok_or_else(|| format!("import: missing tensor for brain param {name}"))?;
        if data.len() != numel {
            return Err(format!("import: {name} element count {} != expected {numel}", data.len()));
        }
        init.insert(name, data);
    }
    if !brain.is_empty() {
        let mut extra: Vec<&String> = brain.keys().collect();
        extra.sort();
        return Err(format!("import: {} mapped HF tensors unused: {extra:?}", brain.len()));
    }
    Ok(init)
}

/// Import `<hf_dir>/config.json` + its weights into the brain checkpoint
/// `out_path`. Never writes a partial checkpoint.
pub fn import(hf_dir: &str, out_path: &str) -> Result<(), String> {
    import_as(hf_dir, out_path, None)
}

/// [`import`] with the card `id` overridden - the model-store auto-fetch
/// dispatcher needs the fully-qualified `vendor/repo` reference rather than a
/// filesystem-derived name.
pub fn import_as(hf_dir: &str, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
    let dir = Path::new(hf_dir);
    let cfg_json =
        std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("read config.json: {e}"))?;
    let cfg = config_from_hf(&cfg_json)?;

    let plan: Vec<(String, Vec<u64>)> = cfg
        .tensor_manifest()
        .into_iter()
        .map(|(name, shape)| (name, shape.into_iter().map(|d| d as u64).collect()))
        .collect();
    let param_count: u64 = plan.iter().map(|(_, s)| s.iter().product::<u64>()).sum();
    let id = id_override
        .unwrap_or_else(|| Path::new(out_path).file_stem().and_then(|s| s.to_str()).unwrap_or("decide"));
    let mut card = checkpoint::st::ModelCard::new(id, "decide");
    card.context_length = Some(cfg.max_positions as u64);
    card.param_count = Some(param_count);

    let mut writer = checkpoint::weightio::StWriter::create(out_path, &plan, &cfg.to_json(), Some(&card))
        .map_err(|e| format!("create {out_path}: {e}"))?;
    let reader =
        checkpoint::weightio::WeightReader::open_hf_dir(dir).map_err(|e| format!("open {hf_dir}: {e}"))?;

    let mut err: Option<String> = None;
    let mut unmapped: Vec<String> = Vec::new();
    let mut joins: HashMap<(u32, bool), QkvJoin> = HashMap::new();
    let mut n_written = 0usize;
    reader.for_each(|name, _shape, data| {
        if err.is_some() {
            return;
        }
        let mut put = |bn: String, d: &[f32]| {
            if let Err(e) = writer.write(&bn, d) {
                err = Some(format!("write {bn}: {e}"));
            } else {
                n_written += 1;
            }
        };
        match hf_to_brain(name) {
            Some(Dest::Param(bn)) => put(bn, &data),
            Some(Dest::Qkv { layer, part, bias }) => {
                let j = joins.entry((layer, bias)).or_insert_with(QkvJoin::new);
                if let Some(fused) = j.put(part, data.to_vec()) {
                    let leaf = if bias { "bias" } else { "weight" };
                    put(format!("blocks.{layer}.qkv.{leaf}"), &fused);
                }
            }
            Some(Dest::Drop) => {}
            None => unmapped.push(name.to_string()),
        }
    });
    if let Some(e) = err {
        let _ = std::fs::remove_file(out_path);
        return Err(e);
    }
    if !unmapped.is_empty() {
        let _ = std::fs::remove_file(out_path);
        return Err(format!("import: {} unmapped HF tensors: {unmapped:?}", unmapped.len()));
    }
    if n_written != plan.len() {
        let _ = std::fs::remove_file(out_path);
        return Err(format!("import: wrote {n_written} of {} parameters", plan.len()));
    }
    writer.finish().map_err(|e| format!("finish {out_path}: {e}"))?;
    Ok(())
}
