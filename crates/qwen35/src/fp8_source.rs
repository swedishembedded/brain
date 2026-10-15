// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A lazy [`checkpoint::TensorSource`] over a Qwen3.5/3.8 HuggingFace checkpoint
//! directory (the shipped `Qwen/Qwen3.8-27B-FP8` layout: `layers-N.safetensors`
//! plus `outside.safetensors`, or any bf16/fp32 HF layout of the same tensor
//! names).
//!
//! [`crate::import::import_dir`] dequantizes the whole model into one host f32
//! map (about 108 GB for the 27B) and so cannot run under the host limit, and
//! [`crate::import::import_layer`] is per layer. This source does the same work
//! per TENSOR, on demand, under the brain-canonical names the model builders
//! ask for (`blocks.{l}.mlp.gate.weight`, `tok.weight`, ...): FP8 blocks are
//! dequantized by the same `model::fp8::dequant_block128`, names go through the
//! same `import::classify`, and plain RMSNorm weights get the same `(1+w)` fold,
//! so every tensor is bit-identical to what the eager importers produce (the
//! unit tests assert exactly that). Host peak while a tensor is consumed is
//! that one tensor (`with_tensor`) or `O(chunk)` (`with_tensor_chunks`, which
//! dequantizes whole 128-row block bands at a time so a 5 GB embedding or head
//! never has to exist on the host at once).
//!
//! The model builder pulls each tensor, uploads it at the requested tier
//! (`Weight::upload`: bf16, or NF4 quantized on the way in) and drops it, so the
//! frozen base can sit on the device without a host copy of the model.
//!
//! Swedish Embedded AB implements solutions for loading very large quantized
//! checkpoints onto a single accelerator within a small host memory budget for
//! its clients. If your team needs expertise in memory-bounded model loading or
//! low-precision fine-tuning then you can procure our services by sending an
//! email to info@swedishembedded.com.

use std::collections::HashMap;
use std::path::Path;

use checkpoint::mmap::MmapSafetensors;
use checkpoint::TensorSource;
use serde_json::Value;

use crate::config::Qwen35Config;
use crate::import::{classify, is_plain_rmsnorm_weight};

/// Element budget of the transient dequantized band `with_tensor` assembles a
/// whole FP8 tensor from (bounds the extra scratch next to the output).
const WHOLE_TENSOR_BAND_ELEMS: usize = 1 << 22;

/// Where one canonical tensor lives and what must be done to its bytes.
struct Entry {
    shard: usize,
    hf_name: String,
    /// `[rows, cols]` when the tensor is an FP8 block-scaled matrix.
    fp8: Option<Fp8>,
    /// Plain RMSNorm weight: add 1.0 on read.
    fold_one: bool,
}

struct Fp8 {
    scale_shard: usize,
    scale_name: String,
    rows: usize,
    cols: usize,
}

pub struct Qwen35Source {
    shards: Vec<MmapSafetensors>,
    entries: HashMap<String, Entry>,
    block: usize,
}

impl Qwen35Source {
    /// Open the checkpoint in `dir` for model `cfg`; `block` is the FP8
    /// `weight_block_size` (128 for the real release, ignored by tensors
    /// without a `weight_scale_inv` sibling).
    ///
    /// Reads only headers. Fails, naming the tensors, when the checkpoint does
    /// not cover `cfg.param_list()` (LoRA adapter tensors excepted: they are
    /// not in a base checkpoint), when a tensor's element count disagrees with
    /// the config, when a source tensor maps to no known name, or when an FP8
    /// weight's scale plane has the wrong shape. `mtp.*` and `model.visual.*`
    /// tensors are out of scope here exactly as in [`crate::import`].
    pub fn open_dir(dir: &Path, cfg: &Qwen35Config, block: usize) -> Result<Qwen35Source, String> {
        let files = shard_files(dir)?;
        let mut shards = Vec::with_capacity(files.len());
        for f in &files {
            shards.push(MmapSafetensors::open(dir.join(f)).map_err(|e| format!("qwen35 source: open {f}: {e}"))?);
        }

        let mut scales: HashMap<String, (usize, String)> = HashMap::new();
        let mut entries: HashMap<String, Entry> = HashMap::new();
        for (si, reader) in shards.iter().enumerate() {
            for name in reader.names() {
                if let Some(base) = name.strip_suffix("_scale_inv") {
                    // A scale plane belongs to its weight: out-of-scope weights
                    // (mtp.*, model.visual.*) take their scales with them.
                    if classify(base, cfg)?.is_some() {
                        scales.insert(base.to_string(), (si, name.clone()));
                    }
                    continue;
                }
                let Some(canon) = classify(name, cfg)? else { continue };
                let entry = Entry { shard: si, hf_name: name.clone(), fp8: None, fold_one: is_plain_rmsnorm_weight(&canon) };
                if entries.insert(canon.clone(), entry).is_some() {
                    return Err(format!("qwen35 source: two source tensors mapped onto {canon}"));
                }
            }
        }
        for entry in entries.values_mut() {
            let Some((scale_shard, scale_name)) = scales.remove(&entry.hf_name) else { continue };
            let shape = shards[entry.shard].shape(&entry.hf_name).unwrap_or(&[]).to_vec();
            let [rows, cols] = shape[..] else {
                return Err(format!("qwen35 source: {} has shape {shape:?}, FP8 block scaling needs 2 dims", entry.hf_name));
            };
            let want = model::fp8::scale_shape(rows, cols, block);
            let got = shards[scale_shard].shape(&scale_name).unwrap_or(&[]);
            if got != [want.0, want.1] {
                return Err(format!("qwen35 source: {scale_name} has shape {got:?}, expected {want:?} for {} {shape:?}", entry.hf_name));
            }
            entry.fp8 = Some(Fp8 { scale_shard, scale_name, rows, cols });
        }
        if let Some(orphan) = scales.keys().min() {
            return Err(format!("qwen35 source: {orphan}_scale_inv has no matching weight in scope"));
        }

        let src = Qwen35Source { shards, entries, block };
        src.check_coverage(cfg)?;
        Ok(src)
    }

    /// Two-way coverage against `cfg.param_list()` from shapes alone.
    fn check_coverage(&self, cfg: &Qwen35Config) -> Result<(), String> {
        let expected: Vec<(String, usize)> =
            cfg.param_list().into_iter().filter(|(n, _)| !model::adapter::device::is_adapter_param(n)).collect();
        let mut problems = Vec::new();
        for (name, numel) in &expected {
            match self.numel(name) {
                None => problems.push(format!("missing {name}")),
                Some(n) if n != *numel => problems.push(format!("{name} has {n} elements, config expects {numel}")),
                Some(_) => {}
            }
        }
        let want: std::collections::HashSet<&str> = expected.iter().map(|(n, _)| n.as_str()).collect();
        for name in self.entries.keys().filter(|n| !want.contains(n.as_str())) {
            problems.push(format!("{name} is in the checkpoint but not in the config's parameter list"));
        }
        if problems.is_empty() {
            return Ok(());
        }
        problems.sort();
        let total = problems.len();
        problems.truncate(16);
        Err(format!("qwen35 source: checkpoint does not match the config ({total} problems): {}", problems.join("; ")))
    }

    /// Canonical names this source serves, sorted.
    pub fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.entries.keys().map(String::as_str).collect();
        v.sort_unstable();
        v
    }

    /// Dequantize rows `[r0, r0 + nr)` of an FP8 matrix. `r0` is block aligned.
    fn dequant_band(&self, e: &Entry, fp8: &Fp8, scale: &[f32], r0: usize, nr: usize) -> Vec<f32> {
        let raw = self.shards[e.shard].tensor_f32_range(&e.hf_name, r0 * fp8.cols, nr * fp8.cols).expect("band lies inside the validated tensor");
        let (_, cb) = model::fp8::scale_shape(fp8.rows, fp8.cols, self.block);
        let first = (r0 / self.block) * cb;
        let blocks = nr.div_ceil(self.block) * cb;
        model::fp8::dequant_block128(&raw, &scale[first..first + blocks], nr, fp8.cols, self.block)
    }

    /// Row bands of at most `max_elems` elements (but at least one block row
    /// band), in order, handed to `f` with their element offset.
    fn for_each_fp8_band(&self, e: &Entry, fp8: &Fp8, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) {
        let scale = self.shards[fp8.scale_shard].tensor_f32(&fp8.scale_name).expect("scale validated at open");
        let rows_per = if max_elems == 0 { fp8.rows } else { ((max_elems / fp8.cols) / self.block).max(1) * self.block };
        let mut r0 = 0;
        while r0 < fp8.rows {
            let nr = rows_per.min(fp8.rows - r0);
            f((r0 * fp8.cols) as u64, &self.dequant_band(e, fp8, &scale, r0, nr));
            r0 += nr;
        }
        self.shards[e.shard].advise_dontneed_tensor(&e.hf_name);
    }
}

impl TensorSource for Qwen35Source {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        let Some(e) = self.entries.get(name) else { return false };
        if let Some(fp8) = &e.fp8 {
            let mut whole = Vec::with_capacity(fp8.rows * fp8.cols);
            self.for_each_fp8_band(e, fp8, WHOLE_TENSOR_BAND_ELEMS, &mut |_, band| whole.extend_from_slice(band));
            f(&whole);
            return true;
        }
        let Some(mut data) = self.shards[e.shard].tensor_f32(&e.hf_name) else { return false };
        if e.fold_one {
            data.iter_mut().for_each(|v| *v += 1.0);
        }
        f(&data);
        true
    }

    fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        let Some(e) = self.entries.get(name) else { return false };
        match &e.fp8 {
            Some(fp8) => {
                self.for_each_fp8_band(e, fp8, max_elems, f);
                true
            }
            // Unscaled and unfolded: the mmap's own bounded chunked decode.
            None if !e.fold_one => self.shards[e.shard].with_tensor_chunks(&e.hf_name, max_elems, f),
            // Norm weights are a few thousand elements: one chunk.
            None => self.with_tensor(name, &mut |d| f(0, d)),
        }
    }

    fn numel(&self, name: &str) -> Option<usize> {
        let e = self.entries.get(name)?;
        self.shards[e.shard].numel(&e.hf_name)
    }

    fn advise_drop(&self, name: &str) {
        if let Some(e) = self.entries.get(name) {
            self.shards[e.shard].advise_dontneed_tensor(&e.hf_name);
        }
    }
}

/// Shard files of an HF checkpoint dir, in `model.safetensors.index.json`
/// order when present, else the single `model.safetensors` (the same two
/// layouts `checkpoint::safetensors::read_model_dir` accepts).
fn shard_files(dir: &Path) -> Result<Vec<String>, String> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let v: Value = serde_json::from_slice(&std::fs::read(&index).map_err(|e| format!("qwen35 source: read {}: {e}", index.display()))?)
            .map_err(|e| format!("qwen35 source: {}: {e}", index.display()))?;
        let map = v["weight_map"].as_object().ok_or_else(|| format!("qwen35 source: {} has no weight_map", index.display()))?;
        let mut files: Vec<String> = map.values().filter_map(|s| s.as_str().map(String::from)).collect();
        files.sort();
        files.dedup();
        return Ok(files);
    }
    if dir.join("model.safetensors").exists() {
        return Ok(vec!["model.safetensors".to_string()]);
    }
    Err(format!("qwen35 source: {} has neither model.safetensors.index.json nor model.safetensors", dir.display()))
}

/// The [`Qwen35Config`] of the HF checkpoint in `dir`, read from its
/// `config.json` (the `text_config` object of a multimodal release, or the
/// top level of a text-only one). `block_size` is the model's full
/// `max_position_embeddings`; a trainer narrows it to its own sequence window.
pub fn hf_config(dir: &Path) -> Result<Qwen35Config, String> {
    let path = dir.join("config.json");
    let top: Value = serde_json::from_slice(&std::fs::read(&path).map_err(|e| format!("qwen35 config: read {}: {e}", path.display()))?)
        .map_err(|e| format!("qwen35 config: {}: {e}", path.display()))?;
    let c = if top["text_config"].is_object() { &top["text_config"] } else { &top };
    let rope = &c["rope_parameters"];
    let pick = |keys: &[&str]| keys.iter().find_map(|k| c.get(*k).filter(|v| !v.is_null()).or_else(|| rope.get(*k).filter(|v| !v.is_null()))).cloned();
    let mut j = serde_json::Map::new();
    for (to, from) in [
        ("vocab_size", &["vocab_size"][..]),
        ("n_layers", &["num_hidden_layers"]),
        ("d_model", &["hidden_size"]),
        ("n_heads", &["num_attention_heads"]),
        ("n_kv_heads", &["num_key_value_heads"]),
        ("head_dim", &["head_dim"]),
        ("rms_norm_eps", &["rms_norm_eps"]),
        ("rope_theta", &["rope_theta"]),
        ("max_position_embeddings", &["max_position_embeddings"]),
        ("block_size", &["max_position_embeddings"]),
        ("attention_bias", &["attention_bias"]),
        ("partial_rotary_factor", &["partial_rotary_factor"]),
        ("mrope_section", &["mrope_section"]),
        ("full_attention_interval", &["full_attention_interval"]),
        ("linear_num_key_heads", &["linear_num_key_heads"]),
        ("linear_num_value_heads", &["linear_num_value_heads"]),
        ("linear_key_head_dim", &["linear_key_head_dim"]),
        ("linear_value_head_dim", &["linear_value_head_dim"]),
        ("linear_conv_kernel_dim", &["linear_conv_kernel_dim"]),
        ("intermediate_size", &["intermediate_size"]),
        ("rope_scaling", &["rope_scaling"]),
    ] {
        if let Some(v) = pick(from) {
            j.insert(to.to_string(), v);
        }
    }
    let tie = top["tie_word_embeddings"].as_bool().or_else(|| c["tie_word_embeddings"].as_bool()).unwrap_or(false);
    j.insert("tie_word_embeddings".to_string(), Value::Bool(tie));
    Qwen35Config::from_json_checked(&Value::Object(j))
}


/// The FP8 `weight_block_size` the checkpoint in `dir` declares in its
/// `quantization_config` (128 for the real release); 128 when it declares
/// none, as a bf16/fp32 release does not.
pub fn hf_fp8_block_size(dir: &Path) -> Result<usize, String> {
    let path = dir.join("config.json");
    let top: Value = serde_json::from_slice(&std::fs::read(&path).map_err(|e| format!("qwen35 config: read {}: {e}", path.display()))?)
        .map_err(|e| format!("qwen35 config: {}: {e}", path.display()))?;
    let q = if top["quantization_config"].is_object() { &top["quantization_config"] } else { &top["text_config"]["quantization_config"] };
    match q["weight_block_size"].as_array().map(|a| a.iter().map(|v| v.as_u64()).collect::<Vec<_>>()) {
        None => Ok(128),
        Some(b) if b.len() == 2 && b[0] == b[1] && b[0].is_some_and(|n| n > 0) => Ok(b[0].unwrap_or(128) as usize),
        Some(b) => Err(format!("qwen35 config: weight_block_size {b:?} is not a square block")),
    }
}
