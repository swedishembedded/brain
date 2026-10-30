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
//! ask for (`blocks.{l}.mlp.gate.weight`, `tok.weight`, ...): names go through
//! the same `import::classify`, plain RMSNorm weights get the same `(1+w)`
//! fold, and FP8 blocks are decoded by `checkpoint::packed`'s tile codec, so
//! every tensor is bit-identical to what the eager importers produce (the unit
//! tests assert exactly that). Host peak while a tensor is consumed is that one
//! tensor (`with_tensor`) or `O(chunk)` (`with_tensor_chunks`, which decodes
//! whole-row tiles so a 5 GB embedding or head never has to exist on the host
//! at once).
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

use std::path::Path;

use checkpoint::packed::{Binding, CodecRegistry, PackedReader, PackedSource, PackedTensorReader};
use checkpoint::TensorSource;
use serde_json::Value;

use model::ops::TierPolicy;

use crate::config::Qwen35Config;
use crate::import::{classify, is_plain_rmsnorm_weight};

/// Open the Qwen3.5/3.8 HuggingFace checkpoint in `dir` for model `cfg` as a
/// [`checkpoint::TensorSource`] under brain-canonical names; `block` is the FP8
/// `weight_block_size` (128 for the real release, ignored by tensors without a
/// `weight_scale_inv` sibling).
///
/// This module is only the declarative part: which stored tensor is which
/// canonical parameter ([`classify`]), which norms are stored as `w` and applied
/// as `1 + w`, and the two-way coverage check against the model config.
/// Discovering FP8 scale grids, decoding, tiling and releasing file pages are
/// `checkpoint::packed`'s.
///
/// Reads only headers. Fails, naming the tensors, when the checkpoint does not
/// cover `cfg.param_list()` (LoRA adapter tensors excepted: they are not in a
/// base checkpoint), when a tensor's element count disagrees with the config,
/// when a source tensor maps to no known name, or when an FP8 weight's scale
/// plane has the wrong shape. `mtp.*` and `model.visual.*` tensors are out of
/// scope here exactly as in [`crate::import`].
pub fn open_dir(dir: &Path, cfg: &Qwen35Config, block: usize) -> Result<PackedSource, String> {
    open_dir_with(dir, cfg, block, &[])
}

/// [`open_dir`] over a checkpoint that also holds another component's tensors:
/// names under a `foreign` prefix are skipped (their owner validates them),
/// every other name is held to the same two-way coverage. A multimodal wrapper
/// around this decoder passes its vision tower and projector prefixes here.
pub fn open_dir_with(dir: &Path, cfg: &Qwen35Config, block: usize, foreign: &[&str]) -> Result<PackedSource, String> {
    let reader = PackedReader::open(dir).map_err(|e| format!("qwen35 source: {e}"))?.with_fp8_block(block as u32);
    let bindings = bindings_for(&reader, cfg, foreign)?;
    let src = PackedSource::new(Box::new(reader), CodecRegistry::reference(), bindings).map_err(|e| format!("qwen35 source: {e}"))?;
    check_coverage(&src, cfg)?;
    Ok(src)
}

/// The canonical-name bindings of `reader`'s tensors: which stored tensor is which parameter, and which plain
/// RMSNorm weights are stored as `w` and applied as `1 + w`.
fn bindings_for(reader: &PackedReader, cfg: &Qwen35Config, foreign: &[&str]) -> Result<Vec<(String, Binding)>, String> {
    let mut bindings: Vec<(String, Binding)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for name in reader.names() {
        // A scale plane belongs to its weight (the reader hides it); one whose
        // weight is missing reaches classify and is refused by name.
        let Some(canon) = classify(&name, cfg, foreign)? else { continue };
        if !seen.insert(canon.clone()) {
            return Err(format!("qwen35 source: two source tensors mapped onto {canon}"));
        }
        let add = if is_plain_rmsnorm_weight(&canon) { 1.0 } else { 0.0 };
        bindings.push((canon, Binding { source: name, add }));
    }
    Ok(bindings)
}

/// The checkpoint's multi-token-prediction head as a [`PackedSource`] under the names of a one-layer stage:
/// its decoder layer is the full-attention layer `layer` of a config truncated to one attention period
/// (`blocks.{layer}.*`, the names the layer builders ask for), and its three own norms and the fused
/// `mtp.fc.weight` (`[d, 2d]`, embedding half first) keep their `mtp.*` names. Plain RMSNorm gains carry the same
/// `(1+w)` fold as the main stack. `Ok(None)` when the checkpoint holds no `mtp.*` tensor.
///
/// `cfg` is the full model's config; fails, naming the tensors, when the head's tensors do not cover the layer
/// and its endpoints exactly or disagree with the config on a size.
pub fn open_mtp(dir: &Path, cfg: &Qwen35Config, block: usize) -> Result<Option<PackedSource>, String> {
    let reader = PackedReader::open(dir).map_err(|e| format!("qwen35 mtp source: {e}"))?.with_fp8_block(block as u32);
    let layer = (cfg.full_attention_interval - 1) as usize;
    let mut bindings: Vec<(String, Binding)> = Vec::new();
    for name in reader.names().into_iter().filter(|n| n.starts_with(MTP_TENSOR_PREFIX)) {
        let canon = match name.strip_prefix("mtp.layers.0.") {
            Some(leaf) => {
                let suffix = crate::import::classify_layer_leaf(leaf, crate::config::LayerType::Full).ok_or_else(|| format!("qwen35 mtp source: unrecognized leaf '{leaf}' in the MTP layer"))?;
                format!("blocks.{layer}.{suffix}")
            }
            None => name.clone(),
        };
        let add = if is_plain_rmsnorm_weight(&canon) { 1.0 } else { 0.0 };
        bindings.push((canon, Binding { source: name, add }));
    }
    if bindings.is_empty() {
        return Ok(None);
    }
    let src = PackedSource::new(Box::new(reader), CodecRegistry::reference(), bindings).map_err(|e| format!("qwen35 mtp source: {e}"))?;
    let d = cfg.d_model as usize;
    let mut expected: Vec<(String, usize)> = vec![("mtp.fc.weight".into(), 2 * d * d)];
    expected.extend(["mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"].map(|n| (n.to_string(), d)));
    let one_layer = Qwen35Config { n_layers: cfg.full_attention_interval, mtp: false, ..cfg.clone() };
    expected.extend(one_layer.param_list().into_iter().filter(|(n, _)| n.starts_with(&format!("blocks.{layer}."))));
    let mut problems: Vec<String> = Vec::new();
    for (name, numel) in &expected {
        match src.numel(name) {
            None => problems.push(format!("missing {name}")),
            Some(n) if n != *numel => problems.push(format!("{name} has {n} elements, expected {numel}")),
            Some(_) => {}
        }
    }
    let want: std::collections::HashSet<&str> = expected.iter().map(|(n, _)| n.as_str()).collect();
    problems.extend(src.names().into_iter().filter(|n| !want.contains(&**n)).map(|n| format!("{n} is not a tensor of the MTP head")));
    if !problems.is_empty() {
        problems.sort();
        problems.truncate(16);
        return Err(format!("qwen35 mtp source: the MTP tensors do not match the config: {}", problems.join("; ")));
    }
    Ok(Some(src))
}

/// Tensor-name prefix of the multi-token-prediction head in an HF checkpoint.
const MTP_TENSOR_PREFIX: &str = "mtp.";

/// Host staging a [`PackedBase::weight`] load may hold at once (one tile of the destination encoding).
pub const WEIGHT_STAGING_BYTES: u64 = 64 << 20;

/// A Qwen3.5/3.8 checkpoint directory served in two ways under brain-canonical names: small tensors (norm
/// gains, conv and gate vectors) decoded to f32 on demand, and weight matrices PRESERVED in their stored
/// encoding straight into a resident `Weight` tier through the shared packed-upload path
/// (`model::ops::upload_preserved`: the checkpoint crate's tiled loader into `DeviceWeightSink`). A layer
/// of an FP8 checkpoint therefore reaches the device as FP8 bytes plus block scales, one tile of host
/// staging at a time, and is never expanded to fp32 on the host or the device.
pub struct PackedBase {
    src: PackedSource,
    staging: checkpoint::packed::BoundedWorkspace,
    convert: Option<gpu_core::select::Dtype>,
}

impl PackedBase {
    /// Open `dir` for model `cfg` (`block` as in [`open_dir`]); the same two-way coverage check as
    /// [`open_dir`] applies.
    pub fn open(dir: &Path, cfg: &Qwen35Config, block: usize) -> Result<PackedBase, String> {
        let staging = checkpoint::packed::BoundedWorkspace::new(checkpoint::packed::MemoryBudget::new(WEIGHT_STAGING_BYTES, 0));
        Ok(PackedBase { src: open_dir(dir, cfg, block)?, staging, convert: None })
    }

    /// Draw this base's host staging from `budget` instead of a private one, so one owner (a trainer) accounts the
    /// staging of every load it makes, pinned upload ring included.
    pub fn with_budget(mut self, budget: checkpoint::packed::MemoryBudget) -> PackedBase {
        self.staging = checkpoint::packed::BoundedWorkspace::new(budget);
        self
    }

    /// Convert every weight matrix to `tier` on the way to the device (an explicit, bounded, tiled requantization, e.g.
    /// FP8 to NF4) instead of preserving the stored encoding.
    pub fn converting_to(mut self, tier: gpu_core::select::Dtype) -> PackedBase {
        self.convert = Some(tier);
        self
    }

    /// `name` decoded to f32, in bounded tiles (a small tensor: a norm gain or a conv/gate vector; the `(1+w)`
    /// norm fold applied). Weight matrices go through [`Self::weight`], never this.
    pub fn small_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(self.src.numel(name).ok_or_else(|| format!("qwen35 source: no tensor {name}"))?);
        self.src.with_tensor_chunks(name, 1 << 20, &mut |_, chunk| out.extend_from_slice(chunk));
        Ok(out)
    }

    /// The `[n, k]` matrix `name` as a resident weight in the tier of its stored encoding (Preserve: no numeric
    /// change). An encoding with no tier is an error, never a widening.
    pub fn weight(&self, ops: &model::ops::Ops, name: &str, n: usize, k: usize) -> Result<model::ops::Weight, String> {
        match self.convert {
            None => model::ops::upload_preserved(ops, &self.src, name, n, k, &self.staging).map_err(|e| format!("qwen35 source: {e}")),
            Some(tier) => model::ops::upload_from_source(ops, &self.src, name, n, k, tier, &self.staging).ok_or_else(|| format!("qwen35 source: no tensor {name}")),
        }
    }
}

/// Convert the checkpoint in `dir` to the per-tensor formats of `tiers` ONCE, into a new checkpoint directory
/// `out`: a single packed `model.safetensors` (every bound tensor under its stored name, converted tensors as
/// codes plus scales, the rest as stored) and the config and tokenizer files copied beside it, so that
/// [`PackedBase::open`] on `out` serves each tensor in its target format and a streamed trainer never converts
/// on the host again. The conversion is the checkpoint crate's bounded tiled `Convert`: one tile of host staging
/// ([`WEIGHT_STAGING_BYTES`]) at a time, the output never in memory.
///
/// A layer matrix takes `tiers.want(name)` (the policy's default applies); any other tensor (a norm gain, the
/// embedding table, the head) is converted only when a rule of `tiers` names it, and is otherwise copied as
/// stored. A tensor that cannot take its tier (a matrix whose width is not a whole block, a tier with no
/// encoder) stays as stored; the report counts what changed.
pub fn write_converted_base(dir: &Path, cfg: &Qwen35Config, block: usize, out: &Path, tiers: &TierPolicy) -> Result<checkpoint::packed::ConversionReport, String> {
    use checkpoint::packed::{BoundedWorkspace, LoadConfig, LoadPolicy, Loader, MemoryBudget};
    if out.join("model.safetensors").exists() {
        return Err(format!("qwen35 conversion: {} already holds a converted base", out.display()));
    }
    let reader = PackedReader::open(dir).map_err(|e| format!("qwen35 source: {e}"))?.with_fp8_block(block as u32);
    let bindings = bindings_for(&reader, cfg, &[])?;
    let matrices: std::collections::HashSet<String> = cfg.layer_types().iter().enumerate().flat_map(|(l, ty)| cfg.layer_leaves(*ty).into_iter().map(move |(suffix, _, _)| format!("blocks.{l}.{suffix}"))).collect();
    let mut requests: Vec<(String, LoadPolicy)> = Vec::with_capacity(bindings.len());
    for (canon, binding) in &bindings {
        let desc = reader.descriptor(&binding.source)?;
        let tier = if matrices.contains(canon) { Some(tiers.want(canon)) } else { tiers.rule_for(canon) };
        requests.push((binding.source.clone(), model::ops::conversion_policy(&desc, tier)));
    }
    let codecs = CodecRegistry::reference();
    let loader = Loader { reader: &reader, codecs: &codecs, workspace: BoundedWorkspace::new(MemoryBudget::new(WEIGHT_STAGING_BYTES, 0)), config: LoadConfig::default() };
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    for aux in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"] {
        if dir.join(aux).exists() {
            std::fs::copy(dir.join(aux), out.join(aux)).map_err(|e| format!("copy {aux}: {e}"))?;
        }
    }
    checkpoint::packed::convert_to_file(&loader, &requests, &out.join("model.safetensors"), &serde_json::json!({}))
}

/// Two-way coverage against `cfg.param_list()` from shapes alone.
fn check_coverage(src: &PackedSource, cfg: &Qwen35Config) -> Result<(), String> {
    let expected: Vec<(String, usize)> = cfg.param_list().into_iter().filter(|(n, _)| !model::adapter::device::is_adapter_param(n)).collect();
    let mut problems = Vec::new();
    for (name, numel) in &expected {
        match src.numel(name) {
            None => problems.push(format!("missing {name}")),
            Some(n) if n != *numel => problems.push(format!("{name} has {n} elements, config expects {numel}")),
            Some(_) => {}
        }
    }
    let want: std::collections::HashSet<&str> = expected.iter().map(|(n, _)| n.as_str()).collect();
    for name in src.names().into_iter().filter(|n| !want.contains(n)) {
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

/// The [`Qwen35Config`] of the HF checkpoint in `dir`, read from its
/// `config.json` (the `text_config` object of a multimodal release, or the
/// top level of a text-only one; EdgeRock's `fixed_memory_*` and `independent_memory` fields included).
/// `block_size` is the model's full `max_position_embeddings`; a trainer narrows it to its own sequence window.
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
    // EdgeRock's window and anchor memory; a plain Qwen3.5 config names none and passes through.
    Qwen35Config::from_json_checked(&Value::Object(j))?.with_edgerock_hf(c)
}


/// The FP8 `weight_block_size` the checkpoint in `dir` declares in its quantization config (128 when it declares
/// none, as a bf16/fp32 release does not). The declaration is parsed once, by `checkpoint::packed::HfQuant`.
pub fn hf_fp8_block_size(dir: &Path) -> Result<usize, String> {
    Ok(checkpoint::packed::HfQuant::from_dir(dir)?.and_then(|q| q.fp8_block()).unwrap_or(128) as usize)
}
