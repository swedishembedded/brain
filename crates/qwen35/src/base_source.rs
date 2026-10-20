// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where a LoRA/QLoRA fine-tune of Qwen3.5/3.8 gets its frozen base from, and
//! at which precision the base is held on the device.
//!
//! [`open_base`] turns a path into a configuration plus a lazy
//! [`checkpoint::TensorSource`]: a HuggingFace directory (the FP8 release, or
//! a bf16/fp32 one) is served by [`crate::fp8_source::open_dir`], a
//! brain-native `.safetensors` by the mmap-backed `WeightReader`. Neither
//! decodes more than one tensor (or one row band of the head) at a time, so
//! a 27B base never exists on the host as a whole. [`lora_base`] layers the
//! small, fresh LoRA adapters over that source, which is the shape the model
//! constructors take (adapters from memory, every frozen tensor streamed).
//!
//! A GGUF (any quantization the checkpoint codecs decode, Q8_0 to Q4_K) is served
//! lazily as well: llama.cpp's names are mapped onto the canonical ones by
//! `gguf_import::classify`, the Gated-DeltaNet leaves get their value fixes on
//! read, and every other tensor streams block by block, so a quantized GGUF
//! base needs no import step.
//!
//! Swedish Embedded AB implements solutions for fine-tuning very large
//! language models on a single accelerator for its clients. If your team
//! needs expertise in parameter-efficient or quantized fine-tuning then you
//! can procure our services by sending an email to info@swedishembedded.com.

use model::adapter::device::AdapterTuning;
use std::path::Path;
use std::str::FromStr;

use checkpoint::remap::{Fetch, RemapSource};
use checkpoint::weightio::WeightReader;
use checkpoint::TensorSource;

use crate::config::{lora_cfg, LoraCfg, Qwen35Config};
use crate::fp8_source::{self, hf_config, hf_fp8_block_size};

/// Precision of the frozen base weights on the device during a fine-tune.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseDtype {
    /// fp32: the legacy resident trainer. Needs ~108 GB for the 27B, so it
    /// only fits small or truncated models.
    F32,
    /// bf16 ("normal LoRA"): half the fp32 footprint, LoRA adapters in fp32.
    Bf16,
    /// 4-bit NormalFloat blocks of 32 with fp32 scales ("QLoRA").
    Nf4,
}

impl BaseDtype {
    pub fn name(self) -> &'static str {
        match self {
            BaseDtype::F32 => "f32",
            BaseDtype::Bf16 => "bf16",
            BaseDtype::Nf4 => "nf4",
        }
    }

    /// Whether the base is stored in a reduced-precision, frozen form: such a
    /// base can only be trained through LoRA adapters, never updated itself.
    pub fn is_frozen_only(self) -> bool {
        self != BaseDtype::F32
    }
}

impl FromStr for BaseDtype {
    type Err = String;
    fn from_str(s: &str) -> Result<BaseDtype, String> {
        match s {
            "f32" => Ok(BaseDtype::F32),
            "bf16" => Ok(BaseDtype::Bf16),
            "nf4" => Ok(BaseDtype::Nf4),
            other => Err(format!("--base-dtype {other:?}: expected f32, bf16 or nf4")),
        }
    }
}

/// A base model opened for streaming: its configuration and its tensors,
/// under brain-canonical names (`blocks.{l}.mlp.gate.weight`, `tok.weight`, ...).
pub struct BaseCheckpoint {
    pub cfg: Qwen35Config,
    pub src: Box<dyn TensorSource>,
}

/// Open the base model at `path`: a HuggingFace checkpoint directory
/// (`config.json` + safetensors shards, FP8 or not) or a brain-native
/// `.safetensors` file. Reads headers only.
pub fn open_base(path: &Path) -> Result<BaseCheckpoint, String> {
    if path.is_dir() {
        let cfg = hf_config(path)?;
        let src = fp8_source::open_dir(path, &cfg, hf_fp8_block_size(path)?)?;
        return Ok(BaseCheckpoint { cfg, src: Box::new(src) });
    }
    if path.extension().is_some_and(|e| e == "gguf") {
        return open_gguf_base(path);
    }
    let name = path.to_str().ok_or_else(|| format!("{}: not a UTF-8 path", path.display()))?;
    let reader = WeightReader::open(name).map_err(|e| format!("open {name}: {e}"))?;
    let cfg = Qwen35Config::from_json_checked(&reader.config()).map_err(|e| format!("{name}: {e}"))?;
    Ok(BaseCheckpoint { cfg, src: Box::new(reader) })
}

/// A GGUF as a streamed base: the config from its metadata, every parameter the
/// config lists mapped to the GGUF tensor that holds it (two-way coverage is
/// checked from shapes before any data is read), the llama.cpp-only value
/// fixes applied on read.
fn open_gguf_base(path: &Path) -> Result<BaseCheckpoint, String> {
    let name = path.to_str().ok_or_else(|| format!("{}: not a UTF-8 path", path.display()))?;
    let mg = checkpoint::gguf::MmapGguf::open(name).map_err(|e| format!("open {name}: {e}"))?;
    let mut cfg = crate::gguf_import::config_from_gguf(&mg).map_err(|e| format!("{name}: {e}"))?;
    // The MTP head is not part of a fine-tuned base.
    cfg.mtp = false;
    let available = crate::int8_gguf_resident::gguf_name_map(&mg, &cfg);
    let want: Vec<(String, usize)> = cfg.param_list().into_iter().filter(|(n, _)| !model::adapter::device::is_adapter_param(n)).collect();
    let (mut plan, mut missing) = (std::collections::HashMap::new(), Vec::new());
    for (param, _) in &want {
        match available.get(param) {
            Some(src) => {
                plan.insert(param.clone(), Fetch::Whole(src.clone()));
            }
            None => missing.push(param.as_str()),
        }
    }
    if !missing.is_empty() {
        missing.truncate(16);
        return Err(format!("{name}: no GGUF tensor for {}", missing.join(", ")));
    }
    let remap = RemapSource::owning(Box::new(mg), plan);
    remap.validate(&want).map_err(|e| format!("{name}: {e}"))?;
    let order = cfg.gdn_head_order();
    Ok(BaseCheckpoint { cfg, src: Box::new(gguf::GdnFixSource::new(remap, order)) })
}

/// Fresh zero-delta LoRA adapters held in memory over a streamed base.
pub struct AdapterOverlay {
    adapters: std::collections::HashMap<String, Vec<f32>>,
    base: Box<dyn TensorSource>,
}

impl AdapterOverlay {
    fn owner(&self, name: &str) -> &dyn TensorSource {
        if self.adapters.contains_key(name) {
            &self.adapters
        } else {
            &*self.base
        }
    }
}

impl TensorSource for AdapterOverlay {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        self.owner(name).with_tensor(name, f)
    }
    fn raw_words(&self, name: &str) -> Option<&[u32]> {
        self.owner(name).raw_words(name)
    }
    fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        self.owner(name).with_tensor_chunks(name, max_elems, f)
    }
    fn numel(&self, name: &str) -> Option<usize> {
        self.owner(name).numel(name)
    }
    fn advise_drop(&self, name: &str) {
        self.owner(name).advise_drop(name)
    }
    /// The base's packed view; adapter names are unknown to it, so they fall
    /// back to the f32 interface and reach `adapters`.
    fn packed(&self) -> Option<&checkpoint::packed::PackedSource> {
        self.base.packed()
    }
}

/// The configuration and initial weights of a LoRA fine-tune of the base at
/// `path`: its architecture with a `rank`/`alpha` adapter, the base tensors
/// streamed from disk.
///
/// A fresh run (`resume_from` `None`) starts as the base: `lora_a` drawn from
/// `seed`, `lora_b` zero, every targetable projection adapted. With
/// `resume_from` naming an adapter file written by `crate::lora::save_adapter`
/// the run continues ITS factors, unfolded, over the targets its card names. An
/// adapter trained at a different rank or alpha, or whose tensors do not fit
/// this base, is refused: its factors only mean what they mean at their own
/// shape and scale.
pub fn lora_base(path: &Path, rank: u32, alpha: f32, seed: u64, resume_from: Option<&Path>) -> Result<(Qwen35Config, AdapterOverlay), String> {
    lora_base_tuned(path, rank, alpha, AdapterTuning::default(), seed, resume_from)
}

/// [`lora_base`] with the run's hyper-parameters (rsLoRA, LoRA+, LoRA-FA). A
/// continued adapter must be continued with the tuning it trained with.
pub fn lora_base_tuned(path: &Path, rank: u32, alpha: f32, tuning: AdapterTuning, seed: u64, resume_from: Option<&Path>) -> Result<(Qwen35Config, AdapterOverlay), String> {
    lora_overlay_tuned(open_base(path)?, rank, alpha, tuning, seed, resume_from)
}

/// [`lora_base`] over a base already opened - a multimodal wrapper's decoder
/// source, say, whose checkpoint also holds tensors this crate does not own.
pub fn lora_overlay(base: BaseCheckpoint, rank: u32, alpha: f32, seed: u64, resume_from: Option<&Path>) -> Result<(Qwen35Config, AdapterOverlay), String> {
    lora_overlay_tuned(base, rank, alpha, AdapterTuning::default(), seed, resume_from)
}

/// [`lora_overlay`] with the run's hyper-parameters.
pub fn lora_overlay_tuned(base: BaseCheckpoint, rank: u32, alpha: f32, tuning: AdapterTuning, seed: u64, resume_from: Option<&Path>) -> Result<(Qwen35Config, AdapterOverlay), String> {
    tuning.validate()?;
    let BaseCheckpoint { mut cfg, src } = base;
    cfg.lora = Some(lora_cfg(rank, alpha).with_tuning(tuning));
    let card_alpha = cfg.lora.as_ref().expect("just set").card_alpha();
    let mut adapter = None;
    if let Some(file) = resume_from {
        let name = file.to_str().ok_or_else(|| format!("{}: not a UTF-8 path", file.display()))?;
        let st = checkpoint::st::load_safetensors(name).map_err(|e| format!("{name}: {e}"))?;
        let card = st.card().and_then(|c| c.adapter).ok_or_else(|| format!("{name}: not an adapter file (no adapter card)"))?;
        if card.kind != "lora" {
            return Err(format!("{name}: a {:?} adapter, only LoRA can be continued", card.kind));
        }
        if card.rank != Some(rank) || card.alpha.is_some_and(|a| !model::adapter::device::alpha_matches(a, card_alpha)) {
            return Err(format!("{name}: trained at rank {:?} alpha {:?}, asked to continue at rank {rank} alpha {card_alpha} ({tuning:?})", card.rank, card.alpha));
        }
        let targets = card.targets.ok_or_else(|| format!("{name}: the adapter card names no targets"))?;
        cfg.lora = Some(LoraCfg::new(rank, alpha, targets).with_tuning(tuning));
        adapter = Some(st.tensors);
    }
    let mut adapters = crate::init::init_lora_only(&cfg, seed);
    for (name, values) in adapter.into_iter().flatten() {
        match adapters.get(&name) {
            Some(slot) if slot.len() == values.len() => {
                adapters.insert(name, values);
            }
            Some(slot) => return Err(format!("adapter tensor {name} holds {} values, this base's has {}", values.len(), slot.len())),
            None => return Err(format!("adapter tensor {name} has no counterpart in this base's LoRA parameters")),
        }
    }
    Ok((cfg, AdapterOverlay { adapters, base: src }))
}
