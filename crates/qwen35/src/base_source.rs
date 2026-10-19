// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where a LoRA/QLoRA fine-tune of Qwen3.5/3.8 gets its frozen base from, and
//! at which precision the base is held on the device.
//!
//! [`open_base`] turns a path into a configuration plus a lazy
//! [`checkpoint::TensorSource`]: a HuggingFace directory (the FP8 release, or
//! a bf16/fp32 one) is served by [`crate::fp8_source::Qwen35Source`], a
//! brain-native `.safetensors` by the mmap-backed `WeightReader`. Neither
//! decodes more than one tensor (or one row band of the head) at a time, so
//! a 27B base never exists on the host as a whole. [`lora_base`] layers the
//! small, fresh LoRA adapters over that source, which is the shape the model
//! constructors take (adapters from memory, every frozen tensor streamed).
//!
//! GGUF is not opened here: convert it once with `brain import-gguf`, which
//! produces the brain-native file this accepts.
//!
//! Swedish Embedded AB implements solutions for fine-tuning very large
//! language models on a single accelerator for its clients. If your team
//! needs expertise in parameter-efficient or quantized fine-tuning then you
//! can procure our services by sending an email to info@swedishembedded.com.

use std::path::Path;
use std::str::FromStr;

use checkpoint::weightio::WeightReader;
use checkpoint::TensorSource;

use crate::config::{lora_cfg, LoraCfg, Qwen35Config};
use crate::fp8_source::{hf_config, hf_fp8_block_size, Qwen35Source};

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
        let src = Qwen35Source::open_dir(path, &cfg, hf_fp8_block_size(path)?)?;
        return Ok(BaseCheckpoint { cfg, src: Box::new(src) });
    }
    if path.extension().is_some_and(|e| e == "gguf") {
        return Err(format!("{}: a GGUF file; convert it with `brain import-gguf` and name the .safetensors it writes", path.display()));
    }
    let name = path.to_str().ok_or_else(|| format!("{}: not a UTF-8 path", path.display()))?;
    let reader = WeightReader::open(name).map_err(|e| format!("open {name}: {e}"))?;
    let cfg = Qwen35Config::from_json_checked(&reader.config()).map_err(|e| format!("{name}: {e}"))?;
    Ok(BaseCheckpoint { cfg, src: Box::new(reader) })
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
    lora_overlay(open_base(path)?, rank, alpha, seed, resume_from)
}

/// [`lora_base`] over a base already opened - a multimodal wrapper's decoder
/// source, say, whose checkpoint also holds tensors this crate does not own.
pub fn lora_overlay(base: BaseCheckpoint, rank: u32, alpha: f32, seed: u64, resume_from: Option<&Path>) -> Result<(Qwen35Config, AdapterOverlay), String> {
    let BaseCheckpoint { mut cfg, src } = base;
    cfg.lora = Some(lora_cfg(rank, alpha));
    let mut adapter = None;
    if let Some(file) = resume_from {
        let name = file.to_str().ok_or_else(|| format!("{}: not a UTF-8 path", file.display()))?;
        let st = checkpoint::st::load_safetensors(name).map_err(|e| format!("{name}: {e}"))?;
        let card = st.card().and_then(|c| c.adapter).ok_or_else(|| format!("{name}: not an adapter file (no adapter card)"))?;
        if card.kind != "lora" {
            return Err(format!("{name}: a {:?} adapter, only LoRA can be continued", card.kind));
        }
        if card.rank != Some(rank) || card.alpha.is_some_and(|a| a != alpha) {
            return Err(format!("{name}: trained at rank {:?} alpha {:?}, asked to continue at rank {rank} alpha {alpha}", card.rank, card.alpha));
        }
        let targets = card.targets.ok_or_else(|| format!("{name}: the adapter card names no targets"))?;
        cfg.lora = Some(LoraCfg { rank, alpha, targets });
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
