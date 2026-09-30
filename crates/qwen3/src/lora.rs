// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Adapter-only save/load for a trained Qwen LoRA.
//!
//! `Qwen::save` (see `model.rs`) writes the WHOLE param store -- base weights
//! plus `.lora_a`/`.lora_b` -- as a full-size checkpoint. That is correct for
//! resuming training, but wasteful for distribution and serving: a rank-8
//! adapter on Qwen3-0.6B is a few MB against a ~2.4 GB fp32 base. This module
//! writes just the adapter tensors (with a `ModelCard` describing them, so
//! `model_dir::register` can catalog the adapter as its own selectable model
//! id) and can fold them into an already-loaded base's weights for
//! inference -- `W_eff = W + (alpha/rank)*B*A`, applied once at load, so the
//! forward pass pays zero extra cost versus the unadapted base.
//!
//! The actual save/fold I/O and `fold_delta` math live once, generically,
//! in `model::lora::device_adapter` (self-improve roadmap P4 -- this used to
//! be a near-verbatim copy shared with `qwen35moe::lora`/`deepseek2::lora`;
//! `qwen35moe`'s and `deepseek2`'s own doc comments called theirs "a direct
//! port" of this file). This module is now just the thin, qwen3-specific
//! wiring: this crate's own `LoraCfg` type and the `"qwen"` family tag.

use std::collections::HashMap;

use crate::config::{LoraCfg, QwenConfig};

/// Write only this model's `.lora_a`/`.lora_b` tensors -- never the frozen
/// base -- to `path`. See `model::lora::device_adapter::save_adapter`.
pub fn save_adapter<M: model::Model<Config = QwenConfig>>(path: &str, model: &M, card_id: &str, base_id: &str, dataset_id: Option<&str>) -> std::io::Result<()> {
    save_adapter_with_lineage(path, model, card_id, base_id, dataset_id, None)
}

/// [`save_adapter`], recording the training run that produced the adapter on
/// its card. See `model::lora::device_adapter::save_adapter_with_lineage`.
pub fn save_adapter_with_lineage<M: model::Model<Config = QwenConfig>>(
    path: &str,
    model: &M,
    card_id: &str,
    base_id: &str,
    dataset_id: Option<&str>,
    training: Option<checkpoint::st::TrainingProvenance>,
) -> std::io::Result<()> {
    let lora = model
        .config()
        .lora
        .as_ref()
        .unwrap_or_else(|| panic!("save_adapter: model was not built with a LoraCfg"));
    model::lora::device_adapter::save_adapter_with_lineage(path, model, lora.rank, lora.alpha, &lora.targets, card_id, base_id, "qwen", dataset_id, training)
}

/// An adapter file read and validated into host memory, with no base model
/// involved - what [`crate::Qwen::attach_adapter_pairs`] applies at runtime.
pub use model::lora::device_adapter::{read_adapter, AdapterSite, DeviceAdapter};

/// Fold an adapter saved by [`save_adapter`] into a base model's host tensor
/// map (name -> row-major `[out, in]` data), in place. `base` must already
/// contain every targeted linear's weight under its plain name (e.g.
/// `blocks.0.attn.wq.weight`). See `model::lora::device_adapter::fold_adapter_into`.
pub fn fold_adapter_into(base: &mut HashMap<String, Vec<f32>>, adapter_path: &str) -> std::io::Result<LoraCfg> {
    let (rank, alpha) = model::lora::device_adapter::fold_adapter_into(base, adapter_path)?;
    Ok(LoraCfg { rank, alpha, targets: vec![] })
}
