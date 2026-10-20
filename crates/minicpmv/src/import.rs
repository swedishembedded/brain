// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The checkpoint is one safetensors file holding three components: the
//! vision tower (`model.vision_tower.*`), the merger (`model.merger.*`) and
//! the Qwen3.5 decoder (`model.language_model.*`). The decoder is served by
//! `brain-qwen35`'s own streaming source with the two vision prefixes
//! declared foreign; this crate accounts for those.

use std::path::Path;

use checkpoint::packed::PackedSource;
use qwen35::fp8_source::{self, hf_fp8_block_size};

use crate::config::MiniCpmVConfig;

/// Tensor-name prefixes this crate owns; the decoder's importer skips them.
pub const VISION_PREFIXES: &[&str] = &["model.vision_tower.", "model.merger."];

/// The decoder's tensors under `brain-qwen35`'s canonical names, streamed
/// from the checkpoint in `dir` with two-way coverage of its own scope.
pub fn open_decoder(dir: &Path, cfg: &MiniCpmVConfig) -> Result<PackedSource, String> {
    fp8_source::open_dir_with(dir, &cfg.text, hf_fp8_block_size(dir)?, VISION_PREFIXES)
}

/// The vision tower and merger weights under [`crate::vision::tensor_manifest`]'s
/// names, read from the checkpoint in `dir` with two-way coverage of
/// [`VISION_PREFIXES`].
pub fn vision_weights(dir: &Path, cfg: &MiniCpmVConfig) -> Result<std::collections::HashMap<String, Vec<f32>>, String> {
    let reader = checkpoint::weightio::WeightReader::open_hf_dir(dir).map_err(|e| format!("minicpmv: open {}: {e}", dir.display()))?;
    let names: Vec<String> = reader.names().map(str::to_string).collect();
    crate::vision::import(&names, &mut |n| reader.tensor(n), cfg)
}
