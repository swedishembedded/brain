// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-checkpoint import - thin on purpose.
//!
//! `crates/gguf`'s `deepseek_ocr` module already owns every decision this would
//! otherwise duplicate: reading the hyperparameters off the GGUF header
//! (including the two that are NOT plain KV reads - `head_dim` derived from
//! `blk.0.attn_q.weight`'s own shape, and `rotary_dim` resolving the file's
//! `rope.dimension_count = 0` to the full head_dim), classifying all ~400
//! tensor names into brain's layout, fanning the stacked `*_exps` tensors out
//! per expert, keeping the shared experts fused, and proving two-way coverage
//! (nothing planned missing, nothing in the file unaccounted for).
//!
//! Because [`crate::config::DeepseekV2Config`] *wraps* that loader's own config
//! struct and delegates `param_list()` to it, this module is only the two lines
//! that consume the loader's output - there is no name translation step and no
//! second manifest that could drift.
//!
//! Two entry points, mirroring `gguf::import`'s own pair:
//! [`import_map`] materialises the weights in host memory (what
//! [`crate::DeepseekV2::new`] takes), [`import_file`] streams them straight to a
//! brain-native `.safetensors` one tensor at a time (what a real 6.7 B-parameter
//! checkpoint wants, since its fp32 expansion is far larger than the Q8_0 file).

use std::collections::HashMap;

use checkpoint::gguf::MmapGguf;
use gguf::deepseek_ocr::{classify, config_from_gguf};
use gguf::ImportStats;

use crate::config::DeepseekV2Config;

const LABEL: &str = "deepseek-ocr";

/// Read a DeepSeek-OCR language-model GGUF's header and return the decoder
/// config it describes, at the given training/eval sequence length.
pub fn config_from_file(path: &str, block_size: u32) -> Result<DeepseekV2Config, String> {
    let mg = MmapGguf::open(path)?;
    Ok(DeepseekV2Config::from_shape(config_from_gguf(&mg)?, block_size))
}

/// Import into an in-memory weight map, ready for [`crate::DeepseekV2::new`].
pub fn import_map(path: &str, block_size: u32) -> Result<(DeepseekV2Config, HashMap<String, Vec<f32>>), String> {
    let mg = MmapGguf::open(path)?;
    let shape = config_from_gguf(&mg)?;
    let params = shape.param_list();
    let weights = gguf::import::to_map(&mg, &params, &|n| classify(n, &shape), LABEL)?;
    Ok((DeepseekV2Config::from_shape(shape, block_size), weights))
}

/// Import to a brain-native `.safetensors` checkpoint, streaming one tensor at
/// a time (peak host memory ≈ one dequantized tensor, never the whole model).
pub fn import_file(path: &str, out_path: &str, id_override: Option<&str>) -> Result<ImportStats, String> {
    let mg = MmapGguf::open(path)?;
    gguf::deepseek_ocr::import(&mg, out_path, id_override)
}

/// Header-only dry run: the identical classification and two-way coverage check
/// with no tensor bytes read and nothing written. Milliseconds regardless of
/// checkpoint size, which is what makes the mapping testable on a machine that
/// cannot hold the fp32 expansion.
pub fn dry_run(path: &str) -> Result<ImportStats, String> {
    let mg = MmapGguf::open(path)?;
    let shape = config_from_gguf(&mg)?;
    let params = shape.param_list();
    gguf::import::dry_run(&mg, &params, &|n| classify(n, &shape), LABEL)
}
