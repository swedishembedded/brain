// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Open a checkpoint for building: its config and its tensors under brain's
//! parameter names, whatever its format.

use crate::config::QwenConfig;
use checkpoint::TensorSource;

/// The config and a tensor source of the checkpoint at `path`, read as it
/// is on disk: nothing is converted ahead of time or written back.
///
/// - A Hugging Face checkpoint directory (`config.json` with safetensors or
///   `pytorch_model*.bin` weights) is configured by [`crate::hf::decoder_config`]
///   and read under brain's names through [`crate::import::owned_source`],
///   each tensor decoded to f32 as it is uploaded.
/// - A GGUF is read under brain's names straight off its mapping
///   ([`crate::gguf_import::open_source`]), with the shape its KV metadata
///   declares.
/// - Any other file is a brain checkpoint, opened with
///   [`checkpoint::weightio::WeightReader`] and configured by
///   [`QwenConfig::from_reader`], which refuses a header missing a shape key.
pub fn open_checkpoint(path: &str) -> Result<(QwenConfig, Box<dyn TensorSource>), String> {
    let dir = std::path::Path::new(path);
    if dir.is_dir() {
        let cfg = hf_dir_config(dir)?;
        let reader = checkpoint::weightio::WeightReader::open_hf_dir(dir).map_err(|e| format!("cannot open {path}: {e}"))?;
        let src = crate::import::owned_source(reader, &cfg).map_err(|e| format!("{path}: {e}"))?;
        return Ok((cfg, Box::new(src)));
    }
    let reader = checkpoint::weightio::WeightReader::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
    if reader.gguf().is_some() {
        drop(reader);
        let (cfg, src) = crate::gguf_import::open_source(path)?;
        return Ok((cfg, Box::new(src)));
    }
    let cfg = QwenConfig::from_reader(&reader).map_err(|e| format!("{path}: {e}"))?;
    Ok((cfg, Box::new(reader)))
}

/// The config of the checkpoint at `path`, as [`open_checkpoint`] resolves it,
/// without opening a tensor source.
pub fn checkpoint_config(path: &str) -> Result<QwenConfig, String> {
    let dir = std::path::Path::new(path);
    if dir.is_dir() {
        return hf_dir_config(dir);
    }
    let reader = checkpoint::weightio::WeightReader::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
    QwenConfig::from_reader(&reader).map_err(|e| format!("{path}: {e}"))
}

fn hf_dir_config(dir: &std::path::Path) -> Result<QwenConfig, String> {
    let json = std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("{}: {e}", dir.join("config.json").display()))?;
    crate::hf::decoder_config(&json).map_err(|e| format!("{}: {e}", dir.display()))
}

/// The path [`open_checkpoint`] should open for a base the model store
/// resolved to `weights` inside `dir`: a `transformers` directory is answered
/// by its `config.json` and is opened as the directory, anything else as the
/// file.
pub fn store_checkpoint_path(weights: &std::path::Path, dir: &std::path::Path) -> std::path::PathBuf {
    if weights.file_name().is_some_and(|n| n == "config.json") {
        dir.to_path_buf()
    } else {
        weights.to_path_buf()
    }
}
