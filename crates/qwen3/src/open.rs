// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Open a checkpoint file for building: its config and its tensors under
//! brain's parameter names, whatever the file's format.

use crate::config::QwenConfig;
use checkpoint::TensorSource;

/// The config and a tensor source of the checkpoint at `path`.
///
/// A GGUF is read under brain's names straight off its mapping
/// ([`crate::gguf_import::open_source`]), with the shape its KV metadata
/// declares. Any other file is a brain checkpoint, opened with
/// [`checkpoint::weightio::WeightReader`] and configured by
/// [`QwenConfig::from_reader`], which refuses a header missing a shape key.
pub fn open_checkpoint(path: &str) -> Result<(QwenConfig, Box<dyn TensorSource>), String> {
    let reader = checkpoint::weightio::WeightReader::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
    if reader.gguf().is_some() {
        drop(reader);
        let (cfg, src) = crate::gguf_import::open_source(path)?;
        return Ok((cfg, Box::new(src)));
    }
    let cfg = QwenConfig::from_reader(&reader).map_err(|e| format!("{path}: {e}"))?;
    Ok((cfg, Box::new(reader)))
}
