// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Load a Nemotron 3.5 ASR checkpoint's tensors into a name→f32 map. HF names are
//! used verbatim as keys (`encoder.*`, `decoder.*`, `joint.*`, `prompt_projector.*`,
//! `encoder_projector.*`); the encoder/decoder builders pull what they need.

use std::collections::HashMap;
use std::path::Path;

pub fn load_tensors(dir: &Path) -> Result<HashMap<String, Vec<f32>>, String> {
    let tensors = checkpoint::safetensors::read_model_dir(dir)?;
    Ok(tensors.into_iter().map(|t| (t.name, t.data)).collect())
}
