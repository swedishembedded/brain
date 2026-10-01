// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Serving what [`crate::train`] trained: a directory holding
//! `adapter.safetensors` (the decoder's LoRA) and, when the aligner trained,
//! `aligner.safetensors`, applied to a loaded composite. The adapter is
//! attached at run time ([`qwen3::Qwen::attach_adapter`]), never folded into
//! the base weights, and the aligner's parameters replace the checkpoint's.

use std::path::Path;

use crate::model::Vlm;
use crate::train::{ADAPTER_FILE, ALIGNER_FILE};

/// Apply the fine-tune in `dir` to `vlm`. A directory with neither file is an
/// error: serving "a fine-tune" that changes nothing would be a silent no-op.
pub fn apply(vlm: &mut Vlm, dir: &Path) -> Result<(), String> {
    let adapter = dir.join(ADAPTER_FILE);
    let aligner = dir.join(ALIGNER_FILE);
    if !adapter.is_file() && !aligner.is_file() {
        return Err(format!("{}: holds neither {ADAPTER_FILE} nor {ALIGNER_FILE}, so it is not a fine-tune of this model", dir.display()));
    }
    if aligner.is_file() {
        let path = aligner.to_str().ok_or("path is not UTF-8")?;
        let st = checkpoint::st::load_safetensors(path).map_err(|e| format!("{path}: {e}"))?;
        vlm.tower.set_aligner(&st.tensors).map_err(|e| format!("{path}: {e}"))?;
    }
    if adapter.is_file() {
        vlm.decoder.attach_adapter(adapter.to_str().ok_or("path is not UTF-8")?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_file_names_are_the_trainer_s() {
        assert_eq!((super::ADAPTER_FILE, super::ALIGNER_FILE), ("adapter.safetensors", "aligner.safetensors"));
    }
}
