// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import the SAM tower out of a DeepSeek-OCR **mmproj** GGUF.
//!
//! There is deliberately no tensor-name table in this file. `brain-gguf`'s
//! `deepseek_ocr_vision` module already derives the config from the real
//! header's KV + tensor shapes and already classifies every one of the file's
//! 476 tensors onto brain-side names; re-deriving either here would be a second
//! source of truth for the same mapping. This module is the **filter**: take
//! that classifier, keep the `vision.sam.*` half, and record every other tensor
//! as a deliberate `Mapped::Dropped` so the two-way coverage check still runs
//! over the whole file.
//!
//! Dropping is what makes this cheap as well as honest -- `import::to_map` never
//! reaches `mg.tensor()` for a dropped name, so the CLIP tower and the
//! projector are not dequantized at all.

use std::collections::HashMap;

use checkpoint::gguf::MmapGguf;
use gguf::deepseek_ocr_vision as dsv;
use gguf::Mapped;

use crate::config::SamViTConfig;

/// The brain-side prefix every tensor of this tower carries.
pub const PREFIX: &str = "vision.sam.";

/// Derive the tower's config from an open mmproj GGUF.
pub fn config_from_gguf(mg: &MmapGguf) -> Result<SamViTConfig, String> {
    Ok(SamViTConfig::from(&dsv::config_from_gguf(mg)?.sam))
}

/// `deepseek_ocr_vision::classify`, narrowed to this tower.
///
/// Returned as a closure factory rather than inlined so the coverage *dry run*
/// and the real load provably classify identically.
fn sam_only(full: &dsv::DeepseekOcrVisionConfig) -> impl Fn(&str) -> Result<Mapped, String> + '_ {
    move |name: &str| match dsv::classify(name, full)? {
        Mapped::Simple(n) if n.starts_with(PREFIX) => Ok(Mapped::Simple(n)),
        _ => Ok(Mapped::Dropped("not part of the SAM ViT tower")),
    }
}

/// Header-only coverage check: every source tensor is classified and every
/// declared parameter is produced, without reading a byte of tensor data.
pub fn dry_run(mg: &MmapGguf) -> Result<(SamViTConfig, gguf::ImportStats), String> {
    let full = dsv::config_from_gguf(mg)?;
    let cfg = SamViTConfig::from(&full.sam);
    let stats = gguf::import::dry_run(mg, &cfg.param_list(), &sam_only(&full), "sam1")?;
    Ok((cfg, stats))
}

/// Load the tower's weights into an init map ready for
/// [`crate::model::SamEncoder::new`].
pub fn weights_from_gguf(mg: &MmapGguf) -> Result<(SamViTConfig, HashMap<String, Vec<f32>>), String> {
    let full = dsv::config_from_gguf(mg)?;
    let cfg = SamViTConfig::from(&full.sam);
    let w = gguf::import::to_map(mg, &cfg.param_list(), &sam_only(&full), "sam1")?;
    Ok((cfg, w))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Skip-if-absent gate against the REAL shipped mmproj. It proves three
    /// things no synthetic fixture can: that the derived config is the one this
    /// crate's `deepseek_ocr()` preset claims, that the SAM filter classifies
    /// every one of the file's tensors (kept or dropped), and that every
    /// parameter of the manifest is produced exactly once.
    #[test]
    fn real_mmproj_covers_the_whole_sam_manifest() {
        // Resolve through the model store, NOT by hand from $HOME. This test
        // used to build `$HOME/.local/share/brain/models/<repo>/<file>`
        // literally, which is the store's default layout but not its only one:
        // a box that sets BRAIN_MODELS_DIR keeps its checkpoints somewhere
        // else, so the path missed, the test skipped, and cargo reported a
        // pass - a misconfigured run indistinguishable from an absent fixture,
        // on a box that had the file all along.
        const REPO: &str = "ggml-org/DeepSeek-OCR-GGUF";
        let Some(dir) = brain_testutil::model_dir(REPO) else {
            return brain_testutil::skip(&format!("no model store to resolve {REPO}"));
        };
        let path = std::path::Path::new(&dir).join("mmproj-DeepSeek-OCR-Q8_0.gguf");
        if !path.exists() {
            brain_testutil::skip(&format!("{} not present (brain fetch {REPO})", path.display()));
            return;
        }
        let mg = MmapGguf::open(path.to_str().expect("utf-8 path")).expect("open mmproj");
        let (cfg, stats) = dry_run(&mg).expect("dry run");
        assert_eq!(cfg, SamViTConfig::deepseek_ocr(), "the shipped file disagrees with the documented preset");
        cfg.check_bindable();
        assert_eq!(stats.written, cfg.param_list().len(), "{stats}");
        assert!(
            stats.dropped.values().sum::<usize>() > 0,
            "the CLIP half must be dropped on the record, not silently missing: {stats}"
        );
        println!("sam1 mmproj coverage: {stats}");
    }
}
