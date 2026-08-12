// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which GGUF is this, and who imports it.
//!
//! `general.architecture` is the primary key, but it is **not** sufficient on
//! its own: every vision/projector file llama.cpp's mtmd tooling produces
//! declares `general.architecture = "clip"` and is distinguished only by a
//! second key, `clip.projector_type` (`"deepseekocr"`, `"qwen2vl"`,
//! `"minicpmv"`, …). So an entry keys on an architecture plus an *optional*
//! secondary `(key, value)` discriminator, and lookup prefers the more
//! specific match.
//!
//! Shaped as a flat table of plain data + one function pointer, the same way
//! the served-model catalog in `crates/cli` is: a closed-but-growing set where
//! adding a model is one literal in one list, and where the table itself can
//! be walked by a test. A trait would buy dynamic extension that nothing here
//! needs.
//!
//! Only architectures whose config lives in *this* crate can appear. A model
//! crate that owns its own config (`qwen35moe`) depends on this crate, so it
//! cannot be listed here without inverting the dependency; it calls
//! [`crate::import`] directly and stays its own entry point.

use checkpoint::gguf::MmapGguf;

use crate::import::ImportStats;
use crate::kv::architecture;
use crate::{deepseek_ocr, deepseek_ocr_vision};

/// One importable GGUF architecture.
pub struct ArchEntry {
    /// The `general.architecture` value this entry claims.
    pub architecture: &'static str,
    /// An extra `(kv key, value)` that must also match, for architectures that
    /// are shared by several models (`clip`).
    pub discriminator: Option<(&'static str, &'static str)>,
    /// The brain-side model id used when the caller supplies no override.
    pub id: &'static str,
    /// Read the whole file into a brain-native checkpoint at `out_path`.
    pub import: fn(&MmapGguf, &str, Option<&str>) -> Result<ImportStats, String>,
}

/// Every architecture this crate can import.
pub const ARCHITECTURES: &[ArchEntry] = &[
    ArchEntry {
        architecture: deepseek_ocr::GGUF_ARCHITECTURE,
        discriminator: None,
        id: "deepseek-ocr",
        import: deepseek_ocr::import,
    },
    ArchEntry {
        architecture: deepseek_ocr_vision::GGUF_ARCHITECTURE,
        discriminator: Some(("clip.projector_type", deepseek_ocr_vision::PROJECTOR_TYPE)),
        id: "deepseek-ocr-vision",
        import: deepseek_ocr_vision::import,
    },
];

/// The entry that claims `mg`, preferring one with a matching discriminator
/// over a bare architecture match.
pub fn lookup(mg: &MmapGguf) -> Option<&'static ArchEntry> {
    let arch = architecture(mg)?;
    let candidates = ARCHITECTURES.iter().filter(|e| e.architecture == arch);
    let mut fallback = None;
    for e in candidates {
        match e.discriminator {
            Some((key, want)) => {
                if mg.kv().get(key).and_then(|v| v.as_str()) == Some(want) {
                    return Some(e);
                }
            }
            None => fallback = Some(e),
        }
    }
    fallback
}

/// Import any supported GGUF, dispatching on its own metadata.
///
/// Fails by naming what the file actually declared, so an unsupported model
/// reports the architecture (and projector type) to add rather than a generic
/// "unsupported".
pub fn import_gguf(gguf_path: &str, out_path: &str, id_override: Option<&str>) -> Result<ImportStats, String> {
    let mg = MmapGguf::open(gguf_path)?;
    let entry = lookup(&mg).ok_or_else(|| {
        let arch = architecture(&mg).unwrap_or("<none>");
        match mg.kv().get("clip.projector_type").and_then(|v| v.as_str()) {
            Some(p) => format!("gguf: no importer for architecture {arch:?} projector_type {p:?}"),
            None => format!("gguf: no importer for architecture {arch:?}"),
        }
    })?;
    (entry.import)(&mg, out_path, id_override)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_entry_is_uniquely_keyed() {
        // Two entries with the same (architecture, discriminator) would make
        // `lookup` order-dependent - the exact silent-drift failure the
        // catalog pattern exists to prevent.
        let mut keys: Vec<(&str, Option<(&str, &str)>)> =
            ARCHITECTURES.iter().map(|e| (e.architecture, e.discriminator)).collect();
        keys.sort_unstable();
        let n = keys.len();
        keys.dedup();
        assert_eq!(keys.len(), n, "duplicate (architecture, discriminator) key in ARCHITECTURES");
    }

    #[test]
    fn a_shared_architecture_must_carry_a_discriminator() {
        // `clip` is claimed by every mmproj ever produced; an entry for it
        // that keyed on the architecture alone would swallow other models'
        // projector files and import them with the wrong tensor map.
        for e in ARCHITECTURES.iter().filter(|e| e.architecture == "clip") {
            assert!(e.discriminator.is_some(), "clip entry {:?} must be discriminated by projector_type", e.id);
        }
    }
}
