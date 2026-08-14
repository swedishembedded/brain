// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Legacy short name → canonical fully-qualified id, for backward compatibility
//! during the migration to `<vendor>/<repo>[-<QUANT>]` names.
//!
//! Each entry is a **deprecation**, not a second API: a request naming the
//! legacy short form still resolves, but `GET /models` never lists it, and the
//! server warns once per name per process (see
//! `crates/apiserve/src/catalog.rs`'s `candidates()` and
//! `residency::Executor::ensure_model`'s D-Bus/`brain do` equivalent — the
//! **only** two places this table is consulted). A [`Manifest`]'s `model` field
//! itself is NEVER a legacy name; it is always canonical.
//!
//! [`Manifest`]: https://docs.rs/brain-capability (crates/capability)

use std::sync::OnceLock;

/// One legacy-name → canonical-id row.
struct Row {
    legacy: &'static str,
    canonical: &'static str,
}

/// The built-ins: every one of these ships inside the `brain` binary itself
/// (no upstream repo — see the module docs' `brain/` vendor), so unlike an
/// env-loaded LLM resident, their canonical id is fixed at compile time and a
/// static table entry is exactly right.
const ROWS: &[Row] = &[
    Row { legacy: "mock", canonical: "brain/mock" },
    Row { legacy: "demo", canonical: "brain/demo" },
    Row { legacy: "imageops", canonical: "brain/imageops" },
    Row { legacy: "fastvlm", canonical: "brain/fastvlm" },
    Row { legacy: "yolo", canonical: "brain/yolo" },
    Row { legacy: "depth", canonical: "brain/depth" },
    Row { legacy: "z-image", canonical: "brain/z-image" },
    Row { legacy: "flux2-klein", canonical: "brain/flux2-klein" },
    Row { legacy: "tts", canonical: "brain/tts" },
    Row { legacy: "chronos2", canonical: "brain/chronos2" },
    Row { legacy: "fincast", canonical: "brain/fincast" },
    Row { legacy: "kronos", canonical: "brain/kronos" },
    Row { legacy: "gpt", canonical: "brain/gpt" },
    Row { legacy: "glm", canonical: "brain/glm" },
    // The imaging stack. These shipped on the p40 branch under bare names
    // before the fully-qualified scheme landed; the rows keep those callers
    // (examples, docs, D-Bus clients) working.
    Row { legacy: "sam2", canonical: "brain/sam2" },
    Row { legacy: "scrfd", canonical: "brain/scrfd" },
    Row { legacy: "arcface", canonical: "brain/arcface" },
    Row { legacy: "vqgan", canonical: "brain/vqgan" },
    Row { legacy: "restore", canonical: "brain/restore" },
    Row { legacy: "clip", canonical: "brain/clip" },
    Row { legacy: "imgpipe", canonical: "brain/imgpipe" },
    Row { legacy: "upscale", canonical: "brain/upscale" },
    // qwen/lfm/nemotron/qwen-asr are env-loaded from an arbitrary checkpoint,
    // so their canonical id depends on what was actually imported (see
    // `resident_llm.rs`'s BRAIN_QWEN_REF / BRAIN_LFM2_REF / BRAIN_NEMOTRONASR_REF /
    // BRAIN_QWEN3ASR_REF) and is NOT fixed at compile time. When that env var
    // is unset, the resident registers under its own brain/<family> fallback
    // (row above/below), so the short legacy name still resolves via THAT row.
    Row { legacy: "qwen", canonical: "brain/qwen" },
    Row { legacy: "lfm", canonical: "brain/lfm" },
    Row { legacy: "nemotron", canonical: "brain/nemotron" },
    Row { legacy: "qwen-asr", canonical: "brain/qwen-asr" },
];

fn table() -> &'static std::collections::HashMap<&'static str, &'static str> {
    static TABLE: OnceLock<std::collections::HashMap<&'static str, &'static str>> = OnceLock::new();
    TABLE.get_or_init(|| ROWS.iter().map(|r| (r.legacy, r.canonical)).collect())
}

/// The canonical id a legacy short name resolves to, or `None` if `name` is
/// not a known legacy name (including: it's already canonical, or it's simply
/// unknown — both cases are the caller's to handle, e.g. by treating `name` as
/// literal and letting the normal "model not found" path fire).
pub fn canonical(name: &str) -> Option<&'static str> {
    table().get(name).copied()
}

/// Every legacy name this table knows, for a completeness test (`model_ids.rs`)
/// that asserts none of them collides with a real canonical id.
pub fn legacy_names() -> impl Iterator<Item = &'static str> {
    ROWS.iter().map(|r| r.legacy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_legacy_names_resolve() {
        assert_eq!(canonical("mock"), Some("brain/mock"));
        assert_eq!(canonical("qwen"), Some("brain/qwen"));
    }

    #[test]
    fn unknown_or_already_canonical_names_resolve_to_none() {
        assert_eq!(canonical("brain/mock"), None);
        assert_eq!(canonical("Qwen/Qwen3-0.6B"), None);
        assert_eq!(canonical("totally-unknown"), None);
    }

    #[test]
    fn every_row_canonical_is_a_valid_ref_under_a_reserved_vendor() {
        for name in legacy_names() {
            let canon = canonical(name).unwrap();
            let r = crate::ModelRef::parse(canon).unwrap_or_else(|e| panic!("{canon}: {e}"));
            assert!(r.is_reserved(), "{canon} should be under a reserved vendor");
        }
    }
}
