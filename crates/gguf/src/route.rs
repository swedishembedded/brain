// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which model is this GGUF? The one answer, for every consumer.
//!
//! Swedish Embedded AB implements format-agnostic checkpoint loading for its
//! clients. If your team needs expertise in shipping one model runtime that
//! reads whatever checkpoint format the ecosystem hands it, then you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! # Why this exists
//!
//! A GGUF names its own architecture, in its own metadata, under
//! `general.architecture`. brain has a canonical architecture registry
//! (`brain_arch`) keyed by exactly that spelling ([`brain_arch::by_gguf`]).
//! Those two facts are all that "load this file" needs, and yet the question
//! "which model is this file" had grown **three** independent answers: the
//! importer table in `cli::gguf_import`, the architecture table this crate's
//! own `registry` used to carry, and a hand-written family-alias `match` in
//! the model-directory scan. Three tables meant a file could be importable
//! through one and reported as unsupported by another, which is what happened
//! to DeepSeek-OCR: its importer lived here, and the generic `brain import`
//! command answered "no GGUF importer yet".
//!
//! [`route`] is the single seam. It reads the file's own metadata, resolves it
//! against `brain_arch`, and hands back a [`Route`]. Everything downstream
//! dispatches on [`Route::id`], which is a `brain_arch` id and therefore the
//! same key the CLI verb, the model card, the fetch recipe and the docs page
//! already use. Adding an architecture is a row in `brain_arch`, not a branch
//! in each consumer.
//!
//! # The projector discriminator
//!
//! `general.architecture` alone is not always sufficient. Every multimodal
//! projector file llama.cpp's mtmd tooling produces (the `mmproj-*.gguf` that
//! ships beside a vision-language model's language half) declares
//! `general.architecture = "clip"` and identifies its real owner in a second
//! key, `clip.projector_type`. [`Route::projector`] carries that value, so a
//! consumer can tell a model file from the projector that belongs to it
//! without opening the file a second time or guessing from the filename.
//!
//! That distinction is not cosmetic. A vision-language GGUF release is TWO
//! files, and pointing a loader at only the language half yields a model that
//! cannot see. [`sibling_projector`] finds the companion by ITS OWN metadata
//! rather than by a hardcoded filename, so a release that spells it
//! `mmproj-F16.gguf` and one that spells it `mmproj-<Model>-Q8_0.gguf` both
//! resolve.

use std::path::{Path, PathBuf};

use checkpoint::gguf::MmapGguf;

use crate::kv::architecture;

/// The GGUF metadata key naming a projector file's real owner.
pub const PROJECTOR_TYPE_KEY: &str = "clip.projector_type";

/// What one GGUF file declares itself to be, resolved against the canonical
/// architecture registry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Route {
    /// The `general.architecture` string the file actually carries, verbatim.
    /// Kept alongside [`Route::arch`] because the two can differ (llama.cpp
    /// spells DeepSeek-OCR `deepseek2-ocr`; brain's id grammar forbids the
    /// hyphen) and an error message should quote what the file said.
    pub tag: String,
    /// The registry row that spelling resolves to.
    pub arch: &'static brain_arch::Arch,
    /// `clip.projector_type` when this file is a multimodal projector.
    pub projector: Option<String>,
}

impl Route {
    /// brain's canonical architecture id: the one key every consumer
    /// dispatches on.
    pub fn id(&self) -> &'static str {
        self.arch.id
    }

    /// Whether this file is a multimodal projector (an `mmproj-*.gguf`) rather
    /// than a model in its own right.
    pub fn is_projector(&self) -> bool {
        self.projector.is_some()
    }
}

/// Read a GGUF's `general.architecture` and resolve it, or say precisely why
/// it could not be resolved.
///
/// Two failures, told apart on purpose: a file that declares no architecture
/// at all is malformed or not a model, while a file that declares one brain
/// has never heard of is a real gap that should be reported by name so it can
/// be added as a registry row.
pub fn route(mg: &MmapGguf) -> Result<Route, String> {
    let tag = architecture(mg).ok_or("gguf: no 'general.architecture' in the metadata")?.to_string();
    let arch = brain_arch::by_gguf(&tag)
        .ok_or_else(|| format!("gguf: unknown GGUF architecture {tag:?} (brain has no architecture registered under that name)"))?;
    let projector = mg.kv().get(PROJECTOR_TYPE_KEY).and_then(|v| v.as_str()).map(str::to_string);
    Ok(Route { tag, arch, projector })
}

/// [`route`] over a path, opening and dropping the mapping itself. Errors are
/// prefixed with the path, since a caller that passed a path is usually
/// holding several.
pub fn route_path(path: &str) -> Result<Route, String> {
    let mg = MmapGguf::open(path).map_err(|e| format!("{path}: {e}"))?;
    route(&mg).map_err(|e| format!("{path}: {e}"))
}

/// Whether `path` names a GGUF at all, by its own leading magic rather than by
/// its extension. `false` for an unreadable path: the caller's own open will
/// report the I/O error with better context than this can.
pub fn is_gguf(path: &Path) -> bool {
    use std::io::Read;
    let mut magic = [0u8; 4];
    std::fs::File::open(path).and_then(|mut f| f.read_exact(&mut magic)).is_ok() && &magic == b"GGUF"
}

/// The multimodal projector that belongs beside `model_path`: a sibling GGUF
/// whose OWN metadata declares it a projector.
///
/// Found by metadata, never by filename, because the filename is the one part
/// of a release that varies (`mmproj-F16.gguf`, `mmproj-BF16.gguf`,
/// `mmproj-<Model>-Q8_0.gguf` are all the same role). `None` when no sibling
/// declares itself one, which is exactly the state a vision-language loader
/// must refuse rather than proceed blind from.
pub fn sibling_projector(model_path: &Path) -> Option<PathBuf> {
    let dir = model_path.parent()?;
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p != model_path && p.extension().is_some_and(|x| x == "gguf"))
        .filter(|p| p.to_str().and_then(|s| route_path(s).ok()).is_some_and(|r| r.is_projector()))
        .collect();
    found.sort();
    found.into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use checkpoint::gguf::GgufValue;
    use checkpoint::gguf_write::{write, TensorOut};

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("brain-gguf-route-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_gguf(path: &Path, kvs: &[(&str, &str)]) {
        let kvs: Vec<(String, GgufValue)> = kvs.iter().map(|(k, v)| (k.to_string(), GgufValue::String(v.to_string()))).collect();
        let tensors = vec![TensorOut { name: "w".into(), shape: vec![1], ty: 0, data: 0f32.to_le_bytes().to_vec() }];
        write(path.to_str().unwrap(), &kvs, &tensors, 32).unwrap();
    }

    /// The requirement in one line: a file's own architecture tag picks the
    /// brain architecture, including where llama.cpp's spelling and brain's
    /// id grammar disagree.
    #[test]
    fn a_file_routes_to_its_own_architecture_by_metadata_alone() {
        let dir = tmp("by-metadata");
        for (tag, want) in [("qwen3", "qwen3"), ("deepseek2-ocr", "deepseek2ocr"), ("wan", "wan"), ("ltxv", "ltxv")] {
            let p = dir.join(format!("{want}.gguf"));
            write_gguf(&p, &[("general.architecture", tag)]);
            let r = route_path(p.to_str().unwrap()).unwrap_or_else(|e| panic!("{tag}: {e}"));
            assert_eq!(r.id(), want, "{tag} must route to brain's {want}");
            assert_eq!(r.tag, tag, "the raw tag is preserved for error messages");
            assert!(!r.is_projector());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An architecture brain has no row for is refused BY NAME. Qwen3-VL's
    /// 30B-A3B release is the live example: llama.cpp registers the MoE
    /// vision-language decoder as its own architecture, which brain does not
    /// build, and the only safe outcome is to say so rather than load the
    /// adjacent dense architecture.
    #[test]
    fn an_unknown_architecture_is_refused_by_name() {
        let dir = tmp("unknown");
        let p = dir.join("mystery.gguf");
        write_gguf(&p, &[("general.architecture", "qwen3vlmoe")]);
        let err = route_path(p.to_str().unwrap()).unwrap_err();
        assert!(err.contains("qwen3vlmoe"), "must name the architecture: {err}");
        assert!(err.contains("unknown GGUF architecture"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_file_with_no_architecture_key_is_refused_naming_the_key() {
        let dir = tmp("noarch");
        let p = dir.join("bare.gguf");
        write_gguf(&p, &[]);
        let err = route_path(p.to_str().unwrap()).unwrap_err();
        assert!(err.contains("general.architecture"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A vision-language release is two files, and the second one is found by
    /// its own metadata rather than by a filename pattern.
    #[test]
    fn a_projector_is_recognized_and_located_beside_its_model() {
        let dir = tmp("mmproj");
        let lm = dir.join("Model-Q8_0.gguf");
        let proj = dir.join("mmproj-F16.gguf");
        write_gguf(&lm, &[("general.architecture", "deepseek2-ocr")]);
        write_gguf(&proj, &[("general.architecture", "clip"), (PROJECTOR_TYPE_KEY, "deepseekocr")]);

        let r = route_path(proj.to_str().unwrap()).unwrap();
        assert!(r.is_projector(), "an mmproj must be recognized as a projector, not as a model");
        assert_eq!(r.projector.as_deref(), Some("deepseekocr"));
        assert_eq!(r.id(), "clip", "the file really is a clip tower; its owner is the projector_type");

        assert_eq!(sibling_projector(&lm), Some(proj), "the companion is found by metadata, not by filename");
        std::fs::remove_dir_all(&dir).ok();

        // A release that ships only the language half has no companion, which
        // is the state a vision-language loader must refuse rather than load
        // a model that cannot see.
        let alone = tmp("mmproj-absent");
        let lm = alone.join("Model-Q8_0.gguf");
        write_gguf(&lm, &[("general.architecture", "deepseek2-ocr")]);
        assert_eq!(sibling_projector(&lm), None);
        std::fs::remove_dir_all(&alone).ok();
    }

    #[test]
    fn is_gguf_reads_the_magic_not_the_extension() {
        let dir = tmp("magic");
        let real = dir.join("real.bin");
        write_gguf(&real, &[("general.architecture", "qwen3")]);
        let fake = dir.join("fake.gguf");
        std::fs::write(&fake, b"not a gguf at all").unwrap();
        assert!(is_gguf(&real));
        assert!(!is_gguf(&fake));
        assert!(!is_gguf(&dir.join("absent.gguf")));
        std::fs::remove_dir_all(&dir).ok();
    }
}
