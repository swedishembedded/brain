// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5's [`ArchSpec`]: `dit`/`vae`/`audio_vae`/`text_encoder`/`tokenizer`.
//!
//! `vae` is the only REQUIRED role - see `crate::pipeline`'s module doc for
//! why: today's DiT is a tiny random-weight smoke-test pipeline and the real
//! 12B Gemma-4 text encoder is not imported in this workspace yet, so `dit`/
//! `text_encoder`/`audio_vae`/`tokenizer` are genuinely opt-in (present, the
//! real weight loads; absent, [`crate::pipeline::generate`]'s own documented
//! fallback runs) - never a placeholder for "not implemented".
//!
//! Every check here reads real content, never a filename: the video VAE's
//! own `encoder.conv_in.conv.weight`/`decoder.conv_in.conv.weight` tensor
//! names (the `.conv.` infix is what tells it apart from an unrelated
//! architecture's own `encoder.conv_in.weight`/`decoder.conv_in.weight`
//! naming - see `crate::import::detect_video_vae_arch`), the audio VAE's own
//! `audio_vae.*` tensor namespace, and a GGUF DiT's `general.architecture ==
//! "ltxv"` KV (confirmed unique to this architecture - see
//! `crates/arch/src/lib.rs`'s own `ltxv` row comment).

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

pub struct LtxvSpec;

const ROLES: &[&str] = &["dit", "vae", "audio_vae", "text_encoder", "tokenizer"];
/// Every role but `vae` is opt-in - see this module's doc.
const OPTIONAL: &[&str] = &["dit", "audio_vae", "text_encoder", "tokenizer"];

fn classify_gguf(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    let Ok(g) = checkpoint::gguf::MmapGguf::open(&rec.path.to_string_lossy()) else { return };
    if g.kv().get("general.architecture").and_then(|v| v.as_str()) == Some("ltxv") {
        out.push((idx, "dit".to_string(), Confidence::Declared));
    }
}

/// A bare vendor-flat safetensors file is how both real LTX-2.5 VAE
/// checkpoints ship - never a diffusers-style directory with its own
/// `config.json`, so a record's own tensor names are the only signal.
fn classify_safetensors(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    let Ok(m) = checkpoint::mmap::MmapSafetensors::open(rec.path.to_string_lossy().as_ref()) else { return };
    let has = |name: &str| m.shape(name).is_some();
    // The audio VAE's own tensors are all namespaced `audio_vae.*` in the
    // file (see `crate::import`'s module doc) - checked FIRST, since the
    // video VAE's bare `encoder.conv_in.conv.weight` is a strict suffix of
    // nothing the audio file carries, but checking order here still keeps
    // the two checks independent and unambiguous either way.
    if has("audio_vae.encoder.conv_in.conv.weight") && has("audio_vae.decoder.conv_out.conv.weight") {
        out.push((idx, "audio_vae".to_string(), Confidence::Declared));
        return;
    }
    if has("encoder.conv_in.conv.weight") && has("decoder.conv_in.conv.weight") {
        out.push((idx, "vae".to_string(), Confidence::Declared));
    }
}

impl ArchSpec for LtxvSpec {
    fn arch(&self) -> &'static str {
        "ltxv"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn optional_roles(&self) -> &'static [&'static str] {
        OPTIONAL
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            if !rec.usable() {
                continue;
            }
            match rec.kind {
                ArtifactKind::Gguf => classify_gguf(idx, rec, &mut out),
                ArtifactKind::Safetensors => classify_safetensors(idx, rec, &mut out),
                _ => {}
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        // No shape-derived variant question (unlike FLUX.2's klein-vs-base):
        // `GenOpts::dit_config` is a plain CLI-typed string naming which
        // config this pipeline runs, not something any weight's own header
        // could answer, and it stays a separate flag entirely.
        if !chosen.contains_key("vae") {
            return Err("ltxv assemble: no vae chosen".to_string());
        }
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/ltxv".to_string(), variant: None }))
    }

    fn validate(&self, _assembly: &Assembly) -> Result<(), String> {
        // Every optional role classified here (dit/audio_vae) is already
        // unambiguous from its own header alone (a GGUF's declared
        // architecture, a safetensors file's own tensor namespace) - there
        // is no further cross-role compatibility this can check without
        // reading tensor bytes, which `ArchSpec::validate` may never do.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::inventory::Completeness;
    use brain_modelstore::resolve::{resolve, Question, Resolution};
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-ltxv-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    fn write_video_vae(path: &Path) {
        checkpoint::st::save_safetensors(
            path.to_str().unwrap(),
            &[
                ("decoder.conv_in.conv.weight".to_string(), vec![512, 128, 3, 3, 3], vec![0.0f32; 512 * 128 * 3 * 3 * 3]),
                ("encoder.conv_in.conv.weight".to_string(), vec![128, 3, 3, 3, 3], vec![0.0f32; 128 * 3 * 3 * 3 * 3]),
            ],
            &serde_json::json!({}),
            None,
        )
        .unwrap();
    }

    fn write_audio_vae(path: &Path) {
        checkpoint::st::save_safetensors(
            path.to_str().unwrap(),
            &[
                ("audio_vae.encoder.conv_in.conv.weight".to_string(), vec![32, 1, 3], vec![0.0f32; 32 * 1 * 3]),
                ("audio_vae.decoder.conv_out.conv.weight".to_string(), vec![1, 32, 3], vec![0.0f32; 1 * 32 * 3]),
                ("vocoder.conv_pre.weight".to_string(), vec![8, 1, 7], vec![0.0f32; 8 * 7]),
            ],
            &serde_json::json!({}),
            None,
        )
        .unwrap();
    }

    fn write_dit_gguf(path: &Path) {
        checkpoint::gguf_write::write(
            path.to_str().unwrap(),
            &[("general.architecture".to_string(), checkpoint::gguf::GgufValue::String("ltxv".to_string()))],
            &[checkpoint::gguf_write::TensorOut { name: "w".to_string(), shape: vec![4], ty: checkpoint::gguf::GgmlType::F32.id(), data: vec![0u8; 16] }],
            32,
        )
        .unwrap();
    }

    /// The gap this migration exists to close: `vae` alone (nothing else on
    /// disk) resolves cleanly, with no env var of any kind involved - `dit`/
    /// `audio_vae`/`text_encoder`/`tokenizer` all being absent is documented
    /// as fine, not a `Missing` outcome.
    #[test]
    fn resolves_with_only_a_vae_present_and_no_env_vars_set() {
        let dir = tmp("vae-only");
        std::fs::create_dir_all(&dir).unwrap();
        let vae_path = dir.join("ltx-2.5-video-vae-conv-bf16.safetensors");
        write_video_vae(&vae_path);
        let records = vec![complete(vae_path.clone(), ArtifactKind::Safetensors)];

        let spec = LtxvSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("ltxv", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                assert_eq!(a.roles["vae"], vae_path);
                assert!(!a.roles.contains_key("dit"), "{a:?}");
                assert!(!a.roles.contains_key("audio_vae"), "{a:?}");
                assert!(!a.roles.contains_key("text_encoder"), "{a:?}");
                assert!(!a.roles.contains_key("tokenizer"), "{a:?}");
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    /// With NO vae candidate at all, the required role still goes `Missing`
    /// - `optional_roles` narrows the exemption to exactly the four roles it
    /// names, not every declared role.
    #[test]
    fn missing_without_a_vae_even_though_every_other_role_is_optional() {
        let records: Vec<ArtifactRecord> = Vec::new();
        let spec = LtxvSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("ltxv", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Missing(m) => {
                assert_eq!(m.roles.len(), 1, "{m:?}");
                assert_eq!(m.roles[0].role, "vae");
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    /// When a real DiT GGUF and audio VAE are ALSO present, they are picked
    /// up too - optional roles are not disabled, only exempt from blocking
    /// resolution at zero.
    #[test]
    fn every_optional_role_present_is_still_picked() {
        let dir = tmp("every-role-present");
        std::fs::create_dir_all(&dir).unwrap();
        let vae_path = dir.join("ltx-2.5-video-vae-conv-bf16.safetensors");
        let audio_vae_path = dir.join("ltx-2.5-audio-vae-bf16.safetensors");
        let dit_path = dir.join("ltx-2.5-22b-distilled-transformer-Q8_0.gguf");
        write_video_vae(&vae_path);
        write_audio_vae(&audio_vae_path);
        write_dit_gguf(&dit_path);
        let records = vec![
            complete(vae_path.clone(), ArtifactKind::Safetensors),
            complete(audio_vae_path.clone(), ArtifactKind::Safetensors),
            complete(dit_path.clone(), ArtifactKind::Gguf),
        ];

        let spec = LtxvSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("ltxv", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                assert_eq!(a.roles["vae"], vae_path);
                assert_eq!(a.roles["audio_vae"], audio_vae_path);
                assert_eq!(a.roles["dit"], dit_path);
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    /// Two vae candidates is a real ambiguity, not silently resolved just
    /// because the role happens to be required - classification never picks
    /// for the caller.
    #[test]
    fn two_vae_candidates_is_ambiguous() {
        let dir = tmp("two-vaes");
        std::fs::create_dir_all(&dir).unwrap();
        let a_path = dir.join("a.safetensors");
        let b_path = dir.join("b.safetensors");
        write_video_vae(&a_path);
        write_video_vae(&b_path);
        let records = vec![complete(a_path, ArtifactKind::Safetensors), complete(b_path, ArtifactKind::Safetensors)];

        let spec = LtxvSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("ltxv", &records, &specs, &BTreeMap::new());
        assert!(matches!(out, Resolution::Ambiguous(a) if a.question == Question::Role { role: "vae".to_string() }));
    }
}
