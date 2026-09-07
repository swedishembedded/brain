// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.2's [`ArchSpec`]: which on-disk artifacts satisfy which FLUX.2 role
//! (`dit`, `vae`, `text_encoder`, `tokenizer`), and the one compatibility
//! gate a chosen assembly must pass - the DiT's `context_in_dim` must equal
//! three times the text encoder's hidden size, since a FLUX.2 prompt embed is
//! three concatenated Qwen3 hidden states.
//!
//! Every check here is header/config only: a GGUF's KV metadata and tensor
//! shapes (via [`checkpoint::gguf::MmapGguf`], mmap'd, no tensor bytes
//! decoded) or an HF directory's `config.json`. Nothing here ever reads a
//! tensor value or touches a device - see [`ArchSpec::validate`]'s contract.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::plan::declared_architecture;
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;
use checkpoint::gguf::MmapGguf;

use crate::config::Flux2Config;
use crate::import::dit_config_from_shapes;

pub struct Flux2Spec;

const ROLES: &[&str] = &["dit", "vae", "text_encoder", "tokenizer"];

/// `img_in.weight`'s second dimension for a shared-name (`general.architecture
/// == "flux"`) GGUF: FLUX.2's 2x2 latent pixel-unshuffle over its 32-channel
/// VAE gives 128; FLUX.1 (`crates/flux1/src/config.rs::Flux1Config::dev`)
/// has no such unshuffle and declares 64. Read off `Flux2Config` rather than
/// hardcoded twice.
fn flux2_in_channels() -> usize {
    Flux2Config::klein_4b().in_channels
}

fn gguf_shapes(g: &MmapGguf) -> Vec<(String, Vec<usize>)> {
    g.names().iter().map(|n| (n.clone(), g.shape(n).map(<[usize]>::to_vec).unwrap_or_default())).collect()
}

fn classify_gguf(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    let Ok(g) = MmapGguf::open(&rec.path.to_string_lossy()) else { return };
    let Some(arch) = g.kv().get("general.architecture").and_then(|v| v.as_str()) else { return };
    match arch {
        // Shared with FLUX.1 - the KV field alone can't tell them apart, so
        // this can never rise above Derived (see the module doc): only a
        // shape check settles it, and only a FULL shape match against a
        // known FLUX.2 size class (never a partial one) counts as a real
        // "dit" candidate at all.
        "flux" => {
            let Some(second) = g.shape("img_in.weight").and_then(|s| s.get(1)).copied() else { return };
            if second != flux2_in_channels() {
                return;
            }
            if dit_config_from_shapes(&gguf_shapes(&g)).is_ok() {
                out.push((idx, "dit".to_string(), Confidence::Derived));
            }
        }
        // Unambiguous on its own - no shape check needed to know what this is.
        "qwen3" => out.push((idx, "text_encoder".to_string(), Confidence::Declared)),
        _ => {}
    }
}

fn classify_hfdir(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    let Ok(bytes) = std::fs::read(rec.path.join("config.json")) else { return };
    let Ok(config) = serde_json::from_slice::<serde_json::Value>(&bytes) else { return };
    if declared_architecture(&config).as_deref() == Some("Qwen3ForCausalLM") {
        out.push((idx, "text_encoder".to_string(), Confidence::Declared));
    } else if config.get("_class_name").and_then(|v| v.as_str()) == Some("AutoencoderKLFlux2") {
        out.push((idx, "vae".to_string(), Confidence::Declared));
    }
}

/// A bare vendor-flat VAE checkpoint (`pipeline::Paths::vae` accepts a file
/// as well as a diffusers `vae/` directory - `build_inner`'s `vp.is_dir()`
/// branch) - `decoder.conv_in.weight`/`encoder.conv_in.weight` are the two
/// tensor names FLUX.2's own VAE always carries (`crates/vae/src/decoder.rs`),
/// present together in no other role's checkpoint, so this reads real header
/// content exactly like `classify_gguf`/`classify_hfdir` do - never the name.
fn classify_safetensors(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    let Ok(m) = checkpoint::mmap::MmapSafetensors::open(rec.path.to_string_lossy().as_ref()) else { return };
    let names = m.names();
    if names.iter().any(|n| n == "decoder.conv_in.weight") && names.iter().any(|n| n == "encoder.conv_in.weight") {
        out.push((idx, "vae".to_string(), Confidence::Declared));
    }
}

fn classify_tokenizer(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    let Ok(bytes) = std::fs::read(&rec.path) else { return };
    if serde_json::from_slice::<serde_json::Value>(&bytes).is_ok() {
        out.push((idx, "tokenizer".to_string(), Confidence::Declared));
    }
}

/// `txt_in.weight`'s second dimension - a pure header shape lookup, zero
/// tensor bytes read.
fn dit_context_in_dim(path: &Path) -> Result<usize, String> {
    let g = MmapGguf::open(&path.to_string_lossy()).map_err(|e| format!("flux2 validate: opening dit {}: {e}", path.display()))?;
    let shape = g.shape("txt_in.weight").ok_or_else(|| format!("flux2 validate: {} has no txt_in.weight", path.display()))?;
    shape.get(1).copied().ok_or_else(|| format!("flux2 validate: {} txt_in.weight has fewer than 2 dimensions", path.display()))
}

/// A text encoder's hidden size, from whichever header it actually has: a
/// GGUF's `qwen3.embedding_length` KV, or an HF directory's `config.json`
/// `hidden_size`.
fn text_encoder_hidden(path: &Path) -> Result<usize, String> {
    if path.extension().is_some_and(|e| e == "gguf") {
        let g = MmapGguf::open(&path.to_string_lossy()).map_err(|e| format!("flux2 validate: opening text_encoder {}: {e}", path.display()))?;
        let hidden = g.kv().get("qwen3.embedding_length").and_then(|v| v.as_u64()).ok_or_else(|| format!("flux2 validate: {} has no qwen3.embedding_length", path.display()))?;
        Ok(hidden as usize)
    } else {
        let config_path = path.join("config.json");
        let bytes = std::fs::read(&config_path).map_err(|e| format!("flux2 validate: reading {}: {e}", config_path.display()))?;
        let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| format!("flux2 validate: parsing {}: {e}", config_path.display()))?;
        let hidden = v.get("hidden_size").and_then(|x| x.as_u64()).ok_or_else(|| format!("flux2 validate: {} has no hidden_size", config_path.display()))?;
        Ok(hidden as usize)
    }
}

impl ArchSpec for Flux2Spec {
    fn arch(&self) -> &'static str {
        "flux2"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            // A record whose only evidence would be its own filename never
            // even reaches these branches - every one of them reads real
            // content (a GGUF header's KV/shapes, or a config.json's own
            // declared fields), never the path.
            if !rec.usable() {
                continue;
            }
            match rec.kind {
                ArtifactKind::Gguf => classify_gguf(idx, rec, &mut out),
                ArtifactKind::HfDir => classify_hfdir(idx, rec, &mut out),
                ArtifactKind::TokenizerJson => classify_tokenizer(idx, rec, &mut out),
                ArtifactKind::Safetensors => classify_safetensors(idx, rec, &mut out),
                _ => {}
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord], overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        let dit_idx = *chosen.get("dit").ok_or("flux2 assemble: no dit chosen")?;
        let dit_rec = &records[dit_idx];
        let g = MmapGguf::open(&dit_rec.path.to_string_lossy()).map_err(|e| format!("flux2 assemble: opening dit {}: {e}", dit_rec.path.display()))?;
        let size = dit_config_from_shapes(&gguf_shapes(&g))?;
        let shape_class = size.as_str().to_string();

        if let Some(variant) = overrides.get("variant") {
            if !variant.ends_with(&format!("-{shape_class}")) {
                return Err(format!("flux2 assemble: --variant {variant} does not match the chosen dit's shape class {shape_class}"));
            }
            return Ok(AssembleOutcome::Assembled(AssembledVariant { id: format!("local/flux2-{variant}"), variant: Some(variant.clone()) }));
        }

        // klein-vs-base is not a weight - identical tensor shapes either way
        // (see `import::dit_config_from_shapes`'s doc) - so nothing on disk
        // can ever answer this. Never default to klein.
        Ok(AssembleOutcome::UnresolvedVariant { shape_class: shape_class.clone(), options: vec![format!("klein-{shape_class}"), format!("base-{shape_class}")] })
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let dit_path = assembly.roles.get("dit").ok_or("flux2 validate: assembly has no dit role")?;
        let te_path = assembly.roles.get("text_encoder").ok_or("flux2 validate: assembly has no text_encoder role")?;
        let context_in_dim = dit_context_in_dim(dit_path)?;
        let hidden = text_encoder_hidden(te_path)?;
        if context_in_dim != 3 * hidden {
            return Err(format!(
                "flux2 validate: dit context_in_dim={context_in_dim} is not 3x the text_encoder hidden size (hidden={hidden}, 3x={}) - dit={}, text_encoder={}",
                3 * hidden,
                dit_path.display(),
                te_path.display()
            ));
        }
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
        std::env::temp_dir().join(format!("brain-flux2-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn tiny_tensor(name: &str, shape: Vec<usize>) -> checkpoint::gguf_write::TensorOut {
        let n: usize = shape.iter().product();
        checkpoint::gguf_write::TensorOut { name: name.to_string(), shape, ty: checkpoint::gguf::GgmlType::F32.id(), data: vec![0u8; n * 4] }
    }

    /// Only the tensors [`crate::import::dit_config_from_shapes`] actually
    /// reads (mirrors `import.rs`'s own `minimal_dit_entries`) - NOT the
    /// full 201-tensor manifest, which at real klein-9b dimensions would
    /// write tens of gigabytes of zeros per fixture.
    fn write_dit_gguf(path: &std::path::Path, cfg: &Flux2Config, arch_kv: &str) {
        let tensors = vec![
            tiny_tensor("img_in.weight", vec![cfg.hidden, cfg.in_channels]),
            tiny_tensor("txt_in.weight", vec![cfg.hidden, cfg.context_in_dim]),
            tiny_tensor("double_blocks.0.img_attn.norm.query_norm.scale", vec![cfg.head_dim()]),
            tiny_tensor(&format!("double_blocks.{}.marker", cfg.depth_double - 1), vec![1]),
            tiny_tensor(&format!("single_blocks.{}.marker", cfg.depth_single - 1), vec![1]),
        ];
        checkpoint::gguf_write::write(path.to_str().unwrap(), &[("general.architecture".to_string(), checkpoint::gguf::GgufValue::String(arch_kv.to_string()))], &tensors, 32).unwrap();
    }

    fn write_qwen3_gguf(path: &std::path::Path, hidden: u64) {
        checkpoint::gguf_write::write(
            path.to_str().unwrap(),
            &[
                ("general.architecture".to_string(), checkpoint::gguf::GgufValue::String("qwen3".to_string())),
                ("general.finetune".to_string(), checkpoint::gguf::GgufValue::String("uncensored-text-encoder".to_string())),
                ("qwen3.embedding_length".to_string(), checkpoint::gguf::GgufValue::U64(hidden)),
                ("qwen3.block_count".to_string(), checkpoint::gguf::GgufValue::U32(36)),
            ],
            &[checkpoint::gguf_write::TensorOut { name: "dummy".to_string(), shape: vec![1], ty: checkpoint::gguf::GgmlType::F32.id(), data: vec![0u8; 4] }],
            32,
        )
        .unwrap();
    }

    fn write_qwen3_hfdir(dir: &std::path::Path, hidden: u64) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"architectures": ["Qwen3ForCausalLM"], "hidden_size": hidden})).unwrap()).unwrap();
    }

    fn write_vae_hfdir(dir: &std::path::Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"_class_name": "AutoencoderKLFlux2"})).unwrap()).unwrap();
    }

    fn write_tokenizer_json(path: &std::path::Path) {
        std::fs::write(path, serde_json::to_vec(&serde_json::json!({"version": "1.0"})).unwrap()).unwrap();
    }

    /// A bare vendor-flat VAE safetensors file (`unsloth`'s own release
    /// shape: both `encoder.*`/`decoder.*` halves in one file, no diffusers
    /// `vae/` directory or `config.json` beside it) - the one tensor pair
    /// [`classify_safetensors`] actually checks for, at real-name minimal
    /// payload.
    fn write_vae_safetensors_flat(path: &std::path::Path) {
        checkpoint::st::save_safetensors(
            path.to_str().unwrap(),
            &[("decoder.conv_in.weight".to_string(), vec![1], vec![0.0f32]), ("encoder.conv_in.weight".to_string(), vec![1], vec![0.0f32])],
            &serde_json::json!({}),
            None,
        )
        .unwrap();
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    /// A full, unambiguous set of records for a 9b assembly: a real 9b dit,
    /// a vae, a tokenizer, and exactly ONE text_encoder candidate.
    fn nine_b_fixture(dir: &std::path::Path) -> Vec<ArtifactRecord> {
        std::fs::create_dir_all(dir).unwrap();
        let dit_path = dir.join("dit.gguf");
        write_dit_gguf(&dit_path, &Flux2Config::klein_9b(), "flux");
        let te_path = dir.join("Qwen3-8B");
        write_qwen3_hfdir(&te_path, 4096);
        let vae_path = dir.join("vae");
        write_vae_hfdir(&vae_path);
        let tok_path = dir.join("tokenizer.json");
        write_tokenizer_json(&tok_path);
        vec![
            complete(dit_path, ArtifactKind::Gguf),
            complete(te_path, ArtifactKind::HfDir),
            complete(vae_path, ArtifactKind::HfDir),
            complete(tok_path, ArtifactKind::TokenizerJson),
        ]
    }

    #[test]
    fn classify_uses_content_not_filename() {
        let dir = tmp("content-not-filename");
        std::fs::create_dir_all(&dir).unwrap();

        // Named exactly like a FLUX.2 DiT release, but its HEADER says
        // qwen3 - proving classification reads content, not the name.
        let suggestive = dir.join("flux2-klein-9b-uncensored-q8_0.gguf");
        write_qwen3_gguf(&suggestive, 4096);

        // Same suggestive naming pattern, but genuinely unreadable (no gguf
        // magic at all) - must produce NO classification whatsoever, since
        // the filename alone is never enough.
        let unreadable = dir.join("flux2-klein-9b-totally-legit-q8_0.gguf");
        std::fs::write(&unreadable, b"not a gguf file").unwrap();

        let records = vec![complete(suggestive.clone(), ArtifactKind::Gguf), complete(unreadable.clone(), ArtifactKind::Gguf)];
        let out = Flux2Spec.classify(&records, dir.as_path());

        assert_eq!(out, vec![(0, "text_encoder".to_string(), Confidence::Declared)], "{out:?}");
        assert!(out.iter().all(|(idx, ..)| *idx != 1), "the unreadable file must never classify, {out:?}");
    }

    /// `pipeline::Paths::vae` accepts a bare file, not only a diffusers
    /// `vae/` directory (`build_inner`'s own `vp.is_dir()` branch) - a real
    /// vendor-flat release (`unsloth/flux2-vae.safetensors`) ships exactly
    /// that shape, so classification must recognize it from its own tensor
    /// names, not only from `classify_hfdir`'s `_class_name` check.
    #[test]
    fn classify_recognizes_a_bare_vendor_flat_vae_safetensors_file() {
        let dir = tmp("flat-vae");
        std::fs::create_dir_all(&dir).unwrap();
        let vae_path = dir.join("flux2-vae.safetensors");
        write_vae_safetensors_flat(&vae_path);

        // A decoy safetensors file with unrelated tensor names must NOT
        // classify as anything - proving this reads real tensor names, not
        // "any bare safetensors file is a vae".
        let decoy_path = dir.join("flux2-vae-decoy.safetensors");
        checkpoint::st::save_safetensors(decoy_path.to_str().unwrap(), &[("some.other.weight".to_string(), vec![1], vec![0.0f32])], &serde_json::json!({}), None).unwrap();

        let records = vec![complete(vae_path, ArtifactKind::Safetensors), complete(decoy_path, ArtifactKind::Safetensors)];
        let out = Flux2Spec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "vae".to_string(), Confidence::Declared)], "{out:?}");
    }

    #[test]
    fn a_9b_dit_with_a_4b_text_encoder_is_rejected_by_validate_before_any_gpu_work() {
        let dir = tmp("mismatched-validate");
        std::fs::create_dir_all(&dir).unwrap();
        let dit_path = dir.join("dit.gguf");
        write_dit_gguf(&dit_path, &Flux2Config::klein_9b(), "flux");
        let te_path = dir.join("Qwen3-4B");
        write_qwen3_hfdir(&te_path, 2560);

        let assembly = Assembly {
            id: "local/flux2-klein-9b".to_string(),
            arch: "flux2".to_string(),
            variant: Some("klein-9b".to_string()),
            roles: BTreeMap::from([("dit".to_string(), dit_path.clone()), ("text_encoder".to_string(), te_path.clone())]),
            provenance: Vec::new(),
        };
        let err = Flux2Spec.validate(&assembly).unwrap_err();
        assert!(err.contains("12288"), "{err}");
        assert!(err.contains("2560"), "{err}");
    }

    #[test]
    fn variant_is_ambiguous_without_a_stated_klein_or_base() {
        let dir = tmp("variant-ambiguous");
        let records = nine_b_fixture(&dir);
        let spec = Flux2Spec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("flux2", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Ambiguous(a) => {
                assert_eq!(a.question, Question::Variant { shape_class: "9b".to_string() });
                assert_eq!(a.choices.len(), 2);
                let selectors: Vec<(String, String)> = a.choices.iter().flat_map(|c| c.selector.clone()).collect();
                assert!(selectors.contains(&("--variant".to_string(), "klein-9b".to_string())), "{selectors:?}");
                assert!(selectors.contains(&("--variant".to_string(), "base-9b".to_string())), "{selectors:?}");
            }
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }

    #[test]
    fn two_text_encoder_candidates_at_equal_confidence_is_ambiguous_not_a_silent_pick() {
        let dir = tmp("two-text-encoders");
        let mut records = nine_b_fixture(&dir);
        // A second, equally Declared text_encoder candidate: the real
        // machine's unsloth uncensored-qwen3 GGUF, same embedding_length.
        let second_te = dir.join("flux2-klein-9b-uncensored-q8_0.gguf");
        write_qwen3_gguf(&second_te, 4096);
        records.push(complete(second_te, ArtifactKind::Gguf));

        let spec = Flux2Spec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let mut overrides = BTreeMap::new();
        overrides.insert("variant".to_string(), "klein-9b".to_string());
        let out = resolve("flux2", &records, &specs, &overrides);
        match out {
            Resolution::Ambiguous(a) => {
                assert_eq!(a.question, Question::Role { role: "text_encoder".to_string() });
                assert_eq!(a.choices.len(), 2);
            }
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }

    #[test]
    fn an_explicit_override_collapses_an_ambiguity() {
        let dir = tmp("override-collapses");
        let mut records = nine_b_fixture(&dir);
        let second_te = dir.join("flux2-klein-9b-uncensored-q8_0.gguf");
        write_qwen3_gguf(&second_te, 4096);
        records.push(complete(second_te.clone(), ArtifactKind::Gguf));

        let spec = Flux2Spec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let mut overrides = BTreeMap::new();
        overrides.insert("variant".to_string(), "klein-9b".to_string());
        overrides.insert("text_encoder".to_string(), second_te.to_string_lossy().into_owned());
        let out = resolve("flux2", &records, &specs, &overrides);
        match out {
            Resolution::Resolved(a) => {
                assert_eq!(a.roles["text_encoder"], second_te);
                assert_eq!(a.variant.as_deref(), Some("klein-9b"));
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }
}
