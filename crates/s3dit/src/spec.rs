// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Z-Image's [`ArchSpec`]: which on-disk artifacts satisfy which Z-Image role
//! (`dit`, `vae`, `text_encoder`, `tokenizer`), and the one compatibility gate
//! a chosen assembly must pass - the DiT's `cap_feat_dim` must equal the
//! Qwen3-4B text encoder's `d_model` (the caption embedder projects the
//! encoder's own hidden states).
//!
//! Every check here is header/config only: a GGUF's KV metadata and tensor
//! shapes ([`checkpoint::gguf::MmapGguf`]) or a safetensors file's own tensor
//! shapes ([`checkpoint::mmap::MmapSafetensors`]). Nothing here ever reads a
//! tensor value or touches a device - see [`ArchSpec::validate`]'s contract.
//!
//! Unlike FLUX.2/Wan there is no variant to disambiguate: Z-Image-Turbo is
//! the one shipped config ([`crate::model::ZImageConfig::turbo`]), so
//! [`S3ditSpec::assemble`] has nothing to ask once a `dit` is chosen.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::plan::declared_architecture;
use brain_modelstore::resolve::{classify_tokenizer_role, ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;
use checkpoint::gguf::MmapGguf;
use checkpoint::mmap::MmapSafetensors;

use crate::import::{dit_config_from_shapes, DISCRIMINATOR_TENSOR, GGUF_ARCHITECTURE};

pub struct S3ditSpec;

const ROLES: &[&str] = &["dit", "vae", "text_encoder", "tokenizer"];

/// Every tensor name + shape a DiT checkpoint declares, in the COMFY
/// spelling [`dit_config_from_shapes`] itself expects - read from whichever
/// header it actually has.
fn dit_shapes(path: &Path) -> Result<Vec<(String, Vec<usize>)>, String> {
    if path.extension().is_some_and(|e| e == "gguf") {
        let g = MmapGguf::open(&path.to_string_lossy()).map_err(|e| format!("s3dit: opening dit {}: {e}", path.display()))?;
        Ok(g.all_shapes())
    } else {
        let m = MmapSafetensors::open(path).map_err(|e| format!("s3dit: opening dit {}: {e}", path.display()))?;
        Ok(m.names().iter().map(|n| (n.clone(), m.shape(n).expect("name came from names()").to_vec())).collect())
    }
}

fn classify_gguf(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    let Ok(g) = MmapGguf::open(&rec.path.to_string_lossy()) else { return };
    let Some(arch) = g.kv().get("general.architecture").and_then(|v| v.as_str()) else { return };
    match arch {
        // Shared with a real Lumina2 GGUF (`crate::import::GGUF_ARCHITECTURE`'s
        // own doc) - never rises above Derived, and only a real Z-Image
        // discriminator tensor plus a full shape match counts as a candidate.
        GGUF_ARCHITECTURE => {
            if g.shape(DISCRIMINATOR_TENSOR).is_none() {
                return;
            }
            if dit_config_from_shapes(&g.all_shapes()).is_ok() {
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
    }
}

/// A bare safetensors file, either the DiT (Comfy tensor names, unique to
/// Z-Image - `layers.N.attention.qkv.weight`/`cap_embedder.*`) or the VAE
/// (the generic `decoder`/`encoder`.conv_in.weight autoencoder names FLUX.2's
/// own VAE also carries - the conv rank tells a 2D image VAE (4 dims) apart
/// from a causal 3D video VAE (5 dims, Wan's own release), the same guard
/// `flux2::spec`'s classifier reads).
fn classify_safetensors(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    let Ok(m) = MmapSafetensors::open(&rec.path) else { return };
    if m.shape(DISCRIMINATOR_TENSOR).is_some() {
        let shapes: Vec<(String, Vec<usize>)> = m.names().iter().map(|n| (n.clone(), m.shape(n).expect("name came from names()").to_vec())).collect();
        if dit_config_from_shapes(&shapes).is_ok() {
            out.push((idx, "dit".to_string(), Confidence::Declared));
        }
        return;
    }
    let is_2d_conv = |name: &str| m.shape(name).is_some_and(|s| s.len() == 4);
    if is_2d_conv("decoder.conv_in.weight") && is_2d_conv("encoder.conv_in.weight") {
        out.push((idx, "vae".to_string(), Confidence::Declared));
    }
}

/// The Qwen3-4B text encoder's own declared hidden size, from whichever
/// header it actually has - mirrors `flux2::spec::text_encoder_hidden`
/// exactly (a different crate's own copy of the same small header read, not
/// a shared function - see this crate's migration notes).
fn text_encoder_hidden(path: &Path) -> Result<usize, String> {
    if path.extension().is_some_and(|e| e == "gguf") {
        let g = MmapGguf::open(&path.to_string_lossy()).map_err(|e| format!("s3dit validate: opening text_encoder {}: {e}", path.display()))?;
        let hidden = g.kv().get("qwen3.embedding_length").and_then(|v| v.as_u64()).ok_or_else(|| format!("s3dit validate: {} has no qwen3.embedding_length", path.display()))?;
        Ok(hidden as usize)
    } else {
        let config_path = path.join("config.json");
        let bytes = std::fs::read(&config_path).map_err(|e| format!("s3dit validate: reading {}: {e}", config_path.display()))?;
        let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| format!("s3dit validate: parsing {}: {e}", config_path.display()))?;
        let hidden = v.get("hidden_size").and_then(|x| x.as_u64()).ok_or_else(|| format!("s3dit validate: {} has no hidden_size", config_path.display()))?;
        Ok(hidden as usize)
    }
}

impl ArchSpec for S3ditSpec {
    fn arch(&self) -> &'static str {
        "s3dit"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            if !rec.usable() {
                continue;
            }
            match rec.kind {
                ArtifactKind::Gguf => classify_gguf(idx, rec, &mut out),
                ArtifactKind::HfDir => classify_hfdir(idx, rec, &mut out),
                ArtifactKind::Safetensors => classify_safetensors(idx, rec, &mut out),
                _ => {}
            }
        }
        let text_encoder_candidates: Vec<&Path> = out.iter().filter(|(_, role, _)| role == "text_encoder").map(|(idx, ..)| records[*idx].path.as_path()).collect();
        classify_tokenizer_role(records, inventory_root, "tokenizer", &text_encoder_candidates, &mut out);
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        let dit_idx = *chosen.get("dit").ok_or("s3dit assemble: no dit chosen")?;
        let dit_rec = &records[dit_idx];
        // A real gate, not a formality: confirms the chosen record really is
        // Z-Image-shaped before assembling - the same check `classify` used
        // to admit it as a candidate in the first place.
        dit_config_from_shapes(&dit_shapes(&dit_rec.path)?)?;
        // No variant to ask about - Z-Image-Turbo is the one shipped config.
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/s3dit-turbo".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let dit_path = assembly.roles.get("dit").ok_or("s3dit validate: assembly has no dit role")?;
        let te_path = assembly.roles.get("text_encoder").ok_or("s3dit validate: assembly has no text_encoder role")?;
        let cfg = dit_config_from_shapes(&dit_shapes(dit_path)?)?;
        let hidden = text_encoder_hidden(te_path)?;
        if cfg.cap_feat_dim as usize != hidden {
            return Err(format!(
                "s3dit validate: dit cap_feat_dim={} does not match text_encoder hidden size={hidden} - dit={}, text_encoder={}",
                cfg.cap_feat_dim,
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
    use brain_modelstore::resolve::{resolve, Resolution};
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-s3dit-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    /// Only the tensors [`dit_config_from_shapes`] actually reads, at real
    /// `ZImageConfig::turbo()` dimensions - NOT the full block-by-block
    /// manifest.
    fn write_dit_safetensors(path: &std::path::Path) {
        let cfg = crate::model::ZImageConfig::turbo();
        let (dim, cap_feat_dim, head_dim) = (cfg.dim as usize, cfg.cap_feat_dim as usize, (cfg.dim / cfg.n_heads) as usize);
        let patch_dim = (cfg.in_channels * cfg.patch_size * cfg.patch_size * cfg.f_patch_size) as usize;
        let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
            // `DISCRIMINATOR_TENSOR` - required for classification to admit
            // this as a candidate at all, distinct from `cap_embedder.1`
            // (which `dit_config_from_shapes` reads dimensions off).
            ("cap_embedder.0.weight".to_string(), vec![1], vec![0.0f32]),
            ("cap_embedder.1.weight".to_string(), vec![dim as u64, cap_feat_dim as u64], vec![0.0f32; dim * cap_feat_dim]),
            ("layers.0.attention.q_norm.weight".to_string(), vec![head_dim as u64], vec![0.0f32; head_dim]),
            ("x_embedder.weight".to_string(), vec![dim as u64, patch_dim as u64], vec![0.0f32; dim * patch_dim]),
        ];
        for prefix in ["layers", "noise_refiner", "context_refiner"] {
            let n = if prefix == "layers" { cfg.n_layers } else { cfg.n_refiner_layers };
            for l in 0..n {
                tensors.push((format!("{prefix}.{l}.attention.qkv.weight"), vec![1], vec![0.0f32]));
            }
        }
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &serde_json::json!({}), None).unwrap();
    }

    fn write_dit_gguf(path: &std::path::Path, arch_kv: &str) {
        let cfg = crate::model::ZImageConfig::turbo();
        let (dim, cap_feat_dim, head_dim) = (cfg.dim as usize, cfg.cap_feat_dim as usize, (cfg.dim / cfg.n_heads) as usize);
        let patch_dim = (cfg.in_channels * cfg.patch_size * cfg.patch_size * cfg.f_patch_size) as usize;
        let f32ty = checkpoint::gguf::GgmlType::F32.id();
        let mut tensors = vec![
            checkpoint::gguf_write::TensorOut { name: "cap_embedder.0.weight".to_string(), shape: vec![1], ty: f32ty, data: vec![0u8; 4] },
            checkpoint::gguf_write::TensorOut { name: "cap_embedder.1.weight".to_string(), shape: vec![dim, cap_feat_dim], ty: f32ty, data: vec![0u8; dim * cap_feat_dim * 4] },
            checkpoint::gguf_write::TensorOut { name: "layers.0.attention.q_norm.weight".to_string(), shape: vec![head_dim], ty: f32ty, data: vec![0u8; head_dim * 4] },
            checkpoint::gguf_write::TensorOut { name: "x_embedder.weight".to_string(), shape: vec![dim, patch_dim], ty: f32ty, data: vec![0u8; dim * patch_dim * 4] },
        ];
        for prefix in ["layers", "noise_refiner", "context_refiner"] {
            let n = if prefix == "layers" { cfg.n_layers } else { cfg.n_refiner_layers };
            for l in 0..n {
                tensors.push(checkpoint::gguf_write::TensorOut { name: format!("{prefix}.{l}.attention.qkv.weight"), shape: vec![1], ty: f32ty, data: vec![0u8; 4] });
            }
        }
        checkpoint::gguf_write::write(path.to_str().unwrap(), &[("general.architecture".to_string(), checkpoint::gguf::GgufValue::String(arch_kv.to_string()))], &tensors, 32).unwrap();
    }

    /// Toy vocab size every fixture writer below agrees on.
    const TOY_VOCAB: usize = 100;

    fn write_qwen3_hfdir(dir: &std::path::Path, hidden: u64) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"architectures": ["Qwen3ForCausalLM"], "hidden_size": hidden, "vocab_size": TOY_VOCAB})).unwrap()).unwrap();
    }

    fn write_tokenizer_json(path: &std::path::Path) {
        let vocab: serde_json::Map<String, serde_json::Value> = (0..TOY_VOCAB).map(|i| (format!("t{i}"), serde_json::json!(i))).collect();
        std::fs::write(path, serde_json::to_vec(&serde_json::json!({"version": "1.0", "model": {"vocab": vocab}, "added_tokens": []})).unwrap()).unwrap();
    }

    fn write_vae_safetensors(path: &std::path::Path) {
        checkpoint::st::save_safetensors(
            path.to_str().unwrap(),
            &[
                ("decoder.conv_in.weight".to_string(), vec![512, 32, 3, 3], vec![0.0f32; 512 * 32 * 3 * 3]),
                ("encoder.conv_in.weight".to_string(), vec![128, 3, 3, 3], vec![0.0f32; 128 * 3 * 3 * 3]),
            ],
            &serde_json::json!({}),
            None,
        )
        .unwrap();
    }

    fn write_causal_3d_vae_safetensors(path: &std::path::Path) {
        checkpoint::st::save_safetensors(
            path.to_str().unwrap(),
            &[
                ("decoder.conv_in.weight".to_string(), vec![384, 16, 3, 3, 3], vec![0.0f32; 384 * 16 * 3 * 3 * 3]),
                ("encoder.conv_in.weight".to_string(), vec![96, 3, 3, 3, 3], vec![0.0f32; 96 * 3 * 3 * 3 * 3]),
            ],
            &serde_json::json!({}),
            None,
        )
        .unwrap();
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    /// A full, unambiguous set of records for a Z-Image-Turbo assembly.
    fn turbo_fixture(dir: &std::path::Path) -> Vec<ArtifactRecord> {
        let vendor = dir.join("Tongyi-MAI");
        std::fs::create_dir_all(&vendor).unwrap();
        let dit_path = vendor.join("dit.safetensors");
        write_dit_safetensors(&dit_path);
        let te_path = vendor.join("Qwen3-4B");
        write_qwen3_hfdir(&te_path, crate::model::ZImageConfig::turbo().cap_feat_dim as u64);
        let vae_path = vendor.join("vae.safetensors");
        write_vae_safetensors(&vae_path);
        let tok_path = vendor.join("tokenizer.json");
        write_tokenizer_json(&tok_path);
        vec![
            complete(dit_path, ArtifactKind::Safetensors),
            complete(te_path, ArtifactKind::HfDir),
            complete(vae_path, ArtifactKind::Safetensors),
            complete(tok_path, ArtifactKind::TokenizerJson),
            complete(dir.join("other-vendor").join("unrelated.bin"), ArtifactKind::Opaque),
        ]
    }

    #[test]
    fn classify_uses_content_not_filename() {
        let dir = tmp("content-not-filename");
        std::fs::create_dir_all(&dir).unwrap();

        let suggestive = dir.join("z-image-turbo-q8_0.gguf");
        // Named like a Z-Image release, but its header says qwen3.
        write_dit_gguf(&suggestive, "qwen3");

        let unreadable = dir.join("z-image-turbo-totally-legit.gguf");
        std::fs::write(&unreadable, b"not a gguf file").unwrap();

        let records = vec![complete(suggestive, ArtifactKind::Gguf), complete(unreadable, ArtifactKind::Gguf)];
        let out = S3ditSpec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "text_encoder".to_string(), Confidence::Declared)], "{out:?}");
    }

    #[test]
    fn classify_recognizes_a_dit_gguf_gated_by_the_discriminator_tensor() {
        let dir = tmp("gguf-dit");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dit.gguf");
        write_dit_gguf(&path, GGUF_ARCHITECTURE);
        let records = vec![complete(path, ArtifactKind::Gguf)];
        let out = S3ditSpec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "dit".to_string(), Confidence::Derived)], "{out:?}");
    }

    /// A real Lumina2 GGUF shares the same `general.architecture` tag but
    /// carries no `cap_embedder.0.weight` - it must never classify as a
    /// Z-Image dit candidate.
    #[test]
    fn classify_rejects_a_shared_tag_gguf_missing_the_discriminator() {
        let dir = tmp("lumina2-decoy");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lumina2.gguf");
        checkpoint::gguf_write::write(
            path.to_str().unwrap(),
            &[("general.architecture".to_string(), checkpoint::gguf::GgufValue::String(GGUF_ARCHITECTURE.to_string()))],
            &[checkpoint::gguf_write::TensorOut { name: "not_cap_embedder".to_string(), shape: vec![1], ty: checkpoint::gguf::GgmlType::F32.id(), data: vec![0u8; 4] }],
            32,
        )
        .unwrap();
        let records = vec![complete(path, ArtifactKind::Gguf)];
        let out = S3ditSpec.classify(&records, dir.as_path());
        assert!(out.is_empty(), "{out:?}");
    }

    #[test]
    fn classify_rejects_a_causal_3d_video_vae_sharing_the_same_tensor_names() {
        let dir = tmp("vae-rank-guard");
        std::fs::create_dir_all(&dir).unwrap();
        let real = dir.join("z-image-vae.safetensors");
        write_vae_safetensors(&real);
        let decoy = dir.join("wan-vae.safetensors");
        write_causal_3d_vae_safetensors(&decoy);
        let records = vec![complete(real, ArtifactKind::Safetensors), complete(decoy, ArtifactKind::Safetensors)];
        let out = S3ditSpec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "vae".to_string(), Confidence::Declared)], "{out:?}");
    }

    #[test]
    fn classify_tokenizer_requires_co_location_with_a_real_text_encoder() {
        let dir = tmp("tokenizer-co-location");
        let mut records = turbo_fixture(&dir);
        let unrelated_dir = dir.join("unrelated-arch");
        std::fs::create_dir_all(&unrelated_dir).unwrap();
        let unrelated_tok = unrelated_dir.join("tokenizer.json");
        write_tokenizer_json(&unrelated_tok);
        records.push(complete(unrelated_tok.clone(), ArtifactKind::TokenizerJson));

        let out = S3ditSpec.classify(&records, dir.as_path());
        let tokenizer_candidates: Vec<_> = out.iter().filter(|(_, role, _)| role == "tokenizer").collect();
        assert_eq!(tokenizer_candidates.len(), 1, "{out:?}");
        let (idx, ..) = tokenizer_candidates[0];
        assert_ne!(records[*idx].path, unrelated_tok, "{out:?}");
    }

    #[test]
    fn a_mismatched_text_encoder_hidden_size_is_rejected_by_validate() {
        let dir = tmp("mismatched-validate");
        std::fs::create_dir_all(&dir).unwrap();
        let dit_path = dir.join("dit.safetensors");
        write_dit_safetensors(&dit_path);
        let te_path = dir.join("Qwen3-mismatch");
        write_qwen3_hfdir(&te_path, 999); // not ZImageConfig::turbo().cap_feat_dim (2560)

        let assembly = Assembly {
            id: "local/s3dit-turbo".to_string(),
            arch: "s3dit".to_string(),
            variant: None,
            roles: BTreeMap::from([("dit".to_string(), dit_path.clone()), ("text_encoder".to_string(), te_path.clone())]),
            provenance: Vec::new(),
        };
        let err = S3ditSpec.validate(&assembly).unwrap_err();
        assert!(err.contains("2560"), "{err}");
        assert!(err.contains("999"), "{err}");
    }

    #[test]
    fn resolves_a_turbo_assembly_from_a_scan_with_no_env_vars() {
        let dir = tmp("full-resolve");
        let records = turbo_fixture(&dir);
        let spec = S3ditSpec;
        let specs: [&dyn ArchSpec; 1] = [&spec];
        let out = resolve("s3dit", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                assert_eq!(a.variant, None);
                assert!(a.roles.contains_key("dit") && a.roles.contains_key("vae") && a.roles.contains_key("text_encoder") && a.roles.contains_key("tokenizer"), "{a:?}");
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }
}
