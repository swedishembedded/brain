// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Wan2.1's [`ArchSpec`]: which on-disk artifacts satisfy which Wan role
//! (`dit`, `vae`, `text_encoder`, `tokenizer`), and the one compatibility
//! gate a chosen assembly must pass - the DiT's `text_dim` must equal the
//! umT5 text encoder's `d_model`.
//!
//! Every check here is header/config only: a GGUF's KV metadata and tensor
//! shapes ([`checkpoint::gguf::MmapGguf`]), a safetensors file's own tensor
//! shapes ([`checkpoint::mmap::MmapSafetensors`]), or a `torch.save`
//! checkpoint's own tensor names/shapes ([`checkpoint::torchpt::shapes`],
//! mmap'd, no tensor data read) - mirroring `crate::import::dit_manifest`'s
//! own "shapes, never values" discipline. Nothing here ever touches a
//! device - see [`ArchSpec::validate`]'s contract.
//!
//! The VAE and text encoder ship as native `torch.save` `.pth` files
//! (`Wan2.1_VAE.pth`, `models_t5_umt5-xxl-enc-bf16.pth`) rather than
//! safetensors/GGUF, so their classification and the tokenizer's own
//! vocab-compatibility check both read a `.pth`'s tensor shapes directly
//! instead of the HfDir/GGUF paths `brain_modelstore::resolve::
//! checkpoint_vocab_size` already knows.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{tokenizer_vocab_count, vendor_dir, vocab_is_compatible, ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;
use checkpoint::gguf::MmapGguf;
use checkpoint::mmap::MmapSafetensors;

use crate::import::{dit_config_from_shapes, GGUF_ARCHITECTURE};

pub struct WanSpec;

const ROLES: &[&str] = &["dit", "vae", "text_encoder", "tokenizer"];

/// Every tensor name + shape a DiT checkpoint declares, read from whichever
/// header it actually has - a GGUF's own header, or a safetensors file's.
fn dit_shapes(path: &Path) -> Result<Vec<(String, Vec<usize>)>, String> {
    if path.extension().is_some_and(|e| e == "gguf") {
        let g = MmapGguf::open(&path.to_string_lossy()).map_err(|e| format!("wan: opening dit {}: {e}", path.display()))?;
        Ok(g.all_shapes())
    } else {
        let m = MmapSafetensors::open(path).map_err(|e| format!("wan: opening dit {}: {e}", path.display()))?;
        Ok(m.names().iter().map(|n| (n.clone(), m.shape(n).expect("name came from names()").to_vec())).collect())
    }
}

fn classify_gguf(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    let Ok(g) = MmapGguf::open(&rec.path.to_string_lossy()) else { return };
    let Some(arch) = g.kv().get("general.architecture").and_then(|v| v.as_str()) else { return };
    // Unlike flux2's shared "flux" tag, "wan" is not shared with any other
    // architecture (`crate::import::GGUF_ARCHITECTURE`'s own doc) - a full
    // shape match against a known Wan variant still gates it, exactly like
    // every other GGUF arm in this workspace, so a same-tagged but malformed
    // or unknown-shaped file never classifies as a candidate.
    if arch != GGUF_ARCHITECTURE {
        return;
    }
    if dit_config_from_shapes(&g.all_shapes()).is_ok() {
        out.push((idx, "dit".to_string(), Confidence::Declared));
    }
}

fn classify_safetensors(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    let Ok(m) = MmapSafetensors::open(&rec.path) else { return };
    if m.shape("patch_embedding.weight").is_none() {
        return;
    }
    let shapes: Vec<(String, Vec<usize>)> = m.names().iter().map(|n| (n.clone(), m.shape(n).expect("name came from names()").to_vec())).collect();
    if dit_config_from_shapes(&shapes).is_ok() {
        out.push((idx, "dit".to_string(), Confidence::Declared));
    }
}

/// A causal-3D-conv VAE (rank-5 `[out, in, kt, kh, kw]`) is Wan's own real
/// checkpoint shape (`crate::vae3d::WanVaeConfig::tensor_manifest`'s
/// `encoder.conv1`/`decoder.conv1`), and the RANK is what tells it apart from
/// a 2D image VAE sharing the same generic autoencoder tensor NAMES - the
/// same distinguishing signal `flux2::spec`'s own VAE classifier reads, in
/// the other direction.
fn is_wan_vae(shapes: &[(String, Vec<usize>)]) -> bool {
    let shape_of = |name: &str| shapes.iter().find(|(n, _)| n == name).map(|(_, s)| s.as_slice());
    let is_5d_conv = |name: &str| shape_of(name).is_some_and(|s| s.len() == 5);
    is_5d_conv("encoder.conv1.weight") && is_5d_conv("decoder.conv1.weight")
}

/// umT5's native (Wan's own) checkpoint shape: a `[vocab, d_model]` embedding
/// table plus at least one block's own attention query projection
/// (`crate::import::wan_to_brain`'s source names) - real, checkpoint-specific
/// structure, not a guess from the filename.
fn is_wan_text_encoder(shapes: &[(String, Vec<usize>)]) -> bool {
    let has_embedding = shapes.iter().any(|(n, s)| n == "token_embedding.weight" && s.len() == 2);
    let has_block = shapes.iter().any(|(n, _)| n.starts_with("blocks.") && n.ends_with(".attn.q.weight"));
    has_embedding && has_block
}

fn classify_pth(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    let Ok(shapes) = checkpoint::torchpt::shapes(&rec.path.to_string_lossy()) else { return };
    if is_wan_vae(&shapes) {
        out.push((idx, "vae".to_string(), Confidence::Declared));
    } else if is_wan_text_encoder(&shapes) {
        out.push((idx, "text_encoder".to_string(), Confidence::Declared));
    }
}

/// The umT5 text encoder's own declared vocabulary: `token_embedding.weight`'s
/// first dimension, read from the `.pth`'s tensor shapes alone.
/// `brain_modelstore::resolve::checkpoint_vocab_size` only knows GGUF/HfDir
/// (a `config.json`'s `vocab_size`, or a GGUF's `tokenizer.ggml.tokens`), so a
/// `.pth` dependency candidate needs this crate's own reader instead - the
/// same discipline that function's own doc anticipates for an architecture
/// whose tokenizer-shaped role does not fit its orchestration.
fn wan_text_encoder_vocab_size(path: &Path) -> Option<usize> {
    let shapes = checkpoint::torchpt::shapes(&path.to_string_lossy()).ok()?;
    shapes.iter().find(|(n, _)| n == "token_embedding.weight").and_then(|(_, s)| s.first().copied())
}

/// The tokenizer role, classified against the `.pth` text-encoder candidates
/// this same `classify` call already found - mirrors
/// [`brain_modelstore::resolve::classify_tokenizer_role`]'s own vendor +
/// vocab-compatibility loop exactly, substituting
/// [`wan_text_encoder_vocab_size`] for that function's `checkpoint_vocab_size`
/// (which cannot read a `.pth`).
fn classify_tokenizer(records: &[ArtifactRecord], root: &Path, te_candidates: &[&Path], out: &mut Vec<(usize, String, Confidence)>) {
    let vendor_dirs: std::collections::BTreeSet<std::path::PathBuf> = te_candidates.iter().filter_map(|p| vendor_dir(p, root)).collect();
    for (idx, rec) in records.iter().enumerate() {
        if !rec.usable() || rec.kind != ArtifactKind::TokenizerJson {
            continue;
        }
        let Ok(bytes) = std::fs::read(&rec.path) else { continue };
        let Some(tok_count) = tokenizer_vocab_count(&bytes) else { continue };
        let Some(vendor) = vendor_dir(&rec.path, root) else { continue };
        if !vendor_dirs.contains(&vendor) {
            continue;
        }
        let compatible = te_candidates
            .iter()
            .filter(|p| vendor_dir(p, root).as_deref() == Some(vendor.as_path()))
            .any(|p| wan_text_encoder_vocab_size(p).is_some_and(|v| vocab_is_compatible(tok_count, v)));
        if compatible {
            out.push((idx, "tokenizer".to_string(), Confidence::Declared));
        }
    }
}

impl ArchSpec for WanSpec {
    fn arch(&self) -> &'static str {
        "wan"
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
                ArtifactKind::Safetensors => classify_safetensors(idx, rec, &mut out),
                ArtifactKind::Pth => classify_pth(idx, rec, &mut out),
                _ => {}
            }
        }
        let text_encoder_candidates: Vec<&Path> = out.iter().filter(|(_, role, _)| role == "text_encoder").map(|(idx, ..)| records[*idx].path.as_path()).collect();
        classify_tokenizer(records, inventory_root, &text_encoder_candidates, &mut out);
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        let dit_idx = *chosen.get("dit").ok_or("wan assemble: no dit chosen")?;
        let dit_rec = &records[dit_idx];
        let shapes = dit_shapes(&dit_rec.path)?;
        let cfg = dit_config_from_shapes(&shapes)?;
        // Unlike flux2's klein-vs-base, nothing about Wan's variant is
        // unrecoverable from the weights: `dit_config_from_shapes` already
        // matches the exact (dim, layer count) pair against a real, named
        // variant, or fails - so a resolved dit always names a full variant,
        // never an ambiguity.
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: format!("local/wan-{}", cfg.name), variant: Some(cfg.name.to_string()) }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let dit_path = assembly.roles.get("dit").ok_or("wan validate: assembly has no dit role")?;
        let te_path = assembly.roles.get("text_encoder").ok_or("wan validate: assembly has no text_encoder role")?;
        let shapes = dit_shapes(dit_path)?;
        let cfg = dit_config_from_shapes(&shapes)?;
        let te_shapes = checkpoint::torchpt::shapes(&te_path.to_string_lossy()).map_err(|e| format!("wan validate: opening text_encoder {}: {e}", te_path.display()))?;
        let d_model = te_shapes
            .iter()
            .find(|(n, _)| n == "token_embedding.weight")
            .and_then(|(_, s)| s.get(1).copied())
            .ok_or_else(|| format!("wan validate: {} has no token_embedding.weight", te_path.display()))?;
        if d_model != cfg.text_dim {
            return Err(format!(
                "wan validate: dit text_dim={} does not match text_encoder d_model={d_model} - dit={}, text_encoder={}",
                cfg.text_dim,
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
        std::env::temp_dir().join(format!("brain-wan-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    /// Only the tensors [`dit_config_from_shapes`] actually reads (mirrors
    /// `import.rs`'s own `shapes_of`) at real t2v-1.3B dimensions, NOT the
    /// full 825-tensor manifest.
    fn write_dit_safetensors(path: &std::path::Path) {
        let cfg = crate::config::WanConfig::t2v_1_3b();
        let (dim, in_ch, pt, ph, pw) = (cfg.dim, cfg.in_channels, cfg.patch_size.0, cfg.patch_size.1, cfg.patch_size.2);
        let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = vec![("patch_embedding.weight".to_string(), vec![dim as u64, in_ch as u64, pt as u64, ph as u64, pw as u64], vec![0.0f32; dim * in_ch * pt * ph * pw])];
        // dit_config_from_shapes counts the highest `blocks.<l>.` index seen -
        // one marker tensor per block is enough, no real block content needed.
        for l in 0..cfg.num_layers {
            tensors.push((format!("blocks.{l}.marker"), vec![1], vec![0.0f32]));
        }
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &serde_json::json!({}), None).unwrap();
    }

    fn write_dit_gguf(path: &std::path::Path) {
        let cfg = crate::config::WanConfig::t2v_1_3b();
        let (dim, in_ch, pt, ph, pw) = (cfg.dim, cfg.in_channels, cfg.patch_size.0, cfg.patch_size.1, cfg.patch_size.2);
        let mut tensors = vec![checkpoint::gguf_write::TensorOut {
            name: "patch_embedding.weight".to_string(),
            shape: vec![dim, in_ch, pt, ph, pw],
            ty: checkpoint::gguf::GgmlType::F32.id(),
            data: vec![0u8; dim * in_ch * pt * ph * pw * 4],
        }];
        for l in 0..cfg.num_layers {
            tensors.push(checkpoint::gguf_write::TensorOut { name: format!("blocks.{l}.marker"), shape: vec![1], ty: checkpoint::gguf::GgmlType::F32.id(), data: vec![0u8; 4] });
        }
        checkpoint::gguf_write::write(path.to_str().unwrap(), &[("general.architecture".to_string(), checkpoint::gguf::GgufValue::String(GGUF_ARCHITECTURE.to_string()))], &tensors, 32).unwrap();
    }

    /// A GGUF sharing the same `general.architecture` tag but shaped nothing
    /// like a real Wan transformer must never classify - the same
    /// content-not-tag discipline `dit_config_from_shapes` itself enforces.
    fn write_non_wan_gguf(path: &std::path::Path) {
        checkpoint::gguf_write::write(
            path.to_str().unwrap(),
            &[("general.architecture".to_string(), checkpoint::gguf::GgufValue::String(GGUF_ARCHITECTURE.to_string()))],
            &[checkpoint::gguf_write::TensorOut { name: "not_a_wan_tensor".to_string(), shape: vec![1], ty: checkpoint::gguf::GgmlType::F32.id(), data: vec![0u8; 4] }],
            32,
        )
        .unwrap();
    }

    /// Only the two tensors [`is_wan_vae`] actually checks, at a tiny (not
    /// real) channel width - real dims would write hundreds of MB of zeros.
    fn write_vae_pth(path: &std::path::Path) {
        checkpoint::torchpt_write::write(
            path.to_str().unwrap(),
            &[
                checkpoint::torchpt_write::TensorOut { name: "encoder.conv1.weight".to_string(), shape: vec![4, 3, 3, 3, 3], data: vec![0.0; 4 * 3 * 3 * 3 * 3] },
                checkpoint::torchpt_write::TensorOut { name: "decoder.conv1.weight".to_string(), shape: vec![4, 4, 3, 3, 3], data: vec![0.0; 4 * 4 * 3 * 3 * 3] },
            ],
        )
        .unwrap();
    }

    /// Toy vocab/d_model every fixture writer below agrees on, so
    /// [`vocab_is_compatible`]/[`WanSpec::validate`] find a real match.
    const TOY_VOCAB: usize = 100;

    fn write_t5_pth(path: &std::path::Path, d_model: usize) {
        checkpoint::torchpt_write::write(
            path.to_str().unwrap(),
            &[
                checkpoint::torchpt_write::TensorOut { name: "token_embedding.weight".to_string(), shape: vec![TOY_VOCAB, d_model], data: vec![0.0; TOY_VOCAB * d_model] },
                checkpoint::torchpt_write::TensorOut { name: "blocks.0.attn.q.weight".to_string(), shape: vec![d_model, d_model], data: vec![0.0; d_model * d_model] },
            ],
        )
        .unwrap();
    }

    fn write_tokenizer_json(path: &std::path::Path) {
        let vocab: serde_json::Map<String, serde_json::Value> = (0..TOY_VOCAB).map(|i| (format!("t{i}"), serde_json::json!(i))).collect();
        std::fs::write(path, serde_json::to_vec(&serde_json::json!({"version": "1.0", "model": {"vocab": vocab}, "added_tokens": []})).unwrap()).unwrap();
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    /// A full, unambiguous set of records for a t2v-1.3B assembly, and a
    /// second vendor's unrelated file so `resolve()`'s own root inference
    /// (the deepest common ancestor) lands on `dir`, not the single vendor
    /// directory - see `flux2::spec`'s identical fixture doc.
    fn t2v_1_3b_fixture(dir: &std::path::Path) -> Vec<ArtifactRecord> {
        let vendor = dir.join("Wan-AI");
        std::fs::create_dir_all(&vendor).unwrap();
        let dit_path = vendor.join("diffusion_pytorch_model.safetensors");
        write_dit_safetensors(&dit_path);
        let vae_path = vendor.join("Wan2.1_VAE.pth");
        write_vae_pth(&vae_path);
        let te_path = vendor.join("models_t5_umt5-xxl-enc-bf16.pth");
        // Real t2v-1.3B text_dim (4096) - `WanSpec::validate`'s own gate must
        // pass for this fixture to reach `Resolved`.
        write_t5_pth(&te_path, crate::config::WanConfig::t2v_1_3b().text_dim);
        let tok_path = vendor.join("tokenizer.json");
        write_tokenizer_json(&tok_path);
        vec![
            complete(dit_path, ArtifactKind::Safetensors),
            complete(vae_path, ArtifactKind::Pth),
            complete(te_path, ArtifactKind::Pth),
            complete(tok_path, ArtifactKind::TokenizerJson),
            complete(dir.join("other-vendor").join("unrelated.bin"), ArtifactKind::Opaque),
        ]
    }

    #[test]
    fn classify_uses_content_not_filename() {
        let dir = tmp("content-not-filename");
        std::fs::create_dir_all(&dir).unwrap();

        // Named exactly like a real Wan DiT release, but its header/tensors
        // say otherwise - proving classification reads content, not the name.
        let suggestive = dir.join("wan2.1-t2v-1.3b-q8_0.gguf");
        write_non_wan_gguf(&suggestive);

        let records = vec![complete(suggestive.clone(), ArtifactKind::Gguf)];
        let out = WanSpec.classify(&records, dir.as_path());
        assert!(out.is_empty(), "a same-tagged but wrongly-shaped GGUF must never classify: {out:?}");
    }

    #[test]
    fn classify_recognizes_a_dit_gguf_by_shape_not_only_its_architecture_tag() {
        let dir = tmp("gguf-dit");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dit.gguf");
        write_dit_gguf(&path);
        let records = vec![complete(path, ArtifactKind::Gguf)];
        let out = WanSpec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "dit".to_string(), Confidence::Declared)], "{out:?}");
    }

    /// A causal 3D video VAE (Wan's own) and a 2D image VAE sharing the exact
    /// same tensor NAMES (`flux2`'s own release) must not cross-classify -
    /// the conv rank is the real, always-present signal that tells them
    /// apart, in both directions.
    #[test]
    fn classify_rejects_a_2d_image_vae_sharing_the_same_tensor_names() {
        let dir = tmp("vae-rank-guard");
        std::fs::create_dir_all(&dir).unwrap();
        let real = dir.join("Wan2.1_VAE.pth");
        write_vae_pth(&real);
        let decoy = dir.join("flux-vae.pth");
        checkpoint::torchpt_write::write(
            decoy.to_str().unwrap(),
            &[
                checkpoint::torchpt_write::TensorOut { name: "encoder.conv1.weight".to_string(), shape: vec![4, 3, 3, 3], data: vec![0.0; 4 * 3 * 3 * 3] },
                checkpoint::torchpt_write::TensorOut { name: "decoder.conv1.weight".to_string(), shape: vec![4, 4, 3, 3], data: vec![0.0; 4 * 4 * 3 * 3] },
            ],
        )
        .unwrap();
        let records = vec![complete(real.clone(), ArtifactKind::Pth), complete(decoy, ArtifactKind::Pth)];
        let out = WanSpec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "vae".to_string(), Confidence::Declared)], "{out:?}");
    }

    /// A tokenizer belonging to an unrelated architecture, sitting nowhere
    /// near any real Wan component, must never classify just because it is
    /// valid tokenizer JSON somewhere on disk.
    #[test]
    fn classify_tokenizer_requires_co_location_with_a_real_text_encoder() {
        let dir = tmp("tokenizer-co-location");
        let mut records = t2v_1_3b_fixture(&dir);
        let unrelated_dir = dir.join("unrelated-arch");
        std::fs::create_dir_all(&unrelated_dir).unwrap();
        let unrelated_tok = unrelated_dir.join("tokenizer.json");
        write_tokenizer_json(&unrelated_tok);
        records.push(complete(unrelated_tok.clone(), ArtifactKind::TokenizerJson));

        let out = WanSpec.classify(&records, dir.as_path());
        let tokenizer_candidates: Vec<_> = out.iter().filter(|(_, role, _)| role == "tokenizer").collect();
        assert_eq!(tokenizer_candidates.len(), 1, "{out:?}");
        let (idx, ..) = tokenizer_candidates[0];
        assert_ne!(records[*idx].path, unrelated_tok, "{out:?}");
    }

    /// The mismatch the DiT's own `text_dim` and the text encoder's real
    /// `d_model` gate - caught before any GPU work, by name.
    #[test]
    fn a_mismatched_text_encoder_d_model_is_rejected_by_validate() {
        let dir = tmp("mismatched-validate");
        std::fs::create_dir_all(&dir).unwrap();
        let dit_path = dir.join("dit.safetensors");
        write_dit_safetensors(&dit_path);
        let te_path = dir.join("t5.pth");
        write_t5_pth(&te_path, 999); // not WanConfig::t2v_1_3b().text_dim (4096)

        let assembly = Assembly {
            id: "local/wan-t2v-1.3B".to_string(),
            arch: "wan".to_string(),
            variant: Some("t2v-1.3B".to_string()),
            roles: BTreeMap::from([("dit".to_string(), dit_path.clone()), ("text_encoder".to_string(), te_path.clone())]),
            provenance: Vec::new(),
        };
        let err = WanSpec.validate(&assembly).unwrap_err();
        assert!(err.contains("4096"), "{err}");
        assert!(err.contains("999"), "{err}");
    }

    /// The whole point: a t2v-1.3B assembly resolves cleanly from a scan,
    /// with the variant read straight off the DiT's own shapes - never
    /// ambiguous, unlike flux2's klein-vs-base.
    #[test]
    fn resolves_a_t2v_1_3b_assembly_from_a_scan_with_no_env_vars() {
        let dir = tmp("full-resolve");
        let records = t2v_1_3b_fixture(&dir);
        let spec = WanSpec;
        let specs: [&dyn ArchSpec; 1] = [&spec];
        let out = resolve("wan", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                assert_eq!(a.variant.as_deref(), Some("t2v-1.3B"));
                assert!(a.roles.contains_key("dit") && a.roles.contains_key("vae") && a.roles.contains_key("text_encoder") && a.roles.contains_key("tokenizer"), "{a:?}");
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }
}
