// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3's [`ArchSpec`]: which on-disk artifacts satisfy which H3 role
//! (`dit`, `video_vae`, `vocoder`, `text_encoder`, `tokenizer`), and the one
//! compatibility gate a chosen assembly must pass - the DiT's `text_dim` must
//! equal the Qwen3-VL text encoder's `hidden_size` (`crate::model`'s
//! `context_embedder` projects the encoder's own hidden states into the
//! packed stream).
//!
//! Every check here is header/config only: a safetensors file's own tensor
//! names and shapes ([`checkpoint::mmap::MmapSafetensors`]) or a component
//! directory's `config.json`. Nothing here ever reads a tensor value or
//! touches a device - see [`ArchSpec::validate`]'s contract. There is no
//! GGUF lane (MiniMax-H3 has no GGUF ecosystem - `crates/arch`'s own
//! `minimaxh3` row carries no `gguf` spelling either).
//!
//! Swedish Embedded AB implements multi-component diffusion-model assembly
//! resolution like this for its clients. If your team needs expertise in
//! wiring large model checkpoints into a model-store resolver, you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::plan::declared_architecture;
use brain_modelstore::resolve::{classify_compound_manifest, classify_tokenizer_role, ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;
use checkpoint::mmap::MmapSafetensors;

pub struct H3Spec;

const ROLES: &[&str] = &["dit", "video_vae", "vocoder", "text_encoder", "tokenizer"];

/// Every tensor name + shape a DiT checkpoint declares - a bare safetensors
/// file or a sharded `transformer/` directory (the release's own layout),
/// mirroring `s3dit::spec::dit_shapes`'s directory handling.
fn dit_shapes(path: &Path) -> Result<Vec<(String, Vec<usize>)>, String> {
    if path.is_dir() {
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(path)
            .map_err(|e| format!("minimaxh3: reading dit dir {}: {e}", path.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(format!("minimaxh3: no .safetensors under {}", path.display()));
        }
        let mut shapes = Vec::new();
        for f in &files {
            let m = MmapSafetensors::open(f).map_err(|e| format!("minimaxh3: opening dit {}: {e}", f.display()))?;
            shapes.extend(m.names().iter().map(|n| (n.clone(), m.shape(n).expect("name came from names()").to_vec())));
        }
        Ok(shapes)
    } else {
        let m = MmapSafetensors::open(path).map_err(|e| format!("minimaxh3: opening dit {}: {e}", path.display()))?;
        Ok(m.names().iter().map(|n| (n.clone(), m.shape(n).expect("name came from names()").to_vec())).collect())
    }
}

/// One opened safetensors file checked against H3's three component
/// discriminators: the DiT's (`adaln_proj.linear.weight` - H3's ONE fused
/// per-modality AdaLN projection, no other architecture in this workspace
/// packs `[6*hidden*3, time_embed_dim]` under that name), the audio VAE
/// decoder's (`dec_in_proj.weight`, `crate::vocoder`'s top-level plain
/// 1x1 latent-input conv), and the video VAE's (`decoder.register_tokens`
/// plus a 5-dim `encoder.conv_in.weight` - the ViT decoder's register row is
/// unique to this causal-3D VAE; Wan's conv-decoder VAE shares the 5-dim
/// encoder names but has no `decoder.register_tokens`). Returns the role the
/// file names, if any.
fn sniff_component(m: &MmapSafetensors) -> Option<&'static str> {
    if m.shape("adaln_proj.linear.weight").is_some_and(|s| s.len() == 2 && s[0] > 0 && s[0] % 18 == 0) {
        return Some("dit");
    }
    if m.shape("dec_in_proj.weight").is_some() {
        return Some("vocoder");
    }
    if m.shape("decoder.register_tokens").is_some_and(|s| s.len() == 3) && m.shape("encoder.conv_in.weight").is_some_and(|s| s.len() == 5) {
        return Some("video_vae");
    }
    None
}

fn classify_safetensors(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    if let Ok(m) = MmapSafetensors::open(&rec.path) {
        if let Some(role) = sniff_component(&m) {
            out.push((idx, role.to_string(), Confidence::Declared));
        }
    }
}

/// A component directory. The release lays every H3 component out as a HF
/// directory of sharded safetensors (`transformer/`, `vae/`, `audio_vae/`),
/// and the text encoder is the released Qwen3-VL checkpoint as downloaded -
/// the same `architectures[0]` value `qwen3vl::spec` classifies by,
/// truncated at an intermediate hidden layer by the caller
/// (`crate::caps::build_text_encoder`), not a different architecture. Sniff
/// the directory's shards first (a directory is classified by what it
/// holds, not by which model it belongs to), and only fall back to the
/// config's `architectures` - the text-encoder lane - when no shard carries
/// an H3 tensor name.
fn classify_hfdir(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    if let Ok(entries) = std::fs::read_dir(&rec.path) {
        let mut files: Vec<std::path::PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        for f in files {
            if let Ok(m) = MmapSafetensors::open(&f) {
                if let Some(role) = sniff_component(&m) {
                    out.push((idx, role.to_string(), Confidence::Declared));
                    return;
                }
            }
        }
    }
    let Ok(bytes) = std::fs::read(rec.path.join("config.json")) else { return };
    let Ok(config) = serde_json::from_slice::<serde_json::Value>(&bytes) else { return };
    if declared_architecture(&config).as_deref() == Some("Qwen3VLForConditionalGeneration") {
        out.push((idx, "text_encoder".to_string(), Confidence::Declared));
    }
}

/// The DiT's `context_embedder.weight` is `[text_dim, hidden]` row-major
/// (PyTorch `nn.Linear` convention, `[out, in]`), so its first dimension IS
/// `text_dim`.
fn dit_text_dim(path: &Path) -> Result<usize, String> {
    let shapes = dit_shapes(path)?;
    shapes
        .iter()
        .find(|(n, _)| n == "context_embedder.weight")
        .map(|(_, s)| s[0])
        .ok_or_else(|| format!("minimaxh3 validate: no context_embedder.weight under {}", path.display()))
}

/// The Qwen3-VL text encoder's own declared `hidden_size`, from its
/// component directory's `config.json`.
fn text_encoder_hidden(path: &Path) -> Result<usize, String> {
    let config_path = path.join("config.json");
    let bytes = std::fs::read(&config_path).map_err(|e| format!("minimaxh3 validate: reading {}: {e}", config_path.display()))?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| format!("minimaxh3 validate: parsing {}: {e}", config_path.display()))?;
    v.get("hidden_size")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| format!("minimaxh3 validate: {} has no hidden_size", config_path.display()))
}

impl ArchSpec for H3Spec {
    fn arch(&self) -> &'static str {
        "minimaxh3"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        // A real conversion step (`crates/loader/src/supply.rs`'s
        // `"minimaxh3"` arm) writes a `brain.manifest.json` (family
        // `"minimaxh3"`) naming each role's component directory directly -
        // `Confidence::Recorded` beats every content-sniffed guess below, so
        // a manifest-declared sharded `transformer/` directory (which no
        // single-shard read can classify: the layer count needs every
        // shard's tensors at once) still resolves unambiguously even when
        // the rest of a shared model store carries other vae/text_encoder-
        // shaped files this recipe never produced.
        classify_compound_manifest(records, inventory_root, "minimaxh3", &mut out);
        for (idx, rec) in records.iter().enumerate() {
            if !rec.usable() {
                continue;
            }
            match rec.kind {
                ArtifactKind::Safetensors => classify_safetensors(idx, rec, &mut out),
                ArtifactKind::HfDir => classify_hfdir(idx, rec, &mut out),
                _ => {}
            }
        }
        let text_encoder_candidates: Vec<&Path> = out.iter().filter(|(_, role, _)| role == "text_encoder").map(|(idx, ..)| records[*idx].path.as_path()).collect();
        classify_tokenizer_role(records, inventory_root, "tokenizer", &text_encoder_candidates, &mut out);
        out
    }

    fn assemble(&self, _chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        // No variant to ask about: the released MiniMax-H3 is the one shipped
        // config (`H3TransformerConfig::real`), and `default_ref` is None
        // because the H3 Community License forbids auto-fetch - see
        // `modelstore::recipe::H3Recipe`'s own doc.
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/minimaxh3".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let dit_path = assembly.roles.get("dit").ok_or("minimaxh3 validate: assembly has no dit role")?;
        let te_path = assembly.roles.get("text_encoder").ok_or("minimaxh3 validate: assembly has no text_encoder role")?;
        let text_dim = dit_text_dim(dit_path)?;
        let hidden = text_encoder_hidden(te_path)?;
        if text_dim != hidden {
            return Err(format!(
                "minimaxh3 validate: dit text_dim={text_dim} does not match text_encoder hidden size={hidden} - dit={}, text_encoder={}",
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
        std::env::temp_dir().join(format!("brain-minimaxh3-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    /// The fixtures carry the REAL tensor names and rank (which is all
    /// `classify`/`validate` read), with deliberately tiny extents - a
    /// full-size `adaln_proj.linear.weight` alone is ~1 GB and every check
    /// here is header-only, so real dimensions would only pay disk and time.
    const TEXT_DIM: u64 = 24;
    const HIDDEN: u64 = 6;

    fn write_dit_safetensors(path: &std::path::Path) {
        // `adaln_proj.linear.weight` is `[6*hidden*MODALITY_NUM,
        // time_embed_dim]` at the real config (6*5376*3 = 96768) - a 1-row
        // stand-in would fail the `% 18` sanity gate, so shape the fixture
        // row count as the real one modulo the width.
        checkpoint::st::save_safetensors(
            path.to_str().unwrap(),
            &[
                ("adaln_proj.linear.weight".to_string(), vec![18, 4], vec![0.0f32; 18 * 4]),
                ("context_embedder.weight".to_string(), vec![TEXT_DIM, HIDDEN], vec![0.0f32; (TEXT_DIM * HIDDEN) as usize]),
            ],
            &serde_json::json!({}),
            None,
        )
        .unwrap();
    }

    /// The two 5-dim convs + the ViT decoder's register row - the tensors
    /// `classify_safetensors`'s video-VAE arm reads.
    fn write_video_vae_safetensors(path: &std::path::Path) {
        checkpoint::st::save_safetensors(
            path.to_str().unwrap(),
            &[
                ("encoder.conv_in.weight".to_string(), vec![1, 1, 1, 1, 1], vec![0.0f32]),
                ("decoder.register_tokens".to_string(), vec![1, 1, 1], vec![0.0f32]),
            ],
            &serde_json::json!({}),
            None,
        )
        .unwrap();
    }

    fn write_vocoder_safetensors(path: &std::path::Path) {
        checkpoint::st::save_safetensors(
            path.to_str().unwrap(),
            &[("dec_in_proj.weight".to_string(), vec![4, 2, 1], vec![0.0f32; 8])],
            &serde_json::json!({}),
            None,
        )
        .unwrap();
    }

    fn write_qwen3vl_hfdir(dir: &std::path::Path, hidden: u64) {
        std::fs::create_dir_all(dir).unwrap();
        // `vocab_size` matching the tokenizer fixture's 100 entries is what
        // `classify_tokenizer_role` pairs the tokenizer on (within 5%).
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"architectures": ["Qwen3VLForConditionalGeneration"], "hidden_size": hidden, "vocab_size": 100})).unwrap()).unwrap();
    }

    fn write_tokenizer_json(path: &std::path::Path) {
        let vocab: serde_json::Map<String, serde_json::Value> = (0..100).map(|i| (format!("t{i}"), serde_json::json!(i))).collect();
        std::fs::write(path, serde_json::to_vec(&serde_json::json!({"version": "1.0", "model": {"vocab": vocab}, "added_tokens": []})).unwrap()).unwrap();
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    /// One full H3 root, laid out exactly as the release downloads it.
    fn h3_fixture(dir: &std::path::Path) -> Vec<ArtifactRecord> {
        let vendor = dir.join("MiniMaxAI");
        std::fs::create_dir_all(&vendor).unwrap();
        let dit_path = vendor.join("transformer");
        std::fs::create_dir_all(&dit_path).unwrap();
        write_dit_safetensors(&dit_path.join("diffusion_pytorch_model.safetensors"));
        let te_path = vendor.join("text_encoder");
        write_qwen3vl_hfdir(&te_path, TEXT_DIM as u64);
        let vae_path = vendor.join("vae.safetensors");
        write_video_vae_safetensors(&vae_path);
        let voc_path = vendor.join("audio_vae.safetensors");
        write_vocoder_safetensors(&voc_path);
        let tok_path = vendor.join("tokenizer.json");
        write_tokenizer_json(&tok_path);
        vec![
            complete(dit_path, ArtifactKind::HfDir),
            complete(te_path, ArtifactKind::HfDir),
            complete(vae_path, ArtifactKind::Safetensors),
            complete(voc_path, ArtifactKind::Safetensors),
            complete(tok_path, ArtifactKind::TokenizerJson),
            // A store never holds just one package: this unrelated record
            // keeps `resolve`'s inferred common root at the STORE (as a
            // multi-package scan would), which is the root every
            // vendor-relative check here is written against.
            complete(dir.join("other-vendor").join("unrelated.bin"), ArtifactKind::Opaque),
        ]
    }

    #[test]
    fn classify_uses_content_not_filename() {
        let dir = tmp("content-not-filename");
        std::fs::create_dir_all(&dir).unwrap();
        // Named like an H3 DiT shard, but carries no H3 tensor.
        let decoy = dir.join("transformer.safetensors");
        checkpoint::st::save_safetensors(decoy.to_str().unwrap(), &[("unrelated.weight".to_string(), vec![1], vec![0.0f32])], &serde_json::json!({}), None).unwrap();
        let records = vec![complete(decoy, ArtifactKind::Safetensors)];
        assert!(H3Spec.classify(&records, dir.as_path()).is_empty());
    }

    /// A Wan-style causal-3D video VAE (5-dim encoder convs, conv-based
    /// decoder, no register tokens) must never classify as this VAE.
    #[test]
    fn classify_rejects_a_conv_decoder_video_vae_sharing_the_encoder_names() {
        let dir = tmp("vae-decoy");
        std::fs::create_dir_all(&dir).unwrap();
        let decoy = dir.join("wan-vae.safetensors");
        checkpoint::st::save_safetensors(
            decoy.to_str().unwrap(),
            &[("encoder.conv_in.weight".to_string(), vec![96, 3, 3, 3, 3], vec![0.0f32; 96 * 3 * 3 * 3 * 3])],
            &serde_json::json!({}),
            None,
        )
        .unwrap();
        let records = vec![complete(decoy, ArtifactKind::Safetensors)];
        assert!(H3Spec.classify(&records, dir.as_path()).is_empty());
    }

    #[test]
    fn resolves_a_minimaxh3_assembly_from_a_scan_with_no_env_vars() {
        let dir = tmp("full-resolve");
        let records = h3_fixture(&dir);
        let specs: [&dyn ArchSpec; 1] = [&H3Spec];
        let out = resolve("minimaxh3", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                assert_eq!(a.variant, None);
                for role in ROLES {
                    assert!(a.roles.contains_key(*role), "missing role {role}: {a:?}");
                }
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn a_mismatched_text_encoder_hidden_size_is_rejected_by_validate() {
        let dir = tmp("mismatched-validate");
        std::fs::create_dir_all(&dir).unwrap();
        let dit_path = dir.join("transformer");
        std::fs::create_dir_all(&dit_path).unwrap();
        write_dit_safetensors(&dit_path.join("diffusion_pytorch_model.safetensors"));
        let te_path = dir.join("Qwen3VL-mismatch");
        write_qwen3vl_hfdir(&te_path, HIDDEN + 1); // not the DiT fixture's text_dim (24)

        let assembly = Assembly {
            id: "local/minimaxh3".to_string(),
            arch: "minimaxh3".to_string(),
            variant: None,
            roles: BTreeMap::from([("dit".to_string(), dit_path.clone()), ("text_encoder".to_string(), te_path.clone())]),
            provenance: Vec::new(),
        };
        let err = H3Spec.validate(&assembly).unwrap_err();
        assert!(err.contains("24"), "{err}");
        assert!(err.contains("7"), "{err}");
    }

    /// `supply.rs`'s `"minimaxh3"` convert arm's own on-disk shape: a
    /// `brain.manifest.json` (family `"minimaxh3"`) naming the component
    /// directories - `Confidence::Recorded` carries the assembly even where
    /// a single shard alone could not be content-classified, and beats
    /// unrelated store noise.
    #[test]
    fn resolves_a_manifest_declared_assembly_over_unrelated_store_noise() {
        let dir = tmp("manifest-sharded-dit");
        let repo = dir.join("MiniMaxAI").join("MiniMax-H3");
        std::fs::create_dir_all(&repo).unwrap();

        // A sharded DiT laid out the way the release ships (`transformer/` with
        // several `.safetensors` shards): the manifest names the directory as
        // a whole at Recorded confidence, which beats every content sniff.
        let dit_dir = repo.join("transformer");
        std::fs::create_dir_all(&dit_dir).unwrap();
        checkpoint::st::save_safetensors(
            dit_dir.join("shard1.safetensors").to_str().unwrap(),
            &[("adaln_proj.linear.weight".to_string(), vec![18, 4], vec![0.0f32; 18 * 4])],
            &serde_json::json!({}),
            None,
        )
        .unwrap();
        checkpoint::st::save_safetensors(
            dit_dir.join("shard2.safetensors").to_str().unwrap(),
            &[("context_embedder.weight".to_string(), vec![TEXT_DIM, HIDDEN], vec![0.0f32; (TEXT_DIM * HIDDEN) as usize])],
            &serde_json::json!({}),
            None,
        )
        .unwrap();

        let te_dir = repo.join("text_encoder");
        write_qwen3vl_hfdir(&te_dir, TEXT_DIM as u64);
        // Every model component is the release's own role DIRECTORY
        // (`convert_h3`'s manifest names these same rel paths) - `read_model_dir`
        // resolves the shards inside each one at load time.
        let vae_path = repo.join("vae");
        std::fs::create_dir_all(&vae_path).unwrap();
        write_video_vae_safetensors(&vae_path.join("model.safetensors"));
        let voc_path = repo.join("audio_vae");
        std::fs::create_dir_all(&voc_path).unwrap();
        write_vocoder_safetensors(&voc_path.join("model.safetensors"));
        let tok_dir = repo.join("tokenizer");
        std::fs::create_dir_all(&tok_dir).unwrap();
        write_tokenizer_json(&tok_dir.join("tokenizer.json"));

        std::fs::write(
            repo.join(brain_modelstore::MANIFEST_FILE),
            serde_json::to_vec(&brain_modelstore::CompoundManifest {
                id: "MiniMaxAI/MiniMax-H3".to_string(),
                family: "minimaxh3".to_string(),
                roles: [
                    ("dit", "transformer"),
                    ("video_vae", "vae"),
                    ("vocoder", "audio_vae"),
                    ("text_encoder", "text_encoder"),
                    ("tokenizer", "tokenizer"),
                ]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            })
            .unwrap(),
        )
        .unwrap();

        // Unrelated store noise: a real (differently-shaped) text encoder
        // this manifest never declared, at Declared confidence only.
        let noise_te = dir.join("other-vendor").join("Qwen3VL-8B");
        std::fs::create_dir_all(noise_te.parent().unwrap()).unwrap();
        write_qwen3vl_hfdir(&noise_te, TEXT_DIM as u64);

        let mut records = vec![
            complete(dit_dir.clone(), ArtifactKind::HfDir),
            complete(te_dir, ArtifactKind::HfDir),
            complete(vae_path, ArtifactKind::HfDir),
            complete(voc_path, ArtifactKind::HfDir),
            complete(tok_dir, ArtifactKind::HfDir),
            complete(noise_te, ArtifactKind::HfDir),
        ];
        let spec = H3Spec;
        let specs: [&dyn ArchSpec; 1] = [&spec];
        records.sort_by(|a, b| a.path.cmp(&b.path));
        let out = resolve("minimaxh3", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                // The manifest's own recorded roles win over every content
                // sniff - including the other vendor's equally-matching
                // Qwen3-VL directory - and the whole sharded `transformer/`
                // directory is named, not either lone shard.
                assert_eq!(a.roles.get("dit").map(std::path::PathBuf::as_path), Some(dit_dir.as_path()));
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }
}