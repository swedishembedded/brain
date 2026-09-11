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
///
/// Also recognizes a bare vendor-flat DiT checkpoint (BFL's own
/// `flux2-dev.safetensors` release: the full, undistilled "dev"/base model,
/// shipped as one plain safetensors file, no GGUF at all) via
/// `crate::import::dit_shapes` - the same canonicalizing shape reader
/// [`dit_config_from_path`]/`validate()`'s `dit_context_in_dim` use, so a
/// diffusers-renamed release classifies exactly as consistently as a
/// BFL-named one, not through a second, narrower hand-rolled reader.
fn classify_safetensors(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    let Ok(m) = checkpoint::mmap::MmapSafetensors::open(rec.path.to_string_lossy().as_ref()) else { return };
    // `decoder`/`encoder`.conv_in.weight are generic autoencoder tensor
    // names, not FLUX.2-specific - a real, unrelated architecture's causal
    // 3D VIDEO vae (Wan's release) carries the exact same two names. The
    // conv kernel's shape RANK is real structure that tells them apart: a
    // 2D image vae is `[out,in,kh,kw]` (4 dims), a causal 3D video vae adds
    // a leading temporal axis, `[out,in,kt,kh,kw]` (5 dims).
    let is_2d_conv = |name: &str| m.shape(name).is_some_and(|s| s.len() == 4);
    if is_2d_conv("decoder.conv_in.weight") && is_2d_conv("encoder.conv_in.weight") {
        out.push((idx, "vae".to_string(), Confidence::Declared));
        return;
    }
    let Ok(shapes) = crate::import::dit_shapes(&rec.path.to_string_lossy()) else { return };
    // Shared with FLUX.1 (`classify_gguf`'s `"flux"` arm's own doc explains
    // why: `dit_config_from_shapes` alone cannot tell the two apart, only
    // `img_in.weight`'s channel count can), so this can never rise above
    // Derived - the same rule, over canonicalized shapes instead of a GGUF's.
    let in_channels = shapes.iter().find(|(n, _)| n == "img_in.weight").and_then(|(_, s)| s.get(1)).copied();
    if in_channels != Some(flux2_in_channels()) {
        return;
    }
    if dit_config_from_shapes(&shapes).is_ok() {
        out.push((idx, "dit".to_string(), Confidence::Derived));
    }
}

/// `txt_in.weight`'s second dimension - a pure header shape lookup, zero
/// tensor bytes read, over whichever format the dit path actually is (see
/// `crate::import::dit_shapes`'s own doc: GGUF, a bare vendor-flat
/// safetensors file, or a diffusers-renamed sharded directory).
fn dit_context_in_dim(path: &Path) -> Result<usize, String> {
    let shapes = crate::import::dit_shapes(&path.to_string_lossy()).map_err(|e| format!("flux2 validate: opening dit {}: {e}", path.display()))?;
    let shape = shapes.iter().find(|(n, _)| n == "txt_in.weight").map(|(_, s)| s.as_slice()).ok_or_else(|| format!("flux2 validate: {} has no txt_in.weight", path.display()))?;
    shape.get(1).copied().ok_or_else(|| format!("flux2 validate: {} txt_in.weight has fewer than 2 dimensions", path.display()))
}

/// [`crate::import::sniff_dit_size`] over `dit_path` - the one place
/// [`ArchSpec::assemble`] decides a chosen dit's shape class, so a GGUF, a
/// bare safetensors file, and a diffusers-renamed sharded `transformer/`
/// directory (BFL's own release shape for the undistilled "dev"/base model)
/// are all handled the same way an explicit `--dit <path>` override forces
/// one of, rather than a second, narrower reader that only knew the first two.
fn dit_config_from_path(dit_path: &Path) -> Result<crate::import::DitSize, String> {
    crate::import::sniff_dit_size(&dit_path.to_string_lossy())
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

    fn classify(&self, records: &[ArtifactRecord], inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        // Pass 1: dit/vae/text_encoder - every one of these reads real
        // content (a GGUF header's KV/shapes, or a config.json's own
        // declared fields), never a record's own filename.
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
        // Pass 2: tokenizer, using pass 1's own results (the text_encoder
        // candidates) as the real signal a bare "is this valid JSON" check
        // cannot provide on its own - shared with every other architecture
        // that has a tokenizer-shaped role, not flux2-specific.
        let text_encoder_candidates: Vec<&Path> = out.iter().filter(|(_, role, _)| role == "text_encoder").map(|(idx, ..)| records[*idx].path.as_path()).collect();
        brain_modelstore::resolve::classify_tokenizer_role(records, inventory_root, "tokenizer", &text_encoder_candidates, &mut out);
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord], overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        let dit_idx = *chosen.get("dit").ok_or("flux2 assemble: no dit chosen")?;
        let dit_rec = &records[dit_idx];
        let size = dit_config_from_path(&dit_rec.path)?;
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

    /// A fixture directory that deletes itself when the test ends.
    ///
    /// These fixtures are real GGUF/safetensors files at REAL klein
    /// dimensions - `txt_in.weight` alone is `[4096, 12288]` fp32, ~200 MB -
    /// and the directory name carries the process id, so nothing ever reused
    /// or overwrote them. Every run of this module left its fixtures in the
    /// system temp directory forever; on a machine that runs the suite often,
    /// that is tens of gigabytes and eventually a build that fails with "no
    /// space left on device". Dropping on the way out (panic included) is the
    /// only version of this that stays correct while the tests keep their
    /// per-run isolation.
    struct TmpDir(PathBuf);

    impl TmpDir {
        fn as_path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TmpDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    impl std::ops::Deref for TmpDir {
        type Target = std::path::Path;
        fn deref(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl AsRef<std::path::Path> for TmpDir {
        fn as_ref(&self) -> &std::path::Path {
            &self.0
        }
    }

    fn tmp(tag: &str) -> TmpDir {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        TmpDir(std::env::temp_dir().join(format!("brain-flux2-spec-test-{tag}-{}-{n}", std::process::id())))
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

    /// Toy vocab size every fixture writer below agrees on, so
    /// [`vocab_is_compatible`] finds a real match between a synthetic
    /// tokenizer and its intended text encoder.
    const TOY_VOCAB: usize = 100;

    fn write_qwen3_gguf(path: &std::path::Path, hidden: u64) {
        let tokens = checkpoint::gguf::GgufValue::Array((0..TOY_VOCAB).map(|i| checkpoint::gguf::GgufValue::String(format!("t{i}"))).collect());
        checkpoint::gguf_write::write(
            path.to_str().unwrap(),
            &[
                ("general.architecture".to_string(), checkpoint::gguf::GgufValue::String("qwen3".to_string())),
                ("general.finetune".to_string(), checkpoint::gguf::GgufValue::String("uncensored-text-encoder".to_string())),
                ("qwen3.embedding_length".to_string(), checkpoint::gguf::GgufValue::U64(hidden)),
                ("qwen3.block_count".to_string(), checkpoint::gguf::GgufValue::U32(36)),
                ("tokenizer.ggml.tokens".to_string(), tokens),
            ],
            &[checkpoint::gguf_write::TensorOut { name: "dummy".to_string(), shape: vec![1], ty: checkpoint::gguf::GgmlType::F32.id(), data: vec![0u8; 4] }],
            32,
        )
        .unwrap();
    }

    fn write_qwen3_hfdir(dir: &std::path::Path, hidden: u64) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"architectures": ["Qwen3ForCausalLM"], "hidden_size": hidden, "vocab_size": TOY_VOCAB})).unwrap()).unwrap();
    }

    fn write_vae_hfdir(dir: &std::path::Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"_class_name": "AutoencoderKLFlux2"})).unwrap()).unwrap();
    }

    /// A real vocab table at [`TOY_VOCAB`] entries, matching what every
    /// synthetic text encoder in this module declares - [`vocab_is_compatible`]
    /// checks real content, so a fixture must actually agree with itself.
    fn write_tokenizer_json(path: &std::path::Path) {
        let vocab: serde_json::Map<String, serde_json::Value> = (0..TOY_VOCAB).map(|i| (format!("t{i}"), serde_json::json!(i))).collect();
        std::fs::write(path, serde_json::to_vec(&serde_json::json!({"version": "1.0", "model": {"vocab": vocab}, "added_tokens": []})).unwrap()).unwrap();
    }

    /// A bare single-file BFL-named DiT safetensors release (`black-forest-
    /// labs/FLUX.2-dev/flux2-dev.safetensors`'s own shape: no GGUF wrapper,
    /// no diffusers `transformer_blocks.` renaming - the same
    /// `img_in.weight`/`txt_in.weight`/`double_blocks.N.*`/`single_blocks.N.*`
    /// names `write_dit_gguf` writes into a GGUF container). Only the tensors
    /// [`crate::import::dit_config_from_shapes`] actually reads, mirroring
    /// `write_dit_gguf`'s own minimal payload.
    fn write_dit_safetensors(path: &std::path::Path, cfg: &Flux2Config) {
        checkpoint::st::save_safetensors(
            path.to_str().unwrap(),
            &[
                ("img_in.weight".to_string(), vec![cfg.hidden as u64, cfg.in_channels as u64], vec![0.0f32; cfg.hidden * cfg.in_channels]),
                ("txt_in.weight".to_string(), vec![cfg.hidden as u64, cfg.context_in_dim as u64], vec![0.0f32; cfg.hidden * cfg.context_in_dim]),
                ("double_blocks.0.img_attn.norm.query_norm.scale".to_string(), vec![cfg.head_dim() as u64], vec![0.0f32; cfg.head_dim()]),
                (format!("double_blocks.{}.marker", cfg.depth_double - 1), vec![1], vec![0.0f32]),
                (format!("single_blocks.{}.marker", cfg.depth_single - 1), vec![1], vec![0.0f32]),
            ],
            &serde_json::json!({}),
            None,
        )
        .unwrap();
    }

    /// The same tensors [`write_dit_safetensors`] writes, split across two
    /// files under `dir` - a sharded diffusers `transformer/` directory's
    /// real on-disk shape (BFL's own undistilled "dev"/base release, when
    /// downloaded from the diffusers repo rather than as one flat file).
    /// `assemble()`'s `dit_config_from_path` used to unconditionally
    /// `MmapGguf::open` the chosen dit - which cannot open a directory at
    /// all - so a `--dit <this directory>` override crashed with "opening
    /// dit ...: Is a directory" instead of resolving; it now delegates to
    /// `crate::import::sniff_dit_size`, which already walks every
    /// `.safetensors` file under a directory.
    fn write_dit_safetensors_sharded(dir: &std::path::Path, cfg: &Flux2Config) {
        std::fs::create_dir_all(dir).unwrap();
        checkpoint::st::save_safetensors(
            dir.join("shard-1.safetensors").to_str().unwrap(),
            &[
                ("img_in.weight".to_string(), vec![cfg.hidden as u64, cfg.in_channels as u64], vec![0.0f32; cfg.hidden * cfg.in_channels]),
                ("txt_in.weight".to_string(), vec![cfg.hidden as u64, cfg.context_in_dim as u64], vec![0.0f32; cfg.hidden * cfg.context_in_dim]),
            ],
            &serde_json::json!({}),
            None,
        )
        .unwrap();
        checkpoint::st::save_safetensors(
            dir.join("shard-2.safetensors").to_str().unwrap(),
            &[
                ("double_blocks.0.img_attn.norm.query_norm.scale".to_string(), vec![cfg.head_dim() as u64], vec![0.0f32; cfg.head_dim()]),
                (format!("double_blocks.{}.marker", cfg.depth_double - 1), vec![1], vec![0.0f32]),
                (format!("single_blocks.{}.marker", cfg.depth_single - 1), vec![1], vec![0.0f32]),
            ],
            &serde_json::json!({}),
            None,
        )
        .unwrap();
    }

    /// A bare vendor-flat VAE safetensors file (`unsloth`'s own release
    /// shape: both `encoder.*`/`decoder.*` halves in one file, no diffusers
    /// `vae/` directory or `config.json` beside it) - the one tensor pair
    /// [`classify_safetensors`] actually checks for, at real-name minimal
    /// payload.
    /// Real FLUX.2 VAE shape rank: a 2D conv, `[out_ch, in_ch, kh, kw]` (4
    /// dims) - `encoder.conv_in.weight` is `[128, 3, 3, 3]` on the real
    /// unsloth release.
    fn write_vae_safetensors_flat(path: &std::path::Path) {
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

    /// A causal 3D VIDEO VAE (Wan's real shape rank): `[out_ch, in_ch, kt,
    /// kh, kw]` (5 dims, the extra leading temporal axis) - carries the
    /// SAME two tensor NAMES FLUX.2's own VAE does (`decoder`/
    /// `encoder`.conv_in.weight are generic autoencoder naming, not
    /// FLUX.2-specific), which is exactly what makes a name-only check a
    /// false positive against a real, unrelated architecture's checkpoint.
    fn write_causal_3d_vae_safetensors_flat(path: &std::path::Path) {
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

    /// A full, unambiguous set of records for a 9b assembly: a real 9b dit,
    /// a vae, a tokenizer, and exactly ONE text_encoder candidate.
    /// Nested one level under a `vendor` directory (not flat under `dir`
    /// itself) - `dir` stands in for the models root, and every real store
    /// layout ([`vendor_dir`]'s doc) puts a release under one such
    /// directory below the root, never directly in it.
    fn nine_b_fixture(dir: &std::path::Path) -> Vec<ArtifactRecord> {
        let vendor = dir.join("vendor");
        std::fs::create_dir_all(&vendor).unwrap();
        let dit_path = vendor.join("dit.gguf");
        write_dit_gguf(&dit_path, &Flux2Config::klein_9b(), "flux");
        let te_path = vendor.join("Qwen3-8B");
        write_qwen3_hfdir(&te_path, 4096);
        let vae_path = vendor.join("vae");
        write_vae_hfdir(&vae_path);
        let tok_path = vendor.join("tokenizer.json");
        write_tokenizer_json(&tok_path);
        vec![
            complete(dit_path, ArtifactKind::Gguf),
            complete(te_path, ArtifactKind::HfDir),
            complete(vae_path, ArtifactKind::HfDir),
            complete(tok_path, ArtifactKind::TokenizerJson),
            // A second vendor's unrelated file, present in every real store
            // - without one, resolve()'s own root inference (the deepest
            // common ancestor of every record it's given) would collapse
            // onto this fixture's single `vendor` directory instead of
            // `dir`, which breaks vendor_dir-based tokenizer matching the
            // same way a single-vendor store never would in practice. Never
            // written to disk: ArtifactRecord literals don't need to be.
            complete(dir.join("other-vendor").join("unrelated.bin"), ArtifactKind::Opaque),
        ]
    }

    /// "is this valid JSON" is true of every tokenizer.json in the entire
    /// store, from every unrelated architecture - a Wan/MiniMax/etc
    /// tokenizer must not classify as a FLUX.2 candidate just because it
    /// happens to also be a tokenizer.json somewhere on disk. Co-location
    /// with an already-classified FLUX.2 component is the real signal.
    #[test]
    fn classify_tokenizer_requires_co_location_with_a_real_component_not_any_tokenizer_in_the_store() {
        let dir = tmp("tokenizer-co-location");
        let mut records = nine_b_fixture(&dir);

        // An unrelated architecture's own checkpoint + its own tokenizer,
        // living nowhere near any FLUX.2 component.
        let unrelated_dir = dir.join("unrelated-arch");
        std::fs::create_dir_all(&unrelated_dir).unwrap();
        let unrelated_tok = unrelated_dir.join("tokenizer.json");
        write_tokenizer_json(&unrelated_tok);
        records.push(complete(unrelated_tok.clone(), ArtifactKind::TokenizerJson));

        let out = Flux2Spec.classify(&records, dir.as_path());
        let tokenizer_candidates: Vec<_> = out.iter().filter(|(_, role, _)| role == "tokenizer").collect();
        assert_eq!(tokenizer_candidates.len(), 1, "{out:?}");
        let (idx, ..) = tokenizer_candidates[0];
        assert_ne!(records[*idx].path, unrelated_tok, "the unrelated tokenizer must never classify: {out:?}");
    }

    /// A vendor commonly publishes several unrelated models under the same
    /// top-level directory (a real store has exactly this: Qwen's own
    /// dense Qwen3-8B and its unrelated, differently-vocabbed Qwen3.5-27B
    /// checkpoint both live under `models/Qwen/`) - a same-vendor tokenizer
    /// belonging to that OTHER, unrelated model must not classify as a
    /// FLUX.2 candidate just because it shares the vendor directory with a
    /// real one.
    #[test]
    fn classify_tokenizer_requires_vocab_compatibility_not_only_the_same_vendor() {
        let dir = tmp("tokenizer-vocab-mismatch");
        let mut records = nine_b_fixture(&dir);

        // A second, unrelated checkpoint published by the SAME vendor as
        // the real text encoder, with a genuinely different vocabulary -
        // not a padding difference, a different model entirely.
        let other_model_dir = dir.join("vendor").join("Other-Model-27B");
        std::fs::create_dir_all(&other_model_dir).unwrap();
        let mismatched_tok = other_model_dir.join("tokenizer.json");
        let big_vocab: serde_json::Map<String, serde_json::Value> = (0..(TOY_VOCAB * 2)).map(|i| (format!("t{i}"), serde_json::json!(i))).collect();
        std::fs::write(&mismatched_tok, serde_json::to_vec(&serde_json::json!({"model": {"vocab": big_vocab}, "added_tokens": []})).unwrap()).unwrap();
        records.push(complete(mismatched_tok.clone(), ArtifactKind::TokenizerJson));

        let out = Flux2Spec.classify(&records, dir.as_path());
        let tokenizer_candidates: Vec<_> = out.iter().filter(|(_, role, _)| role == "tokenizer").collect();
        assert_eq!(tokenizer_candidates.len(), 1, "{out:?}");
        let (idx, ..) = tokenizer_candidates[0];
        assert_ne!(records[*idx].path, mismatched_tok, "the vocab-mismatched same-vendor tokenizer must never classify: {out:?}");
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

    /// A causal 3D video VAE (Wan's real release) carries the exact same
    /// `decoder`/`encoder`.conv_in.weight tensor NAMES FLUX.2's VAE does, so
    /// a name-only check misclassifies an entirely unrelated architecture's
    /// checkpoint as a FLUX.2 vae candidate. The conv's shape RANK (4 for a
    /// 2D image VAE, 5 for a causal 3D video VAE) is
    /// real, always-present structure, not a guess.
    #[test]
    fn classify_rejects_a_causal_3d_video_vae_sharing_the_same_tensor_names() {
        let dir = tmp("flat-vae-video-decoy");
        std::fs::create_dir_all(&dir).unwrap();
        let flux_vae_path = dir.join("flux2-vae.safetensors");
        write_vae_safetensors_flat(&flux_vae_path);
        let video_vae_path = dir.join("wan-vae.safetensors");
        write_causal_3d_vae_safetensors_flat(&video_vae_path);

        let records = vec![complete(flux_vae_path, ArtifactKind::Safetensors), complete(video_vae_path, ArtifactKind::Safetensors)];
        let out = Flux2Spec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "vae".to_string(), Confidence::Declared)], "{out:?}");
    }

    /// A bare single-file BFL-named DiT safetensors release (BFL's own real
    /// `FLUX.2-dev/flux2-dev.safetensors` - the undistilled "base" size class,
    /// never shipped as GGUF) must classify as a "dit" candidate exactly like
    /// its GGUF sibling does - `classify()`'s `ArtifactKind::Safetensors` arm
    /// used to only ever recognize a VAE, so a real safetensors DiT was never
    /// even offered as a candidate, regardless of an explicit `--dit`
    /// override (which reads `chosen`/`records`, populated from `classify`'s
    /// own output).
    #[test]
    fn classify_recognizes_a_bare_dit_safetensors_file() {
        let dir = tmp("flat-dit-safetensors");
        std::fs::create_dir_all(&dir).unwrap();
        let dit_path = dir.join("flux2-dev.safetensors");
        write_dit_safetensors(&dit_path, &Flux2Config::klein_9b());

        let records = vec![complete(dit_path, ArtifactKind::Safetensors)];
        let out = Flux2Spec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "dit".to_string(), Confidence::Derived)], "{out:?}");
    }

    /// A decoy safetensors file shaped like the VAE, not the DiT (no
    /// `img_in.weight`), must never be offered as a "dit" candidate -
    /// classification reads real tensor names, not "any safetensors file
    /// that isn't a VAE is a DiT".
    #[test]
    fn classify_does_not_misclassify_a_vae_safetensors_file_as_a_dit() {
        let dir = tmp("flat-vae-not-dit");
        std::fs::create_dir_all(&dir).unwrap();
        let vae_path = dir.join("flux2-vae.safetensors");
        write_vae_safetensors_flat(&vae_path);

        let records = vec![complete(vae_path, ArtifactKind::Safetensors)];
        let out = Flux2Spec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "vae".to_string(), Confidence::Declared)], "{out:?}");
    }

    /// `assemble()` used to unconditionally open the chosen "dit" as GGUF
    /// (`MmapGguf::open`, no extension check), so even an explicit `--dit
    /// /path/to/some.safetensors` override crashed with `gguf: bad magic`
    /// the moment assembly ran - found live trying to load BFL's real
    /// `FLUX.2-dev/flux2-dev.safetensors`. Must resolve the same shape class
    /// and variant options a GGUF dit at the same dimensions would.
    #[test]
    fn assemble_resolves_a_safetensors_dit_the_same_shape_class_as_its_gguf_sibling() {
        let dir = tmp("assemble-safetensors-dit");
        std::fs::create_dir_all(&dir).unwrap();
        let dit_path = dir.join("flux2-dev.safetensors");
        write_dit_safetensors(&dit_path, &Flux2Config::klein_9b());
        let records = vec![complete(dit_path, ArtifactKind::Safetensors)];
        let chosen = BTreeMap::from([("dit".to_string(), 0usize)]);

        let mut overrides = BTreeMap::new();
        overrides.insert("variant".to_string(), "base-9b".to_string());
        let outcome = Flux2Spec.assemble(&chosen, &records, &overrides).unwrap();
        match outcome {
            AssembleOutcome::Assembled(v) => assert_eq!(v.variant.as_deref(), Some("base-9b")),
            other => panic!("expected Assembled, got {other:?}"),
        }

        // Same file, no variant override: the shape class alone is ambiguous
        // between klein and base, exactly as the GGUF path already tests -
        // a safetensors dit must not silently default to klein either.
        let outcome = Flux2Spec.assemble(&chosen, &records, &BTreeMap::new()).unwrap();
        match outcome {
            AssembleOutcome::UnresolvedVariant { shape_class, options } => {
                assert_eq!(shape_class, "9b");
                assert!(options.contains(&"klein-9b".to_string()) && options.contains(&"base-9b".to_string()), "{options:?}");
            }
            other => panic!("expected UnresolvedVariant, got {other:?}"),
        }
    }

    /// THE regression this module's `dit_shapes` refactor exists for:
    /// `assemble()` used to unconditionally `MmapGguf::open` the chosen dit,
    /// which cannot open a directory - a sharded diffusers `transformer/`
    /// release, force-selected via `--dit <dir>`, crashed with an "opening
    /// dit ...: Is a directory" error instead of resolving. It must now
    /// succeed identically to the single-file case.
    #[test]
    fn assemble_resolves_a_sharded_safetensors_dit_directory() {
        let dir = tmp("assemble-sharded-safetensors-dit");
        let dit_dir = dir.join("transformer");
        write_dit_safetensors_sharded(&dit_dir, &Flux2Config::klein_9b());
        let records = vec![complete(dit_dir, ArtifactKind::Safetensors)];
        let chosen = BTreeMap::from([("dit".to_string(), 0usize)]);

        let mut overrides = BTreeMap::new();
        overrides.insert("variant".to_string(), "base-9b".to_string());
        let outcome = Flux2Spec.assemble(&chosen, &records, &overrides).unwrap();
        match outcome {
            AssembleOutcome::Assembled(v) => assert_eq!(v.variant.as_deref(), Some("base-9b")),
            other => panic!("expected Assembled, got {other:?}"),
        }
    }

    /// [`classify_safetensors`]'s dit arm must recognize a bare safetensors
    /// dit through `crate::import::dit_shapes`'s canonicalization, not only
    /// a file already BFL-named - a diffusers-renamed single-file release
    /// (the split q/k/v naming `diffusers_to_bfl` maps onto `img_in.weight`/
    /// `txt_in.weight`/etc.) must classify exactly as a BFL-named one does.
    #[test]
    fn classify_recognizes_a_diffusers_renamed_dit_safetensors_file() {
        let dir = tmp("classify-diffusers-renamed-dit");
        std::fs::create_dir_all(&dir).unwrap();
        let dit_path = dir.join("diffusion_pytorch_model.safetensors");
        let cfg = Flux2Config::klein_9b();
        // `diffusers_to_bfl`'s own mapping (verified against
        // `crates/flux2/src/import.rs`): `x_embedder.weight` ->
        // `img_in.weight`, `context_embedder.weight` -> `txt_in.weight`,
        // `transformer_blocks.{i}.attn.norm_q.weight` ->
        // `double_blocks.{i}.img_attn.norm.query_norm.scale`,
        // `single_transformer_blocks.{i}.attn.norm_q.weight` ->
        // `single_blocks.{i}.norm.query_norm.scale`. This is a full enough
        // fixture (head-dim marker at block 0 plus a marker at the deepest
        // double/single block) to satisfy `dit_config_from_shapes`'s depth
        // and head-count detection under the diffusers names alone - a
        // fixture too thin to do that (as an earlier version of this test
        // was) can't tell "canonicalization worked but this fixture is
        // incomplete" apart from "canonicalization never ran": both read as
        // an empty `classify()` result.
        checkpoint::st::save_safetensors(
            dit_path.to_str().unwrap(),
            &[
                ("x_embedder.weight".to_string(), vec![cfg.hidden as u64, cfg.in_channels as u64], vec![0.0f32; cfg.hidden * cfg.in_channels]),
                ("context_embedder.weight".to_string(), vec![cfg.hidden as u64, cfg.context_in_dim as u64], vec![0.0f32; cfg.hidden * cfg.context_in_dim]),
                ("transformer_blocks.0.attn.norm_q.weight".to_string(), vec![cfg.head_dim() as u64], vec![0.0f32; cfg.head_dim()]),
                (format!("transformer_blocks.{}.attn.norm_q.weight", cfg.depth_double - 1), vec![cfg.head_dim() as u64], vec![0.0f32; cfg.head_dim()]),
                (format!("single_transformer_blocks.{}.attn.norm_q.weight", cfg.depth_single - 1), vec![cfg.head_dim() as u64], vec![0.0f32; cfg.head_dim()]),
            ],
            &serde_json::json!({}),
            None,
        )
        .unwrap();
        let records = vec![complete(dit_path, ArtifactKind::Safetensors)];
        let out = Flux2Spec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "dit".to_string(), Confidence::Derived)], "{out:?}");
    }

    /// [`dit_context_in_dim`] (what `validate()` calls) must read a
    /// safetensors dit's `txt_in.weight` shape too, not only a GGUF's -
    /// mirrors `a_9b_dit_with_a_4b_text_encoder_is_rejected_by_validate_
    /// before_any_gpu_work` below, but for the safetensors path.
    #[test]
    fn a_9b_safetensors_dit_with_a_4b_text_encoder_is_rejected_by_validate() {
        let dir = tmp("mismatched-validate-safetensors");
        std::fs::create_dir_all(&dir).unwrap();
        let dit_path = dir.join("flux2-dev.safetensors");
        write_dit_safetensors(&dit_path, &Flux2Config::klein_9b());
        let te_path = dir.join("Qwen3-4B");
        write_qwen3_hfdir(&te_path, 2560);

        let assembly = Assembly {
            id: "local/flux2-base-9b".to_string(),
            arch: "flux2".to_string(),
            variant: Some("base-9b".to_string()),
            roles: BTreeMap::from([("dit".to_string(), dit_path.clone()), ("text_encoder".to_string(), te_path.clone())]),
            provenance: Vec::new(),
        };
        let err = Flux2Spec.validate(&assembly).unwrap_err();
        assert!(err.contains("12288"), "{err}");
        assert!(err.contains("2560"), "{err}");
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
        // A second, equally Declared text_encoder candidate: a real
        // release's uncensored-qwen3 GGUF, same embedding_length, published
        // by a second vendor - a real store always has more than one, and
        // resolve()'s own root inference (the deepest common ancestor of
        // every record) needs that second vendor present to land on `dir`
        // rather than collapsing onto the fixture's own single vendor dir.
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
