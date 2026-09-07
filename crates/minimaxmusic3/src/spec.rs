// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax Music 3's [`ArchSpec`]: which on-disk artifacts satisfy which of
//! the six roles (`language_model`, `depth_decoder`, `condition_encoder`,
//! `transformer`, `vocoder`, `tokenizer`), and the one compatibility gate a
//! chosen assembly must pass - the depth decoder's hidden width must equal
//! the Global LLM's own `hidden_size`, since the depth decoder reads the
//! LLM's per-frame hidden state directly with no projection in between
//! (`crate::config`'s own module doc: "both are the Global LLM's hidden
//! width", already checked as an invariant between the crate's own `::real()`
//! configs in that file's tests - this is the same fact, read from real
//! checkpoint headers instead of from two hardcoded structs).
//!
//! `language_model` is an ordinary HF directory (`config.json` + safetensors
//! shards - a real `Qwen3ForCausalLM`), so it classifies exactly like
//! `flux2::spec`'s own `text_encoder` role. The other four brain-native
//! components (`depth_decoder`/`condition_encoder`/`transformer`/`vocoder`)
//! ship with NO `config.json` of their own (`crate::config`'s configs are
//! hardcoded `::real()` structs, never read from a checkpoint) - so they
//! never collapse to an `HfDir` record, and classification reads their
//! safetensors' own tensor NAMES directly
//! ([`checkpoint::mmap::MmapSafetensors`], header-only), the same "real
//! content, never the filename" discipline `flux2::spec::classify_safetensors`
//! uses for FLUX.2's bare vendor-flat VAE file.
//!
//! **A real, must-not-misclassify case**: the checkpoint's OTHER
//! language-model directory, `qwen_7B/qwen_7B/`, declares
//! `"architectures": ["AbabForCausalLM"]` (MiniMax's own native training
//! format, `model_type: "mixtral"`) despite matching `language_model/`'s
//! surface dims (`hidden=4096, layers=36, heads=32, ...`) - confirmed
//! against the real checkpoint in [`global_llm::import`]'s own module doc,
//! not assumed. Reading the declared architecture, not the surface shape,
//! is what tells them apart.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::plan::declared_architecture;
use brain_modelstore::resolve::{classify_tokenizer_role, ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

const ROLES: &[&str] = &["language_model", "depth_decoder", "condition_encoder", "transformer", "vocoder", "tokenizer"];

fn classify_hfdir(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    let Ok(bytes) = std::fs::read(rec.path.join("config.json")) else { return };
    let Ok(config) = serde_json::from_slice::<serde_json::Value>(&bytes) else { return };
    if declared_architecture(&config).as_deref() == Some("Qwen3ForCausalLM") {
        out.push((idx, "language_model".to_string(), Confidence::Declared));
    }
}

/// `depth_decoder`/`condition_encoder`/`transformer`/`vocoder` each carry NO
/// `config.json` of their own (see the module doc), so their real tensor
/// names are the only content signal: a single-file safetensors' own name
/// ("model.safetensors"/"diffusion_pytorch_model.safetensors" - generic
/// convention names, per `checkpoint::safetensors::read_model_dir`) is never
/// enough on its own. Each check below reads a NAME that role's own importer
/// (`crate::depth_decoder`/`condition_encoder`/`dit`/`vocoder`) actually
/// requires, unique to that role.
fn classify_safetensors(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    let Ok(m) = checkpoint::mmap::MmapSafetensors::open(&rec.path) else { return };
    let has = |name: &str| m.shape(name).is_some();
    let any_prefixed = |prefix: &str, suffix: &str| m.names().iter().any(|n| n.starts_with(prefix) && n.ends_with(suffix));

    if has("audio_embeddings.weight") && has("pos_embedding.weight") && any_prefixed("audio_heads.", ".weight") {
        out.push((idx, "depth_decoder".to_string(), Confidence::Declared));
    } else if has("layer_weight_logits") && has("proj.weight") {
        out.push((idx, "condition_encoder".to_string(), Confidence::Declared));
    } else if any_prefixed("transformer_blocks.", ".ff_in.weight") {
        out.push((idx, "transformer".to_string(), Confidence::Declared));
    } else if has("dec_in_proj.weight") || has("dec_in_proj.weight_g") {
        out.push((idx, "vocoder".to_string(), Confidence::Declared));
    }
}

/// The Global LLM's own declared `hidden_size` (`language_model/config.json`).
fn language_model_hidden_size(path: &Path) -> Result<usize, String> {
    let config_path = path.join("config.json");
    let bytes = std::fs::read(&config_path).map_err(|e| format!("minimaxmusic3 validate: reading {}: {e}", config_path.display()))?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| format!("minimaxmusic3 validate: parsing {}: {e}", config_path.display()))?;
    v.get("hidden_size").and_then(serde_json::Value::as_u64).map(|n| n as usize).ok_or_else(|| format!("minimaxmusic3 validate: {} has no hidden_size", config_path.display()))
}

/// The depth decoder's own real hidden width - `norm.weight`'s single
/// dimension, the RMSNorm gain over the residual stream every attention/MLP
/// output in the block is added back into.
fn depth_decoder_hidden_size(path: &Path) -> Result<usize, String> {
    let m = checkpoint::mmap::MmapSafetensors::open(path).map_err(|e| format!("minimaxmusic3 validate: opening depth_decoder {}: {e}", path.display()))?;
    let shape = m.shape("norm.weight").ok_or_else(|| format!("minimaxmusic3 validate: {} has no norm.weight", path.display()))?;
    shape.first().copied().ok_or_else(|| format!("minimaxmusic3 validate: {} norm.weight has no dimensions", path.display()))
}

pub struct MinimaxMusic3Spec;

impl ArchSpec for MinimaxMusic3Spec {
    fn arch(&self) -> &'static str {
        "minimaxmusic3"
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
                ArtifactKind::HfDir => classify_hfdir(idx, rec, &mut out),
                ArtifactKind::Safetensors => classify_safetensors(idx, rec, &mut out),
                _ => {}
            }
        }
        // The Global LLM's own already-classified candidates are the real
        // dependency signal `classify_tokenizer_role` needs - shared with
        // every other architecture with a tokenizer-shaped role, not
        // minimaxmusic3-specific. `language_model` IS an `HfDir` with a real
        // `config.json` `vocab_size`, so (unlike cosyvoice's bare `.pt` llm)
        // that shared helper's own `checkpoint_vocab_size` applies directly.
        let lm_candidates: Vec<&Path> = out.iter().filter(|(_, role, _)| role == "language_model").map(|(idx, ..)| records[*idx].path.as_path()).collect();
        classify_tokenizer_role(records, inventory_root, "tokenizer", &lm_candidates, &mut out);
        out
    }

    fn assemble(&self, _chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        // No named variant: unlike FLUX.2's klein/base or CosyVoice's 2/3,
        // this architecture has exactly one released configuration - once
        // every role resolves to one candidate, the assembly is complete.
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/minimaxmusic3".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let lm_path = assembly.roles.get("language_model").ok_or("minimaxmusic3 validate: assembly has no language_model role")?;
        let depth_path = assembly.roles.get("depth_decoder").ok_or("minimaxmusic3 validate: assembly has no depth_decoder role")?;
        let lm_hidden = language_model_hidden_size(lm_path)?;
        let depth_hidden = depth_decoder_hidden_size(depth_path)?;
        if lm_hidden != depth_hidden {
            return Err(format!(
                "minimaxmusic3 validate: depth_decoder hidden width={depth_hidden} does not match the Global LLM's hidden_size={lm_hidden} - language_model={}, depth_decoder={}",
                lm_path.display(),
                depth_path.display()
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
        std::env::temp_dir().join(format!("brain-minimaxmusic3-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    fn write_st(path: &Path, tensors: &[(&str, Vec<usize>)]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let entries: Vec<(String, Vec<u64>, Vec<f32>)> = tensors.iter().map(|(n, shape)| (n.to_string(), shape.iter().map(|&d| d as u64).collect(), vec![0.0f32; shape.iter().product::<usize>().max(1)])).collect();
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &entries, &serde_json::json!({}), None).unwrap();
    }

    const TOY_VOCAB: usize = 100;

    fn write_lm_hfdir(dir: &Path, hidden: u64, architecture: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"architectures": [architecture], "model_type": if architecture == "Qwen3ForCausalLM" { "qwen3" } else { "mixtral" }, "hidden_size": hidden, "vocab_size": TOY_VOCAB})).unwrap()).unwrap();
        // `hfdir_record` requires a real shard set beside `config.json`, not
        // only the manifest - a loose `model.safetensors` satisfies it.
        write_st(&dir.join("model.safetensors"), &[("dummy", vec![1])]);
    }

    fn write_depth_decoder(dir: &Path, hidden: usize) {
        write_st(
            &dir.join("model.safetensors"),
            &[("audio_embeddings.weight", vec![512, hidden]), ("pos_embedding.weight", vec![16, hidden]), ("audio_heads.0.weight", vec![512, hidden]), ("norm.weight", vec![hidden])],
        );
    }

    fn write_condition_encoder(dir: &Path, hidden: usize) {
        write_st(&dir.join("diffusion_pytorch_model.safetensors"), &[("layer_weight_logits", vec![8]), ("layer_scale", vec![1]), ("proj.weight", vec![32, hidden, 3]), ("proj.bias", vec![32])]);
    }

    fn write_transformer(dir: &Path) {
        write_st(&dir.join("model.safetensors"), &[("transformer_blocks.0.ff_in.weight", vec![16, 8]), ("transformer_blocks.0.attn.to_q.weight", vec![8, 8])]);
    }

    fn write_vocoder(dir: &Path) {
        write_st(&dir.join("diffusion_pytorch_model.safetensors"), &[("dec_in_proj.weight", vec![16, 4, 1]), ("dec_in_proj.bias", vec![16])]);
    }

    fn write_tokenizer_json(path: &Path) {
        let vocab: serde_json::Map<String, serde_json::Value> = (0..TOY_VOCAB).map(|i| (format!("t{i}"), serde_json::json!(i))).collect();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, serde_json::to_vec(&serde_json::json!({"model": {"vocab": vocab}, "added_tokens": []})).unwrap()).unwrap();
    }

    const REAL_HIDDEN: u64 = 32;

    /// A full, unambiguous set of records for one MiniMax Music 3 checkpoint,
    /// laid out under one vendor directory - including the real decoy
    /// (`qwen_7B/qwen_7B`, a materially different architecture despite
    /// matching surface dims) and the tokenizer's own THIRD location
    /// (`qwen_7B/qwen3-8B-tokenizer-music`, same vendor, neither of the
    /// other two role directories) - both real layout facts from
    /// `global_llm::import`'s own module doc, not simplified away.
    fn fixture(dir: &Path) -> Vec<ArtifactRecord> {
        let vendor = dir.join("MiniMaxAI").join("MiniMax-Music3");
        let lm_dir = vendor.join("language_model");
        write_lm_hfdir(&lm_dir, REAL_HIDDEN, "Qwen3ForCausalLM");
        let decoy_dir = vendor.join("qwen_7B").join("qwen_7B");
        write_lm_hfdir(&decoy_dir, REAL_HIDDEN, "AbabForCausalLM");
        let depth_dir = vendor.join("rvq_depth_decoder");
        write_depth_decoder(&depth_dir, REAL_HIDDEN as usize);
        let cond_dir = vendor.join("condition_encoder");
        write_condition_encoder(&cond_dir, REAL_HIDDEN as usize);
        let dit_dir = vendor.join("transformer");
        write_transformer(&dit_dir);
        let voc_dir = vendor.join("vocoder");
        write_vocoder(&voc_dir);
        let tok_path = vendor.join("qwen_7B").join("qwen3-8B-tokenizer-music").join("tokenizer.json");
        write_tokenizer_json(&tok_path);

        vec![
            complete(lm_dir, ArtifactKind::HfDir),
            complete(decoy_dir, ArtifactKind::HfDir),
            complete(depth_dir.join("model.safetensors"), ArtifactKind::Safetensors),
            complete(cond_dir.join("diffusion_pytorch_model.safetensors"), ArtifactKind::Safetensors),
            complete(dit_dir.join("model.safetensors"), ArtifactKind::Safetensors),
            complete(voc_dir.join("diffusion_pytorch_model.safetensors"), ArtifactKind::Safetensors),
            complete(tok_path, ArtifactKind::TokenizerJson),
            // A second, unrelated vendor - see `flux2::spec`'s own fixture
            // doc for why `resolve`'s root inference needs one present.
            complete(dir.join("other-vendor").join("unrelated.bin"), ArtifactKind::Opaque),
        ]
    }

    #[test]
    fn classify_rejects_the_same_shaped_but_differently_architected_decoy_language_model() {
        let dir = tmp("lm-decoy");
        let records = fixture(&dir);
        let out = MinimaxMusic3Spec.classify(&records, dir.as_path());
        let lm_candidates: Vec<_> = out.iter().filter(|(_, role, _)| role == "language_model").collect();
        assert_eq!(lm_candidates.len(), 1, "{out:?}");
        let (idx, ..) = lm_candidates[0];
        assert_eq!(records[*idx].path, dir.join("MiniMaxAI").join("MiniMax-Music3").join("language_model"));
    }

    #[test]
    fn classify_reads_tensor_names_not_the_generic_shard_filename() {
        let dir = tmp("content-not-filename");
        std::fs::create_dir_all(&dir).unwrap();
        // Both real shard filenames `read_model_dir` accepts, but with
        // swapped content - proving classification never trusts the name.
        let decoy = dir.join("depth_decoder_pretending_to_be_vocoder").join("model.safetensors");
        write_st(&decoy, &[("dec_in_proj.weight", vec![4, 2, 1])]);

        let records = vec![complete(decoy, ArtifactKind::Safetensors)];
        let out = MinimaxMusic3Spec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "vocoder".to_string(), Confidence::Declared)], "{out:?}");
    }

    #[test]
    fn a_full_fixture_resolves_cleanly_with_every_role_present() {
        let dir = tmp("resolves");
        let records = fixture(&dir);
        let spec = MinimaxMusic3Spec;
        let specs: [&dyn ArchSpec; 1] = [&spec];
        let out = resolve("minimaxmusic3", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                for role in ROLES {
                    assert!(a.roles.contains_key(*role), "missing role {role}: {a:?}");
                }
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn a_depth_decoder_hidden_width_mismatch_is_rejected_by_validate_before_any_gpu_work() {
        let dir = tmp("mismatched-hidden");
        let lm_dir = dir.join("language_model");
        write_lm_hfdir(&lm_dir, 4096, "Qwen3ForCausalLM");
        let depth_dir = dir.join("rvq_depth_decoder");
        write_depth_decoder(&depth_dir, 8); // deliberately mismatched

        let assembly = Assembly {
            id: "local/minimaxmusic3".to_string(),
            arch: "minimaxmusic3".to_string(),
            variant: None,
            roles: BTreeMap::from([("language_model".to_string(), lm_dir), ("depth_decoder".to_string(), depth_dir.join("model.safetensors"))]),
            provenance: Vec::new(),
        };
        let err = MinimaxMusic3Spec.validate(&assembly).unwrap_err();
        assert!(err.contains("4096"), "{err}");
        assert!(err.contains('8'), "{err}");
    }

    #[test]
    fn two_language_model_candidates_at_equal_confidence_is_ambiguous_not_a_silent_pick() {
        let dir = tmp("two-lms");
        let mut records = fixture(&dir);
        // A second, equally Declared language_model candidate - a real store
        // always has more than one plausible checkpoint under a vendor.
        let second_lm = dir.join("MiniMaxAI").join("MiniMax-Music3-Alt").join("language_model");
        write_lm_hfdir(&second_lm, REAL_HIDDEN, "Qwen3ForCausalLM");
        records.push(complete(second_lm, ArtifactKind::HfDir));

        let spec = MinimaxMusic3Spec;
        let specs: [&dyn ArchSpec; 1] = [&spec];
        let out = resolve("minimaxmusic3", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Ambiguous(a) => assert_eq!(a.question, Question::Role { role: "language_model".to_string() }),
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }
}
