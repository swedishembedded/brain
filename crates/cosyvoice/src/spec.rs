// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CosyVoice's [`ArchSpec`]: which on-disk artifacts satisfy which role
//! (`llm`, `flow`, `hift`, `tokenizer`), and the one compatibility gate a
//! chosen assembly must pass - `llm`/`flow`/`hift` must all be the SAME
//! generation (CosyVoice 2 or CosyVoice 3), never a silent mix of the two.
//!
//! `llm.pt`/`flow.pt`/`hift.pt` are `torch.save` checkpoints
//! ([`checkpoint::torchpt`]), not GGUF/safetensors, so every check here reads
//! a `.pt`'s own tensor NAMES and SHAPES via
//! [`checkpoint::torchpt::read_shapes`] - a pickle-stream-only read that
//! never decodes a single storage byte (see that function's own doc) - the
//! same "header/config only, no tensor bytes decoded" discipline
//! `crate::spec`'s reference, `flux2::spec`, holds for GGUF/safetensors.
//!
//! CosyVoice 2 and CosyVoice 3 ship the exact same FOUR role names
//! (`llm`/`flow`/`hift`/`tokenizer`) with entirely different weights: if a
//! store somehow has both generations' checkpoints, [`resolve`] naturally
//! reports that as a `Role` [`Ambiguity`] (two Declared-confidence `llm`
//! candidates, say) rather than [`ArchSpec::assemble`] silently picking one -
//! there is nothing generation-specific in `classify`'s own role tagging, so
//! this needs no bespoke handling here.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{tokenizer_vocab_count, vendor_dir, vocab_is_compatible, ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

use crate::pipeline::Variant;

pub struct CosyVoiceSpec;

const ROLES: &[&str] = &["llm", "flow", "hift", "tokenizer"];

/// Every (name, shape) `path` declares, or `None` if it cannot be read as a
/// torch checkpoint at all - never a classification on its own, matching
/// `flux2::spec`'s "an unreadable file classifies as nothing" rule.
fn read_pt(path: &Path) -> Option<Vec<(String, Vec<usize>)>> {
    checkpoint::torchpt::read_shapes(path.to_string_lossy().as_ref()).ok()
}

fn has(names: &[(String, Vec<usize>)], name: &str) -> bool {
    names.iter().any(|(n, _)| n == name)
}

fn any_prefixed(names: &[(String, Vec<usize>)], prefix: &str) -> bool {
    names.iter().any(|(n, _)| n.starts_with(prefix))
}

fn shape_of<'a>(names: &'a [(String, Vec<usize>)], name: &str) -> Option<&'a [usize]> {
    names.iter().find(|(n, _)| n == name).map(|(_, s)| s.as_slice())
}

/// Which generation `names` (a `flow.pt`) belongs to, from the estimator's
/// own tensor names - `decoder.estimator.down_blocks.*` is CosyVoice 2's
/// UNet CFM estimator, `decoder.estimator.transformer_blocks.*` is
/// CosyVoice 3's DiT one (`crate::flow_import`/`crate::cv3_flow_import`'s own
/// module docs). `None` when this is not a recognizable flow.pt at all - a
/// filename-only match is never enough, so a `flow.pt` that carries neither
/// prefix is not a candidate.
fn flow_variant(names: &[(String, Vec<usize>)]) -> Option<Variant> {
    if !has(names, "input_embedding.weight") || !has(names, "spk_embed_affine_layer.weight") {
        return None;
    }
    if any_prefixed(names, "decoder.estimator.transformer_blocks.") {
        Some(Variant::CosyVoice3)
    } else if any_prefixed(names, "decoder.estimator.down_blocks.") {
        Some(Variant::CosyVoice2)
    } else {
        None
    }
}

/// Which generation `names` (a `hift.pt`) belongs to. Both generations share
/// every tensor NAME (`crate::cv3_hift_import`'s module doc: "identical key
/// structure to CosyVoice 2's hift.pt"), so the role itself is name-based,
/// but telling the two apart needs a real SHAPE: `conv_pre`'s weight-normed
/// kernel width is 7 for CosyVoice 2, 5 for CosyVoice 3 (both verified
/// against real checkpoints in that module's own doc). Any other width is
/// not a recognized hift.pt at all.
fn hift_variant(names: &[(String, Vec<usize>)]) -> Option<Variant> {
    if !has(names, "f0_predictor.classifier.weight") || !has(names, "m_source.l_linear.weight") {
        return None;
    }
    let k = *shape_of(names, "conv_pre.parametrizations.weight.original1")?.get(2)?;
    match k {
        7 => Some(Variant::CosyVoice2),
        5 => Some(Variant::CosyVoice3),
        _ => None,
    }
}

/// Which generation `names` (an `llm.pt`) belongs to. Both generations carry
/// a full independent Qwen2.5-0.5B backbone under `llm.model.model.*` plus a
/// bolted-on `speech_embedding`/`llm_decoder` (`crate::llm_import`'s module
/// doc); CosyVoice 3's `Qwen2LM`-vs-`CosyVoice3LM` split is that only
/// CosyVoice 2 carries a separate `llm_embedding.weight` table
/// (`SpecialTokenSource::LlmEmbedding`) - a real, verified-absent-not-assumed
/// difference, not a guess.
fn llm_variant(names: &[(String, Vec<usize>)]) -> Option<Variant> {
    if !has(names, "llm.model.model.embed_tokens.weight") || !has(names, "speech_embedding.weight") || !has(names, "llm_decoder.weight") {
        return None;
    }
    Some(if has(names, "llm_embedding.weight") { Variant::CosyVoice2 } else { Variant::CosyVoice3 })
}

/// The backbone's own declared TEXT vocabulary size - `llm.model.model.
/// embed_tokens.weight`'s first dimension - the real content
/// [`classify_tokenizer`] compares the CosyVoice-BlankEN tokenizer's own
/// vocab count against. Not [`brain_modelstore::resolve::checkpoint_vocab_size`]:
/// that helper reads a GGUF KV or an HF directory's `config.json`, neither of
/// which a bare `llm.pt` file has.
fn llm_text_vocab_size(names: &[(String, Vec<usize>)]) -> Option<usize> {
    shape_of(names, "llm.model.model.embed_tokens.weight")?.first().copied()
}

/// Pass 1: `llm`/`flow`/`hift`, from a `.pt`'s own tensor names/shapes.
fn classify_torch(idx: usize, rec: &ArtifactRecord, out: &mut Vec<(usize, String, Confidence)>) {
    let Some(names) = read_pt(&rec.path) else { return };
    if llm_variant(&names).is_some() {
        out.push((idx, "llm".to_string(), Confidence::Declared));
    } else if flow_variant(&names).is_some() {
        out.push((idx, "flow".to_string(), Confidence::Declared));
    } else if hift_variant(&names).is_some() {
        out.push((idx, "hift".to_string(), Confidence::Declared));
    }
}

/// Pass 2: `tokenizer` - a stock Qwen BPE identity (`CosyVoice-BlankEN`) from
/// a genuinely different upstream repo than `llm`/`flow`/`hift`, so it needs
/// the same real signal `brain_modelstore::resolve::classify_tokenizer_role`
/// gives every other architecture's tokenizer-shaped role (same-vendor
/// co-location AND vocab-size compatibility with a dependency candidate) -
/// but not THAT function directly: its own `checkpoint_vocab_size` only
/// reads a GGUF/HF-directory checkpoint, and the dependency here is a bare
/// `llm.pt` file, so this orchestrates the same two primitives
/// ([`vendor_dir`], [`vocab_is_compatible`]) against [`llm_text_vocab_size`]
/// instead - exactly the "different orchestration shape" escape hatch that
/// shared function's own doc calls out.
fn classify_tokenizer(records: &[ArtifactRecord], root: &Path, llm_candidates: &[&Path], out: &mut Vec<(usize, String, Confidence)>) {
    let vendor_dirs: std::collections::BTreeSet<std::path::PathBuf> = llm_candidates.iter().filter_map(|p| vendor_dir(p, root)).collect();
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
        let compatible = llm_candidates
            .iter()
            .filter(|p| vendor_dir(p, root).as_deref() == Some(vendor.as_path()))
            .any(|p| read_pt(p).and_then(|n| llm_text_vocab_size(&n)).is_some_and(|v| vocab_is_compatible(tok_count, v)));
        if compatible {
            out.push((idx, "tokenizer".to_string(), Confidence::Declared));
        }
    }
}

impl ArchSpec for CosyVoiceSpec {
    fn arch(&self) -> &'static str {
        "cosyvoice"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            if !rec.usable() || rec.kind != ArtifactKind::Torch {
                continue;
            }
            classify_torch(idx, rec, &mut out);
        }
        let llm_candidates: Vec<&Path> = out.iter().filter(|(_, role, _)| role == "llm").map(|(idx, ..)| records[*idx].path.as_path()).collect();
        classify_tokenizer(records, inventory_root, &llm_candidates, &mut out);
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        let flow_idx = *chosen.get("flow").ok_or("cosyvoice assemble: no flow chosen")?;
        let flow_path = &records[flow_idx].path;
        let names = read_pt(flow_path).ok_or_else(|| format!("cosyvoice assemble: could not read {}", flow_path.display()))?;
        let variant = flow_variant(&names).ok_or_else(|| format!("cosyvoice assemble: {} is not a recognized flow.pt (neither CosyVoice 2's UNet nor CosyVoice 3's DiT tensor names found)", flow_path.display()))?;
        let variant_str = variant.as_str().to_string();
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: format!("local/cosyvoice-{variant_str}"), variant: Some(variant_str) }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let get = |role: &str| -> Result<&std::path::PathBuf, String> { assembly.roles.get(role).ok_or_else(|| format!("cosyvoice validate: assembly has no {role} role")) };
        let llm_path = get("llm")?;
        let flow_path = get("flow")?;
        let hift_path = get("hift")?;

        let read = |path: &std::path::Path| -> Result<Vec<(String, Vec<usize>)>, String> { read_pt(path).ok_or_else(|| format!("cosyvoice validate: could not read {}", path.display())) };
        let llm_names = read(llm_path)?;
        let flow_names = read(flow_path)?;
        let hift_names = read(hift_path)?;

        let lv = llm_variant(&llm_names).ok_or_else(|| format!("cosyvoice validate: {} is not a recognized llm.pt", llm_path.display()))?;
        let fv = flow_variant(&flow_names).ok_or_else(|| format!("cosyvoice validate: {} is not a recognized flow.pt", flow_path.display()))?;
        let hv = hift_variant(&hift_names).ok_or_else(|| format!("cosyvoice validate: {} is not a recognized hift.pt", hift_path.display()))?;

        if lv != fv || fv != hv {
            return Err(format!(
                "cosyvoice validate: mixed generations chosen - llm is {} ({}), flow is {} ({}), hift is {} ({}) - all three must be the same generation",
                lv.as_str(),
                llm_path.display(),
                fv.as_str(),
                flow_path.display(),
                hv.as_str(),
                hift_path.display()
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
    use checkpoint::torchpt_write::{write as write_pt, TensorOut};
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-cosyvoice-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn t(name: &str, shape: Vec<usize>) -> TensorOut {
        let n: usize = shape.iter().product::<usize>().max(1);
        TensorOut { name: name.to_string(), shape, data: vec![0.0; n] }
    }

    const TOY_TEXT_VOCAB: usize = 200;

    /// Only the tensors this module's own `llm_variant`/`llm_text_vocab_size`
    /// actually read - not the full ~295-tensor Qwen2.5-0.5B manifest, which
    /// a real fixture never needs to write to exercise classification.
    fn write_llm_pt(path: &std::path::Path, generation: Variant) {
        let mut tensors = vec![
            t("llm.model.model.embed_tokens.weight", vec![TOY_TEXT_VOCAB, 32]),
            t("speech_embedding.weight", vec![6561, 32]),
            t("llm_decoder.weight", vec![6561, 32]),
        ];
        if generation == Variant::CosyVoice2 {
            tensors.push(t("llm_embedding.weight", vec![2, 32]));
        }
        write_pt(path.to_str().unwrap(), &tensors).unwrap();
    }

    fn write_flow_pt(path: &std::path::Path, generation: Variant) {
        let mut tensors = vec![t("input_embedding.weight", vec![6561, 80]), t("spk_embed_affine_layer.weight", vec![80, 192])];
        match generation {
            Variant::CosyVoice2 => tensors.push(t("decoder.estimator.down_blocks.0.0.mlp.1.weight", vec![256, 256])),
            Variant::CosyVoice3 => tensors.push(t("decoder.estimator.transformer_blocks.0.attn.to_q.weight", vec![1024, 1024])),
        }
        write_pt(path.to_str().unwrap(), &tensors).unwrap();
    }

    fn write_hift_pt(path: &std::path::Path, generation: Variant) {
        let k = if generation == Variant::CosyVoice2 { 7 } else { 5 };
        let tensors = vec![
            t("f0_predictor.classifier.weight", vec![1, 512]),
            t("m_source.l_linear.weight", vec![1, 8]),
            t("conv_pre.parametrizations.weight.original0", vec![512, 1, 1]),
            t("conv_pre.parametrizations.weight.original1", vec![512, 80, k]),
        ];
        write_pt(path.to_str().unwrap(), &tensors).unwrap();
    }

    fn write_tokenizer_json(path: &std::path::Path, vocab_count: usize) {
        let vocab: serde_json::Map<String, serde_json::Value> = (0..vocab_count).map(|i| (format!("t{i}"), serde_json::json!(i))).collect();
        std::fs::write(path, serde_json::to_vec(&serde_json::json!({"model": {"vocab": vocab}, "added_tokens": []})).unwrap()).unwrap();
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    /// The `llm`/`flow`/`hift` records for one generation, under
    /// `FunAudioLLM/<Generation>-0.5B/` - real world repo naming, real
    /// tensor content. Split out from [`fixture`] so a test that wants BOTH
    /// generations' role files (without a duplicated tokenizer/decoy record)
    /// can call this twice instead.
    fn role_records(dir: &std::path::Path, generation: Variant) -> Vec<ArtifactRecord> {
        let vendor = dir.join("FunAudioLLM").join(format!("{}-0.5B", generation.as_str()));
        std::fs::create_dir_all(&vendor).unwrap();
        let llm_path = vendor.join("llm.pt");
        write_llm_pt(&llm_path, generation);
        let flow_path = vendor.join("flow.pt");
        write_flow_pt(&flow_path, generation);
        let hift_path = vendor.join("hift.pt");
        write_hift_pt(&hift_path, generation);
        vec![complete(llm_path, ArtifactKind::Torch), complete(flow_path, ArtifactKind::Torch), complete(hift_path, ArtifactKind::Torch)]
    }

    /// A full, unambiguous set of records for one generation: `llm`/`flow`/
    /// `hift` under one repo directory, plus a tokenizer under a DIFFERENT
    /// repo - the real `FunAudioLLM/CosyVoice-BlankEN` - published by the
    /// SAME vendor (`FunAudioLLM`, the top-level directory `vendor_dir`
    /// actually keys on) as `FunAudioLLM/CosyVoice2-0.5B`/
    /// `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`, exactly the real-world layout
    /// the module doc describes: "a genuinely different upstream repo" - a
    /// different REPO, same vendor - not an unrelated directory tree.
    fn fixture(dir: &std::path::Path, generation: Variant) -> Vec<ArtifactRecord> {
        let mut records = role_records(dir, generation);

        let tok_vendor = dir.join("FunAudioLLM").join("CosyVoice-BlankEN");
        std::fs::create_dir_all(&tok_vendor).unwrap();
        let tok_path = tok_vendor.join("tokenizer.json");
        write_tokenizer_json(&tok_path, TOY_TEXT_VOCAB);
        records.push(complete(tok_path, ArtifactKind::TokenizerJson));

        // A second, unrelated vendor - without one, `resolve`'s own root
        // inference (the deepest common ancestor of every record) would
        // collapse onto this fixture's single top-level dir instead of
        // `dir`, which breaks vendor_dir-based tokenizer matching the same
        // way flux2's own fixture doc explains.
        records.push(complete(dir.join("other-vendor").join("unrelated.bin"), ArtifactKind::Opaque));
        records
    }

    #[test]
    fn classify_reads_content_not_filenames() {
        let dir = tmp("content-not-filename");
        std::fs::create_dir_all(&dir).unwrap();
        // Named exactly like a real llm.pt, but its content is a flow.pt's -
        // proving classification reads tensor names, not the file's name.
        let suggestive = dir.join("llm.pt");
        write_flow_pt(&suggestive, Variant::CosyVoice2);
        // Same suggestive naming, but genuinely unreadable (not a zip at
        // all) - must classify as nothing.
        let unreadable = dir.join("flow.pt");
        std::fs::write(&unreadable, b"not a torch checkpoint").unwrap();

        let records = vec![complete(suggestive.clone(), ArtifactKind::Torch), complete(unreadable, ArtifactKind::Torch)];
        let out = CosyVoiceSpec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "flow".to_string(), Confidence::Declared)], "{out:?}");
    }

    #[test]
    fn classify_tells_cosyvoice2_and_cosyvoice3_flow_apart_by_tensor_names() {
        let dir = tmp("flow-generation");
        std::fs::create_dir_all(&dir).unwrap();
        let cv2 = dir.join("flow2.pt");
        write_flow_pt(&cv2, Variant::CosyVoice2);
        let cv3 = dir.join("flow3.pt");
        write_flow_pt(&cv3, Variant::CosyVoice3);

        let records = vec![complete(cv2.clone(), ArtifactKind::Torch), complete(cv3.clone(), ArtifactKind::Torch)];
        let names2 = read_pt(&cv2).unwrap();
        let names3 = read_pt(&cv3).unwrap();
        assert_eq!(flow_variant(&names2), Some(Variant::CosyVoice2));
        assert_eq!(flow_variant(&names3), Some(Variant::CosyVoice3));

        let out = CosyVoiceSpec.classify(&records, dir.as_path());
        assert_eq!(out.iter().filter(|(_, r, _)| r == "flow").count(), 2, "{out:?}");
    }

    #[test]
    fn classify_tells_cosyvoice2_and_cosyvoice3_hift_apart_by_shape_not_name() {
        let dir = tmp("hift-generation");
        std::fs::create_dir_all(&dir).unwrap();
        let cv2 = dir.join("hift2.pt");
        write_hift_pt(&cv2, Variant::CosyVoice2);
        let cv3 = dir.join("hift3.pt");
        write_hift_pt(&cv3, Variant::CosyVoice3);
        let names2 = read_pt(&cv2).unwrap();
        let names3 = read_pt(&cv3).unwrap();
        assert_eq!(hift_variant(&names2), Some(Variant::CosyVoice2));
        assert_eq!(hift_variant(&names3), Some(Variant::CosyVoice3));
    }

    #[test]
    fn classify_tells_cosyvoice2_and_cosyvoice3_llm_apart_by_llm_embedding_presence() {
        let dir = tmp("llm-generation");
        std::fs::create_dir_all(&dir).unwrap();
        let cv2 = dir.join("llm2.pt");
        write_llm_pt(&cv2, Variant::CosyVoice2);
        let cv3 = dir.join("llm3.pt");
        write_llm_pt(&cv3, Variant::CosyVoice3);
        let names2 = read_pt(&cv2).unwrap();
        let names3 = read_pt(&cv3).unwrap();
        assert_eq!(llm_variant(&names2), Some(Variant::CosyVoice2));
        assert_eq!(llm_variant(&names3), Some(Variant::CosyVoice3));
    }

    /// Same real-world case flux2's own tests pin: a same-vendor tokenizer
    /// belonging to a DIFFERENT, unrelated model must not classify just
    /// because it shares the vendor directory with a real `llm.pt`.
    #[test]
    fn classify_tokenizer_requires_vocab_compatibility_not_only_the_same_vendor() {
        let dir = tmp("tokenizer-vocab-mismatch");
        let mut records = fixture(&dir, Variant::CosyVoice2);
        let other_model_dir = dir.join("FunAudioLLM").join("Other-Model");
        std::fs::create_dir_all(&other_model_dir).unwrap();
        let mismatched_tok = other_model_dir.join("tokenizer.json");
        write_tokenizer_json(&mismatched_tok, TOY_TEXT_VOCAB * 3);
        records.push(complete(mismatched_tok.clone(), ArtifactKind::TokenizerJson));

        let out = CosyVoiceSpec.classify(&records, dir.as_path());
        let tokenizer_candidates: Vec<_> = out.iter().filter(|(_, role, _)| role == "tokenizer").collect();
        assert_eq!(tokenizer_candidates.len(), 1, "{out:?}");
        let (idx, ..) = tokenizer_candidates[0];
        assert_ne!(records[*idx].path, mismatched_tok, "the vocab-mismatched same-vendor tokenizer must never classify: {out:?}");
    }

    #[test]
    fn a_full_cosyvoice2_fixture_resolves_cleanly() {
        let dir = tmp("resolves-cv2");
        let records = fixture(&dir, Variant::CosyVoice2);
        let spec = CosyVoiceSpec;
        let specs: [&dyn ArchSpec; 1] = [&spec];
        let out = resolve("cosyvoice", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                assert_eq!(a.variant.as_deref(), Some("cosyvoice2"));
                assert!(a.roles.contains_key("llm") && a.roles.contains_key("flow") && a.roles.contains_key("hift") && a.roles.contains_key("tokenizer"), "{a:?}");
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn a_full_cosyvoice3_fixture_resolves_cleanly() {
        let dir = tmp("resolves-cv3");
        let records = fixture(&dir, Variant::CosyVoice3);
        let spec = CosyVoiceSpec;
        let specs: [&dyn ArchSpec; 1] = [&spec];
        let out = resolve("cosyvoice", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => assert_eq!(a.variant.as_deref(), Some("cosyvoice3")),
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    /// THE headline real-world case: a store holding BOTH generations' full
    /// role sets must never let `resolve` silently pick one - both
    /// generations classify at the same (Declared) confidence for every
    /// role, so the ordinary role-ambiguity path already reports this,
    /// with no cosyvoice-specific handling needed.
    #[test]
    fn both_generations_present_at_once_is_ambiguous_never_a_silent_pick() {
        let dir = tmp("both-generations");
        let mut records = fixture(&dir, Variant::CosyVoice2);
        records.extend(role_records(&dir, Variant::CosyVoice3));

        let spec = CosyVoiceSpec;
        let specs: [&dyn ArchSpec; 1] = [&spec];
        let out = resolve("cosyvoice", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Ambiguous(a) => {
                assert_eq!(a.question, Question::Role { role: "llm".to_string() }, "{a:?}");
                assert_eq!(a.choices.len(), 2, "{a:?}");
            }
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }

    /// The last gate: `llm`/`flow`/`hift` from DIFFERENT generations must be
    /// rejected before any weight is loaded, even when each one individually
    /// classifies fine.
    #[test]
    fn mixed_generation_roles_are_rejected_by_validate() {
        let dir = tmp("mixed-generation");
        std::fs::create_dir_all(&dir).unwrap();
        let llm_path = dir.join("llm.pt");
        write_llm_pt(&llm_path, Variant::CosyVoice2);
        let flow_path = dir.join("flow.pt");
        write_flow_pt(&flow_path, Variant::CosyVoice3);
        let hift_path = dir.join("hift.pt");
        write_hift_pt(&hift_path, Variant::CosyVoice2);

        let assembly = Assembly {
            id: "local/cosyvoice-mixed".to_string(),
            arch: "cosyvoice".to_string(),
            variant: Some("cosyvoice2".to_string()),
            roles: BTreeMap::from([("llm".to_string(), llm_path), ("flow".to_string(), flow_path), ("hift".to_string(), hift_path)]),
            provenance: Vec::new(),
        };
        let err = CosyVoiceSpec.validate(&assembly).unwrap_err();
        assert!(err.contains("cosyvoice2"), "{err}");
        assert!(err.contains("cosyvoice3"), "{err}");
    }

    #[test]
    fn an_explicit_role_override_collapses_an_ambiguity() {
        let dir = tmp("override-collapses");
        let mut records = fixture(&dir, Variant::CosyVoice2);
        let (cv2_llm, cv2_flow, cv2_hift) = (records[0].path.clone(), records[1].path.clone(), records[2].path.clone());
        records.extend(role_records(&dir, Variant::CosyVoice3));

        let spec = CosyVoiceSpec;
        let specs: [&dyn ArchSpec; 1] = [&spec];
        let mut overrides = BTreeMap::new();
        overrides.insert("llm".to_string(), cv2_llm.to_string_lossy().into_owned());
        overrides.insert("flow".to_string(), cv2_flow.to_string_lossy().into_owned());
        overrides.insert("hift".to_string(), cv2_hift.to_string_lossy().into_owned());
        let out = resolve("cosyvoice", &records, &specs, &overrides);
        match out {
            Resolution::Resolved(a) => assert_eq!(a.variant.as_deref(), Some("cosyvoice2")),
            other => panic!("expected Resolved, got {other:?}"),
        }
    }
}
