// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CodeFormer's [`ArchSpec`]: which on-disk `.pt`/`.pth` checkpoint satisfies
//! the single `weights` role.
//!
//! Unlike [`crate::config::CodeFormerConfig`]'s sibling in `crates/rrdbnet`
//! (`rrdbnet::spec::RrdbnetSpec`), there is no `from_tensors` here: the
//! released `codeformer.pth` is ONE fixed preset
//! (`CodeFormerConfig::codeformer()`, see that constructor's own doc), not a
//! family of variants differing in width/depth/scale, so there is nothing to
//! derive a config FROM - classification instead checks that a handful of
//! tensor names the reference's `CodeFormer` class alone declares (not the
//! `VQAutoEncoder` it subclasses) are present with the exact shape that fixed
//! preset implies. A bare `vqgan_code1024.pth` (VQGAN's own released
//! checkpoint, `crates/vqgan`) carries NONE of them - only the 329 VQGAN
//! tensors CodeFormer's 515 is a superset of - so this cannot mistake one
//! for the other.
//!
//! `Confidence::Derived`, not `Declared`: a raw `torch.save` state dict has
//! no header/config field that names an architecture the way a GGUF
//! `general.architecture` KV or an HF `config.json` does (the same reasoning
//! `rrdbnet::spec`'s own doc gives for the same file kind).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

use crate::config::CodeFormerConfig;

pub struct CodeFormerSpec;

const ROLES: &[&str] = &["weights"];

/// `basicsr` top-level wrapper keys - mirrors `vqgan::import::STATE_KEYS`
/// exactly (both crates read the same reference framework's checkpoints),
/// but over shapes only: `checkpoint::torchpt::read_shapes` never decodes a
/// tensor's storage bytes, so this cannot reuse `vqgan::import::
/// strip_state_prefix`, which operates on the fully-decoded `Tensors` map.
const STATE_KEYS: [&str; 3] = ["params_ema", "params", "state_dict"];

fn stripped_shapes(path: &Path) -> Option<HashMap<String, Vec<usize>>> {
    let raw = checkpoint::torchpt::read_shapes(path.to_string_lossy().as_ref()).ok()?;
    for key in STATE_KEYS {
        let dot = format!("{key}.");
        if !raw.is_empty() && raw.iter().all(|(n, _)| n.starts_with(&dot)) {
            return Some(raw.into_iter().map(|(n, s)| (n[dot.len()..].to_string(), s)).collect());
        }
    }
    Some(raw.into_iter().collect())
}

/// A handful of tensors ONLY the CodeFormer transformer/CFT declares (never
/// the `VQAutoEncoder` it subclasses), each checked against the exact shape
/// [`CodeFormerConfig::codeformer`]'s fixed preset implies: the code-index
/// head, the feature-embedding projection, the positional embedding, one
/// full transformer layer's fused attention projection, and one
/// controllable-feature-transformation tap. All five present at the right
/// shape is enough to be confident this is a genuine, structurally intact
/// `codeformer.pth` rather than a bare VQGAN checkpoint or unrelated file -
/// `crate::import::load` is what fully validates every one of the 515
/// tensors at actual load time; this only needs enough to classify.
fn looks_like_codeformer(shapes: &HashMap<String, Vec<usize>>, cfg: &CodeFormerConfig) -> bool {
    let e = cfg.dim_embd as usize;
    let get = |n: &str| shapes.get(n).map(Vec::as_slice);

    if get("position_emb") != Some(&[cfg.latent_size as usize, e]) {
        return false;
    }
    if get("feat_emb.weight") != Some(&[e, cfg.vqgan.emb_dim as usize]) {
        return false;
    }
    if get("idx_pred_layer.1.weight") != Some(&[cfg.vqgan.codebook_size as usize, e]) {
        return false;
    }
    if get(&format!("{}.self_attn.in_proj_weight", CodeFormerConfig::layer_prefix(0))) != Some(&[3 * e, e]) {
        return false;
    }
    let Some(tap) = cfg.taps().first().copied() else { return false };
    let c = tap.channels as usize;
    get(&format!("{}.scale.0.weight", CodeFormerConfig::fuse_prefix(&tap))) == Some(&[c, c, 3, 3])
}

impl ArchSpec for CodeFormerSpec {
    fn arch(&self) -> &'static str {
        "codeformer"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let cfg = CodeFormerConfig::codeformer();
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            if !rec.usable() || rec.kind != ArtifactKind::Torch {
                continue;
            }
            if !matches!(rec.path.extension().and_then(|e| e.to_str()), Some("pt" | "pth")) {
                continue;
            }
            let Some(shapes) = stripped_shapes(&rec.path) else { continue };
            if looks_like_codeformer(&shapes, &cfg) {
                out.push((idx, "weights".to_string(), Confidence::Derived));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("weights").ok_or("codeformer assemble: no weights chosen")?;
        // One fixed preset, no variant dimension to report - unlike
        // `rrdbnet::spec`'s `x{scale}-{blocks}b`, there is nothing here that
        // varies checkpoint to checkpoint.
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/codeformer".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("weights").ok_or("codeformer validate: assembly has no weights role")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::inventory::Completeness;
    use brain_modelstore::resolve::{resolve, Resolution};
    use checkpoint::torchpt_write::TensorOut;
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-codeformer-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    fn t(name: impl Into<String>, shape: Vec<usize>) -> TensorOut {
        let n: usize = shape.iter().product::<usize>().max(1);
        TensorOut { name: name.into(), shape, data: vec![0.0; n] }
    }

    /// Exactly the five tensors [`looks_like_codeformer`] reads, at the real
    /// `codeformer()` preset's shapes, under the `params_ema.` prefix a real
    /// release checkpoint uses - the minimum a fixture needs, not the whole
    /// 515-tensor checkpoint [`crate::import`]'s own tests build.
    fn write_codeformer_pt(path: &Path) {
        let cfg = CodeFormerConfig::codeformer();
        let e = cfg.dim_embd as usize;
        let tap = cfg.taps()[0];
        let c = tap.channels as usize;
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let tensors = vec![
            t("params_ema.position_emb", vec![cfg.latent_size as usize, e]),
            t("params_ema.feat_emb.weight", vec![e, cfg.vqgan.emb_dim as usize]),
            t("params_ema.idx_pred_layer.1.weight", vec![cfg.vqgan.codebook_size as usize, e]),
            t(format!("params_ema.{}.self_attn.in_proj_weight", CodeFormerConfig::layer_prefix(0)), vec![3 * e, e]),
            t(format!("params_ema.{}.scale.0.weight", CodeFormerConfig::fuse_prefix(&tap)), vec![c, c, 3, 3]),
        ];
        checkpoint::torchpt_write::write(path.to_str().unwrap(), &tensors).unwrap();
    }

    /// A bare VQGAN checkpoint - real tensor names CodeFormer's own encoder
    /// shares (it subclasses `VQAutoEncoder`), but none of the five
    /// CodeFormer-only ones - must never classify as `codeformer`'s weights.
    fn write_vqgan_only_pt(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let tensors = vec![t("params_ema.quantize.embedding.weight", vec![1024, 256]), t("params_ema.encoder.blocks.0.weight", vec![64, 3, 3, 3])];
        checkpoint::torchpt_write::write(path.to_str().unwrap(), &tensors).unwrap();
    }

    #[test]
    fn classify_recognizes_a_real_codeformer_checkpoint() {
        let dir = tmp("real-shape");
        let path = dir.join("sczhou").join("codeformer.pth");
        write_codeformer_pt(&path);

        let records = vec![complete(path, ArtifactKind::Torch)];
        let out = CodeFormerSpec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "weights".to_string(), Confidence::Derived)], "{out:?}");
    }

    /// The headline case this spec exists to get right: CodeFormer's 515
    /// tensors are a SUPERSET of VQGAN's 329, so a bare `vqgan_code1024.pth`
    /// (real tensor names, just none of the CodeFormer-only ones) must not
    /// be mistaken for a `codeformer.pth`.
    #[test]
    fn classify_rejects_a_bare_vqgan_checkpoint() {
        let dir = tmp("vqgan-only");
        let path = dir.join("weights").join("vqgan_code1024.pth");
        write_vqgan_only_pt(&path);

        let records = vec![complete(path, ArtifactKind::Torch)];
        let out = CodeFormerSpec.classify(&records, dir.as_path());
        assert_eq!(out, Vec::new(), "{out:?}");
    }

    #[test]
    fn classify_rejects_an_unreadable_pth() {
        let dir = tmp("garbage-pth");
        let path = dir.join("sczhou").join("not-really-codeformer.pth");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not a real torch.save archive").unwrap();

        let records = vec![complete(path, ArtifactKind::Torch)];
        let out = CodeFormerSpec.classify(&records, dir.as_path());
        assert_eq!(out, Vec::new(), "{out:?}");
    }

    #[test]
    fn resolves_end_to_end_with_no_variant() {
        let dir = tmp("resolve-end-to-end");
        let path = dir.join("sczhou").join("codeformer.pth");
        write_codeformer_pt(&path);

        let records = vec![complete(path.clone(), ArtifactKind::Torch)];
        let spec = CodeFormerSpec;
        let specs: [&dyn ArchSpec; 1] = [&spec];
        let out = resolve("codeformer", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                assert_eq!(a.roles["weights"], path);
                assert_eq!(a.variant, None);
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn two_candidates_at_equal_confidence_is_ambiguous_not_a_silent_pick() {
        let dir = tmp("two-candidates");
        let a = dir.join("sczhou").join("codeformer.pth");
        write_codeformer_pt(&a);
        let b = dir.join("some-mirror").join("codeformer-copy.pth");
        write_codeformer_pt(&b);

        let records = vec![complete(a, ArtifactKind::Torch), complete(b, ArtifactKind::Torch)];
        let spec = CodeFormerSpec;
        let specs: [&dyn ArchSpec; 1] = [&spec];
        let out = resolve("codeformer", &records, &specs, &BTreeMap::new());
        assert!(matches!(out, Resolution::Ambiguous(_)), "{out:?}");
    }

    /// Regression pin, same discipline `rrdbnet::spec`'s own equivalent test
    /// documents: every other test here hand-builds an `ArtifactRecord` with
    /// an explicitly chosen `kind`, so a `classify` that checked the WRONG
    /// `ArtifactKind` would still pass every one of them. This one goes
    /// through the REAL scanner instead - the same one `loader::
    /// resolve_structured` uses in production.
    #[test]
    fn classify_recognizes_a_real_pth_file_scanned_by_the_real_inventory_scanner() {
        let dir = tmp("real-scanner");
        let path = dir.join("sczhou").join("codeformer.pth");
        write_codeformer_pt(&path);

        let records = brain_modelstore::inventory::scan(&dir);
        assert_eq!(records.len(), 1, "{records:?}");
        assert_eq!(records[0].kind, ArtifactKind::Torch, "{records:?}");

        let out = CodeFormerSpec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "weights".to_string(), Confidence::Derived)], "{out:?}");
    }
}
