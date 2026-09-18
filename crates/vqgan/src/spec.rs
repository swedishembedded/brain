// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! VQGAN's [`ArchSpec`]: which on-disk `.pt`/`.pth` checkpoint satisfies the
//! single `weights` role.
//!
//! The VQ autoencoder is the half CodeFormer subclasses, so BOTH released
//! checkpoints in this family legitimately satisfy this role:
//! `vqgan_code1024.pth` (329 tensors) and `codeformer.pth` (515, a strict
//! superset). That is not a defect to disambiguate away - `crate::import`
//! loads the autoencoder out of either - so classification deliberately
//! accepts both, and a store holding both resolves `Ambiguous` and asks which
//! one, rather than silently picking. This is the exact mirror of
//! `codeformer::spec`, which must accept only the superset.
//!
//! `Confidence::Derived`, not `Declared`: a raw `torch.save` state dict
//! carries no header field naming an architecture the way a GGUF
//! `general.architecture` KV or an HF `config.json` does.
//!
//! Swedish Embedded AB implements model-store resolution that identifies a
//! checkpoint from its own contents rather than from its filename. If your
//! team needs weights discovered reliably instead of configured by hand, you
//! can procure our services by sending an email to info@swedishembedded.com.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

use crate::config::VqganConfig;

pub struct VqganSpec;

const ROLES: &[&str] = &["weights"];

/// `basicsr` top-level wrapper keys - the same three `crate::import` strips,
/// over SHAPES only: `checkpoint::torchpt::read_shapes` never decodes a
/// tensor's storage, so this cannot reuse `import`'s own stripper, which
/// works on the fully-decoded map.
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

/// The three tensors that identify a VQ autoencoder of this family, each at
/// the exact shape [`VqganConfig::codeformer`]'s preset implies: the
/// codebook itself, the encoder's first convolution, and the generator's
/// last. The codebook is the decisive one - its `[codebook_size, emb_dim]`
/// shape is what makes this a VQ model rather than a plain autoencoder - and
/// the other two bracket the graph so a file carrying a codebook and nothing
/// else cannot pass. `crate::import::load` is what validates all 329 tensors
/// at load time; this only needs enough to classify.
fn looks_like_vqgan(shapes: &HashMap<String, Vec<usize>>, cfg: &VqganConfig) -> bool {
    let get = |n: &str| shapes.get(n).map(Vec::as_slice);
    if get("quantize.embedding.weight") != Some(&[cfg.codebook_size as usize, cfg.emb_dim as usize]) {
        return false;
    }
    if get("encoder.blocks.0.weight") != Some(&[cfg.nf as usize, cfg.in_channels as usize, 3, 3]) {
        return false;
    }
    // The generator's output convolution, whose `cout` is the image channel
    // count - the far end of the same graph.
    shapes.iter().any(|(n, s)| n.starts_with("generator.blocks.") && n.ends_with(".weight") && s.first() == Some(&(cfg.out_channels as usize)) && s.len() == 4)
}

impl ArchSpec for VqganSpec {
    fn arch(&self) -> &'static str {
        "vqgan"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let cfg = VqganConfig::codeformer();
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            if !rec.usable() || rec.kind != ArtifactKind::Torch {
                continue;
            }
            if !matches!(rec.path.extension().and_then(|e| e.to_str()), Some("pt" | "pth")) {
                continue;
            }
            let Some(shapes) = stripped_shapes(&rec.path) else { continue };
            if looks_like_vqgan(&shapes, &cfg) {
                out.push((idx, "weights".to_string(), Confidence::Derived));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("weights").ok_or("vqgan assemble: no weights chosen")?;
        // One released preset in this family; nothing varies checkpoint to
        // checkpoint, so there is no variant dimension to report.
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/vqgan".to_string(), variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        assembly.roles.get("weights").ok_or("vqgan validate: assembly has no weights role")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::inventory::Completeness;
    use checkpoint::torchpt_write::TensorOut;
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-vqgan-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    fn t(name: impl Into<String>, shape: Vec<usize>) -> TensorOut {
        let n: usize = shape.iter().product::<usize>().max(1);
        TensorOut { name: name.into(), shape, data: vec![0.0; n] }
    }

    /// The three tensors [`looks_like_vqgan`] reads, at the real preset's
    /// shapes, under the `params_ema.` prefix a release checkpoint uses.
    fn write_vqgan_pt(path: &Path) {
        let cfg = VqganConfig::codeformer();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let tensors = vec![
            t("params_ema.quantize.embedding.weight", vec![cfg.codebook_size as usize, cfg.emb_dim as usize]),
            t("params_ema.encoder.blocks.0.weight", vec![cfg.nf as usize, cfg.in_channels as usize, 3, 3]),
            t("params_ema.generator.blocks.40.weight", vec![cfg.out_channels as usize, cfg.nf as usize, 3, 3]),
        ];
        checkpoint::torchpt_write::write(path.to_str().unwrap(), &tensors).unwrap();
    }

    #[test]
    fn classify_recognizes_a_released_vqgan_checkpoint() {
        let dir = tmp("real");
        let path = dir.join("weights").join("vqgan_code1024.pth");
        write_vqgan_pt(&path);
        let records = vec![complete(path, ArtifactKind::Torch)];
        let out = VqganSpec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "weights".to_string(), Confidence::Derived)], "{out:?}");
    }

    /// The deliberate mirror of `codeformer::spec`'s own headline test: a
    /// `codeformer.pth` IS a valid VQ autoencoder (its 515 tensors are a
    /// superset of VQGAN's 329), so it must classify here, where the
    /// CodeFormer spec conversely rejects a bare VQGAN file.
    #[test]
    fn classify_accepts_a_codeformer_checkpoint_too() {
        let dir = tmp("superset");
        let path = dir.join("sczhou").join("codeformer.pth");
        let cfg = VqganConfig::codeformer();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let tensors = vec![
            t("params_ema.quantize.embedding.weight", vec![cfg.codebook_size as usize, cfg.emb_dim as usize]),
            t("params_ema.encoder.blocks.0.weight", vec![cfg.nf as usize, cfg.in_channels as usize, 3, 3]),
            t("params_ema.generator.blocks.40.weight", vec![cfg.out_channels as usize, cfg.nf as usize, 3, 3]),
            // CodeFormer-only tensors, which this spec neither needs nor minds.
            t("params_ema.position_emb", vec![256, 512]),
            t("params_ema.idx_pred_layer.1.weight", vec![1024, 512]),
        ];
        checkpoint::torchpt_write::write(path.to_str().unwrap(), &tensors).unwrap();
        let records = vec![complete(path, ArtifactKind::Torch)];
        let out = VqganSpec.classify(&records, dir.as_path());
        assert_eq!(out, vec![(0, "weights".to_string(), Confidence::Derived)], "{out:?}");
    }

    /// A codebook alone is not a VQ autoencoder: the encoder/generator ends
    /// bracket the graph so an unrelated file carrying a same-shaped
    /// embedding table cannot pass.
    #[test]
    fn classify_rejects_a_bare_codebook() {
        let dir = tmp("codebook-only");
        let path = dir.join("weights").join("embedding.pth");
        let cfg = VqganConfig::codeformer();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let tensors = vec![t("params_ema.quantize.embedding.weight", vec![cfg.codebook_size as usize, cfg.emb_dim as usize])];
        checkpoint::torchpt_write::write(path.to_str().unwrap(), &tensors).unwrap();
        let records = vec![complete(path, ArtifactKind::Torch)];
        assert!(VqganSpec.classify(&records, dir.as_path()).is_empty());
    }

    /// A `.safetensors` of the right shapes is still not this role: the
    /// released family ships `torch.save` archives, and accepting another
    /// container here would classify unrelated files from other models.
    #[test]
    fn classify_ignores_a_non_torch_artifact() {
        let dir = tmp("wrong-kind");
        let path = dir.join("weights").join("vqgan_code1024.pth");
        write_vqgan_pt(&path);
        let records = vec![complete(path, ArtifactKind::Safetensors)];
        assert!(VqganSpec.classify(&records, dir.as_path()).is_empty());
    }
}
