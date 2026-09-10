// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ArcFace's rules for the model-store resolver: which ONNX graph on disk is
//! insightface's released IResNet-100 identity embedder.
//!
//! One role, `weights` - the `glintr100.onnx` graph itself. It is identified
//! by its own initializer names (`onnx::header`, which seeks past every
//! tensor's `raw_data` rather than reading it), never by its filename: the
//! `ArchSpec` contract forbids reading tensor bytes, and a store may hold any
//! number of unrelated `.onnx` graphs that a name check could not tell apart.
//!
//! Swedish Embedded AB implements content-based model identification for
//! clients whose pipelines mix ONNX, safetensors and GGUF releases in one
//! store. If your team needs weight resolution that identifies graphs by what
//! they contain, you can procure our services by emailing
//! info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

/// The role name this architecture resolves - the released ONNX graph.
pub const ROLES: &[&str] = &["weights"];

/// The embedding width every ArcFace release in this family produces, and the
/// width `crates/arcface` builds its graph for.
const EMBEDDING_DIM: i64 = 512;

/// The final fully-connected layer producing the identity embedding - present
/// in every ArcFace release.
const FC_WEIGHT: &str = "fc.weight";
/// The deepest residual block's first batch-norm. IResNet-100's stage layout
/// is `layer1`×3, `layer2`×13, `layer3`×30, `layer4`×3, so `layer3.29` exists
/// only at depth 100 - this is what separates the released IResNet-100
/// (`glintr100.onnx`) from a differently-sized ArcFace that carries the same
/// `fc.weight`.
const DEEPEST_BLOCK_BN: &str = "layer3.29.bn1.weight";

/// Whether `path` is a released ArcFace IResNet-100 graph, from its own
/// initializer names and the embedding width they declare.
pub fn is_arcface_graph(path: &Path) -> bool {
    let Ok(inits) = onnx::header::read_initializers(path) else { return false };
    let fc = inits.iter().find(|i| i.name == FC_WEIGHT);
    // `fc.weight` is `[embedding, features]`, so its first dimension is the
    // embedding width the graph actually emits - a real declared shape, not a
    // guess from the file's name.
    fc.is_some_and(|i| i.dims.first().copied() == Some(EMBEDDING_DIM)) && inits.iter().any(|i| i.name == DEEPEST_BLOCK_BN)
}

/// Classify every released ArcFace graph in `records` under `role`, appending
/// to `out` in place. Exposed so `pulid::spec` reuses it for its own
/// `arcface` role rather than carrying a second idea of what ArcFace is.
pub fn classify_weights(records: &[ArtifactRecord], role: &str, out: &mut Vec<(usize, String, Confidence)>) {
    for (idx, rec) in records.iter().enumerate() {
        if rec.usable() && rec.kind == ArtifactKind::Onnx && is_arcface_graph(&rec.path) {
            // `Derived`, not `Declared`: an ONNX graph carries no
            // architecture field to declare itself with, so this is computed
            // from the shapes and names its own header reports.
            out.push((idx, role.to_string(), Confidence::Derived));
        }
    }
}

/// ArcFace's [`ArchSpec`].
pub struct ArcFaceSpec;

impl ArchSpec for ArcFaceSpec {
    fn arch(&self) -> &'static str {
        "arcface"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        classify_weights(records, "weights", &mut out);
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        chosen.get("weights").ok_or("arcface assemble: no weights chosen")?;
        // Only one released graph shape is supported, and `classify` already
        // proved this is it - there is no variant left to choose.
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/arcface-iresnet100".to_string(), variant: Some("iresnet100".to_string()) }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let path = assembly.roles.get("weights").ok_or("arcface validate: assembly has no weights role")?;
        if !is_arcface_graph(path) {
            return Err(format!("arcface validate: {} is not a released ArcFace IResNet-100 graph", path.display()));
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
        std::env::temp_dir().join(format!("brain-arcface-spec-{tag}-{}-{n}", std::process::id()))
    }

    /// An ONNX graph carrying exactly the named initializers given, with real
    /// (if tiny) payloads - the header walker must skip those payloads, so
    /// their size is irrelevant to what this test proves.
    fn write_graph(path: &Path, tensors: &[(&str, Vec<i64>)]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let initializer = tensors
            .iter()
            .map(|(name, dims)| onnx::onnx::TensorProto { name: (*name).to_string(), dims: dims.clone(), data_type: 1, raw_data: vec![0u8; 16], ..Default::default() })
            .collect();
        let graph = onnx::onnx::GraphProto { name: "torch-jit-export".to_string(), initializer, ..Default::default() };
        let model = onnx::onnx::ModelProto { ir_version: 6, producer_name: "pytorch".to_string(), graph: Some(graph), ..Default::default() };
        std::fs::write(path, onnx::encode_model(&model)).unwrap();
    }

    fn arcface_tensors() -> Vec<(&'static str, Vec<i64>)> {
        vec![(FC_WEIGHT, vec![EMBEDDING_DIM, 25088]), (DEEPEST_BLOCK_BN, vec![256]), ("bn2.weight", vec![512])]
    }

    fn onnx_record(path: &Path) -> ArtifactRecord {
        ArtifactRecord { path: path.to_path_buf(), size: 1, mtime_ns: 0, kind: ArtifactKind::Onnx, completeness: Completeness::Complete }
    }

    /// The released embedder resolves on its own, from graph content, while a
    /// different ONNX graph sitting beside it in the same directory (the
    /// detector antelopev2 really does ship alongside it) is not mistaken for
    /// it.
    #[test]
    fn the_released_graph_resolves_and_the_detector_beside_it_does_not() {
        let dir = tmp("resolve");
        let repo = dir.join("DIAMONIK7777").join("antelopev2");
        let embedder = repo.join("glintr100.onnx");
        let detector = repo.join("scrfd_10g_bnkps.onnx");
        write_graph(&embedder, &arcface_tensors());
        write_graph(&detector, &[("bbox_head.stride_kps.(8, 8).weight", vec![20, 256, 1, 1])]);

        let records = vec![onnx_record(&embedder), onnx_record(&detector)];
        let specs: Vec<&dyn ArchSpec> = vec![&ArcFaceSpec];
        match resolve("arcface", &records, &specs, &BTreeMap::new()) {
            Resolution::Resolved(a) => assert_eq!(a.roles["weights"], embedder),
            other => panic!("expected Resolved, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Identification is by content, not by name: a graph named
    /// `glintr100.onnx` that is not actually the released embedder must not
    /// classify, and the real embedder must classify under any name.
    #[test]
    fn classification_follows_content_not_the_filename() {
        let dir = tmp("content");
        let impostor = dir.join("vendor").join("glintr100.onnx");
        write_graph(&impostor, &[("something.else.weight", vec![7])]);
        assert!(!is_arcface_graph(&impostor), "a mis-named graph must not classify");

        let renamed = dir.join("vendor").join("hand-placed-embedder.onnx");
        write_graph(&renamed, &arcface_tensors());
        assert!(is_arcface_graph(&renamed), "the real graph must classify under any name");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A same-family ArcFace at a different depth carries `fc.weight` too -
    /// the deepest-block check is what keeps it from being served as the
    /// IResNet-100 `crates/arcface` actually builds.
    #[test]
    fn a_shallower_arcface_is_not_accepted_as_the_released_iresnet100() {
        let dir = tmp("depth");
        let shallow = dir.join("vendor").join("arcface-r50.onnx");
        write_graph(&shallow, &[(FC_WEIGHT, vec![EMBEDDING_DIM, 25088]), ("layer3.5.bn1.weight", vec![256])]);
        assert!(!is_arcface_graph(&shallow));
        std::fs::remove_dir_all(&dir).ok();
    }
}
