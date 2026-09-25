// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LPIPS's rules for the model-store resolver: which two files in the store
//! are the AlexNet trunk and the v0.1 linear heads.
//!
//! Two roles, `trunk` and `heads`, each identified by its own tensor shapes
//! (header-only), never by a filename: the trunk is whatever carries all five
//! of AlexNet's `features.*` convolutions at their torchvision shapes, the
//! heads whatever carries `lin0`..`lin4` at `[1, C, 1, 1]` for the same five
//! channel counts. They come from two upstream releases (torchvision and the
//! LPIPS repository), so the assembly is a `local/` one.
//!
//! Swedish Embedded AB implements content-based checkpoint identification and
//! evaluation pipelines for its clients. If your team needs expertise in
//! perceptual image metrics or model distribution, you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{describe_ambiguity, describe_missing, ArchSpec, AssembleOutcome, AssembledVariant, Confidence, Resolution};
use capability::Assembly;

use crate::config::{head_weight, trunk_bias, trunk_weight, TRUNK};

/// The resolver's name for this metric.
pub const ARCH: &str = "lpips";
/// The roles, in the order the resolver reports them.
pub const ROLES: &[&str] = &["trunk", "heads"];
/// What the two files assemble into.
pub const ID: &str = "local/lpips-alex-v0.1";
/// What puts both files into the store.
pub const FETCH_TOOL: &str = "tools/goldens/lpips_dump_reference.py";

/// The declared shapes in `path` of the tensors LPIPS reads, header-only: a
/// `.safetensors` header or a `torch.save` pickle, never tensor bytes.
fn shapes(path: &Path, kind: ArtifactKind) -> HashMap<String, Vec<usize>> {
    let p = path.to_string_lossy();
    match kind {
        ArtifactKind::Safetensors => {
            let Ok(m) = checkpoint::mmap::MmapSafetensors::open(p.as_ref()) else { return HashMap::new() };
            let mut out = HashMap::new();
            for (i, _) in TRUNK.iter().enumerate() {
                for name in [trunk_weight(i), trunk_bias(i), head_weight(i)] {
                    if let Some(s) = m.shape(&name) {
                        out.insert(name, s.to_vec());
                    }
                }
            }
            out
        }
        ArtifactKind::Torch => checkpoint::torchpt::read_shapes(p.as_ref()).map(|v| v.into_iter().collect()).unwrap_or_default(),
        _ => HashMap::new(),
    }
}

fn is_trunk(s: &HashMap<String, Vec<usize>>) -> bool {
    TRUNK.iter().enumerate().all(|(i, c)| {
        s.get(&trunk_weight(i)).is_some_and(|w| *w == [c.cout as usize, c.cin as usize, c.k as usize, c.k as usize])
            && s.get(&trunk_bias(i)).is_some_and(|b| *b == [c.cout as usize])
    })
}

fn is_heads(s: &HashMap<String, Vec<usize>>) -> bool {
    TRUNK.iter().enumerate().all(|(i, c)| s.get(&head_weight(i)).is_some_and(|w| *w == [1, c.cout as usize, 1, 1]))
}

/// Whether `path` holds LPIPS's AlexNet trunk, from its own header.
pub fn is_trunk_file(path: &Path) -> bool {
    kind_of(path).is_some_and(|k| is_trunk(&shapes(path, k)))
}

/// Whether `path` holds LPIPS v0.1's AlexNet heads, from its own header.
pub fn is_heads_file(path: &Path) -> bool {
    kind_of(path).is_some_and(|k| is_heads(&shapes(path, k)))
}

fn kind_of(path: &Path) -> Option<ArtifactKind> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("safetensors") => Some(ArtifactKind::Safetensors),
        Some("pt" | "pth") => Some(ArtifactKind::Torch),
        _ => None,
    }
}

/// LPIPS's [`ArchSpec`].
pub struct LpipsSpec;

impl ArchSpec for LpipsSpec {
    fn arch(&self) -> &'static str {
        ARCH
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            if !rec.usable() || !matches!(rec.kind, ArtifactKind::Safetensors | ArtifactKind::Torch) {
                continue;
            }
            let s = shapes(&rec.path, rec.kind);
            // `Derived`: a bare checkpoint declares no architecture, so the
            // role is computed from its shapes.
            if is_trunk(&s) {
                out.push((idx, "trunk".to_string(), Confidence::Derived));
            }
            if is_heads(&s) {
                out.push((idx, "heads".to_string(), Confidence::Derived));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        for role in ROLES {
            chosen.get(*role).ok_or_else(|| format!("lpips assemble: no {role} chosen"))?;
        }
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: ID.to_string(), variant: Some("v0.1-alex".to_string()) }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let role = |r: &str| assembly.roles.get(r).ok_or_else(|| format!("lpips validate: the assembly has no {r} role"));
        let trunk = role("trunk")?;
        if !is_trunk_file(trunk) {
            return Err(format!("lpips validate: {} is not torchvision's AlexNet", trunk.display()));
        }
        let heads = role("heads")?;
        if !is_heads_file(heads) {
            return Err(format!("lpips validate: {} is not LPIPS v0.1's AlexNet heads", heads.display()));
        }
        Ok(())
    }

    fn missing_doc(&self, role: &str) -> String {
        format!("no file in the model store holds LPIPS's {role}; run {FETCH_TOOL} to fetch both")
    }
}

/// The trunk and heads files, from the model store
/// ([`brain_modelstore::default_root`]).
///
/// The store is scanned and resolved here with `brain_modelstore`'s own
/// primitives, as `crates/catalog` does, rather than through
/// `loader::resolver`: `brain-loader` reaches `brain-recon` (through
/// `brain-npu` and `brain-worldmirror2`), and `brain-recon` scores with this
/// crate, so depending on it would be a cycle.
pub fn resolve() -> Result<(PathBuf, PathBuf), String> {
    let root = brain_modelstore::default_root().ok_or("lpips: no model store (set BRAIN_MODELS_DIR, XDG_DATA_HOME or HOME)")?;
    let records = brain_modelstore::inventory::scan(&root);
    let specs: [&dyn ArchSpec; 1] = [&LpipsSpec];
    let assembly = match brain_modelstore::resolve::resolve(ARCH, &records, &specs, &BTreeMap::new()) {
        Resolution::Resolved(a) => a,
        Resolution::Ambiguous(a) => return Err(format!("{} (searched {})", describe_ambiguity(&a), root.display())),
        Resolution::Missing(m) => return Err(format!("{} (searched {})", describe_missing(&m), root.display())),
    };
    let role = |r: &str| assembly.roles.get(r).cloned().ok_or_else(|| format!("lpips: the resolved assembly has no {r} role"));
    Ok((role("trunk")?, role("heads")?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("brain-lpips-spec-{tag}-{}", std::process::id()))
    }

    /// A real safetensors file carrying exactly the named tensors (zeros),
    /// written through the workspace's own writer so the header is genuine.
    fn write_st(path: &Path, tensors: &[(String, Vec<usize>)]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let named: Vec<(String, Vec<u64>, Vec<f32>)> =
            tensors.iter().map(|(n, s)| (n.clone(), s.iter().map(|&d| d as u64).collect(), vec![0.0; s.iter().product()])).collect();
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &named, &serde_json::json!({}), None).unwrap();
    }

    fn heads(c_last: usize) -> Vec<(String, Vec<usize>)> {
        TRUNK.iter().enumerate().map(|(i, c)| (head_weight(i), vec![1, if i == 4 { c_last } else { c.cout as usize }, 1, 1])).collect()
    }

    /// The heads classify by their five shapes under any file name; a set
    /// with one head of the wrong width (another trunk's heads) does not, and
    /// neither is mistaken for the trunk.
    #[test]
    fn heads_are_recognised_by_content() {
        let dir = tmp("heads");
        let good = dir.join("anywhere").join("renamed.safetensors");
        write_st(&good, &heads(256));
        let vgg_like = dir.join("richzhang").join("PerceptualSimilarity").join("alex.safetensors");
        write_st(&vgg_like, &heads(512));
        assert!(is_heads_file(&good));
        assert!(!is_trunk_file(&good));
        assert!(!is_heads_file(&vgg_like), "a head of the wrong width is not AlexNet's");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The trunk needs all ten convolution tensors at torchvision's shapes;
    /// dropping one bias is enough to refuse it.
    #[test]
    fn the_trunk_needs_every_convolution() {
        let dir = tmp("trunk");
        let all: Vec<(String, Vec<usize>)> = TRUNK
            .iter()
            .enumerate()
            .flat_map(|(i, c)| [(trunk_weight(i), vec![c.cout as usize, c.cin as usize, c.k as usize, c.k as usize]), (trunk_bias(i), vec![c.cout as usize])])
            .collect();
        let trunk = dir.join("pytorch").join("vision").join("alexnet.safetensors");
        write_st(&trunk, &all);
        let partial = dir.join("partial.safetensors");
        write_st(&partial, &all[..all.len() - 1]);
        assert!(is_trunk_file(&trunk));
        assert!(!is_heads_file(&trunk));
        assert!(!is_trunk_file(&partial));
        std::fs::remove_dir_all(&dir).ok();
    }
}
