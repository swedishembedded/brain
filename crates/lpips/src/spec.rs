// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LPIPS's rules for the model-store resolver: which two files in the store
//! are a trunk (AlexNet, or VGG16) and the v0.1 linear heads trained for it.
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

use crate::config::{head_weight, Trunk};

/// The resolver's name for the AlexNet metric.
pub const ARCH: &str = "lpips";
/// The resolver's name for the VGG16 metric.
pub const ARCH_VGG: &str = "lpips-vgg";
/// The roles, in the order the resolver reports them.
pub const ROLES: &[&str] = &["trunk", "heads"];
/// What the two files assemble into.
pub const ID: &str = "local/lpips-alex-v0.1";
/// What the VGG16 files assemble into.
pub const ID_VGG: &str = "local/lpips-vgg-v0.1";
/// What puts both AlexNet files into the store.
pub const FETCH_TOOL: &str = "tools/goldens/lpips_dump_reference.py";
/// What the VGG16 files are, since nothing fetches them yet: torchvision's `vgg16-397923af.pth` (the trunk) and
/// LPIPS v0.1's `vgg.pth` heads re-serialized as `vgg.safetensors`, as `alex.safetensors` is.
pub const VGG_FILES: &str = "torchvision's vgg16-397923af.pth and LPIPS v0.1's vgg.pth heads (as vgg.safetensors)";

/// The declared shapes in `path` of the tensors LPIPS reads, header-only: a
/// `.safetensors` header or a `torch.save` pickle, never tensor bytes.
fn shapes(path: &Path, kind: ArtifactKind, trunk: Trunk) -> HashMap<String, Vec<usize>> {
    let p = path.to_string_lossy();
    match kind {
        ArtifactKind::Safetensors => {
            let Ok(m) = checkpoint::mmap::MmapSafetensors::open(p.as_ref()) else { return HashMap::new() };
            let mut out = HashMap::new();
            let names = trunk.convs().iter().flat_map(|c| [c.weight(), c.bias()]).chain((0..5).map(head_weight));
            for name in names {
                if let Some(s) = m.shape(&name) {
                    out.insert(name, s.to_vec());
                }
            }
            out
        }
        ArtifactKind::Torch => checkpoint::torchpt::read_shapes(p.as_ref()).map(|v| v.into_iter().collect()).unwrap_or_default(),
        _ => HashMap::new(),
    }
}

fn is_trunk(s: &HashMap<String, Vec<usize>>, trunk: Trunk) -> bool {
    trunk.convs().iter().all(|c| {
        s.get(&c.weight()).is_some_and(|w| *w == [c.cout as usize, c.cin as usize, c.k as usize, c.k as usize]) && s.get(&c.bias()).is_some_and(|b| *b == [c.cout as usize])
    })
}

fn is_heads(s: &HashMap<String, Vec<usize>>, trunk: Trunk) -> bool {
    trunk.tap_channels().iter().enumerate().all(|(i, c)| s.get(&head_weight(i)).is_some_and(|w| *w == [1, *c as usize, 1, 1]))
}

/// Whether `path` holds `trunk`'s torchvision convolutions, from its own header.
pub fn is_trunk_file(trunk: Trunk, path: &Path) -> bool {
    kind_of(path).is_some_and(|k| is_trunk(&shapes(path, k, trunk), trunk))
}

/// Whether `path` holds LPIPS v0.1's heads for `trunk`, from its own header.
pub fn is_heads_file(trunk: Trunk, path: &Path) -> bool {
    kind_of(path).is_some_and(|k| is_heads(&shapes(path, k, trunk), trunk))
}

fn kind_of(path: &Path) -> Option<ArtifactKind> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("safetensors") => Some(ArtifactKind::Safetensors),
        Some("pt" | "pth") => Some(ArtifactKind::Torch),
        _ => None,
    }
}

/// LPIPS's [`ArchSpec`] for one trunk; [`LpipsSpec`] and [`LpipsVggSpec`] are its two instances.
pub struct NetSpec(pub Trunk);

/// The AlexNet metric (`lpips`).
pub struct LpipsSpec;
/// The VGG16 metric (`lpips-vgg`).
pub struct LpipsVggSpec;

impl NetSpec {
    fn arch(&self) -> &'static str {
        match self.0 {
            Trunk::Alex => ARCH,
            Trunk::Vgg16 => ARCH_VGG,
        }
    }

    fn classify(&self, records: &[ArtifactRecord]) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            if !rec.usable() || !matches!(rec.kind, ArtifactKind::Safetensors | ArtifactKind::Torch) {
                continue;
            }
            let s = shapes(&rec.path, rec.kind, self.0);
            // `Derived`: a bare checkpoint declares no architecture, so the
            // role is computed from its shapes.
            if is_trunk(&s, self.0) {
                out.push((idx, "trunk".to_string(), Confidence::Derived));
            }
            if is_heads(&s, self.0) {
                out.push((idx, "heads".to_string(), Confidence::Derived));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>) -> Result<AssembleOutcome, String> {
        for role in ROLES {
            chosen.get(*role).ok_or_else(|| format!("lpips assemble: no {role} chosen"))?;
        }
        let (id, variant) = match self.0 {
            Trunk::Alex => (ID, "v0.1-alex"),
            Trunk::Vgg16 => (ID_VGG, "v0.1-vgg"),
        };
        Ok(AssembleOutcome::Assembled(AssembledVariant { id: id.to_string(), variant: Some(variant.to_string()) }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let role = |r: &str| assembly.roles.get(r).ok_or_else(|| format!("lpips validate: the assembly has no {r} role"));
        let trunk = role("trunk")?;
        if !is_trunk_file(self.0, trunk) {
            return Err(format!("lpips validate: {} is not torchvision's {} trunk", trunk.display(), self.0.name()));
        }
        let heads = role("heads")?;
        if !is_heads_file(self.0, heads) {
            return Err(format!("lpips validate: {} is not LPIPS v0.1's {} heads", heads.display(), self.0.name()));
        }
        Ok(())
    }

    fn missing_doc(&self, role: &str) -> String {
        match self.0 {
            Trunk::Alex => format!("no file in the model store holds LPIPS's {role}; run {FETCH_TOOL} to fetch both"),
            Trunk::Vgg16 => format!("no file in the model store holds LPIPS-VGG's {role}; it needs {VGG_FILES}, which no tool fetches yet"),
        }
    }
}

macro_rules! arch_spec {
    ($ty:ident, $net:expr) => {
        impl ArchSpec for $ty {
            fn arch(&self) -> &'static str {
                NetSpec($net).arch()
            }
            fn roles(&self) -> &'static [&'static str] {
                ROLES
            }
            fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
                NetSpec($net).classify(records)
            }
            fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
                NetSpec($net).assemble(chosen)
            }
            fn validate(&self, assembly: &Assembly) -> Result<(), String> {
                NetSpec($net).validate(assembly)
            }
            fn missing_doc(&self, role: &str) -> String {
                NetSpec($net).missing_doc(role)
            }
        }
    };
}
arch_spec!(LpipsSpec, Trunk::Alex);
arch_spec!(LpipsVggSpec, Trunk::Vgg16);

/// The trunk and heads files, from the model store
/// ([`brain_modelstore::default_root`]).
///
/// The store is scanned and resolved here with `brain_modelstore`'s own
/// primitives, as `crates/catalog` does, rather than through
/// `loader::resolver`: `brain-loader` reaches `brain-recon` (through
/// `brain-npu` and `brain-worldmirror2`), and `brain-recon` scores with this
/// crate, so depending on it would be a cycle.
pub fn resolve() -> Result<(PathBuf, PathBuf), String> {
    resolve_for(Trunk::Alex)
}

/// [`resolve`] for either trunk. A missing file is refused by name: what the role is, and what puts it in the store.
pub fn resolve_for(trunk: Trunk) -> Result<(PathBuf, PathBuf), String> {
    let root = brain_modelstore::default_root().ok_or("lpips: no model store (set BRAIN_MODELS_DIR, XDG_DATA_HOME or HOME)")?;
    let records = brain_modelstore::inventory::scan(&root);
    let (alex, vgg) = (LpipsSpec, LpipsVggSpec);
    let (spec, arch): (&dyn ArchSpec, &str) = match trunk {
        Trunk::Alex => (&alex, ARCH),
        Trunk::Vgg16 => (&vgg, ARCH_VGG),
    };
    let assembly = match brain_modelstore::resolve::resolve(arch, &records, &[spec], &BTreeMap::new()) {
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

    fn heads(trunk: Trunk) -> Vec<(String, Vec<usize>)> {
        trunk.tap_channels().iter().enumerate().map(|(i, c)| (head_weight(i), vec![1, *c as usize, 1, 1])).collect()
    }

    fn convolutions(trunk: Trunk) -> Vec<(String, Vec<usize>)> {
        trunk.convs().iter().flat_map(|c| [(c.weight(), vec![c.cout as usize, c.cin as usize, c.k as usize, c.k as usize]), (c.bias(), vec![c.cout as usize])]).collect()
    }

    /// The heads classify by their five shapes under any file name, for the trunk they were trained with and no other
    /// (AlexNet's and VGG16's differ from the second head on), and neither is mistaken for a trunk.
    #[test]
    fn heads_are_recognised_by_content() {
        let dir = tmp("heads");
        let alex = dir.join("anywhere").join("renamed.safetensors");
        write_st(&alex, &heads(Trunk::Alex));
        let vgg = dir.join("richzhang").join("PerceptualSimilarity").join("vgg.safetensors");
        write_st(&vgg, &heads(Trunk::Vgg16));
        assert!(is_heads_file(Trunk::Alex, &alex) && is_heads_file(Trunk::Vgg16, &vgg));
        assert!(!is_heads_file(Trunk::Alex, &vgg) && !is_heads_file(Trunk::Vgg16, &alex), "heads of the other trunk");
        assert!(!is_trunk_file(Trunk::Alex, &alex) && !is_trunk_file(Trunk::Vgg16, &vgg));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A trunk needs every convolution tensor at torchvision's shapes (dropping one bias refuses it) and is the
    /// trunk of its own network only.
    #[test]
    fn the_trunk_needs_every_convolution() {
        let dir = tmp("trunk");
        for trunk in [Trunk::Alex, Trunk::Vgg16] {
            let other = if trunk == Trunk::Alex { Trunk::Vgg16 } else { Trunk::Alex };
            let all = convolutions(trunk);
            let file = dir.join(trunk.name()).join("trunk.safetensors");
            write_st(&file, &all);
            let partial = dir.join(trunk.name()).join("partial.safetensors");
            write_st(&partial, &all[..all.len() - 1]);
            assert!(is_trunk_file(trunk, &file), "{trunk:?}");
            assert!(!is_trunk_file(other, &file) && !is_heads_file(trunk, &file));
            assert!(!is_trunk_file(trunk, &partial), "{trunk:?} without its last bias");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The VGG16 files being absent is refused by name, saying what they are, not by a generic "missing".
    #[test]
    fn a_missing_vgg_file_is_named() {
        let doc = NetSpec(Trunk::Vgg16).missing_doc("trunk");
        assert!(doc.contains("vgg16-397923af.pth") && doc.contains("vgg.safetensors"), "{doc}");
    }
}
