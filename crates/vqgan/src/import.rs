// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Checkpoint import with **two-way coverage validation**.
//!
//! Same discipline as `flux2::import`: every tensor
//! [`VqganConfig::tensor_manifest`] expects is produced exactly once with the
//! right shape, and every source tensor is either consumed or matches a
//! **declared** non-VQGAN prefix. Anything else is an error naming the tensor —
//! never a silent zero-fill.
//!
//! Two source layouts, both handled here:
//!
//! * `codeformer.pth` / `vqgan_code1024.pth` — a torch zip whose single
//!   top-level key is `params_ema`. `checkpoint::torchpt` flattens that to
//!   `params_ema.encoder.blocks.0.weight`, so the prefix is stripped.
//!   `codeformer.pth` additionally carries the CodeFormer transformer
//!   ([`CODEFORMER_ONLY`]), which this crate does not implement; those are
//!   reported in [`Import::skipped`], not silently dropped.
//! * a `.safetensors` file with the same names.

use std::collections::{HashMap, HashSet};

use vae::blocks::Tensors;

use crate::config::VqganConfig;

/// Top-level keys a `basicsr` checkpoint may wrap its state dict in.
const STATE_KEYS: [&str; 3] = ["params_ema", "params", "state_dict"];

/// Prefixes present in `codeformer.pth` that belong to the **CodeFormer
/// transformer**, not the VQGAN core: the 9-layer `TransformerSALayer` stack,
/// its positional/feature embeddings, the code-index prediction head, and the
/// controllable-feature-transformation fuse convs. A follow-up workflow
/// implements them; until then they are reported, never quietly ignored.
pub const CODEFORMER_ONLY: [&str; 5] =
    ["position_emb", "feat_emb", "ft_layers", "idx_pred_layer", "fuse_convs_dict"];

/// A validated import. `Debug` prints counts, never the tensor data.
pub struct Import {
    /// Exactly the tensors in [`VqganConfig::tensor_manifest`].
    pub tensors: Tensors,
    /// Source tensors deliberately not consumed, sorted. Every one matched a
    /// [`CODEFORMER_ONLY`] prefix; anything else would have been an error.
    pub skipped: Vec<String>,
}

impl std::fmt::Debug for Import {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Import")
            .field("tensors", &self.tensors.len())
            .field("skipped", &self.skipped)
            .finish()
    }
}

/// Read a checkpoint from disk (`.pth`/`.pt` via `checkpoint::torchpt`,
/// `.safetensors` via `checkpoint::safetensors`) and validate it.
pub fn load(path: &str, cfg: &VqganConfig) -> Result<Import, String> {
    let raw: Tensors = if path.ends_with(".safetensors") {
        checkpoint::safetensors::read(path)?
            .into_iter()
            .map(|t| (t.name, (t.shape, t.data)))
            .collect()
    } else {
        checkpoint::torchpt::read(path)?
            .into_iter()
            .map(|t| (t.name, (t.shape, t.data)))
            .collect()
    };
    if raw.is_empty() {
        return Err(format!("vqgan import: {path} contains no tensors"));
    }
    import(raw, cfg)
}

/// Validate an already-loaded name → `(shape, data)` map.
pub fn import(raw: Tensors, cfg: &VqganConfig) -> Result<Import, String> {
    let raw = strip_state_prefix(raw);
    let manifest = cfg.tensor_manifest();

    let mut tensors: Tensors = HashMap::with_capacity(manifest.len());
    for (name, shape) in &manifest {
        let (s, d) = raw
            .get(name)
            .ok_or_else(|| format!("vqgan import: missing tensor {name}"))?;
        if s != shape {
            return Err(format!("vqgan import: {name} shape {s:?}, expected {shape:?}"));
        }
        let n: usize = shape.iter().product();
        if d.len() != n {
            return Err(format!(
                "vqgan import: {name} has {} values, expected {n}",
                d.len()
            ));
        }
        if tensors.insert(name.clone(), (s.clone(), d.clone())).is_some() {
            return Err(format!("vqgan import: {name} produced twice"));
        }
    }

    // Reverse direction: nothing in the source may be silently unused.
    let expected: HashSet<&str> = manifest.iter().map(|(n, _)| n.as_str()).collect();
    let mut skipped = Vec::new();
    let mut unexpected = Vec::new();
    for name in raw.keys() {
        if expected.contains(name.as_str()) {
            continue;
        }
        let head = name.split('.').next().unwrap_or(name);
        if CODEFORMER_ONLY.contains(&head) {
            skipped.push(name.clone());
        } else {
            unexpected.push(name.clone());
        }
    }
    if !unexpected.is_empty() {
        unexpected.sort();
        return Err(format!(
            "vqgan import: {} unused source tensor(s) matching no declared prefix: {:?}",
            unexpected.len(),
            &unexpected[..unexpected.len().min(8)]
        ));
    }
    skipped.sort();
    Ok(Import { tensors, skipped })
}

/// Drop the `params_ema.` / `params.` / `state_dict.` wrapper if every tensor
/// carries the same one. A checkpoint with a mix is left alone, so the
/// coverage check reports the real names rather than a half-stripped set.
///
/// Public because every `basicsr` checkpoint in this workspace wraps its state
/// dict the same way — `crates/restore` reads the *same file* for the
/// CodeFormer half and must strip it identically.
pub fn strip_state_prefix(raw: Tensors) -> Tensors {
    for key in STATE_KEYS {
        let dot = format!("{key}.");
        if !raw.is_empty() && raw.keys().all(|k| k.starts_with(&dot)) {
            return raw
                .into_iter()
                .map(|(k, v)| (k[dot.len()..].to_string(), v))
                .collect();
        }
    }
    raw
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_map(cfg: &VqganConfig) -> Tensors {
        cfg.tensor_manifest()
            .into_iter()
            .map(|(n, s)| {
                let len: usize = s.iter().product();
                (n, (s, vec![0.0f32; len]))
            })
            .collect()
    }

    #[test]
    fn accepts_a_complete_state_dict_and_strips_the_wrapper() {
        let cfg = VqganConfig::codeformer();
        let wrapped: Tensors =
            full_map(&cfg).into_iter().map(|(k, v)| (format!("params_ema.{k}"), v)).collect();
        let im = import(wrapped, &cfg).expect("complete state dict");
        assert_eq!(im.tensors.len(), cfg.tensor_manifest().len());
        assert!(im.skipped.is_empty());
    }

    #[test]
    fn reports_the_codeformer_transformer_as_skipped_not_unused() {
        let cfg = VqganConfig::codeformer();
        let mut m = full_map(&cfg);
        m.insert("position_emb".into(), (vec![256, 512], vec![0.0; 256 * 512]));
        m.insert("ft_layers.0.attn.in_proj_bias".into(), (vec![1536], vec![0.0; 1536]));
        let im = import(m, &cfg).expect("vqgan subset of a codeformer checkpoint");
        assert_eq!(im.skipped, vec!["ft_layers.0.attn.in_proj_bias", "position_emb"]);
    }

    #[test]
    fn a_missing_tensor_is_an_error_naming_it() {
        let cfg = VqganConfig::codeformer();
        let mut m = full_map(&cfg);
        m.remove("encoder.blocks.17.proj_out.weight");
        let e = import(m, &cfg).unwrap_err();
        assert!(e.contains("encoder.blocks.17.proj_out.weight"), "{e}");
    }

    #[test]
    fn a_wrong_shape_is_an_error_naming_it() {
        let cfg = VqganConfig::codeformer();
        let mut m = full_map(&cfg);
        m.insert("quantize.embedding.weight".into(), (vec![512, 256], vec![0.0; 512 * 256]));
        let e = import(m, &cfg).unwrap_err();
        assert!(e.contains("quantize.embedding.weight") && e.contains("expected"), "{e}");
    }

    #[test]
    fn an_undeclared_extra_tensor_is_an_error() {
        let cfg = VqganConfig::codeformer();
        let mut m = full_map(&cfg);
        m.insert("encoder.blocks.99.weight".into(), (vec![4], vec![0.0; 4]));
        let e = import(m, &cfg).unwrap_err();
        assert!(e.contains("encoder.blocks.99.weight"), "{e}");
    }
}
