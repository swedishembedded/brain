// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CodeFormer checkpoint import with **two-way coverage validation**.
//!
//! Same discipline as `flux2::import` and `vqgan::import`, over the WHOLE
//! `codeformer.pth` this time: `crates/vqgan` consumes 329 of its 515 tensors
//! and reports the other 186 as `skipped`; this crate consumes all 515.
//!
//! Both directions are checked and every failure names the tensor:
//!
//! * every entry of [`CodeFormerConfig::tensor_manifest`] must be present
//!   exactly once with the right shape and the right element count — a missing
//!   tensor is an error, never a zero-fill;
//! * every source tensor must be consumed — an unused one is an error, so a
//!   checkpoint carrying a module this port does not implement cannot pass
//!   silently;
//! * the produced runtime map must be exactly
//!   [`CodeFormerConfig::runtime_manifest`], so a split that dropped or
//!   duplicated a slice is caught here rather than as a wrong number 40 layers
//!   later.
//!
//! ## The one transformation at the boundary
//!
//! `nn.MultiheadAttention` stores one fused `in_proj_weight[3E, E]`. The port
//! cannot dispatch it whole because CodeFormer adds the position embedding to
//! **q and k only**, so q/k and v read different inputs. It is split on the host
//! at import — the playbook's "split fused weights at the boundary" — into:
//!
//! ```text
//! in_proj_weight[3E, E] -> qk.weight[2E, E]   (rows 0..2E, already adjacent)
//!                          v.weight [E,  E]   (rows 2E..3E)
//! in_proj_bias  [3E]    -> qk.bias  [2E] , v.bias[E]
//! ```
//!
//! which is also the layout the attention kernels want: `attn_scores_bidir`
//! reads q and k from ONE buffer at `qkv_stride = 2E` with `q_off = 0`,
//! `k_off = E`, and `attn_apply_bidir` reads v from its own buffer at stride
//! `E`. No offset-view gymnastics in the hot path.

use std::collections::{HashMap, HashSet};

use vae::blocks::Tensors;

use crate::config::CodeFormerConfig;

/// A validated import: exactly the tensors
/// [`CodeFormerConfig::runtime_manifest`] names.
pub struct Import {
    pub tensors: Tensors,
    /// Number of tensors read from the source file (before the fused split).
    pub source_tensors: usize,
}

impl std::fmt::Debug for Import {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Import")
            .field("tensors", &self.tensors.len())
            .field("source_tensors", &self.source_tensors)
            .finish()
    }
}

/// Read `codeformer.pth` (`checkpoint::torchpt`) or an equivalent
/// `.safetensors` and validate it.
pub fn load(path: &str, cfg: &CodeFormerConfig) -> Result<Import, String> {
    let raw: Tensors = if path.ends_with(".safetensors") {
        checkpoint::safetensors::read(path)?
            .into_iter()
            .map(|t| (t.name, (t.shape, t.data)))
            .collect()
    } else {
        checkpoint::torchpt::read(path)?.into_iter().map(|t| (t.name, (t.shape, t.data))).collect()
    };
    if raw.is_empty() {
        return Err(format!("codeformer import: {path} contains no tensors"));
    }
    import(raw, cfg)
}

/// Validate an already-loaded name → `(shape, data)` map and produce the
/// runtime tensors.
pub fn import(raw: Tensors, cfg: &CodeFormerConfig) -> Result<Import, String> {
    // The `params_ema.` wrapper is stripped by the ONE implementation of that
    // rule, in `vqgan::import` — both crates read the same basicsr file.
    let raw = vqgan::import::strip_state_prefix(raw);
    let source_tensors = raw.len();
    let manifest = cfg.tensor_manifest();

    // ---- forward direction: every expected tensor, once, right shape --------
    let mut checked: Tensors = HashMap::with_capacity(manifest.len());
    for (name, shape) in &manifest {
        let (s, d) = raw
            .get(name)
            .ok_or_else(|| format!("codeformer import: missing tensor {name}"))?;
        if s != shape {
            return Err(format!("codeformer import: {name} shape {s:?}, expected {shape:?}"));
        }
        let n: usize = shape.iter().product();
        if d.len() != n {
            return Err(format!(
                "codeformer import: {name} has {} values, expected {n}",
                d.len()
            ));
        }
        if checked.insert(name.clone(), (s.clone(), d.clone())).is_some() {
            return Err(format!("codeformer import: {name} produced twice"));
        }
    }

    // ---- reverse direction: nothing in the source may go unused -------------
    let expected: HashSet<&str> = manifest.iter().map(|(n, _)| n.as_str()).collect();
    let mut unused: Vec<&str> =
        raw.keys().map(String::as_str).filter(|n| !expected.contains(n)).collect();
    if !unused.is_empty() {
        unused.sort_unstable();
        return Err(format!(
            "codeformer import: {} unused source tensor(s): {:?}",
            unused.len(),
            &unused[..unused.len().min(8)]
        ));
    }

    // ---- the one boundary transformation: split the fused in_proj ----------
    let e = cfg.dim_embd as usize;
    let mut tensors: Tensors = HashMap::with_capacity(checked.len() + 2 * cfg.n_layers as usize);
    for i in 0..cfg.n_layers as usize {
        let p = CodeFormerConfig::layer_prefix(i);
        let (_, w) = checked
            .remove(&format!("{p}.self_attn.in_proj_weight"))
            .ok_or_else(|| format!("codeformer import: missing {p}.self_attn.in_proj_weight"))?;
        let (_, b) = checked
            .remove(&format!("{p}.self_attn.in_proj_bias"))
            .ok_or_else(|| format!("codeformer import: missing {p}.self_attn.in_proj_bias"))?;
        let (wqk, wv) = w.split_at(2 * e * e);
        let (bqk, bv) = b.split_at(2 * e);
        tensors.insert(format!("{p}.self_attn.qk.weight"), (vec![2 * e, e], wqk.to_vec()));
        tensors.insert(format!("{p}.self_attn.v.weight"), (vec![e, e], wv.to_vec()));
        tensors.insert(format!("{p}.self_attn.qk.bias"), (vec![2 * e], bqk.to_vec()));
        tensors.insert(format!("{p}.self_attn.v.bias"), (vec![e], bv.to_vec()));
    }
    tensors.extend(checked);

    // ---- the produced map must BE the runtime manifest, exactly ------------
    let runtime = cfg.runtime_manifest();
    if tensors.len() != runtime.len() {
        return Err(format!(
            "codeformer import: produced {} runtime tensors, runtime manifest names {}",
            tensors.len(),
            runtime.len()
        ));
    }
    for (name, shape) in &runtime {
        match tensors.get(name) {
            None => return Err(format!("codeformer import: runtime tensor {name} not produced")),
            Some((s, d)) if s != shape || d.len() != shape.iter().product::<usize>() => {
                return Err(format!(
                    "codeformer import: runtime tensor {name} shape {s:?}/{} values, expected {shape:?}",
                    d.len()
                ))
            }
            Some(_) => {}
        }
    }

    Ok(Import { tensors, source_tensors })
}

/// The VQGAN half of a validated import, as `crates/vqgan`'s own graph wants it.
/// Same buffers — this only narrows the map, so there is one import, not two.
pub fn vqgan_tensors(im: &Import, cfg: &CodeFormerConfig) -> Tensors {
    cfg.vqgan
        .tensor_manifest()
        .into_iter()
        .map(|(n, _)| {
            let v = im.tensors.get(&n).unwrap_or_else(|| panic!("vqgan tensor {n} not imported"));
            (n, v.clone())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_map(cfg: &CodeFormerConfig) -> Tensors {
        cfg.tensor_manifest()
            .into_iter()
            .map(|(n, s)| {
                let len: usize = s.iter().product();
                // Distinct values per tensor so a mis-split is visible.
                (n, (s, (0..len).map(|i| i as f32).collect()))
            })
            .collect()
    }

    #[test]
    fn accepts_a_complete_state_dict_and_strips_the_wrapper() {
        let cfg = CodeFormerConfig::codeformer();
        let wrapped: Tensors =
            full_map(&cfg).into_iter().map(|(k, v)| (format!("params_ema.{k}"), v)).collect();
        let im = import(wrapped, &cfg).expect("complete state dict");
        assert_eq!(im.source_tensors, 515);
        assert_eq!(im.tensors.len(), cfg.runtime_manifest().len());
    }

    #[test]
    fn splits_the_fused_in_proj_into_qk_and_v() {
        let cfg = CodeFormerConfig::codeformer();
        let im = import(full_map(&cfg), &cfg).expect("import");
        let e = cfg.dim_embd as usize;
        let (qs, qk) = &im.tensors["ft_layers.3.self_attn.qk.weight"];
        let (vs, v) = &im.tensors["ft_layers.3.self_attn.v.weight"];
        assert_eq!(qs, &vec![2 * e, e]);
        assert_eq!(vs, &vec![e, e]);
        // `full_map` fills each tensor with 0,1,2,...: qk is the prefix, v the
        // suffix, so a swapped or off-by-one split is a wrong first element.
        assert_eq!(qk[0], 0.0);
        assert_eq!(qk[qk.len() - 1], (2 * e * e - 1) as f32);
        assert_eq!(v[0], (2 * e * e) as f32);
        let (_, qb) = &im.tensors["ft_layers.3.self_attn.qk.bias"];
        let (_, vb) = &im.tensors["ft_layers.3.self_attn.v.bias"];
        assert_eq!(qb.len(), 2 * e);
        assert_eq!(vb[0], (2 * e) as f32);
    }

    #[test]
    fn a_missing_tensor_is_an_error_naming_it() {
        let cfg = CodeFormerConfig::codeformer();
        let mut m = full_map(&cfg);
        m.remove("ft_layers.5.linear2.weight");
        let e = import(m, &cfg).unwrap_err();
        assert!(e.contains("ft_layers.5.linear2.weight"), "{e}");
    }

    #[test]
    fn a_wrong_shape_is_an_error_naming_it() {
        let cfg = CodeFormerConfig::codeformer();
        let mut m = full_map(&cfg);
        m.insert("position_emb".into(), (vec![512, 512], vec![0.0; 512 * 512]));
        let e = import(m, &cfg).unwrap_err();
        assert!(e.contains("position_emb") && e.contains("expected"), "{e}");
    }

    /// The VQGAN crate tolerates the CodeFormer-only prefixes because it does
    /// not implement them. This crate implements everything, so an unused
    /// tensor means the checkpoint carries a module the port does not model —
    /// an error, not a note.
    #[test]
    fn an_unused_source_tensor_is_an_error() {
        let cfg = CodeFormerConfig::codeformer();
        let mut m = full_map(&cfg);
        m.insert("ft_layers.9.norm1.weight".into(), (vec![512], vec![0.0; 512]));
        let e = import(m, &cfg).unwrap_err();
        assert!(e.contains("ft_layers.9.norm1.weight"), "{e}");
    }

    #[test]
    fn the_vqgan_half_is_the_same_buffers_not_a_second_import() {
        let cfg = CodeFormerConfig::codeformer();
        let im = import(full_map(&cfg), &cfg).expect("import");
        let t = vqgan_tensors(&im, &cfg);
        assert_eq!(t.len(), cfg.vqgan.tensor_manifest().len());
        assert_eq!(t["quantize.embedding.weight"].0, vec![1024, 256]);
    }
}
