// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA for Mirage's dense kernels: which kernels get a pair, the adapter file, and the exact fold into a
//! checkpoint (`W += (alpha / rank) A B`).
//!
//! Pairs follow the shared host-pair convention in the file (`A [rank, in]`, `B [out, rank]`, named
//! `<kernel>.lora_a` / `<kernel>.lora_b` through `model::adapter::KeyStyle`); in memory they are stored
//! `A [in, rank]`, `B [rank, out]` so they multiply Flax kernels directly. `B` starts at zero, so a fresh
//! adapter is an exact no-op.

use std::collections::BTreeMap;
use std::path::Path;

use data::rng::Lcg;
use model::adapter::{kind, pair_file, KeyStyle};

use super::net::AdapterSet;
use crate::config::MirageConfig;
use crate::import::Checkpoint;
use crate::spec::Manifest;

/// The family tag in an adapter file's header.
pub const FAMILY: &str = "mirage-lora";

/// Kernels that take an adapter: the decoder's self and cross attention, the encoder's attention, the slot
/// projections and the engram key/value projections (the dense projections of the brief's scope).
pub fn is_adaptable(name: &str) -> bool {
    name.ends_with("/kernel")
        && (name.contains("/self_attn/") && (name.starts_with("top_") || name.starts_with("stack/layers/block/"))
            || name.starts_with("stack/layers/block/cross_attn/")
            || name.starts_with("slot_proj/")
            || name.starts_with("slot_pool/")
            || (name.starts_with("engrams_") && (name.contains("/key_proj/") || name.contains("/value_proj/"))))
}

/// The adapter keys of a configuration with their `(in, out)` shapes: a stacked decoder kernel gets one key
/// `<kernel>.<layer>` per layer.
pub fn targets(cfg: &MirageConfig) -> Vec<(String, usize, usize)> {
    let mut out = Vec::new();
    for spec in Manifest::new(cfg).iter().filter(|s| is_adaptable(&s.name)) {
        match spec.shape.as_slice() {
            [inn, o] => out.push((spec.name.clone(), *inn, *o)),
            [layers, inn, o] => (0..*layers).for_each(|l| out.push((format!("{}.{l}", spec.name), *inn, *o))),
            other => panic!("adaptable kernel `{}` has shape {other:?}", spec.name),
        }
    }
    out
}

/// A fresh adapter of `rank` over every adaptable kernel. `A` is drawn from `seed` (scaled `1/sqrt(in)`), `B` is
/// zero.
pub fn fresh(cfg: &MirageConfig, rank: usize, alpha: f32, seed: u64) -> AdapterSet<f32> {
    let mut rng = Lcg::new(seed);
    let pairs = targets(cfg)
        .into_iter()
        .map(|(key, inn, out)| {
            let std = 1.0 / (inn as f32).sqrt();
            let a = (0..inn * rank).map(|_| rng.signed() * std).collect();
            (key, (a, vec![0.0; rank * out]))
        })
        .collect();
    AdapterSet { scale: alpha / rank as f32, rank, pairs }
}

/// Write the adapter to `path` (atomic).
pub fn save(path: &Path, set: &AdapterSet<f32>, cfg: &MirageConfig) -> Result<(), String> {
    let shapes: BTreeMap<String, (usize, usize)> = targets(cfg).into_iter().map(|(k, i, o)| (k, (i, o))).collect();
    let mut tensors = Vec::new();
    for (key, (a, b)) in &set.pairs {
        let (inn, out) = shapes[key];
        let r = set.rank;
        // In memory A [in, r], B [r, out]; in the file A [r, in], B [out, r].
        let mut fa = vec![0f32; r * inn];
        let mut fb = vec![0f32; out * r];
        for i in 0..inn {
            for k in 0..r {
                fa[k * inn + i] = a[i * r + k];
            }
        }
        for k in 0..r {
            for o in 0..out {
                fb[o * r + k] = b[k * out + o];
            }
        }
        tensors.push((KeyStyle::Brain.format(key, kind::LORA_A), vec![r, inn], fa));
        tensors.push((KeyStyle::Brain.format(key, kind::LORA_B), vec![out, r], fb));
    }
    let alpha = set.scale * set.rank as f32;
    pair_file::write(path.to_str().ok_or("adapter path is not UTF-8")?, serde_json::json!({"model": FAMILY, "rank": set.rank, "alpha": alpha}), &tensors)
}

/// Read an adapter file, checking every pair against the configuration's adaptable kernels.
pub fn load(path: &Path, cfg: &MirageConfig) -> Result<AdapterSet<f32>, String> {
    let file = pair_file::read(path.to_str().ok_or("adapter path is not UTF-8")?, FAMILY)?;
    let by_name = file.map();
    let r = file.rank;
    let mut pairs = BTreeMap::new();
    for (key, inn, out) in targets(cfg) {
        let (na, nb) = (KeyStyle::Brain.format(&key, kind::LORA_A), KeyStyle::Brain.format(&key, kind::LORA_B));
        let (Some((sa, fa)), Some((sb, fb))) = (by_name.get(&na), by_name.get(&nb)) else { continue };
        if sa != &[r, inn] || sb != &[out, r] {
            return Err(format!("{}: `{key}` has pair shapes {sa:?} and {sb:?}, expected [{r}, {inn}] and [{out}, {r}]", path.display()));
        }
        let mut a = vec![0f32; inn * r];
        let mut b = vec![0f32; r * out];
        for i in 0..inn {
            for k in 0..r {
                a[i * r + k] = fa[k * inn + i];
            }
        }
        for k in 0..r {
            for o in 0..out {
                b[k * out + o] = fb[o * r + k];
            }
        }
        pairs.insert(key, (a, b));
    }
    let unknown = by_name.keys().filter(|n| !n.ends_with(kind::LORA_A) && !n.ends_with(kind::LORA_B)).count();
    if pairs.is_empty() || unknown > 0 {
        return Err(format!("{}: the file holds no Mirage adapter pairs (or tensors that are not pairs)", path.display()));
    }
    Ok(AdapterSet { scale: file.alpha / r as f32, rank: r, pairs })
}

/// Fold the adapter into the checkpoint: `W += scale * A B` for every pair, in place.
pub fn fold(ck: &mut Checkpoint, set: &AdapterSet<f32>) -> Result<(), String> {
    for (key, (a, b)) in &set.pairs {
        let (name, layer) = match key.rsplit_once('.') {
            Some((n, l)) if l.chars().all(|c| c.is_ascii_digit()) && ck.tensors.get(n).is_some_and(|t| t.shape.len() == 3) => (n, Some(l.parse::<usize>().map_err(|e| e.to_string())?)),
            _ => (key.as_str(), None),
        };
        let t = ck.tensors.get_mut(name).ok_or_else(|| format!("adapter names `{name}`, which the checkpoint lacks"))?;
        let (inn, out) = match (&layer, t.shape.as_slice()) {
            (Some(_), [_, i, o]) | (None, [i, o]) => (*i, *o),
            _ => return Err(format!("adapter `{key}` does not fit tensor `{name}` of shape {:?}", t.shape)),
        };
        let r = set.rank;
        if a.len() != inn * r || b.len() != r * out {
            return Err(format!("adapter `{key}` has pair sizes {} and {}, expected {} and {}", a.len(), b.len(), inn * r, r * out));
        }
        let base = layer.map_or(0, |l| l * inn * out);
        for i in 0..inn {
            for k in 0..r {
                let aik = a[i * r + k] * set.scale;
                if aik != 0.0 {
                    t.data[base + i * out..base + (i + 1) * out].iter_mut().zip(&b[k * out..(k + 1) * out]).for_each(|(w, bk)| *w += aik * bk);
                }
            }
        }
    }
    Ok(())
}
