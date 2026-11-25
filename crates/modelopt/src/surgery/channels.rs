// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Channel selection on declared dimensions.

use std::collections::BTreeMap;

use model::surgery::ChannelDim;

use super::{check_keep, keep_lines, line_count, zero_lines, Change, Checkpoint};

/// The line indices channel set `keep` owns at `block` lines per channel.
fn lines_of(keep: &[usize], block: usize) -> Vec<usize> {
    keep.iter().flat_map(|&c| c * block..(c + 1) * block).collect()
}

/// Every dimension's lines exist at the declared width, and dimensions that
/// share a config key keep one channel count: the new value of each key.
fn check(ck: &Checkpoint, dims: &[&ChannelDim], keep: &[Vec<usize>]) -> Result<BTreeMap<String, usize>, String> {
    if dims.len() != keep.len() {
        return Err(format!("{} dimensions and {} kept sets", dims.len(), keep.len()));
    }
    if dims.is_empty() {
        return Err("no dimension to select channels of".into());
    }
    let mut keys: BTreeMap<String, usize> = BTreeMap::new();
    for (d, k) in dims.iter().zip(keep) {
        check_keep(&d.name, k, d.width)?;
        for l in &d.lines {
            let t = ck.get(&l.tensor)?;
            if line_count(t, l.axis) != d.width * l.block {
                return Err(format!("{}: {} has {} lines along {:?}, the dimension declares {} x {}", d.name, l.tensor, line_count(t, l.axis), l.axis, d.width, l.block));
            }
        }
        for w in &d.keys {
            let value = w.per_channel * k.len();
            if let Some(&other) = keys.get(&w.key) {
                if other != value {
                    return Err(format!("{}: keeps {} channels, but another dimension sized by {} keeps {}", d.name, k.len(), w.key, other / w.per_channel.max(1)));
                }
            }
            keys.insert(w.key.clone(), value);
        }
    }
    Ok(keys)
}

/// Keep channels `keep[i]` (ascending) of dimension `dims[i]`, removing every
/// other channel's lines from every tensor that indexes the dimension, and
/// set each dimension's config keys. Refused, touching nothing: a tensor that
/// is not the declared width, kept sets that are not ascending or exceed the
/// width, and dimensions sharing a key that keep different counts.
pub fn select_channels(ck: &mut Checkpoint, dims: &[&ChannelDim], keep: &[Vec<usize>]) -> Result<Change, String> {
    let keys = check(ck, dims, keep)?;
    let (mut change, before) = Change::begin("select_channels", ck);
    // Every cut is computed from the checkpoint as it stands, then written, so a tensor two dimensions index is
    // refused rather than cut twice by accident.
    let mut cut = BTreeMap::new();
    for (d, k) in dims.iter().zip(keep) {
        for l in &d.lines {
            let t = keep_lines(ck.get(&l.tensor)?, l.axis, &lines_of(k, l.block)).map_err(|e| format!("{}: {}: {e}", d.name, l.tensor))?;
            if cut.insert(l.tensor.clone(), t).is_some() {
                return Err(format!("{}: {} is indexed by two of the selected dimensions", d.name, l.tensor));
            }
        }
        change.targets.push(d.name.clone());
        change.kept.insert(d.name.clone(), k.clone());
    }
    for (name, t) in cut {
        *ck.get_mut(&name)? = t;
    }
    for (key, value) in keys {
        change.set(ck, &key, value.into())?;
    }
    Ok(change.finish(ck, before))
}

/// Zero, at the original shapes, the lines of every dimension's `masks`
/// tensors that `keep` drops: the function [`select_channels`] computes with
/// the same sets. Answers the tensors it changed, with their original
/// values, so a caller can restore them.
pub fn mask_channels(ck: &mut Checkpoint, dims: &[&ChannelDim], keep: &[Vec<usize>]) -> Result<Vec<(String, super::Tensor)>, String> {
    check(ck, dims, keep)?;
    let mut saved = Vec::new();
    for (d, k) in dims.iter().zip(keep) {
        if d.masks.is_empty() {
            return Err(format!("{}: the dimension declares no tensor whose zeroing removes a channel", d.name));
        }
        for m in &d.masks {
            let l = d.lines.iter().find(|l| &l.tensor == m).ok_or_else(|| format!("{}: mask {m} is not one of its lines", d.name))?;
            let t = ck.get_mut(m)?;
            saved.push((m.clone(), t.clone()));
            zero_lines(t, l.axis, &lines_of(k, l.block));
        }
    }
    Ok(saved)
}
