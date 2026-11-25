// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Layer removal on a declared stack.

use std::collections::{BTreeMap, BTreeSet};

use model::surgery::LayerStack;

use super::{Change, Checkpoint, Tensor};

/// The layer a tensor of `stack` belongs to and its leaf, or `None` for a tensor outside the stack.
fn layer_of<'a>(stack: &LayerStack, name: &'a str) -> Option<(usize, &'a str)> {
    let (i, leaf) = name.strip_prefix(stack.prefix.as_str())?.split_once('.')?;
    Some((i.parse().ok()?, leaf))
}

fn check(stack: &LayerStack, remove: &[usize]) -> Result<BTreeSet<usize>, String> {
    let set: BTreeSet<usize> = remove.iter().copied().collect();
    if set.len() != remove.len() {
        return Err(format!("{}: a layer is named twice", stack.name));
    }
    if let Some(&bad) = set.iter().find(|&&l| l >= stack.count) {
        return Err(format!("{}: layer {bad} is past its {} layers", stack.name, stack.count));
    }
    if set.len() == stack.count {
        return Err(format!("{}: removing every layer leaves no stack", stack.name));
    }
    Ok(set)
}

/// Remove layers `remove` of `stack`, renumbering the rest densely in their
/// order, and set the stack's count. Refused, touching nothing: a layer past
/// the stack, one named twice, every layer, and a stack tensor whose index is
/// past the declared count.
pub fn remove_layers(ck: &mut Checkpoint, stack: &LayerStack, remove: &[usize]) -> Result<Change, String> {
    let gone = check(stack, remove)?;
    if let Some(name) = ck.tensors.keys().find(|n| layer_of(stack, n).is_some_and(|(i, _)| i >= stack.count)) {
        return Err(format!("{}: {name} is past the declared {} layers", stack.name, stack.count));
    }
    let mut lists = Vec::new();
    for key in &stack.lists {
        let list = super::lookup(&ck.config, key)?.as_array().ok_or_else(|| format!("{}: config key {key} is not a list", stack.name))?;
        if list.len() != stack.count {
            return Err(format!("{}: config key {key} holds {} entries for {} layers", stack.name, list.len(), stack.count));
        }
        let kept: Vec<serde_json::Value> = list.iter().enumerate().filter(|(i, _)| !gone.contains(i)).map(|(_, v)| v.clone()).collect();
        lists.push((key, kept));
    }
    let (mut change, before) = Change::begin("remove_layers", ck);
    let renumber: BTreeMap<usize, usize> = (0..stack.count).filter(|l| !gone.contains(l)).enumerate().map(|(new, old)| (old, new)).collect();
    let old = std::mem::take(&mut ck.tensors);
    let mut moved: BTreeMap<String, Tensor> = BTreeMap::new();
    for (name, t) in old {
        match layer_of(stack, &name) {
            None => {
                moved.insert(name, t);
            }
            Some((i, leaf)) => {
                if let Some(&new) = renumber.get(&i) {
                    moved.insert(format!("{}{new}.{leaf}", stack.prefix), t);
                }
            }
        }
    }
    ck.tensors = moved;
    change.targets.push(stack.name.clone());
    change.kept.insert(stack.name.clone(), renumber.keys().copied().collect());
    change.set(ck, &stack.key, renumber.len().into())?;
    for (key, kept) in lists {
        change.set(ck, key, kept.into())?;
    }
    // A renamed layer keeps its shapes: the manifest lists the tensors that went (the last indices).
    Ok(change.finish(ck, before))
}

/// Make layers `remove` of `stack` the identity by zeroing their output
/// leaves: the function [`remove_layers`] computes. Answers the original
/// values of the tensors it changed.
pub fn mask_layers(ck: &mut Checkpoint, stack: &LayerStack, remove: &[usize]) -> Result<Vec<(String, Tensor)>, String> {
    check(stack, remove)?;
    let mut saved = Vec::new();
    for &l in remove {
        for leaf in &stack.outputs {
            let name = format!("{}{l}.{leaf}", stack.prefix);
            let t = ck.get_mut(&name)?;
            saved.push((name, t.clone()));
            t.data.fill(0.0);
        }
    }
    Ok(saved)
}
