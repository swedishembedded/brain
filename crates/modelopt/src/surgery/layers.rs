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

/// Choose `count` of `layers` layers to remove, one at a time: each round
/// adds the layer whose removal, together with the layers already chosen,
/// costs least by `cost` (the cost of removing a set of layers, lower is
/// better); ties go to the lower index. Removal costs do not add up layer by
/// layer (two layers that each cost little alone can cost much together), so
/// every round measures again, `layers - round` candidates each. Answers the
/// layers in the order chosen, each with the cost of the set it completes,
/// and tells `on_round` each choice as it is made.
pub fn greedy_removal(layers: usize, count: usize, cost: &mut dyn FnMut(&[usize]) -> Result<f64, String>, on_round: &mut dyn FnMut(usize, f64)) -> Result<Vec<(usize, f64)>, String> {
    if count >= layers {
        return Err(format!("greedy layer removal: {count} of {layers} layers leaves no stack"));
    }
    let mut chosen: Vec<(usize, f64)> = Vec::with_capacity(count);
    for _ in 0..count {
        let mut best: Option<(usize, f64)> = None;
        for l in (0..layers).filter(|l| !chosen.iter().any(|(c, _)| c == l)) {
            let set: Vec<usize> = chosen.iter().map(|(c, _)| *c).chain(std::iter::once(l)).collect();
            let c = cost(&set)?;
            if !c.is_finite() {
                return Err(format!("greedy layer removal: removing {set:?} costs {c}"));
            }
            if best.is_none_or(|(_, b)| c < b) {
                best = Some((l, c));
            }
        }
        let (l, c) = best.expect("a stack longer than the count has a candidate");
        on_round(l, c);
        chosen.push((l, c));
    }
    Ok(chosen)
}

/// Run `f` on the built model `m` with layers `remove` of `stack` made the
/// identity, their output leaves zeroed (what [`remove_layers`] computes on a
/// pre-norm residual stack), and put the weights back after it.
pub fn with_layers_masked<M: model::Model, T>(m: &M, stack: &LayerStack, remove: &[usize], f: impl FnOnce(&M) -> T) -> T {
    let names: Vec<String> = remove.iter().flat_map(|l| stack.outputs.iter().map(move |leaf| format!("{}{l}.{leaf}", stack.prefix))).collect();
    let saved: Vec<Vec<f32>> = names.iter().map(|n| m.read_weight(n)).collect();
    for (n, v) in names.iter().zip(&saved) {
        m.write_weight(n, &vec![0.0; v.len()]);
    }
    let out = f(m);
    for (n, v) in names.iter().zip(&saved) {
        m.write_weight(n, v);
    }
    out
}
