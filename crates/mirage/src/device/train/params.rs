// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Moving checkpoint-layout parameters into a [`ParamStore`] for the device tape and back.
//!
//! The checkpoint stacks the decoder's per-layer tensors on a leading layer axis. The device registers each layer as
//! its own entry `"{name}#{layer}"` (so a layer is a plain buffer); these helpers split on upload and stack on
//! readback, so callers (the trainer, checkpointing, tests) keep working in checkpoint layout.

use std::collections::{BTreeMap, HashMap};

use paramstore::{ParamStore, Role};

use crate::device::Device;
use crate::train::net::Params;

/// Whether the checkpoint tensor `name` carries a leading decoder-layer axis.
pub fn is_stacked(name: &str) -> bool {
    // An adapter pair is per projection and layer already (its key ends in `.{layer}`), never stacked.
    if name.ends_with(model::adapter::kind::LORA_A) || name.ends_with(model::adapter::kind::LORA_B) {
        return false;
    }
    name.starts_with("stack/layers/block/") || name.starts_with("stack/mhc_")
}

/// A store holding every tensor of `params` with the role `role_of(name)` (a frozen tensor gets no gradient or moment
/// buffer), the stacked ones split per layer over `layers`.
pub fn upload_params(dev: &Device, layers: usize, params: &Params<f32>, role_of: impl Fn(&str) -> Role) -> ParamStore {
    assert!(layers > 0, "a decoder has layers");
    let mut entries: Vec<(String, usize, Role)> = Vec::new();
    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    for (name, values) in params {
        if is_stacked(name) {
            assert!(values.len() % layers == 0, "`{name}` does not split over {layers} layers");
            let unit = values.len() / layers;
            for (l, part) in values.chunks_exact(unit).enumerate() {
                let key = format!("{name}#{l}");
                entries.push((key.clone(), unit, role_of(name)));
                init.insert(key, part.to_vec());
            }
        } else {
            entries.push((name.clone(), values.len(), role_of(name)));
            init.insert(name.clone(), values.clone());
        }
    }
    ParamStore::new_with_roles(dev.gpu(), entries, &init)
}

/// Stack per-layer entries back: `read(entry)` for every store entry, grouped by checkpoint name in layer order.
fn stacked(store: &ParamStore, read: impl Fn(&str) -> Vec<f32>) -> BTreeMap<String, Vec<f32>> {
    let mut layered: BTreeMap<String, BTreeMap<usize, Vec<f32>>> = BTreeMap::new();
    let mut out = BTreeMap::new();
    for (key, _) in &store.params {
        match key.rsplit_once('#') {
            Some((name, layer)) if layer.parse::<usize>().is_ok() => {
                layered.entry(name.to_string()).or_default().insert(layer.parse().expect("checked"), read(key));
            }
            _ => {
                out.insert(key.clone(), read(key));
            }
        }
    }
    for (name, layers) in layered {
        out.insert(name, layers.into_values().flatten().collect());
    }
    out
}

/// Every parameter gradient in checkpoint layout (a parameter with no gradient buffer reads as zeros).
pub fn stacked_grads(dev: &Device, store: &ParamStore) -> BTreeMap<String, Vec<f32>> {
    stacked(store, |key| {
        if store.grad.contains_key(key) {
            store.read_grad(dev.gpu(), key)
        } else {
            vec![0.0; store.numel(key)]
        }
    })
}

/// Every parameter value in checkpoint layout.
pub fn stacked_weights(dev: &Device, store: &ParamStore) -> BTreeMap<String, Vec<f32>> {
    stacked(store, |key| store.read_weight(dev.gpu(), key))
}
