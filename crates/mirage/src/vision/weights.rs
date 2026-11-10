// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Named access to an imported checkpoint's tensors for the model builders.

use std::collections::BTreeMap;

use std::sync::Arc;

use model::cq::Cq;

use super::ops::Linear;
use crate::import::{Checkpoint, PackedMatrices, Tensor};

/// A borrowed view of the checkpoint's tensors. Names were validated against
/// the manifest at import, so a lookup of a manifest name cannot miss.
pub struct Weights<'a> {
    tensors: &'a BTreeMap<String, Tensor>,
    packed: &'a PackedMatrices,
    cq_activations: model::cq::CqActivations,
}

impl<'a> Weights<'a> {
    /// View `ck`'s tensors.
    pub fn new(ck: &'a Checkpoint) -> Weights<'a> {
        Weights { tensors: &ck.tensors, packed: &ck.packed, cq_activations: ck.cq_activations }
    }

    fn tensor(&self, name: &str) -> &'a Tensor {
        self.tensors.get(name).unwrap_or_else(|| panic!("checkpoint lacks `{name}`; import validates the manifest"))
    }

    /// A tensor's values.
    pub fn get(&self, name: &str) -> &'a [f32] {
        &self.tensor(name).data
    }

    /// An owned copy of a tensor's values.
    pub fn vec(&self, name: &str) -> Vec<f32> {
        self.get(name).to_vec()
    }

    /// Layer `layer` of a stacked decoder tensor.
    pub fn layer(&self, name: &str, layer: usize) -> &'a [f32] {
        let t = self.tensor(name);
        let unit: usize = t.shape[1..].iter().product();
        &t.data[layer * unit..(layer + 1) * unit]
    }

    /// The stored CQ form of tensor `name` (layer `layer` of a stacked one) when it is an `[out, in]` matrix.
    pub fn packed(&self, name: &str, layer: Option<usize>, out: usize, inn: usize) -> Option<Arc<Cq>> {
        self.packed.get(&(name.to_string(), layer)).filter(|c| c.n == out && c.k == inn).cloned()
    }

    /// `l` with its packed form (if it has one) multiplied the way this checkpoint asks for.
    pub fn with_activations(&self, l: Linear) -> Linear {
        Linear { cq_activations: self.cq_activations, ..l }
    }

    /// A Flax dense kernel `[in, out]` as a [`Linear`].
    pub fn linear(&self, name: &str) -> Linear {
        let t = self.tensor(name);
        assert_eq!(t.shape.len(), 2, "`{name}` is not a matrix");
        self.with_activations(Linear { cq: self.packed(name, None, t.shape[1], t.shape[0]), ..Linear::from_flax(&t.data, t.shape[0], t.shape[1]) })
    }

    /// Layer `layer` of a stacked Flax dense kernel (`[in, out]` per layer) as a [`Linear`].
    pub fn layer_linear(&self, name: &str, layer: usize, inn: usize, out: usize) -> Linear {
        self.with_activations(Linear { cq: self.packed(name, Some(layer), out, inn), ..Linear::from_flax(self.layer(name, layer), inn, out) })
    }
}
