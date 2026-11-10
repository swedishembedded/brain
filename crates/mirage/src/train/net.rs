// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The graph builder: a tape plus the checkpoint's parameters as leaves, created on first use.
//!
//! Every dense kernel is applied through [`Net::proj`], the one place a LoRA adapter can attach.

use std::collections::{BTreeMap, HashMap};

use super::graph::Graph;
use super::real::{r, Real};
use super::tape::{Grads, Id, Tape};
use crate::config::MirageConfig;
use crate::import::Checkpoint;

/// Parameters by checkpoint name (Flax layout), in the graph's scalar type.
pub type Params<R> = BTreeMap<String, Vec<R>>;

/// The checkpoint's tensors converted to the graph's scalar type.
pub fn params_from_checkpoint<R: Real>(ck: &Checkpoint) -> Params<R> {
    ck.tensors.iter().map(|(name, t)| (name.clone(), t.data.iter().map(|&v| r::<R>(f64::from(v))).collect())).collect()
}

/// A low-rank update for some dense kernels: `y = x W + scale * (x A) B`, keyed by kernel (see [`Net::proj`]).
#[derive(Debug, Clone)]
pub struct AdapterSet<R: Real> {
    /// `alpha / rank`.
    pub scale: R,
    /// Rank.
    pub rank: usize,
    /// `key -> (A [in, rank], B [rank, out])`.
    pub pairs: BTreeMap<String, (Vec<R>, Vec<R>)>,
}

/// A graph under construction over fixed parameters.
///
/// `T` evaluates the graph: the host [`Tape`] (default) or the device tape.
pub struct Net<'a, R: Real, T: Graph<R> = Tape<R>> {
    /// The tape.
    pub tape: T,
    /// The model configuration.
    pub cfg: &'a MirageConfig,
    params: Option<&'a Params<R>>,
    adapter: Option<&'a AdapterSet<R>>,
    ids: HashMap<String, Id>,
    stacked: HashMap<(String, usize), Id>,
    lora_ids: BTreeMap<String, (Id, Id)>,
}

impl<'a, R: Real> Net<'a, R> {
    /// A fresh host graph over `params`, optionally with an adapter on some dense kernels.
    pub fn new(cfg: &'a MirageConfig, params: &'a Params<R>, adapter: Option<&'a AdapterSet<R>>) -> Net<'a, R> {
        Net { tape: Tape::new(), cfg, params: Some(params), adapter, ids: HashMap::new(), stacked: HashMap::new(), lora_ids: BTreeMap::new() }
    }
}

impl<'a, R: Real, T: Graph<R>> Net<'a, R, T> {
    /// A graph over an evaluator that holds the parameters (and any adapter) itself.
    pub fn resident(cfg: &'a MirageConfig, tape: T) -> Net<'a, R, T> {
        Net { tape, cfg, params: None, adapter: None, ids: HashMap::new(), stacked: HashMap::new(), lora_ids: BTreeMap::new() }
    }

    /// The parameter `name` as a leaf (one leaf per name per graph, so gradients accumulate).
    pub fn p(&mut self, name: &str) -> Id {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = match self.tape.resident_param(name) {
            Some(id) => id,
            None => {
                let v = self.params.expect("a host graph carries its parameters").get(name).unwrap_or_else(|| panic!("the parameter set lacks `{name}`"));
                self.tape.leaf(v.clone())
            }
        };
        self.ids.insert(name.to_string(), id);
        id
    }

    /// Layer `layer` of the stacked parameter `name` (a leading layer axis).
    pub fn layer(&mut self, name: &str, layer: usize) -> Id {
        if let Some(&id) = self.stacked.get(&(name.to_string(), layer)) {
            return id;
        }
        let id = match self.tape.resident_layer(name, layer) {
            Some(id) => id,
            None => {
                let whole = self.p(name);
                let unit = self.tape.len(whole) / self.cfg.dec_layers;
                self.tape.slice(whole, layer * unit, unit)
            }
        };
        self.stacked.insert((name.to_string(), layer), id);
        id
    }

    /// A constant (no parameter behind it).
    pub fn constant(&mut self, v: Vec<R>) -> Id {
        self.tape.leaf(v)
    }

    /// `x W` for the kernel `name` (layer `layer` of a stacked kernel), plus the adapter's update when one is set
    /// for that kernel. The adapter key is the kernel name, with `.{layer}` for a stacked one.
    pub fn proj(&mut self, x: Id, name: &str, layer: Option<usize>, rows: usize, inn: usize, out: usize) -> Id {
        let w = match layer {
            Some(l) => self.layer(name, l),
            None => self.p(name),
        };
        let y = self.tape.linear(x, w, rows, inn, out);
        let key = match layer {
            Some(l) => format!("{name}.{l}"),
            None => name.to_string(),
        };
        let (ia, ib, rank, scale) = match self.adapter {
            Some(set) => {
                let Some((a, b)) = set.pairs.get(&key) else { return y };
                assert_eq!(a.len(), inn * set.rank, "adapter `{key}`: A is not {inn}x{}", set.rank);
                assert_eq!(b.len(), set.rank * out, "adapter `{key}`: B is not {}x{out}", set.rank);
                let (ia, ib) = match self.lora_ids.get(&key) {
                    Some(&ids) => ids,
                    None => {
                        let ids = (self.tape.leaf(a.clone()), self.tape.leaf(b.clone()));
                        self.lora_ids.insert(key.clone(), ids);
                        ids
                    }
                };
                (ia, ib, set.rank, set.scale.to_f64())
            }
            None => match self.tape.resident_lora(&key) {
                Some(found) => found,
                None => return y,
            },
        };
        let h = self.tape.linear(x, ia, rows, inn, rank);
        let delta = self.tape.linear(h, ib, rows, rank, out);
        let delta = self.tape.scale(delta, scale);
        self.tape.add(y, delta)
    }
}

impl<R: Real> Net<'_, R> {
    /// Gradients of every parameter the graph used.
    pub fn param_grads(&self, grads: &mut Grads<R>) -> BTreeMap<String, Vec<R>> {
        self.ids.iter().map(|(name, &id)| (name.clone(), grads.take(id, self.params.expect("host graph")[name].len()))).collect()
    }

    /// Gradients of the adapter factors the graph used: `key -> (dA, dB)`.
    pub fn adapter_grads(&self, grads: &mut Grads<R>) -> BTreeMap<String, (Vec<R>, Vec<R>)> {
        self.lora_ids
            .iter()
            .map(|(key, &(a, b))| {
                let (pa, pb) = &self.adapter.expect("adapter ids exist only with an adapter").pairs[key];
                (key.clone(), (grads.take(a, pa.len()), grads.take(b, pb.len())))
            })
            .collect()
    }
}
