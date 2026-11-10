// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The graph builder: a tape plus the checkpoint's parameters as leaves, created on first use.
//!
//! Every dense kernel is applied through [`Net::proj`], the one place a LoRA adapter can attach.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use model::saved::SavedPolicy;

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

/// Refuse a saved-activation policy the Mirage graph cannot honour: its recompute boundaries stay in the
/// evaluator's own precision, so the narrow-boundary rungs of [`SavedPolicy`] do not apply.
pub fn check_saved_policy(policy: SavedPolicy) -> Result<(), String> {
    match policy {
        SavedPolicy::Quantized(format) => Err(format!("mirage training keeps its recompute boundaries in full precision; {} boundaries are not supported", format.label())),
        SavedPolicy::Keep | SavedPolicy::Recompute => Ok(()),
    }
}

/// The code of a recompute block: records the block's part of the graph and returns its outputs.
type Build<'a, R, T> = Rc<dyn Fn(&mut Net<'a, R, T>) -> Vec<Id>>;

/// One sealed recompute block: how to rebuild it, and the outputs the forward kept.
struct Block<'a, R: Real, T: Graph<R>> {
    run: Build<'a, R, T>,
    outs: Vec<Id>,
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
    saved: SavedPolicy,
    blocks: Vec<Block<'a, R, T>>,
    /// A block is being rebuilt inside the backward: nested blocks run inline.
    replaying: bool,
}

impl<'a, R: Real> Net<'a, R> {
    /// A fresh host graph over `params`, optionally with an adapter on some dense kernels.
    pub fn new(cfg: &'a MirageConfig, params: &'a Params<R>, adapter: Option<&'a AdapterSet<R>>) -> Net<'a, R> {
        Net { tape: Tape::new(), cfg, params: Some(params), adapter, ids: HashMap::new(), stacked: HashMap::new(), lora_ids: BTreeMap::new(), saved: SavedPolicy::Keep, blocks: Vec::new(), replaying: false }
    }
}

impl<'a, R: Real, T: Graph<R>> Net<'a, R, T> {
    /// A graph over an evaluator that holds the parameters (and any adapter) itself.
    pub fn resident(cfg: &'a MirageConfig, tape: T) -> Net<'a, R, T> {
        Net { tape, cfg, params: None, adapter: None, ids: HashMap::new(), stacked: HashMap::new(), lora_ids: BTreeMap::new(), saved: SavedPolicy::Keep, blocks: Vec::new(), replaying: false }
    }

    /// What the backward keeps of the forward: [`SavedPolicy::Keep`] holds every intermediate; `Recompute` holds
    /// only the outputs of each [`Net::block`] and rebuilds the block inside [`Net::backward`]. A narrow boundary
    /// format is refused: the graph's blocks hold their boundaries in the evaluator's own precision.
    pub fn with_saved_policy(mut self, policy: SavedPolicy) -> Result<Self, String> {
        check_saved_policy(policy)?;
        self.saved = policy;
        Ok(self)
    }

    /// Build a recompute block: `build` records part of the graph and returns the values later work reads. Under
    /// `Keep` it simply runs. Under `Recompute` everything it recorded is dropped after the forward except those
    /// outputs; the backward runs `build` again from the same inputs and differentiates that copy, so `build` must
    /// be deterministic and may read only values created before the block (its inputs and the parameters), through
    /// ids it captured by value.
    pub fn block(&mut self, build: impl Fn(&mut Self) -> Vec<Id> + 'static) -> Vec<Id> {
        if !self.saved.recomputes() || self.replaying {
            return build(self);
        }
        let mark = self.tape.mark();
        let outs = build(self);
        // Parameters first touched inside the block must survive it: later blocks reuse their ids.
        let mut keep = outs.clone();
        keep.extend(self.ids.values().chain(self.lora_ids.values().flat_map(|(a, b)| [a, b])).filter(|id| id.0 >= mark.values()));
        self.tape.seal(mark, &keep, self.blocks.len());
        // A stacked layer's slice is a block-internal value on the host; it is rebuilt when next asked for.
        self.stacked.retain(|_, id| id.0 < mark.values());
        self.blocks.push(Block { run: Rc::new(build), outs: outs.clone() });
        outs
    }

    /// [`Net::block`] for a block with one output and a map the caller needs outside it.
    pub fn block_map(&mut self, build: impl Fn(&mut Self) -> (Id, crate::vision::ops::Map) + 'static) -> (Id, crate::vision::ops::Map) {
        let map = Rc::new(Cell::new(None));
        let sink = Rc::clone(&map);
        let out = self.block(move |net| {
            let (id, m) = build(net);
            sink.set(Some(m));
            vec![id]
        });
        (out[0], map.get().expect("a block ran at least once"))
    }

    /// Differentiate the scalar `loss`, rebuilding each sealed block as the backward reaches it. The gradients are
    /// left with the evaluator (host: [`Tape::take_grads`]; device: the parameter store).
    pub fn backward(&mut self, loss: Id) {
        self.tape.backward_begin(loss);
        while let Some(index) = self.tape.backward_to_block() {
            let run = Rc::clone(&self.blocks[index].run);
            let sealed = self.blocks[index].outs.clone();
            let mark = self.tape.mark();
            self.replaying = true;
            let rebuilt = run(self);
            self.replaying = false;
            assert_eq!(rebuilt.len(), sealed.len(), "a recompute block rebuilt a different number of outputs");
            let pairs: Vec<(Id, Id)> = rebuilt.into_iter().zip(sealed).collect();
            self.tape.replay_backward(mark, &pairs);
            self.stacked.retain(|_, id| id.0 < mark.values());
        }
        self.tape.backward_end();
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
