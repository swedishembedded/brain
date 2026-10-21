// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in full fine-tuning of models larger
// than your accelerator memory then you can procure our services by sending an
// email to info@swedishembedded.com.

//! The streamed parameter update: one [`Rule`], any state format, any
//! backing, bounded workspace.
//!
//! For every tensor, tile by tile: decode the value (or its master), the
//! gradient and the state slots into f32, apply the rule, re-encode state, master
//! and value in their own formats. Staging is five f32 tiles plus the storage's
//! byte workspace, independent of tensor and model size.
//!
//! # The seam
//!
//! A trainer supplies [`ParamTiles`] (value and gradient tile access) and owns
//! the schedule; this module owns the update and the state. The optimiser
//! state lives in an [`OptimizerStateStorage`] whose layout is derived from
//! the rule and the [`TensorSpec`]s by [`state_layout`] - a state slot per
//! rule slot, plus one master slot when the parameter's precision asks for one.
//!
//! # Determinism
//!
//! Stochastic rounding is keyed on `(seed, step, tensor, slot, element)`, so a
//! step's result is independent of the tile size and of whether the run was
//! resumed from a checkpoint in between (`checkpoint` restores seed and step).

use std::io;

use paramstore::precision::{ParamPrecision, StateFormat, ValueDtype};

use crate::codec::{rounding_key, Rounding, SlotKind, StateCodec};
use crate::factored::{self, Factored, SqAccumulator};
use crate::rule::{Rule, StepScalars};
use crate::storage::{BlockStateStorage, Backing, OptimizerStateStorage, StateLayout};

/// One optimised tensor: identity, size and its own precision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub numel: usize,
    pub precision: ParamPrecision,
    /// Row length `k` of a `[numel / k, k]` matrix whose second moment a factored rule
    /// ([`Rule::Adafactor`]) keeps as row and column statistics; `None` for a vector (or a rule with
    /// elementwise state). Tiles of such a tensor are whole rows.
    pub row_len: Option<usize>,
}

/// Value and gradient access, in tiles. `start` is an element offset.
pub trait ParamTiles {
    fn read_value(&mut self, tensor: usize, start: usize, out: &mut [f32]) -> io::Result<()>;
    fn write_value(&mut self, tensor: usize, start: usize, vals: &[f32], rnd: Rounding) -> io::Result<()>;
    fn read_grad(&mut self, tensor: usize, start: usize, out: &mut [f32]) -> io::Result<()>;
    /// Deliver a gradient tile into the store [`Self::read_grad`] reads, for a
    /// trainer that produces gradients one layer at a time through this seam.
    /// Stores whose gradients the backward pass writes itself keep the default.
    fn write_grad(&mut self, tensor: usize, _start: usize, _grad: &[f32]) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::Unsupported, format!("tensor {tensor}: this store does not accept gradient writes")))
    }
}

/// Codecs of every slot of one tensor: the rule's slots (in `state` format)
/// followed by the master slot if the precision has one.
fn slot_codecs(rule: &Rule, p: &ParamPrecision) -> Vec<StateCodec> {
    let mut v: Vec<StateCodec> = rule.slots().iter().map(|&kind| StateCodec { format: p.state, kind }).collect();
    if let Some(f) = p.master.format() {
        v.push(StateCodec { format: f, kind: SlotKind::Signed });
    }
    v
}

/// The state layout for `rule` over `specs`.
pub fn state_layout(rule: &Rule, specs: &[TensorSpec]) -> StateLayout {
    let t: Vec<(usize, Vec<StateCodec>)> = specs.iter().map(|s| (s.numel, slot_codecs(rule, &s.precision))).collect();
    StateLayout::new(&t)
}

/// Layout of a value or gradient store: one unblocked slot per tensor.
pub fn tensor_layout(numels: &[usize], format: StateFormat) -> StateLayout {
    let t: Vec<(usize, Vec<StateCodec>)> =
        numels.iter().map(|&n| (n, vec![StateCodec { format, kind: SlotKind::Signed }])).collect();
    StateLayout::new(&t)
}

/// Value and gradient stores over two backings; the reference [`ParamTiles`].
pub struct HostParams<V: Backing, G: Backing> {
    pub values: BlockStateStorage<V>,
    pub grads: BlockStateStorage<G>,
}

impl<V: Backing, G: Backing> ParamTiles for HostParams<V, G> {
    fn read_value(&mut self, tensor: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        self.values.read_tile(tensor, 0, start, out)
    }
    fn write_value(&mut self, tensor: usize, start: usize, vals: &[f32], rnd: Rounding) -> io::Result<()> {
        self.values.write_tile(tensor, 0, start, vals, rnd)
    }
    fn read_grad(&mut self, tensor: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        self.grads.read_tile(tensor, 0, start, out)
    }
    fn write_grad(&mut self, tensor: usize, start: usize, grad: &[f32]) -> io::Result<()> {
        self.grads.write_tile(tensor, 0, start, grad, Rounding::NEAREST)
    }
}

/// Read-only [`ParamTiles`] whose values are the optimiser's MASTER copies
/// (the stored value where a tensor has none). Export from this to publish the
/// highest-precision weights in a chosen format rather than re-quantizing the
/// already-quantized training values.
pub struct MasterView<'a, S: OptimizerStateStorage> {
    pub opt: &'a mut StreamedOptimizer<S>,
    pub values: &'a mut dyn ParamTiles,
}

impl<S: OptimizerStateStorage> ParamTiles for MasterView<'_, S> {
    fn read_value(&mut self, tensor: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        match self.opt.master_slot(tensor) {
            // An export may read tiles wider than the state's own tile (a destination encoding with a larger write
            // quantum): the master is read in state-sized pieces.
            Some(ms) => {
                let piece = self.opt.tile_elems();
                for (i, chunk) in out.chunks_mut(piece).enumerate() {
                    self.opt.state.read_tile(tensor, ms, start + i * piece, chunk)?;
                }
                Ok(())
            }
            None => self.values.read_value(tensor, start, out),
        }
    }
    fn write_value(&mut self, tensor: usize, _start: usize, _vals: &[f32], _rnd: Rounding) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::Unsupported, format!("tensor {tensor}: a master view is read-only")))
    }
    fn read_grad(&mut self, tensor: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        self.values.read_grad(tensor, start, out)
    }
}

/// Facts about one step, for reporting and gates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepReport {
    pub step: u32,
    /// What every gradient was multiplied by (accumulation average x clip).
    pub grad_multiplier: f32,
}

fn lcm(a: usize, b: usize) -> usize {
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    a / gcd(a, b) * b
}

/// The tile size an optimiser over `specs` actually uses for a requested one:
/// rounded up to a multiple of every block size in use, so tiles start on
/// block boundaries. Size a storage's `max_tile_elems` with this.
pub fn effective_tile(specs: &[TensorSpec], requested: usize) -> usize {
    let mut tile = requested.max(1);
    for s in specs {
        if let StateFormat::Int8Block { block } = s.precision.state {
            if block > 0 {
                tile = lcm(tile, block as usize);
            }
        }
        if let Some(k) = s.row_len {
            tile = lcm(tile, k);
        }
    }
    tile
}

const VALUE_ROUNDING_SLOT: usize = 255;

/// Streamed optimiser over a state storage.
pub struct StreamedOptimizer<S: OptimizerStateStorage> {
    rule: Rule,
    specs: Vec<TensorSpec>,
    state: S,
    seed: u64,
    step: u32,
    tile: usize,
    // Five f32 tiles, allocated once: value/master, grad, two state slots.
    w: Vec<f32>,
    g: Vec<f32>,
    st: [Vec<f32>; 2],
    master: Vec<f32>,
    /// Factored second-moment statistics, one per tensor, when the rule is [`Rule::Adafactor`]; empty otherwise.
    factored: Vec<Factored>,
}

impl<S: OptimizerStateStorage> StreamedOptimizer<S> {
    /// `tile_elems` is rounded up to a multiple of every block size in use; the
    /// storage must have been created with at least that tile bound
    /// (see [`Self::tile_elems`]) and the layout from [`state_layout`].
    pub fn new(rule: Rule, specs: Vec<TensorSpec>, state: S, seed: u64, tile_elems: usize) -> io::Result<Self> {
        let tile = effective_tile(&specs, tile_elems);
        for s in &specs {
            if let StateFormat::Int8Block { block: 0 } = s.precision.state {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("{}: block size 0", s.name)));
            }
            if s.precision.master.format().is_some_and(|f| matches!(f, StateFormat::Int8Block { .. })) {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("{}: an 8-bit master is not supported", s.name)));
            }
        }
        let factored = if matches!(rule, Rule::Adafactor { .. }) { specs.iter().map(|s| Factored::new(s.numel, s.row_len)).collect() } else { Vec::new() };
        Ok(StreamedOptimizer {
            rule,
            specs,
            state,
            seed,
            step: 0,
            tile,
            w: vec![0.0; tile],
            g: vec![0.0; tile],
            st: [vec![0.0; tile], vec![0.0; tile]],
            master: vec![0.0; tile],
            factored,
        })
    }

    /// The effective tile size (>= the requested one, block-aligned).
    pub fn tile_elems(&self) -> usize {
        self.tile
    }

    /// Host staging this optimiser holds: its f32 tiles plus the storage's.
    pub fn staging_bytes(&self) -> usize {
        5 * self.tile * 4 + self.state.staging_bytes()
    }

    /// Physical bytes of the optimiser state (scales included).
    pub fn state_physical_bytes(&self) -> u64 {
        self.state.physical_bytes() + self.factored.iter().map(Factored::bytes).sum::<u64>()
    }

    /// Append the factored statistics (none unless the rule is [`Rule::Adafactor`]) to a checkpoint's state stream.
    pub fn export_factored(&self, w: &mut dyn io::Write) -> io::Result<()> {
        self.factored.iter().try_for_each(|f| factored::write_stats(f, w))
    }

    /// Read back what [`Self::export_factored`] wrote.
    pub fn import_factored(&mut self, r: &mut dyn io::Read) -> io::Result<()> {
        self.factored.iter_mut().try_for_each(|f| factored::read_stats(f, r))
    }

    pub fn step_count(&self) -> u32 {
        self.step
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn rule(&self) -> &Rule {
        &self.rule
    }

    pub fn specs(&self) -> &[TensorSpec] {
        &self.specs
    }

    pub fn state_mut(&mut self) -> &mut S {
        &mut self.state
    }

    /// Change the update rule between steps (e.g. a schedule that moves
    /// `wd`). The new rule must use the same state slots, since the state
    /// layout was fixed at construction.
    pub fn set_rule(&mut self, rule: Rule) {
        assert_eq!(rule.slots(), self.rule.slots(), "set_rule: {} cannot replace {}: different state slots", rule.name(), self.rule.name());
        self.rule = rule;
    }

    /// Restore the position in the run (checkpoint resume).
    pub fn set_position(&mut self, step: u32, seed: u64) {
        self.step = step;
        self.seed = seed;
    }

    fn master_slot(&self, ti: usize) -> Option<usize> {
        self.specs[ti].precision.master.format().map(|_| self.rule.slots().len())
    }

    /// Seed every master slot from the current values. Call once on a fresh
    /// state; a resumed state already carries its masters.
    pub fn init_master(&mut self, params: &mut dyn ParamTiles) -> io::Result<()> {
        for ti in 0..self.specs.len() {
            let Some(slot) = self.master_slot(ti) else { continue };
            let sr = self.specs[ti].precision.stochastic_rounding;
            let key = rounding_key(self.seed, 0, ti, slot);
            let numel = self.specs[ti].numel;
            for start in (0..numel).step_by(self.tile) {
                let n = self.tile.min(numel - start);
                params.read_value(ti, start, &mut self.w[..n])?;
                self.state.write_tile(ti, slot, start, &self.w[..n], Rounding { stochastic: sr, key })?;
            }
        }
        Ok(())
    }

    /// Overwrite tensor `ti`'s master copy from `start` with `vals` (a trainer restoring a weight it holds
    /// elsewhere, e.g. the best evaluation of a run). Returns `false`, writing nothing, for a tensor with no master.
    pub fn set_master(&mut self, ti: usize, start: usize, vals: &[f32]) -> io::Result<bool> {
        let Some(slot) = self.master_slot(ti) else { return Ok(false) };
        self.state.write_tile(ti, slot, start, vals, Rounding::NEAREST)?;
        Ok(true)
    }

    /// Write every master copy into the stored values, rounded to nearest: the values a trainer resumes from
    /// are then the quantization of the (authoritative) masters, whatever format the value file recorded.
    pub fn publish_masters(&mut self, params: &mut dyn ParamTiles) -> io::Result<()> {
        for ti in 0..self.specs.len() {
            let Some(slot) = self.master_slot(ti) else { continue };
            let numel = self.specs[ti].numel;
            for start in (0..numel).step_by(self.tile) {
                let n = self.tile.min(numel - start);
                self.state.read_tile(ti, slot, start, &mut self.w[..n])?;
                params.write_value(ti, start, &self.w[..n], Rounding::NEAREST)?;
            }
        }
        Ok(())
    }

    /// Sum of squares of every gradient, f64, tile by tile. With
    /// [`crate::grad_multiplier`] this is the global clip; a layer-streamed
    /// trainer that cannot hold every gradient computes it as it goes and passes
    /// the resulting multiplier to [`Self::update_tensor`].
    pub fn grad_sum_sq(&mut self, params: &mut dyn ParamTiles) -> io::Result<f64> {
        let mut acc = 0f64;
        for ti in 0..self.specs.len() {
            let numel = self.specs[ti].numel;
            for start in (0..numel).step_by(self.tile) {
                let n = self.tile.min(numel - start);
                params.read_grad(ti, start, &mut self.g[..n])?;
                acc += self.g[..n].iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
            }
        }
        Ok(acc)
    }

    /// One update step over every tensor. Gradients are multiplied by
    /// [`crate::grad_multiplier`] (`extra_scale`, and the global clip when
    /// `clip` is set). Not atomic: after an `Err` the state and values may be
    /// partially updated - recover from the last committed checkpoint.
    pub fn step(&mut self, params: &mut dyn ParamTiles, lr: f32, clip: Option<f32>, extra_scale: f32) -> io::Result<StepReport> {
        let sumsq = if clip.is_some() { self.grad_sum_sq(params)? } else { 0.0 };
        let mult = crate::grad_multiplier(sumsq, clip, extra_scale);
        let step = self.begin_step();
        for ti in 0..self.specs.len() {
            self.update_tensor(params, ti, lr, mult)?;
        }
        Ok(StepReport { step, grad_multiplier: mult })
    }

    /// Start the next optimisation step and return its 1-based index. The
    /// first half of the incremental form of [`Self::step`]: a trainer that
    /// finishes tensors one layer at a time calls this once, then
    /// [`Self::update_tensor`] per tensor in any order, each exactly once.
    pub fn begin_step(&mut self) -> u32 {
        self.step += 1;
        self.step
    }

    /// Update tensor `ti` for the step opened by [`Self::begin_step`], with its
    /// gradient multiplied by `grad_multiplier`. Reads the tensor's value (or
    /// master), gradient and state, writes state, master and value.
    pub fn update_tensor(&mut self, params: &mut dyn ParamTiles, ti: usize, lr: f32, grad_multiplier: f32) -> io::Result<()> {
        if let Rule::Adafactor { decay, eps, clip_threshold, wd } = self.rule {
            return self.update_factored(params, ti, lr, grad_multiplier, (decay, eps, clip_threshold, wd));
        }
        let scalars = StepScalars { t: self.step, lr, grad_scale: grad_multiplier };
        let nslots = self.rule.slots().len();
        let prec = self.specs[ti].precision;
        let numel = self.specs[ti].numel;
        let master_slot = self.master_slot(ti);
        let sr = prec.stochastic_rounding;
        for start in (0..numel).step_by(self.tile) {
            let n = self.tile.min(numel - start);
            match master_slot {
                Some(ms) => self.state.read_tile(ti, ms, start, &mut self.w[..n])?,
                None => params.read_value(ti, start, &mut self.w[..n])?,
            }
            params.read_grad(ti, start, &mut self.g[..n])?;
            for s in 0..nslots {
                self.state.read_tile(ti, s, start, &mut self.st[s][..n])?;
            }
            for i in 0..n {
                let mut st = [self.st[0][i], self.st[1][i]];
                self.rule.update(scalars, &mut self.w[i], self.g[i], &mut st);
                self.st[0][i] = st[0];
                self.st[1][i] = st[1];
            }
            for s in 0..nslots {
                let rnd = Rounding { stochastic: sr, key: rounding_key(self.seed, self.step, ti, s) };
                self.state.write_tile(ti, s, start, &self.st[s][..n], rnd)?;
            }
            if let Some(ms) = master_slot {
                let rnd = Rounding { stochastic: sr, key: rounding_key(self.seed, self.step, ti, ms) };
                self.master[..n].copy_from_slice(&self.w[..n]);
                self.state.write_tile(ti, ms, start, &self.master[..n], rnd)?;
            }
            let vr = Rounding {
                stochastic: sr && prec.value != ValueDtype::F32,
                key: rounding_key(self.seed, self.step, ti, VALUE_ROUNDING_SLOT),
            };
            params.write_value(ti, start, &self.w[..n], vr)?;
        }
        Ok(())
    }
}

impl<S: OptimizerStateStorage> StreamedOptimizer<S> {
    /// Tile length for tensor `ti`: the optimiser's tile, a whole number of rows for a factored matrix.
    fn row_tile(&self, ti: usize) -> usize {
        match self.specs[ti].row_len {
            Some(k) => (self.tile / k).max(1) * k,
            None => self.tile,
        }
    }

    /// [`Rule::Adafactor`] on tensor `ti`: three streamed passes over its gradient (second-moment statistics,
    /// update RMS for the clip, apply), the statistics held in host memory (`rows + cols` floats).
    fn update_factored(&mut self, params: &mut dyn ParamTiles, ti: usize, lr: f32, mult: f32, (decay, eps, clip_threshold, wd): (f32, f32, f32, f32)) -> io::Result<()> {
        let numel = self.specs[ti].numel;
        let tile = self.row_tile(ti);
        let prec = self.specs[ti].precision;
        let master_slot = self.master_slot(ti);
        let sr = prec.stochastic_rounding;
        let scaled = |g: &mut [f32]| g.iter_mut().for_each(|x| *x *= mult);

        let mut acc = SqAccumulator::new(&self.factored[ti], eps);
        for start in (0..numel).step_by(tile) {
            let n = tile.min(numel - start);
            params.read_grad(ti, start, &mut self.g[..n])?;
            scaled(&mut self.g[..n]);
            acc.add(start, &self.g[..n]);
        }
        acc.finish(&mut self.factored[ti], factored::beta2(decay, self.step));

        let mut sum_sq_u = 0f64;
        for start in (0..numel).step_by(tile) {
            let n = tile.min(numel - start);
            params.read_grad(ti, start, &mut self.g[..n])?;
            scaled(&mut self.g[..n]);
            factored::precondition(&self.factored[ti], start, &mut self.g[..n]);
            sum_sq_u += self.g[..n].iter().map(|&u| f64::from(u) * f64::from(u)).sum::<f64>();
        }
        let clip = factored::clip_scale(sum_sq_u, numel, clip_threshold);

        for start in (0..numel).step_by(tile) {
            let n = tile.min(numel - start);
            match master_slot {
                Some(ms) => self.state.read_tile(ti, ms, start, &mut self.w[..n])?,
                None => params.read_value(ti, start, &mut self.w[..n])?,
            }
            params.read_grad(ti, start, &mut self.g[..n])?;
            scaled(&mut self.g[..n]);
            factored::precondition(&self.factored[ti], start, &mut self.g[..n]);
            for (w, &u) in self.w[..n].iter_mut().zip(&self.g[..n]) {
                *w -= lr * (clip * u + wd * *w);
            }
            if let Some(ms) = master_slot {
                let rnd = Rounding { stochastic: sr, key: rounding_key(self.seed, self.step, ti, ms) };
                self.master[..n].copy_from_slice(&self.w[..n]);
                self.state.write_tile(ti, ms, start, &self.master[..n], rnd)?;
            }
            let vr = Rounding { stochastic: sr && prec.value != ValueDtype::F32, key: rounding_key(self.seed, self.step, ti, VALUE_ROUNDING_SLOT) };
            params.write_value(ti, start, &self.w[..n], vr)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::MemBacking;
    use paramstore::precision::{GradDtype, MasterPolicy};

    fn spec(name: &str, numel: usize, state: StateFormat, master: MasterPolicy, value: ValueDtype) -> TensorSpec {
        TensorSpec {
            name: name.into(),
            numel,
            precision: ParamPrecision { value, grad: GradDtype::F32, state, master, stochastic_rounding: false },
            row_len: None,
        }
    }

    fn host(specs: &[TensorSpec], init: &[Vec<f32>], tile: usize) -> HostParams<MemBacking, MemBacking> {
        let numels: Vec<usize> = specs.iter().map(|s| s.numel).collect();
        // one value format for the test store: f32 values keep the comparison exact
        let vl = tensor_layout(&numels, StateFormat::F32);
        let gl = tensor_layout(&numels, StateFormat::F32);
        let mut values = BlockStateStorage::new(vl.clone(), MemBacking::zeroed(vl.total_bytes()), tile).unwrap();
        let grads = BlockStateStorage::new(gl.clone(), MemBacking::zeroed(gl.total_bytes()), tile).unwrap();
        for (i, v) in init.iter().enumerate() {
            for (c, chunk) in v.chunks(tile).enumerate() {
                values.write_tile(i, 0, c * tile, chunk, Rounding::NEAREST).unwrap();
            }
        }
        HostParams { values, grads }
    }

    fn set_grad(p: &mut HostParams<MemBacking, MemBacking>, ti: usize, g: &[f32], tile: usize) {
        for (c, chunk) in g.chunks(tile).enumerate() {
            p.grads.write_tile(ti, 0, c * tile, chunk, Rounding::NEAREST).unwrap();
        }
    }

    fn values(p: &mut HostParams<MemBacking, MemBacking>, ti: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0f32; n];
        for (c, chunk) in out.chunks_mut(64).enumerate() {
            p.values.read_tile(ti, 0, c * 64, chunk).unwrap();
        }
        out
    }

    fn make(specs: Vec<TensorSpec>, tile: usize, rule: Rule) -> StreamedOptimizer<BlockStateStorage<MemBacking>> {
        let layout = state_layout(&rule, &specs);
        let max_tile = 4096;
        let st = BlockStateStorage::new(layout.clone(), MemBacking::zeroed(layout.total_bytes()), max_tile).unwrap();
        StreamedOptimizer::new(rule, specs, st, 11, tile).unwrap()
    }

    /// Spec: with f32 state the streamed step is bit-identical to the
    /// reference `Adam::update_slice`, whatever the tile size, with the clip
    /// and the accumulation scale applied.
    #[test]
    fn f32_streamed_step_equals_reference_adamw_for_any_tile() {
        let n = 1000;
        let w0: Vec<f32> = (0..n).map(|i| ((i * 7 % 101) as f32 / 101.0) - 0.5).collect();
        let adam = crate::Adam { beta1: 0.8, beta2: 0.95, eps: 1e-6 };
        let rule = Rule::AdamW { adam, wd: 0.01 };
        let specs = vec![spec("w", n, StateFormat::F32, MasterPolicy::None, ValueDtype::F32)];
        for tile in [64usize, 333, 4096] {
            let mut p = host(&specs, std::slice::from_ref(&w0), tile);
            let mut opt = make(specs.clone(), tile, rule);
            let (mut wr, mut mr, mut vr) = (w0.clone(), vec![0f32; n], vec![0f32; n]);
            for t in 1..=4u32 {
                let g: Vec<f32> = (0..n).map(|i| (((i as u32 + t) % 13) as f32 / 13.0 - 0.5) * 0.1).collect();
                set_grad(&mut p, 0, &g, 64);
                let rep = opt.step(&mut p, 1e-3, Some(0.5), 2.0).unwrap();
                let sumsq: f64 = g.iter().map(|&x| (x as f64) * (x as f64)).sum();
                let mult = crate::grad_multiplier(sumsq, Some(0.5), 2.0);
                assert_eq!(rep.grad_multiplier, mult);
                adam.update_slice(t, 1e-3, 0.01, mult, &mut wr, &mut mr, &mut vr, &g);
            }
            assert_eq!(values(&mut p, 0, n), wr, "tile {tile}");
        }
    }

    /// Spec: an fp32 master accumulates updates a bf16 value cannot hold. With
    /// updates far below the bf16 spacing, a master-less nearest-rounded bf16
    /// weight never moves; with a master the exported value follows.
    #[test]
    fn master_keeps_updates_a_low_precision_value_would_lose() {
        let n = 16;
        let rule = Rule::SgdMomentum { momentum: 0.0, wd: 0.0, nesterov: false };
        let run = |master: MasterPolicy| {
            let specs = vec![spec("w", n, StateFormat::F32, master, ValueDtype::Bf16)];
            let numels = [n];
            let vl = tensor_layout(&numels, StateFormat::Bf16);
            let gl = tensor_layout(&numels, StateFormat::F32);
            let mut values = BlockStateStorage::new(vl.clone(), MemBacking::zeroed(vl.total_bytes()), 16).unwrap();
            values.write_tile(0, 0, 0, &[1.0; 16], Rounding::NEAREST).unwrap();
            let grads = BlockStateStorage::new(gl.clone(), MemBacking::zeroed(gl.total_bytes()), 16).unwrap();
            let mut p = HostParams { values, grads };
            p.grads.write_tile(0, 0, 0, &[1.0; 16], Rounding::NEAREST).unwrap();
            let mut opt = make(specs, 16, rule);
            opt.init_master(&mut p).unwrap();
            for _ in 0..100 {
                opt.step(&mut p, 1e-4, None, 1.0).unwrap();
            }
            let mut out = [0f32; 16];
            p.values.read_tile(0, 0, 0, &mut out).unwrap();
            out[0]
        };
        assert_eq!(run(MasterPolicy::None), 1.0, "bf16 spacing at 1.0 is 7.8e-3; 1e-4 steps vanish");
        let with = run(MasterPolicy::F32);
        assert!((with - 0.99).abs() < 0.008, "master-backed value {with} should track 1 - 100*1e-4");
    }
}
