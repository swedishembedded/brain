// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements full-parameter fine-tuning of models larger than one accelerator, their parameters
// and optimizer state sharded over the cards and nodes of a cluster, for its clients. If your team needs expertise in
// ZeRO-3-style parameter sharding, you can procure our services by sending an email to info@swedishembedded.com.

//! Parameter sharding (ZeRO-3) for a layer-streamed resident full fine-tune ([`crate::resident_ft`],
//! [`crate::resident_ft::ResidentFtConfig::shard`]).
//!
//! The layer window is the seam: a layer-streamed build already asks for a layer's weights where the forward and the
//! backward reach it ([`crate::resident_ft::ResidentFullFt::load_layer`]) and finishes a layer once its gradient is
//! complete ([`crate::resident_ft::ResidentFullFt::finish_layer`]). Sharded, the window's slots and gradient lanes stay
//! as they are, but what backs them changes: each layer (and each table) is one unit of a [`ZeroOptimizer`], every
//! rank holding its slice of the unit's f32 master, AdamW moments and bf16 value on the card. Loading a layer
//! all-gathers its unit into the slots; finishing it reduce-scatters the lanes to their owners, each owner measures
//! its slice and, in an updating pass, updates it on the card. Nothing here names a model: the model drives the same
//! window protocol whether its parameters are sharded or not.
//!
//! Every rank must load and finish the same layers in the same order (the forward and backward of every rank visit
//! the same layers, whatever its batch), since each load and finish is a collective.

use std::collections::BTreeMap;
use std::io;

use checkpoint::TensorSource;
use gpu_core::{CollectiveDtype, DeviceBuffer, Gpu};
use optim::rule::Rule;
use paramstore::precision::{GradDtype, MasterPolicy, StateFormat};

use crate::collective::Payload;
use crate::ops::{DeviceIo, Weight};
use crate::resident_ft::{ResidentFtConfig, ResidentSpec, ResidentValue};
use crate::zero::{DeviceParam, DeviceParams, ShardGroup, UnitUpdate, ZeroOptimizer};

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

/// Refuse a resident configuration the parameter-sharded build does not run, naming what differs.
pub(crate) fn check_config(cfg: &ResidentFtConfig) -> Result<(), String> {
    let refuse = |what: String| Err(format!("parameter sharding: {what}"));
    if cfg.window.is_none() {
        return refuse("it gathers each layer where it is used, which needs a layer window (window=N)".into());
    }
    if cfg.value != ResidentValue::Bf16 || cfg.grad != GradDtype::Bf16 {
        return refuse(format!("the sharded units hold bf16 values and bf16 gradients, not {:?} values with {:?} gradients", cfg.value, cfg.grad));
    }
    if cfg.master != MasterPolicy::F32 || cfg.state != StateFormat::F32 || cfg.stochastic_state {
        return refuse(format!("the sharded state is an f32 master with f32 AdamW moments, not a {:?} master with {:?} state", cfg.master, cfg.state));
    }
    if !matches!(cfg.rule, Rule::AdamW { .. }) {
        return refuse(format!("the sharded update is AdamW, not {:?}", cfg.rule));
    }
    Ok(())
}

/// The value and gradient buffers of a resident tensor: its window slot's bf16 words and its lane.
fn slot_param(spec: &ResidentSpec, weight: &Weight, grad: &DeviceBuffer) -> Result<DeviceParam, String> {
    let Weight::BF16 { w, .. } = weight else { return Err(format!("{}: a parameter-sharded slot holds bf16 words, not a {:?} weight", spec.name, weight.dtype())) };
    Ok(DeviceParam {
        name: spec.name.clone(),
        value: w.clone(),
        value_dtype: CollectiveDtype::Bf16,
        grad: grad.clone(),
        grad_dtype: CollectiveDtype::Bf16,
        numel: u64::from(spec.n) * u64::from(spec.k),
        lr_mult: 1.0,
    })
}

/// A parameter-sharded build's slices and group (module documentation).
pub(crate) struct ShardedLayers {
    pub(crate) group: ShardGroup,
    zero: ZeroOptimizer,
    /// The optimizer step the open updating pass applies (the last applied otherwise).
    pub(crate) step: u32,
    /// This rank's share of the step's items: every gradient it reduces is weighted by it.
    pub(crate) scale: f32,
    /// Units reduced in the open pass.
    reduced: Vec<bool>,
}

impl ShardedLayers {
    /// Shard `specs` (window slots `weights`, lanes `grads`) over `group`, each rank reading its slices of the initial
    /// values from `src`. Layers are units; every table is a unit of its own.
    pub(crate) fn new(group: ShardGroup, specs: &[ResidentSpec], weights: &[Weight], grads: &[DeviceBuffer], src: &dyn TensorSource, gpu: &Gpu) -> Result<ShardedLayers, String> {
        let tensors = specs.iter().zip(weights).zip(grads).map(|((s, w), g)| slot_param(s, w, g)).collect::<Result<Vec<_>, _>>()?;
        // A unit per layer; a table (no layer) is its own unit, keyed apart from every layer.
        let groups: Vec<u32> = specs.iter().enumerate().map(|(i, s)| s.layer().unwrap_or((1 << 24) + i as u32)).collect();
        let mut read = |t: usize, offset: u64, out: &mut [f32]| -> Result<(), String> {
            if checkpoint::read_range_into(src, &specs[t].name, offset as usize, out) {
                Ok(())
            } else {
                Err(format!("range {offset}..{} is not in the source", offset as usize + out.len()))
            }
        };
        let clock = std::time::Instant::now();
        let zero = ZeroOptimizer::new_parameter_shards(DeviceParams { gpu: gpu.share(), tensors }, &groups, &*group.collective, group.rank, &mut read)?;
        gpu_core::profile::stage_time("parameter sharding: read this rank's slices and build the state on the card", clock);
        let units = zero.unit_count();
        Ok(ShardedLayers { group, zero, step: 0, scale: 1.0, reduced: vec![false; units] })
    }

    pub(crate) fn report(&self) -> String {
        format!(
            "parameter-sharded over {} ranks (rank {}): {} units, {} bytes of slices and staging on the card",
            self.group.collective.world_size(),
            self.group.rank,
            self.zero.unit_count(),
            self.zero.device_bytes()
        )
    }

    /// Open a backward pass: the next step for an updating one.
    pub(crate) fn begin_pass(&mut self, updating: bool) -> u32 {
        self.reduced.iter_mut().for_each(|r| *r = false);
        if updating {
            self.step += 1;
            self.zero.applied(self.step);
            self.step
        } else {
            0
        }
    }

    /// Gather the units of `tensors` whose slots do not hold their current values into those slots.
    pub(crate) fn load(&mut self, ops: &dyn DeviceIo, tensors: &[usize], slot_holds: &mut [Option<(usize, u64)>], slot_of: &[usize], versions: &[u64], _by_layer: &BTreeMap<u32, Vec<usize>>) -> io::Result<()> {
        let mut units: Vec<usize> = tensors.iter().filter(|&&ti| slot_holds[slot_of[ti]] != Some((ti, versions[ti]))).map(|&ti| self.zero.unit_of(ti)).collect();
        units.dedup();
        if units.is_empty() {
            return Ok(());
        }
        // The work the model recorded against the slots (the layer that used them last) reaches the device first;
        // the gather's copies are ordered after it.
        ops.gpu().flush();
        let coll = &*self.group.collective;
        for u in units {
            self.zero.gather_unit(coll, u).map_err(io::Error::other)?;
            for ti in (0..slot_of.len()).filter(|&ti| self.zero.unit_of(ti) == u) {
                slot_holds[slot_of[ti]] = Some((ti, versions[ti]));
            }
        }
        Ok(())
    }

    /// Reduce the unit of `tensors` (a finished layer) to its owners and return this rank's sum of squares of its
    /// slice; with `updating` (`lr`, the gradient multiplier) update the slice on the card.
    pub(crate) fn finish(&mut self, ops: &dyn DeviceIo, tensors: &[usize], updating: Option<(f32, f32)>, rule: Rule) -> io::Result<f64> {
        let Some(&first) = tensors.first() else { return Ok(0.0) };
        let u = self.zero.unit_of(first);
        self.reduce(ops, u)?;
        let local = self.zero.unit_sum_sq(u);
        if let Some((lr, multiplier)) = updating {
            self.update(u, lr, multiplier, rule)?;
        }
        Ok(local)
    }

    fn reduce(&mut self, ops: &dyn DeviceIo, u: usize) -> io::Result<()> {
        ops.gpu().flush();
        self.zero.reduce_unit(&*self.group.collective, u, self.scale).map_err(io::Error::other)?;
        self.reduced[u] = true;
        Ok(())
    }

    fn update(&self, u: usize, lr: f32, multiplier: f32, rule: Rule) -> io::Result<()> {
        let Rule::AdamW { adam, wd } = rule else { return Err(invalid(format!("the sharded update is AdamW, not {rule:?}"))) };
        self.zero.update_unit(u, &UnitUpdate { t: self.step, lr, wd, adam }, multiplier).map_err(invalid)
    }

    /// The group's sum of squares given this rank's `local` share.
    pub(crate) fn global_sum(&self, local: f64) -> io::Result<f64> {
        let coll = &*self.group.collective;
        let sum = pollster::block_on(coll.all_reduce(self.group.rank, Payload::f32(vec![local as f32]))).map_err(io::Error::other)?.data;
        Ok(f64::from(sum.first().copied().unwrap_or(f32::NAN)))
    }

    /// Sum `grads` (the remainder every rank holds whole) over the group in place, each rank's weighted by its share.
    pub(crate) fn reduce_remainder(&self, grads: &crate::DeviceGrads) -> io::Result<()> {
        let coll = &*self.group.collective;
        let rank = self.group.rank;
        for dtype in [CollectiveDtype::F32, CollectiveDtype::Bf16] {
            let bufs: Vec<(&DeviceBuffer, u64)> = grads.tensors.iter().filter(|t| t.2 == dtype).map(|(b, n, _)| (b, *n)).collect();
            if bufs.is_empty() {
                continue;
            }
            if pollster::block_on(coll.all_reduce_device(rank, &grads.gpu, &bufs, dtype, self.scale)).map_err(io::Error::other)? {
                continue;
            }
            for (buf, n) in bufs {
                let local: Vec<f32> = crate::zero::read_elems(&grads.gpu, buf, 0, n, dtype).into_iter().map(|g| g * self.scale).collect();
                let sum = pollster::block_on(coll.all_reduce(rank, Payload::f32(local))).map_err(io::Error::other)?.data;
                crate::zero::write_elems(&grads.gpu, buf, 0, &sum, dtype);
            }
        }
        Ok(())
    }

    /// Reduce the tables' gradients (complete once the backward is) and return the group's sum of their squares.
    pub(crate) fn tables_sum_sq(&mut self, ops: &dyn DeviceIo, tables: &[usize]) -> io::Result<f64> {
        let mut local = 0.0;
        for &ti in tables {
            let u = self.zero.unit_of(ti);
            if !self.reduced[u] {
                self.reduce(ops, u)?;
            }
            local += self.zero.unit_sum_sq(u);
        }
        self.global_sum(local)
    }

    /// Update the tables' slices (reducing their gradients first where this pass has not yet).
    pub(crate) fn update_tables(&mut self, ops: &dyn DeviceIo, tables: &[usize], lr: f32, multiplier: f32, rule: Rule) -> io::Result<()> {
        for &ti in tables {
            let u = self.zero.unit_of(ti);
            if !self.reduced[u] {
                self.reduce(ops, u)?;
            }
            self.update(u, lr, multiplier, rule)?;
        }
        self.zero.settle(&*self.group.collective).map_err(io::Error::other)
    }
}
