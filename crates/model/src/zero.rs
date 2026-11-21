// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sharded data-parallel training for its clients: full-parameter fine-tunes of
// models whose optimizer state does not fit one accelerator, spread over the cards and nodes of a cluster. If your
// team needs expertise in ZeRO-style optimizer and gradient sharding, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Sharded data-parallel training: the optimizer state (and, from stage 2 on, the reduced gradient) of a
//! data-parallel group is split by rank instead of replicated, so every rank holds `1/world` of it.
//!
//! # Layout
//!
//! [`ShardLayout`] is a flat, deterministic partition of the trained tensors: they are packed in backward order
//! (last tensor first, the order a backward pass completes them) into **units** of at most a cap of elements, each
//! member starting on a multiple of the alignment (two for bf16, so a copy moves whole 32-bit words). A unit's flat
//! vector is padded to a multiple of `world * align` elements and rank `r` owns its slice `r` of every unit. The
//! layout is a pure function of the tensors' names, sizes and element types, the cap and the world size, so every
//! rank computes the same one, and a checkpoint records it: loading into another world size reassembles the logical
//! tensors from it ([`read_logical`]) and slices them again.
//!
//! # A step
//!
//! For every unit the members' gradients are packed into one flat staging buffer and reduced to their owner: by an
//! all-reduce of which the owner keeps its slice ([`Sharding::Optimizer`], ZeRO-1) or by a reduce-scatter
//! ([`Sharding::Gradients`], ZeRO-2, half the traffic). Each rank weighs its gradient by its share of the step's
//! items before the sum, as [`crate::reduce_grads_weighted`] does. The global gradient norm is the all-reduced sum
//! of the shards' sums of squares; a non-finite norm, or a non-finite loss on any rank, skips the step on every rank.
//! The owner runs AdamW on its slice against an f32 master copy and moments in host memory
//! ([`optim::Adam::update`], the arithmetic of the device kernel), writes the new values (in the parameter's element
//! type) into its device shard, and the unit's shards are all-gathered and unpacked into the model's parameters.
//!
//! Collectives run on the device when the transport has a device path ([`Collective::reduce_scatter_device`],
//! [`Collective::all_gather_device`]: NCCL for a job's ranks on CUDA cards) and through its payload ops otherwise,
//! staged through host memory: the same results, slower. Gradients reduced by NCCL are not bitwise the host
//! star's rank-ordered sum, so a sharded step equals the replicated one within float tolerance, not bit for bit; for
//! an unchanged layout, card set and library a repeat is bit-identical ("repeatable"), and every rank receives the
//! same parameters from the all-gather.
//!
//! # What this does not do (yet)
//!
//! [`Sharding::Parameters`] (ZeRO-3) needs the forward and backward to ask for each block's parameters before using
//! them; a trainer that keeps every parameter resident is refused by that reason. Narrow optimizer-state formats and
//! factored rules are refused by name: the sharded state is f32 AdamW.

use std::ops::Range;

use serde::{Deserialize, Serialize};

/// What a data-parallel group shards across its ranks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Sharding {
    /// Every rank holds everything (plain data parallelism).
    #[default]
    Replicated,
    /// ZeRO-1: the optimizer state (master copy and moments) is sharded; gradients are all-reduced.
    Optimizer,
    /// ZeRO-2: the optimizer state is sharded and gradients are reduce-scattered to their owner.
    Gradients,
    /// ZeRO-3: the parameters are sharded too, gathered block by block where they are used.
    Parameters,
}

impl Sharding {
    /// Parse a stage: `0`-`3` or `replicated`, `optimizer`, `gradients`, `parameters`.
    pub fn parse(s: &str) -> Result<Sharding, String> {
        match s {
            "0" | "replicated" => Ok(Sharding::Replicated),
            "1" | "optimizer" => Ok(Sharding::Optimizer),
            "2" | "gradients" => Ok(Sharding::Gradients),
            "3" | "parameters" => Ok(Sharding::Parameters),
            other => Err(format!("sharding {other:?}: expected 0-3 or replicated, optimizer, gradients, parameters")),
        }
    }

    /// The ZeRO stage number.
    pub fn stage(self) -> u8 {
        self as u8
    }

    /// Whether this is the default, which a plan file leaves out.
    pub fn is_replicated(&self) -> bool {
        *self == Sharding::Replicated
    }
}

/// Default cap of a unit, in elements (128 MiB of f32): the size of the staging buffer a step packs through, unless
/// a single tensor is larger.
pub const DEFAULT_UNIT_ELEMS: u64 = 32 << 20;

/// One unit of a [`ShardLayout`]: members `(tensor index, offset, count)` in its flat vector, padded to `padded`
/// elements (a multiple of `world * align`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Unit {
    pub members: Vec<(usize, u64, u64)>,
    pub padded: u64,
}

/// The part of a tensor a rank owns within one unit: `len` elements from `tensor_offset` in the tensor, at
/// `shard_offset` in the rank's slice of the unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub tensor: usize,
    pub tensor_offset: u64,
    pub shard_offset: u64,
    pub len: u64,
}

/// A flat, deterministic partition of tensors into units sharded evenly over `world` ranks (module documentation).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardLayout {
    pub names: Vec<String>,
    pub numels: Vec<u64>,
    pub world: usize,
    pub align: u64,
    pub units: Vec<Unit>,
}

impl ShardLayout {
    /// Pack tensors `names`/`numels` in backward order into units of at most `cap` elements (a larger tensor is a
    /// unit of its own); only tensors with the same `group` share a unit (element type, learning-rate multiplier).
    pub fn plan(names: &[String], numels: &[u64], groups: &[u32], cap: u64, world: usize, align: u64) -> Result<ShardLayout, String> {
        if world == 0 {
            return Err("a sharded layout needs at least one rank".into());
        }
        if names.len() != numels.len() || names.len() != groups.len() {
            return Err(format!("{} names, {} sizes and {} groups do not describe one tensor set", names.len(), numels.len(), groups.len()));
        }
        let align = align.max(1);
        let quantum = world as u64 * align;
        let mut units = Vec::new();
        let mut open: Vec<(usize, u64, u64)> = Vec::new();
        let mut open_group = None;
        let mut filled = 0u64;
        let close = |open: &mut Vec<(usize, u64, u64)>, filled: &mut u64, units: &mut Vec<Unit>| {
            if !open.is_empty() {
                units.push(Unit { members: std::mem::take(open), padded: filled.div_ceil(quantum).max(1) * quantum });
                *filled = 0;
            }
        };
        for index in (0..names.len()).rev() {
            let count = numels[index];
            if count == 0 {
                continue;
            }
            let start = filled.div_ceil(align) * align;
            if open_group != Some(groups[index]) || (filled > 0 && start + count > cap) {
                close(&mut open, &mut filled, &mut units);
                open_group = Some(groups[index]);
            }
            let start = filled.div_ceil(align) * align;
            open.push((index, start, count));
            filled = start + count;
        }
        close(&mut open, &mut filled, &mut units);
        Ok(ShardLayout { names: names.to_vec(), numels: numels.to_vec(), world, align, units })
    }

    /// Elements of unit `u` each rank owns.
    pub fn shard_len(&self, u: usize) -> u64 {
        self.units[u].padded / self.world as u64
    }

    /// The elements of unit `u`'s flat vector rank `rank` owns.
    pub fn slice(&self, u: usize, rank: usize) -> Range<u64> {
        let n = self.shard_len(u);
        rank as u64 * n..(rank as u64 + 1) * n
    }

    /// The tensors' parts rank `rank` owns in unit `u`, in member order.
    pub fn pieces(&self, u: usize, rank: usize) -> Vec<Piece> {
        let slice = self.slice(u, rank);
        self.units[u]
            .members
            .iter()
            .filter_map(|&(tensor, offset, count)| {
                let start = slice.start.max(offset);
                let end = slice.end.min(offset + count);
                (start < end).then(|| Piece { tensor, tensor_offset: start - offset, shard_offset: start - slice.start, len: end - start })
            })
            .collect()
    }

    /// Elements one rank's shards add up to.
    pub fn shard_elems(&self) -> u64 {
        (0..self.units.len()).map(|u| self.shard_len(u)).sum()
    }

    /// The largest unit, in elements: what the staging buffer holds.
    pub fn max_unit(&self) -> u64 {
        self.units.iter().map(|u| u.padded).max().unwrap_or(0)
    }

    /// Reassemble whole tensors from every rank's unit shards (`shards[rank][unit]`).
    pub fn gather(&self, shards: &[Vec<Vec<f32>>]) -> Result<Vec<Vec<f32>>, String> {
        if shards.len() != self.world {
            return Err(format!("{} ranks' shards for a layout of {} ranks", shards.len(), self.world));
        }
        let mut tensors: Vec<Vec<f32>> = self.numels.iter().map(|&n| vec![0.0; n as usize]).collect();
        for (rank, units) in shards.iter().enumerate() {
            if units.len() != self.units.len() {
                return Err(format!("rank {rank} holds {} unit shards, the layout has {}", units.len(), self.units.len()));
            }
            for (u, shard) in units.iter().enumerate() {
                if shard.len() as u64 != self.shard_len(u) {
                    return Err(format!("rank {rank} unit {u}: {} elements, the layout's shard holds {}", shard.len(), self.shard_len(u)));
                }
                for p in self.pieces(u, rank) {
                    let (t, s) = (p.tensor_offset as usize, p.shard_offset as usize);
                    tensors[p.tensor][t..t + p.len as usize].copy_from_slice(&shard[s..s + p.len as usize]);
                }
            }
        }
        Ok(tensors)
    }

    /// Rank `rank`'s unit shards of whole `tensors` (padding zero).
    pub fn split(&self, tensors: &[Vec<f32>], rank: usize) -> Result<Vec<Vec<f32>>, String> {
        for (i, t) in tensors.iter().enumerate() {
            if t.len() as u64 != self.numels[i] {
                return Err(format!("{}: {} elements, the layout's tensor has {}", self.names[i], t.len(), self.numels[i]));
            }
        }
        Ok((0..self.units.len())
            .map(|u| {
                let mut shard = vec![0.0; self.shard_len(u) as usize];
                for p in self.pieces(u, rank) {
                    let (t, s) = (p.tensor_offset as usize, p.shard_offset as usize);
                    shard[s..s + p.len as usize].copy_from_slice(&tensors[p.tensor][t..t + p.len as usize]);
                }
                shard
            })
            .collect())
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use device::*;

/// The sharded optimizer and its checkpoints: native only (it runs a rank of a group on a card).
#[cfg(not(target_arch = "wasm32"))]
mod device {
    use std::collections::BTreeMap;
    use std::io;
    use std::path::{Path, PathBuf};

    use gpu_core::{CollectiveDtype, DeviceBuffer, Gpu};
    use serde::Deserialize;

    use super::{ShardLayout, Sharding};
    use crate::collective::{Collective, CollectiveError, CollectiveResult, Payload};

    /// One trained tensor where it lives on the device: its value and gradient buffers and their element types.
    #[derive(Clone)]
    pub struct DeviceParam {
        pub name: String,
        pub value: DeviceBuffer,
        pub value_dtype: CollectiveDtype,
        pub grad: DeviceBuffer,
        pub grad_dtype: CollectiveDtype,
        pub numel: u64,
        /// The tensor's learning-rate multiplier (1.0 unless the trainer sets one).
        pub lr_mult: f32,
    }

    /// A model's trained tensors on its card ([`crate::Model::device_params`]), in [`crate::Model::param_names`] order.
    pub struct DeviceParams {
        pub gpu: Gpu,
        pub tensors: Vec<DeviceParam>,
    }

    impl DeviceParams {
        /// The fp32 values and gradients of `names` in `store`, or which one it does not hold.
        pub fn from_store(gpu: &Gpu, store: &paramstore::ParamStore, names: &[String]) -> Result<DeviceParams, String> {
            let tensors = names
                .iter()
                .map(|n| match (store.weight.get(n), store.grad.get(n)) {
                    (Some(value), Some(grad)) => Ok(DeviceParam {
                        name: n.clone(),
                        value: value.clone(),
                        value_dtype: CollectiveDtype::F32,
                        grad: grad.clone(),
                        grad_dtype: CollectiveDtype::F32,
                        numel: store.numel(n) as u64,
                        lr_mult: store.lr_mult_of(n),
                    }),
                    _ => Err(format!("{n}: the parameter store holds no value and gradient for it on the device")),
                })
                .collect::<Result<_, _>>()?;
            Ok(DeviceParams { gpu: gpu.share(), tensors })
        }
    }

    /// The hyperparameters of one sharded AdamW step.
    #[derive(Clone, Copy, Debug)]
    pub struct StepArgs {
        /// AdamW's 1-based step (the bias corrections).
        pub t: u32,
        pub lr: f32,
        pub wd: f32,
        pub adam: optim::Adam,
        pub clip: Option<f32>,
        /// Multiplies every gradient (the `1/K` of `K` accumulated micro-batches).
        pub extra_scale: f32,
    }

    /// What a sharded step did.
    #[derive(Clone, Debug, PartialEq)]
    pub enum StepOutcome {
        /// The update was applied; the global gradient norm after the reduction (before `extra_scale` and the clip).
        Applied { grad_norm: f32 },
        /// No rank's parameters or state changed, for the reason given (the same on every rank).
        Skipped(String),
    }

    /// Element bytes of `dtype`, and its alignment in elements so that a member starts on a 32-bit word.
    fn align_of(dtype: CollectiveDtype) -> u64 {
        4 / dtype.bytes()
    }

    /// Words of `n` elements of `dtype`.
    fn words(n: u64, dtype: CollectiveDtype) -> u64 {
        (n * dtype.bytes()).div_ceil(4)
    }

    /// Read `n` elements of `dtype` from `buf` at element `offset` (even for bf16) as f32.
    pub fn read_elems(gpu: &Gpu, buf: &DeviceBuffer, offset: u64, n: u64, dtype: CollectiveDtype) -> Vec<f32> {
        match dtype {
            CollectiveDtype::F32 => {
                let mut out = vec![0.0; n as usize];
                gpu.read_into(buf, offset, &mut out);
                out
            }
            CollectiveDtype::Bf16 => {
                debug_assert!(offset.is_multiple_of(2), "bf16 reads start on a word");
                let mut w = vec![0.0f32; words(n, dtype) as usize];
                gpu.read_into(buf, offset / 2, &mut w);
                w.iter().flat_map(|w| [f32::from_bits(w.to_bits() << 16), f32::from_bits(w.to_bits() & 0xFFFF_0000)]).take(n as usize).collect()
            }
        }
    }

    /// Write `data` as `dtype` into `buf` at element `offset` (even for bf16; bf16 rounds to nearest even).
    pub fn write_elems(gpu: &Gpu, buf: &DeviceBuffer, offset: u64, data: &[f32], dtype: CollectiveDtype) {
        match dtype {
            CollectiveDtype::F32 => gpu.write_f32_at(buf, offset, data),
            CollectiveDtype::Bf16 => {
                debug_assert!(offset.is_multiple_of(2), "bf16 writes start on a word");
                gpu.write_at(buf, offset / 2, &crate::half::pack_bf16(data))
            }
        }
    }

    /// Reduce-scatter on the device when the transport can, through its payload ops otherwise (module documentation).
    #[allow(clippy::too_many_arguments)]
    fn reduce_scatter(coll: &dyn Collective, rank: usize, gpu: &Gpu, send: &DeviceBuffer, recv: &DeviceBuffer, count: u64, dtype: CollectiveDtype, scale: f32) -> CollectiveResult<()> {
        if pollster::block_on(coll.reduce_scatter_device(rank, gpu, send, recv, count, dtype, scale))? {
            return Ok(());
        }
        let world = coll.world_size() as u64;
        let local: Vec<f32> = read_elems(gpu, send, 0, count * world, dtype).into_iter().map(|g| g * scale).collect();
        let mine = pollster::block_on(coll.reduce_scatter(rank, Payload::f32(local)))?.data;
        write_elems(gpu, recv, 0, &mine, dtype);
        Ok(())
    }

    /// All-reduce of `buf`'s first `count` elements in place, each rank's weighted by `scale`.
    fn all_reduce(coll: &dyn Collective, rank: usize, gpu: &Gpu, buf: &DeviceBuffer, count: u64, dtype: CollectiveDtype, scale: f32) -> CollectiveResult<()> {
        if pollster::block_on(coll.all_reduce_device(rank, gpu, &[(buf, count)], dtype, scale))? {
            return Ok(());
        }
        let local: Vec<f32> = read_elems(gpu, buf, 0, count, dtype).into_iter().map(|g| g * scale).collect();
        let sum = pollster::block_on(coll.all_reduce(rank, Payload::f32(local)))?.data;
        write_elems(gpu, buf, 0, &sum, dtype);
        Ok(())
    }

    /// All-gather on the device when the transport can, through its payload ops otherwise.
    fn all_gather(coll: &dyn Collective, rank: usize, gpu: &Gpu, send: &DeviceBuffer, recv: &DeviceBuffer, count: u64, dtype: CollectiveDtype) -> CollectiveResult<()> {
        if pollster::block_on(coll.all_gather_device(rank, gpu, send, recv, count, dtype))? {
            return Ok(());
        }
        let mine = read_elems(gpu, send, 0, count, dtype);
        let all = pollster::block_on(coll.all_gather(rank, Payload::f32(mine)))?.data;
        write_elems(gpu, recv, 0, &all, dtype);
        Ok(())
    }

    /// The state one rank holds for one unit.
    struct UnitState {
        /// The unit's element types.
        value_dtype: CollectiveDtype,
        grad_dtype: CollectiveDtype,
        lr_mult: f32,
        /// This rank's reduced gradient slice, in host memory (kept between steps: its pages are faulted in once).
        grad: Vec<f32>,
        /// f32 master copy and AdamW moments of the slice, in host memory.
        master: Vec<f32>,
        m: Vec<f32>,
        v: Vec<f32>,
    }

    /// Sharded AdamW over a data-parallel group (module documentation). One per rank; every rank of the group builds it
    /// from the same model and calls every method in the same order.
    pub struct ZeroOptimizer {
        sharding: Sharding,
        layout: ShardLayout,
        rank: usize,
        gpu: Gpu,
        params: Vec<DeviceParam>,
        units: Vec<UnitState>,
        /// One unit's flat vector: packed gradients before a reduction, gathered values after one.
        staging: DeviceBuffer,
        /// One unit's slice on the device: the reduced gradient this rank owns, then the new values it gathers out.
        shard: DeviceBuffer,
        /// The optimizer steps applied so far.
        step: u32,
    }

    /// Stable small group key of a tensor's element types and learning-rate multiplier: only tensors with the same one
    /// share a unit.
    fn group_keys(params: &[DeviceParam]) -> Vec<u32> {
        let mut keys: Vec<(CollectiveDtype, CollectiveDtype, u32)> = Vec::new();
        params
            .iter()
            .map(|p| {
                let key = (p.value_dtype, p.grad_dtype, p.lr_mult.to_bits());
                match keys.iter().position(|k| *k == key) {
                    Some(i) => i as u32,
                    None => {
                        keys.push(key);
                        keys.len() as u32 - 1
                    }
                }
            })
            .collect()
    }

    impl ZeroOptimizer {
        /// Shard the optimizer state of `params` over `coll`'s ranks as `sharding` says, seeding each rank's master slice
        /// from the parameters' current values. `unit_elems` caps a unit ([`DEFAULT_UNIT_ELEMS`]). Every rank calls it
        /// with the same parameters; refused for [`Sharding::Replicated`] (nothing to shard) and [`Sharding::Parameters`]
        /// (these parameters are resident on every rank).
        pub fn new(params: DeviceParams, sharding: Sharding, coll: &dyn Collective, rank: usize, unit_elems: u64) -> Result<ZeroOptimizer, String> {
            match sharding {
                Sharding::Replicated => return Err("a replicated run has no state to shard".into()),
                Sharding::Parameters => {
                    return Err("parameter sharding (ZeRO-3) needs a trainer that gathers each block's parameters where it uses them; this one keeps every parameter resident on every rank".into())
                }
                Sharding::Optimizer | Sharding::Gradients => {}
            }
            let world = coll.world_size();
            if rank >= world {
                return Err(format!("rank {rank} is outside a group of {world}"));
            }
            if params.tensors.is_empty() {
                return Err("there are no trained parameters to shard".into());
            }
            for p in &params.tensors {
                for dtype in [p.value_dtype, p.grad_dtype] {
                    if !p.numel.is_multiple_of(align_of(dtype)) {
                        return Err(format!("{}: {} elements of {dtype:?} are not a whole number of 32-bit words; sharding moves whole words", p.name, p.numel));
                    }
                }
            }
            let names: Vec<String> = params.tensors.iter().map(|p| p.name.clone()).collect();
            let numels: Vec<u64> = params.tensors.iter().map(|p| p.numel).collect();
            let align = params.tensors.iter().flat_map(|p| [align_of(p.value_dtype), align_of(p.grad_dtype)]).max().unwrap_or(1);
            let layout = ShardLayout::plan(&names, &numels, &group_keys(&params.tensors), unit_elems.max(1), world, align)?;
            let gpu = params.gpu.share();
            let staging = gpu.try_storage(layout.max_unit()).map_err(|e| format!("the sharded step's staging buffer ({} elements): {e:?}", layout.max_unit()))?;
            let max_shard = (0..layout.units.len()).map(|u| layout.shard_len(u)).max().unwrap_or(0);
            let shard = gpu.try_storage(max_shard.max(1)).map_err(|e| format!("the sharded step's shard buffer ({max_shard} elements): {e:?}"))?;
            let mut units = Vec::with_capacity(layout.units.len());
            for (u, unit) in layout.units.iter().enumerate() {
                let first = &params.tensors[unit.members[0].0];
                let n = layout.shard_len(u);
                let mut state = UnitState {
                    value_dtype: first.value_dtype,
                    grad_dtype: first.grad_dtype,
                    lr_mult: first.lr_mult,
                    grad: vec![0.0; n as usize],
                    master: Vec::new(),
                    m: vec![0.0; n as usize],
                    v: vec![0.0; n as usize],
                };
                // The master slice is this rank's part of the current values: packed into the staging buffer, then read.
                Self::pack(&gpu, &layout, u, &params.tensors, &staging, |p| (&p.value, p.value_dtype));
                gpu.poll_wait();
                state.master = read_elems(&gpu, &staging, layout.slice(u, rank).start, n, state.value_dtype);
                units.push(state);
            }
            Ok(ZeroOptimizer { sharding, layout, rank, gpu, params: params.tensors, units, staging, shard, step: 0 })
        }

        pub fn sharding(&self) -> Sharding {
            self.sharding
        }

        pub fn layout(&self) -> &ShardLayout {
            &self.layout
        }

        /// The optimizer steps applied so far.
        pub fn step_count(&self) -> u32 {
            self.step
        }

        /// Host bytes this rank's optimizer state (master and moments) and its gradient slice hold.
        pub fn host_bytes(&self) -> u64 {
            self.units.iter().map(|u| 4 * (u.grad.len() + u.master.len() + u.m.len() + u.v.len()) as u64).sum()
        }

        /// Device bytes the step adds to the model: one unit's staging buffer and one slice.
        pub fn device_bytes(&self) -> u64 {
            4 * (self.layout.max_unit() + (0..self.units.len()).map(|u| self.layout.shard_len(u)).max().unwrap_or(0))
        }

        /// Copy unit `u`'s members (`pick` chooses the buffer and type) into `staging` and zero its gaps and padding.
        fn pack(gpu: &Gpu, layout: &ShardLayout, u: usize, params: &[DeviceParam], staging: &DeviceBuffer, pick: impl Fn(&DeviceParam) -> (&DeviceBuffer, CollectiveDtype)) {
            let unit = &layout.units[u];
            let mut at = 0u64;
            for &(tensor, offset, count) in &unit.members {
                let (buf, dtype) = pick(&params[tensor]);
                if offset > at {
                    write_elems(gpu, staging, at, &vec![0.0; (offset - at) as usize], dtype);
                }
                gpu.copy_to(buf, 0, gpu, staging, offset * dtype.bytes(), 4 * words(count, dtype));
                at = offset + count;
            }
            if unit.padded > at {
                let (_, dtype) = pick(&params[unit.members[0].0]);
                write_elems(gpu, staging, at, &vec![0.0; (unit.padded - at) as usize], dtype);
            }
        }

        /// Reduce every unit's gradient to this rank's slice in host memory, each rank's gradient weighted by `scale`.
        fn reduce(&mut self, coll: &dyn Collective, scale: f32) -> CollectiveResult<()> {
            let gpu = &self.gpu;
            for (u, state) in self.units.iter_mut().enumerate() {
                let dtype = state.grad_dtype;
                Self::pack(gpu, &self.layout, u, &self.params, &self.staging, |p| (&p.grad, p.grad_dtype));
                let n = self.layout.shard_len(u);
                let (from, at) = match self.sharding {
                    Sharding::Optimizer => {
                        all_reduce(coll, self.rank, gpu, &self.staging, self.layout.units[u].padded, dtype, scale)?;
                        (&self.staging, self.layout.slice(u, self.rank).start)
                    }
                    _ => {
                        reduce_scatter(coll, self.rank, gpu, &self.staging, &self.shard, n, dtype, scale)?;
                        (&self.shard, 0)
                    }
                };
                match dtype {
                    CollectiveDtype::F32 => gpu.read_into(from, at, &mut state.grad),
                    CollectiveDtype::Bf16 => state.grad = read_elems(gpu, from, at, n, dtype),
                }
            }
            Ok(())
        }

        /// Write every unit's value slice from its master and all-gather the units into the model's parameters.
        fn publish(&self, coll: &dyn Collective) -> CollectiveResult<()> {
            let gpu = &self.gpu;
            for (u, state) in self.units.iter().enumerate() {
                let dtype = state.value_dtype;
                write_elems(gpu, &self.shard, 0, &state.master, dtype);
                all_gather(coll, self.rank, gpu, &self.shard, &self.staging, self.layout.shard_len(u), dtype)?;
                for &(tensor, offset, count) in &self.layout.units[u].members {
                    gpu.copy_to(&self.staging, offset * dtype.bytes(), gpu, &self.params[tensor].value, 0, 4 * words(count, dtype));
                }
            }
            gpu.poll_wait();
            Ok(())
        }

        /// One sharded step after this rank's backward: `local_weight` is how many items this rank's gradient averages
        /// over (as [`crate::reduce_grads_weighted`] takes it) and `loss_finite` whether its loss was. A collective
        /// failure leaves this rank's parameters and state untouched (other ranks may have moved on: restart from the
        /// last commit).
        pub fn step(&mut self, coll: &dyn Collective, local_weight: f32, loss_finite: bool, args: &StepArgs) -> CollectiveResult<StepOutcome> {
            let rank = self.rank;
            if args.t != self.step + 1 {
                return Err(CollectiveError::Transport(format!("sharded step {} after {} applied steps: the optimizer and the trainer disagree on the step", args.t, self.step)));
            }
            let totals = pollster::block_on(coll.all_reduce(rank, Payload::f32(vec![local_weight, if loss_finite { 0.0 } else { 1.0 }])))?.data;
            if totals.len() != 2 {
                return Err(CollectiveError::Transport(format!("the step's weight reduction returned {} values", totals.len())));
            }
            if totals[1] > 0.0 {
                return Ok(StepOutcome::Skipped(format!("non-finite loss on {} rank(s)", totals[1] as u32)));
            }
            let clock = std::time::Instant::now();
            self.reduce(coll, local_weight / totals[0].max(1e-12))?;
            gpu_core::profile::stage_time("sharded step: reduce the gradients to their owners and read them", clock);
            let clock = std::time::Instant::now();
            let local_sq: f64 = self.units.iter().map(|s| backend_cpu::par::sum_sq_f64(std::slice::from_ref(&s.grad))).sum();
            let global = pollster::block_on(coll.all_reduce(rank, Payload::f32(vec![local_sq as f32])))?.data;
            let sum_sq = f64::from(*global.first().ok_or_else(|| CollectiveError::Transport("the norm reduction returned nothing".into()))?);
            if !sum_sq.is_finite() {
                return Ok(StepOutcome::Skipped("non-finite gradient".into()));
            }
            let multiplier = optim::grad_multiplier(sum_sq, args.clip, args.extra_scale);
            let bc = args.adam.bias_corrections(args.t);
            for state in &mut self.units {
                let (adam, lr, wd) = (args.adam, args.lr * state.lr_mult, args.wd);
                backend_cpu::par::zip3_mut(&mut state.master, &mut state.m, &mut state.v, &state.grad, |w, m, v, g| adam.update(bc, lr, wd, g * multiplier, w, m, v));
            }
            gpu_core::profile::stage_time("sharded step: norm and AdamW on the host shard", clock);
            self.step = args.t;
            let clock = std::time::Instant::now();
            self.publish(coll)?;
            gpu_core::profile::stage_time("sharded step: write the value shard and all-gather the parameters", clock);
            Ok(StepOutcome::Applied { grad_norm: sum_sq.sqrt() as f32 })
        }

        /// This rank's checkpoint file for `step` in `dir`.
        pub fn shard_path(dir: &Path, step: u32, rank: usize, world: usize) -> PathBuf {
            dir.join(format!("shard-{step:08}-rank{rank:05}-of{world:05}.safetensors"))
        }

        /// Write this rank's shard (master, moments, the layout and `extra`: the trainer's per-rank state such as its
        /// RNG) durably to `dir` as of the steps applied so far; returns the file. Every rank writes its own;
        /// [`commit_generation`] names them together.
        pub fn save(&self, dir: &Path, extra: serde_json::Value) -> io::Result<PathBuf> {
            std::fs::create_dir_all(dir)?;
            let path = Self::shard_path(dir, self.step, self.rank, self.layout.world);
            let mut tensors = Vec::with_capacity(3 * self.units.len());
            for (u, s) in self.units.iter().enumerate() {
                for (kind, data) in [("master", &s.master), ("m", &s.m), ("v", &s.v)] {
                    tensors.push((format!("{kind}:{u}"), vec![data.len() as u64], data.clone()));
                }
            }
            let header = serde_json::json!({
                "format": SHARD_FORMAT,
                "step": self.step,
                "rank": self.rank,
                "sharding": self.sharding,
                "layout": self.layout,
                "extra": extra,
            });
            let p = path.to_str().ok_or_else(|| io::Error::other(format!("{}: not a UTF-8 path", path.display())))?;
            checkpoint::st::save_safetensors(p, &tensors, &header, None)?;
            Ok(path)
        }

        /// Remove this rank's shard files in `dir` but the newest `keep` steps.
        pub fn prune(&self, dir: &Path, keep: usize) -> io::Result<()> {
            let suffix = format!("-rank{:05}-of{:05}.safetensors", self.rank, self.layout.world);
            let mut steps: Vec<(u32, PathBuf)> = std::fs::read_dir(dir)?
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let name = e.file_name().into_string().ok()?;
                    let step = name.strip_prefix("shard-")?.strip_suffix(&suffix)?.parse().ok()?;
                    Some((step, e.path()))
                })
                .collect();
            steps.sort_by_key(|(step, _)| std::cmp::Reverse(*step));
            for (_, path) in steps.into_iter().skip(keep) {
                std::fs::remove_file(path)?;
            }
            Ok(())
        }

        /// Restore the state saved at `step` in `dir` and push the restored values into the model's parameters; returns
        /// this rank's `extra`. When the saved layout is this one, this rank reads its own file; otherwise (another
        /// world size or unit cap) every rank's file of that step is read and the logical tensors are sliced again
        /// ([`read_logical`]), and the `extra` is rank 0's. Every rank calls it.
        pub fn restore(&mut self, coll: &dyn Collective, dir: &Path, step: u32) -> io::Result<serde_json::Value> {
            let own = Self::shard_path(dir, step, self.rank, self.layout.world);
            let saved = if own.exists() { Some(read_shard(&own)?).filter(|(h, _)| h.layout == self.layout) } else { None };
            let extra = if let Some((header, mut tensors)) = saved {
                for (u, s) in self.units.iter_mut().enumerate() {
                    for (kind, slot) in [("master", &mut s.master), ("m", &mut s.m), ("v", &mut s.v)] {
                        *slot = tensors.remove(&format!("{kind}:{u}")).ok_or_else(|| invalid(&own, format!("no {kind} for unit {u}")))?;
                    }
                }
                header.extra
            } else {
                let logical = read_logical(dir, step)?;
                if logical.names != self.layout.names || logical.numels != self.layout.numels {
                    return Err(io::Error::other(format!("{}: the checkpoint at step {step} holds other tensors than this model trains", dir.display())));
                }
                let split = |what: &[Vec<f32>]| self.layout.split(what, self.rank).map_err(io::Error::other);
                let (master, m, v) = (split(&logical.master)?, split(&logical.m)?, split(&logical.v)?);
                for (s, ((master, m), v)) in self.units.iter_mut().zip(master.into_iter().zip(m).zip(v)) {
                    (s.master, s.m, s.v) = (master, m, v);
                }
                logical.extra
            };
            self.step = step;
            self.publish(coll).map_err(io::Error::other)?;
            Ok(extra)
        }
    }

    /// The format tag of a shard file.
    const SHARD_FORMAT: &str = "brain-zero-shard-v1";

    /// What a shard file's header records.
    #[derive(Deserialize)]
    struct ShardHeader {
        format: String,
        step: u32,
        rank: usize,
        layout: ShardLayout,
        #[serde(default)]
        extra: serde_json::Value,
    }

    fn invalid(path: &Path, why: String) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, format!("{}: {why}", path.display()))
    }

    fn read_shard(path: &Path) -> io::Result<(ShardHeader, std::collections::HashMap<String, Vec<f32>>)> {
        let p = path.to_str().ok_or_else(|| invalid(path, "not a UTF-8 path".into()))?;
        let st = checkpoint::st::load_safetensors(p)?;
        let header: ShardHeader = serde_json::from_value(st.config().clone()).map_err(|e| invalid(path, format!("not a shard header: {e}")))?;
        if header.format != SHARD_FORMAT {
            return Err(invalid(path, format!("format {:?}, expected {SHARD_FORMAT}", header.format)));
        }
        Ok((header, st.tensors))
    }

    /// The whole tensors a sharded checkpoint holds, in its layout's order.
    pub struct LogicalState {
        pub names: Vec<String>,
        pub numels: Vec<u64>,
        pub step: u32,
        pub master: Vec<Vec<f32>>,
        pub m: Vec<Vec<f32>>,
        pub v: Vec<Vec<f32>>,
        /// Rank 0's `extra`.
        pub extra: serde_json::Value,
    }

    /// Reassemble the logical tensors of the shards every rank saved at `step` in `dir`: the path a run with another
    /// layout resumes through. Every rank's file must be there, agree on the layout and the step, and cover its slice.
    pub fn read_logical(dir: &Path, step: u32) -> io::Result<LogicalState> {
        let prefix = format!("shard-{step:08}-rank");
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(&prefix) && n.ends_with(".safetensors"))).collect();
        files.sort();
        let first = files.first().ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("{}: no shard saved at step {step}", dir.display())))?;
        let (head, _) = read_shard(first)?;
        let layout = head.layout;
        let world = layout.world;
        let mut by_rank: BTreeMap<usize, (ShardHeader, std::collections::HashMap<String, Vec<f32>>)> = BTreeMap::new();
        for f in &files {
            let (h, t) = read_shard(f)?;
            if h.layout != layout || h.step != step {
                return Err(invalid(f, "its layout or step differs from the other shards of this step".into()));
            }
            if by_rank.insert(h.rank, (h, t)).is_some() {
                return Err(invalid(f, "a second shard of the same rank".into()));
            }
        }
        if by_rank.len() != world || by_rank.keys().copied().ne(0..world) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: step {step} has shards of ranks {:?}, its layout has {world}", dir.display(), by_rank.keys().collect::<Vec<_>>())));
        }
        let mut gathered = Vec::new();
        for kind in ["master", "m", "v"] {
            let shards: Vec<Vec<Vec<f32>>> = by_rank
                .values_mut()
                .map(|(h, t)| (0..layout.units.len()).map(|u| t.remove(&format!("{kind}:{u}")).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("rank {} has no {kind} for unit {u}", h.rank)))).collect::<io::Result<Vec<_>>>())
                .collect::<io::Result<_>>()?;
            gathered.push(layout.gather(&shards).map_err(io::Error::other)?);
        }
        let v = gathered.pop().unwrap_or_default();
        let m = gathered.pop().unwrap_or_default();
        let master = gathered.pop().unwrap_or_default();
        let extra = by_rank.remove(&0).map(|(h, _)| h.extra).unwrap_or_default();
        Ok(LogicalState { names: layout.names, numels: layout.numels, step, master, m, v, extra })
    }

    /// Rank 0 names the shard files every rank saved at `step` in `dir` in one committed generation
    /// ([`crate::CheckpointCommit`]), after every rank's file is durable (the call is a barrier over `coll`). Returns the
    /// manifest's path on rank 0, `None` elsewhere.
    pub fn commit_generation(coll: &dyn Collective, rank: usize, dir: &Path, step: u32, job: &str, plan: &str) -> io::Result<Option<PathBuf>> {
        pollster::block_on(coll.all_reduce(rank, Payload::f32(vec![1.0]))).map_err(io::Error::other)?;
        if rank != 0 {
            return Ok(None);
        }
        let world = coll.world_size();
        let paths: Vec<(String, PathBuf)> = (0..world).map(|r| (format!("shard-{r:05}"), ZeroOptimizer::shard_path(dir, step, r, world))).collect();
        let files: Vec<(&str, &Path)> = paths.iter().map(|(role, p)| (role.as_str(), p.as_path())).collect();
        let commit = crate::CheckpointCommit::build(job, plan, step, world, dir, &files)?;
        commit.write(dir).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("t{i}")).collect()
    }

    /// Every element of every tensor is owned by exactly one rank, units respect the cap (unless one tensor is
    /// larger), are padded to the rank quantum, and members start on the alignment.
    #[test]
    fn a_layout_partitions_every_element_once() {
        let numels = [10u64, 3, 0, 64, 7, 100, 2];
        for (world, align, cap) in [(1, 1, 16), (2, 2, 16), (3, 1, 1000), (4, 2, 20)] {
            let numels: Vec<u64> = numels.iter().map(|n| n.div_ceil(align) * align).collect();
            let layout = ShardLayout::plan(&names(numels.len()), &numels, &vec![0; numels.len()], cap, world, align).unwrap();
            let mut owned: Vec<Vec<u32>> = numels.iter().map(|&n| vec![0; n as usize]).collect();
            for (u, unit) in layout.units.iter().enumerate() {
                assert_eq!(unit.padded % (world as u64 * align), 0);
                assert!(unit.members.len() == 1 || unit.members.last().map(|m| m.1 + m.2).unwrap() <= cap, "unit {u} over the cap");
                for &(_, offset, _) in &unit.members {
                    assert_eq!(offset % align, 0);
                }
                for rank in 0..world {
                    for p in layout.pieces(u, rank) {
                        for i in p.tensor_offset..p.tensor_offset + p.len {
                            owned[p.tensor][i as usize] += 1;
                        }
                    }
                }
            }
            assert!(owned.iter().all(|t| t.iter().all(|&c| c == 1)), "world {world}: {owned:?}");
            assert_eq!(layout, ShardLayout::plan(&names(numels.len()), &numels, &vec![0; numels.len()], cap, world, align).unwrap());
        }
    }

    /// Splitting whole tensors by one layout and gathering them by it is the identity; through another world size
    /// too.
    #[test]
    fn split_then_gather_is_the_identity_across_world_sizes() {
        let numels = vec![6u64, 9, 31, 4];
        let tensors: Vec<Vec<f32>> = numels.iter().enumerate().map(|(t, &n)| (0..n).map(|i| (t * 100) as f32 + i as f32).collect()).collect();
        let mut prev: Option<Vec<Vec<f32>>> = None;
        for world in [1, 2, 3, 4] {
            let layout = ShardLayout::plan(&names(4), &numels, &[0, 0, 1, 0], 16, world, 1).unwrap();
            let shards: Vec<Vec<Vec<f32>>> = (0..world).map(|r| layout.split(&tensors, r).unwrap()).collect();
            let back = layout.gather(&shards).unwrap();
            assert_eq!(back, tensors);
            if let Some(p) = &prev {
                assert_eq!(p, &back);
            }
            prev = Some(back);
        }
    }

    /// Tensors of different groups never share a unit.
    #[test]
    fn groups_never_share_a_unit() {
        let layout = ShardLayout::plan(&names(4), &[2, 2, 2, 2], &[0, 1, 0, 1], 1000, 2, 1).unwrap();
        for unit in &layout.units {
            let g: Vec<usize> = unit.members.iter().map(|m| m.0 % 2).collect();
            assert!(g.windows(2).all(|w| w[0] == w[1]), "{unit:?}");
        }
    }

    #[test]
    fn stages_parse_by_number_and_name() {
        assert_eq!(Sharding::parse("2").unwrap(), Sharding::Gradients);
        assert_eq!(Sharding::parse("optimizer").unwrap(), Sharding::Optimizer);
        assert!(Sharding::parse("4").is_err());
    }
}
