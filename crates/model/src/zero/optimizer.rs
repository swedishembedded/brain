// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sharded data-parallel training for its clients: full-parameter fine-tunes of
// models whose optimizer state does not fit one accelerator, spread over the cards and nodes of a cluster. If your
// team needs expertise in ZeRO-style optimizer and gradient sharding, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The sharded optimizer of a data-parallel rank ([`super`]): the reduction of each unit to its owner, the owner's
//! AdamW over its slice and the gather of the units back into the model's parameters.

use gpu_core::{CollectiveDtype, DeviceBuffer, Gpu, Step};
use optim::gradnorm::{DeviceGradNorm, NormItem};
use paramstore::precision::GradDtype;

use super::{ShardLayout, Sharding};
use crate::collective::{Collective, CollectiveError, CollectiveResult, Payload};

/// The kernels a sharded step dispatches on the device tier: what a model's device must register (or what the
/// optimizer registers on a handle of its own, where the backend shares the device between handles).
pub const KERNELS: &[(&str, &str)] = &[
    ("adamw", kernels::ADAMW),
    ("gradnorm_part", kernels::GRADNORM_PART),
    ("gradnorm_part_bf16", kernels::GRADNORM_PART_BF16),
    ("pack_bf16", kernels::PACK_BF16),
    ("unpack_bf16", kernels::UNPACK_BF16),
];

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

/// Where a rank's slice of the optimizer state lives.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StateTier {
    /// On the card, beside the parameters: the reduced gradient slice, the f32 master and the moments, updated by the
    /// AdamW kernel. The default.
    #[default]
    Device,
    /// In host memory (`moments=host`): the device holds one unit's staging and one slice; the update runs on the
    /// host. For a model whose sharded state does not fit beside it on the card.
    Host,
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

/// Elements of `dtype` per 32-bit word: a member starts on one.
pub(super) fn align_of(dtype: CollectiveDtype) -> u64 {
    4 / dtype.bytes()
}

/// Words of `n` elements of `dtype`.
pub(super) fn words(n: u64, dtype: CollectiveDtype) -> u64 {
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

/// Reduce-scatter on the device when the transport can, through its payload ops otherwise.
#[allow(clippy::too_many_arguments)] // the NCCL shape: rank, card, two buffers, count, type, scale
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
pub(crate) fn all_gather(coll: &dyn Collective, rank: usize, gpu: &Gpu, send: &DeviceBuffer, recv: &DeviceBuffer, count: u64, dtype: CollectiveDtype) -> CollectiveResult<()> {
    if pollster::block_on(coll.all_gather_device(rank, gpu, send, recv, count, dtype))? {
        return Ok(());
    }
    let mine = read_elems(gpu, send, 0, count, dtype);
    let all = pollster::block_on(coll.all_gather(rank, Payload::f32(mine)))?.data;
    write_elems(gpu, recv, 0, &all, dtype);
    Ok(())
}

/// A unit's slice of the state on the card ([`StateTier::Device`]).
pub(super) struct DeviceSlice {
    /// f32 master (the value itself for an f32 unit), moments and reduced gradient.
    pub(super) w: DeviceBuffer,
    pub(super) m: DeviceBuffer,
    pub(super) v: DeviceBuffer,
    pub(super) g: DeviceBuffer,
    /// The bf16 reduced gradient (a bf16-gradient unit) and bf16 value (a bf16-value unit) of the slice.
    pub(super) g16: Option<DeviceBuffer>,
    pub(super) w16: Option<DeviceBuffer>,
    /// The AdamW kernel's per-dispatch descriptor (`kernels::adamw_desc`).
    desc: DeviceBuffer,
}

/// A unit's slice of the state in host memory ([`StateTier::Host`]).
pub(super) struct HostSlice {
    /// The reduced gradient slice (kept between steps: its pages are faulted in once).
    pub(super) grad: Vec<f32>,
    pub(super) master: Vec<f32>,
    pub(super) m: Vec<f32>,
    pub(super) v: Vec<f32>,
}

pub(super) enum Slice {
    Device(DeviceSlice),
    Host(HostSlice),
}

/// The state one rank holds for one unit.
pub(super) struct UnitState {
    pub(super) value_dtype: CollectiveDtype,
    pub(super) grad_dtype: CollectiveDtype,
    lr_mult: f32,
    pub(super) slice: Slice,
}

/// The kernels of the device tier and the handle they are registered on.
struct DeviceKernels {
    gpu: Gpu,
    adamw: usize,
    pack: usize,
    unpack: usize,
    norm: DeviceGradNorm,
    /// AdamW's per-step hyperparameters, one uniform for every unit.
    hparams: DeviceBuffer,
    /// AdamW's device coefficient: 1 (the clip is folded into the host-known scale).
    one: DeviceBuffer,
}

impl DeviceKernels {
    /// The model's handle when it registers every kernel the step needs, else a handle of its own on the same card.
    /// A backend that cannot share a card between handles and a model handle without the kernels is refused.
    fn resolve(gpu: &Gpu) -> Result<DeviceKernels, String> {
        let has_all = KERNELS.iter().all(|(n, _)| gpu.kernel_index(n).is_some());
        let gpu = if has_all {
            gpu.share()
        } else if gpu.kind() == "cpu" {
            // The CPU JIT builds a fresh device for another kernel set; its buffers would not be this model's.
            let missing: Vec<&str> = KERNELS.iter().map(|(n, _)| *n).filter(|n| gpu.kernel_index(n).is_none()).collect();
            return Err(format!("the model's CPU device does not register the sharded step's kernels {missing:?} (register model::zero::KERNELS, or keep the state on the host: moments=host)"));
        } else {
            gpu.new_like(KERNELS)
        };
        let find = |n: &str| gpu.kernel_index(n).ok_or_else(|| format!("kernel '{n}' is not registered"));
        let (adamw, pack, unpack) = (find("adamw")?, find("pack_bf16")?, find("unpack_bf16")?);
        let norm = DeviceGradNorm::new(&gpu)?;
        let hparams = gpu.uniform_dynamic(8);
        let one = gpu.storage(1);
        gpu.write_f32(&one, &[1.0]);
        Ok(DeviceKernels { gpu, adamw, pack, unpack, norm, hparams, one })
    }
}

/// Sharded AdamW over a data-parallel group ([`super`]). One per rank; every rank of the group builds it from the
/// same model and calls every method in the same order.
pub struct ZeroOptimizer {
    pub(super) sharding: Sharding,
    pub(super) layout: ShardLayout,
    pub(super) rank: usize,
    pub(super) gpu: Gpu,
    params: Vec<DeviceParam>,
    pub(super) units: Vec<UnitState>,
    /// Two units' flat vectors, used alternately so the packing of one overlaps the collective of the other.
    staging: [DeviceBuffer; 2],
    /// The host tier's one device slice: the reduced gradient this rank owns, then the values it gathers out.
    shard: Option<DeviceBuffer>,
    kernels: Option<DeviceKernels>,
    /// The optimizer steps applied so far.
    pub(super) step: u32,
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

fn grad_dtype(dtype: CollectiveDtype) -> GradDtype {
    match dtype {
        CollectiveDtype::F32 => GradDtype::F32,
        CollectiveDtype::Bf16 => GradDtype::Bf16,
    }
}

impl ZeroOptimizer {
    /// Shard the optimizer state of `params` over `coll`'s ranks as `sharding` says, its slices held where `tier`
    /// says, seeding each rank's master slice from the parameters' current values. `unit_elems` caps a unit
    /// ([`super::DEFAULT_UNIT_ELEMS`]). Every rank calls it with the same parameters; refused for
    /// [`Sharding::Replicated`] (nothing to shard) and [`Sharding::Parameters`] (these parameters are resident on
    /// every rank).
    pub fn new(params: DeviceParams, sharding: Sharding, tier: StateTier, coll: &dyn Collective, rank: usize, unit_elems: u64) -> Result<ZeroOptimizer, String> {
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
        let kernels = match tier {
            StateTier::Device => Some(DeviceKernels::resolve(&gpu).map_err(|why| format!("the sharded state on the device: {why}"))?),
            StateTier::Host => None,
        };
        let alloc = |n: u64, what: &str| gpu.try_storage(n.max(1)).map_err(|e| format!("{what} ({n} words): {e:?}"));
        let staging = [alloc(layout.max_unit(), "a staging buffer")?, alloc(layout.max_unit(), "a staging buffer")?];
        let max_shard = (0..layout.units.len()).map(|u| layout.shard_len(u)).max().unwrap_or(0);
        let shard = match tier {
            StateTier::Host => Some(alloc(max_shard, "the slice buffer")?),
            StateTier::Device => None,
        };
        let mut units = Vec::with_capacity(layout.units.len());
        for (u, unit) in layout.units.iter().enumerate() {
            let first = &params.tensors[unit.members[0].0];
            let n = layout.shard_len(u);
            let (value_dtype, grad_dtype) = (first.value_dtype, first.grad_dtype);
            // The master slice is this rank's part of the current values: packed into the staging buffer, then taken.
            Self::pack(&gpu, &layout, u, &params.tensors, &staging[0], |p| (&p.value, p.value_dtype));
            let at = layout.slice(u, rank).start;
            let slice = match tier {
                StateTier::Host => {
                    gpu.poll_wait();
                    Slice::Host(HostSlice { grad: vec![0.0; n as usize], master: read_elems(&gpu, &staging[0], at, n, value_dtype), m: vec![0.0; n as usize], v: vec![0.0; n as usize] })
                }
                StateTier::Device => {
                    let k = kernels.as_ref().expect("the device tier resolved its kernels");
                    let w = alloc(n, "a master slice")?;
                    let w16 = (value_dtype == CollectiveDtype::Bf16).then(|| alloc(words(n, value_dtype), "a bf16 value slice")).transpose()?;
                    match &w16 {
                        // A bf16 unit's master starts as its bf16 values, widened on the card.
                        Some(w16) => {
                            gpu.copy_to(&staging[0], at * 2, &gpu, w16, 0, 4 * words(n, value_dtype));
                            gpu.flush();
                            k.gpu.submit(&[], &[k.gpu.step(k.unpack, &[w16, &w], &[n as u32], n as u32)]);
                        }
                        None => {
                            gpu.copy_to(&staging[0], at * 4, &gpu, &w, 0, 4 * n);
                        }
                    }
                    let desc = alloc(3, "a descriptor")?;
                    gpu.write(&desc, &kernels::adamw_desc(n as usize, first.lr_mult));
                    let g16 = (grad_dtype == CollectiveDtype::Bf16).then(|| alloc(words(n, grad_dtype), "a bf16 gradient slice")).transpose()?;
                    Slice::Device(DeviceSlice { w, m: alloc(n, "a moment slice")?, v: alloc(n, "a moment slice")?, g: alloc(n, "a gradient slice")?, g16, w16, desc })
                }
            };
            units.push(UnitState { value_dtype, grad_dtype, lr_mult: first.lr_mult, slice });
        }
        gpu.poll_wait();
        Ok(ZeroOptimizer { sharding, layout, rank, gpu, params: params.tensors, units, staging, shard, kernels, step: 0 })
    }

    pub fn sharding(&self) -> Sharding {
        self.sharding
    }

    pub fn layout(&self) -> &ShardLayout {
        &self.layout
    }

    /// Where this rank's state slices live.
    pub fn tier(&self) -> StateTier {
        if self.kernels.is_some() {
            StateTier::Device
        } else {
            StateTier::Host
        }
    }

    /// The optimizer steps applied so far.
    pub fn step_count(&self) -> u32 {
        self.step
    }

    /// Host bytes this rank's optimizer state (master and moments) and its gradient slice hold.
    pub fn host_bytes(&self) -> u64 {
        self.units
            .iter()
            .map(|u| match &u.slice {
                Slice::Host(h) => 4 * (h.grad.len() + h.master.len() + h.m.len() + h.v.len()) as u64,
                Slice::Device(_) => 0,
            })
            .sum()
    }

    /// Device bytes the optimizer adds to the model: the staging buffers and every slice it keeps on the card.
    pub fn device_bytes(&self) -> u64 {
        let slices: u64 = (0..self.units.len())
            .map(|u| {
                let n = self.layout.shard_len(u);
                match &self.units[u].slice {
                    Slice::Device(d) => 4 * 4 * n + d.g16.as_ref().map_or(0, |_| 2 * n) + d.w16.as_ref().map_or(0, |_| 2 * n),
                    Slice::Host(_) => 0,
                }
            })
            .sum();
        let shard = if self.shard.is_some() { 4 * (0..self.units.len()).map(|u| self.layout.shard_len(u)).max().unwrap_or(0) } else { 0 };
        slices + shard + 2 * 4 * self.layout.max_unit()
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

    /// Reduce every unit's gradient to this rank's slice (on the card, or read into host memory), each rank's
    /// gradient weighted by `scale`. Units alternate between the two staging buffers, so the next unit is packed
    /// while the last one's collective runs.
    fn reduce(&mut self, coll: &dyn Collective, scale: f32) -> CollectiveResult<()> {
        let gpu = &self.gpu;
        for (u, state) in self.units.iter_mut().enumerate() {
            let staging = &self.staging[u % 2];
            let dtype = state.grad_dtype;
            Self::pack(gpu, &self.layout, u, &self.params, staging, |p| (&p.grad, p.grad_dtype));
            let n = self.layout.shard_len(u);
            let target: DeviceBuffer = match &state.slice {
                Slice::Device(d) => d.g16.as_ref().unwrap_or(&d.g).clone(),
                Slice::Host(_) => self.shard.clone().expect("the host tier keeps a slice buffer"),
            };
            let target = &target;
            match self.sharding {
                Sharding::Optimizer => {
                    all_reduce(coll, self.rank, gpu, staging, self.layout.units[u].padded, dtype, scale)?;
                    gpu.copy_to(staging, self.layout.slice(u, self.rank).start * dtype.bytes(), gpu, target, 0, 4 * words(n, dtype));
                }
                _ => reduce_scatter(coll, self.rank, gpu, staging, target, n, dtype, scale)?,
            }
            match &mut state.slice {
                Slice::Host(h) => match dtype {
                    CollectiveDtype::F32 => gpu.read_into(target, 0, &mut h.grad),
                    CollectiveDtype::Bf16 => h.grad = read_elems(gpu, target, 0, n, dtype),
                },
                Slice::Device(d) => {
                    if let (Some(g16), Some(k)) = (&d.g16, &self.kernels) {
                        gpu.flush();
                        k.gpu.submit(&[], &[k.gpu.step(k.unpack, &[g16, &d.g], &[n as u32], n as u32)]);
                        k.gpu.flush();
                    }
                }
            }
        }
        Ok(())
    }

    /// This rank's sum of squares of its reduced gradient slices.
    fn local_sum_sq(&self) -> f64 {
        match &self.kernels {
            Some(k) => {
                let items: Vec<NormItem<'_>> = self
                    .units
                    .iter()
                    .enumerate()
                    .filter_map(|(u, s)| match &s.slice {
                        Slice::Device(d) => Some(NormItem { buf: &d.g, numel: self.layout.shard_len(u) as usize, dtype: grad_dtype(CollectiveDtype::F32) }),
                        Slice::Host(_) => None,
                    })
                    .collect();
                self.gpu.flush();
                k.norm.sum_sq(&k.gpu, &items)
            }
            None => self
                .units
                .iter()
                .map(|s| match &s.slice {
                    Slice::Host(h) => backend_cpu::par::sum_sq_f64(std::slice::from_ref(&h.grad)),
                    Slice::Device(_) => 0.0,
                })
                .sum(),
        }
    }

    /// Gather unit `u`'s value slices from every rank into the model's parameters. On the device tier this is
    /// enqueued behind the unit's update and ordered on the device; the caller waits once for every unit.
    fn gather_unit(&self, coll: &dyn Collective, u: usize) -> CollectiveResult<()> {
        let gpu = &self.gpu;
        let state = &self.units[u];
        let dtype = state.value_dtype;
        let staging = &self.staging[u % 2];
        let n = self.layout.shard_len(u);
        let send: &DeviceBuffer = match &state.slice {
            Slice::Device(d) => d.w16.as_ref().unwrap_or(&d.w),
            Slice::Host(h) => {
                let shard = self.shard.as_ref().expect("the host tier keeps a slice buffer");
                write_elems(gpu, shard, 0, &h.master, dtype);
                shard
            }
        };
        all_gather(coll, self.rank, gpu, send, staging, n, dtype)?;
        for &(tensor, offset, count) in &self.layout.units[u].members {
            gpu.copy_to(staging, offset * dtype.bytes(), gpu, &self.params[tensor].value, 0, 4 * words(count, dtype));
        }
        Ok(())
    }

    /// The AdamW dispatch of unit `u`'s device slice, and the cast of its master to bf16 for a bf16 unit.
    fn update_steps(&self, u: usize) -> Vec<Step> {
        let k = self.kernels.as_ref().expect("the device tier resolved its kernels");
        let n = self.layout.shard_len(u) as u32;
        let Slice::Device(d) = &self.units[u].slice else { return Vec::new() };
        let mut steps = vec![k.gpu.step_buf(k.adamw, &k.hparams, &[&d.w, &d.g, &d.m, &d.v, &d.desc, &k.one], n)];
        if let Some(w16) = &d.w16 {
            steps.push(k.gpu.step(k.pack, &[&d.w, w16], &[n], n.div_ceil(2)));
        }
        steps
    }

    /// Write every unit's values and gather them into the model's parameters (after a restore).
    pub(super) fn publish(&self, coll: &dyn Collective) -> CollectiveResult<()> {
        if let Some(k) = &self.kernels {
            for u in 0..self.units.len() {
                if let Slice::Device(DeviceSlice { w, w16: Some(w16), .. }) = &self.units[u].slice {
                    let n = self.layout.shard_len(u) as u32;
                    self.gpu.flush();
                    k.gpu.submit(&[], &[k.gpu.step(k.pack, &[w, w16], &[n], n.div_ceil(2))]);
                    k.gpu.flush();
                }
            }
        }
        for u in 0..self.units.len() {
            self.gather_unit(coll, u)?;
        }
        pollster::block_on(coll.settle_device(self.rank, &self.gpu))?;
        self.gpu.poll_wait();
        Ok(())
    }

    /// One sharded step after this rank's backward: `local_weight` is how many items this rank's gradient averages
    /// over (as [`crate::reduce_grads_weighted`] takes it) and `loss_finite` whether its loss was. A collective
    /// failure leaves this rank's state as it was before the update began (other ranks may have moved on: restart
    /// from the last commit).
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
        gpu_core::profile::stage_time("sharded step: reduce the gradients to their owners", clock);
        let clock = std::time::Instant::now();
        let local_sq = self.local_sum_sq();
        let global = pollster::block_on(coll.all_reduce(rank, Payload::f32(vec![local_sq as f32])))?.data;
        let sum_sq = f64::from(*global.first().ok_or_else(|| CollectiveError::Transport("the norm reduction returned nothing".into()))?);
        gpu_core::profile::stage_time("sharded step: the global gradient norm", clock);
        if !sum_sq.is_finite() {
            return Ok(StepOutcome::Skipped("non-finite gradient".into()));
        }
        let multiplier = optim::grad_multiplier(sum_sq, args.clip, args.extra_scale);
        let bc = args.adam.bias_corrections(args.t);
        let clock = std::time::Instant::now();
        match &self.kernels {
            Some(k) => {
                let f = f32::to_bits;
                k.gpu.write(&k.hparams, &[f(args.lr), f(args.adam.beta1), f(args.adam.beta2), f(args.adam.eps), f(args.wd), f(bc.0), f(bc.1), f(multiplier)]);
                // Unit u's gather runs on the collective's stream while unit u + 1 updates on the compute stream.
                self.gpu.flush();
                for u in 0..self.units.len() {
                    let steps = self.update_steps(u);
                    k.gpu.submit(&[], &steps);
                    k.gpu.flush();
                    self.gather_unit(coll, u)?;
                }
            }
            None => {
                for state in &mut self.units {
                    let (adam, lr, wd) = (args.adam, args.lr * state.lr_mult, args.wd);
                    if let Slice::Host(h) = &mut state.slice {
                        backend_cpu::par::zip3_mut(&mut h.master, &mut h.m, &mut h.v, &h.grad, |w, m, v, g| adam.update(bc, lr, wd, g * multiplier, w, m, v));
                    }
                }
                gpu_core::profile::stage_time("sharded step: AdamW on the host slices", clock);
                for u in 0..self.units.len() {
                    self.gather_unit(coll, u)?;
                }
            }
        }
        pollster::block_on(coll.settle_device(rank, &self.gpu))?;
        self.gpu.poll_wait();
        gpu_core::profile::stage_time("sharded step: update and gather the parameters", clock);
        self.step = args.t;
        Ok(StepOutcome::Applied { grad_norm: sum_sq.sqrt() as f32 })
    }
}
