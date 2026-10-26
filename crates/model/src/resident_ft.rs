// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of language models in bf16 and FP8 for its
// clients. If your team needs expertise in training every weight of a large model on a fixed accelerator budget,
// with the master copy and optimizer state off the device, then you can procure our services by sending an email
// to info@swedishembedded.com.

//! **Resident low-precision full fine-tuning**: the linear weights of a model live on the device in the
//! encoding they are served in (bf16 or block-scaled FP8, as [`Weight`] tiers every forward and `dX` product
//! already runs), their gradients come from [`Ops::weight_grad`], and the update runs against a
//! higher-precision MASTER copy that lives off the device together with the optimizer state.
//!
//! ```text
//!   device                                        host (or a file)
//!   Weight (bf16 / fp8 + scales)  <-- encode --   master (f32 or bf16) + moments (f32 / bf16 / int8)
//!   dW (f32 or bf16, accumulating) --- tile ---> TrainState::update_tensor
//! ```
//!
//! Per step: the backward accumulates each tensor's `dW` ([`ResidentFullFt::weight_grad`]); [`ResidentFullFt::step`]
//! multiplies by the clip and accumulation scale, updates the master tile by tile and re-encodes the stored
//! value into the device weight in place ([`crate::ops::rewrite_tile`]). The stored value is the quantization
//! of the master (the straight-through reading of quantization-aware training: the gradient taken at the
//! quantized value updates the master).
//!
//! A trainer that finishes tensors one layer at a time uses [`ResidentFullFt::layered_step`]: each
//! tensor is updated as soon as its gradient is complete, the new values wait in an
//! [`optim::deferred::DeferredPublish`] store and reach the device together once every gradient of the step
//! has been taken, so all gradients see one parameter version.
//!
//! **Layer streaming** ([`LayerWindow`]): when the weights do not all fit, only a window of layers is resident. The stored
//! value of every tensor lives on the host in its device encoding; a layer is copied into one of a bounded set of
//! device slots ([`ResidentFullFt::load_layer`]) when the forward or backward reaches it, and a layer's weights are
//! updated between the backward segments, as soon as its gradient is complete ([`ResidentFullFt::finish_layer`]).
//! The global clip norm is measured on the device. Two forms: [`ClipPass::TwoPass`] runs the backward twice (the first
//! pass only measures, so the second applies the exact global multiplier and the result equals the resident step bit
//! for bit), [`ClipPass::Running`] runs it once and clips each layer by the norm accumulated so far (never harder than
//! the global clip). No gradient outlives its layer, so a step is one micro-batch.
//!
//! **Tables** ([`ResidentRole::Table`]: the embedding table, an untied LM head) live in the same state, in bf16 with
//! the master and moments off the device like the linears, so a full fine-tune holds no fp32 `[vocab, d]` value,
//! gradient or moment on the device. They are resident for the whole step; a layer-streamed step updates them once
//! the backward is complete, by the same multiplier as the parameters outside this state.
//!
//! Nothing here names a model. A model contributes the [`ResidentSpec`] of its linears and tables and a
//! [`checkpoint::TensorSource`] of initial values; the planner refuses a configuration that does not fit
//! before the first byte is allocated ([`ResidentFullFt::demands`]).

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::io;
use std::path::PathBuf;

use checkpoint::packed::{CodecRegistry, FloatFormat, QuantEncoding, TensorDescriptor, TensorSink, Tile, TileBuf};
use checkpoint::TensorSource;
use gpu_core::{DeviceBuffer, Gpu};
use optim::checkpoint::{CheckpointDir, RunState};
use optim::codec::Rounding;
use optim::deferred::DeferredPublish;
use optim::encoded::{aligned_tile, export_values, EncodedValues, ExportReport};
use optim::rule::Rule;
use optim::Adam;
use optim::storage::{BlockStateStorage, MemBacking};
use optim::streamed::{tensor_layout, ParamTiles, StepReport};
use optim::trainstate::{TrainState, TrainStateConfig};
use paramstore::precision::{GradDtype, MasterPolicy, ParamPrecision, PrecisionPolicy, StateFormat, ValueDtype};
use gpu_core::Step;

use optim::gradnorm::{DeviceGradNorm, NormItem};
use crate::memplan::{self, Authorized, Capacity, Category, Demand, Lever, PlanError, Variant};
use crate::ops::{rewrite_tile, DeviceIo, DeviceWeightSink, Ops, Weight};

/// The resident storage format of the trained weights.
///
/// The narrow formats ([`Self::Int8`], [`Self::Nf4`]) make this quantization-aware training: the forward runs on
/// the quantized weight, the gradient taken there updates the master (the straight-through estimator), and the
/// stored weight is re-quantized from the master after every step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentValue {
    /// f32, the value its own master: what a model whose kernels read f32 weights trains with when its layers stream
    /// through a window or its state lives off the device.
    F32,
    Bf16,
    /// OCP FP8 E4M3 with one scale per `block x block` tile.
    Fp8E4M3 { block: u32 },
    /// Signed int8 with one f32 scale per 32 elements (`Weight::I8`).
    Int8,
    /// NF4 codes with one f32 scale per 32 elements (`Weight::NF4`).
    Nf4,
}

impl ResidentValue {
    pub fn encoding(self) -> QuantEncoding {
        match self {
            ResidentValue::F32 => QuantEncoding::f32(),
            ResidentValue::Bf16 => QuantEncoding::bf16(),
            ResidentValue::Fp8E4M3 { block } => QuantEncoding::fp8_e4m3_tiled(block),
            ResidentValue::Int8 => QuantEncoding::int8_blocked(32),
            ResidentValue::Nf4 => QuantEncoding::nf4_blocked(32),
        }
    }

    /// The device words a `[n, k]` weight of this format holds: payload plus, for FP8, one f32 scale per tile
    /// (the device widens bf16 scales to f32).
    pub fn device_bytes(self, n: u64, k: u64) -> u64 {
        match self {
            ResidentValue::F32 => 4 * n * k,
            ResidentValue::Bf16 => 2 * n * k,
            ResidentValue::Fp8E4M3 { block } => (n * k).div_ceil(4) * 4 + 4 * n.div_ceil(block as u64) * k.div_ceil(block as u64),
            ResidentValue::Int8 => n * k + 4 * (n * k / 32),
            ResidentValue::Nf4 => (n * k).div_ceil(8) * 4 + 4 * (n * k / 32),
        }
    }

    /// Row-length multiple a weight of this format needs.
    fn k_multiple(self) -> u32 {
        match self {
            ResidentValue::F32 => 1,
            ResidentValue::Bf16 => 2,
            ResidentValue::Fp8E4M3 { .. } => 4,
            ResidentValue::Int8 | ResidentValue::Nf4 => 32,
        }
    }
}

/// How the trained weights, their gradients, their master copy and the optimizer state are held.
#[derive(Clone, Debug)]
pub struct ResidentFtConfig {
    pub value: ResidentValue,
    /// Format of the master copy; a resident low-precision value needs one (`None` is refused: updates smaller
    /// than the value's spacing would vanish), and an f32 value is its own (anything but `None` is refused).
    pub master: MasterPolicy,
    pub state: StateFormat,
    pub rule: Rule,
    pub grad: GradDtype,
    /// Stochastic rounding of the optimizer state and master (not of the stored value, which is the nearest
    /// quantization of the master).
    pub stochastic_state: bool,
    pub seed: Option<u64>,
    /// Requested host tile, in elements; raised to what the value encoding can be written in.
    pub tile_elems: usize,
    /// Directory for the on-disk optimizer state; `None` keeps it in memory.
    pub state_dir: Option<PathBuf>,
    /// Hold the updated values in an [`optim::deferred::DeferredPublish`] store and publish them together once every
    /// tensor of the step is updated, as a trainer that updates tensors while their gradients are still being
    /// produced must. The result is the same bits; the price is one f32 copy of the resident weights in host memory.
    pub defer_publication: bool,
    /// Layer streaming: keep only a window of layers on the device and the stored values on the host. `None` keeps
    /// every layer resident.
    pub window: Option<LayerWindow>,
    /// How a layer-streamed step clips by the global gradient norm.
    pub clip_pass: ClipPass,
}

/// The layers a layer-streamed build keeps on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerWindow {
    /// Layers resident per tensor shape (at least one); two lets a loader overlap the next layer with the current.
    pub layers: u32,
}

/// How a layer-streamed step applies the global gradient-norm clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipPass {
    /// Measure the global norm in a first backward pass, update in a second: exact, one extra backward.
    TwoPass,
    /// Update in the only backward pass, clipping each layer by the norm accumulated so far: never harder than the
    /// global clip, equal to it when nothing clips.
    Running,
}

/// What a backward pass over the layers of a layer-streamed build does with each finished layer gradient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StreamPass {
    /// Measure the gradient norm on the device; no update.
    Measure,
    /// Update with a multiplier decided beforehand (the global clip of a [`StreamPass::Measure`] pass, and the
    /// accumulation scale).
    Update { lr: f32, grad_multiplier: f32 },
    /// Update with the multiplier of the norm accumulated so far: that of the layers finished and of any gradients
    /// outside this state the trainer added with [`ResidentFullFt::add_external_sum_sq`].
    Running { lr: f32, clip: Option<f32>, extra_scale: f32 },
}

/// What a finished pass measured or did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PassReport {
    /// The optimizer step the pass applied (0 for a measuring pass).
    pub step: u32,
    /// Sum of squares of the gradients this state holds: all of them after a measuring pass, those of the layers
    /// seen so far after a running one.
    pub sum_sq: f64,
    /// The smallest gradient multiplier a layer was updated with (1.0 for a measuring pass).
    pub min_multiplier: f32,
}

impl ResidentFtConfig {
    fn precision(&self) -> ParamPrecision {
        let value = if self.value == ResidentValue::F32 { ValueDtype::F32 } else { ValueDtype::Bf16 };
        ParamPrecision { value, grad: self.grad, state: self.state, master: self.master, stochastic_rounding: self.stochastic_state }
    }
}

/// One resident tensor of shape `[n, k]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentSpec {
    pub name: String,
    pub n: u32,
    pub k: u32,
    pub role: ResidentRole,
}

/// What a resident tensor is to the model, which decides its encoding and when a layer-streamed build holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentRole {
    /// A linear `y = x W^T` of one layer, held in the configured [`ResidentFtConfig::value`]. A layer-streamed build
    /// loads and updates the tensors of a layer together.
    Linear { layer: u32 },
    /// A `[rows, width]` table of the whole model: the embedding table, an untied LM head. Held in bf16 under any
    /// narrower linears' encoding (the embedding gather reads a float table, and a narrower head is the precision floor
    /// no trainer here goes below) and in f32 beside f32 linears, kept on the device for the whole step even when the layers stream, and updated once
    /// the backward is complete: in a layer-streamed step together with the parameters outside this state, by the
    /// same multiplier.
    Table,
}

impl ResidentSpec {
    pub fn linear(name: impl Into<String>, n: u32, k: u32, layer: u32) -> ResidentSpec {
        ResidentSpec { name: name.into(), n, k, role: ResidentRole::Linear { layer } }
    }

    pub fn table(name: impl Into<String>, rows: u32, width: u32) -> ResidentSpec {
        ResidentSpec { name: name.into(), n: rows, k: width, role: ResidentRole::Table }
    }

    /// The layer of a linear; `None` for a table.
    pub fn layer(&self) -> Option<u32> {
        match self.role {
            ResidentRole::Linear { layer } => Some(layer),
            ResidentRole::Table => None,
        }
    }

    pub fn is_table(&self) -> bool {
        self.role == ResidentRole::Table
    }

    /// The encoding this tensor is held in under `cfg`.
    pub fn value(&self, cfg: &ResidentFtConfig) -> ResidentValue {
        match self.role {
            ResidentRole::Linear { .. } => cfg.value,
            ResidentRole::Table if cfg.value == ResidentValue::F32 => ResidentValue::F32,
            ResidentRole::Table => ResidentValue::Bf16,
        }
    }

    fn numel(&self) -> usize {
        self.n as usize * self.k as usize
    }

    fn device_bytes(&self, cfg: &ResidentFtConfig) -> u64 {
        self.value(cfg).device_bytes(self.n as u64, self.k as u64)
    }
}

/// What the device and host offer, and what else the run already needs (activations, the other parameters).
pub struct ResidentBudget<'a> {
    pub capacity: Capacity,
    pub others: &'a [Demand],
    pub authorized: Authorized,
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

/// Which device weight slot and which gradient lane every tensor of a layer-streamed build uses.
///
/// Tensors are pooled by `(n, k, j)`, `j` counting the tensors of one shape inside a layer, so a model with
/// several equal-shaped linears per layer (key and value projections) keeps them apart. A pool owns
/// `min(window, layers holding the shape)` slots, handed out round-robin in layer order, and one gradient lane: a
/// layer's gradients are consumed before the next layer's backward starts. A table owns one slot and one lane of its
/// own: it stays resident and its gradient accumulates over the whole backward.
struct Pools {
    slot_of: Vec<usize>,
    /// The tensor whose shape each slot has.
    slot_rep: Vec<usize>,
    lane_of: Vec<usize>,
    lane_rep: Vec<usize>,
}

fn plan_pools(specs: &[ResidentSpec], window: u32) -> Pools {
    let window = window.max(1) as usize;
    let by_layer = layers_of(specs);
    struct Pool {
        lane: usize,
        slots: Vec<usize>,
        layers: usize,
    }
    let mut pools: BTreeMap<(u32, u32, usize), Pool> = BTreeMap::new();
    let mut out = Pools { slot_of: vec![0; specs.len()], slot_rep: Vec::new(), lane_of: vec![0; specs.len()], lane_rep: Vec::new() };
    for tensors in by_layer.values() {
        let mut seen: BTreeMap<(u32, u32), usize> = BTreeMap::new();
        for &i in tensors {
            let shape = (specs[i].n, specs[i].k);
            let j = seen.entry(shape).or_insert(0);
            let pool = pools.entry((shape.0, shape.1, *j)).or_insert_with(|| {
                out.lane_rep.push(i);
                Pool { lane: out.lane_rep.len() - 1, slots: Vec::new(), layers: 0 }
            });
            *j += 1;
            if pool.slots.len() < window {
                out.slot_rep.push(i);
                pool.slots.push(out.slot_rep.len() - 1);
            }
            out.slot_of[i] = pool.slots[pool.layers % pool.slots.len()];
            out.lane_of[i] = pool.lane;
            pool.layers += 1;
        }
    }
    for (i, _) in specs.iter().enumerate().filter(|(_, s)| s.is_table()) {
        out.slot_rep.push(i);
        out.slot_of[i] = out.slot_rep.len() - 1;
        out.lane_rep.push(i);
        out.lane_of[i] = out.lane_rep.len() - 1;
    }
    out
}

/// The linears of every layer, in specification order (tables belong to no layer).
fn layers_of(specs: &[ResidentSpec]) -> BTreeMap<u32, Vec<usize>> {
    let mut by_layer: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, s) in specs.iter().enumerate() {
        if let Some(layer) = s.layer() {
            by_layer.entry(layer).or_default().push(i);
        }
    }
    by_layer
}

/// The planner categories of the resident weights, gradients, master and optimizer state. The master and the
/// optimizer state are host bytes (the offloaded layout); only the weights and gradients cost device memory. A
/// layer-streamed build keeps its window of weights and one layer of gradients on the device, and the stored values
/// (in the device encoding) on the host.
pub fn demands(cfg: &ResidentFtConfig, specs: &[ResidentSpec]) -> Result<Vec<Demand>, PlanError> {
    let numel: usize = specs.iter().map(ResidentSpec::numel).sum();
    let stored: u64 = specs.iter().map(|s| s.device_bytes(cfg)).sum();
    let bytes = cfg.precision().bytes(numel, cfg.rule.slots().len());
    let (weights, grads, stored_on_host) = match cfg.window {
        None => (stored, bytes.grad, 0),
        Some(w) => {
            let pools = plan_pools(specs, w.layers);
            let resident: u64 = pools.slot_rep.iter().map(|&i| specs[i].device_bytes(cfg)).sum();
            let lanes: u64 = pools.lane_rep.iter().map(|&i| cfg.grad.bytes(specs[i].numel())).sum();
            (resident, lanes, stored)
        }
    };
    let manifest: Vec<(String, usize)> = specs.iter().map(|s| (s.name.clone(), s.n as usize * s.k as usize)).collect();
    let ladder = paramstore::precision::optimizer_state_ladder(&manifest, cfg.rule.slots().len(), 256);
    let requested = ladder.iter().position(|(f, _)| *f == cfg.state).map(|i| ladder[i].1).unwrap_or(bytes.moments);
    // An Adafactor run keeps row and column statistics (`n + k` floats a matrix) instead of elementwise moments.
    let factored: u64 = if matches!(cfg.rule, Rule::Adafactor { .. }) { specs.iter().map(|s| 4 * (s.n as u64 + s.k as u64)).sum() } else { 0 };
    let tables = if specs.iter().any(ResidentSpec::is_table) { " and bf16 tables" } else { "" };
    let label = match cfg.window {
        None => format!("resident {:?} weights{tables}", cfg.value),
        Some(w) => format!("{:?} weights of a {}-layer window{tables}", cfg.value, w.layers.max(1)),
    };
    let grad_label = match cfg.window {
        None => format!("{:?} gradients", cfg.grad),
        Some(_) => format!("{:?} gradients of one layer{tables}", cfg.grad),
    };
    Ok(vec![
        Demand::fixed(Category::Weights, label, weights),
        Demand::fixed(Category::Grads, grad_label, grads),
        Demand {
            category: Category::OptimizerState,
            variants: vec![Variant::new(Lever::Base, format!("{:?} master + {:?} state on the host", cfg.master, cfg.state), 0, bytes.master + requested + factored + stored_on_host)],
        },
    ])
}

/// Refuse, before anything is allocated, a configuration the planner cannot fit.
pub fn check_fits(cfg: &ResidentFtConfig, specs: &[ResidentSpec], budget: &ResidentBudget<'_>) -> Result<memplan::PlanReport, String> {
    let mut all = demands(cfg, specs).map_err(|e| e.to_string())?;
    all.extend_from_slice(budget.others);
    memplan::plan(&all, budget.capacity, &budget.authorized).map_err(|e| format!("resident full fine-tune does not fit: {e}"))
}

/// Elements of a bf16 gradient unpacked per parallel work item.
const GRAD_UNPACK_CHUNK: usize = 1 << 16;

/// Initial values read from a [`TensorSource`] as the master's and the first stored value's source: each tile is
/// decoded from exactly the elements it covers ([`TensorSource::with_tensor_range`]), so the host holds the tile
/// being read and nothing else, and every element of the source is decoded once per pass over it.
struct SourceTiles<'a> {
    src: &'a dyn TensorSource,
    names: Vec<&'a str>,
    /// Added to the tensor index the caller passes (an export of one tensor starts at 0).
    base: usize,
}

impl<'a> SourceTiles<'a> {
    fn new(src: &'a dyn TensorSource, names: Vec<&'a str>, base: usize) -> Self {
        SourceTiles { src, names, base }
    }
}

impl ParamTiles for SourceTiles<'_> {
    fn read_value(&mut self, tensor: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        let name = self.names[self.base + tensor];
        if !checkpoint::read_range_into(self.src, name, start, out) {
            return Err(invalid(format!("initial values: tensor '{name}' range {start}..{} is not in the source", start + out.len())));
        }
        Ok(())
    }
    fn write_value(&mut self, tensor: usize, _: usize, _: &[f32], _: Rounding) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::Unsupported, format!("tensor {tensor}: the initial values are read-only")))
    }
    fn read_grad(&mut self, tensor: usize, _: usize, _: &mut [f32]) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::Unsupported, format!("tensor {tensor}: the initial values carry no gradient")))
    }
}

/// [`SourceTiles`] for a tensor upload that also seeds the tensor's master copy from every tile it reads.
struct SeedingTiles<'a, 's> {
    source: SourceTiles<'a>,
    seed: Option<&'s mut TrainState>,
    /// The tensor's index in the optimizer state.
    tensor: usize,
}

impl ParamTiles for SeedingTiles<'_, '_> {
    fn read_value(&mut self, _: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        self.source.read_value(0, start, out)?;
        match &mut self.seed {
            Some(state) => state.init_master_tile(self.tensor, start, out),
            None => Ok(()),
        }
    }
    fn write_value(&mut self, tensor: usize, start: usize, vals: &[f32], rnd: Rounding) -> io::Result<()> {
        self.source.write_value(tensor, start, vals, rnd)
    }
    fn read_grad(&mut self, tensor: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        self.source.read_grad(tensor, start, out)
    }
}

/// Where the host time of the update went, accumulated over the steps of a run: reading gradients off the device, encoding
/// the new stored values, writing them to the device, and the whole update (which contains the rest: the master and state
/// tiles and the arithmetic). The profile a run is judged against the transfer ceiling by.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostTimes {
    pub grad_read: std::time::Duration,
    pub value_encode: std::time::Duration,
    pub value_write: std::time::Duration,
    pub update: std::time::Duration,
    pub steps: u32,
}

impl HostTimes {
    fn add(&mut self, o: &HostTimes) {
        self.grad_read += o.grad_read;
        self.value_encode += o.value_encode;
        self.value_write += o.value_write;
        self.update += o.update;
        self.steps += o.steps;
    }
}

/// The device end of [`ParamTiles`]: values decode from (and re-encode into) the resident weights, gradients
/// read from the accumulating device buffers.
struct DeviceTiles<'a> {
    ops: &'a dyn DeviceIo,
    /// The stored values of a layer-streamed build, on the host; `None` reads and writes the resident weights.
    host: Option<HostValues<'a>>,
    weights: &'a [Weight],
    grads: &'a [DeviceBuffer],
    descs: &'a [TensorDescriptor],
    registry: &'a CodecRegistry,
    grad_dtype: GradDtype,
    /// The words of a bf16 gradient tile as read off the device, before they are unpacked: warm scratch reused by
    /// every tile.
    grad_words: Vec<f32>,
    /// The stored value of the last tensor decoded on the host (read whole: only an export, a commit or a tensor
    /// without a master reads values off the device).
    value_cache: Option<(usize, Vec<u8>, Vec<Vec<u8>>)>,
    enc: TileBuf,
    times: HostTimes,
}

/// The host copy of the stored values of a layer-streamed build, with the version of each tensor (bumped by every
/// write, so a device slot knows whether it still holds the current value).
struct HostValues<'a> {
    store: &'a mut EncodedValues<MemBacking>,
    versions: &'a mut [u64],
}

fn buffers(w: &Weight) -> (&DeviceBuffer, Option<&DeviceBuffer>) {
    match w {
        Weight::F32 { w, .. } | Weight::BF16 { w, .. } | Weight::F16 { w, .. } => (w, None),
        Weight::I8 { w, s, .. } | Weight::NF4 { w, s, .. } | Weight::FP8 { w, s, .. } | Weight::Q4 { w, s, .. } | Weight::Micro { w, s, .. } => (w, Some(s)),
        Weight::KQuant { w, .. } => (w, None),
    }
}

impl DeviceTiles<'_> {
    fn gpu(&self) -> &Gpu {
        self.ops.gpu()
    }

    /// Payload bytes and scale sidecar bytes of tensor `ti`, as the descriptor's codec stores them.
    fn load_value(&mut self, ti: usize) -> &(usize, Vec<u8>, Vec<Vec<u8>>) {
        if self.value_cache.as_ref().map(|c| c.0) != Some(ti) {
            let d = &self.descs[ti];
            let (payload, scales) = buffers(&self.weights[ti]);
            let words = self.gpu().read(payload, (d.payload_bytes() as usize).div_ceil(4));
            let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_bits().to_le_bytes()).collect();
            bytes.truncate(d.payload_bytes() as usize);
            let sidecars = match scales {
                Some(buf) => {
                    let level = &d.encoding.scales[0];
                    let floats = self.gpu().read(buf, d.scale_numel(0) as usize);
                    let bytes = if level.format == FloatFormat::Bf16 {
                        floats.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect()
                    } else {
                        floats.iter().flat_map(|v| v.to_le_bytes()).collect()
                    };
                    vec![bytes]
                }
                None => Vec::new(),
            };
            self.value_cache = Some((ti, bytes, sidecars));
        }
        self.value_cache.as_ref().expect("filled above")
    }
}

impl ParamTiles for DeviceTiles<'_> {
    fn read_value(&mut self, ti: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        if let Some(host) = &mut self.host {
            return host.store.read(ti, start, out);
        }
        let d = self.descs[ti].clone();
        let e = start as u64..(start + out.len()) as u64;
        let pr = d.payload_range(&e);
        let registry = self.registry;
        let (_, payload, scales) = self.load_value(ti);
        let slices: Vec<&[u8]> = scales
            .iter()
            .enumerate()
            .map(|(l, s)| {
                let w = d.encoding.scales[l].format.bits() as usize / 8;
                let r = d.scale_range(l, &e);
                &s[r.start as usize * w..r.end as usize * w]
            })
            .collect();
        let tile = Tile { elems: e, payload: &payload[pr.start as usize..pr.end as usize], scales: slices };
        let mut dec = Vec::with_capacity(out.len());
        registry.decoder(&d.encoding).map_err(invalid)?.decode_tile(&d, &tile, &mut dec).map_err(invalid)?;
        out.copy_from_slice(&dec);
        Ok(())
    }

    fn write_value(&mut self, ti: usize, start: usize, vals: &[f32], _rnd: Rounding) -> io::Result<()> {
        // The stored value is the nearest quantization of the master; the optimizer's stochastic request is for
        // the state it keeps beside it.
        if let Some(host) = &mut self.host {
            let t0 = std::time::Instant::now();
            host.store.write(ti, start, vals)?;
            host.versions[ti] += 1;
            self.times.value_encode += t0.elapsed();
            return Ok(());
        }
        let d = &self.descs[ti];
        let e = start as u64..(start + vals.len()) as u64;
        let t0 = std::time::Instant::now();
        self.registry.encoder(&d.encoding).map_err(invalid)?.encode_tile(d, &e, vals, &checkpoint::packed::EncodeCtx::default(), &mut self.enc).map_err(invalid)?;
        let t1 = std::time::Instant::now();
        let tile = Tile { elems: e, payload: &self.enc.payload, scales: self.enc.scales.iter().map(|s| s.as_slice()).collect() };
        rewrite_tile(self.ops, &self.weights[ti], d, &tile).map_err(invalid)?;
        self.times.value_encode += t1 - t0;
        self.times.value_write += t1.elapsed();
        if self.value_cache.as_ref().is_some_and(|c| c.0 == ti) {
            self.value_cache = None;
        }
        Ok(())
    }

    fn read_grad(&mut self, ti: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        let t0 = std::time::Instant::now();
        match self.grad_dtype {
            GradDtype::F32 => self.ops.read_range_into(&self.grads[ti], start as u64, out),
            GradDtype::Bf16 => {
                // Two bf16 values to a word: a tile starts on a word (the tiles are even) and ends on one (every
                // tensor has an even element count).
                debug_assert!(start.is_multiple_of(2) && out.len().is_multiple_of(2), "a bf16 gradient tile is whole words");
                self.grad_words.resize(out.len() / 2, 0.0);
                self.ops.read_range_into(&self.grads[ti], (start / 2) as u64, &mut self.grad_words);
                let words = &self.grad_words;
                backend_cpu::par::chunks_mut(out, GRAD_UNPACK_CHUNK, |c, dst| {
                    for (pair, word) in dst.as_chunks_mut::<2>().0.iter_mut().zip(&words[c * GRAD_UNPACK_CHUNK / 2..]) {
                        let bits = word.to_bits();
                        *pair = [f32::from_bits(bits << 16), f32::from_bits(bits & 0xffff_0000)];
                    }
                });
            }
        }
        self.times.grad_read += t0.elapsed();
        Ok(())
    }

    fn write_grad(&mut self, ti: usize, _: usize, _: &[f32]) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::Unsupported, format!("tensor {ti}: gradients are produced on the device by Ops::weight_grad")))
    }
}

/// The [`ParamTiles`] of the build `$self` is: stored values from the host copy when layer-streamed, from the resident
/// weights otherwise. A macro so the borrow covers only the fields it names (the optimizer state stays free).
macro_rules! tiles {
    ($self:ident, $ops:expr) => {
        DeviceTiles {
            ops: $ops,
            host: $self.window.as_mut().map(|w| HostValues { store: &mut w.host, versions: &mut w.versions }),
            weights: &$self.weights,
            grads: &$self.grads,
            descs: &$self.descs,
            registry: &$self.registry,
            grad_dtype: $self.cfg.grad,
            grad_words: Vec::new(),
            value_cache: None,
            enc: TileBuf::default(),
            times: HostTimes::default(),
        }
    };
}

/// What an export of a resident build wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentExport {
    /// The streamed linears: tensors, payload bytes and the host staging the export used.
    pub linears: ExportReport,
    /// Bytes of the whole file's tensors (linears in the chosen encoding, the remainder in fp32).
    pub bytes: u64,
}

/// The trainable linears and tables of a model, resident on the device in a low-precision encoding with a master copy
/// and optimizer state off it.
///
/// **Ordering.** This state reads gradients and rewrites weights through the `Ops` handle every method takes. A model
/// that records its forward and backward on another handle of the same device must send that handle's recorded work
/// to the device (`Gpu::flush`) before each call that touches device data: a handle may hold recorded work back until
/// it flushes, so a gradient read could otherwise run before the backward producing it, and a rewritten weight could
/// overtake a forward still waiting to read the old one. [`adamw_step`] does it through
/// [`ResidentBackward::flush_backward`]; a model calling in directly (loading a layer, reading a master) flushes first.
pub struct ResidentFullFt {
    cfg: ResidentFtConfig,
    specs: Vec<ResidentSpec>,
    descs: Vec<TensorDescriptor>,
    /// Tensor `i`'s device weight: its own when resident, a handle to its slot's weight when layer-streamed.
    weights: Vec<Weight>,
    /// Tensor `i`'s gradient buffer: its own when resident, its lane's when layer-streamed.
    grads: Vec<DeviceBuffer>,
    state: TrainState,
    registry: CodecRegistry,
    norm: DeviceGradNorm,
    tile: usize,
    times: HostTimes,
    /// Device bytes of the weights this build holds (every layer, or the window's slots).
    weight_bytes: u64,
    window: Option<WindowState>,
}

/// What a layer-streamed build holds besides the shared fields: the host copy of the stored values, the device slots
/// and gradient lanes, and the backward pass in progress.
struct WindowState {
    host: EncodedValues<MemBacking>,
    /// Bumped by every write of a tensor's stored value; a slot holding an older version is stale.
    versions: Vec<u64>,
    by_layer: BTreeMap<u32, Vec<usize>>,
    slot_of: Vec<usize>,
    slot_weights: Vec<Weight>,
    /// The `(tensor, version)` each slot's device weight holds.
    slot_holds: Vec<Option<(usize, u64)>>,
    lane_buffers: Vec<DeviceBuffer>,
    pass: Option<PassState>,
}

struct PassState {
    pass: StreamPass,
    /// Sum of squares of each tensor's gradient, once measured.
    tensor_sum: Vec<Option<f64>>,
    updated: Vec<bool>,
    running: f64,
    min_multiplier: f32,
    step: u32,
}

impl ResidentFullFt {
    /// Check the plan, then upload every weight from `src` in the resident encoding (one tile of host staging
    /// per tensor at a time) and seed the master copies from the same f32 values. Nothing is allocated when the
    /// planner refuses.
    pub fn new(ops: &dyn DeviceIo, cfg: ResidentFtConfig, specs: Vec<ResidentSpec>, src: &dyn TensorSource, budget: &ResidentBudget<'_>) -> Result<ResidentFullFt, String> {
        match (cfg.value, cfg.master) {
            (ResidentValue::F32, MasterPolicy::None) => {}
            (ResidentValue::F32, master) => return Err(format!("a resident f32 weight is its own master; a {master:?} master copy would only duplicate it")),
            (_, MasterPolicy::None) => {
                return Err("a resident bf16/fp8 weight needs a master copy (master: f32 or bf16); without one, updates below the value's spacing are lost".into())
            }
            _ => {}
        }
        for s in &specs {
            let value = s.value(&cfg);
            if !s.k.is_multiple_of(value.k_multiple()) {
                return Err(format!("{}: K = {} must be a multiple of {} for {:?} weights", s.name, s.k, value.k_multiple(), value));
            }
            if cfg.grad == GradDtype::Bf16 && !s.k.is_multiple_of(2) {
                return Err(format!("{}: a bf16 gradient needs an even K (got {})", s.name, s.k));
            }
            // Every kernel binds a tensor's weight and its gradient as one buffer each.
            let binding = ops.gpu().max_storage_binding_bytes();
            let (value_bytes, grad_bytes) = (s.device_bytes(&cfg), cfg.grad.bytes(s.numel()));
            if value_bytes.max(grad_bytes) > binding {
                return Err(format!(
                    "{}: its {value_bytes}-byte {value:?} value or {grad_bytes}-byte {:?} gradient exceeds the device's {binding}-byte storage binding{}",
                    s.name,
                    cfg.grad,
                    if cfg.grad == GradDtype::F32 { " (a bf16 gradient halves it)" } else { "" }
                ));
            }
        }
        check_fits(&cfg, &specs, budget)?;
        let descs: Vec<TensorDescriptor> = specs.iter().map(|s| TensorDescriptor::new(s.name.clone(), vec![s.n as u64, s.k as u64], s.value(&cfg).encoding())).collect();
        let registry = CodecRegistry::reference();
        for d in &descs {
            d.validate()?;
        }
        // A tile every tensor can be written in: whole scale bands for FP8, whole payload words for all.
        let tile = descs.iter().fold(aligned_tile(&descs, cfg.tile_elems.max(1)) as u64, |t, _| t);
        let tile = lcm(tile, 4) as usize;
        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        let mut sink = DeviceWeightSink::new(ops);
        // Each tile of the initial values is decoded from the source once per use: `seed` additionally feeds it to the
        // master copy, so the pass that uploads a tensor is the pass that seeds its master.
        let mut upload = |ti: usize, seed: Option<&mut TrainState>| -> Result<Weight, String> {
            let d = &descs[ti];
            let mut init = SeedingTiles { source: SourceTiles::new(src, names.clone(), ti), seed, tensor: ti };
            export_values(&mut init, std::slice::from_ref(d), &registry, &mut sink, tile).map_err(|e| e.to_string())?;
            sink.take_weight().ok_or_else(|| format!("{}: the upload produced no weight", d.name))
        };
        let manifest: Vec<(String, usize)> = specs.iter().map(|s| (s.name.clone(), s.n as usize * s.k as usize)).collect();
        let state_clock = std::time::Instant::now();
        let mut state = TrainState::create(TrainStateConfig {
            manifest,
            policy: PrecisionPolicy::uniform(cfg.precision()),
            rule: cfg.rule,
            seed: cfg.seed,
            tile_elems: tile,
            state_dir: cfg.state_dir.clone(),
            row_len: specs.iter().map(|s| (s.name.clone(), s.k as usize)).collect(),
        })
        .map_err(|e| e.to_string())?;
        gpu_core::profile::stage_time("resident build: allocate and fault in the master and optimizer state", state_clock);
        let upload_clock = std::time::Instant::now();
        let gpu = ops.gpu();
        let grad_words = |s: &ResidentSpec| {
            let elems = s.n as u64 * s.k as u64;
            if cfg.grad == GradDtype::Bf16 {
                elems / 2
            } else {
                elems
            }
        };
        let norm = DeviceGradNorm::new(gpu)?;
        let (weights, grads, window, weight_bytes) = match cfg.window {
            None => {
                let weights = (0..specs.len()).map(|ti| upload(ti, Some(&mut state))).collect::<Result<Vec<_>, _>>()?;
                let grads = specs.iter().map(|s| gpu.storage(grad_words(s))).collect();
                let bytes = specs.iter().map(|s| s.device_bytes(&cfg)).sum();
                (weights, grads, None, bytes)
            }
            Some(w) => {
                let pools = plan_pools(&specs, w.layers);
                let mut host = EncodedValues::new(descs.clone(), CodecRegistry::reference(), MemBacking::zeroed(EncodedValues::<MemBacking>::required_bytes(&descs)), tile).map_err(|e| e.to_string())?;
                let mut initial = SourceTiles::new(src, names.clone(), 0);
                let mut values = vec![0f32; tile];
                for (ti, d) in descs.iter().enumerate() {
                    let numel = d.numel() as usize;
                    for start in (0..numel).step_by(tile) {
                        let n = tile.min(numel - start);
                        initial.read_value(ti, start, &mut values[..n]).map_err(|e| e.to_string())?;
                        state.init_master_tile(ti, start, &values[..n]).map_err(|e| e.to_string())?;
                        host.write(ti, start, &values[..n]).map_err(|e| e.to_string())?;
                    }
                }
                let slot_weights = pools.slot_rep.iter().map(|&ti| upload(ti, None)).collect::<Result<Vec<_>, _>>()?;
                let lane_buffers: Vec<DeviceBuffer> = pools.lane_rep.iter().map(|&ti| gpu.storage(grad_words(&specs[ti]))).collect();
                let weights = pools.slot_of.iter().map(|&s| slot_weights[s].clone()).collect();
                let grads = pools.lane_of.iter().map(|&l| lane_buffers[l].clone()).collect();
                let bytes = pools.slot_rep.iter().map(|&ti| specs[ti].device_bytes(&cfg)).sum();
                // The tables were uploaded into their own slots above: those hold the current values.
                let slot_holds = (0..slot_weights.len()).map(|slot| Some(pools.slot_rep[slot]).filter(|&ti| specs[ti].is_table()).map(|ti| (ti, 0))).collect();
                let window = WindowState {
                    host,
                    versions: vec![0; specs.len()],
                    by_layer: layers_of(&specs),
                    slot_of: pools.slot_of,
                    slot_holds,
                    slot_weights,
                    lane_buffers,
                    pass: None,
                };
                (weights, grads, Some(window), bytes)
            }
        };
        gpu_core::profile::stage_time("resident build: read the base once, seed the master, encode and upload the weights", upload_clock);
        Ok(ResidentFullFt { cfg, specs, descs, weights, grads, state, registry, norm, tile, times: HostTimes::default(), weight_bytes, window })
    }

    pub fn specs(&self) -> &[ResidentSpec] {
        &self.specs
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.specs.iter().position(|s| s.name == name)
    }

    /// The resident weight of tensor `i`, for the forward and `dX` products.
    pub fn weight(&self, i: usize) -> &Weight {
        &self.weights[i]
    }

    /// The packed bf16 words of table `i` (two values a word, low half first), for a model that reads its tables as
    /// raw bf16 rows.
    pub fn table_words(&self, i: usize) -> &DeviceBuffer {
        match &self.weights[i] {
            Weight::BF16 { w, .. } if self.specs[i].is_table() => w,
            other => panic!("{}: not a table (a {:?} weight)", self.specs[i].name, other.dtype()),
        }
    }

    pub fn grad_buffer(&self, i: usize) -> &DeviceBuffer {
        &self.grads[i]
    }

    pub fn grad_dtype(&self) -> GradDtype {
        self.cfg.grad
    }

    /// The host tile (elements) every update and export of this state runs in.
    pub fn tile_elems(&self) -> usize {
        self.tile
    }

    /// Record tensor `i`'s weight gradient `dW += dY^T X` for `m` rows into its accumulating buffer.
    pub fn weight_grad(&self, ops: &Ops, s: &mut Vec<Step>, i: usize, x: &DeviceBuffer, dy: &DeviceBuffer, m: u32) {
        ops.weight_grad(s, &self.weights[i], x, dy, m, &self.grads[i], self.cfg.grad);
    }

    /// Zero every gradient: call once before the accumulating backward passes of a step.
    pub fn zero_grads(&self, ops: &dyn DeviceIo) {
        let clears: Vec<&DeviceBuffer> = match &self.window {
            Some(w) => w.lane_buffers.iter().collect(),
            None => self.grads.iter().collect(),
        };
        ops.gpu().submit(&clears, &[]);
        // The backward that accumulates next may be recorded on another handle: the clear must reach the device first.
        ops.gpu().flush();
    }

    /// Device bytes of the weights this build holds: every layer's, or the window's slots.
    pub fn device_weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// Whether this build streams its layers through a bounded window.
    pub fn is_layer_streamed(&self) -> bool {
        self.window.is_some()
    }

    /// Per-tensor precision choices and the physical byte totals.
    pub fn report(&self) -> String {
        self.state.report()
    }

    pub fn step_count(&self) -> u32 {
        self.state.step_count()
    }

    pub fn seed(&self) -> u64 {
        self.state.seed()
    }

    /// Sum of squares of every gradient, reduced on the device (for the global clip of [`optim::grad_multiplier`]).
    /// Summed per tensor in tensor order, the order a layer-streamed measuring pass reports in.
    pub fn grad_sum_sq(&mut self, ops: &dyn DeviceIo) -> io::Result<f64> {
        self.require_resident("grad_sum_sq")?;
        let items: Vec<NormItem<'_>> = self.grads.iter().zip(&self.specs).map(|(buf, s)| NormItem { buf, numel: s.n as usize * s.k as usize, dtype: self.cfg.grad }).collect();
        let t0 = std::time::Instant::now();
        let sum = self.norm.sum_sq_each(ops.gpu(), &items).iter().sum();
        self.times.grad_read += t0.elapsed();
        Ok(sum)
    }

    fn require_resident(&self, what: &str) -> io::Result<()> {
        if self.window.is_some() {
            return Err(invalid(format!("{what}: a layer-streamed build holds no whole-model gradient; use the layer pass API (begin_pass, finish_layer, end_pass)")));
        }
        Ok(())
    }

    /// Host time spent in the updates of this run so far, by phase.
    pub fn host_times(&self) -> HostTimes {
        self.times
    }

    /// One update step over every tensor: the gradients are multiplied by [`optim::grad_multiplier`]
    /// (`extra_scale`, and the global clip when `clip` is set), the master and state are updated tile by tile
    /// and the device weights re-encoded in place. The gradients are zeroed afterwards.
    pub fn step(&mut self, ops: &dyn DeviceIo, lr: f32, clip: Option<f32>, extra_scale: f32) -> io::Result<StepReport> {
        self.require_resident("step")?;
        let mut tiles = tiles!(self, ops);
        let report = self.state.step(&mut tiles, lr, clip, extra_scale)?;
        self.zero_grads(ops);
        Ok(report)
    }

    /// [`Self::step`] with the gradient multiplier already decided, for a trainer whose global clip spans
    /// parameters this state does not hold (it adds its own [`Self::grad_sum_sq`] to theirs).
    pub fn step_scaled(&mut self, ops: &dyn DeviceIo, lr: f32, grad_multiplier: f32) -> io::Result<StepReport> {
        self.require_resident("step_scaled")?;
        if self.cfg.defer_publication {
            // Backward order: the last tensor's gradient completes first.
            let n = self.specs.len();
            self.layered_step(ops, lr, grad_multiplier, |step| (0..n).rev().try_for_each(|i| step.update(i)))?;
            return Ok(StepReport { step: self.state.step_count(), grad_multiplier });
        }
        let mut tiles = tiles!(self, ops);
        let t0 = std::time::Instant::now();
        let step = self.state.begin_step();
        for i in 0..self.specs.len() {
            self.state.update_tensor(&mut tiles, i, lr, grad_multiplier)?;
        }
        tiles.times.update += t0.elapsed();
        tiles.times.steps += 1;
        if gpu_core::profile::enabled() {
            let t = &tiles.times;
            eprintln!(
                "stage resident update: {:.0} ms total = grad read {:.0} + value encode {:.0} + value write {:.0} + master/state/arithmetic {:.0}",
                t.update.as_secs_f64() * 1e3,
                t.grad_read.as_secs_f64() * 1e3,
                t.value_encode.as_secs_f64() * 1e3,
                t.value_write.as_secs_f64() * 1e3,
                (t.update.saturating_sub(t.grad_read + t.value_encode + t.value_write)).as_secs_f64() * 1e3,
            );
        }
        if gpu_core::profile::enabled() {
            eprintln!("stage resident update phases: {:?}", self.state.take_phase_times());
        }
        self.times.add(&tiles.times);
        self.zero_grads(ops);
        Ok(StepReport { step, grad_multiplier })
    }

    // ---- layer streaming -------------------------------------------------------------------------------------

    /// The tensors of `layer`, in specification order.
    pub fn layer_tensors(&self, layer: usize) -> &[usize] {
        self.window.as_ref().and_then(|w| w.by_layer.get(&(layer as u32))).map_or(&[], Vec::as_slice)
    }

    /// Copy `layer`'s stored values into the device slots its tensors use (a no-op where a slot already holds the
    /// current value, and for a build with every layer resident). Call before the forward or the backward reaches the
    /// layer.
    pub fn load_layer(&mut self, ops: &dyn DeviceIo, layer: usize) -> io::Result<()> {
        let Some(tensors) = self.window.as_ref().and_then(|w| w.by_layer.get(&(layer as u32)).cloned()) else { return Ok(()) };
        self.load_tensors(ops, &tensors)
    }

    /// Copy the stored values of `tensors` from the host into their device slots, where a slot does not already hold
    /// the current value.
    fn load_tensors(&mut self, ops: &dyn DeviceIo, tensors: &[usize]) -> io::Result<()> {
        let Some(w) = &mut self.window else { return Ok(()) };
        let mut waited = false;
        let mut buf = TileBuf::default();
        for &ti in tensors {
            let slot = w.slot_of[ti];
            if w.slot_holds[slot] == Some((ti, w.versions[ti])) {
                continue;
            }
            if !waited {
                // Work queued by the layer that used the slot last still reads it.
                ops.gpu().poll_wait();
                waited = true;
            }
            let d = &self.descs[ti];
            let numel = d.numel() as usize;
            let t0 = std::time::Instant::now();
            for start in (0..numel).step_by(self.tile) {
                let n = self.tile.min(numel - start);
                w.host.read_encoded(ti, start, n, &mut buf)?;
                let tile = Tile { elems: start as u64..(start + n) as u64, payload: &buf.payload, scales: buf.scales.iter().map(|s| s.as_slice()).collect() };
                rewrite_tile(ops, &w.slot_weights[slot], d, &tile).map_err(invalid)?;
            }
            self.times.value_write += t0.elapsed();
            w.slot_holds[slot] = Some((ti, w.versions[ti]));
        }
        Ok(())
    }

    /// [`Self::load_layer`] for the backward of `layer`, and clear the gradient lanes its tensors accumulate into.
    pub fn begin_layer_backward(&mut self, ops: &dyn DeviceIo, layer: usize) -> io::Result<()> {
        self.load_layer(ops, layer)?;
        if let Some(w) = &self.window {
            let lanes: Vec<&DeviceBuffer> = w.by_layer.get(&(layer as u32)).into_iter().flatten().map(|&ti| &self.grads[ti]).collect();
            ops.gpu().submit(&lanes, &[]);
            ops.gpu().flush();
        }
        Ok(())
    }

    /// Open a backward pass over the layers (see [`StreamPass`]). An updating pass opens the next optimizer step.
    pub fn begin_pass(&mut self, pass: StreamPass) {
        let n = self.specs.len();
        let step = match pass {
            StreamPass::Measure => 0,
            StreamPass::Update { .. } | StreamPass::Running { .. } => self.state.begin_step(),
        };
        if let Some(w) = &mut self.window {
            w.pass = Some(PassState { pass, tensor_sum: vec![None; n], updated: vec![false; n], running: 0.0, min_multiplier: f32::INFINITY, step });
        }
    }

    /// Add `sum_sq`, the squared norm of gradients this state does not hold that are already complete (a trainer's
    /// head), to the running norm of the open pass; layers finished after this clip by a norm that includes it.
    pub fn add_external_sum_sq(&mut self, sum_sq: f64) {
        if let Some(pass) = self.window.as_mut().and_then(|w| w.pass.as_mut()) {
            pass.running += sum_sq;
        }
    }

    /// How a layer-streamed step clips by the global norm.
    pub fn clip_pass(&self) -> ClipPass {
        self.cfg.clip_pass
    }

    /// The device gradient-norm reduction of this build, for a trainer measuring gradients it holds itself.
    pub fn device_norm(&self) -> &DeviceGradNorm {
        &self.norm
    }

    /// `layer`'s gradients are complete (its backward has been submitted): measure them on the device, and in an
    /// updating pass apply the update right away. The layer's new stored values are visible to the next load of it;
    /// no later layer of this pass reads them. With `defer_publication` the layer's new values wait in a
    /// [`DeferredPublish`] store and reach the host copy together, so a layer is never half updated.
    pub fn finish_layer(&mut self, ops: &dyn DeviceIo, layer: usize) -> io::Result<()> {
        let w = self.window.as_mut().ok_or_else(|| invalid("finish_layer: this build keeps every layer resident".into()))?;
        let mut pass = w.pass.take().ok_or_else(|| invalid("finish_layer: no pass is open (call begin_pass)".into()))?;
        let tensors = w.by_layer.get(&(layer as u32)).cloned().unwrap_or_default();
        let result = self.finish_layer_in(ops, &mut pass, &tensors);
        if let Some(w) = &mut self.window {
            w.pass = Some(pass);
        }
        result
    }

    fn finish_layer_in(&mut self, ops: &dyn DeviceIo, pass: &mut PassState, tensors: &[usize]) -> io::Result<()> {
        let items: Vec<NormItem<'_>> = tensors.iter().map(|&ti| NormItem { buf: &self.grads[ti], numel: self.specs[ti].n as usize * self.specs[ti].k as usize, dtype: self.cfg.grad }).collect();
        let t0 = std::time::Instant::now();
        let sums = self.norm.sum_sq_each(ops.gpu(), &items);
        self.times.grad_read += t0.elapsed();
        for (&ti, &sum) in tensors.iter().zip(&sums) {
            pass.tensor_sum[ti] = Some(sum);
            pass.running += sum;
        }
        let (lr, multiplier) = match pass.pass {
            StreamPass::Measure => return Ok(()),
            StreamPass::Update { lr, grad_multiplier } => (lr, grad_multiplier),
            StreamPass::Running { lr, clip, extra_scale, .. } => (lr, optim::grad_multiplier(pass.running, clip, extra_scale)),
        };
        pass.min_multiplier = pass.min_multiplier.min(multiplier);
        let t0 = std::time::Instant::now();
        let mut tiles = tiles!(self, ops);
        if self.cfg.defer_publication {
            let numels: Vec<usize> = (0..self.specs.len()).map(|ti| if tensors.contains(&ti) { self.specs[ti].n as usize * self.specs[ti].k as usize } else { 0 }).collect();
            let layout = tensor_layout(&numels, StateFormat::F32);
            let pending = BlockStateStorage::new(layout.clone(), MemBacking::zeroed(layout.total_bytes()), self.tile).map_err(|e| invalid(e.to_string()))?;
            let mut params = DeferredPublish::new(&mut tiles, pending, numels, self.tile);
            for &ti in tensors {
                self.state.update_tensor(&mut params, ti, lr, multiplier)?;
            }
            params.publish()?;
        } else {
            for &ti in tensors {
                self.state.update_tensor(&mut tiles, ti, lr, multiplier)?;
            }
        }
        tiles.times.update += t0.elapsed();
        self.times.add(&tiles.times);
        for &ti in tensors {
            pass.updated[ti] = true;
        }
        Ok(())
    }

    /// Close the pass opened by [`Self::begin_pass`]: every tensor of the model must have been finished.
    pub fn end_pass(&mut self) -> io::Result<PassReport> {
        let w = self.window.as_mut().ok_or_else(|| invalid("end_pass: this build keeps every layer resident".into()))?;
        let pass = w.pass.take().ok_or_else(|| invalid("end_pass: no pass is open".into()))?;
        let updating = !matches!(pass.pass, StreamPass::Measure);
        let missing = |i: usize| invalid(format!("end_pass: tensor '{}' (layer {:?}) was not finished before the pass ended", self.specs[i].name, self.specs[i].layer()));
        let linear = |i: &usize| !self.specs[*i].is_table();
        if let Some(i) = (0..self.specs.len()).filter(linear).find(|&i| pass.tensor_sum[i].is_none()) {
            return Err(missing(i));
        }
        if let Some(i) = (0..self.specs.len()).filter(linear).find(|&i| updating && !pass.updated[i]) {
            return Err(missing(i));
        }
        if updating {
            self.times.steps += 1;
        }
        let sum_sq = match pass.pass {
            StreamPass::Running { .. } => pass.running,
            _ => pass.tensor_sum.iter().flatten().sum(),
        };
        Ok(PassReport { step: pass.step, sum_sq, min_multiplier: if updating { pass.min_multiplier } else { 1.0 } })
    }

    /// The tensors of this build that are tables ([`ResidentRole::Table`]).
    pub fn tables(&self) -> Vec<usize> {
        (0..self.specs.len()).filter(|&i| self.specs[i].is_table()).collect()
    }

    /// Sum of squares of the tables' gradients, reduced on the device. A layer-streamed pass covers only the layers:
    /// its trainer counts the tables with the parameters it holds itself.
    pub fn table_grad_sum_sq(&mut self, ops: &dyn DeviceIo) -> f64 {
        let tables = self.tables();
        let items: Vec<NormItem<'_>> = tables.iter().map(|&ti| NormItem { buf: &self.grads[ti], numel: self.specs[ti].numel(), dtype: self.cfg.grad }).collect();
        let t0 = std::time::Instant::now();
        let sum = self.norm.sum_sq_each(ops.gpu(), &items).iter().sum();
        self.times.grad_read += t0.elapsed();
        sum
    }

    /// Update every table of a layer-streamed build from its complete gradient, multiplied by `grad_multiplier` (the
    /// clip and the accumulation scale, decided once the backward is done), in the step the latest pass opened, and
    /// put the new values on the device. A resident build updates its tables with every other tensor in
    /// [`Self::step_scaled`].
    pub fn update_tables(&mut self, ops: &dyn DeviceIo, lr: f32, grad_multiplier: f32) -> io::Result<()> {
        if self.window.is_none() {
            return Err(invalid("update_tables: a resident build updates its tables in step_scaled".into()));
        }
        let tables = self.tables();
        let t0 = std::time::Instant::now();
        let mut tiles = tiles!(self, ops);
        for &ti in &tables {
            self.state.update_tensor(&mut tiles, ti, lr, grad_multiplier)?;
        }
        tiles.times.update += t0.elapsed();
        self.times.add(&tiles.times);
        self.load_tensors(ops, &tables)
    }

    /// The update rule in force.
    pub fn rule(&self) -> Rule {
        self.cfg.rule
    }

    /// Use `rule` from the next step on (same state slots as the configured rule).
    pub fn set_rule(&mut self, rule: Rule) {
        self.cfg.rule = rule;
        self.state.set_rule(rule);
    }

    /// The master copy of tensor `i` (the stored value, decoded, when it has none).
    pub fn read_master(&mut self, ops: &dyn DeviceIo, i: usize) -> io::Result<Vec<f32>> {
        let numel = self.descs[i].numel() as usize;
        let mut out = vec![0f32; numel];
        let mut tiles = tiles!(self, ops);
        for start in (0..numel).step_by(self.tile) {
            let n = self.tile.min(numel - start);
            self.state.read_master(&mut tiles, i, start, &mut out[start..start + n])?;
        }
        Ok(out)
    }

    /// Replace tensor `i`'s master copy with `data` and re-encode the stored weight from it.
    pub fn write_master(&mut self, ops: &dyn DeviceIo, i: usize, data: &[f32]) -> io::Result<()> {
        let numel = self.descs[i].numel() as usize;
        if data.len() != numel {
            return Err(invalid(format!("{}: {} values for {numel} elements", self.specs[i].name, data.len())));
        }
        let mut tiles = tiles!(self, ops);
        for start in (0..numel).step_by(self.tile) {
            let n = self.tile.min(numel - start);
            self.state.set_master(i, start, &data[start..start + n])?;
            tiles.write_value(i, start, &data[start..start + n], Rounding::NEAREST)?;
        }
        self.refresh_tables(ops)
    }

    /// Put the host copy of every table on the device again after it changed outside an update (a restore, a written
    /// master): a layer reloads when the backward or forward next reaches it, a table holds its slot for the whole run.
    /// A no-op for a build whose weights are all resident (it writes the device weights directly).
    fn refresh_tables(&mut self, ops: &dyn DeviceIo) -> io::Result<()> {
        let tables = self.tables();
        self.load_tensors(ops, &tables)
    }

    /// The layer-at-a-time step: `body` updates tensors with [`LayeredStep::update`] as their gradients
    /// complete; the new values are published to the device together after `body` returns, so every gradient of
    /// the step was taken against one parameter version. `grad_multiplier` folds the clip and the accumulation
    /// scale (a trainer that cannot hold every gradient computes the norm as it goes). Gradients are zeroed
    /// after the publication.
    pub fn layered_step<R>(&mut self, ops: &dyn DeviceIo, lr: f32, grad_multiplier: f32, body: impl FnOnce(&mut LayeredStep<'_, '_>) -> io::Result<R>) -> io::Result<R> {
        self.require_resident("layered_step")?;
        let numels: Vec<usize> = self.specs.iter().map(|s| s.n as usize * s.k as usize).collect();
        let layout = tensor_layout(&numels, StateFormat::F32);
        let mut live = tiles!(self, ops);
        let pending = BlockStateStorage::new(layout.clone(), MemBacking::zeroed(layout.total_bytes()), self.tile).map_err(|e| invalid(e.to_string()))?;
        let mut params = DeferredPublish::new(&mut live, pending, numels, self.tile);
        self.state.begin_step();
        let out = {
            let mut step = LayeredStep { state: &mut self.state, params: &mut params, lr, grad_multiplier };
            body(&mut step)?
        };
        params.publish()?;
        self.zero_grads(ops);
        Ok(out)
    }

    /// Stream the master weights (the stored value where a tensor has none) into `sink` encoded as `dst[i]` for
    /// tensor `i`, one tile at a time: the export is a re-encoding of the highest-precision copy, not of the
    /// already-quantized resident weights.
    pub fn export(&mut self, ops: &dyn DeviceIo, dst: &[QuantEncoding], sink: &mut dyn TensorSink) -> io::Result<ExportReport> {
        let descs = self.export_descriptors(dst)?;
        let mut tiles = tiles!(self, ops);
        self.state.export_masters(&mut tiles, &descs, &self.registry, sink, self.tile)
    }

    /// The descriptors an export to `dst` writes, one per tensor.
    pub fn export_descriptors(&self, dst: &[QuantEncoding]) -> io::Result<Vec<TensorDescriptor>> {
        if dst.len() != self.specs.len() {
            return Err(invalid(format!("{} destination encodings for {} tensors", dst.len(), self.specs.len())));
        }
        Ok(self.specs.iter().zip(dst).map(|(s, e)| TensorDescriptor::new(s.name.clone(), vec![s.n as u64, s.k as u64], e.clone())).collect())
    }

    /// Atomically commit values, master, moments, step, seed and the trainer's schedule.
    pub fn commit(&mut self, ops: &dyn DeviceIo, dir: &CheckpointDir, run: &RunState) -> io::Result<PathBuf> {
        let mut tiles = tiles!(self, ops);
        self.state.commit(dir, &mut tiles, run)
    }

    /// The run state to commit now, given the trainer's schedule scalars and RNG counter.
    pub fn run_state(&self, rng_counter: u64, schedule: Vec<(String, f64)>) -> RunState {
        self.state.run_state(rng_counter, schedule)
    }

    /// Restore the newest valid checkpoint into the master, the state and the device weights.
    pub fn restore(&mut self, ops: &dyn DeviceIo, dir: &CheckpointDir) -> io::Result<Option<RunState>> {
        let mut tiles = tiles!(self, ops);
        let run = self.state.restore(dir, &mut tiles)?;
        self.refresh_tables(ops)?;
        Ok(run)
    }

    /// [`Self::restore`] of exactly the checkpoint committed at optimizer step `step`.
    pub fn restore_at(&mut self, ops: &dyn DeviceIo, dir: &CheckpointDir, step: u32) -> io::Result<Option<RunState>> {
        let mut tiles = tiles!(self, ops);
        let run = self.state.restore_at(dir, &mut tiles, step)?;
        self.refresh_tables(ops)?;
        Ok(run)
    }
}

/// The updates of one layer-at-a-time step. See [`ResidentFullFt::layered_step`].
pub struct LayeredStep<'a, 'b> {
    state: &'a mut TrainState,
    params: &'a mut DeferredPublish<'b, MemBacking>,
    lr: f32,
    grad_multiplier: f32,
}

impl LayeredStep<'_, '_> {
    /// Update tensor `i` from its (complete) device gradient. The new value waits for the publication.
    pub fn update(&mut self, i: usize) -> io::Result<()> {
        self.state.update_tensor(self.params, i, self.lr, self.grad_multiplier)
    }
}

/// A resident low-precision full fine-tune a model is built for: how the linears are held and updated
/// (`cfg`), what the machine offers (`capacity`) and what else the run needs (`others`: the activations and the
/// parameters outside the resident state). Shared by every model that builds on [`ResidentFullFt`].
#[derive(Clone, Debug)]
pub struct ResidentPlan {
    pub cfg: ResidentFtConfig,
    pub capacity: Capacity,
    pub others: Vec<Demand>,
}

impl ResidentPlan {
    pub fn budget(&self) -> ResidentBudget<'_> {
        ResidentBudget { capacity: self.capacity, others: &self.others, authorized: Authorized::none() }
    }
}

/// What happens inside a backward pass over the layers of a layer-streamed build (see [`ResidentBackward`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerEvent {
    /// The head and final norm gradients are complete; layers follow, the last first.
    AfterHead,
    /// The backward of `layer` is about to be submitted: load its weights and clear its gradients.
    BeforeBackward(usize),
    /// The backward of `layer` has been submitted: its gradients are complete.
    AfterBackward(usize),
}

/// The part of a model the generic resident update drives: the parameters outside the resident state (the norms, and
/// whatever else the model keeps in its own store), trained by the model's own device optimizer, and the model's
/// backward.
pub trait ResidentBackward {
    /// Send the work the model has recorded on its own handle to the device (see [`ResidentFullFt`]'s ordering note).
    fn flush_backward(&self);
    /// Clear the gradients of the parameters outside the resident state.
    fn zero_remainder_grads(&self);
    /// Their sum of squares, reduced on the device with `norm`.
    fn remainder_grad_sum_sq(&self, norm: &DeviceGradNorm) -> f64;
    /// Run one backward over the layers of a layer-streamed build, reporting each [`LayerEvent`] in order; the first
    /// error stops the pass.
    fn backward_pass(&self, on: &mut dyn FnMut(LayerEvent) -> io::Result<()>) -> io::Result<()>;
    /// Update the remainder with the device optimizer for the step `args` describes, its gradients multiplied by
    /// `grad_multiplier` (the clip is already folded in: the optimizer must not clip again).
    fn update_remainder(&self, args: &StepArgs, grad_multiplier: f32);
}

/// The run's per-step values of [`adamw_step`].
#[derive(Clone, Copy, Debug)]
pub struct StepArgs {
    pub step: u32,
    pub lr: f32,
    pub wd: f32,
    pub adam: Adam,
    /// The global gradient-norm bound over the resident linears AND the remainder.
    pub clip: Option<f32>,
    /// The accumulation scale folded into every gradient.
    pub extra_scale: f32,
}

/// One optimizer step of a resident build, whether its layers are all resident or streamed through a window: the
/// global gradient norm spans the resident linears and the model's remainder, and one multiplier (the accumulation
/// scale and the clip) reaches both. Returns the global norm the clip measured, `None` without a clip.
///
/// A resident build updates from the gradients the model's backward already left. A layer-streamed build runs the
/// backward here, because its layers are loaded and updated as it goes: once without a clip, twice with the exact
/// clip ([`ClipPass::TwoPass`]: the first pass measures the norm, the second updates), or once with the running clip.
pub fn adamw_step(host: &dyn ResidentBackward, ft: &RefCell<ResidentFullFt>, ops: &dyn DeviceIo, args: &StepArgs) -> io::Result<Option<f32>> {
    let StepArgs { step, lr, wd, adam, clip, extra_scale } = *args;
    {
        // The run's AdamW hyper-parameters reach an AdamW state; an Adafactor state keeps its own.
        let mut ft = ft.borrow_mut();
        if matches!(ft.rule(), Rule::AdamW { .. }) {
            ft.set_rule(Rule::AdamW { adam, wd });
        }
        debug_assert_eq!(ft.step_count() + 1, step, "the resident state and the run agree on the step index");
    }
    let norm = |sum_sq: f64| clip.map(|_| (sum_sq.sqrt() * f64::from(extra_scale)) as f32);
    host.flush_backward();
    if !ft.borrow().is_layer_streamed() {
        let remainder = || host.remainder_grad_sum_sq(ft.borrow().device_norm());
        let sum_sq = if clip.is_some() {
            let resident = ft.borrow_mut().grad_sum_sq(ops)?;
            resident + remainder()
        } else {
            0.0
        };
        let multiplier = optim::grad_multiplier(sum_sq, clip, extra_scale);
        ft.borrow_mut().step_scaled(ops, lr, multiplier)?;
        host.update_remainder(args, multiplier);
        return Ok(norm(sum_sq));
    }
    // Everything a layer pass does not cover: the model's own parameters and the tables. Both are updated once the
    // backward is complete, by the step's final multiplier.
    let remainder = || {
        host.flush_backward();
        let mut ft = ft.borrow_mut();
        host.remainder_grad_sum_sq(ft.device_norm()) + ft.table_grad_sum_sq(ops)
    };
    // One backward over the layers under `pass`: layers are loaded as the backward reaches them, measured or updated
    // as their gradients complete.
    let run = |pass: StreamPass, after_head: &dyn Fn()| -> io::Result<PassReport> {
        host.zero_remainder_grads();
        host.flush_backward();
        ft.borrow().zero_grads(ops);
        ft.borrow_mut().begin_pass(pass);
        host.backward_pass(&mut |event| {
            host.flush_backward();
            match event {
                LayerEvent::AfterHead => {
                    after_head();
                    Ok(())
                }
                LayerEvent::BeforeBackward(layer) => ft.borrow_mut().begin_layer_backward(ops, layer),
                LayerEvent::AfterBackward(layer) => ft.borrow_mut().finish_layer(ops, layer),
            }
        })?;
        ft.borrow_mut().end_pass()
    };
    let clip_pass = ft.borrow().clip_pass();
    let (multiplier, sum_sq) = match (clip, clip_pass) {
        (None, _) => {
            run(StreamPass::Update { lr, grad_multiplier: extra_scale }, &|| {})?;
            (extra_scale, 0.0)
        }
        (Some(_), ClipPass::TwoPass) => {
            let measured = run(StreamPass::Measure, &|| {})?;
            let sum_sq = measured.sum_sq + remainder();
            let multiplier = optim::grad_multiplier(sum_sq, clip, extra_scale);
            run(StreamPass::Update { lr, grad_multiplier: multiplier }, &|| {})?;
            (multiplier, sum_sq)
        }
        (Some(_), ClipPass::Running) => {
            // The head and final norm are complete after the first segment: they open the running norm. The rest of
            // the remainder (the embedding gradient completes last) is clipped by the full norm.
            let head = Cell::new(0.0);
            let report = run(StreamPass::Running { lr, clip, extra_scale }, &|| {
                head.set(remainder());
                ft.borrow_mut().add_external_sum_sq(head.get());
            })?;
            let sum_sq = report.sum_sq - head.get() + remainder();
            (optim::grad_multiplier(sum_sq, clip, extra_scale), sum_sq)
        }
    };
    host.flush_backward();
    ft.borrow_mut().update_tables(ops, lr, multiplier)?;
    host.update_remainder(args, multiplier);
    Ok(norm(sum_sq))
}

fn lcm(a: u64, b: u64) -> u64 {
    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    a / gcd(a, b) * b
}
