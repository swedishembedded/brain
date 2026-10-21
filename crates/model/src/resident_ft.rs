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
//! Nothing here names a model. A model contributes the `(name, n, k)` of its linears and a
//! [`checkpoint::TensorSource`] of initial values; the planner refuses a configuration that does not fit
//! before the first byte is allocated ([`ResidentFullFt::demands`]).

use std::io;
use std::path::PathBuf;

use checkpoint::packed::{CodecRegistry, FloatFormat, QuantEncoding, TensorDescriptor, TensorSink, Tile, TileBuf};
use checkpoint::TensorSource;
use gpu_core::{DeviceBuffer, Gpu};
use optim::checkpoint::{CheckpointDir, RunState};
use optim::codec::Rounding;
use optim::deferred::DeferredPublish;
use optim::encoded::{aligned_tile, export_values, ExportReport};
use optim::rule::Rule;
use optim::storage::{BlockStateStorage, MemBacking};
use optim::streamed::{tensor_layout, ParamTiles, StepReport};
use optim::trainstate::{TrainState, TrainStateConfig};
use paramstore::precision::{GradDtype, MasterPolicy, ParamPrecision, PrecisionPolicy, StateFormat, ValueDtype};
use gpu_core::Step;

use crate::memplan::{self, Authorized, Capacity, Category, Demand, Lever, PlanError, Variant};
use crate::ops::{rewrite_tile, DeviceWeightSink, Ops, Weight};

/// The resident storage format of the trained weights.
///
/// The narrow formats ([`Self::Int8`], [`Self::Nf4`]) make this quantization-aware training: the forward runs on
/// the quantized weight, the gradient taken there updates the master (the straight-through estimator), and the
/// stored weight is re-quantized from the master after every step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentValue {
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
            ResidentValue::Bf16 => 2 * n * k,
            ResidentValue::Fp8E4M3 { block } => (n * k).div_ceil(4) * 4 + 4 * n.div_ceil(block as u64) * k.div_ceil(block as u64),
            ResidentValue::Int8 => n * k + 4 * (n * k / 32),
            ResidentValue::Nf4 => (n * k).div_ceil(8) * 4 + 4 * (n * k / 32),
        }
    }

    /// Row-length multiple a weight of this format needs.
    fn k_multiple(self) -> u32 {
        match self {
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
    /// than the value's spacing would vanish).
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
}

impl ResidentFtConfig {
    fn precision(&self) -> ParamPrecision {
        ParamPrecision { value: ValueDtype::Bf16, grad: self.grad, state: self.state, master: self.master, stochastic_rounding: self.stochastic_state }
    }
}

/// One resident linear: `y = x W^T` with `W` of shape `[n, k]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentSpec {
    pub name: String,
    pub n: u32,
    pub k: u32,
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

/// The planner categories of the resident weights, gradients, master and optimizer state. The master and the
/// optimizer state are host bytes (the offloaded layout); only the weights and gradients cost device memory.
pub fn demands(cfg: &ResidentFtConfig, specs: &[ResidentSpec]) -> Result<Vec<Demand>, PlanError> {
    let numel: usize = specs.iter().map(|s| s.n as usize * s.k as usize).sum();
    let weights: u64 = specs.iter().map(|s| cfg.value.device_bytes(s.n as u64, s.k as u64)).sum();
    let bytes = cfg.precision().bytes(numel, cfg.rule.slots().len());
    let manifest: Vec<(String, usize)> = specs.iter().map(|s| (s.name.clone(), s.n as usize * s.k as usize)).collect();
    let ladder = paramstore::precision::optimizer_state_ladder(&manifest, cfg.rule.slots().len(), 256);
    let requested = ladder.iter().position(|(f, _)| *f == cfg.state).map(|i| ladder[i].1).unwrap_or(bytes.moments);
    // An Adafactor run keeps row and column statistics (`n + k` floats a matrix) instead of elementwise moments.
    let factored: u64 = if matches!(cfg.rule, Rule::Adafactor { .. }) { specs.iter().map(|s| 4 * (s.n as u64 + s.k as u64)).sum() } else { 0 };
    let label = format!("resident {:?} weights", cfg.value);
    Ok(vec![
        Demand::fixed(Category::Weights, label, weights),
        Demand::fixed(Category::Grads, format!("{:?} gradients", cfg.grad), bytes.grad),
        Demand { category: Category::OptimizerState, variants: vec![Variant::new(Lever::Base, format!("{:?} master + {:?} state on the host", cfg.master, cfg.state), 0, bytes.master + requested + factored)] },
    ])
}

/// Refuse, before anything is allocated, a configuration the planner cannot fit.
pub fn check_fits(cfg: &ResidentFtConfig, specs: &[ResidentSpec], budget: &ResidentBudget<'_>) -> Result<memplan::PlanReport, String> {
    let mut all = demands(cfg, specs).map_err(|e| e.to_string())?;
    all.extend_from_slice(budget.others);
    memplan::plan(&all, budget.capacity, &budget.authorized).map_err(|e| format!("resident full fine-tune does not fit: {e}"))
}

/// Elements of the window [`SourceTiles`] keeps of the tensor it is reading (16 MiB of f32).
const SOURCE_WINDOW: usize = 1 << 22;

/// Initial values read from a [`TensorSource`] as the master's and the first stored value's source, through the
/// bounded chunk reader: the host holds one window of the tensor being read, never the tensor.
struct SourceTiles<'a> {
    src: &'a dyn TensorSource,
    names: Vec<&'a str>,
    /// Added to the tensor index the caller passes (an export of one tensor starts at 0).
    base: usize,
    window: Vec<f32>,
    /// `(tensor, first element)` the window holds.
    at: Option<(usize, usize)>,
}

impl<'a> SourceTiles<'a> {
    fn new(src: &'a dyn TensorSource, names: Vec<&'a str>, base: usize) -> Self {
        SourceTiles { src, names, base, window: Vec::new(), at: None }
    }

    fn fill(&mut self, ti: usize, start: usize, need: usize) -> io::Result<()> {
        let name = self.names[ti];
        let len = SOURCE_WINDOW.max(need);
        let mut window = std::mem::take(&mut self.window);
        window.clear();
        window.resize(len, 0.0);
        let mut filled = 0usize;
        let found = self.src.with_tensor_chunks(name, SOURCE_WINDOW, &mut |off, chunk| {
            let (lo, hi) = (off as usize, off as usize + chunk.len());
            let (from, to) = (lo.max(start), hi.min(start + len));
            if from < to {
                window[from - start..to - start].copy_from_slice(&chunk[from - lo..to - lo]);
                filled = filled.max(to - start);
            }
        });
        if !found || filled < need {
            return Err(invalid(format!("initial values: tensor '{name}' range {start}..{} is not in the source", start + need)));
        }
        window.truncate(filled);
        self.window = window;
        self.at = Some((ti, start));
        Ok(())
    }
}

impl ParamTiles for SourceTiles<'_> {
    fn read_value(&mut self, tensor: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        let ti = self.base + tensor;
        let covered = self.at.is_some_and(|(t, first)| t == ti && start >= first && start + out.len() <= first + self.window.len());
        if !covered {
            self.fill(ti, start, out.len())?;
        }
        let (_, first) = self.at.expect("filled above");
        out.copy_from_slice(&self.window[start - first..start - first + out.len()]);
        Ok(())
    }
    fn write_value(&mut self, tensor: usize, _: usize, _: &[f32], _: Rounding) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::Unsupported, format!("tensor {tensor}: the initial values are read-only")))
    }
    fn read_grad(&mut self, tensor: usize, _: usize, _: &mut [f32]) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::Unsupported, format!("tensor {tensor}: the initial values carry no gradient")))
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
    ops: &'a Ops,
    weights: &'a [Weight],
    grads: &'a [DeviceBuffer],
    descs: &'a [TensorDescriptor],
    registry: &'a CodecRegistry,
    grad_dtype: GradDtype,
    /// Tensor-sized caches of the last tensor read (the device read is whole-buffer): host staging is the
    /// largest tensor, not the model.
    grad_cache: Option<(usize, Vec<f32>)>,
    value_cache: Option<(usize, Vec<u8>, Vec<Vec<u8>>)>,
    enc: TileBuf,
    times: HostTimes,
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
        if self.grad_cache.as_ref().map(|c| c.0) != Some(ti) {
            let t0 = std::time::Instant::now();
            let numel = self.descs[ti].numel() as usize;
            let g = match self.grad_dtype {
                GradDtype::F32 => self.gpu().read(&self.grads[ti], numel),
                GradDtype::Bf16 => {
                    let words = self.gpu().read(&self.grads[ti], numel / 2);
                    words.iter().flat_map(|w| [f32::from_bits(w.to_bits() << 16), f32::from_bits(w.to_bits() & 0xffff_0000)]).collect()
                }
            };
            self.grad_cache = Some((ti, g));
            self.times.grad_read += t0.elapsed();
        }
        let g = &self.grad_cache.as_ref().expect("filled above").1;
        out.copy_from_slice(&g[start..start + out.len()]);
        Ok(())
    }

    fn write_grad(&mut self, ti: usize, _: usize, _: &[f32]) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::Unsupported, format!("tensor {ti}: gradients are produced on the device by Ops::weight_grad")))
    }
}

/// What an export of a resident build wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentExport {
    /// The streamed linears: tensors, payload bytes and the host staging the export used.
    pub linears: ExportReport,
    /// Bytes of the whole file's tensors (linears in the chosen encoding, the remainder in fp32).
    pub bytes: u64,
}

/// The trainable linears of a model, resident on the device in a low-precision encoding with a master copy and
/// optimizer state off it.
pub struct ResidentFullFt {
    cfg: ResidentFtConfig,
    specs: Vec<ResidentSpec>,
    descs: Vec<TensorDescriptor>,
    weights: Vec<Weight>,
    grads: Vec<DeviceBuffer>,
    state: TrainState,
    registry: CodecRegistry,
    tile: usize,
    times: HostTimes,
}

impl ResidentFullFt {
    /// Check the plan, then upload every weight from `src` in the resident encoding (one tile of host staging
    /// per tensor at a time) and seed the master copies from the same f32 values. Nothing is allocated when the
    /// planner refuses.
    pub fn new(ops: &Ops, cfg: ResidentFtConfig, specs: Vec<ResidentSpec>, src: &dyn TensorSource, budget: &ResidentBudget<'_>) -> Result<ResidentFullFt, String> {
        if cfg.master == MasterPolicy::None {
            return Err("a resident bf16/fp8 weight needs a master copy (master: f32 or bf16); without one, updates below the value's spacing are lost".into());
        }
        for s in &specs {
            if !s.k.is_multiple_of(cfg.value.k_multiple()) {
                return Err(format!("{}: K = {} must be a multiple of {} for {:?} weights", s.name, s.k, cfg.value.k_multiple(), cfg.value));
            }
            if cfg.grad == GradDtype::Bf16 && !s.k.is_multiple_of(2) {
                return Err(format!("{}: a bf16 gradient needs an even K (got {})", s.name, s.k));
            }
        }
        check_fits(&cfg, &specs, budget)?;
        let enc = cfg.value.encoding();
        let descs: Vec<TensorDescriptor> = specs.iter().map(|s| TensorDescriptor::new(s.name.clone(), vec![s.n as u64, s.k as u64], enc.clone())).collect();
        let registry = CodecRegistry::reference();
        for d in &descs {
            d.validate()?;
        }
        // A tile every tensor can be written in: whole scale bands for FP8, whole payload words for all.
        let tile = descs.iter().fold(aligned_tile(&descs, cfg.tile_elems.max(1)) as u64, |t, _| t);
        let tile = lcm(tile, 4) as usize;
        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        let mut weights = Vec::with_capacity(specs.len());
        let mut sink = DeviceWeightSink::new(ops);
        for (ti, d) in descs.iter().enumerate() {
            let mut init = SourceTiles::new(src, names.clone(), ti);
            export_values(&mut init, std::slice::from_ref(d), &registry, &mut sink, tile).map_err(|e| e.to_string())?;
            weights.push(sink.take_weight().ok_or_else(|| format!("{}: the upload produced no weight", d.name))?);
        }
        let manifest: Vec<(String, usize)> = specs.iter().map(|s| (s.name.clone(), s.n as usize * s.k as usize)).collect();
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
        state.init_master(&mut SourceTiles::new(src, names, 0)).map_err(|e| e.to_string())?;
        let gpu = ops.gpu();
        let grads = specs
            .iter()
            .map(|s| {
                let elems = s.n as u64 * s.k as u64;
                gpu.storage(if cfg.grad == GradDtype::Bf16 { elems / 2 } else { elems })
            })
            .collect();
        Ok(ResidentFullFt { cfg, specs, descs, weights, grads, state, registry, tile, times: HostTimes::default() })
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
    pub fn zero_grads(&self, ops: &Ops) {
        let clears: Vec<&DeviceBuffer> = self.grads.iter().collect();
        ops.gpu().submit(&clears, &[]);
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

    /// Sum of squares of every gradient (for the global clip of [`optim::grad_multiplier`]).
    pub fn grad_sum_sq(&mut self, ops: &Ops) -> io::Result<f64> {
        let mut tiles = device_tiles(ops, &self.weights, &self.grads, &self.descs, &self.registry, self.cfg.grad);
        let sum = self.state.grad_sum_sq(&mut tiles);
        self.times.add(&tiles.times);
        sum
    }

    /// Host time spent in the updates of this run so far, by phase.
    pub fn host_times(&self) -> HostTimes {
        self.times
    }

    /// One update step over every tensor: the gradients are multiplied by [`optim::grad_multiplier`]
    /// (`extra_scale`, and the global clip when `clip` is set), the master and state are updated tile by tile
    /// and the device weights re-encoded in place. The gradients are zeroed afterwards.
    pub fn step(&mut self, ops: &Ops, lr: f32, clip: Option<f32>, extra_scale: f32) -> io::Result<StepReport> {
        let mut tiles = device_tiles(ops, &self.weights, &self.grads, &self.descs, &self.registry, self.cfg.grad);
        let report = self.state.step(&mut tiles, lr, clip, extra_scale)?;
        self.zero_grads(ops);
        Ok(report)
    }

    /// [`Self::step`] with the gradient multiplier already decided, for a trainer whose global clip spans
    /// parameters this state does not hold (it adds its own [`Self::grad_sum_sq`] to theirs).
    pub fn step_scaled(&mut self, ops: &Ops, lr: f32, grad_multiplier: f32) -> io::Result<StepReport> {
        let mut tiles = device_tiles(ops, &self.weights, &self.grads, &self.descs, &self.registry, self.cfg.grad);
        let t0 = std::time::Instant::now();
        let step = self.state.begin_step();
        for i in 0..self.specs.len() {
            self.state.update_tensor(&mut tiles, i, lr, grad_multiplier)?;
        }
        tiles.times.update += t0.elapsed();
        tiles.times.steps += 1;
        self.times.add(&tiles.times);
        self.zero_grads(ops);
        Ok(StepReport { step, grad_multiplier })
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
    pub fn read_master(&mut self, ops: &Ops, i: usize) -> io::Result<Vec<f32>> {
        let numel = self.descs[i].numel() as usize;
        let mut out = vec![0f32; numel];
        let mut tiles = device_tiles(ops, &self.weights, &self.grads, &self.descs, &self.registry, self.cfg.grad);
        for start in (0..numel).step_by(self.tile) {
            let n = self.tile.min(numel - start);
            self.state.read_master(&mut tiles, i, start, &mut out[start..start + n])?;
        }
        Ok(out)
    }

    /// Replace tensor `i`'s master copy with `data` and re-encode the stored weight from it.
    pub fn write_master(&mut self, ops: &Ops, i: usize, data: &[f32]) -> io::Result<()> {
        let numel = self.descs[i].numel() as usize;
        if data.len() != numel {
            return Err(invalid(format!("{}: {} values for {numel} elements", self.specs[i].name, data.len())));
        }
        let mut tiles = device_tiles(ops, &self.weights, &self.grads, &self.descs, &self.registry, self.cfg.grad);
        for start in (0..numel).step_by(self.tile) {
            let n = self.tile.min(numel - start);
            self.state.set_master(i, start, &data[start..start + n])?;
            tiles.write_value(i, start, &data[start..start + n], Rounding::NEAREST)?;
        }
        Ok(())
    }

    /// The layer-at-a-time step: `body` updates tensors with [`LayeredStep::update`] as their gradients
    /// complete; the new values are published to the device together after `body` returns, so every gradient of
    /// the step was taken against one parameter version. `grad_multiplier` folds the clip and the accumulation
    /// scale (a trainer that cannot hold every gradient computes the norm as it goes). Gradients are zeroed
    /// after the publication.
    pub fn layered_step<R>(&mut self, ops: &Ops, lr: f32, grad_multiplier: f32, body: impl FnOnce(&mut LayeredStep<'_, '_>) -> io::Result<R>) -> io::Result<R> {
        let numels: Vec<usize> = self.specs.iter().map(|s| s.n as usize * s.k as usize).collect();
        let layout = tensor_layout(&numels, StateFormat::F32);
        let mut live = device_tiles(ops, &self.weights, &self.grads, &self.descs, &self.registry, self.cfg.grad);
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
    pub fn export(&mut self, ops: &Ops, dst: &[QuantEncoding], sink: &mut dyn TensorSink) -> io::Result<ExportReport> {
        let descs = self.export_descriptors(dst)?;
        let mut tiles = device_tiles(ops, &self.weights, &self.grads, &self.descs, &self.registry, self.cfg.grad);
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
    pub fn commit(&mut self, ops: &Ops, dir: &CheckpointDir, run: &RunState) -> io::Result<PathBuf> {
        let mut tiles = device_tiles(ops, &self.weights, &self.grads, &self.descs, &self.registry, self.cfg.grad);
        self.state.commit(dir, &mut tiles, run)
    }

    /// The run state to commit now, given the trainer's schedule scalars and RNG counter.
    pub fn run_state(&self, rng_counter: u64, schedule: Vec<(String, f64)>) -> RunState {
        self.state.run_state(rng_counter, schedule)
    }

    /// Restore the newest valid checkpoint into the master, the state and the device weights.
    pub fn restore(&mut self, ops: &Ops, dir: &CheckpointDir) -> io::Result<Option<RunState>> {
        let mut tiles = device_tiles(ops, &self.weights, &self.grads, &self.descs, &self.registry, self.cfg.grad);
        self.state.restore(dir, &mut tiles)
    }

    /// [`Self::restore`] of exactly the checkpoint committed at optimizer step `step`.
    pub fn restore_at(&mut self, ops: &Ops, dir: &CheckpointDir, step: u32) -> io::Result<Option<RunState>> {
        let mut tiles = device_tiles(ops, &self.weights, &self.grads, &self.descs, &self.registry, self.cfg.grad);
        self.state.restore_at(dir, &mut tiles, step)
    }
}

fn device_tiles<'a>(
    ops: &'a Ops,
    weights: &'a [Weight],
    grads: &'a [DeviceBuffer],
    descs: &'a [TensorDescriptor],
    registry: &'a CodecRegistry,
    grad_dtype: GradDtype,
) -> DeviceTiles<'a> {
    DeviceTiles { ops, weights, grads, descs, registry, grad_dtype, grad_cache: None, value_cache: None, enc: TileBuf::default(), times: HostTimes::default() }
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
