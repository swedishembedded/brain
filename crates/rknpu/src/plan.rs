// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running neural networks on the
// NPUs of low-cost embedded Linux SoCs for its clients. If your team needs
// expertise in embedded NPU bring-up then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Compiling a float [`Graph`] for the NPU: quantization, memory layout and
//! the order of NPU and CPU work. Pure and host-testable; [`crate::model`]
//! loads a plan onto the device.
//!
//! - **Quantization.** Every tensor gets affine int8 parameters from its
//!   calibrated range. Ops that only move or select values (concatenation, max
//!   pooling, upsampling) force their inputs and output onto one shared set,
//!   so they need no requantization.
//! - **Memory.** One data region holds every tensor (native16), then the
//!   weights. Tensors whose lifetimes do not overlap share bytes (first fit,
//!   largest first). A concatenation is free: its inputs are written straight
//!   into adjacent planes of its output, which needs every input but the last
//!   to have a multiple of 16 channels.
//! - **Schedule.** An Add becomes a concatenation plus a 1x1 identity-pair
//!   convolution ([`Graph::lower_adds`]), so it runs on the NPU. Consecutive
//!   convolutions form one NPU segment (one linked submission). Max pool,
//!   upsample and any Add that could not be lowered run on the CPU between
//!   segments, in place on the native16 buffers, with explicit cache
//!   maintenance.
//!
//! [`Plan::simulate`] executes the whole plan on a host byte arena with the
//! exact integer arithmetic, so a board run is checked byte for byte.

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use crate::conv::{self, ConvAddrs, ConvGeometry};
use crate::cpu;
use crate::graph::{Calibration, ConvParams, Graph, Op, TensorId};
use crate::job::Task;

use crate::layout::{Shape, LANES};
use crate::quant::{ConvQuant, QParams};
use crate::{Error, Result};

/// Where a tensor lives in the data region and how it is quantized.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    /// Byte offset of its first plane.
    pub offset: usize,
    pub shape: Shape,
    pub q: QParams,
}

impl Placement {
    /// The byte range its planes cover.
    pub fn range(&self) -> Range<usize> {
        self.offset..self.offset + self.shape.bytes()
    }
}

/// A unit of work, in execution order.
#[derive(Clone, Debug)]
pub enum Step {
    /// Run tasks `tasks` on the NPU as one submission.
    Npu { tasks: Range<u32> },
    /// `out = requant(a) + requant(b)` per element.
    Add { a: TensorId, b: TensorId, output: TensorId, scale_a: i32, scale_b: i32 },
    MaxPool { input: TensorId, output: TensorId, kernel: usize, stride: usize, pad: usize },
    Upsample2x { input: TensorId, output: TensorId },
}

/// Upper bound of one constant buffer. Weights are split across buffers of
/// about this size (a layer never straddles two), because a board's
/// contiguous NPU memory is fragmented: several 1 MiB blocks are available
/// when one 7 MiB block is not.
const CONSTANT_CHUNK_BYTES: usize = 1 << 20;

/// One compiled convolution. Its int8 weights are not kept: they are
/// re-derived from the float source when they are written, so a plan for a
/// whole network costs kilobytes, not the model's size.
struct ConvLayer {
    /// After [`odd_kernel`]; `source_kernel` is the graph's own.
    geometry: ConvGeometry,
    source_kernel: usize,
    params: Arc<dyn ConvParams>,
    relu: bool,
    input: TensorId,
    output: TensorId,
    /// Which constant buffer holds its weights and bias, and where.
    chunk: usize,
    weights_at: usize,
    bias_at: usize,
}

impl ConvLayer {
    fn quantize(&self, placements: &[Placement]) -> Result<ConvQuant> {
        let (si, so) = (placements[self.input], placements[self.output]);
        let source = self.params.weights();
        let weights = odd_kernel_weights(&source, self.geometry.out_channels * self.geometry.input.c, self.source_kernel, self.geometry.kernel);
        ConvQuant::quantize(&weights, &self.params.bias(), so.shape.c, si.shape.c, self.geometry.kernel, si.q, so.q, self.relu)
    }
}

/// Where the plan's regions start, relative to the job base.
#[derive(Clone, Debug)]
pub struct AddressMap {
    pub activations: u32,
    /// One per [`Plan::constant_chunks`] entry.
    pub constants: Vec<u32>,
}

pub struct Plan {
    placements: Vec<Placement>,
    steps: Vec<Step>,
    convs: Vec<ConvLayer>,
    /// The conv layers each NPU step runs, by index into `convs`.
    segments: Vec<Range<usize>>,
    task_count: usize,
    activation_bytes: usize,
    constant_chunks: Vec<usize>,
    input: TensorId,
    outputs: Vec<TensorId>,
}

impl Plan {
    /// Compile `graph`, quantizing activations from `calibration` and the
    /// input with `input_q` (the caller decides how pixels map to int8).
    /// Weights are not read here; [`Plan::emit`] quantizes them.
    pub fn new(graph: &Graph, calibration: &Calibration, input_q: QParams) -> Result<Plan> {
        if calibration.ranges.len() != graph.tensor_count() {
            return Err(Error::Quantization(format!("calibration covers {} tensors, the graph has {}", calibration.ranges.len(), graph.tensor_count())));
        }
        let (lowered, calibration) = graph.lower_adds(calibration);
        let (graph, calibration) = (&lowered, &calibration);
        let q = quant_classes(graph, calibration, input_q)?;
        let (placements, activation_bytes) = layout(graph, &q)?;

        let mut convs: Vec<ConvLayer> = Vec::new();
        let mut constant_chunks: Vec<usize> = Vec::new();
        let mut task_count = 0usize;
        let mut steps = Vec::new();
        let mut segments = Vec::new();
        let mut segment_start = (0usize, 0usize);
        let close = |steps: &mut Vec<Step>, segments: &mut Vec<Range<usize>>, start: &mut (usize, usize), tasks: usize, convs: usize| {
            if tasks > start.0 {
                steps.push(Step::Npu { tasks: start.0 as u32..tasks as u32 });
                segments.push(start.1..convs);
            }
            *start = (tasks, convs);
        };
        for op in graph.ops() {
            match op {
                Op::Conv { input, output, params, kernel, stride, pad, relu } => {
                    let (si, so) = (placements[*input], placements[*output]);
                    let geometry = odd_kernel(ConvGeometry { input: si.shape, out_channels: so.shape.c, kernel: *kernel, stride: *stride, pad: *pad });
                    if geometry.output() != so.shape {
                        return Err(Error::Unsupported(format!("conv {geometry:?} produces {:?}, the graph says {:?}", geometry.output(), so.shape)));
                    }
                    task_count += conv::bands(&geometry)?.len();
                    let size = geometry.weight_bytes().next_multiple_of(64) + geometry.bias_bytes().next_multiple_of(64);
                    if constant_chunks.last().is_none_or(|&used| used + size > CONSTANT_CHUNK_BYTES) {
                        constant_chunks.push(0);
                    }
                    let chunk = constant_chunks.len() - 1;
                    let weights_at = constant_chunks[chunk];
                    let bias_at = weights_at + geometry.weight_bytes().next_multiple_of(64);
                    constant_chunks[chunk] += size;
                    convs.push(ConvLayer { geometry, source_kernel: *kernel, params: Arc::clone(params), relu: *relu, input: *input, output: *output, chunk, weights_at, bias_at });
                }
                Op::Concat { .. } => {}
                Op::Add { a, b, output } => {
                    close(&mut steps, &mut segments, &mut segment_start, task_count, convs.len());
                    let so = placements[*output].q.scale;
                    let fixed = |s: f32| ((f64::from(s) / f64::from(so)) * f64::from(1u32 << cpu::ADD_FRAC_BITS)).round() as i32;
                    steps.push(Step::Add { a: *a, b: *b, output: *output, scale_a: fixed(placements[*a].q.scale), scale_b: fixed(placements[*b].q.scale) });
                }
                Op::MaxPool { input, output, kernel, stride, pad } => {
                    close(&mut steps, &mut segments, &mut segment_start, task_count, convs.len());
                    steps.push(Step::MaxPool { input: *input, output: *output, kernel: *kernel, stride: *stride, pad: *pad });
                }
                Op::Upsample2x { input, output } => {
                    close(&mut steps, &mut segments, &mut segment_start, task_count, convs.len());
                    steps.push(Step::Upsample2x { input: *input, output: *output });
                }
            }
        }
        close(&mut steps, &mut segments, &mut segment_start, task_count, convs.len());
        Ok(Plan { placements, steps, convs, segments, task_count, activation_bytes, constant_chunks, input: graph.input(), outputs: graph.outputs().to_vec() })
    }

    pub fn placement(&self, t: TensorId) -> Placement {
        self.placements[t]
    }

    pub fn input(&self) -> TensorId {
        self.input
    }

    pub fn outputs(&self) -> &[TensorId] {
        &self.outputs
    }

    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    pub fn task_count(&self) -> usize {
        self.task_count
    }

    /// Bytes of the activation region (every tensor, native16).
    pub fn activation_bytes(&self) -> usize {
        self.activation_bytes
    }

    /// Bytes of each constant buffer (weight tables and bias blocks).
    pub fn constant_chunks(&self) -> &[usize] {
        &self.constant_chunks
    }

    /// Quantize every layer (one layer's parameters in memory at a time),
    /// write its constants into `chunks` (one buffer per
    /// [`Plan::constant_chunks`] entry) and return the task programs for the
    /// regions at `at`.
    pub fn emit(&self, at: &AddressMap, chunks: &mut [&mut [u8]]) -> Result<Vec<Task>> {
        if at.constants.len() != self.constant_chunks.len() || chunks.len() != self.constant_chunks.len() {
            return Err(Error::Unsupported(format!("the plan has {} constant buffers; got {} addresses and {} buffers", self.constant_chunks.len(), at.constants.len(), chunks.len())));
        }
        let mut tasks = Vec::with_capacity(self.task_count);
        for c in &self.convs {
            let quant = c.quantize(&self.placements)?;
            let chunk = &mut chunks[c.chunk];
            conv::pack_weights(&c.geometry, &quant, &mut chunk[c.weights_at..c.weights_at + c.geometry.weight_bytes()]);
            conv::pack_bias(&quant, &mut chunk[c.bias_at..c.bias_at + c.geometry.bias_bytes()]);
            let (si, so) = (self.placements[c.input], self.placements[c.output]);
            let addrs = ConvAddrs {
                input: at.activations + si.offset as u32,
                input_plane_bytes: si.shape.plane_bytes(),
                output: at.activations + so.offset as u32,
                output_plane_bytes: so.shape.plane_bytes(),
                weights: at.constants[c.chunk] + c.weights_at as u32,
                bias: at.constants[c.chunk] + c.bias_at as u32,
            };
            for program in conv::programs(&c.geometry, &quant, si.q.zero_point, &addrs)? {
                tasks.push(Task { program, enable: conv::ENABLE, int_mask: conv::INT_MASK });
            }
        }
        Ok(tasks)
    }

    /// Run one CPU step on `data`, the activation region.
    pub fn run_cpu_step(&self, step: &Step, data: &mut [u8]) {
        match *step {
            Step::Npu { .. } => unreachable!("NPU steps run on the device"),
            Step::Add { a, b, output, scale_a, scale_b } => {
                let (pa, pb, po) = (self.placements[a], self.placements[b], self.placements[output]);
                let (out, [x, y]) = split(data, po.range(), [pa.range(), pb.range()]);
                cpu::add(x, y, out, pa.q.zero_point, pb.q.zero_point, po.q.zero_point, scale_a, scale_b);
            }
            Step::MaxPool { input, output, kernel, stride, pad } => {
                let (pi, po) = (self.placements[input], self.placements[output]);
                let (out, [x]) = split(data, po.range(), [pi.range()]);
                let (staged_in, mut staged_out) = (x.to_vec(), vec![0u8; out.len()]);
                cpu::max_pool(&staged_in, pi.shape, &mut staged_out, po.shape, kernel, stride, pad, &mut Vec::new());
                out.copy_from_slice(&staged_out);
            }
            Step::Upsample2x { input, output } => {
                let (pi, po) = (self.placements[input], self.placements[output]);
                let (out, [x]) = split(data, po.range(), [pi.range()]);
                cpu::upsample2x(x, pi.shape, out, po.shape);
            }
        }
    }

    /// The conv layers NPU step `index` (counting NPU steps only) runs.
    fn segment_convs(&self, index: usize) -> &[ConvLayer] {
        &self.convs[self.segments[index].clone()]
    }

    /// Execute the whole plan on a host activation region with the NPU's exact
    /// integer arithmetic. `data` must be [`Plan::activation_bytes`] long with
    /// the input already written; afterwards it holds what the board would.
    pub fn simulate(&self, data: &mut [u8]) -> Result<()> {
        let mut npu_steps = 0;
        for step in &self.steps {
            match step {
                Step::Npu { .. } => {
                    for c in self.segment_convs(npu_steps) {
                        let (pi, po) = (self.placements[c.input], self.placements[c.output]);
                        let input = pi.shape.unpack(&data[pi.offset..]);
                        let result = conv::reference(&c.geometry, &c.quantize(&self.placements)?, pi.q.zero_point, &input);
                        pack_into(po.shape, &result, &mut data[po.offset..]);
                    }
                    npu_steps += 1;
                }
                cpu => self.run_cpu_step(cpu, data),
            }
        }
        Ok(())
    }
}

/// Pack HWC `values` of `shape` into native16 planes at `dst` without
/// touching bytes outside the tensor's own planes (a concat neighbour may
/// share the surrounding region).
fn pack_into(shape: Shape, values: &[i8], dst: &mut [u8]) {
    for (pixel, px) in values.chunks(shape.c).enumerate() {
        for (c, &v) in px.iter().enumerate() {
            dst[(c / LANES) * shape.plane_bytes() + pixel * LANES + c % LANES] = v as u8;
        }
    }
}

/// An even kernel becomes the next odd one: zero taps on its last row and
/// column, one more row and column of bottom/right padding. Same outputs.
fn odd_kernel(g: ConvGeometry) -> ConvGeometry {
    if g.kernel % 2 == 1 {
        return g;
    }
    let [t, l, b, r] = g.pad;
    ConvGeometry { kernel: g.kernel + 1, pad: [t, l, b + 1, r + 1], ..g }
}

/// `weights` (`filters` filters of `k x k`) re-laid for an `k2 x k2` kernel
/// with the extra taps zero; borrowed unchanged when `k2 == k`.
fn odd_kernel_weights(weights: &[f32], filters: usize, k: usize, k2: usize) -> Cow<'_, [f32]> {
    if k == k2 {
        return Cow::Borrowed(weights);
    }
    let mut w = vec![0f32; filters * k2 * k2];
    for f in 0..filters {
        for ky in 0..k {
            for kx in 0..k {
                w[(f * k2 + ky) * k2 + kx] = weights[(f * k + ky) * k + kx];
            }
        }
    }
    Cow::Owned(w)
}

/// Quantization parameters per tensor. Concat, max pool and upsample share
/// one set across their inputs and output, covering all of their ranges.
fn quant_classes(graph: &Graph, calibration: &Calibration, input_q: QParams) -> Result<Vec<QParams>> {
    let n = graph.tensor_count();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for op in graph.ops() {
        if matches!(op, Op::Concat { .. } | Op::MaxPool { .. } | Op::Upsample2x { .. }) {
            let o = find(&mut parent, op.output());
            for i in op.inputs() {
                let r = find(&mut parent, i);
                parent[r] = o;
            }
        }
    }
    let mut class_range = vec![(f32::INFINITY, f32::NEG_INFINITY); n];
    for t in 0..n {
        let r = find(&mut parent, t);
        let (lo, hi) = calibration.ranges[t];
        class_range[r] = (class_range[r].0.min(lo), class_range[r].1.max(hi));
    }
    let input_class = find(&mut parent, graph.input());
    (0..n)
        .map(|t| {
            let r = find(&mut parent, t);
            if r == input_class {
                return Ok(input_q);
            }
            let (lo, hi) = class_range[r];
            QParams::from_range(lo, hi).map_err(|e| Error::Quantization(format!("tensor {t}: {e}")))
        })
        .collect()
}

/// Lay out every tensor in the data region. A concat output owns its planes
/// and its inputs live inside them; everything else gets its own planes.
fn layout(graph: &Graph, q: &[QParams]) -> Result<(Vec<Placement>, usize)> {
    let n = graph.tensor_count();
    // parent[t] = (concat output, plane offset) for a tensor stored inside a concat.
    let mut parent: Vec<Option<(TensorId, usize)>> = vec![None; n];
    for op in graph.ops() {
        if let Op::Concat { inputs, output } = op {
            let mut plane = 0;
            for (i, &t) in inputs.iter().enumerate() {
                let c = graph.shape(t).c;
                if i + 1 < inputs.len() && !c.is_multiple_of(LANES) {
                    return Err(Error::Unsupported(format!("concat input {t} has {c} channels; every input but the last needs a multiple of {LANES}")));
                }
                if parent[t].is_some() || t == graph.input() {
                    return Err(Error::Unsupported(format!("tensor {t} is concatenated twice (or is the graph input); it would need a copy")));
                }
                parent[t] = Some((*output, plane));
                plane += c / LANES;
            }
        }
    }
    // Storage roots (tensors not stored inside a concat) and the op interval
    // each must stay intact for: from the first write of any tensor stored in
    // it to the last read of any of them. The graph input is written before
    // op 0; outputs are read after the last op.
    let root = |mut t: TensorId| {
        while let Some((p, _)) = parent[t] {
            t = p;
        }
        t
    };
    let after_last = graph.ops().len();
    let mut live: Vec<Option<(usize, usize)>> = vec![None; n];
    let mut touch = |t: TensorId, at: usize| {
        let r = root(t);
        live[r] = Some(live[r].map_or((at, at), |(a, b)| (a.min(at), b.max(at))));
    };
    touch(graph.input(), 0);
    for (i, op) in graph.ops().iter().enumerate() {
        touch(op.output(), i);
        for t in op.inputs() {
            touch(t, i);
        }
    }
    for &t in graph.outputs() {
        touch(t, after_last);
    }
    // First fit, largest first: a root may share bytes with any root whose
    // interval does not intersect its own (intervals are inclusive, so an op's
    // inputs never share with its output).
    let mut roots: Vec<TensorId> = (0..n).filter(|&t| parent[t].is_none() && live[t].is_some()).collect();
    roots.sort_by_key(|&t| std::cmp::Reverse(graph.shape(t).bytes()));
    let mut offset = vec![usize::MAX; n];
    let mut placed: Vec<(usize, usize, (usize, usize))> = Vec::new();
    let mut cursor = 0usize;
    for t in roots {
        let size = graph.shape(t).bytes().next_multiple_of(64);
        let (a, b) = live[t].expect("filtered");
        let mut busy: Vec<(usize, usize)> = placed.iter().filter(|(_, _, (c, d))| a <= *d && *c <= b).map(|&(o, s, _)| (o, o + s)).collect();
        busy.sort_unstable();
        let mut at = 0;
        for (lo, hi) in busy {
            if at + size <= lo {
                break;
            }
            at = at.max(hi);
        }
        offset[t] = at;
        placed.push((at, size, (a, b)));
        cursor = cursor.max(at + size);
    }
    // Resolve nested concats outermost first (a parent always has a higher id
    // than its inputs, since ops are in execution order).
    for t in (0..n).rev() {
        if let Some((p, plane)) = parent[t] {
            offset[t] = offset[p] + plane * graph.shape(p).plane_bytes();
        }
    }
    let placements = (0..n).map(|t| Placement { offset: offset[t], shape: graph.shape(t), q: q[t] }).collect();
    Ok((placements, cursor))
}

/// `data[out]` mutably and each `data[ins[i]]` shared. Panics if an input
/// overlaps the output: the planner never lets an op's output share bytes
/// with its inputs (their lifetimes intersect at the op).
fn split<const N: usize>(data: &mut [u8], out: Range<usize>, ins: [Range<usize>; N]) -> (&mut [u8], [&[u8]; N]) {
    assert!(out.end <= data.len() && ins.iter().all(|r| r.end <= data.len()), "tensor outside the activation region");
    for r in &ins {
        assert!(r.end <= out.start || out.end <= r.start, "op input {r:?} overlaps its output {out:?}");
    }
    let base = data.as_mut_ptr();
    // SAFETY: every range is inside `data` and none of the inputs overlaps
    // the output, so the one mutable slice aliases nothing; the shared slices
    // may overlap each other, which shared borrows allow.
    unsafe { (std::slice::from_raw_parts_mut(base.add(out.start), out.len()), ins.map(|r| &*std::ptr::slice_from_raw_parts(base.add(r.start), r.len()))) }
}
