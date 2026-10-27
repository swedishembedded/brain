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
//! - **Schedule.** Consecutive convolutions form one NPU segment (one linked
//!   submission). Add, max pool and upsample run on the CPU between segments,
//!   in place on the native16 buffers, with explicit cache maintenance.
//!
//! [`Plan::simulate`] executes the whole plan on a host byte arena with the
//! exact integer arithmetic, so a board run is checked byte for byte.

use std::ops::Range;

use crate::conv::{self, ConvAddrs, ConvGeometry};
use crate::graph::{Calibration, Graph, Op, TensorId};
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

/// Fixed-point fraction bits of the CPU Add's operand scales.
const ADD_FRAC_BITS: u32 = 16;

struct ConvLayer {
    geometry: ConvGeometry,
    quant: ConvQuant,
    input: TensorId,
    output: TensorId,
    weights_at: usize,
    bias_at: usize,
}

pub struct Plan {
    placements: Vec<Placement>,
    steps: Vec<Step>,
    tasks: Vec<Task>,
    convs: Vec<ConvLayer>,
    /// The conv layers each NPU step runs, by index into `convs`.
    segments: Vec<Range<usize>>,
    data_bytes: usize,
    input: TensorId,
    outputs: Vec<TensorId>,
}

impl Plan {
    /// Compile `graph`, quantizing activations from `calibration` and the
    /// input with `input_q` (the caller decides how pixels map to int8).
    pub fn new(graph: &Graph, calibration: &Calibration, input_q: QParams) -> Result<Plan> {
        let n = graph.tensor_count();
        if calibration.ranges.len() != n {
            return Err(Error::Quantization(format!("calibration covers {} tensors, the graph has {n}", calibration.ranges.len())));
        }
        let q = quant_classes(graph, calibration, input_q)?;
        let (mut placements, mut data_bytes) = layout(graph, &q)?;

        let mut convs = Vec::new();
        let mut tasks = Vec::new();
        let mut steps = Vec::new();
        let mut segments = Vec::new();
        let mut segment_start = (0u32, 0usize);
        let close = |steps: &mut Vec<Step>, segments: &mut Vec<Range<usize>>, start: &mut (u32, usize), tasks: usize, convs: usize| {
            if tasks as u32 > start.0 {
                steps.push(Step::Npu { tasks: start.0..tasks as u32 });
                segments.push(start.1..convs);
            }
            *start = (tasks as u32, convs);
        };
        for op in graph.ops() {
            match op {
                Op::Conv { input, output, weights, bias, kernel, stride, pad, relu } => {
                    let (si, so) = (placements[*input], placements[*output]);
                    let (geometry, weights) = odd_kernel(ConvGeometry { input: si.shape, out_channels: so.shape.c, kernel: *kernel, stride: *stride, pad: *pad }, weights);
                    if geometry.output() != so.shape {
                        return Err(Error::Unsupported(format!("conv {geometry:?} produces {:?}, the graph says {:?}", geometry.output(), so.shape)));
                    }
                    let quant = ConvQuant::quantize(&weights, bias, so.shape.c, si.shape.c, geometry.kernel, si.q, so.q, *relu)?;
                    let weights_at = data_bytes;
                    let bias_at = weights_at + geometry.weight_bytes().next_multiple_of(64);
                    data_bytes = bias_at + geometry.bias_bytes().next_multiple_of(64);
                    let addrs = ConvAddrs {
                        input: si.offset as u32,
                        input_plane_bytes: si.shape.plane_bytes(),
                        output: so.offset as u32,
                        output_plane_bytes: so.shape.plane_bytes(),
                        weights: weights_at as u32,
                        bias: bias_at as u32,
                    };
                    for program in conv::programs(&geometry, &quant, si.q.zero_point, &addrs)? {
                        tasks.push(Task { program, enable: conv::ENABLE, int_mask: conv::INT_MASK });
                    }
                    convs.push(ConvLayer { geometry, quant, input: *input, output: *output, weights_at, bias_at });
                }
                Op::Concat { .. } => {}
                Op::Add { a, b, output } => {
                    close(&mut steps, &mut segments, &mut segment_start, tasks.len(), convs.len());
                    let so = placements[*output].q.scale;
                    let fixed = |s: f32| ((f64::from(s) / f64::from(so)) * f64::from(1u32 << ADD_FRAC_BITS)).round() as i32;
                    steps.push(Step::Add { a: *a, b: *b, output: *output, scale_a: fixed(placements[*a].q.scale), scale_b: fixed(placements[*b].q.scale) });
                }
                Op::MaxPool { input, output, kernel, stride, pad } => {
                    close(&mut steps, &mut segments, &mut segment_start, tasks.len(), convs.len());
                    steps.push(Step::MaxPool { input: *input, output: *output, kernel: *kernel, stride: *stride, pad: *pad });
                }
                Op::Upsample2x { input, output } => {
                    close(&mut steps, &mut segments, &mut segment_start, tasks.len(), convs.len());
                    steps.push(Step::Upsample2x { input: *input, output: *output });
                }
            }
        }
        close(&mut steps, &mut segments, &mut segment_start, tasks.len(), convs.len());
        // Every tensor keeps its graph shape; the input is quantized as asked.
        placements[graph.input()].q = input_q;
        Ok(Plan { placements, steps, tasks, convs, segments, data_bytes, input: graph.input(), outputs: graph.outputs().to_vec() })
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

    pub fn tasks(&self) -> &[Task] {
        &self.tasks
    }

    /// Bytes of the data region (tensors, then weights and biases).
    pub fn data_bytes(&self) -> usize {
        self.data_bytes
    }

    /// Write every constant (weight tables, bias blocks) into `data`.
    pub fn write_constants(&self, data: &mut [u8]) {
        for c in &self.convs {
            conv::pack_weights(&c.geometry, &c.quant, &mut data[c.weights_at..c.weights_at + c.geometry.weight_bytes()]);
            conv::pack_bias(&c.quant, &mut data[c.bias_at..c.bias_at + c.geometry.bias_bytes()]);
        }
    }

    /// Run one CPU step on `data`, the data region.
    pub fn run_cpu_step(&self, step: &Step, data: &mut [u8]) {
        match *step {
            Step::Npu { .. } => unreachable!("NPU steps run on the device"),
            Step::Add { a, b, output, scale_a, scale_b } => {
                let (pa, pb, po) = (self.placements[a], self.placements[b], self.placements[output]);
                let (za, zb, zo) = (i32::from(pa.q.zero_point), i32::from(pb.q.zero_point), i32::from(po.q.zero_point));
                let half = 1i32 << (ADD_FRAC_BITS - 1);
                for i in 0..po.shape.planes() * po.shape.plane_bytes() {
                    let x = i32::from(data[pa.offset + i] as i8) - za;
                    let y = i32::from(data[pb.offset + i] as i8) - zb;
                    let v = ((x * scale_a + y * scale_b + half) >> ADD_FRAC_BITS) + zo;
                    data[po.offset + i] = v.clamp(-128, 127) as i8 as u8;
                }
            }
            Step::MaxPool { input, output, kernel, stride, pad } => {
                let (pi, po) = (self.placements[input], self.placements[output]);
                let (si, so) = (pi.shape, po.shape);
                for plane in 0..si.planes() {
                    let src = pi.offset + plane * si.plane_bytes();
                    let dst = po.offset + plane * so.plane_bytes();
                    for oy in 0..so.h {
                        let ys = (oy * stride).saturating_sub(pad)..(oy * stride + kernel - pad).min(si.h);
                        for ox in 0..so.w {
                            let xs = (ox * stride).saturating_sub(pad)..(ox * stride + kernel - pad).min(si.w);
                            let mut m = [i8::MIN; LANES];
                            for y in ys.clone() {
                                for x in xs.clone() {
                                    let atom = &data[src + (y * si.w + x) * LANES..][..LANES];
                                    for (m, &v) in m.iter_mut().zip(atom) {
                                        *m = (*m).max(v as i8);
                                    }
                                }
                            }
                            for (l, v) in m.iter().enumerate() {
                                data[dst + (oy * so.w + ox) * LANES + l] = *v as u8;
                            }
                        }
                    }
                }
            }
            Step::Upsample2x { input, output } => {
                let (pi, po) = (self.placements[input], self.placements[output]);
                let (si, so) = (pi.shape, po.shape);
                for plane in 0..si.planes() {
                    let src = pi.offset + plane * si.plane_bytes();
                    let dst = po.offset + plane * so.plane_bytes();
                    for y in 0..so.h {
                        for x in 0..so.w {
                            let from = src + ((y / 2) * si.w + x / 2) * LANES;
                            data.copy_within(from..from + LANES, dst + (y * so.w + x) * LANES);
                        }
                    }
                }
            }
        }
    }

    /// The conv layers NPU step `index` (counting NPU steps only) runs.
    fn segment_convs(&self, index: usize) -> &[ConvLayer] {
        &self.convs[self.segments[index].clone()]
    }

    /// Execute the whole plan on a host arena with the NPU's exact integer
    /// arithmetic. `data` must be [`Plan::data_bytes`] long with the input
    /// already written; afterwards it holds what the board would hold.
    pub fn simulate(&self, data: &mut [u8]) {
        let mut npu_steps = 0;
        for step in &self.steps {
            match step {
                Step::Npu { .. } => {
                    for c in self.segment_convs(npu_steps) {
                        let (pi, po) = (self.placements[c.input], self.placements[c.output]);
                        let input = pi.shape.unpack(&data[pi.offset..]);
                        let result = conv::reference(&c.geometry, &c.quant, pi.q.zero_point, &input);
                        pack_into(po.shape, &result, &mut data[po.offset..]);
                    }
                    npu_steps += 1;
                }
                cpu => self.run_cpu_step(cpu, data),
            }
        }
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
fn odd_kernel(g: ConvGeometry, weights: &[f32]) -> (ConvGeometry, Vec<f32>) {
    if g.kernel % 2 == 1 {
        return (g, weights.to_vec());
    }
    let (k, k2) = (g.kernel, g.kernel + 1);
    let n = g.out_channels * g.input.c;
    let mut w = vec![0f32; n * k2 * k2];
    for oc_ic in 0..n {
        for ky in 0..k {
            for kx in 0..k {
                w[(oc_ic * k2 + ky) * k2 + kx] = weights[(oc_ic * k + ky) * k + kx];
            }
        }
    }
    let [t, l, b, r] = g.pad;
    (ConvGeometry { kernel: k2, pad: [t, l, b + 1, r + 1], ..g }, w)
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
