// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running neural networks on the
// NPUs of low-cost embedded Linux SoCs for its clients. If your team needs
// expertise in embedded NPU bring-up then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The float network a model hands the NPU compiler, and its host reference.
//!
//! A [`Graph`] is a single-input feed-forward network over HWC feature maps,
//! with ops in execution order. A model crate builds it from its checkpoint
//! (weights stay float, `[out][in][k][k]`); the compiler quantizes it from a
//! [`Calibration`]. [`Graph::run`] is the float reference used to calibrate and
//! to measure what quantization costs.

use crate::layout::Shape;
use crate::{Error, Result};

pub type TensorId = usize;

#[derive(Clone, Debug)]
pub enum Op {
    /// Square-kernel convolution with bias, optionally followed by ReLU.
    Conv { input: TensorId, output: TensorId, weights: Vec<f32>, bias: Vec<f32>, kernel: usize, stride: usize, pad: [usize; 4], relu: bool },
    /// Element-wise sum of two same-shape tensors.
    Add { a: TensorId, b: TensorId, output: TensorId },
    /// Channel concatenation.
    Concat { inputs: Vec<TensorId>, output: TensorId },
    /// Square max pooling (padding never wins the max).
    MaxPool { input: TensorId, output: TensorId, kernel: usize, stride: usize, pad: usize },
    /// Nearest-neighbour 2x spatial upsampling.
    Upsample2x { input: TensorId, output: TensorId },
}

impl Op {
    pub fn output(&self) -> TensorId {
        match self {
            Op::Conv { output, .. } | Op::Add { output, .. } | Op::Concat { output, .. } | Op::MaxPool { output, .. } | Op::Upsample2x { output, .. } => *output,
        }
    }

    pub fn inputs(&self) -> Vec<TensorId> {
        match self {
            Op::Conv { input, .. } | Op::MaxPool { input, .. } | Op::Upsample2x { input, .. } => vec![*input],
            Op::Add { a, b, .. } => vec![*a, *b],
            Op::Concat { inputs, .. } => inputs.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Graph {
    shapes: Vec<Shape>,
    ops: Vec<Op>,
    input: TensorId,
    outputs: Vec<TensorId>,
}

impl Graph {
    /// A graph with one input of `shape`; build it with the op methods, which
    /// check shapes as they go and return the new tensor.
    pub fn new(input: Shape) -> Graph {
        Graph { shapes: vec![input], ops: Vec::new(), input: 0, outputs: Vec::new() }
    }

    pub fn input(&self) -> TensorId {
        self.input
    }

    pub fn outputs(&self) -> &[TensorId] {
        &self.outputs
    }

    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    pub fn shape(&self, t: TensorId) -> Shape {
        self.shapes[t]
    }

    pub fn tensor_count(&self) -> usize {
        self.shapes.len()
    }

    pub fn mark_output(&mut self, t: TensorId) {
        self.outputs.push(t);
    }

    fn push(&mut self, shape: Shape, op: impl FnOnce(TensorId) -> Op) -> TensorId {
        let id = self.shapes.len();
        self.shapes.push(shape);
        self.ops.push(op(id));
        id
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv(&mut self, input: TensorId, weights: Vec<f32>, bias: Vec<f32>, out_channels: usize, kernel: usize, stride: usize, pad: [usize; 4], relu: bool) -> Result<TensorId> {
        let s = self.shapes[input];
        if weights.len() != out_channels * s.c * kernel * kernel || bias.len() != out_channels {
            return Err(Error::Unsupported(format!("conv on {s:?}: weights must be [{out_channels}, {}, {kernel}, {kernel}] with {out_channels} biases", s.c)));
        }
        let [t, l, b, r] = pad;
        if stride == 0 || kernel == 0 || s.h + t + b < kernel || s.w + l + r < kernel {
            return Err(Error::Unsupported(format!("conv on {s:?}: kernel {kernel}, stride {stride}, pad {pad:?} has no output")));
        }
        let out = Shape::new((s.h + t + b - kernel) / stride + 1, (s.w + l + r - kernel) / stride + 1, out_channels);
        Ok(self.push(out, |output| Op::Conv { input, output, weights, bias, kernel, stride, pad, relu }))
    }

    pub fn add(&mut self, a: TensorId, b: TensorId) -> Result<TensorId> {
        let (sa, sb) = (self.shapes[a], self.shapes[b]);
        if sa != sb {
            return Err(Error::Unsupported(format!("add of {sa:?} and {sb:?}")));
        }
        Ok(self.push(sa, |output| Op::Add { a, b, output }))
    }

    pub fn concat(&mut self, inputs: &[TensorId]) -> Result<TensorId> {
        let first = *inputs.first().ok_or_else(|| Error::Unsupported("concat of nothing".into()))?;
        let s = self.shapes[first];
        let mut c = 0;
        for &i in inputs {
            let si = self.shapes[i];
            if (si.h, si.w) != (s.h, s.w) {
                return Err(Error::Unsupported(format!("concat of {s:?} and {si:?}")));
            }
            c += si.c;
        }
        let inputs = inputs.to_vec();
        Ok(self.push(Shape::new(s.h, s.w, c), |output| Op::Concat { inputs, output }))
    }

    pub fn max_pool(&mut self, input: TensorId, kernel: usize, stride: usize, pad: usize) -> Result<TensorId> {
        let s = self.shapes[input];
        if stride == 0 || pad >= kernel || s.h + 2 * pad < kernel || s.w + 2 * pad < kernel {
            return Err(Error::Unsupported(format!("max pool on {s:?}: kernel {kernel}, stride {stride}, pad {pad}")));
        }
        let out = Shape::new((s.h + 2 * pad - kernel) / stride + 1, (s.w + 2 * pad - kernel) / stride + 1, s.c);
        Ok(self.push(out, |output| Op::MaxPool { input, output, kernel, stride, pad }))
    }

    pub fn upsample2x(&mut self, input: TensorId) -> TensorId {
        let s = self.shapes[input];
        self.push(Shape::new(s.h * 2, s.w * 2, s.c), |output| Op::Upsample2x { input, output })
    }

    /// Run the float reference on one HWC input, calling `observe` with every
    /// tensor as it is produced (the input first). Returns the outputs.
    pub fn run(&self, input: &[f32], mut observe: impl FnMut(TensorId, &[f32])) -> Result<Vec<Vec<f32>>> {
        let s = self.shapes[self.input];
        if input.len() != s.h * s.w * s.c {
            return Err(Error::Unsupported(format!("graph input is {s:?}, got {} values", input.len())));
        }
        let mut values: Vec<Option<Vec<f32>>> = vec![None; self.shapes.len()];
        observe(self.input, input);
        values[self.input] = Some(input.to_vec());
        let last_use = self.last_uses();
        for (step, op) in self.ops.iter().enumerate() {
            let get = |t: TensorId| values[t].as_deref().expect("graph ops are in execution order");
            let out = match op {
                Op::Conv { input, weights, bias, kernel, stride, pad, relu, output } => conv_f32(get(*input), self.shapes[*input], self.shapes[*output], weights, bias, *kernel, *stride, *pad, *relu),
                Op::Add { a, b, .. } => get(*a).iter().zip(get(*b)).map(|(x, y)| x + y).collect(),
                Op::Concat { inputs, output } => {
                    let so = self.shapes[*output];
                    let mut out = Vec::with_capacity(so.h * so.w * so.c);
                    for pixel in 0..so.h * so.w {
                        for &i in inputs {
                            let c = self.shapes[i].c;
                            out.extend_from_slice(&get(i)[pixel * c..(pixel + 1) * c]);
                        }
                    }
                    out
                }
                Op::MaxPool { input, output, kernel, stride, pad } => max_pool_f32(get(*input), self.shapes[*input], self.shapes[*output], *kernel, *stride, *pad),
                Op::Upsample2x { input, output } => upsample2x(get(*input), self.shapes[*input], self.shapes[*output]),
            };
            observe(op.output(), &out);
            values[op.output()] = Some(out);
            // Free what no later op reads, so a reference run at full size
            // holds only the live tensors.
            for t in op.inputs() {
                if last_use[t] == step && !self.outputs.contains(&t) {
                    values[t] = None;
                }
            }
        }
        Ok(self.outputs.iter().map(|&t| values[t].clone().expect("outputs are kept")).collect())
    }

    /// For every tensor, the index of the last op that reads it.
    pub(crate) fn last_uses(&self) -> Vec<usize> {
        let mut last = vec![0; self.shapes.len()];
        for (i, op) in self.ops.iter().enumerate() {
            for t in op.inputs() {
                last[t] = i;
            }
        }
        last
    }
}

#[allow(clippy::too_many_arguments)]
fn conv_f32(x: &[f32], si: Shape, so: Shape, w: &[f32], b: &[f32], k: usize, stride: usize, pad: [usize; 4], relu: bool) -> Vec<f32> {
    let mut out = vec![0f32; so.h * so.w * so.c];
    for oy in 0..so.h {
        for ox in 0..so.w {
            let acc = &mut out[(oy * so.w + ox) * so.c..(oy * so.w + ox + 1) * so.c];
            acc.copy_from_slice(b);
            for ky in 0..k {
                let y = (oy * stride + ky) as isize - pad[0] as isize;
                if y < 0 || y >= si.h as isize {
                    continue;
                }
                for kx in 0..k {
                    let xx = (ox * stride + kx) as isize - pad[1] as isize;
                    if xx < 0 || xx >= si.w as isize {
                        continue;
                    }
                    let px = &x[((y as usize) * si.w + xx as usize) * si.c..][..si.c];
                    for (o, a) in acc.iter_mut().enumerate() {
                        let wo = &w[o * si.c * k * k..];
                        *a += px.iter().enumerate().map(|(c, v)| v * wo[(c * k + ky) * k + kx]).sum::<f32>();
                    }
                }
            }
            if relu {
                acc.iter_mut().for_each(|v| *v = v.max(0.0));
            }
        }
    }
    out
}

fn max_pool_f32(x: &[f32], si: Shape, so: Shape, k: usize, stride: usize, pad: usize) -> Vec<f32> {
    let mut out = vec![f32::NEG_INFINITY; so.h * so.w * so.c];
    for oy in 0..so.h {
        for ox in 0..so.w {
            for ky in 0..k {
                for kx in 0..k {
                    let (y, xx) = ((oy * stride + ky) as isize - pad as isize, (ox * stride + kx) as isize - pad as isize);
                    if y < 0 || xx < 0 || y >= si.h as isize || xx >= si.w as isize {
                        continue;
                    }
                    for c in 0..si.c {
                        let v = x[((y as usize) * si.w + xx as usize) * si.c + c];
                        let o = &mut out[(oy * so.w + ox) * so.c + c];
                        *o = o.max(v);
                    }
                }
            }
        }
    }
    out
}

fn upsample2x<T: Copy>(x: &[T], si: Shape, so: Shape) -> Vec<T> {
    let mut out = Vec::with_capacity(so.h * so.w * so.c);
    for y in 0..so.h {
        for xx in 0..so.w {
            out.extend_from_slice(&x[((y / 2) * si.w + xx / 2) * si.c..][..si.c]);
        }
    }
    out
}

/// The value range every tensor took over a set of calibration inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    pub ranges: Vec<(f32, f32)>,
}

impl Calibration {
    /// Run the float reference over `inputs` and record each tensor's range.
    pub fn measure<'a>(graph: &Graph, inputs: impl IntoIterator<Item = &'a [f32]>) -> Result<Calibration> {
        let mut ranges = vec![(f32::INFINITY, f32::NEG_INFINITY); graph.tensor_count()];
        let mut seen = 0;
        for input in inputs {
            graph.run(input, |t, v| {
                let r = &mut ranges[t];
                for &x in v {
                    r.0 = r.0.min(x);
                    r.1 = r.1.max(x);
                }
            })?;
            seen += 1;
        }
        if seen == 0 {
            return Err(Error::Quantization("calibration needs at least one input".into()));
        }
        Ok(Calibration { ranges })
    }
}
