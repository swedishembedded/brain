// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for validating edge detectors against
// a trusted fp32 reference for its clients. If your team needs expertise in
// reference implementations and numerical parity gates for accelerator ports then
// you can procure our services by sending an email to info@swedishembedded.com.

//! The fp32 forward of a YOLOv5 [`Graph`] through `gpu_core`: the host reference
//! an accelerator port is validated against.
//!
//! Each graph node becomes one stage of the shared `vision` blocks (`Conv`,
//! `MaxPool`, `Up`, `Cat`, `Acc`), built for one fixed input size and run in
//! graph order; every convolution is a raw biased conv with its epilogue
//! (activation, or the head's sigmoid), so no batch-norm kernels are involved.
//! The network is fully convolutional: any input height and width that are
//! multiples of 32 work, and the weights do not change with them.
//!
//! Nothing outside this module touches a device. [`crate::Checkpoint`],
//! [`crate::decode`] and the NMS it ends in run without one.

use std::sync::OnceLock;

use gpu_core::{DeviceBuffer, Gpu};
use paramstore::{ParamStore, Role};
use vision::{Acc, Act, Cat, Conv, ConvKernelIds, ConvNames, ConvSpec, Ctx, MaxPool, Norm, PoolSpec, Shape, Up};

use crate::config::{Activation, Yolov5Config, STRIDES};
use crate::decode::{self, HeadMap};
use crate::error::Error;
use crate::graph::{Epilogue, Graph, Op};
use crate::weights::{self, Weights};

use imaging::nms::Detection;

/// The kernels this model dispatches, resolved by name by the shared blocks.
/// ReLU is `leaky_relu` at slope 0 (see `vision::Act::Relu`).
pub const PIPELINES: &[(&str, &str)] = &[
    ("conv2d", kernels::CONV2D),
    ("conv_bias", kernels::CONV_BIAS),
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
    ("leaky_relu", kernels::LEAKY_RELU),
    ("leaky_relu_bwd", kernels::LEAKY_RELU_BWD),
    ("silu", kernels::SILU),
    ("silu_bwd", kernels::SILU_BWD),
    ("sigmoid", kernels::SIGMOID),
    ("sigmoid_bwd", kernels::SIGMOID_BWD),
    ("maxpool2d", kernels::MAXPOOL2D),
    ("upsample2", kernels::UPSAMPLE2),
    ("concat2", kernels::CONCAT2),
    ("add2", kernels::ADD2),
];

fn ids() -> &'static ConvKernelIds {
    static IDS: OnceLock<ConvKernelIds> = OnceLock::new();
    IDS.get_or_init(|| ConvKernelIds::resolve(PIPELINES))
}

/// Grey letterbox fill (114/255), the convention the weights were trained with.
const LETTERBOX_PAD: f32 = 114.0 / 255.0;


enum Stage {
    Conv(Box<Conv>),
    Add(Acc),
    /// A left fold of two-input concats (`Cat` joins two maps).
    Cat(Vec<Cat>),
    Pool(MaxPool),
    Up(Up),
}

/// The three head maps of one forward, post-sigmoid, batch 1.
#[derive(Clone, Debug)]
pub struct HeadOutputs {
    /// `[anchors * (5 + nc), h, w]` per scale, P3 first.
    pub maps: [Vec<f32>; 3],
    /// `(h, w)` per scale.
    pub dims: [(usize, usize); 3],
}

impl HeadOutputs {
    pub fn heads(&self) -> [HeadMap<'_>; 3] {
        std::array::from_fn(|i| HeadMap { data: &self.maps[i], h: self.dims[i].0, w: self.dims[i].1 })
    }
}

/// A YOLOv5 detector resident on a device, for one input size.
pub struct Yolov5 {
    cfg: Yolov5Config,
    graph: Graph,
    gpu: Gpu,
    ps: ParamStore,
    stages: Vec<Stage>,
    /// Every graph value's device buffer, indexed by [`crate::graph::ValueId`].
    values: Vec<DeviceBuffer>,
    shapes: Vec<Shape>,
    height: u32,
    width: u32,
}

fn conv_spec(op: &crate::graph::ConvOp) -> ConvSpec {
    let act = match op.epilogue {
        Epilogue::None => Act::None,
        Epilogue::Act(Activation::Relu) => Act::Relu,
        Epilogue::Act(Activation::Silu) => Act::Silu,
        Epilogue::Sigmoid => Act::Sigmoid,
    };
    let spec = ConvSpec { cout: op.cout, k: op.k, stride: op.stride, pad: op.pad, groups: 1, dilation: 1, norm: Norm::None, act, bias: false };
    if op.bias {
        spec.with_bias()
    } else {
        spec
    }
}

impl Yolov5 {
    /// Load a brain-format checkpoint on the device the process selected
    /// (`--device`), for a square `input` (`0` = the checkpoint's own).
    pub fn load(path: &std::path::Path, input: u32) -> Result<Yolov5, Error> {
        let ckpt = weights::open(path)?;
        let side = if input == 0 { ckpt.config.input } else { input };
        Yolov5::new_on(Gpu::new(PIPELINES), ckpt.config, &ckpt.weights, side, side)
    }

    /// Build on `gpu` for a `height x width` input (each a positive multiple of 32).
    pub fn new_on(gpu: Gpu, cfg: Yolov5Config, weights: &Weights, height: u32, width: u32) -> Result<Yolov5, Error> {
        let max_stride = STRIDES[2];
        if height == 0 || width == 0 || !height.is_multiple_of(max_stride) || !width.is_multiple_of(max_stride) {
            return Err(Error::Input(format!("input {height}x{width} must be positive multiples of {max_stride}")));
        }
        let graph = Graph::build(&cfg);
        let roles = graph.param_list().into_iter().map(|(n, s)| (n, s.iter().product(), Role::Frozen)).collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, weights.tensors());

        let ctx = Ctx::new(&gpu, ids());
        let img_shape = Shape::new(1, 3, height, width);
        let img = ctx.act(img_shape.numel());
        let mut shapes = vec![img_shape];
        let mut values = vec![img.clone()];
        let mut stages = Vec::with_capacity(graph.nodes.len());
        for node in &graph.nodes {
            let x = shapes[node.inputs[0]];
            let (stage, shape, out) = match &node.op {
                Op::Conv(op) => {
                    let conv = Conv::with_names(&ctx, &op.name, ConvNames::torch_flat(&op.name), x, conv_spec(op), false);
                    let (shape, out) = (conv.out_shape, conv.out().clone());
                    (Stage::Conv(Box::new(conv)), shape, out)
                }
                Op::Add => {
                    let acc = Acc::new(&ctx, x);
                    let out = acc.out.clone();
                    (Stage::Add(acc), x, out)
                }
                Op::Concat => {
                    let mut cats: Vec<Cat> = Vec::new();
                    let mut acc_shape = x;
                    for &i in &node.inputs[1..] {
                        let cat = Cat::new(&ctx, acc_shape, shapes[i]);
                        acc_shape = cat.out_shape();
                        cats.push(cat);
                    }
                    let out = cats.last().expect("a concat has at least two inputs").out.clone();
                    (Stage::Cat(cats), acc_shape, out)
                }
                Op::MaxPool { k, stride, pad } => {
                    let pool = MaxPool::new(&ctx, x, PoolSpec::new(*k, *stride, *pad));
                    let (shape, out) = (pool.out_shape, pool.out().clone());
                    (Stage::Pool(pool), shape, out)
                }
                Op::Upsample2x => {
                    let up = Up::new(&ctx, x);
                    let (shape, out) = (up.out_shape, up.out.clone());
                    (Stage::Up(up), shape, out)
                }
            };
            stages.push(stage);
            shapes.push(shape);
            values.push(out);
        }
        Ok(Yolov5 { cfg, graph, gpu, ps, stages, values, shapes, height, width })
    }

    pub fn config(&self) -> &Yolov5Config {
        &self.cfg
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The input `(height, width)` this instance was built for.
    pub fn input_size(&self) -> (u32, u32) {
        (self.height, self.width)
    }

    /// Run the network on a CHW fp32 image `[3, height, width]` (RGB, `0..1`)
    /// and read the three post-sigmoid head maps back.
    pub fn forward(&self, chw: &[f32]) -> Result<HeadOutputs, Error> {
        let want = 3 * (self.height * self.width) as usize;
        if chw.len() != want {
            return Err(Error::Input(format!("image has {} values, expected {want} (3 x {} x {})", chw.len(), self.height, self.width)));
        }
        self.gpu.write_f32(&self.values[0], chw);
        let ctx = Ctx::new(&self.gpu, ids());
        {
            let _pass = self.gpu.pass_scope();
            for (node, stage) in self.graph.nodes.iter().zip(&self.stages) {
                let input = |k: usize| &self.values[node.inputs[k]];
                match stage {
                    Stage::Conv(c) => c.forward(&ctx, &self.ps, input(0)),
                    Stage::Add(a) => a.add(&ctx, input(0), input(1)),
                    Stage::Cat(cats) => {
                        let mut left = input(0);
                        for (k, cat) in cats.iter().enumerate() {
                            cat.forward(&ctx, left, input(k + 1));
                            left = &cat.out;
                        }
                    }
                    Stage::Pool(p) => p.forward(&ctx, input(0)),
                    Stage::Up(u) => u.forward(&ctx, input(0)),
                }
            }
        }
        let read = |scale: usize| {
            let id = self.graph.outputs[scale];
            self.gpu.read(&self.values[id], self.shapes[id].numel() as usize)
        };
        let dims = std::array::from_fn(|i| {
            let s = self.shapes[self.graph.outputs[i]];
            (s.h as usize, s.w as usize)
        });
        Ok(HeadOutputs { maps: std::array::from_fn(read), dims })
    }

    /// Detect objects in one interleaved-RGB HWC image (`0..1`, `w0 * h0 * 3`
    /// values): letterbox to the (square) input, forward, decode, NMS, and map
    /// the boxes back to original-image pixels.
    pub fn detect(&self, hwc: &[f32], w0: u32, h0: u32, conf_thresh: f32, iou_thresh: f32) -> Result<Vec<Detection>, Error> {
        if self.height != self.width {
            return Err(Error::Input(format!("detect letterboxes into a square input; this instance is {}x{}", self.height, self.width)));
        }
        if w0 == 0 || h0 == 0 || hwc.len() != (w0 as usize) * (h0 as usize) * 3 {
            return Err(Error::Input(format!("image of {w0}x{h0} needs {} values, got {}", (w0 as usize) * (h0 as usize) * 3, hwc.len())));
        }
        let (chw, lb) = imaging::letterbox_rgb(hwc, w0, h0, self.height, LETTERBOX_PAD);
        let out = self.forward(&chw)?;
        decode::detections(&self.cfg, out.heads(), &lb, (w0, h0), conf_thresh, iou_thresh, decode::MAX_DETECTIONS)
    }
}
