// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for compiling trained networks onto
// NPUs for its clients. If your team needs expertise in lowering detection
// graphs to fixed-function accelerators then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The YOLOv5 layer graph: the one description of the network that the fp32
//! reference forward ([`crate::model`]) executes and that an accelerator
//! back-end walks to lower it.
//!
//! The graph is a flat, topologically ordered list of [`Node`]s over numbered
//! values. Value `0` is the input image `[1, 3, H, W]`; node `i` produces value
//! `i + 1`. Every node is one of five operators, and every convolution carries
//! its activation, bias presence and its tensor name, so a consumer needs
//! nothing else but the [`crate::Weights`] to run or compile it.
//!
//! ## What the layout is
//!
//! The Ultralytics YOLOv5 v6 architecture as published: a stem, four
//! stride-2 convolutions each followed by a C3 block, an SPP-fast block, a
//! top-down then bottom-up PAN neck of C3 blocks, and a three-scale head of 1x1
//! convolutions. Two spellings follow the checkpoint this crate imports (an
//! NPU-oriented export) rather than the training code, and both are exactly
//! equivalent in fp32:
//!
//! * the stem is a 2x2 stride-2 convolution (a fixed space-to-depth, 3 -> 12
//!   channels, no bias, no activation) followed by a 3x3 convolution, i.e. the
//!   `Focus` form of the stem expressed with a single convolution;
//! * SPP-fast's three chained 5x5 max-pools are six chained 3x3 stride-1
//!   max-pools, tapped after the 2nd, 4th and 6th (receptive fields 5, 9, 13).
//!
//! ## Tensor names
//!
//! A convolution named `P` owns `P.weight` (`[cout, cin, k, k]`) and, when
//! `bias`, `P.bias` (`[cout]`). Batch norm is already folded into the
//! convolution in the checkpoints this crate loads, so no norm tensors exist.
//! The names are the module paths of the training code (`model.2.cv1.conv`,
//! `model.2.m.0.cv2.conv`, `model.24.m.0`), which is what makes importing from
//! such a checkpoint a 1:1 rename.

use crate::config::{Activation, Yolov5Config, ANCHORS_PER_SCALE, BOX_FIELDS};

/// A value id: `0` is the network input, node `i` produces value `i + 1`.
pub type ValueId = usize;

/// What follows a convolution's accumulate (+ bias).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Epilogue {
    None,
    /// The checkpoint's [`Activation`] (ReLU or SiLU).
    Act(Activation),
    /// The logistic function: the detection head's output nonlinearity, applied
    /// inside the network so its outputs are probabilities and offsets.
    Sigmoid,
}

/// One convolution: dense (`groups = 1`), square kernel, square stride, symmetric
/// zero padding.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvOp {
    /// Tensor-name prefix: `<name>.weight`, and `<name>.bias` when `bias`.
    pub name: String,
    pub cin: u32,
    pub cout: u32,
    pub k: u32,
    pub stride: u32,
    pub pad: u32,
    pub bias: bool,
    pub epilogue: Epilogue,
}

impl ConvOp {
    /// Weight tensor shape `[cout, cin, k, k]`.
    pub fn weight_shape(&self) -> [usize; 4] {
        [self.cout as usize, self.cin as usize, self.k as usize, self.k as usize]
    }

    pub fn weight_name(&self) -> String {
        format!("{}.weight", self.name)
    }

    pub fn bias_name(&self) -> String {
        format!("{}.bias", self.name)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Conv(ConvOp),
    /// Elementwise sum of the two inputs (a bottleneck's residual).
    Add,
    /// Channel-wise concatenation of the inputs, in order.
    Concat,
    /// Max-pool, square window and stride, symmetric padding (padding never wins).
    MaxPool { k: u32, stride: u32, pad: u32 },
    /// Nearest-neighbour 2x upsample.
    Upsample2x,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<ValueId>,
}

/// The whole network.
#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    /// In execution (topological) order.
    pub nodes: Vec<Node>,
    /// The three head output values, P3 (stride 8) first. Each is
    /// `[1, anchors * (5 + nc), H/stride, W/stride]` after the sigmoid.
    pub outputs: [ValueId; 3],
    /// Channels of every value, indexed by [`ValueId`] (`channels[0] == 3`).
    pub channels: Vec<u32>,
}

/// `ceil(x / 8) * 8`, the channel rounding of the scaled architecture.
fn round_channels(base: u32, width: f32) -> u32 {
    let scaled = (base as f32 * width).ceil() as u32;
    scaled.div_ceil(8) * 8
}

/// Bottleneck count of a C3 block with `n` bottlenecks at full depth.
fn round_depth(n: u32, depth: f32) -> u32 {
    if n > 1 {
        ((n as f32 * depth).round() as u32).max(1)
    } else {
        n
    }
}

struct Builder {
    nodes: Vec<Node>,
    channels: Vec<u32>,
    act: Activation,
}

impl Builder {
    fn push(&mut self, op: Op, inputs: Vec<ValueId>, channels: u32) -> ValueId {
        self.nodes.push(Node { op, inputs });
        self.channels.push(channels);
        self.nodes.len()
    }

    fn conv_ex(&mut self, name: &str, x: ValueId, cout: u32, (k, stride, pad): (u32, u32, u32), bias: bool, epilogue: Epilogue) -> ValueId {
        let op = ConvOp { name: name.to_string(), cin: self.channels[x], cout, k, stride, pad, bias, epilogue };
        self.push(Op::Conv(op), vec![x], cout)
    }

    /// A biased convolution followed by the network activation (`Conv` of the
    /// training code, batch norm folded).
    fn conv(&mut self, name: &str, x: ValueId, cout: u32, k: u32, stride: u32) -> ValueId {
        let act = self.act;
        self.conv_ex(name, x, cout, (k, stride, k / 2), true, Epilogue::Act(act))
    }

    fn add(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let c = self.channels[a];
        self.push(Op::Add, vec![a, b], c)
    }

    fn concat(&mut self, inputs: &[ValueId]) -> ValueId {
        let c = inputs.iter().map(|&i| self.channels[i]).sum();
        self.push(Op::Concat, inputs.to_vec(), c)
    }

    /// C3: two 1x1 branches, one through `n` bottlenecks, concatenated and
    /// fused by a third 1x1.
    fn c3(&mut self, name: &str, x: ValueId, c2: u32, n: u32, shortcut: bool) -> ValueId {
        let hidden = c2 / 2;
        let mut m = self.conv(&format!("{name}.cv1.conv"), x, hidden, 1, 1);
        for i in 0..n {
            let b1 = self.conv(&format!("{name}.m.{i}.cv1.conv"), m, hidden, 1, 1);
            let b2 = self.conv(&format!("{name}.m.{i}.cv2.conv"), b1, hidden, 3, 1);
            m = if shortcut { self.add(m, b2) } else { b2 };
        }
        let side = self.conv(&format!("{name}.cv2.conv"), x, hidden, 1, 1);
        let cat = self.concat(&[m, side]);
        self.conv(&format!("{name}.cv3.conv"), cat, c2, 1, 1)
    }

    /// SPP-fast with the pooling chain spelled as six 3x3 stride-1 pools.
    fn sppf(&mut self, name: &str, x: ValueId, c2: u32) -> ValueId {
        let hidden = self.channels[x] / 2;
        let y = self.conv(&format!("{name}.cv1.conv"), x, hidden, 1, 1);
        let mut taps = vec![y];
        let mut p = y;
        for i in 0..6 {
            p = self.push(Op::MaxPool { k: 3, stride: 1, pad: 1 }, vec![p], hidden);
            if i % 2 == 1 {
                taps.push(p);
            }
        }
        let cat = self.concat(&taps);
        self.conv(&format!("{name}.cv2.conv"), cat, c2, 1, 1)
    }

    fn up(&mut self, x: ValueId) -> ValueId {
        let c = self.channels[x];
        self.push(Op::Upsample2x, vec![x], c)
    }
}

impl Graph {
    /// Build the graph `cfg` describes.
    pub fn build(cfg: &Yolov5Config) -> Graph {
        let w = cfg.width_multiple;
        let d = cfg.depth_multiple;
        let ch = |base: u32| round_channels(base, w);
        let n = |base: u32| round_depth(base, d);
        let (c0, c1, c2, c3, c4) = (ch(64), ch(128), ch(256), ch(512), ch(1024));

        let mut b = Builder { nodes: Vec::new(), channels: vec![3], act: cfg.activation };

        // Backbone (layers 0..=9).
        let sp = b.conv_ex("model.0.convsp", 0, 12, (2, 2, 0), false, Epilogue::None);
        let x = b.conv("model.0.conv.conv", sp, c0, 3, 1);
        let x = b.conv("model.1.conv", x, c1, 3, 2);
        let x = b.c3("model.2", x, c1, n(3), true);
        let x = b.conv("model.3.conv", x, c2, 3, 2);
        let p3_src = b.c3("model.4", x, c2, n(6), true);
        let x = b.conv("model.5.conv", p3_src, c3, 3, 2);
        let p4_src = b.c3("model.6", x, c3, n(9), true);
        let x = b.conv("model.7.conv", p4_src, c4, 3, 2);
        let x = b.c3("model.8", x, c4, n(3), true);
        let x = b.sppf("model.9", x, c4);

        // Neck (layers 10..=23).
        let l10 = b.conv("model.10.conv", x, c3, 1, 1);
        let u = b.up(l10);
        let cat = b.concat(&[u, p4_src]);
        let x = b.c3("model.13", cat, c3, n(3), false);
        let l14 = b.conv("model.14.conv", x, c2, 1, 1);
        let u = b.up(l14);
        let cat = b.concat(&[u, p3_src]);
        let p3 = b.c3("model.17", cat, c2, n(3), false);
        let x = b.conv("model.18.conv", p3, c2, 3, 2);
        let cat = b.concat(&[x, l14]);
        let p4 = b.c3("model.20", cat, c3, n(3), false);
        let x = b.conv("model.21.conv", p4, c3, 3, 2);
        let cat = b.concat(&[x, l10]);
        let p5 = b.c3("model.23", cat, c4, n(3), false);

        // Head (layer 24): one 1x1 convolution per scale, sigmoid applied.
        let head_c = ANCHORS_PER_SCALE as u32 * (BOX_FIELDS + cfg.nc);
        let mut outputs = [0; 3];
        for (i, feat) in [p3, p4, p5].into_iter().enumerate() {
            outputs[i] = b.conv_ex(&format!("model.24.m.{i}"), feat, head_c, (1, 1, 0), true, Epilogue::Sigmoid);
        }

        Graph { nodes: b.nodes, outputs, channels: b.channels }
    }

    /// Every convolution, in execution order.
    pub fn convs(&self) -> impl Iterator<Item = &ConvOp> {
        self.nodes.iter().filter_map(|n| match &n.op {
            Op::Conv(c) => Some(c),
            _ => None,
        })
    }

    /// The checkpoint's tensor manifest: every `(name, shape)` the graph needs,
    /// in execution order.
    pub fn param_list(&self) -> Vec<(String, Vec<usize>)> {
        let mut v = Vec::new();
        for c in self.convs() {
            v.push((c.weight_name(), c.weight_shape().to_vec()));
            if c.bias {
                v.push((c.bias_name(), vec![c.cout as usize]));
            }
        }
        v
    }

    /// The spatial size of every value for an `h x w` input (multiples of 32
    /// are required so the three scales line up).
    pub fn spatial(&self, h: u32, w: u32) -> Vec<(u32, u32)> {
        let mut s = vec![(h, w)];
        for n in &self.nodes {
            let (ih, iw) = s[n.inputs[0]];
            s.push(match &n.op {
                Op::Conv(c) => ((ih + 2 * c.pad - c.k) / c.stride + 1, (iw + 2 * c.pad - c.k) / c.stride + 1),
                Op::MaxPool { k, stride, pad } => ((ih + 2 * pad - k) / stride + 1, (iw + 2 * pad - k) / stride + 1),
                Op::Upsample2x => (ih * 2, iw * 2),
                Op::Add | Op::Concat => (ih, iw),
            });
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Yolov5Config;

    fn census(g: &Graph) -> [usize; 5] {
        let mut c = [0; 5];
        for n in &g.nodes {
            c[match n.op {
                Op::Conv(_) => 0,
                Op::Add => 1,
                Op::Concat => 2,
                Op::MaxPool { .. } => 3,
                Op::Upsample2x => 4,
            }] += 1;
        }
        c
    }

    /// The operator census of the reference export this graph must reproduce:
    /// 61 Conv, 7 Add, 13 Concat, 6 MaxPool, 2 Resize (and 3 Sigmoid, fused
    /// into the head convolutions here).
    #[test]
    fn yolov5s_matches_the_reference_operator_census() {
        let g = Graph::build(&Yolov5Config::yolov5s(Activation::Relu));
        assert_eq!(census(&g), [61, 7, 13, 6, 2]);
        let sig = g.convs().filter(|c| c.epilogue == Epilogue::Sigmoid).count();
        assert_eq!(sig, 3);
    }

    #[test]
    fn yolov5s_head_outputs_are_the_three_255_channel_scales() {
        let cfg = Yolov5Config::yolov5s(Activation::Relu);
        let g = Graph::build(&cfg);
        for o in g.outputs {
            assert_eq!(g.channels[o], 255);
        }
        let s = g.spatial(640, 640);
        assert_eq!(g.outputs.map(|o| s[o]), [(80, 80), (40, 40), (20, 20)]);
        let s = g.spatial(320, 192);
        assert_eq!(g.outputs.map(|o| s[o]), [(40, 24), (20, 12), (10, 6)]);
    }

    #[test]
    fn the_stem_is_a_pad_free_space_to_depth_conv_without_bias() {
        let g = Graph::build(&Yolov5Config::yolov5s(Activation::Relu));
        let stem = g.convs().next().unwrap();
        assert_eq!((stem.name.as_str(), stem.cin, stem.cout, stem.k, stem.stride, stem.pad, stem.bias), ("model.0.convsp", 3, 12, 2, 2, 0, false));
        assert_eq!(stem.epilogue, Epilogue::None);
    }

    #[test]
    fn the_manifest_names_are_unique_and_scale_with_width() {
        let g = Graph::build(&Yolov5Config::yolov5s(Activation::Relu));
        let names: std::collections::BTreeSet<_> = g.param_list().into_iter().map(|(n, _)| n).collect();
        assert_eq!(names.len(), 61 * 2 - 1);
        let mut wide = Yolov5Config::yolov5s(Activation::Relu);
        wide.width_multiple = 1.0;
        let first_conv = |g: &Graph| g.convs().nth(1).unwrap().cout;
        assert_eq!(first_conv(&Graph::build(&wide)), 64);
        assert_eq!(first_conv(&g), 32);
    }
}
