// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The released networks' geometry, one table per graph.
//!
//! Every OpenSeeFace network is the same shape: a MobileNetV3 encoder (a
//! stem, stages of depthwise-separable `ds`, inverted-residual `ir` and
//! pointwise `cn` blocks, squeeze-excite where the release has it), a
//! decoder that upsamples and concatenates encoder skips, and a grouped
//! head. What varies is in these tables.
//!
//! The tables are the pinned release's facts - read off its graphs' node
//! attributes (kernel, stride, group, fused activation) and initializer
//! shapes - and the import walk re-asserts every entry against the file it
//! loads, so a different release fails at import rather than computing a
//! plausible wrong answer.

use vision::Act;

/// One encoder block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    /// Depthwise k x k, optional SE, pointwise projection.
    Ds,
    /// Pointwise expansion, depthwise k x k, optional SE, linear projection.
    Ir,
    /// One pointwise conv with an activation.
    Cn,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Block {
    pub kind: Kind,
    pub k: u32,
    pub stride: u32,
    /// The expanded (depthwise) width; for `Ds` the input width.
    pub mid: u32,
    pub out: u32,
    /// The activation of the expansion and depthwise convs (`Cn`: its conv).
    pub act: Act,
    /// The squeeze-excite bottleneck width, 0 for none.
    pub se: u32,
}

impl Block {
    pub const fn ds(k: u32, stride: u32, out: u32, act: Act, se: u32) -> Block {
        Block { kind: Kind::Ds, k, stride, mid: 0, out, act, se }
    }
    pub const fn ir(k: u32, stride: u32, mid: u32, out: u32, act: Act, se: u32) -> Block {
        Block { kind: Kind::Ir, k, stride, mid, out, act, se }
    }
    pub const fn cn(out: u32, act: Act) -> Block {
        Block { kind: Kind::Cn, k: 1, stride: 1, mid: 0, out, act, se: 0 }
    }
}

/// One decoder stage: upsample to the skip's size (bilinear, corners
/// aligned), concatenate `[skip, upsampled]`, depthwise 3x3, pointwise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Up {
    /// The encoder stage whose output is the skip.
    pub skip: usize,
    pub dw_act: Act,
    pub out: u32,
}

/// The grouped head: a grouped 3x3 widening to `mid`, then a grouped 1x1
/// down to the output channels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Head {
    pub groups: u32,
    pub mid: u32,
    pub mid_act: Act,
    pub out_groups: u32,
    pub out: u32,
    pub out_act: Act,
}

/// One released network.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NetConfig {
    /// The release's file name.
    pub file: &'static str,
    /// Input side, pixels (square).
    pub input: u32,
    pub stem: u32,
    pub stem_act: Act,
    pub stages: &'static [&'static [Block]],
    pub ups: &'static [Up],
    pub head: Head,
    /// The detector's 3x3 stride-1 max-pool over the output, for peak
    /// picking.
    pub peak_pool: bool,
}

use Act::{HardSwish, Relu, Relu6};

/// The face detector: a 2-channel 56x56 map, face-centre likelihood and box
/// radius.
pub const DETECTOR: NetConfig = NetConfig {
    file: "mnv3_detection_opt.onnx",
    input: 224,
    stem: 8,
    stem_act: HardSwish,
    stages: &[
        &[Block::ds(3, 1, 8, Relu, 0)],
        &[Block::ir(3, 2, 32, 8, Relu, 0), Block::ir(3, 1, 24, 8, Relu, 0)],
        &[Block::ir(5, 2, 24, 8, Relu, 8), Block::ir(5, 1, 24, 8, Relu, 8), Block::ir(5, 1, 24, 8, Relu, 8)],
        &[Block::ir(3, 2, 48, 8, HardSwish, 0), Block::ir(3, 1, 24, 8, HardSwish, 0), Block::ir(3, 1, 24, 8, HardSwish, 0), Block::ir(3, 1, 24, 8, HardSwish, 0)],
        &[Block::ir(3, 1, 48, 16, HardSwish, 16), Block::ir(3, 1, 96, 16, HardSwish, 24)],
        &[Block::ir(5, 2, 96, 16, HardSwish, 24), Block::ir(5, 1, 96, 16, HardSwish, 24), Block::ir(5, 1, 96, 16, HardSwish, 24)],
        &[Block::cn(96, HardSwish)],
    ],
    ups: &[Up { skip: 4, dw_act: Act::None, out: 256 }, Up { skip: 2, dw_act: Act::None, out: 128 }, Up { skip: 1, dw_act: Act::None, out: 64 }],
    head: Head { groups: 64, mid: 256, mid_act: Act::None, out_groups: 2, out: 2, out_act: Relu6 },
    peak_pool: true,
};

/// The fastest landmark network (`lm_model0`): 66 heatmaps and 132 offset
/// maps at 28x28.
pub const LANDMARKS_FAST: NetConfig = NetConfig {
    file: "lm_model0_opt.onnx",
    input: 224,
    stem: 8,
    stem_act: HardSwish,
    stages: &[
        &[Block::ds(3, 2, 8, Relu, 8)],
        &[Block::ir(3, 2, 40, 16, Relu, 0), Block::ir(3, 1, 56, 16, Relu, 0)],
        &[Block::ir(5, 2, 64, 24, HardSwish, 16), Block::ir(5, 1, 144, 24, HardSwish, 40), Block::ir(5, 1, 144, 24, HardSwish, 40)],
        &[Block::ir(5, 1, 72, 24, HardSwish, 24), Block::ir(5, 1, 72, 24, HardSwish, 24)],
        &[Block::ir(5, 2, 144, 48, HardSwish, 40), Block::ir(5, 1, 288, 48, HardSwish, 72), Block::ir(5, 1, 288, 48, HardSwish, 72)],
        &[Block::cn(288, HardSwish)],
    ],
    ups: &[Up { skip: 2, dw_act: Relu6, out: 256 }, Up { skip: 1, dw_act: Relu6, out: 198 }],
    head: Head { groups: 198, mid: 792, mid_act: Relu6, out_groups: 3, out: 198, out_act: Relu6 },
    peak_pool: false,
};

/// The highest-quality landmark network (`lm_model3`, the reference
/// tracker's default).
pub const LANDMARKS_BEST: NetConfig = NetConfig {
    file: "lm_model3_opt.onnx",
    input: 224,
    stem: 16,
    stem_act: HardSwish,
    stages: &[
        &[Block::ds(3, 1, 16, Relu, 0)],
        &[Block::ir(3, 2, 64, 24, Relu, 0), Block::ir(3, 1, 72, 24, Relu, 0)],
        &[Block::ir(5, 2, 72, 40, Relu, 24), Block::ir(5, 1, 120, 40, Relu, 32), Block::ir(5, 1, 120, 40, Relu, 32)],
        &[Block::ir(3, 2, 240, 80, HardSwish, 0), Block::ir(3, 1, 200, 80, HardSwish, 0), Block::ir(3, 1, 184, 80, HardSwish, 0), Block::ir(3, 1, 184, 80, HardSwish, 0)],
        &[Block::ir(3, 1, 480, 112, HardSwish, 120), Block::ir(3, 1, 672, 112, HardSwish, 168)],
        &[Block::ir(5, 2, 672, 160, HardSwish, 168), Block::ir(5, 1, 960, 160, HardSwish, 240), Block::ir(5, 1, 960, 160, HardSwish, 240)],
        &[Block::cn(960, HardSwish)],
    ],
    ups: &[Up { skip: 4, dw_act: Relu6, out: 256 }, Up { skip: 2, dw_act: Relu6, out: 198 }],
    head: Head { groups: 198, mid: 792, mid_act: Relu6, out_groups: 3, out: 198, out_act: Relu6 },
    peak_pool: false,
};
