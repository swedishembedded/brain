// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The device bytes a graph of this crate's units allocates when built, summed
//! by walking the same shapes the build walks.
//!
//! Each unit's buffers are decided where the unit is constructed
//! (`Conv::with_names`, `BatchNorm::new`, ...), so the count for one unit lives
//! here beside them rather than in every model that wants to price a build: a
//! model's own `device_bytes` replays its construction on a [`Footprint`] and
//! the per-unit accounting stays in one place. Only buffers an inference
//! build allocates eagerly are counted; the lazily created backward and
//! im2col scratch is not, because nothing here runs them.
//!
//! Swedish Embedded AB implements solutions for fitting vision pipelines onto
//! the accelerators a machine really has for its clients. If your team needs
//! expertise in device memory planning then you can procure our services by
//! sending an email to info@swedishembedded.com.

use crate::blocks::{Act, ConvSpec, Norm, PoolSpec};
use crate::net::Shape;

/// Words (f32) a `BatchNorm`'s statistics scratch holds per channel: `mean`,
/// `var`, `mv`, `gb`, `mvg`, `bp` (`bn::BnCore::new`).
const BN_CORE_WORDS_PER_CHANNEL: u64 = 1 + 1 + 2 + 2 + 3 + 5;

/// Running total of the device words a walk over a graph's units allocates.
#[derive(Clone, Copy, Debug, Default)]
pub struct Footprint {
    words: u64,
}

impl Footprint {
    pub fn new() -> Footprint {
        Footprint::default()
    }

    /// `n` f32 words of any buffer the graph owns outside a unit (an input, a
    /// residual sum, a head's sigmoid output).
    pub fn buffer(&mut self, n: u64) {
        self.words += n;
    }

    /// A buffer holding a whole `shape` map.
    pub fn map(&mut self, shape: Shape) {
        self.buffer(shape.numel() as u64);
    }

    /// A `Conv` over `x` with `spec`; returns its output shape.
    pub fn conv(&mut self, x: Shape, spec: ConvSpec) -> Shape {
        let out = spec.out_shape(x);
        let on = out.numel() as u64;
        let raw = spec.norm == Norm::None;
        // `conv_out` is the unit's output when there is no norm and is a
        // one-word placeholder otherwise; `act` the reverse for a raw conv
        // with no activation.
        self.buffer(if raw { on } else { 1 });
        self.buffer(if raw && spec.act == Act::None { 1 } else { on });
        self.buffer(2 * out.c as u64);
        self.buffer(BN_CORE_WORDS_PER_CHANNEL * out.c as u64);
        out
    }

    /// A `BatchNorm` over `shape`.
    pub fn batchnorm(&mut self, shape: Shape) {
        self.buffer(BN_CORE_WORDS_PER_CHANNEL * shape.c as u64);
        self.map(shape);
    }

    /// A `PReLU` over `shape`.
    pub fn prelu(&mut self, shape: Shape) {
        self.buffer(2 * shape.numel() as u64);
    }

    /// A `MaxPool` over `x`; returns its output shape.
    pub fn maxpool(&mut self, x: Shape, spec: PoolSpec) -> Shape {
        let out = spec.out_shape(x);
        self.buffer(2 * out.numel() as u64 + x.numel() as u64);
        out
    }

    /// An `AvgPool` halving `x`; returns its output shape.
    pub fn avgpool_half(&mut self, x: Shape) -> Shape {
        let out = Shape::new(x.n, x.c, x.h / 2, x.w / 2);
        self.buffer(out.numel() as u64 + x.numel() as u64);
        out
    }

    /// The accumulated bytes (f32 words times four).
    pub fn bytes(&self) -> u64 {
        self.words * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bn_relu(cout: u32) -> ConvSpec {
        ConvSpec { cout, k: 3, stride: 1, pad: 1, groups: 1, dilation: 1, norm: Norm::Bn, act: Act::Relu, bias: false }
    }

    /// A normed, activated conv keeps a one-word placeholder for the raw
    /// output and a full map for the activation, plus the per-channel scratch.
    #[test]
    fn a_normed_conv_charges_its_activation_map_and_per_channel_scratch() {
        let mut f = Footprint::new();
        let out = f.conv(Shape::new(1, 3, 8, 8), bn_relu(4));
        assert_eq!(out, Shape::new(1, 4, 8, 8));
        let on = 4 * 8 * 8;
        assert_eq!(f.bytes(), (1 + on + 2 * 4 + 14 * 4) * 4);
    }

    /// A raw conv with no activation aliases its output to `conv_out`, so it
    /// holds one map, not two.
    #[test]
    fn a_raw_conv_without_activation_holds_one_map() {
        let mut f = Footprint::new();
        let spec = ConvSpec { norm: Norm::None, act: Act::None, ..bn_relu(4) };
        f.conv(Shape::new(1, 3, 8, 8), spec);
        let on = 4 * 8 * 8;
        assert_eq!(f.bytes(), (on + 1 + 2 * 4 + 14 * 4) * 4);
    }
}
