// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The one seam between the tracker and whatever runs a network.
//!
//! The tracker needs exactly this from a network: run it on a normalized
//! input, hand back its output map (and the detector's peak pool). Two
//! executors implement it: [`crate::Net`] on brain's own kernels (any
//! `gpu_core` backend: the CPU JIT, wgpu, Vulkan, CUDA), and
//! [`crate::openvino::OvNet`], the same network exported by
//! [`crate::export`] and compiled by OpenVINO for an Intel NPU, GPU or CPU.
//! Both are parity-tested against the same goldens, so swapping one for the
//! other changes where the work runs, not what it computes.

use crate::config::NetConfig;

/// A network's outputs for one input.
#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    /// `[C, H, W]` flattened.
    pub map: Vec<f32>,
    /// The detector's 3x3 max-pooled map, same shape; `None` for landmarks.
    pub pooled: Option<Vec<f32>>,
}

/// Something that runs one released network.
pub trait Infer {
    /// Which network.
    fn config(&self) -> &NetConfig;
    /// Run on one normalized `[1, 3, input, input]` NCHW tensor.
    fn infer(&mut self, x: &[f32]) -> Result<Output, String>;
    /// Where it runs, as a person would want it reported.
    fn device(&self) -> String;
}

impl Infer for crate::model::Net {
    fn config(&self) -> &NetConfig {
        &self.cfg
    }

    fn infer(&mut self, x: &[f32]) -> Result<Output, String> {
        self.forward(x);
        Ok(Output { map: self.read_output(), pooled: self.read_pooled() })
    }

    fn device(&self) -> String {
        format!("brain {}", self.gpu().kind())
    }
}
