// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`GraphBuilder`] — the ergonomic surface brain's exporters use to assemble an
//! ONNX graph and serialize it to bytes. Wraps the owned [`Graph`] model and
//! handles encoding via prost.

use crate::graph::{Elem, Graph, Node, Tensor, TensorData, ValueInfo};
use prost::Message;

/// Default ONNX opset: 13 is the floor that provides per-axis
/// `QuantizeLinear`/`DequantizeLinear` (the `axis` attribute used for
/// per-output-channel weight quantization) and `Resize-13`.
pub const DEFAULT_OPSET: i64 = 13;
/// ONNX IR version paired with opset 13.
pub const DEFAULT_IR_VERSION: i64 = 8;

/// Assembles an ONNX [`Graph`] and serializes it. Add inputs/outputs/initializers
/// and nodes, then call [`GraphBuilder::finish`] (or [`GraphBuilder::finish_with`]).
pub struct GraphBuilder {
    graph: Graph,
}

impl GraphBuilder {
    pub fn new(name: &str) -> GraphBuilder {
        GraphBuilder { graph: Graph { name: name.to_string(), ..Default::default() } }
    }

    /// Register a typed FLOAT graph input with a static shape.
    pub fn input_f32(&mut self, name: &str, dims: &[i64]) -> &mut Self {
        self.graph.inputs.push(ValueInfo { name: name.to_string(), dims: dims.to_vec(), elem: Elem::F32 });
        self
    }

    /// Register a typed FLOAT graph output with a static shape.
    pub fn output_f32(&mut self, name: &str, dims: &[i64]) -> &mut Self {
        self.graph.outputs.push(ValueInfo { name: name.to_string(), dims: dims.to_vec(), elem: Elem::F32 });
        self
    }

    /// Add an FLOAT constant initializer. Panics if `data.len()` disagrees with `dims`.
    pub fn init_f32(&mut self, name: &str, dims: &[i64], data: Vec<f32>) -> &mut Self {
        self.push_init(name, dims, TensorData::F32(data))
    }

    /// Add an INT8 constant initializer (e.g. per-channel-quantized conv weights).
    pub fn init_i8(&mut self, name: &str, dims: &[i64], data: Vec<i8>) -> &mut Self {
        self.push_init(name, dims, TensorData::I8(data))
    }

    /// Add an INT64 constant initializer (e.g. a Resize `scales`/`sizes` helper).
    pub fn init_i64(&mut self, name: &str, dims: &[i64], data: Vec<i64>) -> &mut Self {
        self.push_init(name, dims, TensorData::I64(data))
    }

    fn push_init(&mut self, name: &str, dims: &[i64], data: TensorData) -> &mut Self {
        let want: i64 = dims.iter().product();
        assert_eq!(
            want as usize,
            data.len(),
            "initializer {name}: dims {dims:?} imply {want} elems but got {}",
            data.len()
        );
        self.graph.initializers.push(Tensor { name: name.to_string(), dims: dims.to_vec(), data });
        self
    }

    /// Append a node (build it with [`Node::new`] + `attr_*`).
    pub fn add(&mut self, node: Node) -> &mut Self {
        self.graph.nodes.push(node);
        self
    }

    /// Borrow the underlying owned graph (e.g. for inspection in tests).
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Serialize to ONNX bytes with the default opset/IR version.
    pub fn finish(&self) -> Vec<u8> {
        self.finish_with(DEFAULT_OPSET, DEFAULT_IR_VERSION)
    }

    /// Serialize to ONNX bytes with an explicit opset / IR version.
    pub fn finish_with(&self, opset: i64, ir_version: i64) -> Vec<u8> {
        self.graph.to_proto(opset, ir_version).encode_to_vec()
    }
}
