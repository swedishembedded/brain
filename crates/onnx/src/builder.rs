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

    /// Register a typed INT64 graph input with a static shape (e.g. token ids).
    pub fn input_i64(&mut self, name: &str, dims: &[i64]) -> &mut Self {
        self.graph.inputs.push(ValueInfo { name: name.to_string(), dims: dims.to_vec(), elem: Elem::I64 });
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

    /// Append a 1-D `ConvTranspose`: registers the weight (`[Cin, Cout/group, K]`,
    /// the brain `audio::conv` layout) and optional bias (`[Cout]`) as
    /// initializers, then the node. `node_name` names both the op and its
    /// weight/bias initializers; the output tensor is `out`. See
    /// [`crate::conv::ConvTranspose1d`] for the shape/padding conventions.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose1d(
        &mut self,
        node_name: &str,
        x: &str,
        out: &str,
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
        c: &crate::conv::ConvTranspose1d,
    ) -> &mut Self {
        let cout_g = c.cout / c.groups;
        let wname = format!("{node_name}.weight");
        self.init_f32(&wname, &[c.cin as i64, cout_g as i64, c.k as i64], weight);
        let mut inputs = vec![x.to_string(), wname];
        if let Some(b) = bias {
            let bname = format!("{node_name}.bias");
            self.init_f32(&bname, &[c.cout as i64], b);
            inputs.push(bname);
        }
        let in_refs: Vec<&str> = inputs.iter().map(|s| s.as_str()).collect();
        let node = crate::conv::conv_transpose_node(
            node_name,
            &in_refs,
            out,
            &[c.k as i64],
            &[c.stride as i64],
            &[c.pad_begin as i64, c.pad_end as i64],
            &[c.dilation as i64],
            &[c.output_padding as i64],
            c.groups as i64,
        );
        self.add(node)
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

    /// Write the model to `model_path` plus a sidecar `<model_path>.data` holding
    /// every initializer larger than `threshold` bytes (ONNX external data). This
    /// keeps the `.onnx` proto small enough to parse (under protobuf's 2GB limit)
    /// for multi-GB models. Read it back with a file-based loader (so the reader
    /// resolves the sidecar relative to the model directory).
    pub fn finish_external(&self, model_path: &str, threshold: usize) -> std::io::Result<()> {
        self.finish_external_with(model_path, threshold, DEFAULT_OPSET, DEFAULT_IR_VERSION)
    }

    pub fn finish_external_with(
        &self,
        model_path: &str,
        threshold: usize,
        opset: i64,
        ir_version: i64,
    ) -> std::io::Result<()> {
        let p = std::path::Path::new(model_path);
        let data_name = format!("{}.data", p.file_name().and_then(|s| s.to_str()).unwrap_or("model.onnx"));
        let (model, sidecar) = self.graph.to_proto_external(opset, ir_version, threshold, &data_name);
        std::fs::write(model_path, model.encode_to_vec())?;
        std::fs::write(p.with_file_name(&data_name), sidecar)?;
        Ok(())
    }
}
