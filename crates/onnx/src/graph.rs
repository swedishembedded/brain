// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! An owned, ergonomic ONNX graph model, decoupled from the prost wire types in
//! [`crate::onnx`]. Brain's exporters build against these types; [`Graph::to_proto`]
//! bridges to the wire representation for serialization.
//!
//! The split is deliberate: keeping a clean owned model (rather than poking the
//! generated `*Proto` structs directly) is what would let a future ONNX *import*
//! frontend reuse the same vocabulary via a `from_proto` (not implemented now —
//! only the export/serialize direction exists today).

use crate::onnx as p;

/// Element type for a graph value or initializer. Maps to ONNX `TensorProto::DataType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    F32,
    I8,
    I32,
    I64,
    U8,
}

impl Elem {
    fn data_type(self) -> i32 {
        use p::tensor_proto::DataType::*;
        match self {
            Elem::F32 => Float as i32,
            Elem::I8 => Int8 as i32,
            Elem::I32 => Int32 as i32,
            Elem::I64 => Int64 as i32,
            Elem::U8 => Uint8 as i32,
        }
    }
}

/// A typed, shaped graph input/output (`ValueInfoProto`). A `None` dim is a
/// dynamic dimension; brain always exports static shapes so dims are concrete.
#[derive(Clone, Debug)]
pub struct ValueInfo {
    pub name: String,
    pub dims: Vec<i64>,
    pub elem: Elem,
}

/// A constant tensor stored in the graph (`TensorProto` initializer). Numeric
/// payloads are serialized into little-endian `raw_data` (the canonical compact
/// form ONNX/OpenVINO expect).
#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub dims: Vec<i64>,
    pub data: TensorData,
}

#[derive(Clone, Debug)]
pub enum TensorData {
    F32(Vec<f32>),
    I8(Vec<i8>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl TensorData {
    fn data_type(&self) -> i32 {
        use p::tensor_proto::DataType::*;
        match self {
            TensorData::F32(_) => Float as i32,
            TensorData::I8(_) => Int8 as i32,
            TensorData::I32(_) => Int32 as i32,
            TensorData::I64(_) => Int64 as i32,
        }
    }
    fn raw(&self) -> Vec<u8> {
        match self {
            TensorData::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::I8(v) => v.iter().map(|x| *x as u8).collect(),
            TensorData::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }
    /// Number of scalar elements (for shape sanity checks).
    pub fn len(&self) -> usize {
        match self {
            TensorData::F32(v) => v.len(),
            TensorData::I8(v) => v.len(),
            TensorData::I32(v) => v.len(),
            TensorData::I64(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A node attribute value.
#[derive(Clone, Debug)]
pub enum AttrVal {
    Int(i64),
    Ints(Vec<i64>),
    Float(f32),
    Floats(Vec<f32>),
    Str(String),
    Tensor(Tensor),
}

#[derive(Clone, Debug)]
pub struct Attr {
    pub name: String,
    pub val: AttrVal,
}

/// A graph operator instance (`NodeProto`).
#[derive(Clone, Debug)]
pub struct Node {
    pub op_type: String,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: Vec<Attr>,
}

/// Fluent constructor for a [`Node`]. Inputs/outputs are taken as string slices;
/// an empty string `""` denotes a skipped optional ONNX input (e.g. Resize `roi`).
impl Node {
    pub fn new(op_type: &str, inputs: &[&str], outputs: &[&str]) -> Node {
        Node {
            op_type: op_type.to_string(),
            name: String::new(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attrs: Vec::new(),
        }
    }
    pub fn name(mut self, n: &str) -> Node {
        self.name = n.to_string();
        self
    }
    pub fn attr_int(mut self, n: &str, v: i64) -> Node {
        self.attrs.push(Attr { name: n.to_string(), val: AttrVal::Int(v) });
        self
    }
    pub fn attr_ints(mut self, n: &str, v: &[i64]) -> Node {
        self.attrs.push(Attr { name: n.to_string(), val: AttrVal::Ints(v.to_vec()) });
        self
    }
    pub fn attr_float(mut self, n: &str, v: f32) -> Node {
        self.attrs.push(Attr { name: n.to_string(), val: AttrVal::Float(v) });
        self
    }
    pub fn attr_floats(mut self, n: &str, v: &[f32]) -> Node {
        self.attrs.push(Attr { name: n.to_string(), val: AttrVal::Floats(v.to_vec()) });
        self
    }
    pub fn attr_str(mut self, n: &str, v: &str) -> Node {
        self.attrs.push(Attr { name: n.to_string(), val: AttrVal::Str(v.to_string()) });
        self
    }
}

/// An owned ONNX graph: typed inputs/outputs, ordered nodes, and constant
/// initializers. Convert to the wire form with [`Graph::to_proto`].
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub name: String,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
    pub nodes: Vec<Node>,
    pub initializers: Vec<Tensor>,
}

impl Graph {
    fn value_info_to_proto(v: &ValueInfo) -> p::ValueInfoProto {
        let dims = v
            .dims
            .iter()
            .map(|&d| p::tensor_shape_proto::Dimension {
                dim_value: d,
                ..Default::default()
            })
            .collect();
        p::ValueInfoProto {
            name: v.name.clone(),
            r#type: Some(p::TypeProto {
                tensor_type: Some(p::type_proto::Tensor {
                    elem_type: v.elem.data_type(),
                    shape: Some(p::TensorShapeProto { dim: dims }),
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn tensor_to_proto(t: &Tensor) -> p::TensorProto {
        p::TensorProto {
            dims: t.dims.clone(),
            data_type: t.data.data_type(),
            name: t.name.clone(),
            raw_data: t.data.raw(),
            ..Default::default()
        }
    }

    fn attr_to_proto(a: &Attr) -> p::AttributeProto {
        use p::attribute_proto::AttributeType as AT;
        let mut out = p::AttributeProto { name: a.name.clone(), ..Default::default() };
        match &a.val {
            AttrVal::Int(v) => {
                out.i = *v;
                out.r#type = AT::Int as i32;
            }
            AttrVal::Ints(v) => {
                out.ints = v.clone();
                out.r#type = AT::Ints as i32;
            }
            AttrVal::Float(v) => {
                out.f = *v;
                out.r#type = AT::Float as i32;
            }
            AttrVal::Floats(v) => {
                out.floats = v.clone();
                out.r#type = AT::Floats as i32;
            }
            AttrVal::Str(v) => {
                out.s = v.clone().into_bytes();
                out.r#type = AT::String as i32;
            }
            AttrVal::Tensor(t) => {
                out.t = Some(Self::tensor_to_proto(t));
                out.r#type = AT::Tensor as i32;
            }
        }
        out
    }

    fn to_graph_proto(&self) -> p::GraphProto {
        p::GraphProto {
            node: self
                .nodes
                .iter()
                .map(|n| p::NodeProto {
                    input: n.inputs.clone(),
                    output: n.outputs.clone(),
                    name: n.name.clone(),
                    op_type: n.op_type.clone(),
                    attribute: n.attrs.iter().map(Self::attr_to_proto).collect(),
                    ..Default::default()
                })
                .collect(),
            name: self.name.clone(),
            initializer: self.initializers.iter().map(Self::tensor_to_proto).collect(),
            input: self.inputs.iter().map(Self::value_info_to_proto).collect(),
            output: self.outputs.iter().map(Self::value_info_to_proto).collect(),
            ..Default::default()
        }
    }

    /// Build the full `ModelProto` (opset `domain ""` version `opset`, the given
    /// `ir_version`), with brain's producer tag.
    pub fn to_proto(&self, opset: i64, ir_version: i64) -> p::ModelProto {
        p::ModelProto {
            ir_version,
            opset_import: vec![p::OperatorSetIdProto { domain: String::new(), version: opset }],
            producer_name: "brain".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            graph: Some(self.to_graph_proto()),
            ..Default::default()
        }
    }
}
