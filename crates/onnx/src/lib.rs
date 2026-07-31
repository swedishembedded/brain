// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain-onnx` — a small, pure-Rust ONNX graph model + serializer.
//!
//! Architecture-agnostic: it knows nothing about YOLO, brain, or any hardware.
//! It exists so brain can *export* its models to the ONNX interchange format and
//! hand them to an external graph compiler (today: OpenVINO for the Intel NPU,
//! via `brain-npu`). Only the **export/serialize** direction is implemented; the
//! owned [`graph`] model is deliberately decoupled from the wire types so a
//! future ONNX *import* frontend could add a `from_proto` without reshaping the
//! crate. The **import** direction now exists in part: [`read`] decodes a
//! serialized model back into tensors + nodes (used by `crates/facenet`, whose
//! reference release ships ONNX only). It stops short of a `Graph::from_proto`
//! — see that module for why.
//!
//! The only dependency is the `prost` runtime crate. The protobuf bindings in
//! [`onnx`] are vendored (generated offline from `proto/onnx.proto`), so no
//! `protoc`/codegen runs in brain's build.
//!
//! ```
//! use onnx::{GraphBuilder, Node};
//! let mut g = GraphBuilder::new("relu");
//! g.input_f32("x", &[1, 4]);
//! g.output_f32("y", &[1, 4]);
//! g.add(Node::new("Relu", &["x"], &["y"]).name("act"));
//! let bytes = g.finish();
//! assert!(!bytes.is_empty());
//! ```

pub mod builder;
pub mod conv;
pub mod graph;
pub mod onnx;
pub mod read;

pub use builder::{GraphBuilder, DEFAULT_IR_VERSION, DEFAULT_OPSET};
pub use conv::{conv_transpose1d_ref, conv_transpose_node, ConvTranspose1d};
pub use graph::{Attr, AttrVal, Elem, Graph, Node, Tensor, TensorData, ValueInfo};
pub use read::{initializers, read_file, OnnxTensor};

/// Decode serialized ONNX bytes back into a [`onnx::ModelProto`] (for inspection
/// / tests). Keeps `prost` an implementation detail of this crate.
pub fn decode_model(bytes: &[u8]) -> Result<onnx::ModelProto, String> {
    use prost::Message;
    onnx::ModelProto::decode(bytes).map_err(|e| e.to_string())
}
