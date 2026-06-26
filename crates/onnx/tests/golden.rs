// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Build → encode → re-decode (with prost) round-trip tests. These prove the
//! owned graph model maps onto the correct ONNX wire fields without needing an
//! external ONNX validator.

use onnx::onnx as proto;
use onnx::{GraphBuilder, Node};
use prost::Message;

#[test]
fn two_node_graph_roundtrips() {
    let mut g = GraphBuilder::new("tiny");
    g.input_f32("x", &[1, 3, 4, 4]);
    g.output_f32("y", &[1, 2, 4, 4]);
    // A 2-node SiLU-ish chain: Sigmoid then Mul(x, sigmoid).
    g.init_f32("w", &[2, 3, 1, 1], vec![0.5; 6]);
    g.add(Node::new("Conv", &["x", "w"], &["c"]).name("conv").attr_ints("kernel_shape", &[1, 1]));
    g.add(Node::new("Sigmoid", &["c"], &["s"]).name("sig"));
    g.add(Node::new("Mul", &["c", "s"], &["y"]).name("silu"));

    let bytes = g.finish();
    assert!(!bytes.is_empty());

    let model = proto::ModelProto::decode(&bytes[..]).expect("decode");
    assert_eq!(model.ir_version, onnx::DEFAULT_IR_VERSION);
    assert_eq!(model.opset_import.len(), 1);
    assert_eq!(model.opset_import[0].version, onnx::DEFAULT_OPSET);
    assert_eq!(model.producer_name, "brain");

    let gr = model.graph.expect("graph");
    assert_eq!(gr.name, "tiny");
    assert_eq!(gr.node.len(), 3);
    assert_eq!(gr.node[0].op_type, "Conv");
    assert_eq!(gr.node[2].op_type, "Mul");
    assert_eq!(gr.node[2].input, vec!["c".to_string(), "s".to_string()]);

    // IO typing + static shapes survive.
    assert_eq!(gr.input.len(), 1);
    let xt = gr.input[0].r#type.as_ref().unwrap().tensor_type.as_ref().unwrap();
    assert_eq!(xt.elem_type, proto::tensor_proto::DataType::Float as i32);
    let dims: Vec<i64> = xt.shape.as_ref().unwrap().dim.iter().map(|d| d.dim_value).collect();
    assert_eq!(dims, vec![1, 3, 4, 4]);

    // Initializer survives with raw little-endian f32 payload.
    assert_eq!(gr.initializer.len(), 1);
    let w = &gr.initializer[0];
    assert_eq!(w.name, "w");
    assert_eq!(w.dims, vec![2, 3, 1, 1]);
    assert_eq!(w.data_type, proto::tensor_proto::DataType::Float as i32);
    assert_eq!(w.raw_data.len(), 6 * 4);
    let first = f32::from_le_bytes(w.raw_data[0..4].try_into().unwrap());
    assert_eq!(first, 0.5);

    // Conv attribute survives as INTS.
    let a = &gr.node[0].attribute[0];
    assert_eq!(a.name, "kernel_shape");
    assert_eq!(a.r#type, proto::attribute_proto::AttributeType::Ints as i32);
    assert_eq!(a.ints, vec![1, 1]);
}

#[test]
fn int8_initializer_packs_one_byte_each() {
    let mut g = GraphBuilder::new("q");
    g.init_i8("wq", &[4], vec![-128, -1, 0, 127]);
    let bytes = g.finish();
    let model = proto::ModelProto::decode(&bytes[..]).unwrap();
    let t = &model.graph.unwrap().initializer[0];
    assert_eq!(t.data_type, proto::tensor_proto::DataType::Int8 as i32);
    assert_eq!(t.raw_data, vec![0x80, 0xFF, 0x00, 0x7F]);
}

#[test]
#[should_panic(expected = "imply")]
fn initializer_shape_mismatch_panics() {
    let mut g = GraphBuilder::new("bad");
    g.init_f32("w", &[2, 2], vec![1.0, 2.0, 3.0]); // 3 != 4
}
