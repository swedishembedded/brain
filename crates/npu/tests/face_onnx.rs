// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The face stack on the NPU: ArcFace needs **no topology**, SCRFD needs shape
//! freezing.
//!
//! Every other model in this tree needs a hand-written `*_topology` because
//! brain holds its weights as tensors and has to emit a graph. The antelopev2
//! face stack is the exception: it SHIPS as ONNX, and
//! `npu::openvino::load(onnx_path)` takes a plain `.onnx`. So the question is
//! not "how do we build the graph" but "will the released graph compile for the
//! NPU", and that is answerable here, without the hardware, by reading it.
//!
//! The answers differ for the two models, which is exactly why it was worth
//! asking:
//!
//! * **`glintr100` (ArcFace)** - `Add`, `BatchNormalization`, `Conv`, `Flatten`,
//!   `Gemm`, `PRelu`. All NPU-supported, all static. It needs no topology and no
//!   conversion: point OpenVINO at the file.
//! * **`scrfd_10g_bnkps` (the detector)** - carries `Shape`, `Slice`, `Gather`,
//!   `Unsqueeze` and `Reshape` driven by them, and declares its spatial dims as
//!   `?`. That is a DYNAMIC-shape graph, which the NPU plugin does not compile;
//!   it needs its input frozen to a concrete size and the shape subgraph
//!   constant-folded first. Naming that here is the point - it is the actual
//!   work item, and it is not obvious from the outside.
//!
//! The two models are two crates and two served models, but they read the same
//! two released files and the interesting fact is the CONTRAST between them, so
//! both live in one test file. Each finds its weights through its own directory
//! variable (`BRAIN_ARCFACE_DIR`, `BRAIN_SCRFD_DIR`), which is what a caller
//! actually sets.
//!
//! Skips when the checkpoints are absent, like brain's other weight-gated tests.

use std::path::PathBuf;

/// Ops the OpenVINO NPU plugin compiles for a static conv net. Not exhaustive -
/// it is the set these two models are allowed to use, so a re-exported
/// checkpoint that introduces something else fails here rather than at
/// `compile_model` on a machine far away.
const NPU_SAFE: &[&str] = &[
    "Add", "BatchNormalization", "Concat", "Conv", "Flatten", "Gemm", "GlobalAveragePool",
    "Identity", "MaxPool", "AveragePool", "Mul", "PRelu", "Relu", "Sigmoid", "Sub", "Resize",
    "Transpose",
];

/// Ops that imply a data-dependent shape. Their presence is the diagnosis, not
/// a style complaint: the NPU plugin needs static shapes.
const DYNAMIC_SHAPE_OPS: &[&str] = &["Shape", "Slice", "Gather", "Unsqueeze", "NonMaxSuppression"];

fn weights_dir(var: &str) -> Option<PathBuf> {
    let p = std::env::var(var)
        .unwrap_or_else(|_| "/data/workspace/resources/identity/weights/antelopev2".into());
    let p = PathBuf::from(p);
    p.is_dir().then_some(p)
}

fn read(dir: &PathBuf, name: &str) -> Option<onnx::onnx::GraphProto> {
    let p = dir.join(name);
    if !p.exists() {
        eprintln!("SKIP: {} absent", p.display());
        return None;
    }
    let m = onnx::read::read_file(&p).unwrap_or_else(|e| panic!("read {name}: {e}"));
    m.graph
}

fn ops(g: &onnx::onnx::GraphProto) -> Vec<String> {
    let mut v: Vec<String> = g.node.iter().map(|n| n.op_type.clone()).collect();
    v.sort();
    v.dedup();
    v
}

/// Spatial dims of the first input, `None` where the graph left it symbolic.
fn input_dims(g: &onnx::onnx::GraphProto) -> Vec<Option<i64>> {
    g.input[0]
        .r#type
        .as_ref()
        .and_then(|t| t.tensor_type.as_ref())
        .and_then(|tt| tt.shape.as_ref())
        .map(|s| {
            s.dim
                .iter()
                .map(|d| if d.dim_param.is_empty() { Some(d.dim_value) } else { None })
                .collect()
        })
        .unwrap_or_default()
}

/// ArcFace is ready as shipped - this is the test that says "no topology
/// needed", so nobody writes one.
#[test]
fn arcface_is_npu_ready_as_shipped() {
    let Some(dir) = weights_dir("BRAIN_ARCFACE_DIR") else {
        eprintln!("SKIP: set BRAIN_ARCFACE_DIR to a directory holding glintr100.onnx");
        return;
    };
    let Some(g) = read(&dir, "glintr100.onnx") else { return };

    let ops = ops(&g);
    eprintln!("glintr100 ops: {ops:?}");
    for op in &ops {
        assert!(NPU_SAFE.contains(&op.as_str()), "glintr100 uses `{op}`, outside the NPU-safe set");
        assert!(!DYNAMIC_SHAPE_OPS.contains(&op.as_str()), "glintr100 has dynamic-shape op `{op}`");
    }

    // Everything but the batch dim must be static, or the plugin cannot plan.
    let dims = input_dims(&g);
    assert_eq!(dims.len(), 4, "expected NCHW input, got {dims:?}");
    assert_eq!(&dims[1..], &[Some(3), Some(112), Some(112)], "ArcFace's 112x112 RGB input");
}

/// SCRFD is NOT ready, and this records precisely why - so the work item is a
/// sentence rather than a debugging session on the machine with the NPU.
#[test]
fn scrfd_needs_its_shapes_frozen_before_the_npu_can_take_it() {
    let Some(dir) = weights_dir("BRAIN_SCRFD_DIR") else {
        eprintln!("SKIP: set BRAIN_SCRFD_DIR to a directory holding scrfd_10g_bnkps.onnx");
        return;
    };
    let Some(g) = read(&dir, "scrfd_10g_bnkps.onnx") else { return };

    let ops = ops(&g);
    let dynamic: Vec<&String> = ops.iter().filter(|o| DYNAMIC_SHAPE_OPS.contains(&o.as_str())).collect();
    let dims = input_dims(&g);
    eprintln!("scrfd ops: {ops:?}");
    eprintln!("scrfd input dims: {dims:?}   dynamic-shape ops: {dynamic:?}");

    // This asserts the PROBLEM, so the day someone fixes it the test fails and
    // they update it to the ready-as-shipped shape above. A test that asserted
    // "it works" would be a lie; one that asserts nothing would be silence.
    assert!(
        !dynamic.is_empty() || dims[2..].contains(&None),
        "scrfd looks static now - freeze the input and re-point this test at \
         `arcface_is_npu_ready_as_shipped`'s assertions"
    );
}
