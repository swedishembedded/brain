// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Hardware smoke for the generic [`npu::openvino::NpuGraph`] runner — the reuse
//! seam every model's NPU export shares. Builds a tiny ONNX (MatMul + Relu) with the
//! shared `onnx::GraphBuilder`, compiles it on the NPU (fp16), runs it, and checks
//! the output against a host reference. Skips when no NPU/OpenVINO is present.

use npu::openvino::{npu_present, Feed, NpuConfig, NpuDevice, NpuGraph, PerfHint};

#[test]
fn npugraph_matmul_relu_on_npu() {
    if !npu_present() {
        eprintln!("skip: no NPU/OpenVINO present");
        return;
    }
    // y = relu(x @ w),  x:[1,64], w:[64,64]
    let n = 64usize;
    let w: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) * 0.03).collect();

    let mut g = onnx::GraphBuilder::new("tiny_mm_relu");
    g.input_f32("x", &[1, n as i64]);
    g.init_f32("w", &[n as i64, n as i64], w.clone());
    g.add(onnx::Node::new("MatMul", &["x", "w"], &["mm"]));
    g.add(onnx::Node::new("Relu", &["mm"], &["y"]));
    g.output_f32("y", &[1, n as i64]);
    let bytes = g.finish_with(onnx::DEFAULT_OPSET, onnx::DEFAULT_IR_VERSION);

    let cfg = NpuConfig { device: NpuDevice::Npu, perf_hint: PerfHint::Latency, allow_fallback: true, ..Default::default() };
    let mut graph = NpuGraph::compile_bytes(&bytes, &cfg).expect("compile tiny graph on NPU");
    eprintln!("NpuGraph compiled on device {}, inputs {:?}, outputs {:?}", graph.device(), graph.input_names(), graph.output_names());

    let out = graph.run(&[("x", Feed::F32(&x, vec![1, n as i64]))]).expect("run");
    assert_eq!(out.len(), 1);
    let (_name, shape, data) = &out[0];
    assert_eq!(shape, &vec![1, n]);

    // host reference
    let mut refy = vec![0.0f32; n];
    for j in 0..n {
        let mut acc = 0.0f32;
        for k in 0..n {
            acc += x[k] * w[k * n + j];
        }
        refy[j] = acc.max(0.0);
    }
    let maxdiff = data.iter().zip(&refy).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("NpuGraph matmul+relu maxdiff vs host = {maxdiff:.2e}");
    assert!(maxdiff < 5e-2, "NPU fp16 output should match host within fp16 tol, got {maxdiff}");
}
