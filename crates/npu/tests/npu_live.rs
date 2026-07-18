// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Live NPU smoke: build a trivial ONNX with brain's own `onnx` crate, compile it
//! on the real Intel NPU via `NpuSession`, run it, and check the result matches a
//! hand-computed reference. Proves the whole brain -> ONNX -> OpenVINO -> NPU path
//! is wired, before any model-specific topology depends on it.
//!
//! Gated on the NPU actually being present: if OpenVINO can't find the runtime or
//! the device, the test SKIPS (so it stays green on machines without an NPU).
use npu::openvino::{NpuConfig, NpuDevice, NpuSession};
use onnx::{GraphBuilder, Node};

/// A 1x1 conv that scales channel c by (c+1), plus a per-channel bias of c*10, over
/// a [1,3,2,2] input — small enough to verify every output by hand.
fn tiny_scale_onnx() -> Vec<u8> {
    let mut g = GraphBuilder::new("tiny");
    g.input_f32("x", &[1, 3, 2, 2]);
    g.output_f32("y", &[1, 3, 2, 2]);
    // weight [3,3,1,1]: diagonal, w[c,c]=c+1.
    let mut w = vec![0f32; 3 * 3];
    for c in 0..3 {
        w[c * 3 + c] = (c + 1) as f32;
    }
    g.init_f32("w", &[3, 3, 1, 1], w);
    g.init_f32("b", &[3], vec![0.0, 10.0, 20.0]);
    g.add(
        Node::new("Conv", &["x", "w", "b"], &["y"])
            .name("conv")
            .attr_ints("kernel_shape", &[1, 1])
            .attr_ints("strides", &[1, 1])
            .attr_ints("pads", &[0, 0, 0, 0])
            .attr_int("group", 1),
    );
    g.finish()
}

#[test]
fn brain_onnx_runs_on_the_intel_npu() {
    let bytes = tiny_scale_onnx();
    let cfg = NpuConfig { device: NpuDevice::Npu, allow_fallback: false, ..Default::default() };
    let mut sess = match NpuSession::load_bytes(&bytes, &cfg) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP: NPU unavailable ({e})");
            return;
        }
    };
    assert_eq!(sess.device(), "NPU", "must have compiled for the NPU, not a fallback");
    assert_eq!(sess.input_shape(), [1, 3, 2, 2]);

    // Input: channel c filled with value (c+1)*0.5 -> 0.5, 1.0, 1.5.
    let mut x = vec![0f32; 3 * 2 * 2];
    for c in 0..3 {
        for i in 0..4 {
            x[c * 4 + i] = (c + 1) as f32 * 0.5;
        }
    }
    let out = sess.run(&x, [1, 3, 2, 2]).expect("NPU inference");
    let (_, shape, data) = &out.tensors[0];
    assert_eq!(shape, &vec![1, 3, 2, 2]);
    // y[c] = (c+1)*x[c] + c*10 = (c+1)^2*0.5 + c*10.
    for c in 0..3 {
        let want = ((c + 1) * (c + 1)) as f32 * 0.5 + (c as f32) * 10.0;
        for i in 0..4 {
            let got = data[c * 4 + i];
            assert!((got - want).abs() < 1e-2, "channel {c}: NPU gave {got}, expected {want}");
        }
    }
    eprintln!("NPU OK: brain-built ONNX compiled and ran on {} with correct output", sess.device());
}
