// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Hardware smoke for [`npu::roofline::measure`]: mirrors `tests/npugraph.rs`'s
//! convention of building a tiny graph purely in-process via `onnx::GraphBuilder`
//! and compiling it through `NpuGraph::compile_bytes`, but exercises the actual
//! `roofline` module end to end. Skips when no NPU/OpenVINO is present, so this
//! stays green on CI machines without one - the no-hardware contract itself is
//! covered by the non-ignored unit test in `src/roofline.rs`.

use npu::openvino::{npu_present, NpuDevice};
use npu::roofline::measure;

#[test]
#[ignore = "needs a real Intel NPU + OpenVINO runtime"]
fn roofline_round_trip_on_real_npu() {
    if !npu_present() {
        brain_testutil::skip_unavailable("no NPU/OpenVINO present");
        return;
    }
    let r = measure(NpuDevice::Npu).expect("measure() should succeed when the NPU is actually present");
    eprintln!(
        "NPU roofline: {} caps={:?} fp16_gops={:?} int8_gops={:?}",
        r.device_name, r.capabilities, r.fp16_gops, r.int8_gops
    );
    assert!(!r.device_name.is_empty());
    // Every measured number the device claims must be a positive, finite rate -
    // never a fabricated or degenerate zero.
    if r.capabilities.iter().any(|c| c.eq_ignore_ascii_case("FP16")) {
        let v = r.fp16_gops.expect("device claims FP16 but no fp16 rate was measured");
        assert!(v.is_finite() && v > 0.0, "fp16_gops = {v}");
    } else {
        assert!(r.fp16_gops.is_none(), "device does not claim FP16 but a rate was reported");
    }
    if r.capabilities.iter().any(|c| c.eq_ignore_ascii_case("INT8")) {
        let v = r.int8_gops.expect("device claims INT8 but no int8 rate was measured");
        assert!(v.is_finite() && v > 0.0, "int8_gops = {v}");
    } else {
        assert!(r.int8_gops.is_none(), "device does not claim INT8 but a rate was reported");
    }
}
