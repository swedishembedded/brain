// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end exporter structure tests (no OpenVINO, no NPU). Build a tiny YOLO,
//! save a checkpoint, export fp32 + INT8-QDQ ONNX, and decode the bytes back to
//! verify the graph topology, static IO shapes, op coverage, and the INT8 Q/DQ
//! representation. Also confirms the calibration keys (`*.conv.weight` prefixes
//! from `full_param_list`) match the topology's conv prefixes exactly.

use std::collections::HashMap;

use onnx::onnx::ModelProto;
use yolov8::{Yolo, YoloConfig};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").map(|v| !v.is_empty()).unwrap_or(false)
}

fn tmp(name: &str) -> String {
    let dir = std::env::temp_dir();
    dir.join(format!("brain_npu_export_{}_{}", std::process::id(), name)).to_string_lossy().into_owned()
}

fn op_counts(m: &ModelProto) -> HashMap<String, usize> {
    let mut h = HashMap::new();
    for n in &m.graph.as_ref().unwrap().node {
        *h.entry(n.op_type.clone()).or_insert(0) += 1;
    }
    h
}

fn save_tiny(path: &str) -> YoloConfig {
    let cfg = YoloConfig::tiny(3);
    let init = yolov8::init_weights(&cfg, 7);
    let model = Yolo::new(cfg.clone(), 1, 0, &init);
    model.save(path);
    cfg
}

#[test]
fn fp32_export_topology() {
    if skip() {
        return;
    }
    let wpath = tmp("fp32.safetensors");
    let cfg = save_tiny(&wpath);
    let bytes = npu::build_fp32_bytes(&wpath, None, onnx::DEFAULT_OPSET);
    let m = onnx::decode_model(&bytes).expect("decode");
    let g = m.graph.as_ref().unwrap();

    // Opset 13, single static input images:[1,3,128,128].
    assert_eq!(m.opset_import[0].version, 13);
    assert_eq!(g.input.len(), 1);
    assert_eq!(g.input[0].name, "images");
    let in_dims: Vec<i64> = g.input[0].r#type.as_ref().unwrap().tensor_type.as_ref().unwrap()
        .shape.as_ref().unwrap().dim.iter().map(|d| d.dim_value).collect();
    assert_eq!(in_dims, vec![1, 3, cfg.input as i64, cfg.input as i64]);

    // 6 head outputs with the right names + NCHW shapes (strides 8/16/32).
    let out_names: Vec<&str> = g.output.iter().map(|o| o.name.as_str()).collect();
    for s in 0..3 {
        assert!(out_names.contains(&format!("head.{s}.cls").as_str()), "missing head.{s}.cls");
        assert!(out_names.contains(&format!("head.{s}.reg").as_str()), "missing head.{s}.reg");
    }
    assert_eq!(g.output.len(), 6);
    // cls channel = nc, reg channel = 4*reg_max, spatial = input/stride.
    for o in &g.output {
        let dims: Vec<i64> = o.r#type.as_ref().unwrap().tensor_type.as_ref().unwrap()
            .shape.as_ref().unwrap().dim.iter().map(|d| d.dim_value).collect();
        let want_c = if o.name.ends_with(".cls") { cfg.nc as i64 } else { (4 * cfg.reg_max) as i64 };
        assert_eq!(dims[1], want_c, "{}: channel {} != {want_c}", o.name, dims[1]);
    }

    // Op coverage: the YOLO op set must all be present, and NO Q/DQ in fp32.
    let ops = op_counts(&m);
    for op in ["Conv", "Sigmoid", "Mul", "Split", "Concat", "MaxPool", "Resize", "Add"] {
        assert!(ops.contains_key(op), "fp32 graph missing op {op}");
    }
    assert!(!ops.contains_key("QuantizeLinear"), "fp32 graph must have no Q/DQ");
    // SPPF has exactly 3 maxpools per detector (one SPPF), each scale upsample x2.
    assert_eq!(ops["MaxPool"], 3);
    assert_eq!(ops["Resize"], 2);
    std::fs::remove_file(&wpath).ok();
}

#[test]
fn int8_export_has_qdq() {
    if skip() {
        return;
    }
    let wpath = tmp("int8.safetensors");
    let cfg = save_tiny(&wpath);

    // Calibration keys = every conv prefix (`X.conv.weight` -> X). This is exactly
    // what the calibrator will produce; here we stub a uniform scale.
    let mut q = npu::Quant::new();
    for (name, _) in cfg.full_param_list() {
        if let Some(pfx) = name.strip_suffix(".conv.weight") {
            q.act_scales.insert(pfx.to_string(), 0.05);
        }
    }
    assert!(!q.is_empty());

    let opath = tmp("int8.onnx");
    npu::export_int8(&wpath, &q, &opath, None, onnx::DEFAULT_OPSET).expect("export int8");
    let bytes = std::fs::read(&opath).unwrap();
    let m = onnx::decode_model(&bytes).expect("decode");
    let ops = op_counts(&m);

    // Every quantized conv contributes one activation Q + one activation DQ + one
    // weight DQ. So QuantizeLinear count == number of quantized convs, and
    // DequantizeLinear == 2x that.
    let n_qconv = q.len();
    assert_eq!(ops.get("QuantizeLinear").copied().unwrap_or(0), n_qconv);
    assert_eq!(ops.get("DequantizeLinear").copied().unwrap_or(0), 2 * n_qconv);

    // INT8 weight initializers exist (data_type INT8 = 3).
    let g = m.graph.as_ref().unwrap();
    let n_i8 = g.initializer.iter().filter(|t| t.data_type == 3).count();
    assert!(n_i8 >= n_qconv, "expected >= {n_qconv} INT8 initializers, got {n_i8}");

    std::fs::remove_file(&wpath).ok();
    std::fs::remove_file(&opath).ok();
}
