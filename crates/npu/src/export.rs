// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export a trained YOLO `.safetensors` checkpoint to ONNX (fp32 or INT8-QDQ).

use onnx::GraphBuilder;

use crate::quant::Quant;
use crate::topology::build_graph;

/// The model config stored in a checkpoint header.
pub fn config_of(weights_path: &str) -> yolo::YoloConfig {
    let r = checkpoint::weightio::WeightReader::open(weights_path).unwrap_or_else(|e| panic!("open {weights_path}: {e}"));
    yolo::YoloConfig::from_json(&r.config())
}

/// Open the checkpoint (header only) + config, optionally overriding the
/// (square) input size.
fn load(weights_path: &str, input: Option<u32>) -> (yolo::YoloConfig, checkpoint::weightio::WeightReader) {
    let r = checkpoint::weightio::WeightReader::open(weights_path).unwrap_or_else(|e| panic!("open {weights_path}: {e}"));
    let mut cfg = yolo::YoloConfig::from_json(&r.config());
    if let Some(s) = input {
        cfg.input = s;
    }
    (cfg, r)
}

/// Export an fp32 ONNX graph. `input` overrides the static input resolution
/// (default: the checkpoint's training resolution).
pub fn export_fp32(weights_path: &str, out_path: &str, input: Option<u32>, opset: i64) -> std::io::Result<()> {
    std::fs::write(out_path, build_fp32_bytes(weights_path, input, opset))
}

/// Export an INT8 Q/DQ ONNX graph using calibrated activation scales.
pub fn export_int8(
    weights_path: &str,
    quant: &Quant,
    out_path: &str,
    input: Option<u32>,
    opset: i64,
) -> std::io::Result<()> {
    let (cfg, r) = load(weights_path, input);
    let mut g = GraphBuilder::new("yolov8-int8");
    build_graph(&cfg, &r, Some(quant), &mut g);
    std::fs::write(out_path, g.finish_with(opset, onnx::DEFAULT_IR_VERSION))
}

/// fp32 ONNX bytes in-memory (used by `yolo detect --device npu` to compile
/// without writing a temp file).
pub fn build_fp32_bytes(weights_path: &str, input: Option<u32>, opset: i64) -> Vec<u8> {
    let (cfg, r) = load(weights_path, input);
    let mut g = GraphBuilder::new("yolov8");
    build_graph(&cfg, &r, None, &mut g);
    g.finish_with(opset, onnx::DEFAULT_IR_VERSION)
}
