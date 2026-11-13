// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export a trained YOLO `.safetensors` checkpoint to ONNX (fp32 or INT8-QDQ).

use onnx::GraphBuilder;

use crate::quant::Quant;
use crate::topology::build_graph;

/// The model config of a checkpoint (a brain file's header, or yolov8n's for
/// an Ultralytics `.pt`).
///
/// Panics on an unreadable checkpoint; [`try_config_of`] reports it.
pub fn config_of(weights_path: &str) -> yolov8::YoloConfig {
    try_config_of(weights_path).unwrap_or_else(|e| panic!("{e}"))
}

/// [`config_of`] reporting an unreadable checkpoint as an error.
pub fn try_config_of(weights_path: &str) -> Result<yolov8::YoloConfig, String> {
    let r = yolov8::import::open(weights_path)?;
    Ok(yolov8::YoloConfig::from_json(&r.config()))
}

/// Open the checkpoint (header only) + config, optionally overriding the
/// (square) input size.
fn load(weights_path: &str, input: Option<u32>) -> Result<(yolov8::YoloConfig, checkpoint::weightio::WeightReader), String> {
    let r = yolov8::import::open(weights_path)?;
    let mut cfg = yolov8::YoloConfig::from_json(&r.config());
    if let Some(s) = input {
        cfg.input = s;
    }
    Ok((cfg, r))
}

/// Export an fp32 ONNX graph. `input` overrides the static input resolution
/// (default: the checkpoint's training resolution).
pub fn export_fp32(weights_path: &str, out_path: &str, input: Option<u32>, opset: i64) -> std::io::Result<()> {
    std::fs::write(out_path, try_build_fp32_bytes(weights_path, input, opset).map_err(std::io::Error::other)?)
}

/// Export an INT8 Q/DQ ONNX graph using calibrated activation scales.
pub fn export_int8(
    weights_path: &str,
    quant: &Quant,
    out_path: &str,
    input: Option<u32>,
    opset: i64,
) -> std::io::Result<()> {
    std::fs::write(out_path, try_build_int8_bytes(weights_path, quant, input, opset).map_err(std::io::Error::other)?)
}

/// The INT8 Q/DQ ONNX graph's bytes, for a caller that places them itself.
pub fn try_build_int8_bytes(weights_path: &str, quant: &Quant, input: Option<u32>, opset: i64) -> Result<Vec<u8>, String> {
    let (cfg, r) = load(weights_path, input)?;
    let mut g = GraphBuilder::new("yolov8-int8");
    build_graph(&cfg, &r, Some(quant), &mut g);
    Ok(g.finish_with(opset, onnx::DEFAULT_IR_VERSION))
}

/// fp32 ONNX bytes in-memory, to compile without writing a temp file.
/// Panics on an unreadable checkpoint; [`try_build_fp32_bytes`] reports it.
pub fn build_fp32_bytes(weights_path: &str, input: Option<u32>, opset: i64) -> Vec<u8> {
    try_build_fp32_bytes(weights_path, input, opset).unwrap_or_else(|e| panic!("{e}"))
}

/// [`build_fp32_bytes`] reporting an unreadable checkpoint as an error.
pub fn try_build_fp32_bytes(weights_path: &str, input: Option<u32>, opset: i64) -> Result<Vec<u8>, String> {
    let (cfg, r) = load(weights_path, input)?;
    let mut g = GraphBuilder::new("yolov8");
    build_graph(&cfg, &r, None, &mut g);
    Ok(g.finish_with(opset, onnx::DEFAULT_IR_VERSION))
}
