// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain-npu` — deploy brain's YOLO detector to the Intel NPU.
//!
//! Pipeline: **export** the trained model to ONNX, brain-native **INT8
//! post-training quantization** (calibrate → Q/DQ ONNX), then **compile + run**
//! on the Intel NPU via OpenVINO (behind the optional `openvino` feature). DFL
//! decode + NMS stay on the host in Rust.
//!
//! The export / quantize / fake-quant-simulate core is pure Rust and builds &
//! tests anywhere (no NPU, no OpenVINO). Only `run`/`bench` need the `openvino`
//! feature + an Intel NPU.

// Pure-Rust core (always compiled, hardware-free).
pub mod calib;
pub mod decode;
pub mod export;
pub mod fold;
pub mod quant;
pub mod sim;
pub mod topology;

// OpenVINO runtime seam (real on x86_64 linux/windows, stub elsewhere).
pub mod openvino;

pub use calib::{calibrate, calibrate_from_weights, load_calib_images, RangeCollector};
pub use decode::{decode_npu_outputs, detect_image, detect_weights_on_npu};
pub use export::{build_fp32_bytes, config_of, export_fp32, export_int8};
pub use quant::Quant;
pub use sim::{reference_logits, simulate_logits, simulate_map, FakeQuantTap};
pub use topology::{build_graph, WeightSource};
