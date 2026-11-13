// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A YOLOv8 detector compiled for the Intel NPU (or any OpenVINO device):
//! built either from a brain checkpoint (exported to fp32 ONNX in memory, so
//! nothing is written) or from an ONNX graph a previous `export` / `quantize`
//! wrote, then asked for detections of any number of images. The box decode
//! and NMS run on the host ([`crate::decode`]), the same tail the engine's own
//! `Yolo::detect` uses.
//!
//! Swedish Embedded AB implements object detection on NPUs and other edge
//! accelerators for its clients. If your team needs expertise in deploying and
//! quantizing detectors for OpenVINO then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::path::Path;

use yolov8::nms::Detection;
use yolov8::YoloConfig;

use crate::openvino::{NpuConfig, NpuSession};

/// A compiled YOLOv8 graph and the config its outputs decode with.
pub struct YoloSession {
    session: NpuSession,
    cfg: YoloConfig,
}

impl YoloSession {
    /// Compile the checkpoint `weights` (`input` overrides its input side).
    pub fn from_weights(weights: &str, input: Option<u32>, npu: &NpuConfig) -> Result<YoloSession, String> {
        let mut cfg = crate::export::try_config_of(weights)?;
        if let Some(side) = input {
            cfg.input = side;
        }
        let bytes = crate::export::try_build_fp32_bytes(weights, input, onnx::DEFAULT_OPSET)?;
        let session = NpuSession::load_bytes(&bytes, npu).map_err(|e| e.to_string())?;
        Ok(YoloSession { session, cfg })
    }

    /// Compile the ONNX graph at `onnx`. The graph does not carry its class
    /// count or DFL width, so they are given (defaults are yolov8n's: 80 and
    /// 16); the input side is the graph's own unless overridden.
    pub fn from_onnx(onnx: &Path, nc: Option<u32>, reg_max: Option<u32>, input: Option<u32>, npu: &NpuConfig) -> Result<YoloSession, String> {
        let session = NpuSession::load(onnx, npu).map_err(|e| e.to_string())?;
        let mut cfg = YoloConfig::yolov8n();
        cfg.input = input.unwrap_or(session.input_shape()[2] as u32);
        if let Some(n) = nc {
            cfg.nc = n;
        }
        if let Some(r) = reg_max {
            cfg.reg_max = r;
        }
        Ok(YoloSession { session, cfg })
    }

    /// The OpenVINO device the graph compiled for.
    pub fn device(&self) -> &str {
        self.session.device()
    }

    pub fn config(&self) -> &YoloConfig {
        &self.cfg
    }

    /// Detections `[x1,y1,x2,y2,conf,class]` of the `w x h` interleaved-RGB
    /// frame `hwc` in `[0,1]`, in the frame's own pixels.
    pub fn detect(&mut self, hwc: &[f32], w: u32, h: u32, conf: f32, iou: f32) -> Result<Vec<Detection>, String> {
        crate::decode::detect_image(&mut self.session, hwc, w, h, &self.cfg, conf, iou).map_err(|e| e.to_string())
    }
}
