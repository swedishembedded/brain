// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `yolov8` - the YOLOv8 detector from the command line: every served action
//! (`detect`, `train`, `finetune`, `eval`) with every parameter, and the NPU
//! route (`npu-export`, `npu-quantize`, `npu-sim`, `npu-run`) that deploys a
//! trained checkpoint to the Intel NPU through OpenVINO.
//!
//! The first four are the model's own served actions (`yolov8::caps`), so this
//! program, `brain serve` and D-Bus clients drive the same code with the same
//! defaults; `yolov8 <action> --help` lists every parameter. The NPU actions
//! ([`npu_actions`]) are the part that is an application rather than the served
//! model: they write ONNX files for a runtime brain does not itself serve
//! through, and need the NPU crate, which depends on the model crates.
//!
//! Swedish Embedded AB implements object detection and person re-identification
//! on its clients' own cameras and NPUs. If your team needs expertise in
//! deploying and quantizing detectors for OpenVINO then you can procure our
//! services by sending an email to info@swedishembedded.com.

mod npu_actions;

use std::sync::Arc;

fn main() {
    // Identity verification resolves SCRFD + ArcFace from the model store.
    let components: Arc<dyn capability::Components> = Arc::new(loader::components::StoreComponents::new(yolov8_faces::component_specs()));
    let build = move |assembly: &capability::Assembly| npu_actions::provider(assembly, Arc::new(yolov8_faces::StoreFaces::new(components.clone())));
    let model = invoke::Model { manifest: npu_actions::manifest(), weights: Some(("yolov8", &yolov8::spec::YoloSpec)), build: &build };
    invoke::main_with("yolov8", &model, &[])
}
