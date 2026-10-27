// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for real-time object detection on
// the NPUs of low-cost camera SoCs for its clients. If your team needs
// expertise in embedded vision on Rockchip-class hardware then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Calibrate a YOLOv5 checkpoint for the Rockchip NPU, on a host:
//!
//! ```text
//! npu_calibrate <model.safetensors> <images-dir> [input-size, default 320]
//! ```
//!
//! Every JPEG/PNG in `<images-dir>` is letterboxed to the input size and run
//! through the float network; the range of every tensor is written to
//! `<model>.npu-calibration.json` beside the checkpoint, which is what the
//! board loads with it. Use images like the ones the camera will see.

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(model), Some(dir)) = (args.first(), args.get(1)) else {
        eprintln!("usage: npu_calibrate <model.safetensors> <images-dir> [input-size, default 320]");
        return ExitCode::from(2);
    };
    let size: u32 = match args.get(2).map(|s| s.parse()) {
        None => 320,
        Some(Ok(s)) => s,
        Some(Err(_)) => {
            eprintln!("input size must be an integer, got {:?}", args[2]);
            return ExitCode::from(2);
        }
    };
    let mut paths: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| matches!(e.to_ascii_lowercase().to_str(), Some("jpg" | "jpeg" | "png")))).collect(),
        Err(e) => {
            eprintln!("{dir}: {e}");
            return ExitCode::FAILURE;
        }
    };
    paths.sort();
    let mut images = Vec::new();
    for p in &paths {
        match std::fs::read(p).map_err(|e| e.to_string()).and_then(|b| imaging::codec::decode(&b)) {
            Ok(img) => images.push((img.px, img.w, img.h)),
            Err(e) => {
                eprintln!("{}: {e}", p.display());
                return ExitCode::FAILURE;
            }
        }
    }
    if images.is_empty() {
        eprintln!("{dir}: no JPEG or PNG images");
        return ExitCode::FAILURE;
    }
    let model = PathBuf::from(model);
    match yolov5::npu::calibrate(&model, size, &images) {
        Ok(file) => {
            println!("calibrated {} at {size}x{size} over {} images -> {}", model.display(), file.images, yolov5::npu::calibration_path(&model).display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
