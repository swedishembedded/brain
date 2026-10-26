// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of object detectors for its clients. If your team needs
// expertise in training vision models within a fixed device budget, you can procure our services by sending an email
// to info@swedishembedded.com.

//! Activation recompute through the shared contract (`model::saved::check_recompute`): a detector that keeps only each
//! conv unit's output and rebuilds the unit's post-conv and post-BatchNorm intermediates from its input before its
//! backward (the normalisation replayed with the batch statistics the forward kept) computes the loss and gradients
//! of the build that keeps every intermediate, bit for bit, at a lower step peak. One test in its own binary, so
//! nothing else allocates while the device bytes are compared.

use yolov8::{GtBox, LossMode, Yolo, YoloConfig};

fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| optim::with_moment_kernels(yolov8::net::PIPELINES))
}

#[test]
fn yolov8_recomputes_its_units_exactly() {
    let nc = 3u32;
    let cfg = YoloConfig::tiny(nc);
    let init = yolov8::init_weights(&cfg, 1234);
    let pixels = (3 * cfg.input * cfg.input) as usize;
    let image: Vec<f32> = (0..2 * pixels).map(|i| ((i * 37 % 101) as f32) / 100.0).collect();
    let gts = [
        GtBox { img: 0, cls: 0, cx: 0.30, cy: 0.35, w: 0.25, h: 0.30 },
        GtBox { img: 1, cls: 2, cx: 0.70, cy: 0.65, w: 0.30, h: 0.25 },
    ];
    let build = || Yolo::new_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 2, cfg.input, &init);
    let step = |m: &Yolo| {
        m.set_mode(LossMode::Detection);
        m.zero_grads();
        m.set_image(&image);
        m.set_targets(&gts);
        let loss = m.forward();
        m.backward();
        m.poll_wait();
        loss
    };
    model::saved::check_recompute("yolov8", build, step).unwrap();
}
