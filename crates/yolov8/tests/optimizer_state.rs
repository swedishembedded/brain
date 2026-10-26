// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of object detectors for edge devices for its clients.
// If your team needs expertise in training vision models with narrow optimizer state, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The run's optimizer-state precision reaches YOLOv8 through the generic path every `ParamStore` trainer shares
//! (`model::trainprec::check_optimizer_state_formats`): bf16 and int8-block moments train like f32 on the first step,
//! and a host-only format is refused by name.

use yolov8::{GtBox, LossMode, Yolo, YoloConfig};

fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| optim::with_moment_kernels(yolov8::net::PIPELINES))
}

#[test]
fn yolov8_trains_with_narrow_optimizer_state() {
    let nc = 3u32;
    let cfg = YoloConfig::tiny(nc);
    let init = yolov8::init_weights(&cfg, 1234);
    let pixels = (3 * cfg.input * cfg.input) as usize;
    let image: Vec<f32> = (0..pixels).map(|i| ((i * 37 % 101) as f32) / 100.0).collect();
    let gts = [
        GtBox { img: 0, cls: 0, cx: 0.30, cy: 0.35, w: 0.25, h: 0.30 },
        GtBox { img: 0, cls: 2, cx: 0.70, cy: 0.65, w: 0.30, h: 0.25 },
    ];
    let lr = 1e-3;
    let build = || Yolo::new_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 1, cfg.input, &init);
    let step = |m: &Yolo| {
        m.set_mode(LossMode::Detection);
        m.zero_grads();
        m.set_image(&image);
        m.set_targets(&gts);
        m.forward();
        m.backward();
        m.adamw_step(1, lr, 0.0, model::Adam::default(), None, 1.0);
        m.poll_wait();
    };
    model::trainprec::check_optimizer_state_formats("yolov8", build, step, lr).unwrap();
}
