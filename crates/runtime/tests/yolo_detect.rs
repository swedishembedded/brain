// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end: a real (tiny, random-weight) YOLO detector plugged into the
//! Controller produces an `object_detected` event for a `camera_frame`. Gated
//! behind `MOE_SKIP_GPU_TESTS` since a tiny YOLO forward on the CPU JIT is slow.

use events::{base64, Event};
use runtime::{Controller, FakeInferModel, GenConfig, Registry, YoloDetect};

#[test]
fn real_tiny_yolo_emits_object_detected() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    // Random-weight tiny YOLO (no training): we only assert the controller emits
    // an `object_detected` event with finite box coordinates (not the contents).
    let cfg = yolov8::YoloConfig::tiny(3);
    let side = cfg.input; // square model input
    let init = yolov8::init_weights(&cfg, 1234);
    let m = yolov8::Yolo::new(cfg, 1, 0, &init);
    let detect = Box::new(YoloDetect::from_model(m).with_thresholds(0.0, 0.45));

    let infer = Box::new(FakeInferModel::echoing("x"));
    let reg = Registry::with_models(infer, detect);
    let mut ctrl = Controller::with_config(
        reg,
        GenConfig { max_new: 8, temperature: 0.0, top_k: 0, eos: Some(256), seed: 0 },
    );

    // A square rgb8 frame matching the model's input size (mid-grey pixels).
    let (w, h) = (side, side);
    let px = vec![128u8; (w * h * 3) as usize];
    let frame = Event::CameraFrame {
        format: "rgb8".into(),
        w,
        h,
        data: Some(base64::encode(&px)),
        path: None,
    };
    let env_out = ctrl.feed_line(&events::encode_line(&frame));
    let out: Vec<Event> = env_out.iter().map(|e| e.event.clone()).collect();

    let detected: Vec<&Event> =
        out.iter().filter(|e| matches!(e, Event::ObjectDetected { .. })).collect();
    assert_eq!(detected.len(), 1, "expected exactly one object_detected, got {out:?}");
    if let Event::ObjectDetected { dets, .. } = detected[0] {
        // boxes (if any) must be finite — the adapter + decode must not produce
        // NaN/inf even on random weights.
        for d in dets {
            for v in d {
                assert!(v.is_finite(), "non-finite detection value: {d:?}");
            }
        }
    }
}
