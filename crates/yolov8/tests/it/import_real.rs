// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Opt-in end-to-end test of `yolov8::import::import_yolov8n` against a REAL,
//! unmodified Ultralytics checkpoint -- the decisive proof no synthetic
//! fixture can give, since it exercises `checkpoint::torchpt`'s
//! generic-unrecognized-class fallback against the actual pickled
//! `ultralytics.nn.tasks.DetectionModel` object graph, not a hand-built one.
//!
//! Confirmed passing this session against a freshly-downloaded
//! `https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt`
//! (297 tensors, exact match against `YoloConfig::yolov8n().full_param_list()`).
//!
//! ## Gating
//! Needs a real `yolov8n.pt` on disk (no torch/network required to RUN the
//! test, only to have obtained the file beforehand):
//!
//! ```text
//! YOLO_RAW_PT=/path/to/yolov8n.pt cargo test -p brain-yolo --test import_real -- --nocapture
//! ```
//!
//! When `YOLO_RAW_PT` is unset or the file is missing, the test prints a skip
//! notice and returns OK (so plain `cargo test` is green everywhere, matching
//! `crates/yolo/tests/parity.rs`'s convention for the same reason).

#[test]
fn imports_a_real_yolov8n_checkpoint_with_exact_coverage() {
    let path = match std::env::var("YOLO_RAW_PT") {
        Ok(p) if std::path::Path::new(&p).is_file() => p,
        Ok(p) => {
            brain_testutil::skip(&format!("imports_a_real_yolov8n_checkpoint_with_exact_coverage: YOLO_RAW_PT={p:?} does not exist"));
            return;
        }
        Err(_) => {
            brain_testutil::skip("imports_a_real_yolov8n_checkpoint_with_exact_coverage: set YOLO_RAW_PT to a real yolov8n.pt");
            return;
        }
    };

    let tensors = yolov8::import::import_yolov8n(&path).expect("import must succeed against a real, unmodified yolov8n.pt");

    let expected = yolov8::config::YoloConfig::yolov8n().full_param_list();
    assert_eq!(tensors.len(), expected.len(), "tensor count must match YoloConfig::yolov8n().full_param_list() exactly");

    let expected_by_name: std::collections::BTreeMap<&str, usize> = expected.iter().map(|(n, c)| (n.as_str(), *c)).collect();
    for (name, shape, data) in &tensors {
        let &want = expected_by_name.get(name.as_str()).unwrap_or_else(|| panic!("unexpected tensor {name:?}"));
        assert_eq!(data.len(), want, "{name}: element count");
        assert_eq!(shape.iter().product::<usize>(), want, "{name}: shape {shape:?} does not match its own element count");
        assert!(data.iter().all(|v| v.is_finite()), "{name}: contains a non-finite value");
    }
}

/// A real `yolov8n.pt` served as downloaded: `Yolo::load` reads the `.pt`
/// itself and computes exactly what the same weights converted to a brain
/// checkpoint compute.
#[test]
fn a_real_yolov8n_pt_serves_as_its_conversion_does() {
    let Some(path) = std::env::var("YOLO_RAW_PT").ok().filter(|p| std::path::Path::new(p).is_file()) else {
        brain_testutil::skip("a_real_yolov8n_pt_serves_as_its_conversion_does: set YOLO_RAW_PT to a real yolov8n.pt");
        return;
    };
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let tensors = yolov8::import::import_yolov8n(&path).unwrap();
    let converted = std::env::temp_dir().join(format!("yolov8n-converted-{}.safetensors", std::process::id()));
    let st: Vec<(String, Vec<u64>, Vec<f32>)> = tensors.into_iter().map(|(n, s, d)| (n, s.into_iter().map(|x| x as u64).collect(), d)).collect();
    checkpoint::st::save_safetensors(converted.to_str().unwrap(), &st, &yolov8::config::YoloConfig::yolov8n().to_json(), None).unwrap();

    let side = yolov8::config::YoloConfig::yolov8n().input as usize;
    let image: Vec<f32> = (0..3 * side * side).map(|i| ((i * 37 % 255) as f32) / 255.0).collect();
    let logits = |m: yolov8::Yolo| {
        m.set_eval(true);
        m.set_image(&image);
        m.forward_net_pub();
        let (cls, reg) = m.raw_logits();
        cls.into_iter().chain(reg).map(f32::to_bits).collect::<Vec<_>>()
    };
    assert_eq!(logits(yolov8::Yolo::load(&path, 1)), logits(yolov8::Yolo::load(converted.to_str().unwrap(), 1)));
    std::fs::remove_file(&converted).ok();
}
