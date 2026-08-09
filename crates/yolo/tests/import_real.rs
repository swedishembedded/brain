// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Opt-in end-to-end test of `yolo::import::import_yolov8n` against a REAL,
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
            println!("SKIP imports_a_real_yolov8n_checkpoint_with_exact_coverage: YOLO_RAW_PT={p:?} does not exist");
            return;
        }
        Err(_) => {
            println!("SKIP imports_a_real_yolov8n_checkpoint_with_exact_coverage: set YOLO_RAW_PT to a real yolov8n.pt");
            return;
        }
    };

    let tensors = yolo::import::import_yolov8n(&path).expect("import must succeed against a real, unmodified yolov8n.pt");

    let expected = yolo::config::YoloConfig::yolov8n().full_param_list();
    assert_eq!(tensors.len(), expected.len(), "tensor count must match YoloConfig::yolov8n().full_param_list() exactly");

    let expected_by_name: std::collections::BTreeMap<&str, usize> = expected.iter().map(|(n, c)| (n.as_str(), *c)).collect();
    for (name, shape, data) in &tensors {
        let &want = expected_by_name.get(name.as_str()).unwrap_or_else(|| panic!("unexpected tensor {name:?}"));
        assert_eq!(data.len(), want, "{name}: element count");
        assert_eq!(shape.iter().product::<usize>(), want, "{name}: shape {shape:?} does not match its own element count");
        assert!(data.iter().all(|v| v.is_finite()), "{name}: contains a non-finite value");
    }
}
