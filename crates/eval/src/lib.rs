// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Unified evaluation harness — the same metrics across models, so different
//! architectures can be compared on identical data (README §3 discipline: hold
//! the input distribution fixed, separate the metric from how you test). A
//! model's own task metrics live with the model (`gpt2::eval`,
//! `lfm2::train::evaluate`, `yolov8::evaluate`).

/// Calibration metrics for a decision model's probability output (ECE,
/// AdaECE, classwise-ECE, NLL, squared-error probability score, reliability bins, coverage-vs-accuracy,
/// failure-AUROC) - re-exported from the leaf layer (`crates/rlcd`) rather
/// than owned here, so a sample can use them without this crate's own
/// heavier dependency closure.
pub use rlcd::metrics as calibration;
// The detection metrics live with the detector they score (`yolov8` serves its own
// `eval` action), so they cannot depend on this crate; the paths stay valid here.
pub use yolov8::{detection, detection_report};

pub mod asr;
pub mod tts;
