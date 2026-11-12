// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The OpenSeeFace port against the released graphs run by onnxruntime
//! (`tools/goldens/openseeface_dump_reference.py`).
//!
//! Fixtures, both outside git: the released `.onnx` files in the model store
//! under `emilianavt/OpenSeeFace`, and the goldens under
//! `testdata/face/openseeface`. Every test skips itself, naming what is
//! missing, when either is absent.
//!
//! ```bash
//! make test/one P=brain-openseeface T=it ARGS=--nocapture
//! ```

mod import;
mod parity;
mod pipeline;

use std::collections::HashMap;
use std::path::PathBuf;

use brain_testutil::golden::Source;
use openseeface::config::{NetConfig, DETECTOR, LANDMARKS_BEST, LANDMARKS_FAST};

pub const DUMPER: &str = "tools/goldens/openseeface_dump_reference.py";

/// The model-store directory holding the released graphs, or `None` (having
/// skipped) when they are not there.
pub fn weights() -> Option<PathBuf> {
    let d = PathBuf::from(brain_testutil::model_dir("emilianavt/OpenSeeFace").unwrap_or_default());
    for cfg in [DETECTOR, LANDMARKS_FAST, LANDMARKS_BEST] {
        if !d.join(cfg.file).exists() {
            brain_testutil::skip(&format!("{} not in {} (samples/vision/fishtank/fetch-data.sh fetches it)", cfg.file, d.display()));
            return None;
        }
    }
    Some(d)
}

/// The golden directory, its provenance checked against the graphs' identity.
pub fn goldens() -> Option<PathBuf> {
    let d = PathBuf::from(brain_testutil::testdata("face/openseeface"));
    let src = Source::open(&d, DUMPER)?;
    let identity = [("input", 224), ("landmarks", 66), ("detector_stem", DETECTOR.stem as i64), ("lm0_stem", LANDMARKS_FAST.stem as i64), ("lm3_stem", LANDMARKS_BEST.stem as i64)];
    src.require(&identity).then_some(d)
}

pub type Golden = HashMap<String, (Vec<usize>, Vec<f32>)>;

pub fn golden(file: &str) -> Option<Golden> {
    let d = goldens()?;
    let t = checkpoint::safetensors::read(d.join(file).to_str().expect("utf-8 path")).unwrap_or_else(|e| panic!("read {file}: {e}"));
    Some(t.into_iter().map(|t| (t.name, (t.shape, t.data))).collect())
}

/// The three released networks and the golden file each is dumped into.
pub const NETS: [(&str, NetConfig); 3] = [("detector", DETECTOR), ("lm0", LANDMARKS_FAST), ("lm3", LANDMARKS_BEST)];

/// Print and gate a comparison on cosine AND max_abs: cosine cannot see a
/// scale error, max_abs cannot see a rotation.
pub fn gate(label: &str, got: &[f32], want: &[f32], min_cos: f32, max_abs: f32) {
    assert_eq!(got.len(), want.len(), "{label}: {} values vs golden {}", got.len(), want.len());
    let cos = model::hostmath::cosine(got, want);
    let abs = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("  {label:<28} cos {cos:.7}  max_abs {abs:.3e}  (n={})", got.len());
    assert!(cos >= min_cos && abs <= max_abs, "{label}: cos {cos} (need >= {min_cos}), max_abs {abs} (need <= {max_abs})");
}
