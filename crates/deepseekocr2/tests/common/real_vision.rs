// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared scaffolding for this crate's real-weight test binaries: the model-
//! store lookup and the CPU-backend pin, mirroring `crates/deepseek2ocr`'s own
//! `tests/common/real_vision.rs` so the two crates' real-weight suites read
//! the same way. Included via `#[path]`, not published as a module - this is
//! test glue, not a second copy of the import logic (that lives in
//! `deepseekocr2::import`, and every binary here calls it directly so a
//! served path and these tests can never load different tensors under
//! different names).
//!
//! **Backend: CPU.** v1's own SAM tower is untrustworthy on wgpu past a
//! handful of resident blocks at production shape
//! (`crates/sam1/tests/parity.rs`'s own header); this crate's Qwen2 tower
//! reuses the same `sam1::SamEncoder` for its SAM half, so the same pin
//! applies here until that wgpu defect is independently re-verified fixed
//! for a 24-block-deep second tower sharing the device.
//!
//! Every entry point SKIPS (returns `None` after an `eprintln!`) when the
//! checkpoint is absent - a missing real checkpoint is never a panic.

#![allow(dead_code)]

#[allow(unused_imports)]
pub use deepseekocr2::import::{Files, LM, LM_EXPANDED, MMPROJ, VISION_EXPANDED};

/// Pin the CPU backend.
///
/// # Safety
/// Call before any device exists and while single-threaded; no test in these
/// binaries touches `BRAIN_DEVICE` afterwards.
pub fn pin_cpu_backend() {
    unsafe { std::env::set_var("BRAIN_DEVICE", "cpu") };
}

const REPO: &str = "deepseek-ai/DeepSeek-OCR-2";

/// The resolved checkpoint layout, or `None`. **Owns its skip**: a `None`
/// here has already gone through [`brain_testutil::skip`], so a caller may
/// write the bare `let Some(f) = real_files() else { return };` and still be
/// covered - the decision lives in exactly one place.
pub fn real_files() -> Option<Files> {
    let Some(dir) = brain_testutil::model_dir(REPO) else {
        brain_testutil::skip(&format!("no model store to resolve {REPO}"));
        return None;
    };
    match Files::locate(&dir) {
        Ok(f) => Some(f),
        Err(e) => {
            brain_testutil::skip(&e);
            None
        }
    }
}

/// A tensor's shape-and-sanity summary. Returns the finite count so a caller
/// can assert on it.
pub fn describe(name: &str, v: &[f32]) -> usize {
    let finite = v.iter().filter(|x| x.is_finite()).count();
    let (mut lo, mut hi, mut sq) = (f32::INFINITY, f32::NEG_INFINITY, 0f64);
    for x in v.iter().filter(|x| x.is_finite()) {
        lo = lo.min(*x);
        hi = hi.max(*x);
        sq += (*x as f64) * (*x as f64);
    }
    println!("  {name:<22} n={:<8} finite={finite:<8} min={lo:>9.4} max={hi:>9.4} rms={:.5}", v.len(), (sq / finite.max(1) as f64).sqrt());
    finite
}
