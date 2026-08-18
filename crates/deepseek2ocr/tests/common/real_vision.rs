// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared scaffolding for this crate's real-weight test binaries.
//!
//! Included via `#[path]` rather than published as a crate module, because what
//! is left here IS pure test glue: the model-store lookup, the CPU-backend pin,
//! the synthetic page these tests feed, and the tensor summaries they print. It
//! lives here and not copy-pasted into each binary so they can never disagree
//! about *which* weights they ran, which is exactly what lets one test treat
//! another's verified numbers as its anchor.
//!
//! **The import itself is no longer here.** The mmproj → init-map pass moved to
//! `deepseek2ocr::import`, production code, the moment the served path needed the
//! same three steps; [`encoder_weights`] below is a one-line wrapper over it, so
//! `brain serve` and these tests provably load the same tensors under the same
//! names. The path constants are re-exported from there for the same reason.
//!
//! **Backend: CPU**, pinned before any device exists, for the reason
//! `crates/sam1/tests/parity.rs` documents: at 1024x1024 the SAM tower's
//! per-block buffers go wrong on wgpu as soon as the graph holds three or more
//! blocks, so the tower cannot presently be trusted there at production shape.
//!
//! Every entry point SKIPS (returns `None` after an `eprintln!`) when the
//! checkpoint is absent - a missing real checkpoint is never a panic.

#![allow(dead_code)]

use std::collections::HashMap;

use checkpoint::gguf::MmapGguf;

// `#![allow(dead_code)]` covers an unused *item*, but not an unused *re-export*
// - and each binary that `#[path]`-includes this file uses a different subset.
#[allow(unused_imports)]
pub use deepseek2ocr::import::{EXPANDED, LM, MMPROJ, STORE};

/// Pin the CPU backend.
///
/// # Safety
/// Call before any device exists and while single-threaded; no test in these
/// binaries touches `BRAIN_DEVICE` afterwards.
pub fn pin_cpu_backend() {
    unsafe { std::env::set_var("BRAIN_DEVICE", "cpu") };
}

/// The model-store directory holding the real DeepSeek-OCR checkpoints, or
/// `None`.
///
/// **These helpers own the skip.** A `None` here has already gone through
/// [`brain_testutil::skip`], so it has printed its reason and, under
/// `BRAIN_REQUIRE_FIXTURES=1`, has already panicked. A caller may therefore
/// write the bare `let Some(d) = store_dir() else { return };` and still be
/// covered - the decision lives in exactly one place. It used to live in two:
/// the helper printed and the caller returned in silence, so a caller that
/// forgot to print was indistinguishable from one that ran.
pub fn store_dir() -> Option<std::path::PathBuf> {
    match brain_testutil::model_dir(STORE).map(std::path::PathBuf::from) {
        Some(d) => Some(d),
        None => {
            brain_testutil::skip("no model store (set BRAIN_MODELS_DIR or HOME)");
            None
        }
    }
}

/// The mmproj's path, or `None`. Owns its skip - see [`store_dir`].
pub fn mmproj_path() -> Option<std::path::PathBuf> {
    let p = store_dir()?.join(MMPROJ);
    if !p.exists() {
        brain_testutil::skip(&format!("{} absent", p.display()));
        return None;
    }
    Some(p)
}

/// The encoder's whole init map - `deepseek2ocr::import::encoder_weights`, the
/// production path, with its `Result` turned into the panic a test wants.
pub fn encoder_weights(mg: &MmapGguf) -> HashMap<String, Vec<f32>> {
    deepseek2ocr::import::encoder_weights(mg).unwrap_or_else(|e| panic!("mmproj import: {e}"))
}

/// `MemAvailable` in GiB, read live from `/proc/meminfo`.
///
/// The composite stages below have a measured ~24 GiB construction peak (read
/// off `/proc/self/status`, not estimated - the same `brain_testutil::mem`
/// reporter these tests print), so on a box that cannot host one the honest
/// outcome is a printed skip, not an OOM kill that takes whatever else is
/// running down with it.
pub fn mem_available_gib() -> f64 {
    let Ok(s) = std::fs::read_to_string("/proc/meminfo") else { return f64::INFINITY };
    s.lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.split_whitespace().next())
        .and_then(|kb| kb.parse::<f64>().ok())
        .map(|kb| kb / (1024.0 * 1024.0))
        .unwrap_or(f64::INFINITY)
}

/// Headroom a composite (encoder + the 2.9 B decoder) stage needs before it is
/// worth starting.
pub const DECODER_GIB: f64 = 16.0;

/// A deliberately awkward source extent for [`synthetic_page`]: not square, not
/// a multiple of 1024, and landscape, so the letterbox bars land on the TOP and
/// BOTTOM.
pub const SRC_W: u32 = 1600;
pub const SRC_H: u32 = 1131;

/// A synthetic scanned page: a light ground with darker horizontal bars of
/// varying length and one vertical rule, plus a coloured margin block so a
/// channel swap cannot hide. Structured, not noise - noise resizes to a flat
/// mean and would be nearly as blind as the constant fill the parity tests use.
pub fn synthetic_page(w: u32, h: u32) -> Vec<f32> {
    let mut v = vec![0f32; 3 * (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = 3 * (y * w + x) as usize;
            // Paper.
            let mut px = [0.93f32, 0.92, 0.89];
            // Text lines: 24 px tall bands every 48 px, each a different length.
            let line = y / 48;
            let in_band = (y % 48) < 24 && (60..h - 60).contains(&y);
            let len = 200 + (line * 137) % (w - 400);
            if in_band && (80..80 + len).contains(&x) {
                px = [0.10, 0.11, 0.13];
            }
            // A vertical rule at one third, so an h/w transpose cannot survive.
            if (w / 3..w / 3 + 6).contains(&x) {
                px = [0.35, 0.36, 0.40];
            }
            // A saturated margin block: R, G and B all differ, in the top-left.
            if x < 120 && y < 90 {
                px = [0.85, 0.25, 0.10];
            }
            v[i..i + 3].copy_from_slice(&px);
        }
    }
    v
}

/// A tensor's shape-and-sanity summary. Returns the finite count so a caller can
/// assert on it.
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
