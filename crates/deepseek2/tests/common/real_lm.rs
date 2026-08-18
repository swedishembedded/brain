// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared scaffolding for this crate's two real-weight test binaries
//! (`parity.rs`, the per-stage gate; `generate.rs`, the composed-loop gate).
//!
//! Included via `#[path]` rather than published as a crate module, because it
//! is pure test glue - the model-store lookup, the one-off fp32 expansion of
//! the shipped Q8_0 GGUF, the CPU-backend pin, and the inference-only build of
//! the ~2.9 B-parameter decoder. It lives here and not copy-pasted into both
//! binaries so the two can never disagree about *which* weights they ran, which
//! is the whole basis for `generate.rs` treating `parity.rs`'s verified
//! single-step argmax as the anchor of its multi-step reference.
//!
//! **Backend: CPU**, pinned before any device exists, for the reasons
//! `crates/sam1/tests/parity.rs` documents (a wgpu device-level buffer
//! corruption at production shape, and an integrated GPU whose reported
//! `max_buffer_size` is 2047 MiB against a ~12 GB working set).
//!
//! Every entry point SKIPS (returns `None` after an `eprintln!`) when the
//! checkpoint is absent - a missing real checkpoint is never a panic.

#![allow(dead_code)]

use checkpoint::weightio::WeightReader;
use deepseek2::{DeepseekV2, DeepseekV2Config};

pub const STORE: &str = "ggml-org/DeepSeek-OCR-GGUF";
pub const GGUF: &str = "DeepSeek-OCR-Q8_0.gguf";
/// Default name of the cached fp32 expansion, beside the checkpoint it came from.
pub const EXPANDED: &str = "DeepSeek-OCR-brain-fp32.safetensors";

/// Pin the CPU backend before any device is built - see the module header.
///
/// # Safety
/// Single-threaded at this point; no other test in these binaries touches
/// `BRAIN_DEVICE`.
pub fn pin_cpu_backend() {
    unsafe { std::env::set_var("BRAIN_DEVICE", "cpu") };
}

/// The real checkpoint plus the fp32 expansion path to use for it, or `None`
/// when the checkpoint is not in the model store.
pub fn paths() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let dir = std::path::PathBuf::from(brain_testutil::model_dir(STORE)?);
    let gguf = dir.join(GGUF);
    if !gguf.exists() {
        return None;
    }
    let st = std::env::var("BRAIN_DEEPSEEK_OCR_LM_ST")
        .ok()
        .filter(|s| !s.is_empty())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| dir.join(EXPANDED));
    Some((gguf, st))
}

/// The GGUF's fp32 expansion, converting it on first use.
pub fn expanded(gguf: &std::path::Path, st: &std::path::Path) -> String {
    let st_s = st.to_str().expect("utf-8 path").to_string();
    if st.exists() {
        return st_s;
    }
    println!("  expanding {} -> {} (once; ~12 GB)", gguf.display(), st.display());
    let stats = deepseek2::import::import_file(gguf.to_str().expect("utf-8 path"), &st_s, None)
        .unwrap_or_else(|e| panic!("import_file: {e}"));
    println!("  import: {stats}");
    st_s
}

/// An **inference** (`train = false`) decoder on the real LM weights, sized for
/// a `t`-token sequence at batch 1, or `None` when the checkpoint is absent.
///
/// No gradient or AdamW buffers are allocated for ~2.9 B parameters, and the
/// weights are streamed one tensor at a time from the cached expansion - they
/// never become a host `HashMap`. The shipped file's own config is asserted
/// equal to the documented preset on the way through, so a checkpoint that
/// silently changed shape fails here rather than as a numeric mismatch later.
pub fn open(t: u32) -> Option<DeepseekV2> {
    let Some((gguf, st)) = paths() else {
        brain_testutil::skip(&format!("{STORE}/{GGUF} not in the model store"));
        return None;
    };
    pin_cpu_backend();
    let cfg = deepseek2::import::config_from_file(gguf.to_str().expect("utf-8 path"), t).unwrap_or_else(|e| panic!("config_from_file: {e}"));
    assert_eq!(cfg, DeepseekV2Config::deepseek_ocr(t), "shipped file vs documented preset");
    let src = WeightReader::open(&expanded(&gguf, &st)).unwrap_or_else(|e| panic!("open expansion: {e}"));
    let gpu = gpu_core::testgpu::dev(deepseek2::PIPELINES);
    Some(DeepseekV2::new_on(gpu, cfg, 1, t, &src, false))
}
