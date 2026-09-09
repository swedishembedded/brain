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
//! **Both backends are reachable**, and the device is an argument rather than
//! a process-global pin: [`open_on`] takes the `Gpu` the caller built, so one
//! test binary can gate the CPU Cranelift JIT and a real wgpu/Vulkan card
//! against the same llama.cpp reference without either run's `BRAIN_DEVICE`
//! write deciding the other's backend. [`cpu`] and [`wgpu`] are the two
//! devices this crate's real-weight tests compare; `wgpu` returns `None` (and
//! skips) on a box with no discrete card, where a software rasteriser would
//! only be comparing the host against itself.
//!
//! Every entry point SKIPS (returns `None` after an `eprintln!`) when the
//! checkpoint is absent - a missing real checkpoint is never a panic.

#![allow(dead_code)]

use checkpoint::weightio::WeightReader;
use deepseek2::{DeepseekV2, DeepseekV2Config};
use gpu_core::Gpu;

pub const STORE: &str = "ggml-org/DeepSeek-OCR-GGUF";
pub const GGUF: &str = "DeepSeek-OCR-Q8_0.gguf";
/// Default name of the cached fp32 expansion, beside the checkpoint it came from.
pub const EXPANDED: &str = "DeepSeek-OCR-brain-fp32.safetensors";

/// The CPU Cranelift JIT, named explicitly rather than selected through
/// `BRAIN_DEVICE` - a process-global env write cannot express "this build on
/// the CPU, that one on the card" in one test binary, which is exactly what
/// the cross-backend gates below need.
pub fn cpu() -> Gpu {
    Gpu::new_cpu(deepseek2::PIPELINES)
}

/// A real wgpu/Vulkan device, or `None` (having skipped) when this box has no
/// discrete card - on a software rasteriser a "GPU" run would compare the host
/// against itself and prove nothing.
pub fn wgpu() -> Option<Gpu> {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        brain_testutil::skip("MOE_SKIP_GPU_TESTS");
        return None;
    }
    if gpu_core::discrete_gpu_count() == 0 {
        brain_testutil::skip("no discrete GPU - a software rasteriser would compare the host against itself");
        return None;
    }
    Some(Gpu::new_wgpu(deepseek2::PIPELINES))
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
///
/// A cached expansion whose tensor names are not the ones `cfg`'s manifest
/// asks for is REBUILT rather than used: it is derived, so a stale one (left
/// by a build with a different tensor layout) is a cache to refresh, not a
/// checkpoint to refuse.
pub fn expanded(gguf: &std::path::Path, st: &std::path::Path, cfg: &DeepseekV2Config) -> String {
    let st_s = st.to_str().expect("utf-8 path").to_string();
    if st.exists() {
        let want = cfg.param_list();
        match WeightReader::open(&st_s) {
            Ok(r) if want.iter().all(|(n, _)| r.shape(n).is_some()) => return st_s,
            _ => {
                println!("  {st_s}: cached expansion does not match this build's tensor layout - rebuilding");
                std::fs::remove_file(st).expect("removing the stale expansion");
            }
        }
    }
    println!("  expanding {} -> {} (once; ~12 GB)", gguf.display(), st.display());
    let stats = deepseek2::import::import_file(gguf.to_str().expect("utf-8 path"), &st_s, None)
        .unwrap_or_else(|e| panic!("import_file: {e}"));
    println!("  import: {stats}");
    st_s
}

/// [`open_on`] on the CPU Cranelift JIT.
pub fn open(t: u32) -> Option<DeepseekV2> {
    open_on(t, cpu())
}

/// An **inference** (`train = false`) decoder on the real LM weights, built on
/// `gpu` and sized for a `t`-token sequence at batch 1, or `None` when the
/// checkpoint is absent.
///
/// No gradient or AdamW buffers are allocated for ~2.9 B parameters, and the
/// weights are streamed one tensor at a time from the cached expansion - they
/// never become a host `HashMap`. The shipped file's own config is asserted
/// equal to the documented preset on the way through, so a checkpoint that
/// silently changed shape fails here rather than as a numeric mismatch later.
pub fn open_on(t: u32, gpu: Gpu) -> Option<DeepseekV2> {
    let Some((gguf, st)) = paths() else {
        brain_testutil::skip(&format!("{STORE}/{GGUF} not in the model store"));
        return None;
    };
    let cfg = deepseek2::import::config_from_file(gguf.to_str().expect("utf-8 path"), t).unwrap_or_else(|e| panic!("config_from_file: {e}"));
    assert_eq!(cfg, DeepseekV2Config::deepseek_ocr(t), "shipped file vs documented preset");
    let src = WeightReader::open(&expanded(&gguf, &st, &cfg)).unwrap_or_else(|e| panic!("open expansion: {e}"));
    Some(DeepseekV2::new_on(gpu, cfg, 1, t, &src, false))
}
