// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `--target qwen35moe:<gguf>[:kvf32|:kvbf16|:kvint8]` for the request-level
//! `brain perf run` scenarios (`sweep`, `serve`, ...): the released
//! Qwen3.6-35B-A3B GGUF behind `qwen35moe::serve::Engine` and the shared
//! `model::serve::Scheduler`, so its goodput, TTFT and KV-pressure numbers are
//! measured the way `qwen35:` measures the dense model and can be compared with
//! it. The weight tier comes from the engine's own loader (`BRAIN_QWEN35MOE_TIER`,
//! default int8) and the KV tier from the spec or `BRAIN_QWEN35_KV`.
//!
//! Swedish Embedded AB implements solutions for measuring LLM serving
//! performance against production behaviour for its clients. If your team needs
//! expertise in serving and benchmarking sparse mixture-of-experts models on
//! GPUs then you can procure our services by sending an email to
//! info@swedishembedded.com.

use checkpoint::gguf::MmapGguf;
use perf::target::{PerfTarget, TargetInfo};

/// Longest `prompt + max_new` of one sequence. This engine reserves a whole
/// block of this many rows per resident sequence, so it is the sizing knob.
const CONTEXT: u32 = 8192;

/// Sequences resident at once; the sweep's concurrency ladder tops out here.
const MAX_BATCH: u32 = 16;

/// The engine target for one released GGUF, on the first `--device` card.
pub(crate) fn build(rest: &str) -> Result<Box<dyn PerfTarget>, String> {
    let (path, kv) = crate::perf_qwen35::parse_spec(rest).map_err(|e| e.replace("qwen35:", "qwen35moe:"))?;
    if !std::path::Path::new(path).is_file() {
        return Err(format!("qwen35moe target: GGUF not found: {path}"));
    }
    let kv_tier = match kv {
        Some(kv) => kv,
        None => qwen35moe::gguf_load::kv_tier_from_env()?,
    };
    let tier = qwen35moe::gguf_load::tier_from_env();
    let mg = MmapGguf::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let cfg = qwen35moe::gguf_load::resident_config(&mg, CONTEXT)?;
    let src = qwen35moe::gguf_load::source(&mg, &cfg)?;
    let opts = qwen35moe::serve::EngineOptions::new(CONTEXT, MAX_BATCH).with_tier(tier).with_kv_tier(kv_tier);
    let engine = qwen35moe::serve::Engine::from_source(cfg, &src, opts);
    let vocab = model::serve::PagedDecoder::vocab(&engine) as u32;
    let info = TargetInfo::new("qwen35moe", "token")
        .with("weights", path.into())
        .with("max_batch", MAX_BATCH.into())
        .with("ctx", CONTEXT.into())
        .with("kv_dtype", kv_tier.as_str().into());
    let sched = qwen35moe::serve::Scheduler::new(engine, MAX_BATCH as usize);
    // Synthetic runs must produce the requested length: no stop ids.
    Ok(Box::new(perf::targets::PagedLlmTarget::new(sched, info, Vec::new(), vocab)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spec_without_a_checkpoint_names_this_target_not_the_dense_one() {
        let e = build(":kvint8").err().expect("an empty path is refused");
        assert!(e.contains("qwen35moe:") && !e.contains("qwen35:"), "{e}");
    }

    #[test]
    fn a_missing_checkpoint_is_reported_before_any_device_work() {
        let e = build("/nonexistent/q.gguf").err().expect("a missing file is refused");
        assert!(e.contains("GGUF not found"), "{e}");
    }
}
