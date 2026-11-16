// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `--target qwen35moe:<gguf>[:kvf32|:kvbf16|:kvint8]` for the request-level
//! `brain perf run` scenarios (`sweep`, `serve`, ...): the released
//! Qwen3.6-35B-A3B GGUF served exactly as `brain serve` serves it.
//!
//! The engine is configured by `catalog::resident_qwen35moe::gguf_resident`
//! (weight tier, KV tier, batch and context planned for the card) and put under
//! the scheduler by `Qwen35MoeGgufResident::activate_scheduler`, the two
//! functions the server calls, so goodput, TTFT and KV-pressure numbers are the
//! served ones and compare with `qwen35:`'s. `--weight-format` and the `:kv<tier>`
//! suffix pin what they name; everything else is the server's own default.
//!
//! Swedish Embedded AB implements solutions for measuring LLM serving
//! performance against production behaviour for its clients. If your team needs
//! expertise in serving and benchmarking sparse mixture-of-experts models on
//! GPUs then you can procure our services by sending an email to
//! info@swedishembedded.com.

use perf::target::{PerfTarget, TargetInfo};
use residency::ResidentModel;

/// The engine target for one released GGUF, on the card of the `--device` set the server would plan it for.
pub(crate) fn build(rest: &str, weights: Option<&model::ops::TierPolicy>, max_batch: Option<u32>) -> Result<Box<dyn PerfTarget>, String> {
    let (path, kv) = crate::perf_qwen35::parse_spec(rest).map_err(|e| e.replace("qwen35:", "qwen35moe:"))?;
    let checkpoint = std::path::Path::new(path);
    if !checkpoint.is_file() {
        return Err(format!("qwen35moe target: GGUF not found: {path}"));
    }
    let machine = serving::Machine::probe(crate::compute_set());
    let id = checkpoint.file_stem().map_or_else(|| path.to_string(), |s| s.to_string_lossy().into_owned());
    let cfg = catalog::resident_qwen35moe::Qwen35MoeGgufServeConfig { kv, tier: weights.cloned(), max_batch, ..Default::default() };
    let served = catalog::resident_qwen35moe::gguf_resident(path, &id, &machine.gpus, crate::perf_qwen35::RESERVE_BYTES, &cfg)?;
    let device = served.planned_device();
    let (sched, _tokenizer, _stop) = served.activate_scheduler(device)?;
    let vocab = model::serve::PagedDecoder::vocab(sched.decoder()) as u32;
    let manifest = served.manifest();
    let mut info = TargetInfo::new("qwen35moe", "token")
        .with("weights", path.into())
        .with("max_batch", served.max_batch().into())
        .with("ctx", served.cap().into())
        .with("kv_dtype", served.kv_tier().as_str().into())
        .with("devices", vec![format!("{device:?}")].into())
        .with("summary", manifest.actions.first().map(|a| a.summary.clone()).unwrap_or_default().into());
    for (i, line) in served.chosen_defaults().iter().enumerate() {
        info = info.with(&format!("default_{i}"), line.as_str().into());
    }
    // Synthetic runs must produce the requested length: no stop ids.
    Ok(Box::new(perf::targets::PagedLlmTarget::new(sched, info, Vec::new(), vocab)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spec_without_a_checkpoint_names_this_target_not_the_dense_one() {
        let e = build(":kvint8", None, None).err().expect("an empty path is refused");
        assert!(e.contains("qwen35moe:") && !e.contains("qwen35:"), "{e}");
    }

    #[test]
    fn a_missing_checkpoint_is_reported_before_any_device_work() {
        let e = build("/nonexistent/q.gguf", None, None).err().expect("a missing file is refused");
        assert!(e.contains("GGUF not found"), "{e}");
    }
}
