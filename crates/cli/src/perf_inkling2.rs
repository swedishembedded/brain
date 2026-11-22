// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `--target inkling2:<dir>[:kvf32|:kvbf16]` for the request-level `brain perf run` scenarios: a pulled Inkling 2
//! directory served exactly as `brain serve` serves it.
//!
//! The engine is sized by `catalog::resident_inkling2::Inkling2Resident` (context, K/V tier, the visit pool and the
//! batch the server would take from the same settings) and put under the scheduler by `Inkling2Resident::scheduler`,
//! the builder its `activate` uses, so throughput, TTFT and decode rates are the served ones. `--max-batch` and the
//! `:kv<tier>` suffix pin what they name; everything else is the server's default.
//!
//! Swedish Embedded AB implements solutions for measuring LLM serving performance against production behaviour for
//! its clients. If your team needs expertise in serving looped mixture-of-experts models on GPUs then you can procure
//! our services by sending an email to info@swedishembedded.com.

use perf::target::{PerfTarget, TargetInfo};

/// The engine target for one Inkling 2 directory on the current device; `max_batch` (`--max-batch`) is the number of
/// sequences that run at once.
pub(crate) fn build(rest: &str, max_batch: Option<u32>) -> Result<Box<dyn PerfTarget>, String> {
    let (path, kv) = crate::perf_qwen35::parse_spec(rest).map_err(|e| e.replace("qwen35:", "inkling2:"))?;
    if !std::path::Path::new(path).is_dir() {
        return Err(format!("inkling2 target: checkpoint directory not found: {path}"));
    }
    let mut sizing = catalog::resident_qwen35::Qwen35Sizing { kv_cache: kv, ..Default::default() };
    if let Some(batch) = max_batch {
        sizing.max_batch = batch;
    }
    let served = catalog::resident_inkling2::Inkling2Resident::from_hf_dir(path, "perf/inkling2", sizing)?;
    let opts = served.options();
    let mut sched = served.scheduler()?;
    let vocab = model::serve::PagedDecoder::vocab(sched.decoder_mut()) as u32;
    let info = TargetInfo::new("inkling2", "token")
        .with("weights", path.into())
        .with("ctx", (opts.max_seq_len as u64).into())
        .with("kv_dtype", opts.kv_tier.as_str().into())
        .with("max_batch", opts.sizing.live_rows.into())
        .with("prefill_chunk", opts.prefill_chunk.into());
    // Synthetic runs must produce the requested length: no stop ids.
    Ok(Box::new(perf::targets::PagedLlmTarget::new(sched, info, Vec::new(), vocab)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spec_is_refused_naming_this_target_before_any_device_work() {
        let e = build(":kvbf16", None).err().expect("an empty path is refused");
        assert!(e.contains("inkling2:") && !e.contains("qwen35:"), "{e}");
        let e = build("/nonexistent/Inkling-2-Mini:kvbf16", None).err().expect("a missing directory is refused");
        assert!(e.contains("/nonexistent/Inkling-2-Mini"), "{e}");
    }
}
