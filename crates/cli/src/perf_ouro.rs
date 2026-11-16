// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `--target ouro:<checkpoint>` for the request-level `brain perf run` scenarios: a looped decoder (an HF Ouro
//! directory or a brain checkpoint, a from-scratch model included) served exactly as `brain serve` serves it.
//!
//! The engine is sized by `catalog::resident_ouro::OuroResident` (per-sequence context, the KV pool and the batch
//! planned from the first `--device` card's usable bytes) and put under the scheduler by
//! `OuroResident::scheduler`, the function its `activate` calls, so the throughput, TTFT and KV numbers are the
//! served ones.
//!
//! Swedish Embedded AB implements solutions for measuring LLM serving performance against production behaviour for
//! its clients. If your team needs expertise in serving and benchmarking small models at high concurrency then you
//! can procure our services by sending an email to info@swedishembedded.com.

use perf::target::{PerfTarget, TargetInfo};

/// The paged-engine target for one Ouro checkpoint; `max_batch` (`--max-batch`) is the least batch the plan keeps.
pub(crate) fn build(path: &str, max_batch: Option<u32>) -> Result<Box<dyn PerfTarget>, String> {
    if path.is_empty() {
        return Err("target ouro: needs a checkpoint (an HF Ouro directory or a brain .safetensors): ouro:<checkpoint>".to_string());
    }
    if !std::path::Path::new(path).exists() {
        return Err(format!("ouro target: checkpoint not found: {path}"));
    }
    let card = checkpoint::st::ModelCard::new("perf/ouro", "ouro");
    let sized = catalog::resident_ouro::OuroResident::from_card(path, &card, None)?.with_sizing(catalog::resident_ouro::DEFAULT_CTX, catalog::resident_ouro::DEFAULT_BLOCK, max_batch.unwrap_or(catalog::resident_ouro::DEFAULT_BATCH));
    let served = match crate::perf_longctx::first_gpu_usable_bytes() {
        Some(bytes) => sized.with_card_budget(bytes),
        None => sized,
    };
    let (ctx, pool, batch) = served.sizing();
    let mut sched = served.scheduler()?;
    let vocab = model::serve::PagedDecoder::vocab(sched.decoder_mut()) as u32;
    let info = TargetInfo::new("ouro", "token").with("weights", path.into()).with("ctx", ctx.into()).with("kv_pool_tokens", pool.into()).with("max_batch", batch.into());
    // Synthetic runs must produce the requested length: no stop ids.
    Ok(Box::new(perf::targets::PagedLlmTarget::new(sched, info, Vec::new(), vocab)))
}
