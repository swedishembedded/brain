// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `--target qwen35:<gguf>[:kvf32|:kvbf16|:kvint8]` for the request-level
//! `brain perf run` scenarios: the released Qwen3.8-27B GGUF served exactly as
//! `brain serve` serves it.
//!
//! The engine is configured by `catalog::resident_qwen35::gguf_resident` (tier,
//! context, KV pool, prefix slots, speculation) and put under the scheduler by
//! `Qwen35GgufResident::activate_scheduler`, the two functions the server calls,
//! so goodput, TTFT, prefix-hit and KV-pressure numbers are the served ones.
//!
//! Swedish Embedded AB implements solutions for measuring LLM serving
//! performance against production behaviour for its clients. If your team needs
//! expertise in serving and benchmarking large models on GPUs then you can
//! procure our services by sending an email to info@swedishembedded.com.

use model::kv_tier::KvTier;
use perf::target::{PerfTarget, TargetInfo};
use residency::multi::MultiDeviceResidentModel;
use residency::ResidentModel;

/// Bytes `brain serve` keeps free per card (its `--reserve-gb` default of 2).
const RESERVE_BYTES: u64 = 2 << 30;

/// Split `qwen35:<gguf>[:kvf32|:kvbf16|:kvint8]` (the part after `qwen35:`) into
/// the checkpoint path and the KV tier it names. `None` leaves the tier to the
/// caller's default.
pub(crate) fn parse_spec(rest: &str) -> Result<(&str, Option<KvTier>), String> {
    let (path, kv) = match rest.rsplit_once(":kv") {
        Some((path, tier @ ("f32" | "bf16" | "int8"))) => (path, Some(KvTier::parse(tier)?)),
        Some((_, tier)) => return Err(format!("target \"qwen35:{rest}\": :kv{tier} is not a KV tier - use :kvf32, :kvbf16 or :kvint8")),
        None => (rest, None),
    };
    if path.is_empty() {
        return Err("target qwen35: needs a GGUF path: qwen35:<gguf>[:kvf32|:kvbf16|:kvint8]".to_string());
    }
    Ok((path, kv))
}

/// The paged-engine target for one released GGUF. Devices are the cards
/// `--device` makes schedulable, budgeted with `brain serve`'s reserve.
pub(crate) fn build(rest: &str) -> Result<Box<dyn PerfTarget>, String> {
    let (path, kv) = parse_spec(rest)?;
    if !std::path::Path::new(path).is_file() {
        return Err(format!("qwen35 target: GGUF not found: {path}"));
    }
    let machine = serving::Machine::probe(crate::compute_set());
    let id = std::path::Path::new(path).file_stem().map_or_else(|| path.to_string(), |s| s.to_string_lossy().into_owned());
    // Everything but the KV tier the target names is left to the server's own defaults.
    let cfg = catalog::resident_qwen35::Qwen35GgufServeConfig { kv, ..Default::default() };
    let served = catalog::resident_qwen35::gguf_resident(path, &id, &machine.gpus, RESERVE_BYTES, &cfg)?;
    let key = served.instance_key("generate", &capability::Invocation::new());
    let devices: Vec<residency::Device> = served.estimate_multi(&key).devices().collect();
    if devices.is_empty() {
        return Err(format!("qwen35 target: {path} does not fit the schedulable GPUs: {}", served.unavailable_reason(&key).unwrap_or_default()));
    }
    let (mut sched, _tokenizer, _stop) = served.activate_scheduler(&devices)?;
    let vocab = model::serve::PagedDecoder::vocab(sched.decoder_mut()) as u32;
    let mut info = TargetInfo::new("qwen35", "token")
        .with("weights", path.into())
        .with("max_batch", served.max_batch().into())
        .with("ctx", served.cap().into())
        .with("kv_dtype", served.kv_tier().as_str().into())
        .with("prefix_slots", served.prefix_slots().into())
        .with("devices", devices.iter().map(|d| format!("{d:?}")).collect::<Vec<_>>().into());
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
    fn the_spec_names_a_checkpoint_and_optionally_its_kv_tier() {
        assert_eq!(parse_spec("/m/q.gguf").unwrap(), ("/m/q.gguf", None));
        assert_eq!(parse_spec("/m/q.gguf:kvint8").unwrap(), ("/m/q.gguf", Some(KvTier::INT8)));
        assert_eq!(parse_spec("/m/q.gguf:kvbf16").unwrap().1, Some(KvTier::BF16));
        assert_eq!(parse_spec("/m/q.gguf:kvf32").unwrap().1, Some(KvTier::F32));
    }

    #[test]
    fn an_unknown_kv_suffix_or_empty_path_is_refused_with_the_valid_choices() {
        let e = parse_spec("/m/q.gguf:kvfoo").unwrap_err();
        assert!(e.contains("kvfoo") && e.contains(":kvint8"), "{e}");
        assert!(parse_spec(":kvint8").unwrap_err().contains("GGUF path"));
    }

    #[test]
    fn a_missing_checkpoint_errors_naming_the_file() {
        let e = build("/nonexistent/qwen35.gguf:kvint8").err().expect("must refuse");
        assert!(e.contains("/nonexistent/qwen35.gguf"), "{e}");
    }
}
