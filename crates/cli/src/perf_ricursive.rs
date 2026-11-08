// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain perf run longctx --target ricursive:<dir>[:kvbf16|:kvint8|:kvfp8]` - Ripple-65B's paged serving engine on the
//! cards `--device` selected, for the long-context scenario: decode tokens/s per batch size at a context, and cold
//! prefill time per prompt length, against the roofline.
//!
//! Swedish Embedded AB implements solutions for measuring what a deployed long-context language model delivers per
//! card, with the arithmetic of its memory plan beside the measurement, for its clients. If your team needs expertise
//! in serving-throughput measurement of large hybrid-attention decoders then you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! The engine is built as `brain serve` builds it for the same checkpoint: weights as the checkpoint stores them
//! (the NVFP4 release stays NVFP4), the layers cut over the fewest selected cards that hold the sizing
//! ([`ricursive::placement::auto_plan`]), the fused paged attention kernels, the KV cache at the tier the target
//! names (bf16 when unnamed, the server's own default without an index). A decode context is synthetic: a
//! sequence's block table is extended to the measured position without prefilling, which costs the decode what a
//! real context of that length costs.

use std::time::Instant;

use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::serve::{PagedDecoder, PrefillProgress};
use perf::scenarios::longctx::{ContextKind, EngineInfo, Fit, LongContextEngine};
use ricursive::placement::{auto_plan, footprint};
use ricursive::{open_shared_checkpoint, resolve_tiers, LsaCharge, RicursiveConfig, RicursiveEngine, ServeConfig};

/// Rows per global KV block and per prefill forward: the server's own sizes.
const BLOCK_SIZE: u32 = 64;
const PREFILL_CHUNK: u32 = 512;

/// `ricursive:<dir>[:kv<tier>]` after the prefix: the checkpoint directory and the KV tier it names.
pub(crate) fn parse_spec(rest: &str) -> Result<(&str, KvTier), String> {
    let (dir, kv) = match rest.rsplit_once(":kv") {
        Some((dir, tier)) => (dir, KvTier::parse(tier).map_err(|e| format!("target \"ricursive:{rest}\": :kv{tier}: {e}"))?),
        None => (rest, KvTier::BF16),
    };
    if dir.is_empty() {
        return Err("target ricursive: needs a checkpoint directory: ricursive:<dir>[:kvbf16|:kvint8|:kvfp8]".to_string());
    }
    Ok((dir, kv))
}

pub struct RicursiveLongCtx {
    dir: String,
    cfg: RicursiveConfig,
    kv: KvTier,
    engine: Option<RicursiveEngine>,
    tables: Vec<BlockTable>,
    prefill_chunk: usize,
}

impl RicursiveLongCtx {
    pub fn open(spec: &str) -> Result<RicursiveLongCtx, String> {
        let (dir, kv) = parse_spec(spec)?;
        let cfg = RicursiveConfig::from_file(&format!("{dir}/config.json"))?;
        Ok(RicursiveLongCtx { dir: dir.to_string(), cfg, kv, engine: None, tables: Vec::new(), prefill_chunk: PREFILL_CHUNK as usize })
    }

    /// The sizing of `batch` sequences of up to `context` tokens: a block each beyond the context for the one being
    /// written, one cached-prefix slot for none.
    fn serve(&self, batch: u32, context: u32) -> ServeConfig {
        let per_seq = context.div_ceil(BLOCK_SIZE) + 1;
        ServeConfig { kv_cache: self.kv, ..ServeConfig::new(BLOCK_SIZE, per_seq * batch + batch, per_seq * BLOCK_SIZE, batch, 0, PREFILL_CHUNK) }
    }

    fn layers(&self) -> Vec<usize> {
        (0..self.cfg.num_hidden_layers).collect()
    }
}

impl LongContextEngine for RicursiveLongCtx {
    fn describe(&self) -> EngineInfo {
        let cards = model::shard::StreamPlan::cards(None, 0);
        EngineInfo {
            model: format!("ricursive:{}", std::path::Path::new(&self.dir).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()),
            weight_tier: "as stored".into(),
            kv_precision: crate::perf_longctx::kv_label(self.kv).into(),
            backend: gpu_core::backend_name().to_string(),
            devices: cards.iter().map(|(i, free)| format!("gpu{i} ({:.1} GiB free)", *free as f64 / (1u64 << 30) as f64)).collect(),
            context_kind: ContextKind::Synthetic,
        }
    }

    /// The config the engine parses states no limit (the release advertises 2 Mi positions in a field the engine
    /// does not read), so a sweep is bounded by the cards alone.
    fn max_context(&self) -> Option<u32> {
        None
    }

    fn plan(&mut self, batch: u32, context: u32) -> Result<Fit, String> {
        let cards = model::shard::StreamPlan::cards(None, 0);
        if cards.is_empty() {
            return Err("no GPU with queryable memory: longctx is GPU-only and has no host fallback".into());
        }
        let serve = self.serve(batch, context);
        let layers = self.layers();
        let src = open_shared_checkpoint(&self.dir, &self.cfg, &layers)?;
        let tiers = resolve_tiers(&self.cfg, src.as_ref(), None);
        let needed = footprint(&self.cfg, &serve, &tiers, LsaCharge::default()).single_card_bytes();
        let usable = cards.iter().map(|c| c.1).sum();
        Ok(Fit { fits: auto_plan(&self.cfg, &serve, &tiers, LsaCharge::default(), None).is_ok(), needed_bytes: needed, usable_bytes: usable })
    }

    fn load(&mut self, batch: u32, context: u32) -> Result<(), String> {
        self.engine = None;
        self.tables.clear();
        let serve = self.serve(batch, context);
        let layers = self.layers();
        let src = open_shared_checkpoint(&self.dir, &self.cfg, &layers)?;
        let tiers = resolve_tiers(&self.cfg, src.as_ref(), None);
        let plan = auto_plan(&self.cfg, &serve, &tiers, LsaCharge::default(), None)?;
        // An allocation failure on the device surfaces as a panic from the constructor: it is the out-of-memory
        // boundary of the sweep, not a crash.
        let cfg = self.cfg.clone();
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| RicursiveEngine::from_source_staged(cfg, src, serve, None, &plan)))
            .map_err(|p| p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "engine construction panicked".into()))??;
        self.engine = Some(built);
        Ok(())
    }

    fn prefill(&mut self, prompt_tokens: u32) -> Result<f64, String> {
        let engine = self.engine.as_mut().ok_or("prefill before load")?;
        let vocab = self.cfg.vocab_size as u64;
        // Deterministic in-vocabulary ids: the cost of a prefill does not depend on which.
        let prompt: Vec<u32> = (0..u64::from(prompt_tokens)).map(|i| ((i * 7919 + 13) % vocab) as u32).collect();
        let mut table = BlockTable::new();
        let started = Instant::now();
        // The last hidden row returns to the host, so the clock covers the device.
        while !matches!(engine.prefill_chunk(&mut table, &prompt, self.prefill_chunk), PrefillProgress::Done(_)) {}
        let seconds = started.elapsed().as_secs_f64();
        engine.release_table(&mut table);
        Ok(seconds)
    }

    fn decode_step(&mut self, positions: &[u32]) -> Result<f64, String> {
        let engine = self.engine.as_mut().ok_or("decode before load")?;
        self.tables.resize_with(positions.len(), BlockTable::new);
        // A sequence is brought to its position by reserving (or truncating) blocks: the synthetic context.
        for (table, &p) in self.tables.iter_mut().zip(positions) {
            if table.len() < p {
                table.reserve(p - table.len(), engine.alloc_mut())?;
            } else if table.len() > p {
                table.truncate(p, engine.alloc_mut());
            }
        }
        let tokens = vec![1u32; positions.len()];
        let mut rows: Vec<&mut BlockTable> = self.tables.iter_mut().collect();
        let started = Instant::now();
        // Greedy decode reads each row's next token back to the host.
        let out = engine.forward_batched_greedy(&mut rows, &tokens);
        let ms = started.elapsed().as_secs_f64() * 1e3;
        if out.len() != positions.len() {
            return Err(format!("engine returned {} tokens for {} rows", out.len(), positions.len()));
        }
        Ok(ms)
    }

    fn device_used_bytes(&self) -> Option<u64> {
        crate::perf_longctx::driver_used_bytes()
    }

    /// The first stage's device: a `--kernels` profile of a staged engine covers that stage only.
    fn devices(&self) -> Vec<&gpu_core::Gpu> {
        self.engine.as_ref().map(|e| vec![e.gpu()]).unwrap_or_default()
    }

    fn unload(&mut self) {
        self.tables.clear();
        self.engine = None;
    }

    fn release_batch(&mut self) {
        if let Some(engine) = self.engine.as_mut() {
            for table in &mut self.tables {
                engine.release_table(table);
            }
        }
        self.tables.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kv_tier_is_named_by_a_suffix_and_defaults_to_bf16() {
        assert_eq!(parse_spec("/m/ripple").unwrap(), ("/m/ripple", KvTier::BF16));
        assert_eq!(parse_spec("/m/ripple:kvint8").unwrap(), ("/m/ripple", KvTier::INT8));
        assert_eq!(parse_spec("/m/ripple:kvfp8").unwrap(), ("/m/ripple", KvTier::FP8));
        assert!(parse_spec("/m/ripple:kvq3").is_err(), "an unknown tier is refused by name");
        assert!(parse_spec(":kvint8").is_err(), "a target without a directory is refused");
    }
}
