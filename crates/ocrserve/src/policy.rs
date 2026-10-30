// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements batched document-OCR serving for clients who
// convert large archives to text. If your team needs expertise in sizing a
// vision-language server to the cards it runs on then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The serving policy of an OCR model: which of the model-agnostic flags it honours ([`OCR_POLICY`]) and how the ones it does
//! shape the decoder engine ([`OcrServeConfig::engine_config`]).
//!
//! * `--max-batch N` - pages decoded together (default [`MAX_BATCH`]). The K/V pool is sized by bytes, not by rows: a bigger batch
//!   shares the same pool, so each page gets fewer tokens before the scheduler preempts and recomputes.
//! * `--ctx N` - tokens one page may hold (prompt plus generated), at most the model's window.
//! * `--kv-cache f32|bf16` - how K/V is stored (bf16 default; an f32 pool holds half the tokens in the same bytes).
//! * `--gemm fp32` - the exact fp32 kernels in the decoder AND the vision towers instead of tensor cores.
//!
//! The weights have one format (the checkpoint's own, quantised at load), so `--weight-format` is refused by name.

use std::sync::OnceLock;

use deepseek2::serve::{EngineConfig, Precision};
use gpu_core::provider::library::GemmPrecision;
use model::kv_tier::KvTier;
use model::policy::{PolicySupport, ServePolicy};

use crate::{MAX_CONTEXT, MAX_BATCH, POOL_TOKENS, PREFILL_CHUNK};

/// What an OCR model honours of the model-agnostic serving policy.
pub const OCR_POLICY: PolicySupport = PolicySupport { model: "deepseek-ocr", weights: &[], weight_rules: false, kv_cache: &["f32", "bf16"], gemm: true, fuse_norm: false, kv_calibration: false, exact_decode: false, act_precision: false };

/// Most pages one engine decodes together: its step buffers and recorded steps are sized by it.
pub const MAX_BATCH_CEILING: u32 = 128;

/// The settings a person chose; `None` leaves the served default.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OcrServeConfig {
    pub max_batch: Option<u32>,
    pub ctx: Option<u32>,
    pub kv: Option<KvTier>,
    pub gemm: Option<GemmPrecision>,
    /// Why the policy cannot be honoured; building an OCR model then refuses with it ([`OcrServeConfig::check_servable`]).
    pub refused: Option<&'static str>,
}

impl OcrServeConfig {
    /// `Err` with the reason when the policy asked for something this model cannot do: the model is not served rather than served
    /// differently from what was asked.
    pub fn check_servable(&self) -> Result<(), String> {
        self.refused.map_or(Ok(()), |why| Err(why.to_string()))
    }

    /// Take what the policy asks of an OCR model. A choice it cannot honour is refused naming the model and the supported set; a
    /// limit that has no meaning here is ignored with a note.
    pub fn apply_policy(&mut self, p: &ServePolicy) -> Result<Vec<String>, String> {
        let refuse = |this: &mut Self, why: String| {
            // Leaked once per process start: a refusal is a startup event, and the config stays `Copy`.
            this.refused = Some(Box::leak(why.clone().into_boxed_str()));
            Err(why)
        };
        if let Err(why) = OCR_POLICY.check(p) {
            return refuse(self, why);
        }
        let l = &p.limits;
        if l.max_batch == Some(0) || l.max_batch.is_some_and(|n| n > MAX_BATCH_CEILING) {
            return refuse(self, format!("{}: --max-batch must be 1 to {MAX_BATCH_CEILING}", OCR_POLICY.model));
        }
        if l.ctx.is_some_and(|n| n < 64) {
            return refuse(self, format!("{}: --ctx must be at least 64 tokens", OCR_POLICY.model));
        }
        self.max_batch = l.max_batch.or(self.max_batch);
        self.ctx = l.ctx.or(self.ctx);
        self.kv = p.kv_cache.or(self.kv);
        self.gemm = p.gemm.or(self.gemm);
        let mut notes = Vec::new();
        for (set, name) in [(l.kv_offload_gb.is_some(), "--kv-offload-gb"), (l.kv_tokens.is_some(), "kv-tokens"), (l.max_prefill.is_some(), "max-prefill"), (l.prefix_slots.is_some(), "prefix-slots"), (l.itl_target_ms.is_some(), "--itl-target-ms")] {
            if set {
                notes.push(format!("{}: {name} has no meaning for this engine and is ignored", OCR_POLICY.model));
            }
        }
        Ok(notes)
    }

    /// Whether the exact fp32 kernels were asked for.
    pub fn fp32_reference(&self) -> bool {
        self.gemm == Some(GemmPrecision::Fp32Reference)
    }

    /// The arithmetic of the vision towers' linears.
    pub fn tower_precision(&self) -> GemmPrecision {
        self.gemm.unwrap_or(GemmPrecision::Auto)
    }

    /// Pages decoded together.
    pub fn batch(&self) -> u32 {
        self.max_batch.unwrap_or(MAX_BATCH).clamp(1, MAX_BATCH_CEILING)
    }

    /// The decoder engine's shape: the K/V pool is the same bytes whatever the batch (an f32 pool holds half the tokens of a bf16
    /// one), the window is `--ctx` capped at the model's.
    pub fn engine_config(&self) -> EngineConfig {
        let kv = self.kv.unwrap_or(KvTier::BF16);
        let pool_tokens = if kv == KvTier::F32 { POOL_TOKENS / 2 } else { POOL_TOKENS };
        let max_seq = self.ctx.map_or(MAX_CONTEXT, |c| c.min(MAX_CONTEXT));
        let precision = if self.fp32_reference() { Precision::Reference } else { Precision::Auto };
        EngineConfig { max_seq, pool_tokens, max_batch: self.batch(), chunk: PREFILL_CHUNK, block_size: 64, kv, topk: 64, precision }
    }
}

static CONFIG: OnceLock<OcrServeConfig> = OnceLock::new();

/// Set the process-wide serving configuration, once, before any OCR model loads (the CLI parses the flags; models read it when
/// they are built). A second call is ignored.
pub fn configure(config: OcrServeConfig) {
    let _ = CONFIG.set(config);
}

/// The serving configuration this process runs with: what [`configure`] set, else the defaults.
pub fn config() -> OcrServeConfig {
    CONFIG.get().copied().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(set: &[(&str, &str)]) -> ServePolicy {
        let mut p = ServePolicy::default();
        for (flag, value) in set {
            p.set(flag, value).expect("a policy flag");
        }
        p
    }

    #[test]
    fn the_flags_an_ocr_model_honours_shape_its_engine() {
        let mut c = OcrServeConfig::default();
        assert!(c.apply_policy(&policy(&[("--max-batch", "16"), ("--ctx", "4096"), ("--kv-cache", "f32"), ("--gemm", "fp32")])).unwrap().is_empty());
        let e = c.engine_config();
        assert_eq!((e.max_batch, e.max_seq, e.kv, e.pool_tokens), (16, 4096, KvTier::F32, POOL_TOKENS / 2));
        assert_eq!(e.precision, Precision::Reference);
        assert_eq!(c.tower_precision(), GemmPrecision::Fp32Reference);
        let d = OcrServeConfig::default().engine_config();
        assert_eq!((d.max_batch, d.max_seq, d.kv, d.pool_tokens, d.precision), (MAX_BATCH, MAX_CONTEXT, KvTier::BF16, POOL_TOKENS, Precision::Auto));
    }

    #[test]
    fn what_an_ocr_model_cannot_honour_is_refused_by_name_and_a_meaningless_limit_is_noted() {
        let mut c = OcrServeConfig::default();
        let e = c.apply_policy(&policy(&[("--weight-format", "int8")])).unwrap_err();
        assert!(e.contains("deepseek-ocr") && e.contains("weight format"), "{e}");
        assert!(c.apply_policy(&policy(&[("--kv-cache", "fp8")])).unwrap_err().contains("supported: f32, bf16"));
        assert!(c.check_servable().is_err(), "a refused policy stops the model from being served");
        assert!(c.apply_policy(&policy(&[("--max-batch", "0")])).is_err());
        assert!(c.apply_policy(&policy(&[("--max-batch", "500")])).is_err());
        let notes = c.apply_policy(&policy(&[("--kv-offload-gb", "4")])).unwrap();
        assert_eq!(notes.len(), 1);
    }
}
