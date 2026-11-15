// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving large language models
// from killable worker processes without changing what they compute for its
// clients. If your team needs expertise in reproducible inference engine
// sizing then you can procure our services by sending an email to
// info@swedishembedded.com.

//! What a [`super::QwenResident`] serves by, once every choice is resolved:
//! the checkpoint, the engine's sizing (`ctx`, the KV pool, batch and prefill
//! caps), the weight and KV tiers and the adapter bank. The serving process
//! resolves it from the operator's flags and the budgets of its cards
//! ([`super::QwenServeConfig`]); a worker hosting the resident is sent the
//! resolved plan as its host spec and builds from it as it stands, so both
//! sides size the identical engine even though only the serving process knows
//! the budgets the plan was resolved against.

use gpu_core::provider::library::GemmPrecision;
use llmserve::adapter_registry::{AdapterBankSettings, AdapterTargets};
use model::ops::TierPolicy;
use serde::{Deserialize, Serialize};

/// The resolved serving plan of one Qwen3 resident. Every field is a value
/// the engine is built from as it stands: nothing in it is re-derived from a
/// budget, a device or the checkpoint's size.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct QwenPlan {
    /// The served model id (the card's, or `brain/qwen3` for a checkpoint named by path).
    pub(super) id: String,
    /// The checkpoint the engine opens: a file, or a Hugging Face directory.
    pub(super) path: String,
    /// The `tokenizer.json` to read; empty for a GGUF's embedded one.
    pub(super) tokenizer: String,
    /// The architecture the card names, declared on the manifest ([`brain_arch::ARCHS`] id).
    pub(super) architecture: Option<String>,
    /// The id adapters trained on this base name as theirs.
    pub(super) base_id: Option<String>,
    /// The context the KV pool is sized for: explicit, auto-sized against the card's budget, or the
    /// historical default ([`super::QwenResident::resolve_ctx`]).
    pub(super) ctx: u32,
    pub(super) max_batch: u32,
    pub(super) kv_int8_requested: bool,
    pub(super) kv_fp8_requested: bool,
    /// `--sparse-attention`: the indexer file the engine reads through.
    pub(super) sparse_attention: Option<String>,
    pub(super) kv_calib_opt_in: bool,
    pub(super) kv_offload_bytes: u64,
    /// The tiers of the linears and the head as resolved: the operator's `--weight-format`, else the
    /// checkpoint's stored tier under the card rule ([`super::qwen3_weight_policy`]).
    #[serde(with = "tier_spelling")]
    pub(super) weights: TierPolicy,
    pub(super) max_prefill_cap: u32,
    /// The KV pool in tokens as resolved (`None`: two contexts).
    pub(super) kv_tokens: Option<u32>,
    /// The share of a card the plan must fit, which activation refuses by before it loads.
    pub(super) card_budget_bytes: Option<u64>,
    #[serde(with = "gemm_spelling")]
    pub(super) gemm_precision: GemmPrecision,
    pub(super) fuse_residual_norm: bool,
    pub(super) exact_decode: bool,
    /// Why the serving policy cannot be honoured; activation refuses with it.
    pub(super) refused: Option<String>,
    /// The adapter bank built beside the weights; `None` builds none.
    #[serde(with = "bank_spelling")]
    pub(super) adapter_bank: Option<AdapterBankSettings>,
}

impl QwenPlan {
    /// The plan a host spec carries, refusing an unknown field and a value no engine is built with.
    pub(super) fn from_spec(args: &serde_json::Value) -> Result<QwenPlan, String> {
        let plan: QwenPlan = serde_json::from_value(args.clone()).map_err(|e| format!("a Qwen3 host spec: {e}"))?;
        if plan.ctx == 0 || plan.max_batch == 0 || !(1..=512).contains(&plan.max_prefill_cap) {
            return Err(format!("a Qwen3 host spec sizes no engine (ctx {}, max_batch {}, max_prefill_cap {})", plan.ctx, plan.max_batch, plan.max_prefill_cap));
        }
        if let Some(arch) = &plan.architecture {
            if brain_arch::by_id(arch).is_none() {
                return Err(format!("a Qwen3 host spec names architecture {arch:?}, which is not registered"));
            }
        }
        if let Some(bank) = &plan.adapter_bank {
            bank.validate().map_err(|e| format!("a Qwen3 host spec's adapter bank: {e}"))?;
        }
        Ok(plan)
    }

    /// The plan as the host spec's arguments.
    pub(super) fn to_spec(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("a plan of strings, numbers and flags serializes")
    }
}

/// A weight policy as its `--weight-format` spelling ([`TierPolicy::describe`]).
mod tier_spelling {
    use super::TierPolicy;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(tiers: &TierPolicy, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&tiers.describe())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<TierPolicy, D::Error> {
        TierPolicy::parse(&String::deserialize(d)?).map_err(serde::de::Error::custom)
    }
}

/// The GEMM arithmetic as a word: `auto`, `fp32` (the `--gemm` spellings) or `bf16-tensor-core`.
mod gemm_spelling {
    use super::GemmPrecision;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(g: &GemmPrecision, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(match g {
            GemmPrecision::Auto => "auto",
            GemmPrecision::Fp32Reference => "fp32",
            GemmPrecision::Bf16TensorCore => "bf16-tensor-core",
        })
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<GemmPrecision, D::Error> {
        match String::deserialize(d)?.as_str() {
            "auto" => Ok(GemmPrecision::Auto),
            "fp32" => Ok(GemmPrecision::Fp32Reference),
            "bf16-tensor-core" => Ok(GemmPrecision::Bf16TensorCore),
            other => Err(serde::de::Error::custom(format!("{other:?} is not a GEMM precision (auto, fp32 or bf16-tensor-core)"))),
        }
    }
}

/// An adapter bank as `{slots, rank_cap, targets}`, the targets in their `--adapter-targets` spelling.
mod bank_spelling {
    use super::{AdapterBankSettings, AdapterTargets};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Bank {
        slots: u32,
        rank_cap: u32,
        targets: String,
    }

    pub fn serialize<S: Serializer>(bank: &Option<AdapterBankSettings>, s: S) -> Result<S::Ok, S::Error> {
        bank.as_ref().map(|b| Bank { slots: b.slots, rank_cap: b.rank_cap, targets: b.targets.to_string() }).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<AdapterBankSettings>, D::Error> {
        Option::<Bank>::deserialize(d)?
            .map(|b| AdapterTargets::parse(&b.targets).map(|targets| AdapterBankSettings { slots: b.slots, rank_cap: b.rank_cap, targets }).map_err(serde::de::Error::custom))
            .transpose()
    }
}
