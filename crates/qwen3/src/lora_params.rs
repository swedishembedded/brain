// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The parameters of a LoRA fine-tune, declared once for every served action
//! that trains a LoRA over a Qwen-family decoder (Qwen3's `lora_train`,
//! DeepSeek-VL's and Janus-Pro's `finetune`, ...), and read back into one
//! [`LoraRun`].
//!
//! The names, meanings and bounds are the same everywhere, so a caller who
//! has tuned one model's adapter knows every other's; only the defaults a
//! model is trained with differ ([`LoraDefaults`]).
//!
//! Swedish Embedded AB implements parameter-efficient fine-tuning of language
//! and vision-language models for its clients. If your team needs expertise
//! in adapting models to its own data then you can procure our services by
//! sending an email to info@swedishembedded.com.

use capability::{Invocation, ParamSpec, ParamType};
use serde_json::json;

use crate::finetune::{default_lora_targets, parse_lora_targets, LoraHyper, LORA_TARGETS};

/// Largest LoRA rank a run may ask for.
const MAX_RANK: f64 = 1_024.0;
/// Largest LoRA alpha a run may ask for.
const MAX_ALPHA: f64 = 4_096.0;
/// Most examples one step may average over.
const MAX_BATCH: f64 = 1_024.0;
/// Largest gradient-norm clip a run may ask for.
const MAX_CLIP: f64 = 1_000_000.0;

/// What a model trains with when the caller names nothing.
#[derive(Clone, Copy, Debug)]
pub struct LoraDefaults {
    pub rank: u32,
    pub steps: u32,
    pub lr: f32,
    pub batch: u32,
    /// Training context, tokens; `None` fits the longest example.
    pub block: Option<u32>,
}

/// One LoRA run, as an invocation of [`specs`] describes it.
#[derive(Clone, Debug, PartialEq)]
pub struct LoraRun {
    pub rank: u32,
    pub alpha: f32,
    pub steps: u32,
    pub lr: f32,
    pub batch: u32,
    pub block: Option<u32>,
    pub seed: u64,
    pub targets: Vec<String>,
    pub hyper: LoraHyper,
    /// What the backward keeps of the forward; `None` is the model's own default.
    pub saved: Option<model::saved::SavedPolicy>,
}

/// The parameters every LoRA fine-tune takes, with `d`'s defaults.
pub fn specs(d: &LoraDefaults) -> Vec<ParamSpec> {
    let hyper = LoraHyper::default();
    let block = match d.block {
        Some(b) => ParamSpec::new("block", ParamType::Int, "training context, tokens; bounded only by the memory training at it needs").default(json!(b)),
        None => ParamSpec::new("block", ParamType::Int, "training context, tokens; omit to fit the longest example"),
    }
    .min(1.0)
    .max(capability::bounds::CONTEXT_TOKENS);
    vec![
        ParamSpec::new("rank", ParamType::Int, "LoRA rank (capacity/size tradeoff)").default(json!(d.rank)).min(1.0).max(MAX_RANK),
        ParamSpec::new("alpha", ParamType::Float, "LoRA alpha (the delta is scaled by alpha/rank); omit for 2 x rank").min(0.0).max(MAX_ALPHA),
        ParamSpec::new("steps", ParamType::Int, "optimizer steps").default(json!(d.steps)).min(1.0).max(capability::bounds::TRAIN_STEPS),
        ParamSpec::new("lr", ParamType::Float, "peak learning rate (cosine schedule down to min_lr)").default(json!(d.lr)).min(0.0).max(capability::bounds::UNIT),
        ParamSpec::new("batch", ParamType::Int, "examples averaged into each step").default(json!(d.batch)).min(1.0).max(MAX_BATCH),
        block,
        ParamSpec::new("seed", ParamType::Int, "RNG seed (omit for random; the outcome echoes the one used)").min(0.0),
        ParamSpec::new("targets", ParamType::Str, &format!("comma-separated projections to adapt, of {}", LORA_TARGETS.join(","))).default(json!(LORA_TARGETS.join(","))),
        ParamSpec::new("weight_decay", ParamType::Float, "AdamW decoupled weight decay").default(json!(hyper.weight_decay)).min(0.0).max(capability::bounds::UNIT),
        ParamSpec::new("grad_clip", ParamType::Float, "global gradient-norm clip; 0 disables it").default(json!(hyper.grad_clip)).min(0.0).max(MAX_CLIP),
        ParamSpec::new("warmup", ParamType::Int, "linear warmup steps; omit for 5% of steps").min(0.0).max(capability::bounds::TRAIN_COUNT),
        ParamSpec::new("min_lr", ParamType::Float, "rate the cosine decays to; omit for lr/10").min(0.0).max(capability::bounds::UNIT),
        ParamSpec::new("beta1", ParamType::Float, "AdamW first-moment decay").default(json!(hyper.adam.beta1)).min(0.0).max(0.9999),
        ParamSpec::new("beta2", ParamType::Float, "AdamW second-moment decay").default(json!(hyper.adam.beta2)).min(0.0).max(0.99999),
        ParamSpec::new("eps", ParamType::Float, "AdamW denominator epsilon").default(json!(hyper.adam.eps)).min(1e-12).max(1e-3),
        ParamSpec::new(
            "activations",
            ParamType::Enum(model::saved::SavedPolicy::ACTIVATIONS_CHOICES.map(String::from).to_vec()),
            "what the backward keeps of the forward: every unit's intermediates, or only each unit's input, rebuilt in the backward; a choice the model cannot run is refused by name before the first step (default: the model's own)",
        ),
    ]
}

/// The run `inv` describes, `d` filling what it leaves out. `action` names
/// the action in an error.
pub fn from_invocation(inv: &Invocation, d: &LoraDefaults, action: &str) -> Result<LoraRun, String> {
    let rank = inv.get_i64("rank").map_or(d.rank, |r| r.max(1) as u32);
    let base = LoraHyper::default();
    let f = |name: &str, default: f32| inv.get_f64(name).map_or(default, |v| v as f32);
    let hyper = LoraHyper {
        weight_decay: f("weight_decay", base.weight_decay),
        grad_clip: f("grad_clip", base.grad_clip),
        warmup: inv.get_i64("warmup").map(|w| w.max(0) as u32),
        min_lr: inv.get_f64("min_lr").map(|v| v as f32),
        adam: model::Adam { beta1: f("beta1", base.adam.beta1), beta2: f("beta2", base.adam.beta2), eps: f("eps", base.adam.eps) },
    };
    let targets = match inv.get_str("targets") {
        Some(t) => parse_lora_targets(&t).map_err(|e| format!("{action}: {e}"))?,
        None => default_lora_targets(),
    };
    let saved = inv.get_str("activations").map(|v| model::saved::SavedPolicy::parse_activations(&v).map_err(|e| format!("{action}: {e}"))).transpose()?;
    Ok(LoraRun {
        rank,
        alpha: inv.get_f64("alpha").map_or(rank as f32 * 2.0, |a| a as f32),
        steps: inv.get_i64("steps").map_or(d.steps, |s| s.max(1) as u32),
        lr: f("lr", d.lr),
        batch: inv.get_i64("batch").map_or(d.batch, |b| b.max(1) as u32),
        block: inv.get_i64("block").map(|b| b.max(1) as u32).or(d.block),
        seed: inv.seed()?.unwrap_or_else(data::rng::random_seed),
        targets,
        hyper,
        saved,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: LoraDefaults = LoraDefaults { rank: 16, steps: 200, lr: 1e-4, batch: 1, block: None };

    fn spec() -> capability::ActionSpec {
        capability::ActionSpec::new("finetune", "test").params(specs(&D))
    }

    /// What a caller leaves out is the model's default; alpha follows rank.
    #[test]
    fn an_empty_invocation_is_the_models_defaults() {
        let inv = spec().validate(Invocation::new().set("seed", json!(3))).unwrap();
        let run = from_invocation(&inv, &D, "finetune").unwrap();
        assert_eq!((run.rank, run.alpha, run.steps, run.lr, run.batch, run.block, run.seed), (16, 32.0, 200, 1e-4, 1, None, 3));
        assert_eq!(run.targets, default_lora_targets());
        assert_eq!(run.hyper, LoraHyper::default());
        assert_eq!(run.saved, None);
    }

    #[test]
    fn every_knob_reaches_the_run_and_bad_values_are_refused() {
        let inv = spec()
            .validate(Invocation::new().set("rank", json!(4)).set("alpha", json!(2.0)).set("block", json!(256)).set("targets", json!("wq,wv")).set("warmup", json!(7)).set("activations", json!("recompute")))
            .unwrap();
        let run = from_invocation(&inv, &D, "finetune").unwrap();
        assert_eq!((run.rank, run.alpha, run.block, run.hyper.warmup), (4, 2.0, Some(256), Some(7)));
        assert_eq!(run.targets, ["wq", "wv"]);
        assert_eq!(run.saved, Some(model::saved::SavedPolicy::Recompute));
        assert!(spec().validate(Invocation::new().set("rank", json!(0))).is_err());
        let bad = spec().validate(Invocation::new().set("targets", json!("wq,nope"))).unwrap();
        assert!(from_invocation(&bad, &D, "finetune").unwrap_err().contains("nope"));
    }
}
