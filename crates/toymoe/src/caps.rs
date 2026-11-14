// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The toy sparse Mixture-of-Experts Transformer's served actions: `train`
//! it on its synthetic corpus, `eval` how well a checkpoint continues that
//! corpus, and `generate` (or read the next-token logits) from a prompt of
//! token ids.
//!
//! The model is brain's own reference architecture with no released
//! checkpoint and no store spec, so a checkpoint is named by path (host
//! side); `train` writes one.
//!
//! Swedish Embedded AB implements sparse Mixture-of-Experts models and their
//! training on its clients' own hardware. If your team needs expertise in
//! expert-routed architectures then you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Invocation, Manifest, Outcome, ParamSpec, ParamType, Progress, Provider};
use serde_json::json;

use crate::train::{TrainArgs, CORPUS_SEED};

pub const MODEL: &str = "brain/toymoe";

/// Largest block a run may train at: the toy corpus window.
const MAX_BLOCK: f64 = 64.0;
/// Most sequences one step may hold.
const MAX_BATCH: f64 = 1_024.0;
/// Longest continuation one call may generate.
const MAX_NEW: f64 = 65_536.0;
/// Tokens `generate` adds when the caller names no count.
const DEFAULT_NEW: i64 = 64;
/// Most positions one accuracy measurement may score.
const MAX_SAMPLES: f64 = 1_000_000.0;

fn weights_param() -> ParamSpec {
    ParamSpec::new("weights", ParamType::Str, "the checkpoint (.safetensors) `train` wrote, on the serving machine").required().host_resolved()
}

pub fn train_spec() -> ActionSpec {
    ActionSpec::new("train", "train the toy MoE on its synthetic substitution-orbit corpus (the generic trainer, AdamW)").cancellable()
        .training()
        .streaming()
        .from_scratch()
        .param(ParamSpec::new("save", ParamType::Str, "where to write the checkpoint (.safetensors), on the serving machine").required().host_resolved())
        .param(ParamSpec::new("steps", ParamType::Int, "optimizer steps").default(json!(2000)).min(1.0).max(capability::bounds::TRAIN_STEPS))
        .param(ParamSpec::new("batch", ParamType::Int, "sequences per step").default(json!(16)).min(1.0).max(MAX_BATCH))
        .param(ParamSpec::new("block", ParamType::Int, "tokens per sequence (the corpus window is 64)").default(json!(64)).min(1.0).max(MAX_BLOCK))
        .param(ParamSpec::new("lr", ParamType::Float, "AdamW learning rate").default(json!(3e-4)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("weight_decay", ParamType::Float, "AdamW decoupled weight decay").default(json!(0.1)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("seed", ParamType::Int, "seeds the initialization and the batches (omit for random; the outcome echoes it)").min(0.0))
        .param(ParamSpec::new(
            "activations",
            ParamType::Enum(model::saved::SavedPolicy::ACTIVATIONS_CHOICES.map(String::from).to_vec()),
            "what the backward keeps of the forward (default: every layer's activations)",
        ))
}

pub fn eval_spec() -> ActionSpec {
    ActionSpec::new("eval", "next-token accuracy on the training, validation and a never-seen orbit of the corpus, at the context length the checkpoint does best at").cancellable()
        .param(weights_param())
        .param(ParamSpec::new("corpus_seed", ParamType::Int, "which toy corpus to score on; the trainer's by default").default(json!(CORPUS_SEED)).min(0.0))
        .param(ParamSpec::new("samples", ParamType::Int, "positions scored per measurement").default(json!(400)).min(1.0).max(MAX_SAMPLES))
}

pub fn generate_spec() -> ActionSpec {
    ActionSpec::new("generate", "continue a prompt of token ids, or return the logits of the next token").cancellable()
        .param(weights_param())
        .param(ParamSpec::new("prompt", ParamType::Str, "comma- or space-separated token ids").default(json!("0,1,2,3")))
        .param(ParamSpec::new("max_new", ParamType::Int, "tokens to generate").default(json!(DEFAULT_NEW)).min(0.0).max(MAX_NEW))
        .param(ParamSpec::new("temperature", ParamType::Float, "sampling temperature").default(json!(0.8)).min(0.0).max(capability::bounds::UNIT * 10.0))
        .param(ParamSpec::new("top_k", ParamType::Int, "sample among the k most likely tokens (omit for all)").min(1.0).max(capability::bounds::TOP_K))
        .param(ParamSpec::new("seed", ParamType::Int, "sampling seed (omit for random; the outcome echoes it)").min(0.0))
        .param(ParamSpec::new("logits_only", ParamType::Bool, "return the next token's logits instead of generating").default(json!(false)))
}

pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "Toy sparse Mixture-of-Experts Transformer (top-k routed experts, RMSNorm, RoPE, tied head): brain's reference MoE.", vec![train_spec(), eval_spec(), generate_spec()])
}

fn parse_prompt(text: &str) -> Result<Vec<u32>, String> {
    text.split([',', ' ']).filter(|s| !s.is_empty()).map(|t| t.parse::<u32>().map_err(|_| format!("prompt token {t:?} is not a token id"))).collect()
}

#[derive(Default)]
pub struct ToyMoeProvider;

impl Provider for ToyMoeProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        manifest().actions.into_iter().find(|a| a.name == name).map(|spec| Arc::new(ToyMoeAction { spec }) as Arc<dyn Action>)
    }
}

struct ToyMoeAction {
    spec: ActionSpec,
}

impl Action for ToyMoeAction {
    fn spec(&self) -> ActionSpec {
        self.spec.clone()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let weights = || inv.get_str("weights").ok_or_else(|| format!("toymoe {}: missing required param 'weights'", self.spec.name));
        let seed = || inv.seed().map(|s| s.unwrap_or_else(data::rng::random_seed));
        match self.spec.name.as_str() {
            "train" => {
                let args = TrainArgs {
                    steps: inv.get_i64("steps").unwrap_or(2000).max(1) as u32,
                    b: inv.get_i64("batch").unwrap_or(16).max(1) as u32,
                    t: inv.get_i64("block").unwrap_or(64) as u32,
                    lr: inv.get_f64("lr").unwrap_or(3e-4) as f32,
                    wd: inv.get_f64("weight_decay").unwrap_or(0.1) as f32,
                    seed: seed()?,
                    out: inv.get_str("save").ok_or("toymoe train: missing required param 'save'")?,
                    saved: inv.get_str("activations").map(|a| model::saved::SavedPolicy::parse_activations(&a)).transpose()?,
                };
                let (seed, save, steps) = (args.seed, args.out.clone(), args.steps);
                let report = crate::train::train(args, &mut |s| {
                    progress(Progress::step(s.step, s.steps, format!("loss {:.4}", s.loss)));
                    !inv.cancel.is_cancelled()
                })?;
                if report.interrupted {
                    progress(Progress::step(report.steps_completed, steps, format!("cancelled; the checkpoint so far is at {save}")));
                    return Err(capability::failure::CANCELLED.to_string());
                }
                Ok(Outcome::new().set("initial_loss", json!(report.initial_loss)).set("final_loss", json!(report.final_loss)).set("steps", json!(report.steps_completed)).set("seed", json!(seed)).set("saved", json!(save)))
            }
            "eval" => {
                let r = crate::evaluate(&weights()?, inv.get_i64("corpus_seed").map_or(CORPUS_SEED, |s| s as u64), inv.get_i64("samples").unwrap_or(400).max(1) as usize)?;
                Ok(Outcome::new()
                    .set("random", json!(r.random))
                    .set("best_context", json!(r.best_context))
                    .set("train_orbit", json!(r.train_orbit))
                    .set("val_orbit", json!(r.val_orbit))
                    .set("new_orbit", json!(r.new_orbit))
                    .set("by_context", json!(r.by_context)))
            }
            "generate" => {
                let prompt = parse_prompt(&inv.get_str("prompt").unwrap_or_else(|| "0,1,2,3".into()))?;
                if inv.get_bool("logits_only").unwrap_or(false) {
                    return Ok(Outcome::new().set("logits", json!(crate::logits(&weights()?, &prompt)?)));
                }
                let seed = seed()?;
                let top_k = inv.get_i64("top_k").map(|k| k.max(1) as usize);
                let tokens = crate::generate(&weights()?, &prompt, inv.get_i64("max_new").unwrap_or(DEFAULT_NEW).max(0) as usize, inv.get_f64("temperature").unwrap_or(0.8) as f32, top_k, seed)?;
                Ok(Outcome::new().set("tokens", json!(tokens)).set("seed", json!(seed)))
            }
            other => Err(format!("toymoe: unknown action {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::Registry;

    /// Train for a few steps, then evaluate and sample the checkpoint it
    /// wrote, all through the served actions.
    #[test]
    fn train_then_eval_and_generate_through_the_actions() {
        let dir = std::env::temp_dir().join(format!("toymoe-caps-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let save = dir.join("moe.safetensors");
        let mut reg = Registry::new();
        reg.register(Arc::new(ToyMoeProvider));
        let run = |action: &str, inv: Invocation| reg.run(MODEL, action, inv, &mut |_| {}).unwrap_or_else(|e| panic!("{action}: {e}"));
        let trained = run("train", Invocation::new().set("save", json!(save.to_str().unwrap())).set("steps", json!(3)).set("batch", json!(2)).set("block", json!(16)).set("seed", json!(1)));
        assert_eq!((trained.outputs["steps"].as_u64(), trained.outputs["seed"].as_u64()), (Some(3), Some(1)));
        let w = json!(save.to_str().unwrap());
        let evaluated = run("eval", Invocation::new().set("weights", w.clone()).set("samples", json!(20)));
        assert!(evaluated.outputs["new_orbit"].as_f64().is_some_and(|a| (0.0..=1.0).contains(&a)));
        let generated = run("generate", Invocation::new().set("weights", w.clone()).set("prompt", json!("1,2")).set("max_new", json!(4)).set("seed", json!(9)));
        let again = run("generate", Invocation::new().set("weights", w.clone()).set("prompt", json!("1,2")).set("max_new", json!(4)).set("seed", json!(9)));
        assert_eq!(generated.outputs["tokens"].as_array().unwrap().len(), 6);
        assert_eq!(generated.outputs["tokens"], again.outputs["tokens"], "the seed reproduces the sample");
        let logits = run("generate", Invocation::new().set("weights", w).set("prompt", json!("1")).set("logits_only", json!(true)));
        assert_eq!(logits.outputs["logits"].as_array().unwrap().len(), 64);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bad_input_is_refused_by_name() {
        assert!(parse_prompt("1,x").unwrap_err().contains("\"x\""));
        assert!(train_spec().validate(Invocation::new().set("save", json!("s")).set("block", json!(65))).is_err());
        let mut reg = Registry::new();
        reg.register(Arc::new(ToyMoeProvider));
        let e = reg.run(MODEL, "eval", Invocation::new().set("weights", json!("/nonexistent/moe.safetensors")), &mut |_| {}).unwrap_err();
        assert!(e.contains("cannot read MoE weights"), "{e}");
    }
}
