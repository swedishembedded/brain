// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Federated expert training over a toy MoE checkpoint, as actions beside the
//! model's served ones: `split` a checkpoint into a content-addressed shard
//! directory (the shared backbone plus one shard per expert), `train_expert`
//! one expert against the frozen backbone and keep only its shard, `assemble`
//! a base directory and expert overlays into one checkpoint, `merge` a shard
//! directory back into one, and `verify` a directory's hashes.
//!
//! They are the application around the model (`federated` does the sharding,
//! `toymoe::train::train_expert` the worker step), so they live in this
//! program rather than in the model crate.
//!
//! Swedish Embedded AB implements federated and distributed training of
//! Mixture-of-Experts models for its clients. If your team needs expertise in
//! training experts on separate machines then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::path::Path;
use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Invocation, Manifest, Outcome, ParamSpec, ParamType, Progress, Provider};
use serde_json::json;

/// Most steps one expert run may take.
const MAX_STEPS: f64 = capability::bounds::TRAIN_STEPS;
/// Defaults of an expert run.
const DEFAULT_STEPS: i64 = 200;
const DEFAULT_BATCH: i64 = 16;
const DEFAULT_BLOCK: i64 = 64;
const DEFAULT_LR: f64 = 6e-4;

fn path(name: &str, help: &str) -> ParamSpec {
    ParamSpec::new(name, ParamType::Str, help).required().host_resolved()
}

fn specs() -> Vec<ActionSpec> {
    vec![
        ActionSpec::new("split", "split a checkpoint into a shard directory: the shared backbone and one shard per expert, with a manifest of their hashes")
            .from_scratch()
            .param(path("weights", "the checkpoint (.safetensors) to split"))
            .param(path("dest", "the shard directory to write")),
        ActionSpec::new("verify", "check every shard of a directory against its manifest's hashes").from_scratch().param(path("dir", "the shard directory")),
        ActionSpec::new("merge", "merge a shard directory back into one checkpoint")
            .from_scratch()
            .param(path("dir", "the shard directory"))
            .param(path("dest", "where to write the .safetensors")),
        ActionSpec::new("assemble", "assemble a base shard directory and expert overlay directories into one checkpoint")
            .from_scratch()
            .param(path("base", "the base shard directory"))
            .param(ParamSpec::new("overlays", ParamType::Str, "comma-separated overlay shard directories (each from train_expert)").default(json!("")).host_resolved())
            .param(path("dest", "where to write the .safetensors")),
        ActionSpec::new("train_expert", "train one expert against the frozen shared backbone of a checkpoint and keep only its shard, as an overlay directory for assemble")
            .training()
            .streaming()
            .from_scratch()
            .param(path("weights", "the base checkpoint (.safetensors)"))
            .param(path("dest", "the overlay directory to write"))
            .param(ParamSpec::new("expert", ParamType::Int, "which expert to train").min(0.0).max(1024.0).required())
            .param(ParamSpec::new("steps", ParamType::Int, "optimizer steps").default(json!(DEFAULT_STEPS)).min(1.0).max(MAX_STEPS))
            .param(ParamSpec::new("batch", ParamType::Int, "sequences per step").default(json!(DEFAULT_BATCH)).min(1.0).max(1024.0))
            .param(ParamSpec::new("block", ParamType::Int, "tokens per sequence").default(json!(DEFAULT_BLOCK)).min(1.0).max(64.0))
            .param(ParamSpec::new("lr", ParamType::Float, "AdamW learning rate").default(json!(DEFAULT_LR)).min(0.0).max(capability::bounds::UNIT))
            .param(ParamSpec::new("seed", ParamType::Int, "seeds the batches (omit for random; the outcome echoes it)").min(0.0))
            .param(ParamSpec::new(
                "activations",
                ParamType::Enum(model::saved::SavedPolicy::ACTIVATIONS_CHOICES.map(String::from).to_vec()),
                "what the backward keeps of the forward (default: every layer's activations)",
            )),
    ]
}

/// The model's served manifest plus the federated actions.
pub fn manifest() -> Manifest {
    let mut m = toymoe::caps::manifest();
    m.actions.extend(specs());
    m
}

/// The served provider, extended with the federated actions.
pub struct WithFederated;

impl Provider for WithFederated {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match specs().into_iter().find(|s| s.name == name) {
            Some(spec) => Some(Arc::new(Federated { spec })),
            None => toymoe::caps::ToyMoeProvider.action(name),
        }
    }
}

struct Federated {
    spec: ActionSpec,
}

fn io(what: &str, e: std::io::Error) -> String {
    format!("{what}: {e}")
}

impl Action for Federated {
    fn spec(&self) -> ActionSpec {
        self.spec.clone()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let name = self.spec.name.as_str();
        let s = |p: &str| inv.get_str(p).ok_or_else(|| format!("{name}: missing required param '{p}'"));
        match name {
            "split" => {
                let m = federated::split(&s("weights")?, Path::new(&s("dest")?)).map_err(|e| io("split", e))?;
                Ok(Outcome::new().set("experts", json!(m.experts)).set("files", json!(m.files.len())))
            }
            "verify" => {
                let m = federated::verify(Path::new(&s("dir")?)).map_err(|e| io("verify", e))?;
                Ok(Outcome::new().set("experts", json!(m.experts)).set("files", json!(m.files.len())).set("verified", json!(true)))
            }
            "merge" => {
                let dest = s("dest")?;
                federated::merge_to_full(Path::new(&s("dir")?), &dest).map_err(|e| io("merge", e))?;
                Ok(Outcome::new().set("saved", json!(dest)))
            }
            "assemble" => {
                let dest = s("dest")?;
                let overlays: Vec<String> = inv.get_str("overlays").unwrap_or_default().split(',').map(str::trim).filter(|o| !o.is_empty()).map(String::from).collect();
                let overlays: Vec<&Path> = overlays.iter().map(Path::new).collect();
                federated::assemble(Path::new(&s("base")?), &overlays, &dest).map_err(|e| io("assemble", e))?;
                Ok(Outcome::new().set("saved", json!(dest)))
            }
            "train_expert" => train_expert(inv, progress),
            other => Err(format!("toymoe: no action '{other}'")),
        }
    }
}

/// Train one expert into a full checkpoint beside the overlay, keep only its
/// shard (and the shared one) as the overlay, and remove the full checkpoint.
fn train_expert(inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let dest = inv.get_str("dest").ok_or("train_expert: missing required param 'dest'")?;
    let expert = inv.get_i64("expert").ok_or("train_expert: missing required param 'expert'")? as u32;
    let seed = inv.seed()?.unwrap_or_else(data::rng::random_seed);
    let steps = inv.get_i64("steps").unwrap_or(DEFAULT_STEPS) as u32;
    let out_dir = Path::new(&dest);
    std::fs::create_dir_all(out_dir).map_err(|e| io(&dest, e))?;
    let full = out_dir.join(".worker_full.safetensors");
    let args = toymoe::train::ExpertTrainArgs {
        base_weights: inv.get_str("weights").ok_or("train_expert: missing required param 'weights'")?,
        expert,
        out: full.to_string_lossy().into_owned(),
        steps,
        b: inv.get_i64("batch").unwrap_or(DEFAULT_BATCH) as u32,
        t: inv.get_i64("block").unwrap_or(DEFAULT_BLOCK) as u32,
        lr: inv.get_f64("lr").unwrap_or(DEFAULT_LR) as f32,
        seed,
        saved: inv.get_str("activations").map(|a| model::saved::SavedPolicy::parse_activations(&a)).transpose()?,
    };
    let trained = toymoe::train::train_expert(args, &mut |step, loss| {
        progress(Progress::step(step, steps, format!("loss {loss:.4}")));
        !inv.cancel.is_cancelled()
    });
    let trained = match trained {
        Ok(t) => t,
        Err(e) => {
            let _ = std::fs::remove_file(&full);
            return Err(format!("train_expert: {e}"));
        }
    };
    let manifest = federated::split_filtered(&full.to_string_lossy(), out_dir, Some(&[expert]));
    let _ = std::fs::remove_file(&full);
    let manifest = manifest.map_err(|e| io("train_expert: keeping the expert's shard", e))?;
    Ok(Outcome::new()
        .set("expert", json!(expert))
        .set("initial_loss", json!(trained.initial_loss))
        .set("final_loss", json!(trained.final_loss))
        .set("seed", json!(seed))
        .set("files", json!(manifest.files.len()))
        .set("saved", json!(dest)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Train a base, split it, train expert 0 as an overlay, assemble the
    /// overlay over the base and verify the base directory: the whole
    /// federated round on a tiny run, through the actions.
    #[test]
    fn a_federated_round_runs_through_the_actions() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("toymoe-federated-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = |n: &str| dir.join(n).to_string_lossy().into_owned();
        let mut reg = capability::Registry::new();
        reg.register(Arc::new(WithFederated));
        let run = |action: &str, inv: Invocation| reg.run(toymoe::caps::MODEL, action, inv, &mut |_| {}).unwrap_or_else(|e| panic!("{action}: {e}"));
        run("train", Invocation::new().set("save", json!(p("base.safetensors"))).set("steps", json!(2)).set("batch", json!(2)).set("block", json!(16)).set("seed", json!(1)));
        let split = run("split", Invocation::new().set("weights", json!(p("base.safetensors"))).set("dest", json!(p("base"))));
        assert!(split.outputs["experts"].as_array().is_some_and(|e| !e.is_empty()));
        run("verify", Invocation::new().set("dir", json!(p("base"))));
        let trained = run(
            "train_expert",
            Invocation::new().set("weights", json!(p("base.safetensors"))).set("dest", json!(p("exp0"))).set("expert", json!(0)).set("steps", json!(2)).set("batch", json!(2)).set("block", json!(16)).set("seed", json!(3)),
        );
        assert_eq!(trained.outputs["seed"].as_u64(), Some(3));
        assert!(!dir.join("exp0").join(".worker_full.safetensors").exists());
        run("assemble", Invocation::new().set("base", json!(p("base"))).set("overlays", json!(p("exp0"))).set("dest", json!(p("final.safetensors"))));
        run("merge", Invocation::new().set("dir", json!(p("base"))).set("dest", json!(p("merged.safetensors"))));
        assert!(dir.join("final.safetensors").is_file() && dir.join("merged.safetensors").is_file());
        let err = reg.run(toymoe::caps::MODEL, "train_expert", Invocation::new().set("weights", json!(p("base.safetensors"))).set("dest", json!(p("x"))).set("expert", json!(1000)).set("block", json!(16)), &mut |_| {}).unwrap_err();
        assert!(err.contains("does not exist"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
