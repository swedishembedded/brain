// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The PID controller transformer's served actions: `train` it on simulated
//! plants, `rollout` a checkpoint in closed loop against each plant's oracle
//! controller (on the training plants and on interpolated ones it never
//! saw), and `profile` one inference.
//!
//! The model is brain's own with no released checkpoint and no store spec,
//! so a checkpoint is named by path (host side); `train` writes one.
//!
//! Swedish Embedded AB implements learned controllers and their real-time
//! inference on embedded targets for its clients. If your team needs
//! expertise in control with sequence models then you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Invocation, Manifest, Outcome, ParamSpec, ParamType, Progress, Provider};
use serde_json::json;

use crate::data as pid_data;
use crate::train::{closed_loop, load, profiling_context, weights_of, PlantReport, TrainCfg};
use crate::Pid;

pub const MODEL: &str = "brain/toypid";

/// Closed-loop steps each plant is run for.
const ROLLOUT_STEPS: usize = 180;
/// Most sequences one step may average.
const MAX_BATCH: f64 = 4_096.0;
/// Largest model width, depth or head count a run may ask for.
const MAX_DIM: f64 = 8_192.0;
/// Most trajectories or trajectory steps a run may simulate.
const MAX_SIM: f64 = 1_000_000.0;
/// Largest device budget a run may plan against, bytes (1 TiB).
const MAX_BUDGET: f64 = 1_099_511_627_776.0;
/// Most timed inferences one profile may run.
const MAX_CYCLES: f64 = 1_000_000.0;

fn weights_param() -> ParamSpec {
    ParamSpec::new("weights", ParamType::Str, "the checkpoint (.safetensors) `train` wrote, on the serving machine").required().host_resolved()
}

pub fn train_spec() -> ActionSpec {
    let d = TrainCfg::default();
    let int = |name: &str, what: &str, default: f64, min: f64, max: f64| ParamSpec::new(name, ParamType::Int, what).default(json!(default as u64)).min(min).max(max);
    ActionSpec::new("train", "train the controller on simulated plant trajectories, then report it in closed loop on the training and interpolated plants").cancellable()
        .training()
        .streaming()
        .from_scratch()
        .param(ParamSpec::new("save", ParamType::Str, "where to write the checkpoint (.safetensors), on the serving machine").required().host_resolved())
        .param(int("steps", "optimizer steps", d.steps as f64, 1.0, capability::bounds::TRAIN_STEPS))
        .param(int("batch", "sequences averaged into each step (split into microbatches that fit mem_budget)", d.eff_batch as f64, 1.0, MAX_BATCH))
        .param(int("seq_steps", "event/effect records per training window", d.seq_steps as f64, 1.0, MAX_SIM))
        .param(int("block", "tokens per window; raised to fit seq_steps records", d.t_pad as f64, 1.0, capability::bounds::CONTEXT_TOKENS))
        .param(int("d_model", "model width", d.d_model as f64, 1.0, MAX_DIM))
        .param(int("layers", "transformer layers", d.layers as f64, 1.0, MAX_DIM))
        .param(int("heads", "attention heads", d.heads as f64, 1.0, MAX_DIM))
        .param(ParamSpec::new("lr", ParamType::Float, "AdamW learning rate").default(json!(d.lr)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("weight_decay", ParamType::Float, "AdamW decoupled weight decay").default(json!(d.wd)).min(0.0).max(capability::bounds::UNIT))
        .param(int("trajectories", "simulated trajectories per training plant", d.n_traj as f64, 1.0, MAX_SIM))
        .param(int("trajectory_len", "steps per simulated trajectory", d.traj_len as f64, 1.0, MAX_SIM))
        .param(int("mem_budget", "device bytes the microbatch planner may use", d.mem_budget as f64, 1.0, MAX_BUDGET))
        .param(int("checkpoint_every", "steps between checkpoint writes; 0 writes only at the end", d.ckpt_every as f64, 0.0, capability::bounds::TRAIN_STEPS))
        .param(ParamSpec::new("seed", ParamType::Int, "seeds the plants' trajectories, the initialization and the batches (omit for random; the outcome echoes it)").min(0.0))
        .param(ParamSpec::new(
            "activations",
            ParamType::Enum(model::saved::SavedPolicy::ACTIVATIONS_CHOICES.map(String::from).to_vec()),
            "what the backward keeps of the forward (default: the model's own)",
        ))
}

pub fn rollout_spec() -> ActionSpec {
    ActionSpec::new("rollout", "run a checkpoint in closed loop on the training and interpolated plants, against each plant's oracle controller").cancellable().param(weights_param())
}

pub fn profile_spec() -> ActionSpec {
    ActionSpec::new("profile", "time one inference (a forward to the DECIDE position) over a context of the given length").cancellable()
        .param(weights_param())
        .param(ParamSpec::new("cycles", ParamType::Int, "timed inferences").default(json!(100)).min(1.0).max(MAX_CYCLES))
        .param(ParamSpec::new("warmup", ParamType::Int, "untimed inferences first").default(json!(10)).min(0.0).max(MAX_CYCLES))
        .param(ParamSpec::new("seq_len", ParamType::Int, "context tokens (at most the checkpoint's block size)").default(json!(64)).min(1.0).max(capability::bounds::CONTEXT_TOKENS))
}

pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "PID event/effect control transformer: a learned controller for first-order plants, judged in closed loop against each plant's oracle.", vec![train_spec(), rollout_spec(), profile_spec()])
}

fn reports(dec: &Pid, cfg: &crate::PidConfig) -> serde_json::Value {
    let side = |plants: &[pid_data::PlantSpec]| {
        let r: Vec<PlantReport> = closed_loop(dec, cfg, plants, ROLLOUT_STEPS);
        let gap = r.iter().map(|p| p.model_mse - p.oracle_mse).sum::<f32>() / r.len().max(1) as f32;
        let plants: Vec<_> = r.iter().map(|p| json!({"tau": p.tau, "gain": p.gain, "model_mse": p.model_mse, "oracle_mse": p.oracle_mse, "steady_state_error": p.steady_state_error})).collect();
        json!({"plants": plants, "mean_mse_gap": gap})
    };
    json!({"training": side(&pid_data::training_plants()), "validation": side(&pid_data::validation_plants())})
}

#[derive(Default)]
pub struct ToyPidProvider;

impl Provider for ToyPidProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        manifest().actions.into_iter().find(|a| a.name == name).map(|spec| Arc::new(ToyPidAction { spec }) as Arc<dyn Action>)
    }
}

struct ToyPidAction {
    spec: ActionSpec,
}

impl Action for ToyPidAction {
    fn spec(&self) -> ActionSpec {
        self.spec.clone()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let n = |name: &str, default: u64| inv.get_i64(name).map_or(default, |v| v.max(0) as u64);
        let weights = || inv.get_str("weights").ok_or_else(|| format!("toypid {}: missing required param 'weights'", self.spec.name));
        match self.spec.name.as_str() {
            "train" => {
                let d = TrainCfg::default();
                let tc = TrainCfg {
                    steps: n("steps", d.steps as u64).max(1) as u32,
                    eff_batch: n("batch", d.eff_batch as u64).max(1) as u32,
                    seq_steps: n("seq_steps", d.seq_steps as u64).max(1) as u32,
                    t_pad: n("block", d.t_pad as u64).max(1) as u32,
                    d_model: n("d_model", d.d_model as u64).max(1) as u32,
                    layers: n("layers", d.layers as u64).max(1) as u32,
                    heads: n("heads", d.heads as u64).max(1) as u32,
                    lr: inv.get_f64("lr").map_or(d.lr, |v| v as f32),
                    wd: inv.get_f64("weight_decay").map_or(d.wd, |v| v as f32),
                    seed: inv.seed()?.unwrap_or_else(data::rng::random_seed),
                    n_traj: n("trajectories", d.n_traj as u64).max(1) as usize,
                    traj_len: n("trajectory_len", d.traj_len as u64).max(1) as usize,
                    mem_budget: n("mem_budget", d.mem_budget).max(1),
                    out: inv.get_str("save").ok_or("toypid train: missing required param 'save'")?,
                    ckpt_every: n("checkpoint_every", d.ckpt_every as u64) as u32,
                    activations: inv.get_str("activations").map(|a| model::saved::SavedPolicy::parse_activations(&a)).transpose()?,
                };
                let t = crate::train::train(&tc, &mut |step, loss| {
                    progress(Progress::step(step, tc.steps, loss.map_or_else(String::new, |l| format!("loss {l:.4}"))));
                    !inv.cancel.is_cancelled()
                })?;
                if t.steps < tc.steps {
                    progress(Progress::step(t.steps, tc.steps, format!("cancelled; the checkpoint so far is at {}", tc.out)));
                    return Err(capability::failure::CANCELLED.to_string());
                }
                let dec = Pid::new(t.cfg.clone(), 1, t.cfg.block_size, &weights_of(&t.model));
                Ok(Outcome::new()
                    .set("final_loss", json!(t.final_loss))
                    .set("microbatch", json!(t.microbatch))
                    .set("accumulation", json!(t.accumulation))
                    .set("block", json!(t.cfg.block_size))
                    .set("seed", json!(tc.seed))
                    .set("saved", json!(tc.out))
                    .set("closed_loop", reports(&dec, &t.cfg)))
            }
            "rollout" => {
                let (dec, cfg) = load(&weights()?)?;
                Ok(Outcome::new().set("closed_loop", reports(&dec, &cfg)))
            }
            "profile" => {
                let (dec, cfg) = load(&weights()?)?;
                let len = (n("seq_len", 64) as usize).clamp(1, cfg.block_size as usize);
                let (mean, min, max) = dec.profile_inference(&profiling_context(len), n("cycles", 100).max(1) as usize, n("warmup", 10) as usize);
                Ok(Outcome::new()
                    .set("context", json!(len))
                    .set("mean_ms", json!(mean * 1e3))
                    .set("min_ms", json!(min * 1e3))
                    .set("max_ms", json!(max * 1e3))
                    .set("inferences_per_second", json!(1.0 / mean))
                    .set("config", json!({"d_model": cfg.d_model, "layers": cfg.n_layers, "heads": cfg.n_heads, "block": cfg.block_size})))
            }
            other => Err(format!("toypid: unknown action {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::Registry;

    /// A short run trains, writes a checkpoint the other actions read back,
    /// and reports both plant sets.
    #[test]
    fn train_then_rollout_and_profile_through_the_actions() {
        let dir = std::env::temp_dir().join(format!("toypid-caps-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let save = dir.join("pid.safetensors");
        let mut reg = Registry::new();
        reg.register(Arc::new(ToyPidProvider));
        let run = |action: &str, inv: Invocation| reg.run(MODEL, action, inv, &mut |_| {}).unwrap_or_else(|e| panic!("{action}: {e}"));
        let small = Invocation::new()
            .set("save", json!(save.to_str().unwrap()))
            .set("steps", json!(2))
            .set("batch", json!(2))
            .set("seq_steps", json!(2))
            .set("d_model", json!(16))
            .set("heads", json!(2))
            .set("trajectories", json!(2))
            .set("trajectory_len", json!(20))
            .set("seed", json!(4));
        let trained = run("train", small);
        assert_eq!(trained.outputs["seed"].as_u64(), Some(4));
        assert!(trained.outputs["closed_loop"]["validation"]["plants"].as_array().is_some_and(|p| !p.is_empty()));
        let w = json!(save.to_str().unwrap());
        assert!(run("rollout", Invocation::new().set("weights", w.clone())).outputs["closed_loop"]["training"]["mean_mse_gap"].is_number());
        let p = run("profile", Invocation::new().set("weights", w).set("cycles", json!(2)).set("warmup", json!(0)).set("seq_len", json!(8)));
        assert_eq!(p.outputs["context"].as_u64(), Some(8));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_checkpoint_is_a_clean_error() {
        let mut reg = Registry::new();
        reg.register(Arc::new(ToyPidProvider));
        let e = reg.run(MODEL, "rollout", Invocation::new().set("weights", json!("/nonexistent/pid.safetensors")), &mut |_| {}).unwrap_err();
        assert!(e.contains("no such checkpoint"), "{e}");
    }
}
