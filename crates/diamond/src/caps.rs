// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DIAMOND's served actions: `import` (the reference agent checkpoint to a
//! brain checkpoint), `export` (the UNet as fp32 ONNX), `finetune` (on a
//! recorded episode dataset), `rollout` (headless play: the frames of an
//! action script), `replay` (inspect and verify a recorded episode dataset)
//! and `bench` (per-frame timings).
//!
//! `rollout`, `replay` and `bench` run either the pretrained DIAMOND world
//! model or the deterministic GPU-free `fake` one (`model`), on the engine or,
//! given the ONNX export of the UNet, on the Intel NPU. The interactive window
//! is an application's, not the served model's, and lives in
//! `samples/models/diamond`.
//!
//! Swedish Embedded AB implements world-model simulators that train and run on
//! its clients' own hardware. If your team needs expertise in diffusion world
//! models for control and simulation then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use serde_json::json;

use crate::train::{DiamondTrainer, FinetuneConfig, Transition};
use crate::worldrun::{self, Kind, ModelRequest, RolloutTargets, VerifyRequest};

/// The model id the catalog serves DIAMOND under.
pub const MODEL: &str = "brain/diamond";

/// Most game actions an import may declare (the reference games have at most 18).
const MAX_ACTIONS: f64 = 1_024.0;
/// Largest gradient-norm clip a fine-tune may ask for.
const MAX_CLIP: f64 = 1_000.0;
/// Most frames one `bench` call may time.
const MAX_BENCH_FRAMES: f64 = 100_000.0;

fn model_names() -> Vec<String> {
    vec!["diamond".into(), "fake".into()]
}

/// The params that pick which world model a run uses.
fn model_params(spec: ActionSpec) -> ActionSpec {
    spec.param(ParamSpec::new("model", ParamType::Enum(model_names()), "diamond: the pretrained Atari-100k EDM world model, from the checkpoint; fake: the deterministic GPU-free test model").default(json!("diamond")))
        .param(ParamSpec::new("seed", ParamType::Int, "seeds the noise stream (omit for random; the run reports the seed it used)").min(0.0))
        .param(ParamSpec::new("denoise_steps", ParamType::Int, "denoising steps per frame: 3 or more is the full quality, 2 and 1 trade it for speed; 0 = the model's default").default(json!(0)).min(0.0).max(capability::bounds::DENOISE_STEPS))
        .param(ParamSpec::new("onnx", ParamType::Str, "path of the ONNX export of the UNet (from `export`): runs it as a compiled OpenVINO graph on the Intel NPU, the sampler staying on the host; omit to run on the engine").host_resolved())
}

pub fn import_spec() -> ActionSpec {
    ActionSpec::new("import", "convert a reference DIAMOND agent checkpoint (agent.pt) into brain's checkpoint, every denoiser tensor verified; the reward/termination and actor-critic sub-models are not imported")
        .from_scratch()
        .param(ParamSpec::new("source", ParamType::Str, "path of the reference agent.pt, on the serving machine").required().host_resolved())
        .param(ParamSpec::new("dest", ParamType::Str, "path to write the converted .safetensors to, on the serving machine").required().host_resolved())
        .param(ParamSpec::new("actions_count", ParamType::Int, "the game's action count (Breakout: 4), checked against the checkpoint's action embedding").default(json!(4)).min(1.0).max(MAX_ACTIONS))
}

pub fn export_spec() -> ActionSpec {
    ActionSpec::new("export", "write the checkpoint's UNet as an fp32 ONNX graph, which `rollout`, `replay` and `bench` run on the Intel NPU through OpenVINO (the sampler stays on the host)")
        .param(ParamSpec::new("dest", ParamType::Str, "path to write the .onnx to, on the serving machine").required().host_resolved())
}

pub fn finetune_spec() -> ActionSpec {
    ActionSpec::new("finetune", "fine-tune the denoiser's convolutions on a recorded episode dataset (AdamW, batch 1), from the resolved checkpoint; a run whose loss diverges saves nothing")
        .training()
        .streaming()
        .param(ParamSpec::new("data", ParamType::Str, "directory of the recorded episode dataset to train on (from `rollout`'s record), on the serving machine").required().host_resolved())
        .param(ParamSpec::new("save", ParamType::Str, "path to write the fine-tuned .safetensors to, on the serving machine").required().host_resolved())
        .param(ParamSpec::new("steps", ParamType::Int, "optimizer steps").default(json!(300)).min(1.0).max(capability::bounds::TRAIN_STEPS))
        .param(ParamSpec::new("lr", ParamType::Float, "AdamW learning rate (batch-1 fine-tuning diverges when it is too high)").default(json!(1e-4)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("wd", ParamType::Float, "AdamW weight decay").default(json!(1e-2)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("clip", ParamType::Float, "gradient-norm clip").default(json!(1.0)).min(0.0).max(MAX_CLIP))
        .param(ParamSpec::new("seed", ParamType::Int, "seeds the noise, the noise levels and the dataset sampling (omit for random)").min(0.0))
        .param(ParamSpec::new("activations", ParamType::Enum(model::saved::SavedPolicy::ACTIVATIONS_CHOICES.map(String::from).to_vec()), "what the backward keeps of the forward: every unit's intermediates, or only each unit's input, rebuilt in the backward for the same gradients at lower memory (default keep)"))
}

pub fn rollout_spec() -> ActionSpec {
    model_params(ActionSpec::new("rollout", "play the world model headless over an action script and return the frames as a clip, with a hash of the rollout that is equal for equal runs; optionally dump the frames or record them as an episode dataset"))
        .streaming()
        .from_scratch()
        .param(ParamSpec::new("frames", ParamType::Int, "steps to play: with an action script, 0 plays all of it and a smaller count cuts it short; without one, that many no-op steps (0 = one)").default(json!(0)).min(0.0).max(capability::bounds::VIDEO_FRAMES))
        .param(ParamSpec::new("action_seq", ParamType::Str, "the action ids to play, comma separated, one per step (for example 1,2,0); none = no-ops"))
        .param(ParamSpec::new("actions", ParamType::Str, "path of a text file of action ids separated by whitespace, one per step, instead of action_seq").host_resolved())
        .param(ParamSpec::new("fps", ParamType::Int, "frame rate recorded into an episode dataset").default(json!(15)).min(1.0).max(capability::bounds::FPS))
        .param(ParamSpec::new("seed_context", ParamType::Str, "directory of .ppm frames to start the world from, oldest first, at the model's frame size").host_resolved())
        .param(ParamSpec::new("record", ParamType::Str, "directory to record the session into as an episode dataset (`replay` and `finetune` read it); must not exist yet").host_resolved())
        .param(ParamSpec::new("dump_ppm", ParamType::Str, "directory to write every frame into as frame_NNNNNN.ppm").host_resolved())
        .param(ParamSpec::new("hashes", ParamType::Bool, "also list the hash of every frame").default(json!(false)))
        .output(BlobSpec::new("clip", Media::Video, "the rolled-out frames, one per step"))
}

pub fn replay_spec() -> ActionSpec {
    model_params(ActionSpec::new("replay", "describe a recorded episode dataset and, with verify, re-generate it through the model and compare frame by frame; the noise streams align when seed and denoise_steps match the recording's"))
        .from_scratch()
        .param(ParamSpec::new("episode", ParamType::Str, "directory of the recorded episode dataset, on the serving machine").required().host_resolved())
        .param(ParamSpec::new("verify", ParamType::Bool, "re-generate the recording through the model and compare it").default(json!(false)))
        .param(ParamSpec::new("tolerance", ParamType::Float, "the largest mean absolute difference per frame, on a 0-1 scale, that still counts as matching (default 2/255)").default(json!(2.0 / 255.0)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("context", ParamType::Int, "steps burned before the first recorded frames are loaded as context, so verification starts there (diamond only; the fake model verifies from 0)").default(json!(0)).min(0.0).max(capability::bounds::TRAIN_COUNT))
}

pub fn bench_spec() -> ActionSpec {
    model_params(ActionSpec::new("bench", "per-frame timings of an unpaced run, or with profile the per-kernel time of one UNet forward"))
        .from_scratch()
        .param(ParamSpec::new("frames", ParamType::Int, "steps to time").default(json!(200)).min(1.0).max(MAX_BENCH_FRAMES))
        .param(ParamSpec::new("profile", ParamType::Bool, "instead time each kernel of one UNet forward (one submit per step: for ranking kernels, not absolute time; diamond only)").default(json!(false)))
}

/// The full, static capability manifest: safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "DIAMOND, the Atari-100k diffusion world model: import of the reference checkpoint, fine-tuning on recorded episodes, headless rollout, replay verification and benchmarking.", vec![import_spec(), export_spec(), finetune_spec(), rollout_spec(), replay_spec(), bench_spec()])
}

/// The executable DIAMOND model behind the manifest, over one checkpoint
/// (none for `import` and the `fake` model). Construction is free: the
/// checkpoint loads on the call that needs it.
pub struct DiamondProvider {
    weights: Option<String>,
}

impl DiamondProvider {
    pub fn new(weights: Option<String>) -> DiamondProvider {
        DiamondProvider { weights }
    }
}

/// DIAMOND over `assembly`'s `weights` role; an assembly without one serves
/// only the actions that run without a checkpoint.
pub fn provider(assembly: &capability::Assembly) -> Result<Arc<dyn Provider>, String> {
    Ok(Arc::new(DiamondProvider::new(assembly.role_path("weights").ok())))
}

impl Provider for DiamondProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        manifest().actions.iter().any(|a| a.name == name).then(|| Arc::new(DiamondAction { name: name.to_string(), weights: self.weights.clone() }) as Arc<dyn Action>)
    }
}

struct DiamondAction {
    name: String,
    weights: Option<String>,
}

impl Action for DiamondAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == self.name).expect("known action")
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match self.name.as_str() {
            "import" => import(inv),
            "export" => self.export(inv),
            "finetune" => self.finetune(inv, progress),
            "rollout" => self.rollout(inv, progress),
            "replay" => self.replay(inv),
            "bench" => self.bench(inv),
            other => Err(format!("diamond: unknown action {other}")),
        }
    }
}

fn required(inv: &Invocation, action: &str, name: &str) -> Result<String, String> {
    inv.get_str(name).ok_or_else(|| format!("diamond {action}: missing required param '{name}'"))
}

/// The model a call's `model`, `seed`, `denoise_steps`, `onnx` and
/// `seed_context` params name over `weights`, with the seed it will use. Shared
/// by the headless actions and the sample's window, so both build the same
/// model from the same params.
pub fn model_request(inv: &Invocation, weights: Option<String>, action: &str) -> Result<(ModelRequest, u64), String> {
    let kind = match inv.get_str("model").as_deref() {
        Some("fake") => Kind::Fake,
        _ => Kind::Diamond { weights: weights.ok_or_else(|| format!("diamond {action}: no DIAMOND checkpoint (pull one, import one, or name it with --weights)"))? },
    };
    let npu_onnx = inv.get_str("onnx").filter(|p| !p.is_empty());
    if npu_onnx.is_some() && matches!(kind, Kind::Fake) {
        return Err(format!("diamond {action}: the NPU runs the diamond model, not the fake one"));
    }
    let seed = seed_of(inv);
    let denoise_steps = inv.get_i64("denoise_steps").unwrap_or(0).max(0) as u32;
    Ok((ModelRequest { kind, npu_onnx, seed, denoise_steps, seed_context: inv.get_str("seed_context").filter(|p| !p.is_empty()) }, seed))
}

fn seed_of(inv: &Invocation) -> u64 {
    inv.get_i64("seed").map_or_else(data::rng::random_seed, |s| s as u64)
}

fn import(inv: &Invocation) -> ActionResult {
    let source = required(inv, "import", "source")?;
    let dest = required(inv, "import", "dest")?;
    let actions = inv.get_i64("actions_count").unwrap_or(4).max(1) as u32;
    let cfg = crate::import::import(&source, &dest, actions).map_err(|e| format!("diamond import: {e}"))?;
    Ok(Outcome::new()
        .set("tensors", json!(cfg.param_list().len()))
        .set("height", json!(cfg.h))
        .set("width", json!(cfg.w))
        .set("num_actions", json!(cfg.num_actions))
        .set("saved", json!(dest)))
}

impl DiamondAction {
    fn weights(&self, action: &str) -> Result<String, String> {
        self.weights.clone().ok_or_else(|| format!("diamond {action}: no DIAMOND checkpoint (pull one, import one, or name it with --weights)"))
    }

    fn export(&self, inv: &Invocation) -> ActionResult {
        let weights = self.weights("export")?;
        let dest = required(inv, "export", "dest")?;
        let cfg = crate::npu::export_onnx(&weights, &dest).map_err(|e| format!("diamond export: {e}"))?;
        Ok(Outcome::new()
            .set("height", json!(cfg.h))
            .set("width", json!(cfg.w))
            .set("context_frames", json!(cfg.num_steps_conditioning))
            .set("cond_channels", json!(cfg.cond_channels))
            .set("saved", json!(dest)))
    }

    fn model_request(&self, inv: &Invocation, action: &str) -> Result<(ModelRequest, u64), String> {
        model_request(inv, self.weights.clone(), action)
    }

    fn finetune(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let weights = self.weights("finetune")?;
        let data_dir = required(inv, "finetune", "data")?;
        let save = required(inv, "finetune", "save")?;
        let saved_policy = inv.get_str("activations").map(|a| model::saved::SavedPolicy::parse_activations(&a)).transpose()?;
        let seed = seed_of(inv);
        let schedule = FinetuneConfig {
            steps: inv.get_i64("steps").unwrap_or(300).max(1) as u32,
            lr: inv.get_f64("lr").unwrap_or(1e-4) as f32,
            weight_decay: inv.get_f64("wd").unwrap_or(1e-2) as f32,
            clip: Some(inv.get_f64("clip").unwrap_or(1.0) as f32),
            seed,
        };

        let (cfg, tensors) = crate::import::load(&weights).map_err(|e| format!("diamond finetune: {e}"))?;
        let ds = worldrun::open_dataset(&data_dir).map_err(|e| format!("diamond finetune: {e}"))?;
        let nsc = cfg.num_steps_conditioning as usize;
        if (ds.c, ds.h, ds.w) != (cfg.img_channels, cfg.h, cfg.w) || ds.num_actions != cfg.num_actions {
            return Err(format!(
                "diamond finetune: the dataset is {}x{}x{} with {} actions but the model takes {}x{}x{} with {}",
                ds.c, ds.h, ds.w, ds.num_actions, cfg.img_channels, cfg.h, cfg.w, cfg.num_actions
            ));
        }
        if !ds.episodes.iter().any(|e| e.len > nsc) {
            return Err(format!("diamond finetune: no episode in {data_dir} has the {} frames a training window needs", nsc + 1));
        }
        let frame_len = (cfg.img_channels * cfg.h * cfg.w) as usize;

        let mut trainer = DiamondTrainer::from_tensors(cfg.clone(), &tensors, None);
        if let Some(policy) = saved_policy {
            trainer.set_saved_policy(policy).map_err(|e| format!("diamond finetune: activations {policy:?}: {e}"))?;
        }
        let mut dataset_rng = data::rng::Rng::new(seed ^ DATASET_STREAM);
        let sampler = |_: &mut crate::play::NormalRng| {
            let w = ds.sample_window(&mut dataset_rng, nsc + 1).expect("an episode with enough frames was checked above");
            // Dataset frames are [0,1]; the trainer wants [-1,1]. actions[i] is
            // the action that PRODUCED frame i, so the context actions (the
            // action taken AT each context frame) are actions[1..=nsc].
            let to_pm1 = |v: &f32| v * 2.0 - 1.0;
            Transition { obs: w.frames_f32[..nsc * frame_len].iter().map(to_pm1).collect(), clean: w.frames_f32[nsc * frame_len..].iter().map(to_pm1).collect(), actions: w.actions[1..=nsc].to_vec() }
        };
        let steps = schedule.steps;
        let end = crate::train::finetune(&trainer, sampler, &schedule, |step, loss| {
            progress(Progress::step(step, steps, format!("loss {loss:.5}")));
            !inv.cancel.is_cancelled()
        });
        if end.stopped {
            return Err("cancelled".into());
        }
        if !end.last_loss.is_finite() {
            return Err(format!("diamond finetune: training diverged (non-finite loss at step {}); nothing was saved - lower lr", end.steps_run));
        }
        trainer.save(&tensors, &save).map_err(|e| format!("diamond finetune: saving {save}: {e}"))?;
        Ok(Outcome::new()
            .set("first_loss", json!(end.first_loss))
            .set("last_loss", json!(end.last_loss))
            .set("trainable_tensors", json!(crate::train::trainable_list(&cfg).len()))
            .set("seed", json!(seed))
            .set("saved", json!(save)))
    }

    fn rollout(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (req, seed) = self.model_request(inv, "rollout")?;
        let mut built = worldrun::build(&req).map_err(|e| format!("diamond rollout: {e}"))?;
        let steps = rollout_script(inv)?;
        let fps = inv.get_i64("fps").unwrap_or(15).max(1) as u32;
        let targets = RolloutTargets { ppm_dir: inv.get_str("dump_ppm").filter(|p| !p.is_empty()), record_dir: inv.get_str("record").filter(|p| !p.is_empty()) };
        let name = inv.get_str("model").unwrap_or_else(|| "diamond".into());
        let run = worldrun::rollout(built.model.as_mut(), &name, steps, fps, &targets, &inv.cancel, progress).map_err(|e| if e == "cancelled" { e } else { format!("diamond rollout: {e}") })?;
        let clip: Vec<(Vec<f32>, u32, u32)> = run.frames.iter().map(|rgb| (rgb.iter().map(|&b| b as f32 / 255.0).collect(), run.width, run.height)).collect();
        let mut outcome = Outcome::new()
            .set("model", json!(name))
            .set("backend", json!(built.backend))
            .set("seed", json!(seed))
            .set("steps", json!(run.report.steps))
            .set("fps", json!(run.report.fps))
            .set("work_ms_mean", json!(run.report.work_ms_mean))
            .set("work_ms_p95", json!(run.report.work_ms_p95))
            .set("rollout_hash", json!(format!("{:016x}", run.rollout_hash())))
            .blob("clip", capability::blob::video_blob(&clip)?);
        if inv.get_bool("hashes").unwrap_or(false) {
            outcome = outcome.set("frame_hashes", json!(run.frame_hashes.iter().map(|h| format!("{h:016x}")).collect::<Vec<_>>()));
        }
        if let Some(n) = run.recorded {
            outcome = outcome.set("recorded_frames", json!(n));
        }
        Ok(outcome)
    }

    fn replay(&self, inv: &Invocation) -> ActionResult {
        let dir = required(inv, "replay", "episode")?;
        let ds = worldrun::open_dataset(&dir).map_err(|e| format!("diamond replay: {e}"))?;
        let episodes: Vec<_> = ds.episodes.iter().map(|e| json!({"start": e.start, "len": e.len})).collect();
        let mut outcome = Outcome::new()
            .set("frames", json!(ds.n))
            .set("channels", json!(ds.c))
            .set("height", json!(ds.h))
            .set("width", json!(ds.w))
            .set("num_actions", json!(ds.num_actions))
            .set("fps", json!(ds.fps))
            .set("episodes", json!(episodes))
            .set("rewards", json!(ds.rewards().is_some()));
        if !inv.get_bool("verify").unwrap_or(false) {
            return Ok(outcome);
        }
        let (req, seed) = self.model_request(inv, "replay")?;
        let mut built = worldrun::build(&req).map_err(|e| format!("diamond replay: {e}"))?;
        let tolerance = inv.get_f64("tolerance").unwrap_or(2.0 / 255.0) as f32;
        let context = inv.get_i64("context").unwrap_or(0).max(0) as usize;
        let v = worldrun::verify(built.model.as_mut(), &ds, &VerifyRequest { tolerance, context }).map_err(|e| format!("diamond replay: {e}"))?;
        if v.failed > 0 {
            return Err(format!("diamond replay: FAIL - {} of {} frames exceeded the tolerance {tolerance:.6} (max_mad {:.6} at frame {})", v.failed, v.checked, v.max_mad, v.worst_frame));
        }
        outcome = outcome
            .set("seed", json!(seed))
            .set("verified_frames", json!(v.checked))
            .set("max_mad", json!(v.max_mad))
            .set("worst_frame", json!(v.worst_frame))
            .set("tolerance", json!(tolerance));
        Ok(outcome)
    }

    fn bench(&self, inv: &Invocation) -> ActionResult {
        let name = inv.get_str("model").unwrap_or_else(|| "diamond".into());
        if inv.get_bool("profile").unwrap_or(false) {
            if name != "diamond" {
                return Err("diamond bench: profile times the diamond UNet only".into());
            }
            let kernels = worldrun::profile_unet(&self.weights("bench")?).map_err(|e| format!("diamond bench: {e}"))?;
            let total: f64 = kernels.iter().map(|(_, ms, _)| ms).sum();
            let rows: Vec<_> = kernels.iter().map(|(kernel, ms, dispatches)| json!({"kernel": kernel, "ms": ms, "dispatches": dispatches, "share": ms / total})).collect();
            return Ok(Outcome::new().set("kernels", json!(rows)).set("total_ms", json!(total)));
        }
        let (req, seed) = self.model_request(inv, "bench")?;
        let frames = inv.get_i64("frames").unwrap_or(200).max(1) as u64;
        let mut built = worldrun::build(&req).map_err(|e| format!("diamond bench: {e}"))?;
        let b = worldrun::bench(built.model.as_mut(), &name, frames);
        Ok(Outcome::new()
            .set("model", json!(name))
            .set("backend", json!(built.backend))
            .set("seed", json!(seed))
            .set("frames", json!(b.report.steps))
            .set("total_s", json!(b.total_s))
            .set("ms_per_frame_mean", json!(b.report.work_ms_mean))
            .set("ms_per_frame_p95", json!(b.report.work_ms_p95))
            .set("fps", json!(b.report.steps as f32 / b.total_s.max(1e-6))))
    }
}

/// Separates the dataset-sampling stream from the noise stream of one seed.
const DATASET_STREAM: u64 = 0xD5;

/// The action id of every step a rollout plays.
fn rollout_script(inv: &Invocation) -> Result<Vec<u32>, String> {
    let from_file = inv.get_str("actions").filter(|p| !p.is_empty());
    let from_seq = inv.get_str("action_seq").filter(|s| !s.is_empty());
    let script = match (from_file, from_seq) {
        (Some(_), Some(_)) => return Err("diamond rollout: give actions or action_seq, not both".into()),
        (Some(path), None) => Some(worldrun::parse_actions(&std::fs::read_to_string(&path).map_err(|e| format!("diamond rollout: {path}: {e}"))?).map_err(|e| format!("diamond rollout: {e}"))?),
        (None, Some(seq)) => Some(worldrun::parse_actions(&seq).map_err(|e| format!("diamond rollout: {e}"))?),
        (None, None) => None,
    };
    let frames = inv.get_i64("frames").unwrap_or(0).max(0) as usize;
    let steps = match script {
        Some(mut s) if frames > 0 => {
            s.truncate(frames);
            s
        }
        Some(s) => s,
        None => vec![0; frames.max(1)],
    };
    if steps.is_empty() {
        return Err("diamond rollout: the action script is empty".into());
    }
    if steps.len() as f64 > capability::bounds::VIDEO_FRAMES {
        return Err(format!("diamond rollout: {} steps exceed the {} a rollout returns as one clip", steps.len(), capability::bounds::VIDEO_FRAMES));
    }
    Ok(steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::Registry;

    fn run(action: &str, inv: Invocation) -> ActionResult {
        let mut reg = Registry::new();
        reg.register(Arc::new(DiamondProvider::new(None)));
        reg.run(MODEL, action, inv, &mut |_| {})
    }

    /// The error of a call that must be refused.
    fn refused(result: ActionResult) -> String {
        match result {
            Err(e) => e,
            Ok(_) => panic!("expected the call to be refused"),
        }
    }

    #[test]
    fn manifest_declares_every_verb_of_the_world_model() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["import", "export", "finetune", "rollout", "replay", "bench"]);
        let finetune = &m.actions[2];
        assert!(finetune.training && finetune.streaming && !finetune.from_scratch, "fine-tuning trains from a checkpoint");
        assert!(m.actions[0].from_scratch && m.actions[3].from_scratch, "import and the fake model need no checkpoint");
    }

    /// The fake model needs no weights, so the whole rollout path is
    /// checkable anywhere: same seed and script, same hash.
    #[test]
    fn a_fake_rollout_is_deterministic_and_returns_one_frame_per_step() {
        let play = |seed: i64| {
            let inv = Invocation::new().set("model", json!("fake")).set("seed", json!(seed)).set("action_seq", json!("1,2,0,4")).set("hashes", json!(true));
            run("rollout", inv).unwrap()
        };
        let (a, b) = (play(7), play(7));
        assert_eq!(a.outputs["rollout_hash"], b.outputs["rollout_hash"]);
        assert_eq!(a.outputs["steps"], json!(4));
        assert_eq!(a.outputs["frame_hashes"].as_array().unwrap().len(), 4);
        assert_eq!(a.blobs["clip"].meta["frames"], json!(4));
        assert_ne!(a.outputs["rollout_hash"], play(8).outputs["rollout_hash"], "the seed positions the fake world");
    }

    #[test]
    fn a_rollout_refuses_an_action_the_model_does_not_have() {
        let err = refused(run("rollout", Invocation::new().set("model", json!("fake")).set("action_seq", json!("1,99"))));
        assert!(err.contains("out of range"), "{err}");
    }

    #[test]
    fn the_diamond_model_needs_a_checkpoint_and_the_fake_one_refuses_the_npu() {
        let err = refused(run("rollout", Invocation::new()));
        assert!(err.contains("no DIAMOND checkpoint"), "{err}");
        let err = refused(run("bench", Invocation::new().set("model", json!("fake")).set("onnx", json!("/x.onnx"))));
        assert!(err.contains("NPU"), "{err}");
    }

    #[test]
    fn a_cancelled_rollout_reports_cancelled() {
        let mut inv = Invocation::new().set("model", json!("fake")).set("frames", json!(50));
        inv.cancel = capability::CancelToken::armed();
        inv.cancel.cancel();
        assert_eq!(refused(run("rollout", inv)), "cancelled");
    }

    /// A recording made by a rollout verifies against the same seed, and a
    /// different fake-world seed fails it.
    #[test]
    fn a_recorded_rollout_replays_and_verifies() {
        let dir = std::env::temp_dir().join(format!("diamond-replay-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let rec = dir.to_str().unwrap().to_string();
        let inv = Invocation::new().set("model", json!("fake")).set("seed", json!(3)).set("action_seq", json!("1,1,2,3,4,0")).set("record", json!(rec));
        assert_eq!(run("rollout", inv).unwrap().outputs["recorded_frames"], json!(6));
        let verify = |seed: i64| run("replay", Invocation::new().set("episode", json!(rec)).set("verify", json!(true)).set("model", json!("fake")).set("seed", json!(seed)));
        assert_eq!(verify(3).unwrap().outputs["verified_frames"], json!(6));
        assert!(refused(verify(4)).contains("FAIL"), "a different world fails verification");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bench_times_the_fake_model() {
        let out = run("bench", Invocation::new().set("model", json!("fake")).set("frames", json!(5))).unwrap();
        assert_eq!(out.outputs["frames"], json!(5));
    }
}
