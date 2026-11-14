// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniCPM-V's training actions: `finetune` (each part - decoder, connector,
//! vision tower - trained or frozen on its own, from the checkpoint as
//! downloaded) and `export_peft` (a tune as one PEFT adapter over the Hugging
//! Face checkpoint, for `transformers` + `peft`).
//!
//! Both read and write files on the host (a JSONL dataset whose message parts
//! name images and videos; the tune directory), so they are local actions:
//! the scheduled service serves `generate`.
//!
//! Swedish Embedded AB implements fine-tuning of vision-language models on
//! its clients' images, video and conversations. If your team needs
//! expertise in adapting multimodal models then you can procure our services
//! by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use capability::{Action, ActionResult, ActionSpec, Invocation, Outcome, ParamSpec, ParamType, Progress};
use serde_json::json;

use crate::config::Downsample;
use crate::finetune::{store_dir, Event, Job, Options};
use crate::train::PartMode;

/// The label a tune is kept under beside the checkpoint when none is named.
pub const DEFAULT_LABEL: &str = "local:minicpmv-tune:latest";

/// Most supervised turns one optimiser step may hold.
const MAX_BATCH: f64 = 1_024.0;
/// Largest gradient-norm clip a run may ask for.
const MAX_CLIP: f64 = 1_000_000.0;
/// Most image slices a run may cut one picture into.
const MAX_SLICES: f64 = 1_024.0;

fn part_mode_param(name: &str, what: &str, default: &str) -> ParamSpec {
    ParamSpec::new(name, ParamType::Str, &format!("{what}: frozen, full, or lora:RANK[:ALPHA] (alpha defaults to 2 x rank)")).default(json!(default))
}

pub fn finetune_spec() -> ActionSpec {
    let o = Options::new(0);
    ActionSpec::new("finetune", "fine-tune MiniCPM-V on conversations with images and video, each part trained or frozen on its own; by default a LoRA on the decoder with the connector trained and the tower frozen")
        .training()
        .streaming()
        .cancellable()
        .param(ParamSpec::new("dataset", ParamType::Str, "the generic-messages-v2 JSONL training file; a message part {\"type\": \"image\"|\"video\", \"path\": ...} names a file relative to it").required().host_resolved())
        .param(ParamSpec::new("held_out", ParamType::Str, "a JSONL file of the same shape, scored before training and after every epoch").host_resolved())
        .param(ParamSpec::new("save", ParamType::Str, "the tune directory; omit to keep the tune beside the checkpoint in the model store, where serving offers it under its label").host_resolved())
        .param(ParamSpec::new("label", ParamType::Str, "owner:name:tag the tune is recorded (and, without save, stored) under").default(json!(DEFAULT_LABEL)))
        .param(part_mode_param("decoder", "how the decoder trains", &o.decoder.name()))
        .param(part_mode_param("vision", "how the vision tower trains", &o.vision.name()))
        .param(ParamSpec::new("connector", ParamType::Bool, "train the connector between tower and decoder").default(json!(o.connector)))
        .param(ParamSpec::new("epochs", ParamType::Int, "passes over the dataset").default(json!(o.epochs)).min(1.0).max(capability::bounds::TRAIN_COUNT))
        .param(ParamSpec::new("batch", ParamType::Int, "supervised turns per optimiser step").default(json!(o.batch)).min(1.0).max(MAX_BATCH))
        .param(ParamSpec::new("lr", ParamType::Float, "peak rate of the decoder (and of a LoRA tower)").default(json!(o.lr)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("connector_lr", ParamType::Float, "the connector's peak rate; omit to derive it from lr").min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("vision_lr", ParamType::Float, "the tower's peak rate; omit to derive it from lr (a fully trained tower gets a fiftieth)").min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("min_lr_ratio", ParamType::Float, "where the cosine ends, as a fraction of each peak").default(json!(o.min_lr_ratio)).min(0.0).max(capability::bounds::UNIT))
        // perf-number: the default warmup fraction is a training hyperparameter
        .param(ParamSpec::new("warmup", ParamType::Int, "warmup steps; omit for 5% of the run").min(0.0).max(capability::bounds::TRAIN_COUNT))
        .param(ParamSpec::new("weight_decay", ParamType::Float, "AdamW decoupled weight decay").default(json!(o.weight_decay)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("grad_clip", ParamType::Float, "global gradient-norm clip; 0 disables it").default(json!(o.grad_clip.unwrap_or(0.0))).min(0.0).max(MAX_CLIP))
        .param(ParamSpec::new("block", ParamType::Int, "tokens per sample; omit to fit the longest one").min(1.0).max(capability::bounds::CONTEXT_TOKENS))
        .param(ParamSpec::new("downsample", ParamType::Enum(vec!["16x".into(), "4x".into()]), "the tower's token reduction; omit for the checkpoint's own"))
        .param(ParamSpec::new("max_slices", ParamType::Int, "most slices a high-resolution image is cut into; omit for the checkpoint's own").min(1.0).max(MAX_SLICES))
        .param(ParamSpec::new("max_frames", ParamType::Int, "most frames read from a video").default(json!(o.max_frames)).min(1.0).max(capability::bounds::VIDEO_FRAMES))
        .param(ParamSpec::new("checkpoint_every", ParamType::Int, "steps between training-state saves; 0 saves only at epoch ends").default(json!(o.checkpoint_every)).min(0.0).max(capability::bounds::TRAIN_STEPS))
        .param(ParamSpec::new("resume", ParamType::Bool, "continue from the tune directory's saved state when one is there").default(json!(o.resume)))
        .param(ParamSpec::new("seed", ParamType::Int, "seeds the order and any fresh adapter (omit for random; the outcome echoes it)").min(0.0))
        .param(ParamSpec::new(
            "activations",
            ParamType::Enum(model::saved::SavedPolicy::ACTIVATIONS_CHOICES.map(String::from).to_vec()),
            "what the backward keeps of the forward in the parts that train; a choice a part cannot run fails before the first step (default: each part's own)",
        ))
}

pub fn export_peft_spec() -> ActionSpec {
    ActionSpec::new("export_peft", "write a tune as one PEFT adapter over the Hugging Face checkpoint, for transformers + peft")
        .cancellable()
        .param(ParamSpec::new("tune", ParamType::Str, "the tune: its directory, or the owner:name:tag label of one kept beside the checkpoint").required().host_resolved())
        .param(ParamSpec::new("save", ParamType::Str, "the directory the PEFT adapter is written to").required().host_resolved())
}

/// The run `inv` describes.
pub fn options(inv: &Invocation) -> Result<Options, String> {
    let seed = inv.seed()?.unwrap_or_else(data::rng::random_seed);
    let mut o = Options::new(seed);
    if let Some(m) = inv.get_str("decoder") {
        o.decoder = PartMode::parse(&m)?;
    }
    if let Some(m) = inv.get_str("vision") {
        o.vision = PartMode::parse(&m)?;
    }
    let f = |name: &str| inv.get_f64(name).map(|v| v as f32);
    let n = |name: &str| inv.get_i64(name).map(|v| v.max(0) as u32);
    o.connector = inv.get_bool("connector").unwrap_or(o.connector);
    o.epochs = n("epochs").unwrap_or(o.epochs).max(1);
    o.batch = n("batch").unwrap_or(o.batch).max(1);
    o.lr = f("lr").unwrap_or(o.lr);
    o.connector_lr = f("connector_lr");
    o.vision_lr = f("vision_lr");
    o.min_lr_ratio = f("min_lr_ratio").unwrap_or(o.min_lr_ratio);
    o.warmup = n("warmup");
    o.weight_decay = f("weight_decay").unwrap_or(o.weight_decay);
    if let Some(c) = f("grad_clip") {
        o.grad_clip = Some(c).filter(|c| *c > 0.0);
    }
    o.block = n("block");
    o.downsample = inv.get_str("downsample").map(|d| Downsample::parse(&d)).transpose()?;
    o.max_slices = n("max_slices");
    o.max_frames = n("max_frames").unwrap_or(o.max_frames);
    o.checkpoint_every = n("checkpoint_every").unwrap_or(o.checkpoint_every);
    o.resume = inv.get_bool("resume").unwrap_or(o.resume);
    o.saved = inv.get_str("activations").map(|a| model::saved::SavedPolicy::parse_activations(&a)).transpose()?;
    Ok(o)
}

fn checkpoint(dir: &Option<String>, action: &str) -> Result<PathBuf, String> {
    dir.clone().map(PathBuf::from).ok_or_else(|| format!("minicpmv {action}: no MiniCPM-V checkpoint (pull openbmb/MiniCPM-V-4.6, or name its directory with --dir)"))
}

pub(crate) struct FinetuneAction {
    pub dir: Option<String>,
}

impl Action for FinetuneAction {
    fn spec(&self) -> ActionSpec {
        finetune_spec()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let dir = checkpoint(&self.dir, "finetune")?;
        let o = options(inv)?;
        let label = inv.get_str("label").unwrap_or_else(|| DEFAULT_LABEL.to_string());
        let stored = store_dir(&dir, &label)?;
        let save = inv.get_str("save").map_or(stored, PathBuf::from);
        let dataset = PathBuf::from(inv.get_str("dataset").ok_or("minicpmv finetune: missing required param 'dataset'")?);
        let held_out = inv.get_str("held_out").map(PathBuf::from);
        let base_id = brain_modelstore::repo_id(&dir);
        let gpu = gpu_core::Gpu::new(crate::vision_train::train_pipelines());
        let job = Job { dir: &dir, base_id: &base_id, train: &dataset, held_out: held_out.as_deref(), out: &save, label: &label };
        let mut events = Vec::new();
        let outcome = crate::finetune::finetune(
            &gpu,
            job,
            &o,
            &mut |e| match e {
                Event::Step { step, steps, epoch, loss, lr } => progress(Progress::step(*step, *steps, format!("epoch {epoch}  loss {loss:.4}  lr {lr:.2e}"))),
                Event::Eval { epoch, loss } => events.push(json!({"held_out_loss": loss, "after_epoch": epoch})),
                Event::Checkpoint { step } => events.push(json!({"state_saved_at_step": step})),
            },
            &|| inv.cancel.is_cancelled(),
        )?;
        Ok(Outcome::new()
            .set("steps", json!(outcome.steps))
            .set("turns", json!(outcome.turns))
            .set("block", json!(outcome.block))
            .set("initial_eval", json!(outcome.initial_eval))
            .set("final_eval", json!(outcome.final_eval))
            .set("events", json!(events))
            .set("seed", json!(o.seed))
            .set("label", json!(label))
            .set("saved", json!(save.display().to_string())))
    }
}

pub(crate) struct ExportPeftAction {
    pub dir: Option<String>,
}

impl Action for ExportPeftAction {
    fn spec(&self) -> ActionSpec {
        export_peft_spec()
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let dir = checkpoint(&self.dir, "export_peft")?;
        let tune = inv.get_str("tune").ok_or("minicpmv export_peft: missing required param 'tune'")?;
        let save = PathBuf::from(inv.get_str("save").ok_or("minicpmv export_peft: missing required param 'save'")?);
        let tune_dir = if Path::new(&tune).is_dir() { PathBuf::from(&tune) } else { store_dir(&dir, &tune)? };
        crate::export::export_peft(&tune_dir, &dir, &save)?;
        Ok(Outcome::new().set("tune", json!(tune_dir.display().to_string())).set("saved", json!(save.display().to_string())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each part's mode, the schedule and the store label reach the run; a
    /// malformed one is refused before anything loads.
    #[test]
    fn the_parameters_choose_each_parts_mode_and_refuse_what_is_malformed() {
        let spec = finetune_spec();
        let run = |inv: Invocation| options(&spec.validate(inv.set("dataset", json!("d.jsonl")).set("seed", json!(1))).unwrap());
        let o = run(Invocation::new().set("vision", json!("lora:4")).set("connector", json!(false)).set("grad_clip", json!(0.0))).unwrap();
        assert_eq!((o.decoder, o.vision, o.connector, o.grad_clip), (PartMode::Lora { rank: 16, alpha: 32.0 }, PartMode::Lora { rank: 4, alpha: 8.0 }, false, None));
        assert!(run(Invocation::new().set("decoder", json!("lora:0"))).unwrap_err().contains("rank"));
        let r = run(Invocation::new().set("vision", json!("full")).set("connector_lr", json!(3e-5))).unwrap().rates();
        assert_eq!((r.decoder, r.connector, r.vision), (1e-4, 3e-5, 1e-4 / 50.0), "a full tower defaults to a fiftieth of the LoRA rate");
        assert!(spec.validate(Invocation::new().set("dataset", json!("d")).set("batch", json!(0))).is_err());
        assert!(spec.validate(Invocation::new().set("dataset", json!("d")).set("downsample", json!("8x"))).is_err());
        assert!(store_dir(Path::new(""), "a:b").unwrap_err().contains("owner:name:tag"));
        assert!(export_peft_spec().validate(Invocation::new().set("save", json!("o"))).is_err(), "the tune is required");
    }
}
