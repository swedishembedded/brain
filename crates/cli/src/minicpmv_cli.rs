// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning of vision-language models for
// its clients. If your team needs expertise in adapting multimodal models
// to your images, video and conversations then you can procure our services
// by sending an email to info@swedishembedded.com.

//! `brain minicpmv finetune`: fine-tune MiniCPM-V from the checkpoint as
//! downloaded, each part (decoder, connector, vision tower) trained or frozen
//! on its own.
//!
//! ```text
//! brain minicpmv finetune --dataset train.jsonl [--held-out val.jsonl] \
//!     [--decoder lora:16:32] [--vision frozen|full|lora:R:A] [--freeze-connector] \
//!     [--epochs 3 --batch 1 --lr 1e-4 ...] [--label owner:name:tag | --out DIR]
//! ```
//!
//! The dataset is `generic-messages-v2` JSONL whose message content may hold
//! `{"type": "image"|"video", "path": ...}` parts (paths relative to the
//! file). Without `--out` the tune is written beside the checkpoint in the
//! model store, where serving offers it as `brain/minicpmv:<label>`.
//!
//! ```text
//! brain minicpmv export-peft --tune DIR|owner:name:tag --out DIR [--weights DIR|vendor/repo]
//! ```
//!
//! writes a tune as one PEFT adapter over the Hugging Face checkpoint
//! (`minicpmv::export`), for `transformers` + `peft`.
//!
//! Every other `minicpmv` verb goes through the generic capability dispatch.

use std::path::{Path, PathBuf};

use minicpmv::finetune::{Event, Job, Options};
use minicpmv::train::PartMode;
use minicpmv::Downsample;

pub fn run_minicpmv(args: &[String]) {
    let verb = args.first().map(String::as_str);
    let run: fn(&[String]) -> Result<(), String> = match verb {
        Some("finetune") => finetune,
        Some("export-peft") => export_peft,
        _ => std::process::exit(crate::resolver_cli::run_generic_migrated("minicpmv", minicpmv::caps::MODEL, args).expect("minicpmv is a resolver-migrated architecture")),
    };
    if let Err(e) = run(&args[1..]) {
        eprintln!("brain minicpmv {}: {e}", verb.unwrap_or_default());
        std::process::exit(1);
    }
}

pub(crate) const EXPORT_USAGE: &str = "usage: brain minicpmv export-peft --tune DIR|owner:name:tag --out DIR [--weights DIR|vendor/repo] [--models-dir DIR]";

/// `export-peft`'s flags: the tune (a directory, or the label of one stored
/// beside the checkpoint), the output directory, and the checkpoint.
fn export_args(args: &[String]) -> Result<(String, PathBuf, String, Option<String>), String> {
    let (mut tune, mut out, mut weights, mut models_dir) = (None, None, None, None);
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = || it.next().cloned().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--tune" => tune = Some(value()?),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--weights" => weights = Some(value()?),
            "--models-dir" => models_dir = Some(value()?),
            other => return Err(format!("unknown flag {other:?}\n{EXPORT_USAGE}")),
        }
    }
    Ok((
        tune.ok_or_else(|| format!("--tune is required\n{EXPORT_USAGE}"))?,
        out.ok_or_else(|| format!("--out is required\n{EXPORT_USAGE}"))?,
        weights.unwrap_or_else(|| DEFAULT_WEIGHTS.to_string()),
        models_dir,
    ))
}

fn export_peft(args: &[String]) -> Result<(), String> {
    let (tune, out, weights, models_dir) = export_args(args)?;
    let (dir, _) = crate::deepseekvl_cli::checkpoint_dir(&weights, models_dir.as_deref())?;
    let tune_dir = if Path::new(&tune).is_dir() { PathBuf::from(&tune) } else { minicpmv::finetune::store_dir(&dir, &tune)? };
    minicpmv::export::export_peft(&tune_dir, &dir, &out)?;
    println!("PEFT adapter of {} written to {}", tune_dir.display(), out.display());
    Ok(())
}

pub(crate) const USAGE: &str = "usage: brain minicpmv finetune --dataset FILE.jsonl [--held-out FILE.jsonl] [--weights DIR|vendor/repo] \
    [--decoder frozen|full|lora:R[:A]] [--vision frozen|full|lora:R[:A]] [--freeze-connector] [--epochs N] [--batch N] [--lr X] \
    [--connector-lr X] [--vision-lr X] [--min-lr-ratio R] [--warmup N] [--weight-decay W] [--grad-clip C] [--block T] [--downsample 16x|4x] [--max-slices N] [--max-frames N] \
    [--checkpoint-every N] [--resume] [--activations keep|recompute] [--label owner:name:tag] [--out DIR] [--device gpuN] [--models-dir DIR] [--seed S]";

const DEFAULT_WEIGHTS: &str = "openbmb/MiniCPM-V-4.6";
const DEFAULT_LABEL: &str = "local:minicpmv-tune:latest";

/// The flags `finetune` takes, parsed once.
pub struct Args {
    pub weights: String,
    pub dataset: PathBuf,
    pub held_out: Option<PathBuf>,
    pub out: Option<PathBuf>,
    pub label: String,
    pub device: Option<u32>,
    pub models_dir: Option<String>,
    pub options: Options,
}

impl Args {
    pub fn parse(args: &[String]) -> Result<Args, String> {
        let (seed, rest) = crate::args::split_seed(args, "minicpmv finetune");
        let mut o = Options::new(seed);
        let (mut weights, mut dataset, mut held_out, mut out, mut label, mut device, mut models_dir) = (None, None, None, None, None, None, None);
        let mut i = 0;
        while i < rest.len() {
            let flag = rest[i].as_str();
            let mut value = || {
                i += 1;
                rest.get(i).cloned().ok_or_else(|| format!("{flag} needs a value"))
            };
            let float = |v: String| v.parse::<f32>().map_err(|_| format!("{flag} {v:?} is not a number"));
            let whole = |v: String| v.parse::<u32>().map_err(|_| format!("{flag} {v:?} is not a whole number"));
            match flag {
                "--weights" => weights = Some(value()?),
                "--dataset" => dataset = Some(PathBuf::from(value()?)),
                "--held-out" => held_out = Some(PathBuf::from(value()?)),
                "--out" => out = Some(PathBuf::from(value()?)),
                "--label" => label = Some(value()?),
                "--models-dir" => models_dir = Some(value()?),
                "--device" => {
                    let v = value()?;
                    device = Some(v.strip_prefix("gpu").and_then(|n| n.parse().ok()).ok_or_else(|| format!("--device {v:?}: expected gpuN"))?);
                }
                "--decoder" => o.decoder = PartMode::parse(&value()?)?,
                "--vision" => o.vision = PartMode::parse(&value()?)?,
                "--freeze-connector" => o.connector = false,
                "--epochs" => o.epochs = whole(value()?)?,
                "--batch" => o.batch = whole(value()?)?,
                "--lr" => o.lr = float(value()?)?,
                "--connector-lr" => o.connector_lr = Some(float(value()?)?),
                "--vision-lr" => o.vision_lr = Some(float(value()?)?),
                "--min-lr-ratio" => o.min_lr_ratio = float(value()?)?,
                "--warmup" => o.warmup = Some(whole(value()?)?),
                "--weight-decay" => o.weight_decay = float(value()?)?,
                "--grad-clip" => o.grad_clip = Some(float(value()?)?).filter(|c| *c > 0.0),
                "--block" => o.block = Some(whole(value()?)?),
                "--downsample" => o.downsample = Some(Downsample::parse(&value()?)?),
                "--max-slices" => o.max_slices = Some(whole(value()?)?),
                "--max-frames" => o.max_frames = whole(value()?)?,
                "--checkpoint-every" => o.checkpoint_every = whole(value()?)?,
                "--resume" => o.resume = true,
                "--activations" => o.saved = Some(crate::args::parse_activations(&value()?)?),
                other => return Err(format!("unknown flag {other:?}\n{USAGE}")),
            }
            i += 1;
        }
        if o.epochs == 0 || o.batch == 0 {
            return Err("--epochs and --batch must be at least 1".into());
        }
        let label = label.unwrap_or_else(|| DEFAULT_LABEL.to_string());
        minicpmv::finetune::store_dir(Path::new(""), &label)?;
        Ok(Args {
            weights: weights.unwrap_or_else(|| DEFAULT_WEIGHTS.to_string()),
            dataset: dataset.ok_or_else(|| format!("--dataset is required\n{USAGE}"))?,
            held_out,
            out,
            label,
            device,
            models_dir,
            options: o,
        })
    }
}

fn finetune(args: &[String]) -> Result<(), String> {
    let a = Args::parse(args)?;
    let (dir, base_id) = crate::deepseekvl_cli::checkpoint_dir(&a.weights, a.models_dir.as_deref())?;
    let out = match &a.out {
        Some(o) => o.clone(),
        None => minicpmv::finetune::store_dir(&dir, &a.label)?,
    };
    let o = &a.options;
    println!(
        "minicpmv finetune: {base_id} on {} - decoder {}, connector {}, vision {}; {} epoch(s), batch {}, peak rates {:?}",
        a.dataset.display(),
        o.decoder.name(),
        if o.connector { "trained" } else { "frozen" },
        o.vision.name(),
        o.epochs,
        o.batch,
        o.rates()
    );
    let pipelines = minicpmv::vision_train::train_pipelines();
    let gpu = match a.device {
        Some(i) => gpu_core::Gpu::new_on_index(i, pipelines)?,
        None => gpu_core::Gpu::new(pipelines),
    };
    let job = Job { dir: &dir, base_id: &base_id, train: &a.dataset, held_out: a.held_out.as_deref(), out: &out, label: &a.label };
    let outcome = minicpmv::finetune::finetune(&gpu, job, o, &mut |e| match e {
        Event::Step { step, steps, epoch, loss, lr } => println!("step {step:>5}/{steps}  epoch {epoch}  loss {loss:.4}  lr {lr:.2e}"),
        Event::Eval { epoch, loss } => println!("held-out loss after epoch {epoch}: {loss:.4}"),
        Event::Checkpoint { step } => println!("state saved at step {step}"),
    })?;
    println!("trained {} step(s) over {} supervised turn(s) at {} tokens", outcome.steps, outcome.turns, outcome.block);
    if let (Some(b), Some(f)) = (outcome.initial_eval, outcome.final_eval) {
        println!("held-out loss {b:.4} -> {f:.4}");
    }
    println!("saved: {}", out.display());
    if a.out.is_none() {
        println!("serve it as {}:{}", minicpmv::caps::MODEL, a.label);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn the_flags_choose_each_parts_mode_and_refuse_what_is_malformed() {
        let a = Args::parse(&args(&["--dataset", "d.jsonl", "--seed", "1", "--vision", "lora:4", "--freeze-connector", "--grad-clip", "0", "--device", "gpu1"])).unwrap();
        let o = &a.options;
        assert_eq!((o.decoder, o.vision, o.connector, o.grad_clip), (PartMode::Lora { rank: 16, alpha: 32.0 }, PartMode::Lora { rank: 4, alpha: 8.0 }, false, None));
        assert_eq!((a.weights.as_str(), a.label.as_str(), a.device), (DEFAULT_WEIGHTS, DEFAULT_LABEL, Some(1)));
        let err = |l: &[&str]| Args::parse(&args(l)).err().expect("refused");
        assert!(err(&["--seed", "1"]).contains("--dataset"));
        assert!(err(&["--dataset", "d", "--seed", "1", "--decoder", "lora:0"]).contains("rank"));
        assert!(err(&["--dataset", "d", "--seed", "1", "--label", "a:b"]).contains("owner:name:tag"));
        assert!(err(&["--dataset", "d", "--seed", "1", "--device", "1"]).contains("gpuN"));
        assert!(err(&["--dataset", "d", "--seed", "1", "--batch", "0"]).contains("--batch"));
        let r = Args::parse(&args(&["--dataset", "d", "--seed", "1", "--vision", "full", "--connector-lr", "3e-5"])).unwrap().options.rates();
        assert_eq!((r.decoder, r.connector, r.vision), (1e-4, 3e-5, 1e-4 / 50.0), "a full tower defaults to a fiftieth of the LoRA rate");
    }

    #[test]
    fn activations_default_to_each_parts_own_and_refuse_a_bogus_value_naming_the_flag() {
        let with = |extra: &[&str]| Args::parse(&args(&[&["--dataset", "d", "--seed", "1"][..], extra].concat()));
        assert_eq!(with(&[]).unwrap().options.saved, None);
        assert_eq!(with(&["--activations", "recompute"]).unwrap().options.saved, Some(model::saved::SavedPolicy::Recompute));
        assert!(with(&["--activations", "bogus"]).err().unwrap().contains("--activations"));
    }

    #[test]
    fn an_export_needs_its_tune_and_output() {
        let (tune, out, weights, _) = export_args(&args(&["--tune", "me:colors:v1", "--out", "o"])).unwrap();
        assert_eq!((tune.as_str(), out, weights.as_str()), ("me:colors:v1", PathBuf::from("o"), DEFAULT_WEIGHTS));
        assert!(export_args(&args(&["--out", "o"])).unwrap_err().contains("--tune"));
        assert!(export_args(&args(&["--tune", "t"])).unwrap_err().contains("--out"));
        assert!(export_args(&args(&["--tune"])).unwrap_err().contains("needs a value"));
    }
}
