// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning of vision-language models for
// its clients. If your team needs expertise in adapting multimodal models
// to your images and conversations then you can procure our services by
// sending an email to info@swedishembedded.com.

//! `brain deepseekvl finetune`: a LoRA fine-tune of DeepSeek-VL on a dataset
//! of images and replies, from the checkpoint as downloaded.
//!
//! ```text
//! brain deepseekvl finetune --weights deepseek-ai/deepseek-vl-7b-chat \
//!     --dataset DIR --out DIR [--rank 16 --alpha 32 --steps 200 --lr 1e-4 ...]
//! ```
//!
//! `DIR/train.jsonl` holds one example a line: `{"image": "cat.png",
//! "messages": [{"role": "user", "content": "<image_placeholder>What is
//! this?"}, {"role": "assistant", "content": "A cat."}]}`, the image path
//! relative to `DIR`. The towers stay frozen (their features are extracted
//! once and the towers released), the aligner trains from the checkpoint's own
//! weights, and the decoder trains as a LoRA over its frozen base. The adapter
//! and the aligner are written to `--out`; nothing is written beside the
//! checkpoint.
//!
//! Every other `deepseekvl` verb goes through the generic capability dispatch.

use std::path::Path;

use deepseekvl::train::{Options, StepInfo};

pub fn run_deepseekvl(args: &[String]) {
    if args.first().map(String::as_str) != Some("finetune") {
        std::process::exit(
            crate::resolver_cli::run_generic_migrated("deepseekvl", deepseekvl::caps::MODEL, args).expect("deepseekvl is a resolver-migrated architecture"),
        );
    }
    if let Err(e) = finetune(&args[1..]) {
        eprintln!("brain deepseekvl finetune: {e}");
        std::process::exit(1);
    }
}

pub(crate) const USAGE: &str = "usage: brain deepseekvl finetune --weights DIR|vendor/repo --dataset DIR --out DIR \
    [--rank N] [--alpha A] [--lora-targets wq,wk,...] [--steps N] [--lr X] [--aligner-lr X] [--block T] [--batch N] [--seed S] \
    [--base-dtype f32|bf16] [--weight-decay W] [--grad-clip C] [--warmup N] [--min-lr X] [--models-dir DIR]";

/// The flags `finetune` takes, parsed once.
pub struct Args {
    pub weights: String,
    pub dataset: String,
    pub out: String,
    pub models_dir: Option<String>,
    pub options: Options,
}

impl Args {
    pub fn parse(args: &[String]) -> Result<Args, String> {
        let (mut weights, mut dataset, mut out, mut models_dir) = (None, None, None, None);
        let mut o = Options {
            rank: 16,
            alpha: 0.0,
            targets: qwen3::finetune::default_lora_targets(),
            steps: 200,
            lr: 1e-4,
            aligner_lr: None,
            seed: 0,
            block: None,
            batch: 1,
            dtype: qwen3::Dtype::BF16,
            hyper: qwen3::finetune::LoraHyper::default(),
        };
        let mut alpha = None;
        let mut seed = None;
        let mut i = 0;
        while i < args.len() {
            let flag = args[i].as_str();
            let mut value = || {
                i += 1;
                args.get(i).cloned().ok_or_else(|| format!("{flag} needs a value"))
            };
            let number = |v: String| v.parse::<f32>().map_err(|_| format!("{flag} {v:?} is not a number"));
            match flag {
                "--weights" => weights = Some(value()?),
                "--dataset" => dataset = Some(value()?),
                "--out" => out = Some(value()?),
                "--models-dir" => models_dir = Some(value()?),
                "--rank" => o.rank = value()?.parse().map_err(|_| "--rank is not a whole number".to_string())?,
                "--alpha" => alpha = Some(number(value()?)?),
                "--lora-targets" => o.targets = qwen3::finetune::parse_lora_targets(&value()?)?,
                "--steps" => o.steps = value()?.parse().map_err(|_| "--steps is not a whole number".to_string())?,
                "--lr" => o.lr = number(value()?)?,
                "--aligner-lr" => o.aligner_lr = Some(number(value()?)?),
                "--batch" => o.batch = value()?.parse().ok().filter(|&b| b > 0).ok_or_else(|| "--batch is not a positive whole number".to_string())?,
                "--block" => o.block = Some(value()?.parse().map_err(|_| "--block is not a whole number".to_string())?),
                "--seed" => seed = Some(value()?.parse::<u64>().map_err(|_| "--seed is not a whole number".to_string())?),
                "--base-dtype" => {
                    o.dtype = match value()?.as_str() {
                        "f32" => qwen3::Dtype::F32,
                        "bf16" => qwen3::Dtype::BF16,
                        other => return Err(format!("--base-dtype {other:?}: expected f32 or bf16")),
                    }
                }
                "--weight-decay" => o.hyper.weight_decay = number(value()?)?,
                "--grad-clip" => o.hyper.grad_clip = number(value()?)?,
                "--warmup" => o.hyper.warmup = Some(number(value()?)? as u32),
                "--min-lr" => o.hyper.min_lr = Some(number(value()?)?),
                other => return Err(format!("unknown flag {other:?}\n{USAGE}")),
            }
            i += 1;
        }
        let need = |v: Option<String>, flag: &str| v.ok_or_else(|| format!("{flag} is required\n{USAGE}"));
        if o.rank == 0 {
            return Err("--rank must be > 0".to_string());
        }
        o.alpha = alpha.unwrap_or(o.rank as f32 * 2.0);
        o.seed = seed.unwrap_or_else(|| {
            let s = data::rng::random_seed();
            println!("no --seed given, using random seed {s} (pass --seed {s} to reproduce)");
            s
        });
        Ok(Args { weights: need(weights, "--weights")?, dataset: need(dataset, "--dataset")?, out: need(out, "--out")?, models_dir, options: o })
    }
}

/// The checkpoint directory `weights` names, and its `vendor/repo` id: the
/// directory itself, or a `vendor/repo` reference's directory in the model
/// store.
pub(crate) fn checkpoint_dir(weights: &str, models_dir: Option<&str>) -> Result<(std::path::PathBuf, String), String> {
    let path = Path::new(weights);
    if path.is_dir() {
        let parts: Vec<_> = path.components().rev().take(2).map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
        return match parts.as_slice() {
            [repo, vendor] => Ok((path.to_path_buf(), format!("{vendor}/{repo}"))),
            _ => Ok((path.to_path_buf(), format!("local/{weights}"))),
        };
    }
    let reference = brain_modelref::ModelRef::parse(weights).map_err(|e| format!("{weights}: not a directory, and not a model reference ({e})"))?;
    let root = loader::model_dir::resolve(models_dir).ok_or("no models directory resolved (set --models-dir, BRAIN_MODELS_DIR, or HOME)")?;
    let dir = brain_modelstore::Store::new(&root).repo_dir(&reference);
    if !dir.is_dir() {
        return Err(format!("{weights}: not found in the model store at {}", root.display()));
    }
    Ok((dir, reference.to_string()))
}

fn finetune(args: &[String]) -> Result<(), String> {
    let a = Args::parse(args)?;
    let (dir, base_id) = checkpoint_dir(&a.weights, a.models_dir.as_deref())?;
    println!("deepseekvl finetune: {base_id}, rank {} on {}, {} steps", a.options.rank, a.dataset, a.options.steps);
    let mut progress = |s: &StepInfo| println!("step {:>5}/{}  loss {:.4}  lr {:.2e}", s.step, s.steps, s.loss, s.lr);
    let outcome = deepseekvl::train::finetune(&dir, Path::new(&a.dataset), &a.options, &mut progress)?;
    println!(
        "trained on {} example(s) at {} tokens: loss {:.4} -> {}",
        outcome.examples,
        outcome.block,
        outcome.initial_loss,
        outcome.final_loss.map_or("n/a".to_string(), |l| format!("{l:.4}"))
    );
    outcome.save(Path::new(&a.out), &format!("{base_id}:local:vl-adapter:latest"), &base_id)?;
    println!("saved: {}", a.out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_fine_tune_needs_its_checkpoint_dataset_and_output() {
        let err = |a: &[&str]| Args::parse(&args(a)).err().expect("refused");
        assert!(err(&["--dataset", "d", "--out", "o"]).contains("--weights"));
        assert!(err(&["--weights", "w", "--out", "o"]).contains("--dataset"));
        assert!(err(&["--weights", "w", "--dataset", "d"]).contains("--out"));
        assert!(err(&["--weights", "w", "--dataset", "d", "--out", "o", "--rank", "0"]).contains("--rank"));
        assert!(err(&["--weights", "w", "--dataset", "d", "--out", "o", "--base-dtype", "int8"]).contains("f32 or bf16"));
        assert!(err(&["--weights", "w", "--dataset", "d", "--out", "o", "--steps"]).contains("needs a value"));
    }

    #[test]
    fn the_defaults_are_the_lora_recipe_at_bf16() {
        let a = Args::parse(&args(&["--weights", "w", "--dataset", "d", "--out", "o", "--seed", "3", "--rank", "8", "--lr", "2e-4"])).unwrap();
        let o = a.options;
        assert_eq!((o.rank, o.alpha, o.seed, o.dtype), (8, 16.0, 3, qwen3::Dtype::BF16), "alpha defaults to twice the rank");
        assert_eq!(o.lr, 2e-4);
        assert_eq!(o.targets, qwen3::finetune::default_lora_targets());
    }

    #[test]
    fn a_batch_is_a_positive_whole_number() {
        let base = ["--weights", "w", "--dataset", "d", "--out", "o", "--seed", "1"];
        let with = |extra: &[&str]| Args::parse(&args(&[&base[..], extra].concat()));
        assert_eq!(with(&[]).unwrap().options.batch, 1);
        assert_eq!(with(&["--batch", "4"]).unwrap().options.batch, 4);
        assert!(with(&["--batch", "0"]).err().unwrap().contains("--batch"));
    }
}
