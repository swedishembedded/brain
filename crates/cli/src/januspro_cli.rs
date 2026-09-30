// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning of generative multimodal models
// for its clients. If your team needs expertise in teaching a model to draw
// your domain then you can procure our services by sending an email to
// info@swedishembedded.com.

//! `brain januspro finetune`: a LoRA fine-tune of Janus-Pro-7B from the
//! checkpoint as downloaded, in one of its two modes.
//!
//! ```text
//! brain januspro finetune --mode understanding --weights DIR --dataset DIR --out DIR [...]
//! brain januspro finetune --mode generation     --weights DIR --dataset DIR --out DIR [--cfg-dropout 0.1 ...]
//! ```
//!
//! **understanding** trains as `brain deepseekvl finetune` does (the same
//! dataset: an image and the reply to learn per line of `train.jsonl`).
//!
//! **generation** teaches the decoder to draw: each line of `train.jsonl` is
//! a `prompt` and the `image` (relative to `DIR`) to draw for it. The images
//! are encoded once by the frozen VQ-16 (which is then released) and the
//! decoder's adapter, the generation head, the generation aligner and the
//! code embedding train on the codes, with `--cfg-dropout` of the steps run
//! on a padded prompt so the unconditional branch guidance contrasts against
//! learns too. `--out` receives `adapter.safetensors` and
//! `generation.safetensors`.
//!
//! Every other `januspro` verb goes through the generic capability dispatch.

use std::path::Path;

use januspro::train::GenOptions;

use crate::deepseekvl_cli::{checkpoint_dir, Args, USAGE};

pub fn run_januspro(args: &[String]) {
    if args.first().map(String::as_str) != Some("finetune") {
        std::process::exit(crate::resolver_cli::run_generic_migrated("januspro", januspro::caps::MODEL, args).expect("januspro is a resolver-migrated architecture"));
    }
    if let Err(e) = finetune(&args[1..]) {
        eprintln!("brain januspro finetune: {e}");
        std::process::exit(1);
    }
}

/// What the fine-tune trains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Understanding,
    Generation,
}

/// Pull `--flag VALUE` out of `args`, returning the value.
fn take(args: &mut Vec<String>, flag: &str) -> Result<Option<String>, String> {
    let Some(at) = args.iter().position(|a| a == flag) else { return Ok(None) };
    if at + 1 >= args.len() {
        return Err(format!("{flag} needs a value"));
    }
    let value = args.remove(at + 1);
    args.remove(at);
    Ok(Some(value))
}

/// `--mode` and `--cfg-dropout`, and the flags the rest of the verb shares
/// with `deepseekvl finetune`.
fn parse(args: &[String]) -> Result<(Mode, f32, Args), String> {
    let mut rest = args.to_vec();
    let mode = match take(&mut rest, "--mode")?.as_deref() {
        Some("understanding") => Mode::Understanding,
        Some("generation") => Mode::Generation,
        Some(other) => return Err(format!("--mode {other:?}: expected understanding or generation")),
        None => return Err(format!("--mode understanding|generation is required\n{USAGE}")),
    };
    let dropout = match take(&mut rest, "--cfg-dropout")? {
        Some(v) => v.parse::<f32>().ok().filter(|p| (0.0..=1.0).contains(p)).ok_or_else(|| format!("--cfg-dropout {v:?} is not a share between 0 and 1"))?,
        None => 0.1,
    };
    Ok((mode, dropout, Args::parse(&rest)?))
}

fn finetune(args: &[String]) -> Result<(), String> {
    let (mode, cfg_dropout, a) = parse(args)?;
    let (dir, base_id) = checkpoint_dir(&a.weights, a.models_dir.as_deref())?;
    println!("januspro finetune ({mode:?}): {base_id}, rank {} on {}, {} steps", a.options.rank, a.dataset, a.options.steps);
    let mut progress = |s: &deepseekvl::train::StepInfo| println!("step {:>5}/{}  loss {:.4}  lr {:.2e}", s.step, s.steps, s.loss, s.lr);
    let card = format!("{base_id}:local:janus-adapter:latest");
    match mode {
        Mode::Understanding => {
            let outcome = januspro::train::finetune_understanding(&dir, Path::new(&a.dataset), &a.options, &mut progress)?;
            report(outcome.examples, outcome.block, outcome.initial_loss, outcome.final_loss);
            outcome.save(Path::new(&a.out), &card, &base_id)?;
        }
        Mode::Generation => {
            let outcome = januspro::train::finetune_generation(&dir, Path::new(&a.dataset), &GenOptions { options: a.options.clone(), cfg_dropout }, &mut progress)?;
            report(outcome.examples, outcome.block, outcome.initial_loss, outcome.final_loss);
            outcome.save(Path::new(&a.out), &card, &base_id)?;
        }
    }
    println!("saved: {}", a.out);
    Ok(())
}

fn report(examples: usize, block: u32, initial: f32, last: Option<f32>) {
    println!("trained on {examples} example(s) at {block} tokens: loss {initial:.4} -> {}", last.map_or("n/a".to_string(), |l| format!("{l:.4}")));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_fine_tune_names_its_mode_and_takes_the_shared_flags() {
        let base = ["--weights", "w", "--dataset", "d", "--out", "o", "--seed", "1"];
        let with = |extra: &[&str]| parse(&args(&[&base[..], extra].concat()));
        assert!(with(&[]).err().unwrap().contains("--mode"));
        assert!(with(&["--mode", "both"]).err().unwrap().contains("understanding or generation"));
        let (mode, dropout, a) = with(&["--mode", "generation", "--cfg-dropout", "0.25", "--rank", "8"]).unwrap();
        assert_eq!((mode, dropout, a.options.rank), (Mode::Generation, 0.25, 8));
        let (mode, dropout, _) = with(&["--mode", "understanding"]).unwrap();
        assert_eq!((mode, dropout), (Mode::Understanding, 0.1), "guidance dropout defaults to a tenth");
        assert!(with(&["--mode", "generation", "--cfg-dropout", "2"]).err().unwrap().contains("between 0 and 1"));
    }
}
