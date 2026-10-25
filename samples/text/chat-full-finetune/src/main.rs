// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Full-parameter fine-tuning of a chat model with
//! [`brain::ChatFullFineTune`]: every weight of the base trains on a chat
//! dataset (one JSON object per line, `{"messages": [{"role": ..., "content":
//! ..., "train": true|false}, ...]}`, `train` marking the supervised turns) and the result is a checkpoint directory that is a base like
//! any other - `brain serve` serves it and another fine-tune continues it.
//!
//! The run's weights and gradients live on the card and the AdamW moments
//! and master weights in host memory, so the sample reports what a model of
//! a given size costs on both, beside the losses: run it under
//! `/usr/bin/time -v` and next to `nvidia-smi --query-gpu=memory.used -lms
//! 500` to read the peaks.
//!
//! ```text
//! sample-text-chat-full-finetune --base Qwen/Qwen3-1.7B --dataset train.jsonl \
//!                                --out runs/full-1 --steps 50 --device gpu0
//! ```
//!
//! Swedish Embedded AB implements full-parameter fine-tuning of open-weight
//! chat models on a client's own data, sized to the cards they have. If your
//! team needs a model whose every weight learned your domain, you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::process::ExitCode;
use std::time::Instant;

use brain::{ChatFullFineTune, Device, TrainPrecision};

const USAGE: &str = "\
usage: sample-text-chat-full-finetune --base REF --dataset FILE --out DIR
                                      [--steps N] [--lr X] [--max-block T]
                                      [--grad-accum N] [--seed S]
                                      [--precision SPEC] [--device SPEC]
                                      [--models-dir DIR]

  --base REF         a checkpoint file or directory, or a vendor/repo name
                      in the model store (e.g. Qwen/Qwen3-1.7B)
  --dataset FILE     chat records, one {\"messages\": [...]} object per line,
                      each message with \"train\": true|false
  --out DIR          where the run writes its checkpoint and record
  --steps N          optimizer steps (default 50)
  --lr X             peak learning rate (default 1e-5)
  --max-block T      longest record in tokens; longer ones are refused
  --grad-accum N     records averaged into one step (default 1)
  --seed S           batch-order seed (drawn and printed when not given)
  --precision SPEC   training precision, e.g. state=int8,sr=on (default fp32)
  --device SPEC      gpu0, gpu1, cpu, ... (default: every device)
  --models-dir DIR   the model store a vendor/repo base is looked up in
";

/// A tiny flag reader - see `samples/study/document/src/main.rs` for the
/// same pattern; a sample parses its own arguments rather than sharing the
/// engine's parser, and its only brain dependency is the SDK.
struct Args(Vec<String>);

impl Args {
    fn take(&mut self, flag: &str) -> Option<String> {
        let i = self.0.iter().position(|a| a == flag)?;
        if i + 1 >= self.0.len() {
            return None;
        }
        self.0.remove(i);
        Some(self.0.remove(i))
    }
    fn parse<T: std::str::FromStr>(&mut self, flag: &str) -> Result<Option<T>, String> {
        self.take(flag)
            .map(|v| {
                v.parse()
                    .map_err(|_| format!("{flag} {v:?} is not a valid value"))
            })
            .transpose()
    }
}

fn run() -> Result<(), String> {
    let mut args = Args(std::env::args().skip(1).collect());
    let (Some(base), Some(dataset), Some(out)) = (
        args.take("--base"),
        args.take("--dataset"),
        args.take("--out"),
    ) else {
        return Err(USAGE.to_string());
    };
    let mut tune = ChatFullFineTune::from_pretrained(base.clone())
        .dataset(dataset)
        .out_dir(out)
        .steps(args.parse("--steps")?.unwrap_or(50))
        .lr(args.parse("--lr")?.unwrap_or(1e-5))
        .grad_accum(args.parse("--grad-accum")?.unwrap_or(1));
    if let Some(block) = args.parse("--max-block")? {
        tune = tune.max_block(block);
    }
    if let Some(seed) = args.parse("--seed")? {
        tune = tune.seed(seed);
    }
    if let Some(spec) = args.take("--precision") {
        tune = tune.train_precision(
            TrainPrecision::parse(&spec).map_err(|e| format!("--precision {spec:?}: {e}"))?,
        );
    }
    if let Some(spec) = args.take("--device") {
        tune = tune.device(Device::parse(&spec).map_err(|e| format!("--device {spec:?}: {e}"))?);
    }
    if let Some(dir) = args.take("--models-dir") {
        tune = tune.models_dir(dir);
    }
    if let Some(unknown) = args.0.first() {
        return Err(format!("unknown argument {unknown:?}\n\n{USAGE}"));
    }
    let started = Instant::now();
    let outcome = tune.run().map_err(|e| format!("{base}: {e}"))?;
    println!("status: {:?}", outcome.status);
    println!(
        "records: {} trained, {} replayed; block {} tokens; seed {}",
        outcome.train_records, outcome.replay_records, outcome.block, outcome.seed
    );
    println!(
        "steps: {} of {}; loss {:?} -> {:?}",
        outcome.steps_completed, outcome.steps, outcome.initial_loss, outcome.final_loss
    );
    println!("wall: {:.1} s", started.elapsed().as_secs_f64());
    if let Some(checkpoint) = &outcome.checkpoint {
        println!("checkpoint: {}", checkpoint.display());
    }
    if let Some(record) = &outcome.record {
        println!("record: {}", record.display());
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
