// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `brain` native CLI — one binary over every model in the workspace.
//! The model is chosen by the subcommand (no global "model type" flag).
//!
//!   * `gpt`        — dense GPT decoder baseline (nanogpt parity).
//!   * `generate`/`train`/`eval`/`validate` — the sparse-MoE Transformer.
//!   * `federated`  — sharded-MoE shard split/assemble.
//!   * `data`       — dataset generation; `gradcheck` — backprop correctness gate.
//!   * `pid`        — event/effect control Transformer (the WebGPU demo).
//!
//! Run `brain help` for the full usage with examples.

mod data_cli;
mod federated_cli;
mod gpt_cli;
mod pid_cli;

const HELP: &str = "\
brain — train and evaluate neural nets from scratch on the GPU (Rust + WGSL).

USAGE: brain <command> [options]
The model is selected by the command.

DATA
  brain data gen <name> [--out DIR --n N --seed S]
      names: calculator | reverser | wordcalc | timeseries | shakespeare_char | gpt

GPT (dense baseline)
  brain gpt train <data_dir> [--out F --steps N --batch B --block T
                              --layers L --d-model D --heads H --lr X --mask = --align]
  brain gpt eval  --weights F --data <dir> [--batches N --samples M]
  brain gpt gen   --weights F --data <dir> [--prompt \"...\" --max-new N --temp X --top-k K]

SPARSE MoE
  brain train [--steps N --batch-size B --block-size T --lr X --out F]
  brain generate --weights F [--prompt 1,2,3,4 --max-new N --temperature X --top-k K]
  brain eval     --weights F [--samples N]
  brain validate [ref.bin]                # gradient parity gate (if a ref file exists)

FEDERATED MoE (train experts separately, then assemble)
  brain federated split    <base.weights> <out_dir>
  brain federated verify   <dir>
  brain federated merge     <dir> --out <full.weights>
  brain federated assemble  <base_dir> [overlay_dir ...] --out <full.weights>

OTHER
  brain gradcheck                          # finite-difference backprop check (GPT)
  brain pid <validate|stream|train|rollout|profile> ...
  brain help

EXAMPLES
  brain data gen calculator --out data/calculator --n 100000
  brain gpt train data/calculator --out out/gpt.weights --steps 2000 --mask =
  brain gpt eval  --weights out/gpt.weights --data data/calculator
  brain gpt gen   --weights out/gpt.weights --data data/calculator --prompt \"12+7=\" --max-new 8
  brain train --steps 2000 --out moe.weights
  brain generate --weights moe.weights --prompt 1,2,3,4 --max-new 64
  brain federated split moe.weights out/shards && brain federated verify out/shards
  brain gradcheck

Or drive everything via the Makefile:  make data/calculator train/gpt/calculator eval/gpt/calculator
";

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    match argv.get(1).map(|s| s.as_str()) {
        Some("data") => data_cli::run_data(&argv[2..]),
        Some("gpt") => gpt_cli::run_gpt(&argv[2..]),
        Some("federated") => federated_cli::run_federated(&argv[2..]),
        Some("gradcheck") => {
            let report = gradcheck::check_gpt(1);
            report.print();
            let fails = report.failures(4e-3, 8e-2);
            if fails.is_empty() {
                println!("gradcheck OK ({} tensors)", report.checks.len());
            } else {
                eprintln!("gradcheck FAILED for {} tensors", fails.len());
                std::process::exit(1);
            }
        }
        Some("pid") => pid_cli::run_pid(&argv[2..]),
        Some("validate") => {
            let path = argv
                .get(2)
                .map(|s| s.as_str())
                .unwrap_or("scratchpad/weights/train_ref.bin");
            moe::train::validate(path);
        }
        Some("train") => moe::run_train(&argv[2..]),
        Some("eval") => moe::run_eval(&argv[2..]),
        Some("generate") => moe::run_generate(),
        Some("help") | Some("-h") | Some("--help") | None => print!("{HELP}"),
        Some(other) => {
            eprintln!("brain: unknown command '{other}'\n");
            print!("{HELP}");
            std::process::exit(2);
        }
    }
}
