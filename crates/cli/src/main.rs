// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `brain` native CLI — one binary over every model in the workspace.
//!
//!   * `moe`  — sparse-MoE Transformer (RMSNorm/RoPE/top-k experts): the
//!     `brain-moe` crate (`model` inference, `train` training + parity).
//!   * `pid`  — event/effect control Transformer: the `brain-pid` crate
//!     (`model` + `data`). Drives the WebGPU demo.
//!
//! Shared infrastructure lives in dedicated crates: `gpu_core` (device +
//! dispatch), `checkpoint` (weights I/O), `paramstore` (weight/grad/Adam
//! buffers), `optim` (AdamW + grad clip), `kernels` (WGSL source of truth).
//!
//! Subcommands:
//!   brain [--weights F --prompt ... --max-new N]   # MoE inference (default)
//!   brain train|eval|validate [...]                # MoE training / eval / parity
//!   brain pid validate <ref.bin>                   # PID single-step parity gate
//!   brain pid stream   <stream.bin>                # PID fixed-data multi-step check
//!   brain pid train [--steps --effective-batch --mem-budget --seq-len ...]
//!   brain pid rollout --weights F                  # closed-loop generalization report

mod data_cli;
mod federated_cli;
mod gpt_cli;
mod pid_cli;

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
        _ => moe::run_generate(),
    }
}
