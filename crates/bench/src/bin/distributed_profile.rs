// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `distributed_profile` - the single-machine distributed-training profile.
//!
//! Runs the world sizes and transports in [`bench::distributed::ProfileConfig`]
//! and prints one JSON document (optionally also writing it to `--output`).
//! Usage:
//!
//! ```text
//! distributed_profile [--steps N] [--worlds 1,2,4] [--transports host,network,shaped]
//!                     [--dataset-examples N] [--seed N] [--output PATH]
//! ```
//!
//! See `crates/bench/src/distributed.rs` for what the report's fields mean and
//! what they do not include.

use std::process::ExitCode;

use bench::distributed::{run_profile, ProfileConfig, Transport};

const USAGE: &str = "\
usage: distributed_profile [options]

  --steps N               steady-state steps per run (default 3)
  --worlds a,b,...        world sizes to run (default 1,2,4)
  --transports a,b,...    host | network | shaped (default all three)
  --dataset-examples N    calculator examples to generate (default 2000)
  --seed N                weight/order seed (default 7)
  --output PATH           also write the JSON document to PATH
";

fn main() -> ExitCode {
    let mut cfg = ProfileConfig::default();
    let mut output: Option<String> = None;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = args.get(index + 1).map(String::as_str);
        match flag {
            "--help" | "-h" => {
                print!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "--steps" => {
                let Some(value) = value else { return usage_error("--steps needs a value") };
                let Ok(steps) = value.parse::<u32>() else { return usage_error("--steps needs an integer") };
                cfg.steps = steps;
                index += 1;
            }
            "--dataset-examples" => {
                let Some(value) = value else { return usage_error("--dataset-examples needs a value") };
                let Ok(examples) = value.parse::<usize>() else { return usage_error("--dataset-examples needs an integer") };
                cfg.dataset_examples = examples;
                index += 1;
            }
            "--seed" => {
                let Some(value) = value else { return usage_error("--seed needs a value") };
                let Ok(seed) = value.parse::<u64>() else { return usage_error("--seed needs an integer") };
                cfg.seed = seed;
                index += 1;
            }
            "--worlds" => {
                let Some(value) = value else { return usage_error("--worlds needs a comma-separated list") };
                let Ok(worlds) = parse_worlds(value) else { return usage_error("--worlds must be comma-separated positive integers") };
                cfg.worlds = worlds;
                index += 1;
            }
            "--transports" => {
                let Some(value) = value else { return usage_error("--transports needs a comma-separated list") };
                let Ok(transports) = parse_transports(value) else {
                    return usage_error("--transports must name host, network or shaped");
                };
                cfg.transports = transports;
                index += 1;
            }
            "--output" => {
                let Some(value) = value else { return usage_error("--output needs a path") };
                output = Some(value.to_string());
                index += 1;
            }
            other => return usage_error(&format!("unknown option '{other}'")),
        }
        index += 1;
    }

    match run_profile(&cfg) {
        Ok(report) => {
            let text = match serde_json::to_string_pretty(&report) {
                Ok(text) => text,
                Err(e) => {
                    eprintln!("distributed_profile: encode the report: {e}");
                    return ExitCode::FAILURE;
                }
            };
            if let Some(path) = output {
                if let Err(e) = std::fs::write(&path, &text) {
                    eprintln!("distributed_profile: write {path}: {e}");
                    return ExitCode::FAILURE;
                }
            }
            println!("{text}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            // A run that cannot execute names what is missing; it is never
            // reported as a successful empty result.
            eprintln!("distributed_profile: {e}");
            ExitCode::FAILURE
        }
    }
}

fn parse_worlds(value: &str) -> Result<Vec<usize>, ()> {
    let mut worlds = Vec::new();
    for part in value.split(',') {
        let world = part.trim().parse::<usize>().map_err(|_| ())?;
        if world == 0 {
            return Err(());
        }
        worlds.push(world);
    }
    if worlds.is_empty() {
        return Err(());
    }
    Ok(worlds)
}

fn parse_transports(value: &str) -> Result<Vec<Transport>, ()> {
    let mut transports = Vec::new();
    for part in value.split(',') {
        transports.push(Transport::parse(part.trim()).ok_or(())?);
    }
    if transports.is_empty() {
        return Err(());
    }
    Ok(transports)
}

fn usage_error(message: &str) -> ExitCode {
    eprintln!("distributed_profile: {message}\n\n{USAGE}");
    ExitCode::from(2)
}
