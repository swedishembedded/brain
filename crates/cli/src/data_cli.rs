// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain data …` — dataset generation/preparation.
//!
//!   brain data gen <name> [--out DIR] [--n N] [--seed S]
//!   brain data gen pong [--out DIR] [--episodes E] [--steps N] [--seed S] [--policy random|chase]
//!
//! Names: shakespeare_char | calculator | reverser | wordcalc | gpt | timeseries | tts | pong
//! (shakespeare_char/gpt read `<DIR>/input.txt`; the synthetic ones generate it).

use std::path::PathBuf;

use data::prepare::{prepare, Dataset};

pub fn run_data(args: &[String]) {
    match args.first().map(|s| s.as_str()) {
        Some("gen") => gen(&args[1..]),
        other => {
            eprintln!("usage: brain data gen <name> [--out DIR] [--n N] [--seed S]  (got {other:?})");
        }
    }
}

fn gen(args: &[String]) {
    let Some(name) = args.first() else {
        eprintln!("usage: brain data gen <name> [--out DIR] [--n N] [--seed S]");
        return;
    };
    if name == "pong" {
        return gen_pong(&args[1..]);
    }
    let Some(ds) = Dataset::from_name(name) else {
        eprintln!(
            "unknown dataset {name:?}; expected one of: shakespeare_char calculator reverser wordcalc gpt timeseries tts pong detect localization classification scale multi_object background"
        );
        return;
    };

    // Flag defaults: n is examples (text) or steps (timeseries).
    let mut out: PathBuf = PathBuf::from("data").join(ds.name());
    let mut n: usize = match ds {
        Dataset::Timeseries => 200_000,
        // Image datasets are far heavier per example; default to a small corpus.
        Dataset::Detect { .. } => 256,
        // TTS `text->codes`: examples (each ~14 tokens); a few thousand suffice.
        Dataset::Tts => 4_000,
        _ => 100_000,
    };
    let mut seed: u64 = 1337;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                out = PathBuf::from(arg_val(args, &mut i, "--out"));
            }
            "--n" => {
                n = arg_val(args, &mut i, "--n").parse().unwrap_or(n);
            }
            "--seed" => {
                seed = arg_val(args, &mut i, "--seed").parse().unwrap_or(seed);
            }
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }

    println!("preparing {} -> {} (n={n}, seed={seed})", ds.name(), out.display());
    match prepare(ds, &out, n, seed) {
        Ok(()) => println!("done: {}", out.display()),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

/// `brain data gen pong` — roll the pong world-model substrate into an
/// episode dataset (frames + actions + rewards; see `data::episode`).
fn gen_pong(args: &[String]) {
    let mut out: PathBuf = PathBuf::from("data").join("pong");
    let mut episodes: usize = 10;
    let mut steps: usize = 200;
    let mut seed: u64 = 1337;
    let mut policy = data::gen_pong::Policy::Random;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                out = PathBuf::from(arg_val(args, &mut i, "--out"));
            }
            "--episodes" => {
                episodes = arg_val(args, &mut i, "--episodes").parse().unwrap_or(episodes);
            }
            "--steps" => {
                steps = arg_val(args, &mut i, "--steps").parse().unwrap_or(steps);
            }
            "--seed" => {
                seed = arg_val(args, &mut i, "--seed").parse().unwrap_or(seed);
            }
            "--policy" => {
                let p = arg_val(args, &mut i, "--policy");
                policy = data::gen_pong::Policy::from_name(&p).unwrap_or_else(|| {
                    eprintln!("unknown --policy {p:?}; expected random or chase");
                    std::process::exit(2);
                });
            }
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }

    println!(
        "generating pong -> {} (episodes={episodes}, steps={steps}, seed={seed}, policy={policy:?})",
        out.display()
    );
    match data::gen_pong::generate(&out, episodes, steps, seed, policy) {
        Ok(()) => println!("done: {}", out.display()),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn arg_val(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    args.get(*i)
        .cloned()
        .unwrap_or_else(|| {
            eprintln!("{flag} requires a value");
            std::process::exit(2);
        })
}
