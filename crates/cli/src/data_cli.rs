// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain data …` — dataset generation/preparation.
//!
//!   brain data gen <name> [--out DIR] [--n N] [--seed S]
//!
//! Names: shakespeare_char | calculator | reverser | wordcalc | gpt | timeseries
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
    let Some(ds) = Dataset::from_name(name) else {
        eprintln!(
            "unknown dataset {name:?}; expected one of: shakespeare_char calculator reverser wordcalc gpt timeseries"
        );
        return;
    };

    // Flag defaults: n is examples (text) or steps (timeseries).
    let mut out: PathBuf = PathBuf::from("data").join(ds.name());
    let mut n: usize = match ds {
        Dataset::Timeseries => 200_000,
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

fn arg_val(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    args.get(*i)
        .cloned()
        .unwrap_or_else(|| {
            eprintln!("{flag} requires a value");
            std::process::exit(2);
        })
}
