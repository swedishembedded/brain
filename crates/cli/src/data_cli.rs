// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain data …` — dataset generation/preparation.
//!
//!   brain data gen <name> [--out DIR] [--n N] [--seed S]
//!   brain data gen pong [--out DIR] [--episodes E] [--steps N] [--seed S] [--policy random|chase]
//!
//! Names: shakespeare_char | calculator | reverser | wordcalc | gpt | timeseries | tts | pong
//! (shakespeare_char/gpt read `<DIR>/input.txt`; the synthetic ones generate it).

use std::path::{Path, PathBuf};

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
            "unknown dataset {name:?}; expected one of: shakespeare_char calculator reverser wordcalc gpt timeseries tinystories tts pong detect localization classification scale multi_object background"
        );
        return;
    };

    let (seed, rest) = crate::args::split_seed(args, "data gen");
    let args = &rest[..];
    // Flag defaults: n is examples (text) or steps (timeseries).
    let mut out: PathBuf = PathBuf::from("data").join(ds.name());
    let mut n: usize = match ds {
        Dataset::Timeseries => 200_000,
        // Image datasets are far heavier per example; default to a small corpus.
        Dataset::Detect { .. } => 256,
        // TTS `text->codes`: examples (each ~14 tokens); a few thousand suffice.
        Dataset::Tts => 4_000,
        // TinyStories: the number of training stories to encode; 0 is all.
        Dataset::TinyStories => 0,
        _ => 100_000,
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                out = PathBuf::from(arg_val(args, &mut i, "--out"));
            }
            "--n" => {
                n = arg_val(args, &mut i, "--n").parse().unwrap_or(n);
            }
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }

    println!("preparing {} -> {} (n={n}, seed={seed})", ds.name(), out.display());
    match prepare(ds, &out, n, seed).and_then(|()| {
        record_generation(&out, ds.name(), serde_json::json!({ "n": n }), seed)
    }) {
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
    let (seed, rest) = crate::args::split_seed(args, "data gen pong");
    let args = &rest[..];
    let mut out: PathBuf = PathBuf::from("data").join("pong");
    let mut episodes: usize = 10;
    let mut steps: usize = 200;
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
    match data::gen_pong::generate(&out, episodes, steps, seed, policy).and_then(|()| {
        let params = serde_json::json!({ "episodes": episodes, "steps": steps, "policy": format!("{policy:?}") });
        record_generation(&out, "pong", params, seed).map_err(|e| e.to_string())
    }) {
        Ok(()) => println!("done: {}", out.display()),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

/// File beside a generated dataset that records how it was made, so a drawn
/// seed is never lost: `brain data gen <name> --seed <seed> ...` regenerates
/// the same data.
const GENERATION_RECORD: &str = "generation.json";

/// Write [`GENERATION_RECORD`] into `out`: the dataset name, its generation
/// parameters and the seed the data was drawn from.
fn record_generation(out: &Path, dataset: &str, params: serde_json::Value, seed: u64) -> std::io::Result<()> {
    let record = serde_json::json!({ "dataset": dataset, "params": params, "seed": seed });
    let text = serde_json::to_string_pretty(&record).map_err(std::io::Error::other)?;
    std::fs::write(out.join(GENERATION_RECORD), text + "\n")
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A generated dataset says which seed made it, so a drawn seed can be
    /// replayed with `brain data gen ... --seed <seed>`.
    #[test]
    fn a_generated_dataset_records_the_seed_it_was_drawn_from() {
        let dir = std::env::temp_dir().join(format!("brain-data-cli-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        record_generation(&dir, "calculator", serde_json::json!({ "n": 10 }), 4242).unwrap();
        let text = std::fs::read_to_string(dir.join(GENERATION_RECORD)).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let record: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(record["seed"], 4242);
        assert_eq!(record["dataset"], "calculator");
    }
}
