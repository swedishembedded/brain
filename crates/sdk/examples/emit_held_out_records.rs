// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for reproducing the held-out
// records a fine-tune reported its score on, for its clients. If your team
// needs expertise in reproducible evaluation splits of language-model
// fine-tunes then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The held-out split of a chat dataset as the trainer encodes it: one JSONL
//! line per record, `{"ids":[...],"mask":[...]}`, for the tooling that scores a
//! trained adapter against the base on the same ids (a served engine and the
//! in-process trainer both take that form).
//!
//! The split is deterministic (whole `metadata.group`s, from the last record
//! backwards, no seed), and a manifest beside the output names the dataset's
//! digest, the rule and the counts, so the same dataset always emits the same
//! bytes under the same name.
//!
//! ```text
//! cargo run --release -p brain --example emit_held_out_records -- \
//!   --base local:Qwen/Qwen3.8-27B-FP8 --dataset records.jsonl --out held-out.jsonl
//! ```

use std::path::PathBuf;

fn die(why: &str) -> ! {
    eprintln!("emit_held_out_records: {why}");
    std::process::exit(2)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut base, mut dataset, mut out, mut models_dir) = (None, None, None, None);
    let mut thinking = false;
    while let Some(flag) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| die(&format!("{flag} needs a value")));
        match flag.as_str() {
            "--base" => base = Some(value()),
            "--dataset" => dataset = Some(value()),
            "--out" => out = Some(value()),
            "--models-dir" => models_dir = Some(PathBuf::from(value())),
            "--thinking" => thinking = true,
            other => die(&format!("unknown flag {other}; give --base, --dataset, --out, [--models-dir], [--thinking]")),
        }
    }
    let (Some(base), Some(dataset), Some(out)) = (base, dataset, out) else {
        die("usage: emit_held_out_records --base <model> --dataset <records.jsonl> --out <held-out.jsonl> [--models-dir DIR] [--thinking]")
    };
    let outcome = brain::emit_held_out_records(&base, PathBuf::from(&dataset).as_path(), PathBuf::from(&out).as_path(), thinking, models_dir.as_deref()).unwrap_or_else(|e| die(&e.to_string()));
    // The output names what it is: the records file, the manifest that carries
    // the recipe, and the counts a run's own score can be checked against.
    println!("records:   {} (from {} held-out dataset record(s); thinking {})", outcome.emitted, outcome.held_out, thinking);
    println!("positions: {} supervised, {} token(s)", outcome.positions, outcome.tokens);
    println!("out:       {}", outcome.out.display());
    println!("manifest:  {}", outcome.manifest.display());
}
