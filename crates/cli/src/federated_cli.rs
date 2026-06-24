// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain federated …` — drive the sharded-MoE artifact pipeline.
//!
//!   brain federated split   <base.weights> <out_dir>
//!   brain federated verify  <dir>
//!   brain federated merge    <dir> --out <full.weights>
//!   brain federated assemble <base_dir> [overlay_dir ...] --out <full.weights>

use std::path::Path;

pub fn run_federated(args: &[String]) {
    match args.first().map(|s| s.as_str()) {
        Some("split") => split(&args[1..]),
        Some("verify") => verify(&args[1..]),
        Some("merge") => merge(&args[1..]),
        Some("assemble") => assemble(&args[1..]),
        Some("train-expert") => train_expert(&args[1..]),
        other => eprintln!(
            "usage: brain federated <split|verify|merge|assemble|train-expert> ...  (got {other:?})"
        ),
    }
}

/// Train one expert against a frozen backbone and emit an overlay shard dir.
///   brain federated train-expert --base B --expert E --out DIR
///                                [--steps N --batch B --block T --lr X --seed S]
fn train_expert(args: &[String]) {
    let mut base = String::new();
    let mut out = String::new();
    let mut expert = 0u32;
    let mut steps = 200u32;
    let mut b = 16u32;
    let mut t = 64u32;
    let mut lr = 6e-4f32;
    let mut seed = 1337u64;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--base" => base = next(args, &mut i),
            "--out" => out = next(args, &mut i),
            "--expert" => expert = next(args, &mut i).parse().unwrap_or(expert),
            "--steps" => steps = next(args, &mut i).parse().unwrap_or(steps),
            "--batch" => b = next(args, &mut i).parse().unwrap_or(b),
            "--block" => t = next(args, &mut i).parse().unwrap_or(t),
            "--lr" => lr = next(args, &mut i).parse().unwrap_or(lr),
            "--seed" => seed = next(args, &mut i).parse().unwrap_or(seed),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if base.is_empty() || out.is_empty() {
        eprintln!("usage: brain federated train-expert --base <base.weights> --expert E --out <dir> [--steps N --batch B --block T --lr X --seed S]");
        return;
    }

    let out_dir = Path::new(&out);
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        return fail(e);
    }
    // Train expert E (frozen backbone) -> a full updated checkpoint, then keep
    // only that expert's shard (+ shared) as an overlay dir for `assemble`.
    let tmp_full = out_dir.join(".worker_full.weights");
    moe::train::train_expert(moe::train::ExpertTrainArgs {
        base_weights: base,
        expert,
        out: tmp_full.to_string_lossy().into_owned(),
        steps,
        b,
        t,
        lr,
        seed,
    });
    match federated::split_filtered(tmp_full.to_str().unwrap(), out_dir, Some(&[expert])) {
        Ok(_) => {
            let _ = std::fs::remove_file(&tmp_full);
            println!("expert {expert} overlay shard -> {out}");
        }
        Err(e) => fail(e),
    }
}

fn split(a: &[String]) {
    if a.len() < 2 {
        eprintln!("usage: brain federated split <base.weights> <out_dir>");
        return;
    }
    match federated::split(&a[0], Path::new(&a[1])) {
        Ok(m) => println!("split {} -> {} ({} experts: {:?})", a[0], a[1], m.experts.len(), m.experts),
        Err(e) => fail(e),
    }
}

fn verify(a: &[String]) {
    let Some(dir) = a.first() else {
        eprintln!("usage: brain federated verify <dir>");
        return;
    };
    match federated::verify(Path::new(dir)) {
        Ok(m) => println!("OK: {} ({} files, base {})", dir, m.files.len(), &m.base_config_sha256[..16]),
        Err(e) => fail(e),
    }
}

fn merge(a: &[String]) {
    let (pos, out) = split_out(a);
    let (Some(dir), Some(out)) = (pos.first(), out) else {
        eprintln!("usage: brain federated merge <dir> --out <full.weights>");
        return;
    };
    match federated::merge_to_full(Path::new(dir), &out) {
        Ok(()) => println!("merged {dir} -> {out}"),
        Err(e) => fail(e),
    }
}

fn assemble(a: &[String]) {
    let (pos, out) = split_out(a);
    let (Some(base), Some(out)) = (pos.first(), out) else {
        eprintln!("usage: brain federated assemble <base_dir> [overlay_dir ...] --out <full.weights>");
        return;
    };
    let overlays: Vec<&Path> = pos[1..].iter().map(Path::new).collect();
    match federated::assemble(Path::new(base), &overlays, &out) {
        Ok(()) => println!("assembled {base} (+{} overlays) -> {out}", overlays.len()),
        Err(e) => fail(e),
    }
}

/// Split args into positionals and an `--out VALUE`.
fn split_out(a: &[String]) -> (Vec<String>, Option<String>) {
    let mut pos = Vec::new();
    let mut out = None;
    let mut i = 0;
    while i < a.len() {
        if a[i] == "--out" {
            out = a.get(i + 1).cloned();
            i += 2;
        } else {
            pos.push(a[i].clone());
            i += 1;
        }
    }
    (pos, out)
}

/// Advance `*i` to the next arg and return it (empty if absent).
fn next(args: &[String], i: &mut usize) -> String {
    *i += 1;
    args.get(*i).cloned().unwrap_or_default()
}

fn fail(e: std::io::Error) {
    eprintln!("error: {e}");
    std::process::exit(1);
}
