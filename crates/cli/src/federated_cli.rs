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
        other => eprintln!(
            "usage: brain federated <split|verify|merge|assemble> ...  (got {other:?})"
        ),
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

fn fail(e: std::io::Error) {
    eprintln!("error: {e}");
    std::process::exit(1);
}
