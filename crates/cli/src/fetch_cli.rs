// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain fetch` — download a known model into the global model directory so
//! it auto-serves on the next `brain serve` (see `model_dir::discover`, no env
//! vars needed). GGUF is preferred when a model offers it (self-contained,
//! embeds its tokenizer); safetensors + a `tokenizer.json` sidecar otherwise.
//!
//!   brain fetch --list                              # known models
//!   brain fetch <name> [--format gguf|safetensors] [--models-dir DIR]

use crate::args::Args;
use crate::fetch::{find, known_models, Fetcher, HttpFetcher};

pub fn run_fetch(argv: &[String]) {
    let mut a = Args::new(argv);
    if a.take_flag("--list") {
        list();
        return;
    }
    let format = a.take_str("--format");
    let models_dir = a.take_str("--models-dir");
    let name = match a.positional() {
        Some(n) => n,
        None => {
            eprintln!("usage: brain fetch <name> [--format gguf|safetensors] [--models-dir DIR]");
            eprintln!("       brain fetch --list");
            std::process::exit(2);
        }
    };
    a.finish();

    let model = match find(&name) {
        Some(m) => m,
        None => {
            eprintln!("brain fetch: unknown model '{name}' (see `brain fetch --list`)");
            std::process::exit(1);
        }
    };

    let dir = match crate::model_dir::resolve(models_dir.as_deref()) {
        Some(d) => d,
        None => {
            eprintln!("brain fetch: no model directory resolved (no --models-dir, BRAIN_MODELS_DIR, or $HOME)");
            std::process::exit(1);
        }
    };
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("brain fetch: could not create model dir {} ({e})", dir.display());
        std::process::exit(1);
    }

    if let Some(bad) = &format {
        if bad != "gguf" && bad != "safetensors" {
            eprintln!("brain fetch: --format must be gguf or safetensors, got {bad:?}");
            std::process::exit(2);
        }
    }
    let want_safetensors = format.as_deref() == Some("safetensors");
    let want_gguf = format.as_deref() == Some("gguf");

    let fetcher = HttpFetcher;
    // GGUF preferred (no tokenizer sidecar needed) unless the caller pinned a
    // format that isn't offered.
    if let Some(src) = &model.gguf {
        if !want_safetensors {
            let dest = dir.join(format!("{}.gguf", model.name));
            fetch_one(&fetcher, src, &dest);
            eprintln!("brain fetch: wrote {} — `brain serve` will auto-serve it", dest.display());
            return;
        }
    }
    if let Some((weights, tokenizer)) = &model.safetensors {
        if want_gguf {
            eprintln!("brain fetch: {} has no gguf source (see `brain fetch --list`)", model.name);
            std::process::exit(1);
        }
        let wdest = dir.join(format!("{}.safetensors", model.name));
        fetch_one(&fetcher, weights, &wdest);
        // Shared sibling per model-dir convention (crates/cli/src/model_dir.rs)
        // — safe to fetch once and let multiple models share it, but here we
        // just always (re)fetch this model's tokenizer.json alongside it.
        let tdest = dir.join("tokenizer.json");
        fetch_one(&fetcher, tokenizer, &tdest);
        eprintln!("brain fetch: wrote {} + {} — `brain serve` will auto-serve it", wdest.display(), tdest.display());
        return;
    }
    eprintln!("brain fetch: {} has no source for format {:?}", model.name, format.unwrap_or_else(|| "gguf".into()));
    std::process::exit(1);
}

fn fetch_one(fetcher: &dyn Fetcher, src: &crate::fetch::Source, dest: &std::path::Path) {
    eprintln!("brain fetch: {} -> {}", src.url, dest.display());
    let mut last_pct: i64 = -1;
    let mut progress = |got: u64, total: Option<u64>| {
        if let Some(total) = total.filter(|&t| t > 0) {
            let pct = (got * 100 / total) as i64;
            if pct != last_pct {
                last_pct = pct;
                eprint!("\rbrain fetch: {pct:3}% ({} / {} MiB)", got >> 20, total >> 20);
            }
        } else {
            eprint!("\rbrain fetch: {} MiB", got >> 20);
        }
    };
    match fetcher.fetch(src, dest, &mut progress) {
        Ok(()) => eprintln!(),
        Err(e) => {
            eprintln!("\nbrain fetch: {e}");
            std::process::exit(1);
        }
    }
}

fn list() {
    println!("known models (brain fetch <name>):");
    for m in known_models() {
        let formats: Vec<&str> = [m.gguf.is_some().then_some("gguf"), m.safetensors.is_some().then_some("safetensors")]
            .into_iter()
            .flatten()
            .collect();
        println!("  {:<14} {}  [{}]", m.name, m.description, formats.join(", "));
    }
}
