// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements command-line tooling for recurrent-depth
// language models for its clients. If your team needs expertise in importing or
// running a prelude/core/coda decoder locally then you can procure our services
// by sending an email to info@swedishembedded.com.

//! `brain huginn …` - import and run the Huginn recurrent-depth decoder.
//!
//!   brain huginn import   --hf <dir> --out huginn.safetensors
//!   brain huginn generate --checkpoint <hf dir | checkpoint> --prompt "…"
//!                         [--tokenizer tokenizer.json] [--max-new 64] [--ctx 1024]
//!                         [--steps N]
//!
//! `generate` runs the paged serving engine the resident uses, so the local
//! path and the served path share their arithmetic.

use std::time::Instant;

use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use model::paged::BlockTable;
use model::serve::PagedDecoder;

use crate::args::{canon_verb, Args};

pub fn run_huginn(argv: &[String]) {
    match argv.first().map(|s| canon_verb(s)) {
        Some("import") => import(&argv[1..]),
        Some("generate") | Some("infer") => generate(&argv[1..]),
        other => eprintln!("usage: brain huginn <import|generate> ...  (got {other:?})"),
    }
}

fn import(argv: &[String]) {
    let mut a = Args::new(argv);
    let hf = a.take_str("--hf");
    let out = a.str_or("--out", "huginn.safetensors");
    a.finish();
    let Some(hf) = hf else {
        eprintln!("usage: brain huginn import --hf <hf checkpoint dir> --out huginn.safetensors");
        std::process::exit(2);
    };
    if let Err(e) = huginn::import::import(&hf, &out) {
        eprintln!("huginn import: {e}");
        std::process::exit(1);
    }
}

fn generate(argv: &[String]) {
    let mut a = Args::new(argv);
    let checkpoint = a.take_str("--checkpoint").or_else(|| a.take_str("--weights"));
    let tokenizer = a.take_str("--tokenizer");
    let prompt = a.str_or("--prompt", "");
    let max_new = a.u32_or("--max-new", 64);
    let ctx = a.u32_or("--ctx", 1024);
    let block = a.u32_or("--block", 16);
    let steps = match a.take_str("--steps") {
        Some(raw) => match raw.parse::<u32>() {
            Ok(n) if n > 0 => Some(n),
            _ => {
                eprintln!("huginn: --steps must be a positive integer (got {raw:?})");
                std::process::exit(2);
            }
        },
        None => None,
    };
    a.finish();
    let Some(checkpoint) = checkpoint else {
        eprintln!("usage: brain huginn generate --checkpoint <hf dir|checkpoint> --prompt \"…\" [--tokenizer T --max-new N --ctx N --steps N]");
        std::process::exit(2);
    };
    let tokenizer = tokenizer.or_else(|| {
        let p = std::path::Path::new(&checkpoint);
        let dir = if p.is_dir() { p.to_path_buf() } else { p.parent()?.to_path_buf() };
        let t = dir.join("tokenizer.json");
        t.exists().then(|| t.to_string_lossy().to_string())
    });
    let Some(tokenizer) = tokenizer else {
        eprintln!("huginn: no tokenizer.json next to {checkpoint}; pass --tokenizer");
        std::process::exit(2);
    };
    let tok = match QwenBpe::from_file(&tokenizer) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer {tokenizer}: {e}");
            std::process::exit(1);
        }
    };
    let ids = tok.encode(&prompt);
    if ids.is_empty() {
        eprintln!("huginn: empty prompt");
        std::process::exit(2);
    }
    let need = ids.len() as u32 + max_new + 8;
    let ctx = ctx.max(need).div_ceil(block) * block;
    let t0 = Instant::now();
    // Depth is a serving setting; absent it the checkpoint's own
    // `mean_recurrence` is used, which for the released model is 32 stages
    // times four core blocks beside the weights and needs a smaller context.
    let mut eng = match huginn::serve::Engine::open_with_steps(&checkpoint, ctx, block, steps) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("huginn: {e}");
            std::process::exit(1);
        }
    };
    let mut table = BlockTable::new();
    let hidden = eng.prefill(&mut table, &ids);
    let mut next = eng.admit_greedy(&hidden);
    let eos = tok.special_id("<|end_text|>");
    let mut out = Vec::new();
    let mut generated = 0u32;
    while generated < max_new {
        if Some(next) == eos {
            break;
        }
        out.push(next);
        generated += 1;
        if generated == max_new {
            break;
        }
        let mut refs = [&mut table];
        next = eng.forward_batched_greedy(&mut refs, &[next])[0];
    }
    eprintln!(
        "{} prompt tokens, {generated} new, ctx {ctx}, steps {}, {:.2}s",
        ids.len(),
        eng.steps(),
        t0.elapsed().as_secs_f64()
    );
    print!("{}", tok.decode(&out));
}
