// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements efficient neural-network architectures and
// the evidence that they work for its clients. If your team needs expertise
// in structured and spectral model operators, you can procure our services
// by sending an email to info@swedishembedded.com.

//! SpectralLM on a document corpus, end to end:
//!
//!   spectral-lm train <corpus dir> --out F [shape and schedule flags]
//!   spectral-lm eval --checkpoint F --tokenizer T --text valid.txt [--max-doc-bytes N]
//!   spectral-lm generate --checkpoint F --tokenizer T [--prompt "..."]
//!   spectral-lm cost [shape flags]
//!   spectral-lm profile [shape flags] [--batch B --block T --reps R]
//!   spectral-lm qwen-eval --weights W --text valid.txt [--max-doc-bytes N]
//!   spectral-lm qwen-corpus --weights W --text train.txt --valid valid.txt --out DIR [--stories N]
//!   spectral-lm qwen-train --weights W <qwen corpus dir> --out F [schedule flags]
//!   spectral-lm qwen-spectrum --weights W --fit-text train.txt --text valid.txt [--layer L --widths 64,128]
//!
//! `train` with `--mixer swiglu` trains the conventional control, which is
//! the Qwen3 decoder (the crate's parity test holds it to brain's Qwen3
//! implementation). Every measuring subcommand ends with one JSON line.

mod lm;
mod qwen;
mod spectrum;

use appopts::Args;

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let rest = argv.get(1..).unwrap_or(&[]);
    let mut a = Args::new(rest);
    match argv.first().map(String::as_str) {
        Some("train") => lm::train(&mut a),
        Some("eval") => lm::eval(&mut a),
        Some("generate") => lm::generate(&mut a),
        Some("cost") => lm::cost(&mut a),
        Some("profile") => lm::profile(&mut a),
        Some("qwen-eval") => qwen::eval(&mut a),
        Some("qwen-corpus") => qwen::corpus(&mut a),
        Some("qwen-train") => qwen::train(&mut a),
        Some("qwen-spectrum") => spectrum::run(&mut a),
        other => {
            eprintln!("usage: spectral-lm <train|eval|generate|cost|profile|qwen-eval|qwen-corpus|qwen-train|qwen-spectrum> ... (got {other:?}); see README.md");
            std::process::exit(2);
        }
    }
}

/// Print `msg` and exit with the usage status.
pub(crate) fn die(msg: &str) -> ! {
    eprintln!("spectral-lm: {msg}");
    std::process::exit(2);
}

/// The schedule flags every training subcommand shares; `lr` is the
/// subcommand's default peak learning rate.
pub(crate) fn fit_opts(a: &mut Args, what: &str, lr: f32) -> model::FitOpts {
    let mut o = model::FitOpts::default();
    o.steps = a.u32_or("--steps", 2000);
    o.batch_size = a.u32_or("--batch", 16);
    o.block_size = a.u32_or("--block", 256);
    o.grad_accum = a.u32_or("--grad-accum", 1);
    o.lr = a.f32_or("--lr", lr);
    o.min_lr = a.f32_or("--min-lr", o.lr / 10.0);
    o.warmup = a.u32_or("--warmup", 100);
    o.weight_decay = a.f32_or("--weight-decay", 0.1);
    o.grad_clip = a.f32_or("--clip", 1.0);
    o.eval_interval = a.u32_or("--eval-interval", 0);
    o.eval_batches = a.u32_or("--eval-batches", 8);
    o.checkpoint_secs = a.u64_or("--checkpoint-secs", 0);
    o.seed = a.seed(what);
    o.decay_iters = o.steps;
    o
}

/// Training tokens a schedule consumes.
pub(crate) fn tokens_of(o: &model::FitOpts) -> u64 {
    o.steps as u64 * o.batch_size as u64 * o.block_size as u64 * o.grad_accum.max(1) as u64
}

/// The whole text file at `path`.
pub(crate) fn read_text(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| die(&format!("{path}: {e}")))
}
