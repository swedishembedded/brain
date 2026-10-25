// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The pretrained reference: Qwen3-0.6B scored on the same documents in bits
//! per byte, and fine-tuned on the same training stories.

use std::path::{Path, PathBuf};
use std::time::Instant;

use appopts::Args;
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use model::score::ByteScore;
use qwen3::model::Qwen;

use crate::{die, fit_opts, read_text, tokens_of};

/// The one-line record every scoring subcommand ends with.
pub fn print_score(checkpoint: &str, score: &ByteScore, max_bytes: usize, seconds: f64) {
    println!(
        "{}",
        serde_json::json!({
            "event": "eval",
            "checkpoint": checkpoint,
            "bits_per_byte": score.bits_per_byte(),
            "nats": score.nats,
            "bytes": score.bytes,
            "tokens": score.tokens,
            "documents": score.documents,
            "max_doc_bytes": max_bytes,
            "seconds": seconds,
        })
    );
}

/// The tokenizer of a Qwen checkpoint: `--tokenizer` (a `tokenizer.json` or
/// its directory), else the checkpoint's own directory.
pub(crate) fn tokenizer(a: &mut Args, weights: &str) -> (QwenBpe, u32) {
    let path = a.take_str("--tokenizer").unwrap_or_else(|| Path::new(weights).parent().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default());
    let tok = if Path::new(&path).is_dir() { QwenBpe::from_dir(&path) } else { QwenBpe::from_file(&path) };
    let tok = tok.unwrap_or_else(|e| die(&format!("tokenizer {path}: {e}")));
    let eot = tok.special_id("<|endoftext|>").unwrap_or_else(|| die(&format!("tokenizer {path} has no <|endoftext|>")));
    (tok, eot)
}

fn weights(a: &mut Args) -> String {
    a.take_str("--weights").unwrap_or_else(|| die("needs --weights <Qwen3 checkpoint>"))
}

pub fn eval(a: &mut Args) {
    let weights = weights(a);
    let (tok, eot) = tokenizer(a, &weights);
    let text = a.take_str("--text").unwrap_or_else(|| die("qwen-eval needs --text <documents file>"));
    let max_bytes = a.usize_or("--max-doc-bytes", usize::MAX);
    let batch = a.u32_or("--batch", 8);
    a.finish();
    let docs = model::score::byte_docs(&read_text(&text), &tok, eot, max_bytes);
    let longest = docs.iter().map(|d| d.tokens.len()).max().unwrap_or(2) as u32;
    let m = Qwen::load_inference(&weights, batch, longest);
    let start = Instant::now();
    let score = model::score::score_bytes(&m, &docs, batch, longest).unwrap_or_else(|e| die(&e));
    print_score(&weights, &score, max_bytes, start.elapsed().as_secs_f64());
}

/// Encode the training and validation stories with the Qwen tokenizer into a
/// `train.u32.bin`/`val.u32.bin` corpus, each document preceded by
/// `<|endoftext|>` exactly as the byte-level corpus is laid out.
pub fn corpus(a: &mut Args) {
    let weights = weights(a);
    let (tok, eot) = tokenizer(a, &weights);
    let train = a.take_str("--text").unwrap_or_else(|| die("qwen-corpus needs --text <training documents>"));
    let valid = a.take_str("--valid").unwrap_or_else(|| die("qwen-corpus needs --valid <validation documents>"));
    let out = PathBuf::from(a.take_str("--out").unwrap_or_else(|| die("qwen-corpus needs --out <dir>")));
    let stories = a.usize_or("--stories", usize::MAX);
    a.finish();
    std::fs::create_dir_all(&out).unwrap_or_else(|e| die(&format!("{}: {e}", out.display())));
    let encode = |text: &str, limit: usize| {
        let mut ids = Vec::new();
        for doc in data::prepare::split_documents(text).take(limit) {
            ids.push(eot);
            ids.extend(tok.encode(doc));
        }
        ids
    };
    let train_ids = encode(&read_text(&train), stories);
    let val_ids = encode(&read_text(&valid), usize::MAX);
    data::binio::write_u32_bin(&out.join("train.u32.bin"), &train_ids).unwrap_or_else(|e| die(&e.to_string()));
    data::binio::write_u32_bin(&out.join("val.u32.bin"), &val_ids).unwrap_or_else(|e| die(&e.to_string()));
    std::fs::write(out.join("meta.json"), data::binio::Meta::vocab_only(tok.vocab_size())).unwrap_or_else(|e| die(&e.to_string()));
    println!("{}", serde_json::json!({ "event": "corpus", "train_tokens": train_ids.len(), "val_tokens": val_ids.len(), "vocab": tok.vocab_size() }));
}

/// Continue training the pretrained model on the corpus written by
/// [`corpus`]: every weight, the same AdamW loop and schedule flags as the
/// from-scratch models.
pub fn train(a: &mut Args) {
    let weights = weights(a);
    let Some(dir) = a.positional() else { die("usage: spectral-lm qwen-train --weights W <qwen corpus dir> --out F [flags]") };
    let out = a.take_str("--out").map(PathBuf::from).unwrap_or_else(|| die("qwen-train needs --out"));
    // A pretrained base is continued at a rate far below a from-scratch one.
    let o = fit_opts(a, "spectral-lm qwen-train", 3e-5);
    a.finish();
    let (train, val, batch_cfg, _) = model::load_dataset(Path::new(&dir), &o).unwrap_or_else(|e| die(&e.to_string()));
    let m = Qwen::load(&weights, o.batch_size, o.block_size);
    let objective = model::causal_lm::<Qwen>(train, val, batch_cfg, None);
    let start = Instant::now();
    let (first, last) = model::fit_with(m, objective, &o, Some(&out)).unwrap_or_else(|e| die(&e.to_string()));
    let seconds = start.elapsed().as_secs_f64();
    let tokens = tokens_of(&o);
    println!(
        "{}",
        serde_json::json!({
            "event": "train",
            "base": weights,
            "seed": o.seed,
            "steps": o.steps,
            "batch": o.batch_size * o.grad_accum.max(1),
            "block": o.block_size,
            "lr": o.lr,
            "tokens": tokens,
            "initial_loss": first,
            "final_loss": last,
            "seconds": seconds,
            "tokens_per_s": tokens as f64 / seconds,
            "checkpoint": out,
        })
    );
}
