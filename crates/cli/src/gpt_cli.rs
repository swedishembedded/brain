// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain gpt …` — train / sample the dense GPT baseline.
//!
//!   brain gpt train <data_dir> [--out F --steps N --batch B --block T --lr X
//!                               --layers L --d-model D --heads H --mask = --align]
//!   brain gpt gen  --weights F --data <dir> [--prompt "..." --max-new N
//!                               --temp X --top-k K]

use std::path::{Path, PathBuf};

use data::binio::Meta;
use data::rng::Rng;
use data::tokenizer::{CharTokenizer, Tokenizer};
use gpt::model::Gpt;
use gpt::train::TrainOpts;
use gpt::GptConfig;

pub fn run_gpt(args: &[String]) {
    match args.first().map(|s| s.as_str()) {
        Some("train") => train(&args[1..]),
        Some("gen") => gen(&args[1..]),
        Some("eval") => eval(&args[1..]),
        other => eprintln!("usage: brain gpt <train|gen|eval> ...  (got {other:?})"),
    }
}

fn eval(args: &[String]) {
    let mut weights = String::new();
    let mut data_dir = String::new();
    let mut batches = 20usize;
    let mut samples = 200usize;
    let mut seed = 99u64;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--data" => data_dir = val(args, &mut i, "--data"),
            "--batches" => batches = val(args, &mut i, "--batches").parse().unwrap_or(batches),
            "--samples" => samples = val(args, &mut i, "--samples").parse().unwrap_or(samples),
            "--seed" => seed = val(args, &mut i, "--seed").parse().unwrap_or(seed),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() || data_dir.is_empty() {
        eprintln!("usage: brain gpt eval --weights F --data <dir> [--batches N --samples M]");
        return;
    }
    let dir = Path::new(&data_dir);
    match eval::gpt_val_perplexity(&weights, dir, batches, seed) {
        Ok(ppl) => println!("val_perplexity {ppl:.4}"),
        Err(e) => eprintln!("perplexity error: {e}"),
    }
    // exact-match only applies to LHS=RHS char datasets (have meta.json + '=').
    if dir.join("meta.json").exists() {
        match eval::gpt_exact_match(&weights, dir, samples, seed) {
            Ok((acc, n)) if n > 0 => println!("exact_match {:.4} ({}/{} held-out)", acc, (acc * n as f32).round() as usize, n),
            Ok(_) => {}
            Err(e) => eprintln!("exact-match error: {e}"),
        }
    }
}

fn val(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    args.get(*i).cloned().unwrap_or_else(|| {
        eprintln!("{flag} requires a value");
        std::process::exit(2);
    })
}

fn train(args: &[String]) {
    let Some(dir) = args.first().cloned() else {
        eprintln!("usage: brain gpt train <data_dir> [flags]");
        return;
    };
    let mut out: Option<PathBuf> = None;
    let mut o = TrainOpts::default();
    let mut cfg = GptConfig { vocab: 0, block_size: o.block_size, n_layers: 4, d_model: 128, n_heads: 4, d_ff: 0 };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => out = Some(PathBuf::from(val(args, &mut i, "--out"))),
            "--steps" => o.steps = val(args, &mut i, "--steps").parse().unwrap_or(o.steps),
            "--batch" => o.batch_size = val(args, &mut i, "--batch").parse().unwrap_or(o.batch_size),
            "--block" => o.block_size = val(args, &mut i, "--block").parse().unwrap_or(o.block_size),
            "--lr" => o.lr = val(args, &mut i, "--lr").parse().unwrap_or(o.lr),
            "--warmup" => o.warmup = val(args, &mut i, "--warmup").parse().unwrap_or(o.warmup),
            "--eval-interval" => o.eval_interval = val(args, &mut i, "--eval-interval").parse().unwrap_or(o.eval_interval),
            "--grad-accum" => o.grad_accum = val(args, &mut i, "--grad-accum").parse().unwrap_or(o.grad_accum),
            "--layers" => cfg.n_layers = val(args, &mut i, "--layers").parse().unwrap_or(cfg.n_layers),
            "--d-model" => cfg.d_model = val(args, &mut i, "--d-model").parse().unwrap_or(cfg.d_model),
            "--heads" => cfg.n_heads = val(args, &mut i, "--heads").parse().unwrap_or(cfg.n_heads),
            "--seed" => o.seed = val(args, &mut i, "--seed").parse().unwrap_or(o.seed),
            "--mask" => {
                o.mask_before = val(args, &mut i, "--mask").chars().next();
                o.mask_per_line = true;
            }
            "--align" => o.align_to_lines = true,
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    o.decay_iters = o.steps;
    println!(
        "training gpt on {dir}: steps={} batch={} block={} layers={} d_model={} heads={}",
        o.steps, o.batch_size, o.block_size, cfg.n_layers, cfg.d_model, cfg.n_heads
    );
    match gpt::train::train(Path::new(&dir), cfg, &o, out.as_deref()) {
        Ok((i0, i1)) => println!("done: train loss {i0:.4} -> {i1:.4}"),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn gen(args: &[String]) {
    let mut weights = String::new();
    let mut data_dir = String::new();
    let mut prompt = String::new();
    let mut max_new = 200usize;
    let mut temp = 0.8f32;
    let mut top_k = 40usize;
    let mut seed = 1234u64;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--data" => data_dir = val(args, &mut i, "--data"),
            "--prompt" => prompt = val(args, &mut i, "--prompt"),
            "--max-new" => max_new = val(args, &mut i, "--max-new").parse().unwrap_or(max_new),
            "--temp" => temp = val(args, &mut i, "--temp").parse().unwrap_or(temp),
            "--top-k" => top_k = val(args, &mut i, "--top-k").parse().unwrap_or(top_k),
            "--seed" => seed = val(args, &mut i, "--seed").parse().unwrap_or(seed),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() {
        eprintln!("usage: brain gpt gen --weights F [--data <dir>] [--prompt ... --max-new N --temp X --top-k K]");
        return;
    }

    // The char tokenizer comes from the checkpoint itself (vocab embedded at
    // train time) — inference needs no dataset. `--data` is only a fallback for
    // older checkpoints that predate embedded vocab.
    let itos = match gpt::model::Gpt::load_itos(&weights) {
        Some(itos) => itos,
        None => {
            if data_dir.is_empty() {
                eprintln!(
                    "this checkpoint has no embedded vocab (trained before vocab embedding, \
                     or a BPE model); pass --data <dir> with its meta.json"
                );
                return;
            }
            let meta_path = Path::new(&data_dir).join("meta.json");
            match std::fs::read_to_string(&meta_path).ok().and_then(|s| Meta::from_json(&s).ok()) {
                Some(m) => m.itos,
                None => {
                    eprintln!("gen needs a char vocab: none embedded and no meta.json at {}", meta_path.display());
                    return;
                }
            }
        }
    };
    let tok = CharTokenizer::from_itos(itos);

    // Build the model sized for block_size; seed prompt (default = newline).
    let model = Gpt::load(&weights, 1, model_block(&weights));
    let prompt_text = if prompt.is_empty() { "\n" } else { prompt.as_str() };
    let prompt_ids: Vec<u32> = tok.encode(prompt_text).iter().map(|&t| t as u32).collect();
    let mut rng = Rng::new(seed);
    let gen = gpt::sample::generate(&model, &prompt_ids, max_new, temp, top_k, &mut rng);
    let gen_u16: Vec<u16> = gen.iter().map(|&t| t as u16).collect();
    print!("{prompt_text}");
    print!("{}", tok.decode(&gen_u16));
    println!();
}

/// Read the block size from a checkpoint header so the decoder is sized right.
fn model_block(weights: &str) -> u32 {
    let c = checkpoint::load(weights);
    GptConfig::from_json(&c.header["config"]).block_size
}
