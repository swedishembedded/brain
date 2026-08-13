// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain gpt …` — train / evaluate / sample the dense GPT baseline. Uses the
//! shared `args` grammar; `infer` is the canonical verb for `gen`.
//!
//!   brain gpt train <data_dir> [--out F --steps N --batch B --block T --lr X
//!                               --layers L --d-model D --heads H --mask = --align]
//!   brain gpt infer --weights F [--data <dir>] [--prompt "..." --max-new N
//!                               --temp X --top-k K]
//!   brain gpt eval  --weights F --data <dir> [--batches N --samples M]

use std::path::{Path, PathBuf};

use data::binio::Meta;
use data::rng::Rng;
use data::tokenizer::{CharTokenizer, Tokenizer};
use gpt2::model::Gpt;
use gpt2::train::TrainOpts;
use gpt2::GptConfig;

use crate::args::{canon_verb, Args};

pub fn run_gpt(argv: &[String]) {
    match argv.first().map(|s| canon_verb(s)) {
        Some("train") => train(&argv[1..]),
        Some("infer") => gen(&argv[1..]),
        Some("eval") => eval(&argv[1..]),
        other => eprintln!("usage: brain gpt <train|infer|eval> ...  (got {other:?})"),
    }
}

fn eval(args: &[String]) {
    let mut a = Args::new(args);
    let weights = a.str_or("--weights", "");
    let data_dir = a.str_or("--data", "");
    let batches = a.usize_or("--batches", 20);
    let samples = a.usize_or("--samples", 200);
    let seed = a.u64_or("--seed", 99);
    a.finish();
    if weights.is_empty() || data_dir.is_empty() {
        eprintln!("usage: brain gpt eval --weights F --data <dir> [--batches N --samples M]");
        return;
    }
    let dir = Path::new(&data_dir);
    match eval::gpt_val_perplexity(&weights, dir, batches, seed) {
        Ok(ppl) => println!("val_perplexity {ppl:.4}"),
        Err(e) => eprintln!("perplexity error: {e}"),
    }
    if dir.join("meta.json").exists() {
        match eval::gpt_exact_match(&weights, dir, samples, seed) {
            Ok((acc, n)) if n > 0 => println!("exact_match {:.4} ({}/{} held-out)", acc, (acc * n as f32).round() as usize, n),
            Ok(_) => {}
            Err(e) => eprintln!("exact-match error: {e}"),
        }
    }
}

fn train(args: &[String]) {
    let mut a = Args::new(args);
    let Some(dir) = a.positional() else {
        eprintln!("usage: brain gpt train <data_dir> [flags]");
        return;
    };
    let mut o = TrainOpts::default();
    let mut cfg = GptConfig { vocab: 0, block_size: o.block_size, n_layers: 4, d_model: 128, n_heads: 4, d_ff: 0 };
    let out = a.take_str("--out").map(PathBuf::from);
    o.steps = a.u32_or("--steps", o.steps);
    o.batch_size = a.u32_or("--batch", o.batch_size);
    o.block_size = a.u32_or("--block", o.block_size);
    o.lr = a.f32_or("--lr", o.lr);
    o.warmup = a.u32_or("--warmup", o.warmup);
    o.eval_interval = a.u32_or("--eval-interval", o.eval_interval);
    o.grad_accum = a.u32_or("--grad-accum", o.grad_accum);
    cfg.n_layers = a.u32_or("--layers", cfg.n_layers);
    cfg.d_model = a.u32_or("--d-model", cfg.d_model);
    cfg.n_heads = a.u32_or("--heads", cfg.n_heads);
    o.seed = a.u64_or("--seed", o.seed);
    if let Some(m) = a.char_opt("--mask") {
        o.mask_before = Some(m);
        o.mask_per_line = true;
    }
    o.align_to_lines = a.take_flag("--align");
    a.finish();
    o.decay_iters = o.steps;
    println!(
        "training gpt on {dir}: steps={} batch={} block={} layers={} d_model={} heads={}",
        o.steps, o.batch_size, o.block_size, cfg.n_layers, cfg.d_model, cfg.n_heads
    );
    match gpt2::train::train(Path::new(&dir), cfg, &o, out.as_deref()) {
        Ok((i0, i1)) => println!("done: train loss {i0:.4} -> {i1:.4}"),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn gen(args: &[String]) {
    let mut a = Args::new(args);
    let weights = a.str_or("--weights", "");
    let data_dir = a.str_or("--data", "");
    let prompt = a.str_or("--prompt", "");
    let max_new = a.usize_or("--max-new", 200);
    let temp = a.f32_or("--temp", 0.8);
    let top_k = a.usize_or("--top-k", 40);
    let seed = a.u64_or("--seed", 1234);
    a.finish();
    if weights.is_empty() {
        eprintln!("usage: brain gpt infer --weights F [--data <dir>] [--prompt ... --max-new N --temp X --top-k K]");
        return;
    }
    let itos = match gpt2::model::Gpt::load_itos(&weights) {
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
    let model = Gpt::load(&weights, 1, model_block(&weights));
    let prompt_text = if prompt.is_empty() { "\n" } else { prompt.as_str() };
    let prompt_ids: Vec<u32> = tok.encode(prompt_text);
    let mut rng = Rng::new(seed);
    let gen = gpt2::sample::generate_kv(&model, &prompt_ids, max_new, temp, top_k, &mut rng);
    print!("{prompt_text}");
    print!("{}", tok.decode(&gen));
    println!();
}

/// Read the block size from a checkpoint header so the decoder is sized right.
fn model_block(weights: &str) -> u32 {
    let c = checkpoint::load(weights);
    GptConfig::from_json(&c.header["config"]).block_size
}
