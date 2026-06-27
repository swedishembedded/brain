// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain qwen …` — import / run / fine-tune the Qwen3 decoder.
//!
//!   brain qwen import --hf <dir> --out qwen.weights
//!   brain qwen infer  --weights F --tokenizer tokenizer.json --prompt "..."
//!                     [--max-new N --temp X --top-k K --chat --device cpu|gpu]
//!   brain qwen train    <data_dir> --out F [--steps N --batch B --block T --lr X ...]
//!   brain qwen finetune <data_dir> --weights BASE --out F [--steps N --lr X ...]

use std::path::Path;

use data::rng::Rng;
use data::tokenizer::Tokenizer;
use qwen::config::QwenConfig;
use qwen::model::Qwen;

pub fn run_qwen(args: &[String]) {
    match args.first().map(|s| s.as_str()) {
        Some("import") => import(&args[1..]),
        Some("infer") | Some("gen") => infer(&args[1..]),
        Some("train") => train(&args[1..], None),
        Some("finetune") => finetune(&args[1..]),
        other => eprintln!("usage: brain qwen <import|infer|train|finetune> ...  (got {other:?})"),
    }
}

fn val(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    args.get(*i).cloned().unwrap_or_else(|| {
        eprintln!("{flag} requires a value");
        std::process::exit(2);
    })
}

fn import(args: &[String]) {
    let mut hf = String::new();
    let mut out = "qwen.weights".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--hf" => hf = val(args, &mut i, "--hf"),
            "--out" => out = val(args, &mut i, "--out"),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if hf.is_empty() {
        eprintln!("usage: brain qwen import --hf <dir> --out qwen.weights");
        return;
    }
    match qwen::import::import(&hf, &out) {
        Ok(()) => println!("ok: wrote {out}"),
        Err(e) => eprintln!("import failed: {e}"),
    }
}

fn infer(args: &[String]) {
    let mut weights = String::new();
    let mut tokenizer = String::new();
    let mut prompt = String::new();
    let mut max_new = 32usize;
    let mut temp = 0.0f32; // greedy by default
    let mut top_k = 0usize;
    let mut seed = 0u64;
    let mut chat = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--tokenizer" => tokenizer = val(args, &mut i, "--tokenizer"),
            "--prompt" => prompt = val(args, &mut i, "--prompt"),
            "--max-new" => max_new = val(args, &mut i, "--max-new").parse().unwrap_or(max_new),
            "--temp" => temp = val(args, &mut i, "--temp").parse().unwrap_or(temp),
            "--top-k" => top_k = val(args, &mut i, "--top-k").parse().unwrap_or(top_k),
            "--seed" => seed = val(args, &mut i, "--seed").parse().unwrap_or(seed),
            "--chat" => chat = true,
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() || tokenizer.is_empty() {
        eprintln!("usage: brain qwen infer --weights F --tokenizer tokenizer.json --prompt \"...\" [--max-new N --temp X --top-k K --chat]");
        return;
    }
    let tok = match data::qwen_tokenizer::QwenBpe::from_file(&tokenizer) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer load failed: {e}");
            return;
        }
    };
    let text = if chat {
        tok.apply_chat_template(&[("user", &prompt)], true)
    } else {
        prompt.clone()
    };
    let ids = tok.encode(&text);
    if ids.is_empty() {
        eprintln!("empty prompt");
        return;
    }
    let cap = (ids.len() + max_new) as u32;
    let model = Qwen::load_inference(&weights, 1, cap);
    let eos = tok.encode("<|im_end|>").first().copied();
    let mut rng = Rng::new(seed);
    let gen = qwen::sample::generate(&model, &ids, max_new, temp, top_k, eos, &mut rng);
    print!("{prompt}");
    print!("{}", tok.decode(&gen));
    println!();
}

/// Shared core for `train` (fresh) and `finetune` (seeded from `--weights`).
fn train(args: &[String], base: Option<&str>) {
    let mut data_dir = String::new();
    let mut out = "out/qwen.weights".to_string();
    let mut steps = 2000u32;
    let mut batch = 8u32;
    let mut block = 256u32;
    let mut lr = if base.is_some() { 1e-5 } else { 3e-4 };
    let mut seed = 1234u64;
    let mut mask: Option<char> = None;
    let mut align = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => out = val(args, &mut i, "--out"),
            "--steps" => steps = val(args, &mut i, "--steps").parse().unwrap_or(steps),
            "--batch" => batch = val(args, &mut i, "--batch").parse().unwrap_or(batch),
            "--block" => block = val(args, &mut i, "--block").parse().unwrap_or(block),
            "--lr" => lr = val(args, &mut i, "--lr").parse().unwrap_or(lr),
            "--seed" => seed = val(args, &mut i, "--seed").parse().unwrap_or(seed),
            "--mask" => mask = val(args, &mut i, "--mask").chars().next(),
            "--align" => align = true,
            s if !s.starts_with("--") && data_dir.is_empty() => data_dir = s.to_string(),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if data_dir.is_empty() {
        eprintln!("usage: brain qwen {{train|finetune}} <data_dir> --out F [--steps N --batch B --block T --lr X --mask = --align]");
        return;
    }
    // A small default architecture for from-scratch training; finetune reads the
    // architecture from the base checkpoint instead.
    let cfg = match base {
        Some(p) => {
            let c = checkpoint::load(p);
            QwenConfig::from_json(&c.header["config"])
        }
        None => QwenConfig {
            vocab: 0, // filled from the dataset
            block_size: block,
            n_layers: 6,
            d_model: 384,
            n_heads: 6,
            n_kv_heads: 2,
            head_dim: 64,
            d_ff: 1024,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            tie_embeddings: true,
            lora: None,
        },
    };
    let opts = model::FitOpts {
        steps,
        batch_size: batch,
        block_size: block,
        lr,
        warmup: 50,
        decay_iters: steps,
        min_lr: lr * 0.1,
        weight_decay: 0.1,
        grad_clip: 1.0,
        grad_accum: 1,
        eval_interval: (steps / 10).max(1),
        eval_batches: 20,
        mask_before: mask,
        mask_per_line: mask.is_some(),
        align_to_lines: align,
        seed,
    };
    // finetune: seed weights from the base checkpoint by pre-writing `out`.
    if let Some(p) = base {
        if !Path::new(&out).exists() {
            std::fs::copy(p, &out).unwrap_or_else(|e| panic!("seed finetune from {p}: {e}"));
        }
    }
    match model::fit::<Qwen>(Path::new(&data_dir), cfg, &opts, Some(Path::new(&out))) {
        Ok((l0, l1)) => println!("trained: loss {l0:.4} -> {l1:.4}; saved {out}"),
        Err(e) => eprintln!("train error: {e}"),
    }
}

fn finetune(args: &[String]) {
    // Extract --weights as the base, pass the rest through to the shared core.
    let mut base = String::new();
    let mut rest: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--weights" {
            base = val(args, &mut i, "--weights");
        } else {
            rest.push(args[i].clone());
        }
        i += 1;
    }
    if base.is_empty() {
        eprintln!("usage: brain qwen finetune <data_dir> --weights BASE --out F [...]");
        return;
    }
    train(&rest, Some(&base));
}
