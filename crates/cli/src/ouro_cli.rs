// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements command-line tooling for recurrent-depth
// language models for its clients. If your team needs expertise in importing or
// running a looped decoder locally then you can procure our services by sending
// an email to info@swedishembedded.com.

//! `brain ouro …` - import and run the Ouro looped decoder.
//!
//!   brain ouro import   --hf <dir> --out ouro.safetensors
//!   brain ouro generate --checkpoint <hf dir | checkpoint> --prompt "…"
//!                       [--tokenizer tokenizer.json] [--max-new 64] [--ctx 1024]
//!   brain ouro train    --checkpoint <hf dir | checkpoint> [--steps 32]
//!                       [--seq 128] [--batch 2] [--lr 1e-5] [--out trained.safetensors]
//!
//! `generate` runs the paged serving engine the resident uses, so the local
//! path and the served path share their arithmetic.

use std::time::Instant;

use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use model::paged::BlockTable;
use model::serve::PagedDecoder;
use ouro::serve::Engine;

use crate::args::{canon_verb, Args};

pub fn run_ouro(argv: &[String]) {
    match argv.first().map(|s| canon_verb(s)) {
        Some("import") => import(&argv[1..]),
        Some("generate") | Some("infer") => generate(&argv[1..]),
        Some("train") | Some("fit") => train(&argv[1..]),
        other => eprintln!("usage: brain ouro <import|generate|train> ...  (got {other:?})"),
    }
}

fn import(argv: &[String]) {
    let mut a = Args::new(argv);
    let hf = a.take_str("--hf");
    let out = a.str_or("--out", "ouro.safetensors");
    a.finish();
    let Some(hf) = hf else {
        eprintln!("usage: brain ouro import --hf <hf checkpoint dir> --out ouro.safetensors");
        std::process::exit(2);
    };
    if let Err(e) = ouro::import::import(&hf, &out) {
        eprintln!("ouro import: {e}");
        std::process::exit(1);
    }
}

/// A short causal-LM training run on the checkpoint itself: the released
/// objective (every pass's cross-entropy weighted by its exit probability,
/// summed, averaged over supervised rows) through the analytic backward, one
/// AdamW step per batch. Prints the loss every step, so a falling loss is
/// measurable evidence that the shared stack trains.
fn train(argv: &[String]) {
    let mut a = Args::new(argv);
    let checkpoint = a.take_str("--checkpoint").or_else(|| a.take_str("--weights"));
    let tokenizer = a.take_str("--tokenizer");
    let text = a.str_or("--text", DEFAULT_TRAIN_TEXT);
    // Raw token ids, for a vocabulary no tokenizer here produces (a tiny
    // fixture checkpoint, a unit-scale run).
    let ids = a.take_str("--ids");
    let steps = a.u32_or("--steps", 32);
    let seq = a.u32_or("--seq", 128);
    let batch = a.u32_or("--batch", 2);
    let lr = a.f32_or("--lr", 1e-5);
    // Global-norm clip on the averaged gradient. The released checkpoint's
    // logits are large, so an unclipped first step can blow up.
    let clip = a.f32_or("--clip", 1.0);
    let out = a.take_str("--out");
    a.finish();
    let Some(checkpoint) = checkpoint else {
        eprintln!("usage: brain ouro train --checkpoint <hf dir|checkpoint> [--steps N --seq N --batch N --lr F --clip F --out PATH]");
        std::process::exit(2);
    };
    let tokenizer = match ids {
        Some(_) => None,
        None => tokenizer.or_else(|| {
            let p = std::path::Path::new(&checkpoint);
            let dir = if p.is_dir() { p.to_path_buf() } else { p.parent()?.to_path_buf() };
            let t = dir.join("tokenizer.json");
            t.exists().then(|| t.to_string_lossy().to_string())
        }),
    };
    let tokenizer = match tokenizer {
        Some(t) => t,
        None if ids.is_some() => String::new(),
        None => {
            eprintln!("ouro: no tokenizer.json next to {checkpoint}; pass --tokenizer or --ids");
            std::process::exit(2);
        }
    };
    let mut corpus = match &ids {
        Some(raw) => match raw.split(',').map(|t| t.trim().parse::<u32>()).collect::<Result<Vec<u32>, _>>() {
            Ok(v) => v,
            Err(_) => {
                eprintln!("ouro train: --ids takes comma-separated token ids (got {raw:?})");
                std::process::exit(2);
            }
        },
        None => {
            let tok = match QwenBpe::from_file(&tokenizer) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("tokenizer {tokenizer}: {e}");
                    std::process::exit(1);
                }
            };
            tok.encode(&text)
        }
    };
    if corpus.len() < 2 {
        eprintln!("ouro train: the training data is {} token(s)", corpus.len());
        std::process::exit(2);
    }
    // Causal-LM rows: every row predicts the next token of the corpus, the
    // last position of each row unsupervised.
    let (b, t) = (batch.max(1), seq.max(2));
    let need = (b * t + 1) as usize;
    while corpus.len() < need {
        let mut again = corpus.clone();
        corpus.append(&mut again);
    }
    let mut tokens = Vec::with_capacity((b * t) as usize);
    let mut targets = Vec::with_capacity((b * t) as usize);
    for row in 0..b as usize {
        let start = row * t as usize;
        tokens.extend_from_slice(&corpus[start..start + t as usize]);
        targets.extend_from_slice(&corpus[start + 1..start + 1 + t as usize]);
        targets[(t - 1) as usize] = ouro::model::IGNORE;
    }

    let src: Box<dyn checkpoint::TensorSource> = if std::path::Path::new(&checkpoint).is_dir() {
        match ouro::import::open_dir(std::path::Path::new(&checkpoint)) {
            Ok((_, s)) => Box::new(s),
            Err(e) => {
                eprintln!("ouro: {e}");
                std::process::exit(1);
            }
        }
    } else {
        match checkpoint::weightio::WeightReader::open(&checkpoint) {
            Ok(r) => Box::new(r),
            Err(e) => {
                eprintln!("{checkpoint}: {e}");
                std::process::exit(1);
            }
        }
    };
    let cfg = match std::path::Path::new(&checkpoint).is_dir() {
        true => ouro::import::config_from_hf(&std::fs::read_to_string(std::path::Path::new(&checkpoint).join("config.json")).unwrap_or_default()).unwrap_or_else(|e| {
            eprintln!("ouro: {e}");
            std::process::exit(1);
        }),
        false => ouro::OuroConfig::from_json(&checkpoint::weightio::WeightReader::open(&checkpoint).map(|r| r.config()).unwrap_or_default()).unwrap_or_else(|e| {
            eprintln!("ouro: {e}");
            std::process::exit(1);
        }),
    };
    let t0 = Instant::now();
    let model = ouro::model::Ouro::new_train(cfg, b, t, src.as_ref());
    model::Model::set_batch(&model, model::Batch::Lm { tokens: &tokens, targets: &targets });
    let mut first = None;
    for step in 0..steps {
        model::Model::zero_grads(&model);
        let loss = model::Model::forward(&model);
        model::Model::backward(&model);
        // AdamW's bias correction is 1-based: step 0 would divide by zero.
        model::Model::adamw_step(&model, step + 1, lr, 0.0, model::Adam::default(), (clip > 0.0).then_some(clip), 1.0);
        if first.is_none() {
            first = Some(loss);
        }
        eprintln!("step {step}: loss {loss:.4}");
    }
    if let Some(out) = out.as_deref() {
        model::Model::save(&model, out);
        eprintln!("wrote {out}");
    }
    eprintln!(
        "{} steps, {b}x{t} tokens, lr {lr}, {:.1}s, loss {:.4} -> {:.4}",
        steps,
        t0.elapsed().as_secs_f64(),
        first.unwrap_or(f32::NAN),
        model::Model::forward(&model)
    );
}

/// Default training text: enough distinct prose to make a few rows of
/// causal-LM targets without repeating one sentence.
const DEFAULT_TRAIN_TEXT: &str = "The city of Paris is the capital of France. \
Its history reaches back more than two thousand years, from a small settlement \
on the banks of the Seine to the seat of government, learning and art that it \
is today. The Eiffel Tower was completed in 1889 for the world's fair, and the \
Louvre holds one of the largest collections of art anywhere. The cathedral of \
Notre-Dame stands on the island in the middle of the river, and the wide \
boulevards that cross the city were laid out in the nineteenth century.";

fn generate(argv: &[String]) {
    let mut a = Args::new(argv);
    let checkpoint = a.take_str("--checkpoint").or_else(|| a.take_str("--weights"));
    let tokenizer = a.take_str("--tokenizer");
    let prompt = a.str_or("--prompt", "");
    let max_new = a.u32_or("--max-new", 64);
    let ctx = a.u32_or("--ctx", 1024);
    let block = a.u32_or("--block", 16);
    a.finish();
    let Some(checkpoint) = checkpoint else {
        eprintln!("usage: brain ouro generate --checkpoint <hf dir|checkpoint> --prompt \"…\" [--tokenizer T --max-new N --ctx N]");
        std::process::exit(2);
    };
    let tokenizer = tokenizer.or_else(|| {
        let p = std::path::Path::new(&checkpoint);
        let dir = if p.is_dir() { p.to_path_buf() } else { p.parent()?.to_path_buf() };
        let t = dir.join("tokenizer.json");
        t.exists().then(|| t.to_string_lossy().to_string())
    });
    let Some(tokenizer) = tokenizer else {
        eprintln!("ouro: no tokenizer.json next to {checkpoint}; pass --tokenizer");
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
        eprintln!("ouro: empty prompt");
        std::process::exit(2);
    }
    let need = ids.len() as u32 + max_new + 8;
    let ctx = ctx.max(need).div_ceil(block) * block;
    let t0 = Instant::now();
    let mut eng = match Engine::open(&checkpoint, ctx, block) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("ouro: {e}");
            std::process::exit(1);
        }
    };
    let mut table = BlockTable::new();
    let hidden = eng.prefill(&mut table, &ids);
    let mut next = eng.admit_greedy(&hidden);
    let eos = tok.special_id("<|im_end|>");
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
        "{} prompt tokens, {generated} new, ctx {ctx}, {:.2}s",
        ids.len(),
        t0.elapsed().as_secs_f64()
    );
    print!("{}", tok.decode(&out));
}
