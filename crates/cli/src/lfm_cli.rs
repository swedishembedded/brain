// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain lfm2 …` - import / run the LFM2.5-Encoder (bidirectional hybrid
//! short-conv/attention encoder, tied MLM head). Uses the shared `args` grammar.
//!
//!   brain lfm2 import    --hf <dir> --out lfm.safetensors
//!   brain lfm2 fill-mask --weights F --tokenizer tokenizer.json --text "… <|mask|> …"
//!                        [--topk K]
//!   brain lfm2 embed     --weights F --tokenizer tokenizer.json
//!                        (--text "…" | --input FILE) [--out emb.f32] [--seq T]
//!
//! `infer` is accepted as an alias for `fill-mask` -- the canonical verb
//! every architecture answers to, here the mask-filling demo action.
//!
//! Both inference verbs run the chunked long-context path (bounded attention
//! slab), so an 8k-token input works on any backend within the binding budget.

use std::time::Instant;

use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use lfm2::model::Lfm;

use crate::args::{canon_verb, Args};

/// Attention-slab budget for the chunked path (per materialized `[H,chunk,T]`
/// score slab; 512 MiB picks chunk 2048 at T=8192, H=16).
const SLAB_BUDGET: u64 = 512 << 20;

pub fn run_lfm(argv: &[String]) {
    match argv.first().map(|s| canon_verb(s)) {
        Some("import") => import(&argv[1..]),
        Some("fill-mask") | Some("fillmask") | Some("infer") => fill_mask(&argv[1..]),
        Some("embed") => embed(&argv[1..]),
        Some("data") => data_prep(&argv[1..]),
        Some("finetune") => finetune(&argv[1..]),
        Some("eval") => eval(&argv[1..]),
        other => eprintln!("usage: brain lfm2 <import|fill-mask|embed|data|finetune|eval> ...  (got {other:?})"),
    }
}

/// The MLM corruption config for an LFM tokenizer: mask/pad/bos/eos resolved
/// from the tokenizer file, never hardcoded.
fn mlm_config(tok: &QwenBpe) -> data::mlm::MlmConfig {
    let mask = tok.special_id("<|mask|>").expect("tokenizer has <|mask|>");
    let mut specials: Vec<u32> = ["<|pad|>", "<|startoftext|>", "<|im_end|>"]
        .iter()
        .filter_map(|s| tok.special_id(s))
        .collect();
    specials.push(mask);
    data::mlm::MlmConfig { special_ids: specials, ..data::mlm::MlmConfig::new(mask, tok.vocab_size() as u32) }
}

/// Tokenize a text corpus into `train.u32.bin` / `val.u32.bin` + `meta.json`.
fn data_prep(argv: &[String]) {
    let mut a = Args::new(argv);
    let input = a.take_str("--input");
    let tokenizer = a.take_str("--tokenizer");
    let out = a.str_or("--out", "data/lfm");
    let val_frac = a.f32_or("--val-frac", 0.05);
    a.finish();
    let (Some(input), Some(tokenizer)) = (input, tokenizer) else {
        eprintln!("usage: brain lfm2 data --input corpus.txt --tokenizer tokenizer.json --out data/lfm [--val-frac 0.05]");
        std::process::exit(2);
    };
    let tok = load_tokenizer(&tokenizer);
    let text = std::fs::read_to_string(&input).unwrap_or_else(|e| {
        eprintln!("read {input}: {e}");
        std::process::exit(1);
    });
    let ids = tok.encode(&text);
    let split = ((ids.len() as f32) * (1.0 - val_frac)) as usize;
    let dir = std::path::Path::new(&out);
    std::fs::create_dir_all(dir).expect("mkdir");
    data::binio::write_u32_bin(&dir.join("train.u32.bin"), &ids[..split]).expect("write train");
    data::binio::write_u32_bin(&dir.join("val.u32.bin"), &ids[split..]).expect("write val");
    std::fs::write(dir.join("meta.json"), data::binio::Meta::vocab_only(tok.vocab_size())).expect("meta");
    eprintln!("{out}: {} train + {} val tokens (vocab {})", split, ids.len() - split, tok.vocab_size());
}

fn finetune(argv: &[String]) {
    let mut a = Args::new(argv);
    let weights = a.take_str("--weights");
    let tokenizer = a.take_str("--tokenizer");
    let data_dir = a.str_or("--data", "data/lfm");
    let out = a.str_or("--out", "out/lfm-ft.safetensors");
    let steps = a.u32_or("--steps", 100);
    let batch = a.u32_or("--batch", 4);
    let seq = a.u32_or("--seq", 1024);
    let lr = a.f32_or("--lr", 3e-5);
    let seed = a.u32_or("--seed", 0) as u64;
    a.finish();
    let (Some(weights), Some(tokenizer)) = (weights, tokenizer) else {
        eprintln!("usage: brain lfm2 finetune --weights F --tokenizer T [--data D --out F --steps N --batch B --seq T --lr X --seed K]");
        std::process::exit(2);
    };
    let tok = load_tokenizer(&tokenizer);
    let mlm = mlm_config(&tok);
    let dir = std::path::Path::new(&data_dir);
    let train = data::binio::read_tokens_u32(&dir.join("train")).expect("train split");
    let val = data::binio::read_tokens_u32(&dir.join("val")).unwrap_or_default();

    let t0 = Instant::now();
    // Long sequences train on the chunked regime: bounded [H,chunk,T] attention
    // slabs + the gathered supervised-row MLM head (full T×T scores and
    // [n,vocab] logits both exceed the ~2 GiB binding budget at 8k).
    let m = if seq > 2048 {
        let head_cap = ((batch * seq) as f32 * 0.35) as u32 + 64;
        Lfm::load_train_chunked(&weights, batch, seq, 512 << 20, head_cap)
    } else {
        Lfm::load_train(&weights, batch, seq)
    };
    eprintln!("loaded trainable model in {:.1}s (b={batch}, t={seq})", t0.elapsed().as_secs_f64());
    let opts = lfm2::train::MlmTrainOpts { steps, lr, seed, ..Default::default() };
    let val_loss = lfm2::train::finetune(&m, &train, &val, &mlm, batch, seq, &opts, &mut |line| eprintln!("{line}"));
    if let Some(parent) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    m.save(&out);
    eprintln!("saved {out}  (final val loss {val_loss:.4}, pseudo-ppl {:.2})", val_loss.exp());
}

fn eval(argv: &[String]) {
    let mut a = Args::new(argv);
    let weights = a.take_str("--weights");
    let tokenizer = a.take_str("--tokenizer");
    let data_dir = a.str_or("--data", "data/lfm");
    let batches = a.u32_or("--batches", 8);
    let batch = a.u32_or("--batch", 4);
    let seq = a.u32_or("--seq", 1024);
    let seed = a.u32_or("--seed", 0) as u64;
    a.finish();
    let (Some(weights), Some(tokenizer)) = (weights, tokenizer) else {
        eprintln!("usage: brain lfm2 eval --weights F --tokenizer T [--data D --batches N --batch B --seq T --seed K]");
        std::process::exit(2);
    };
    let tok = load_tokenizer(&tokenizer);
    let mlm = mlm_config(&tok);
    match eval::mlm::lfm_mlm_eval(&weights, std::path::Path::new(&data_dir), batches, batch, seq, &mlm, seed) {
        Ok((ppl, acc, n)) => println!("pseudo-perplexity {ppl:.3}  masked-accuracy {:.2}% ({n} masked tokens)", acc * 100.0),
        Err(e) => {
            eprintln!("lfm eval: {e}");
            std::process::exit(1);
        }
    }
}

fn import(argv: &[String]) {
    let mut a = Args::new(argv);
    let hf = a.take_str("--hf");
    let out = a.str_or("--out", "out/lfm.safetensors");
    a.finish();
    let Some(hf) = hf else {
        eprintln!("usage: brain lfm2 import --hf <hf_checkpoint_dir> --out lfm.safetensors");
        std::process::exit(2);
    };
    if let Some(parent) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = lfm2::import::import(&hf, &out) {
        eprintln!("lfm import: {e}");
        std::process::exit(1);
    }
}

fn load_tokenizer(path: &str) -> QwenBpe {
    match QwenBpe::from_file(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer: {e}");
            std::process::exit(1);
        }
    }
}

/// Template prefix + encoded text (the HF-equivalent single-sequence encoding).
fn encode_with_template(tok: &QwenBpe, text: &str) -> Vec<u32> {
    let mut ids: Vec<u32> = tok.template_prefix().to_vec();
    ids.extend(tok.encode(text));
    ids
}

fn fill_mask(argv: &[String]) {
    let mut a = Args::new(argv);
    let weights = a.str_or("--weights", "out/lfm.safetensors");
    let tokenizer = a.take_str("--tokenizer");
    let text = a.take_str("--text");
    let topk = a.u32_or("--topk", 5) as usize;
    a.finish();
    let (Some(tokenizer), Some(text)) = (tokenizer, text) else {
        eprintln!("usage: brain lfm2 fill-mask --weights F --tokenizer tokenizer.json --text \"… <|mask|> …\" [--topk K]");
        std::process::exit(2);
    };
    let tok = load_tokenizer(&tokenizer);
    let Some(mask_id) = tok.special_id("<|mask|>") else {
        eprintln!("tokenizer has no <|mask|> token");
        std::process::exit(1);
    };
    let ids = encode_with_template(&tok, &text);
    let mask_rows: Vec<u32> = ids
        .iter()
        .enumerate()
        .filter_map(|(i, &t)| (t == mask_id).then_some(i as u32))
        .collect();
    if mask_rows.is_empty() {
        eprintln!("no <|mask|> in the input text");
        std::process::exit(1);
    }

    let t0 = Instant::now();
    let m = Lfm::load_inference_chunked(&weights, 1, ids.len() as u32, SLAB_BUDGET, mask_rows.len() as u32);
    let t_load = t0.elapsed();
    m.set_tokens(&ids);
    m.set_probe_rows(&mask_rows);
    let t1 = Instant::now();
    m.forward();
    let logits = m.read_probe_logits();
    let t_fwd = t1.elapsed();

    let v = m.cfg.vocab as usize;
    for (i, &row) in mask_rows.iter().enumerate() {
        let lrow = &logits[i * v..(i + 1) * v];
        let mut idx: Vec<u32> = (0..v as u32).collect();
        idx.sort_unstable_by(|&x, &y| lrow[y as usize].total_cmp(&lrow[x as usize]));
        let picks: Vec<String> = idx[..topk.min(v)]
            .iter()
            .map(|&id| format!("{:?} ({id}, {:.2})", tok.decode(&[id]), lrow[id as usize]))
            .collect();
        println!("mask@{row}: {}", picks.join("  "));
    }
    eprintln!(
        "[{} tokens, chunk {:?}] load {:.2}s  forward {:.3}s",
        ids.len(),
        m.chunk(),
        t_load.as_secs_f64(),
        t_fwd.as_secs_f64()
    );
}

fn embed(argv: &[String]) {
    let mut a = Args::new(argv);
    let weights = a.str_or("--weights", "out/lfm.safetensors");
    let tokenizer = a.take_str("--tokenizer");
    let text = a.take_str("--text");
    let input = a.take_str("--input");
    let out = a.take_str("--out");
    let seq = a.u32_or("--seq", 0);
    a.finish();
    let Some(tokenizer) = tokenizer else {
        eprintln!("usage: brain lfm2 embed --weights F --tokenizer tokenizer.json (--text \"…\" | --input FILE) [--out emb.f32] [--seq T]");
        std::process::exit(2);
    };
    let tok = load_tokenizer(&tokenizer);
    let text = match (text, input) {
        (Some(t), _) => t,
        (None, Some(f)) => std::fs::read_to_string(&f).unwrap_or_else(|e| {
            eprintln!("read {f}: {e}");
            std::process::exit(1);
        }),
        _ => {
            eprintln!("embed: need --text or --input");
            std::process::exit(2);
        }
    };
    let mut ids = encode_with_template(&tok, &text);
    if seq > 0 {
        ids.truncate(seq as usize);
    }

    let t0 = Instant::now();
    let m = Lfm::load_inference_chunked(&weights, 1, ids.len() as u32, SLAB_BUDGET, 0);
    let t_load = t0.elapsed();
    m.set_tokens(&ids);
    let t1 = Instant::now();
    m.forward();
    let hidden = m.read_hidden();
    let t_fwd = t1.elapsed();

    // Mean-pooled sequence embedding (per-token states go to --out raw f32).
    let d = m.cfg.d_model as usize;
    let n = ids.len();
    let mut mean = vec![0.0f32; d];
    for row in hidden.chunks_exact(d) {
        for (m, &x) in mean.iter_mut().zip(row) {
            *m += x;
        }
    }
    for x in &mut mean {
        *x /= n as f32;
    }
    if let Some(out) = out {
        let bytes: Vec<u8> = hidden.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(&out, bytes).unwrap_or_else(|e| {
            eprintln!("write {out}: {e}");
            std::process::exit(1);
        });
        eprintln!("wrote {out}: [{n}, {d}] f32 LE");
    }
    println!(
        "embedding[{d}] mean-pool head: {:?} …",
        &mean[..8.min(d)].iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>()
    );
    eprintln!(
        "[{n} tokens, chunk {:?}] load {:.2}s  forward {:.3}s",
        m.chunk(),
        t_load.as_secs_f64(),
        t_fwd.as_secs_f64()
    );
}
