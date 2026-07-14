// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain glm …` — set up / train / evaluate / run the GLM-5.2 decoder
//! (MLA + sigmoid noaux_tc MoE + shared expert).
//!
//!   brain glm train <data_dir> --out F [--steps N --batch B --block T --lr X
//!                    --layers L --d-model D --heads H --experts E --size S --seed K]
//!   brain glm infer --weights F [--data <dir>] --prompt "..."
//!                    [--max-new N --temp X --top-k K --seed K --device cpu|gpu]
//!   brain glm eval  --weights F --data <dir> [--batches N --block T --seed K]
//!   brain glm finetune <data_dir> --weights BASE --out F [--steps N --lr X ...]
//!   brain glm import --hf <dir> --out glm.weights   (HuggingFace safetensors)

use std::path::Path;

use data::binio::Meta;
use data::rng::Rng;
use data::tokenizer::{CharTokenizer, Tokenizer};
use glm::config::GlmConfig;
use glm::model::Glm;

pub fn run_glm(args: &[String]) {
    match args.first().map(|s| s.as_str()) {
        Some("train") => train(&args[1..], None),
        Some("finetune") => finetune(&args[1..]),
        Some("infer") | Some("gen") => infer(&args[1..]),
        Some("eval") => eval(&args[1..]),
        Some("import") => import(&args[1..]),
        other => eprintln!("usage: brain glm <train|finetune|infer|eval|import> ...  (got {other:?})"),
    }
}

fn val(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    args.get(*i).cloned().unwrap_or_else(|| {
        eprintln!("{flag} requires a value");
        std::process::exit(2);
    })
}

/// Named size presets for from-scratch setup (`--size tiny|small|base`). All keep
/// the MLA head split + a MoE layer; they differ in depth/width/experts.
fn preset(size: &str, block: u32) -> GlmConfig {
    let base = GlmConfig { block_size: block, ..GlmConfig::tiny() };
    match size {
        "tiny" => GlmConfig { vocab: 0, ..base },
        "small" => GlmConfig {
            vocab: 0,
            n_layers: 4,
            d_model: 128,
            n_heads: 4,
            q_lora_rank: 96,
            kv_lora_rank: 48,
            qk_nope_head_dim: 32,
            qk_rope_head_dim: 16,
            v_head_dim: 32,
            n_routed_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 256,
            intermediate_size: 256,
            first_k_dense_replace: 1,
            ..base
        },
        _ => GlmConfig {
            vocab: 0,
            n_layers: 6,
            d_model: 256,
            n_heads: 8,
            q_lora_rank: 192,
            kv_lora_rank: 96,
            qk_nope_head_dim: 32,
            qk_rope_head_dim: 16,
            v_head_dim: 32,
            n_routed_experts: 8,
            num_experts_per_tok: 2,
            moe_intermediate_size: 512,
            intermediate_size: 512,
            first_k_dense_replace: 1,
            ..base
        },
    }
}

/// Shared core for `train` (fresh) and `finetune` (seeded from `--weights`).
fn train(args: &[String], base: Option<&str>) {
    let mut data_dir = String::new();
    let mut out = "out/glm.weights".to_string();
    let mut steps = 2000u32;
    let mut batch = 8u32;
    let mut block = 128u32;
    let mut lr = if base.is_some() { 1e-4 } else { 3e-4 };
    let mut seed = 1234u64;
    let mut size = "small".to_string();
    let mut layers: Option<u32> = None;
    let mut d_model: Option<u32> = None;
    let mut heads: Option<u32> = None;
    let mut experts: Option<u32> = None;
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
            "--size" => size = val(args, &mut i, "--size"),
            "--layers" => layers = val(args, &mut i, "--layers").parse().ok(),
            "--d-model" => d_model = val(args, &mut i, "--d-model").parse().ok(),
            "--heads" => heads = val(args, &mut i, "--heads").parse().ok(),
            "--experts" => experts = val(args, &mut i, "--experts").parse().ok(),
            "--seed" => seed = val(args, &mut i, "--seed").parse().unwrap_or(seed),
            "--mask" => mask = val(args, &mut i, "--mask").chars().next(),
            "--align" => align = true,
            s if !s.starts_with("--") && data_dir.is_empty() => data_dir = s.to_string(),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if data_dir.is_empty() {
        eprintln!("usage: brain glm {{train|finetune}} <data_dir> --out F [--size tiny|small|base --steps N --batch B --block T --lr X --layers L --d-model D --heads H --experts E --mask = --align]");
        return;
    }
    // finetune reads the architecture from the base checkpoint; train builds from
    // the requested preset + explicit overrides.
    let mut cfg = match base {
        Some(p) => GlmConfig::from_json(&checkpoint::load(p).header["config"]),
        None => preset(&size, block),
    };
    if base.is_none() {
        if let Some(l) = layers {
            cfg.n_layers = l;
        }
        if let Some(dm) = d_model {
            cfg.d_model = dm;
        }
        if let Some(h) = heads {
            cfg.n_heads = h;
        }
        if let Some(e) = experts {
            cfg.n_routed_experts = e;
        }
        cfg.block_size = block;
    }
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
    if let Some(p) = base {
        if !Path::new(&out).exists() {
            std::fs::copy(p, &out).unwrap_or_else(|e| panic!("seed finetune from {p}: {e}"));
        }
    }
    println!(
        "training glm on {data_dir}: steps={steps} batch={batch} block={block} layers={} d_model={} heads={} experts={}",
        cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_routed_experts
    );
    match model::fit::<Glm>(Path::new(&data_dir), cfg, &opts, Some(Path::new(&out))) {
        Ok((l0, l1)) => println!("trained: loss {l0:.4} -> {l1:.4}; saved {out}"),
        Err(e) => eprintln!("train error: {e}"),
    }
}

fn finetune(args: &[String]) {
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
        eprintln!("usage: brain glm finetune <data_dir> --weights BASE --out F [...]");
        return;
    }
    train(&rest, Some(&base));
}

fn model_block(weights: &str) -> u32 {
    GlmConfig::from_json(&checkpoint::load(weights).header["config"]).block_size
}

fn infer(args: &[String]) {
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
        eprintln!("usage: brain glm infer --weights F [--data <dir>] [--prompt ... --max-new N --temp X --top-k K]");
        return;
    }
    // Char vocab from the checkpoint (embedded at train time); `--data` is a
    // fallback for checkpoints without embedded vocab.
    let itos = match Glm::load_itos(&weights) {
        Some(itos) => itos,
        None => {
            if data_dir.is_empty() {
                eprintln!("checkpoint has no embedded char vocab; pass --data <dir> with its meta.json");
                return;
            }
            let meta_path = Path::new(&data_dir).join("meta.json");
            match std::fs::read_to_string(&meta_path).ok().and_then(|s| Meta::from_json(&s).ok()) {
                Some(m) => m.itos,
                None => {
                    eprintln!("infer needs a char vocab: none embedded and no meta.json at {}", meta_path.display());
                    return;
                }
            }
        }
    };
    let tok = CharTokenizer::from_itos(itos);
    let model = Glm::load_inference(&weights, 1, model_block(&weights));
    let prompt_text = if prompt.is_empty() { "\n" } else { prompt.as_str() };
    let prompt_ids: Vec<u32> = tok.encode(prompt_text);
    let mut rng = Rng::new(seed);
    let gen = glm::sample::generate(&model, &prompt_ids, max_new, temp, top_k, None, &mut rng);
    print!("{prompt_text}");
    print!("{}", tok.decode(&gen));
    println!();
}

/// Validation perplexity: mean masked-CE over sampled val windows -> exp(). Uses
/// the model's own forward (the same masked-CE the trainer optimises).
fn eval(args: &[String]) {
    let mut weights = String::new();
    let mut data_dir = String::new();
    let mut batches = 20usize;
    let mut block = 0u32;
    let mut seed = 99u64;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--data" => data_dir = val(args, &mut i, "--data"),
            "--batches" => batches = val(args, &mut i, "--batches").parse().unwrap_or(batches),
            "--block" => block = val(args, &mut i, "--block").parse().unwrap_or(block),
            "--seed" => seed = val(args, &mut i, "--seed").parse().unwrap_or(seed),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() || data_dir.is_empty() {
        eprintln!("usage: brain glm eval --weights F --data <dir> [--batches N --block T]");
        return;
    }
    let t = if block > 0 { block } else { model_block(&weights) };
    let val_tokens = match data::binio::read_tokens_u32(&Path::new(&data_dir).join("val")) {
        Ok(v) if v.len() as u32 > t + 1 => v,
        Ok(_) => {
            eprintln!("val split too short for block {t}");
            return;
        }
        Err(e) => {
            eprintln!("cannot read val split in {data_dir}: {e}");
            return;
        }
    };
    let model = Glm::load_inference(&weights, 1, t);
    let mut rng = Rng::new(seed);
    let mut ce_sum = 0.0f64;
    let tt = t as usize;
    for _ in 0..batches {
        let start = (rng.next_u64() as usize) % (val_tokens.len() - tt - 1);
        let x = &val_tokens[start..start + tt];
        let y = &val_tokens[start + 1..start + 1 + tt];
        model.set_batch(x, y);
        ce_sum += model.forward() as f64;
    }
    let mean_ce = ce_sum / batches as f64;
    println!("val_ce {mean_ce:.4}  val_perplexity {:.4}", mean_ce.exp());
}

fn import(args: &[String]) {
    let mut hf = String::new();
    let mut out = "glm.weights".to_string();
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
        eprintln!("usage: brain glm import --hf <dir> --out glm.weights");
        return;
    }
    match glm::import::import(&hf, &out) {
        Ok(()) => println!("ok: wrote {out}"),
        Err(e) => eprintln!("import failed: {e}"),
    }
}
