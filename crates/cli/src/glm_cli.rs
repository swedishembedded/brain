// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain glm …` — set up / train / evaluate / run the GLM-5.2 decoder
//! (MLA + sigmoid noaux_tc MoE + shared expert). Uses the shared `args` grammar.
//!
//!   brain glm train <data_dir> --out F [--steps N --batch B --block T --lr X
//!                    --layers L --d-model D --heads H --experts E --size S --seed K]
//!   brain glm infer --weights F [--data <dir>] --prompt "..."
//!                    [--max-new N --temp X --top-k K --seed K --device cpu|gpu]
//!   brain glm eval  --weights F --data <dir> [--batches N --block T --seed K]
//!   brain glm finetune <data_dir> --weights BASE --out F [--steps N --lr X ...]
//!   brain glm import --hf <dir> --out glm.safetensors
//!   brain glm export --weights F --out model.onnx --seq T

use std::path::Path;

use data::binio::Meta;
use data::rng::Rng;
use data::tokenizer::{CharTokenizer, Tokenizer};
use glmdsa::config::GlmConfig;
use glmdsa::model::Glm;

use crate::args::{canon_verb, Args};

pub fn run_glm(argv: &[String]) {
    match argv.first().map(|s| canon_verb(s)) {
        Some("train") => train(&argv[1..], None),
        Some("finetune") => finetune(&argv[1..]),
        Some("infer") => infer(&argv[1..]),
        Some("eval") => eval(&argv[1..]),
        Some("import") => import(&argv[1..]),
        Some("export") => export(&argv[1..]),
        other => eprintln!("usage: brain glm <train|finetune|infer|eval|import|export> ...  (got {other:?})"),
    }
}

/// Named size presets for from-scratch setup (`--size tiny|small|base`).
fn preset(size: &str, block: u32) -> GlmConfig {
    let base = GlmConfig { block_size: block, ..GlmConfig::tiny() };
    match size {
        "tiny" => GlmConfig { vocab: 0, ..base },
        "base" => GlmConfig {
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
        _ => GlmConfig {
            // "small" (default)
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
    }
}

fn train(args: &[String], base: Option<&str>) {
    let mut a = Args::new(args);
    let data_dir = a.positional().unwrap_or_default();
    let out = a.str_or("--out", "out/glm.safetensors");
    let steps = a.u32_or("--steps", 2000);
    let batch = a.u32_or("--batch", 8);
    let block = a.u32_or("--block", 128);
    let lr = a.f32_or("--lr", if base.is_some() { 1e-4 } else { 3e-4 });
    let seed = a.u64_or("--seed", 1234);
    let size = a.str_or("--size", "small");
    let layers = a.opt_u32("--layers");
    let d_model = a.opt_u32("--d-model");
    let heads = a.opt_u32("--heads");
    let experts = a.opt_u32("--experts");
    let mask = a.char_opt("--mask");
    let align = a.take_flag("--align");
    a.finish();
    if data_dir.is_empty() {
        eprintln!("usage: brain glm {{train|finetune}} <data_dir> --out F [--size tiny|small|base --steps N --batch B --block T --lr X --layers L --d-model D --heads H --experts E --mask = --align]");
        return;
    }
    let mut cfg = match base {
        Some(p) => GlmConfig::from_json(&checkpoint::read_config(p)),
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
    let save_secs = 600u64;
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
        checkpoint_secs: save_secs,
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
    let mut a = Args::new(args);
    let Some(base) = a.take_str("--weights") else {
        eprintln!("usage: brain glm finetune <data_dir> --weights BASE --out F [...]");
        return;
    };
    // Rebuild the remaining args (without --weights) for the shared train core.
    let rest: Vec<String> = {
        let mut r = Vec::new();
        let mut i = 0;
        while i < args.len() {
            if args[i] == "--weights" {
                i += 2;
            } else {
                r.push(args[i].clone());
                i += 1;
            }
        }
        r
    };
    train(&rest, Some(&base));
}

fn model_block(weights: &str) -> u32 {
    GlmConfig::from_json(&checkpoint::read_config(weights)).block_size
}

fn infer(args: &[String]) {
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
        eprintln!("usage: brain glm infer --weights F [--data <dir>] [--prompt ... --max-new N --temp X --top-k K]");
        return;
    }
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
    let prompt_text = if prompt.is_empty() { "\n" } else { prompt.as_str() };
    let prompt_ids: Vec<u32> = tok.encode(prompt_text);
    // NPU / OpenVINO whole-graph path (greedy): export -> compile -> decode.
    if crate::npu_explicit() {
        match npu::glm_decode::generate(&weights, &prompt_ids, max_new, npu::openvino::NpuDevice::Npu, true, None, false) {
            Ok(run) => {
                eprintln!("npu: ran on OpenVINO device {} (load_ms={:.1} gen_ms={:.1})", run.device, run.load_ms, run.gen_ms);
                print!("{prompt_text}");
                print!("{}", tok.decode(&run.tokens));
                println!();
            }
            Err(e) => eprintln!("npu infer failed: {e}"),
        }
        return;
    }
    let model = Glm::load_inference(&weights, 1, model_block(&weights));
    let mut rng = Rng::new(seed);
    let gen = glmdsa::sample::generate_kv(&model, &prompt_ids, max_new, temp, top_k, None, &mut rng);
    print!("{prompt_text}");
    print!("{}", tok.decode(&gen));
    println!();
}

/// Validation perplexity: mean masked-CE over sampled val windows -> exp().
fn eval(args: &[String]) {
    let mut a = Args::new(args);
    let weights = a.str_or("--weights", "");
    let data_dir = a.str_or("--data", "");
    let batches = a.usize_or("--batches", 20);
    let block = a.u32_or("--block", 0);
    let seed = a.u64_or("--seed", 99);
    a.finish();
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
    let mut a = Args::new(args);
    let hf = a.str_or("--hf", "");
    let out = a.str_or("--out", "glm.safetensors");
    a.finish();
    if hf.is_empty() {
        eprintln!("usage: brain glm import --hf <dir> --out glm.safetensors");
        return;
    }
    match glmdsa::import::import(&hf, &out) {
        Ok(()) => println!("ok: wrote {out}"),
        Err(e) => eprintln!("import failed: {e}"),
    }
}

/// `brain glm export --weights F --out model.onnx --seq T` — ONNX decoder graph
/// (dense-expert MoE) for OpenVINO / the NPU.
fn export(args: &[String]) {
    let mut a = Args::new(args);
    let weights = a.str_or("--weights", "");
    let out = a.str_or("--out", "glm.onnx");
    let seq = a.usize_or("--seq", 32);
    let int8 = a.take_flag("--int8");
    a.finish();
    if weights.is_empty() {
        eprintln!("usage: brain glm export --weights F --out model.onnx [--seq T --int8]");
        return;
    }
    let res = if int8 {
        npu::glm_export::export_glm_int8(&weights, &out, seq)
    } else {
        npu::glm_export::export_glm_fp32(&weights, &out, seq)
    };
    match res {
        Ok(()) => println!("ok: wrote {out} (seq_len {seq}{})", if int8 { ", int8" } else { "" }),
        Err(e) => eprintln!("export failed: {e}"),
    }
}
