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
    // Canonical verbs shared with `gpt`/`glm` (`gen`->`infer`, `fine-tune`->`finetune`).
    match args.first().map(|s| crate::args::canon_verb(s)) {
        Some("import") => import(&args[1..]),
        Some("infer") => infer(&args[1..]),
        Some("serve") => serve(&args[1..]),
        Some("export") => export(&args[1..]),
        Some("precompile") => precompile(&args[1..]),
        Some("train") => train(&args[1..], None),
        Some("finetune") => finetune(&args[1..]),
        Some("toolcall") => toolcall(&args[1..]),
        other => {
            eprintln!("usage: brain qwen <import|infer|export|precompile|train|finetune|toolcall> ...  (got {other:?})")
        }
    }
}

/// Whether the NPU/OpenVINO path was requested (`--device npu` or BRAIN_DEVICE=npu).
fn want_npu() -> bool {
    crate::npu_requested()
        || std::env::var("BRAIN_DEVICE").map(|v| v.eq_ignore_ascii_case("npu")).unwrap_or(false)
}

/// `brain qwen export --weights F --out model.onnx --seq T` — emit the ONNX
/// decoder graph (for OpenVINO / `brain npu check`).
fn export(args: &[String]) {
    let mut weights = String::new();
    let mut out = "qwen.onnx".to_string();
    let mut seq = 32usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--out" => out = val(args, &mut i, "--out"),
            "--seq" => seq = val(args, &mut i, "--seq").parse().unwrap_or(seq),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() {
        eprintln!("usage: brain qwen export --weights F --out model.onnx [--seq T]");
        return;
    }
    match npu::qwen_export::export_qwen_fp32(&weights, &out, seq) {
        Ok(()) => println!("ok: wrote {out} (seq_len {seq})"),
        Err(e) => eprintln!("export failed: {e}"),
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
    let mut block: Option<u32> = None;
    let mut hf = String::new();
    let mut out = "qwen.weights".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--hf" => hf = val(args, &mut i, "--hf"),
            "--out" => out = val(args, &mut i, "--out"),
            "--block" => block = val(args, &mut i, "--block").parse().ok(),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if hf.is_empty() {
        eprintln!("usage: brain qwen import --hf <dir> --out qwen.weights");
        return;
    }
    match qwen::import::import_with_block(&hf, &out, block) {
        Ok(()) => println!("ok: wrote {out}"),
        Err(e) => eprintln!("import failed: {e}"),
    }
}

/// `brain qwen precompile --weights F --seq T [--npu-cache D]` — export + compile
/// the NPU decoder once into the cache so later `infer --device npu --seq T`
/// (with the same `--npu-cache`) skips the export + compile wait.
fn precompile(args: &[String]) {
    let mut weights = String::new();
    let mut npu_cache = "out/npu-cache".to_string();
    let mut seq = 0usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--npu-cache" => npu_cache = val(args, &mut i, "--npu-cache"),
            "--seq" => seq = val(args, &mut i, "--seq").parse().unwrap_or(seq),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() || seq == 0 {
        eprintln!("usage: brain qwen precompile --weights F --seq T [--npu-cache out/npu-cache]");
        return;
    }
    match npu::qwen_decode::precompile(&weights, seq, npu::openvino::NpuDevice::Npu, true, std::path::Path::new(&npu_cache)) {
        Ok((dev, ms)) => println!("precompiled seq {seq} on OpenVINO {dev} in {:.1}s -> cache {npu_cache}", ms / 1e3),
        Err(e) => eprintln!("precompile failed: {e}"),
    }
}

/// `brain qwen serve` — continuous-batching serving: submit several prompts and
/// decode them concurrently through the paged Scheduler (one batched forward per
/// iteration over all running sequences).
fn serve(args: &[String]) {
    let mut weights = String::new();
    let mut tokenizer = String::new();
    let mut prompts: Vec<String> = Vec::new();
    let mut max_new = 64usize;
    let mut block_size = 16u32;
    let mut int8 = false;
    let mut weights_int8 = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--tokenizer" => tokenizer = val(args, &mut i, "--tokenizer"),
            "--prompt" => prompts.push(val(args, &mut i, "--prompt")),
            "--max-new" => max_new = val(args, &mut i, "--max-new").parse().unwrap_or(max_new),
            "--block-size" => block_size = val(args, &mut i, "--block-size").parse().unwrap_or(block_size),
            "--int8" => int8 = true,
            "--weights-int8" => weights_int8 = true,
            other => eprintln!("serve: ignoring {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() || tokenizer.is_empty() || prompts.is_empty() {
        eprintln!("usage: brain qwen serve --weights F --tokenizer T --prompt \"...\" [--prompt \"...\"]... [--max-new N --block-size B --int8 --weights-int8]");
        return;
    }
    let tok = match data::qwen_tokenizer::QwenBpe::from_file(&tokenizer) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer load failed: {e}");
            return;
        }
    };
    let eos = tok.encode("<|im_end|>").first().copied();
    let toks: Vec<Vec<u32>> = prompts.iter().map(|p| tok.encode(&tok.apply_chat_template(&[("user", p)], true))).collect();
    let n = prompts.len() as u32;
    let max_prompt = toks.iter().map(|t| t.len()).max().unwrap_or(1) as u32;
    let max_len = max_prompt + max_new as u32 + 8;
    let blocks_per_seq = max_len.div_ceil(block_size);
    let num_blocks = blocks_per_seq * n + n; // headroom for every sequence
    let eng = qwen::serve::Engine::load(&weights, block_size, num_blocks, n, blocks_per_seq, max_prompt.max(1), int8, weights_int8);
    // Report what actually ran: the int8-weights request is capability-gated
    // and may have fallen back to fp32.
    let weights_int8 = eng.weights_int8();
    let mut sched = qwen::serve::Scheduler::new(eng, n as usize);
    let ids: Vec<u64> = toks.iter().map(|t| sched.submit(qwen::serve::Request { prompt: t.clone(), max_new, eos })).collect();

    let t0 = std::time::Instant::now();
    let out = sched.run();
    let secs = t0.elapsed().as_secs_f64();
    let total: usize = out.values().map(|v| v.len()).sum();
    for (i, id) in ids.iter().enumerate() {
        println!("=== prompt {i}: {:?}", prompts[i]);
        println!("{}\n", tok.decode(&out[id]));
    }
    eprintln!(
        "served {n} prompts, {total} tokens in {secs:.2}s ({:.1} tok/s aggregate){}{}",
        total as f64 / secs,
        if int8 { " [int8 KV]" } else { "" },
        if weights_int8 { " [int8 weights]" } else { "" }
    );
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
    let mut npu_cache = "out/npu-cache".to_string();
    let mut seq: Option<usize> = None;
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
            // NPU: cache the exported ONNX + compiled blob here; "" disables.
            "--npu-cache" => npu_cache = val(args, &mut i, "--npu-cache"),
            // NPU: pin the compiled context length so one cache serves all prompts.
            "--seq" => seq = val(args, &mut i, "--seq").parse().ok(),
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
    // NPU / OpenVINO whole-graph path (greedy only): export -> compile -> decode.
    if want_npu() {
        let cache = (!npu_cache.is_empty()).then(|| std::path::PathBuf::from(&npu_cache));
        match npu::qwen_decode::generate(&weights, &ids, max_new, npu::openvino::NpuDevice::Npu, true, cache.as_deref(), seq) {
            Ok(run) => {
                eprintln!("npu: ran on OpenVINO device {} (onnx_cached={})", run.device, run.onnx_cached);
                eprintln!("qwen-timing load_ms={:.1} gen_ms={:.1} tokens={}", run.load_ms, run.gen_ms, run.tokens.len());
                print!("{prompt}");
                print!("{}", tok.decode(&run.tokens));
                println!();
            }
            Err(e) => eprintln!("npu infer failed: {e}"),
        }
        return;
    }
    let cap = (ids.len() + max_new) as u32;
    let t_load = std::time::Instant::now();
    let model = Qwen::load_inference(&weights, 1, cap);
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
    let eos = tok.encode("<|im_end|>").first().copied();
    let mut rng = Rng::new(seed);
    let t_gen = std::time::Instant::now();
    let gen = qwen::sample::generate_kv(&model, &ids, max_new, temp, top_k, eos, &mut rng);
    let gen_ms = t_gen.elapsed().as_secs_f64() * 1e3;
    eprintln!("qwen-timing load_ms={load_ms:.1} gen_ms={gen_ms:.1} tokens={}", gen.len());
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
    let mut save_secs = 600u64;
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
            "--save-secs" => save_secs = val(args, &mut i, "--save-secs").parse().unwrap_or(save_secs),
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
        checkpoint_secs: save_secs,
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


/// `brain qwen toolcall gen  --out DIR [--n N --held M --tools K --vocab V --tokenizer T]`
///   Write a masked tool-call fine-tuning dataset (train + held-out val).
/// `brain qwen toolcall eval --weights F --tokenizer T [--n N --tools K --seq S]`
///   Score a checkpoint's held-out tool-call exact-match (teacher-forced greedy).
fn toolcall(args: &[String]) {
    match args.first().map(|s| s.as_str()) {
        Some("gen") => toolcall_gen(&args[1..]),
        Some("eval") => toolcall_eval(&args[1..]),
        other => eprintln!("usage: brain qwen toolcall <gen|eval> ...  (got {other:?})"),
    }
}

fn toolcall_gen(args: &[String]) {
    let mut out = "data/toolcall".to_string();
    let mut tokenizer = String::new();
    let (mut n, mut held, mut tools, mut vocab, mut seed) = (400usize, 40usize, 3usize, 151936usize, 1u64);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => out = val(args, &mut i, "--out"),
            "--tokenizer" => tokenizer = val(args, &mut i, "--tokenizer"),
            "--n" => n = val(args, &mut i, "--n").parse().unwrap_or(n),
            "--held" => held = val(args, &mut i, "--held").parse().unwrap_or(held),
            "--tools" => tools = val(args, &mut i, "--tools").parse().unwrap_or(tools),
            "--vocab" => vocab = val(args, &mut i, "--vocab").parse().unwrap_or(vocab),
            "--seed" => seed = val(args, &mut i, "--seed").parse().unwrap_or(seed),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if tokenizer.is_empty() {
        eprintln!("usage: brain qwen toolcall gen --tokenizer tokenizer.json --out DIR [--n --held --tools --vocab --seed]");
        return;
    }
    let tok = match data::qwen_tokenizer::QwenBpe::from_file(&tokenizer) {
        Ok(t) => t,
        Err(e) => { eprintln!("tokenizer: {e}"); return; }
    };
    let train: Vec<_> = data::toolcall::generate(n, tools, seed).iter().map(|c| c.to_chat_example()).collect();
    let val: Vec<_> = data::toolcall::generate(held, tools, seed ^ 0xDEAD).iter().map(|c| c.to_chat_example()).collect();
    match data::chat::prepare_chat(&train, &val, &tok, vocab, std::path::Path::new(&out)) {
        Ok(()) => println!("ok: wrote {n} train + {held} val tool-call examples -> {out}"),
        Err(e) => eprintln!("prepare failed: {e}"),
    }
}

fn toolcall_eval(args: &[String]) {
    let mut weights = String::new();
    let mut tokenizer = String::new();
    let (mut n, mut tools, mut seq, mut seed) = (40usize, 3usize, 512usize, 999u64);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--tokenizer" => tokenizer = val(args, &mut i, "--tokenizer"),
            "--n" => n = val(args, &mut i, "--n").parse().unwrap_or(n),
            "--tools" => tools = val(args, &mut i, "--tools").parse().unwrap_or(tools),
            "--seq" => seq = val(args, &mut i, "--seq").parse().unwrap_or(seq),
            "--seed" => seed = val(args, &mut i, "--seed").parse().unwrap_or(seed),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() || tokenizer.is_empty() {
        eprintln!("usage: brain qwen toolcall eval --weights F --tokenizer T [--n --tools --seq --seed]");
        return;
    }
    let tok = match data::qwen_tokenizer::QwenBpe::from_file(&tokenizer) {
        Ok(t) => t, Err(e) => { eprintln!("tokenizer: {e}"); return; }
    };
    let (exact, tacc) = qwen::toolcall_eval::score(&weights, &tok, n, tools, seq, seed);
    println!("tool-call eval: exact-match {:.1}%  token-acc {:.1}%  ({n} held-out cases)", exact * 100.0, tacc * 100.0);
}
