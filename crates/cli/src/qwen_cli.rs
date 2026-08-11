// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain qwen …` — import / run / fine-tune the Qwen3 decoder.
//!
//!   brain qwen import --hf <dir> --out qwen.safetensors
//!   brain qwen infer  --weights F --tokenizer tokenizer.json --prompt "..."
//!                     [--max-new N --temp X --top-k K --chat --device cpu|gpu]
//!   brain qwen train    <data_dir> --out F [--steps N --batch B --block T --lr X ...]
//!   brain qwen finetune <data_dir> --weights BASE --out F [--steps N --lr X ...]
//!   brain qwen finetune --lora RANK --weights BASE --adapter OWNER/NAME[:TAG] --dataset DIR
//!                     [--alpha A --steps N --lr X --batch B --block T --seed S
//!                      --models-dir DIR --dataset-id ID]
//!   brain qwen eval --weights BASE --jsonl FILE [--adapter OWNER/NAME[:TAG]
//!                     --block T --models-dir DIR]
//!   brain qwen calib --weights BASE --jsonl FILE [--report --out kv_calib.json
//!                     --models-dir DIR]

use std::path::{Path, PathBuf};

use data::rng::Rng;
use data::tokenizer::Tokenizer;
use qwen3::config::QwenConfig;
use qwen3::model::Qwen;

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
        Some("eval") => eval_chat(&args[1..]),
        Some("calib") => calib(&args[1..]),
        other => {
            eprintln!("usage: brain qwen <import|infer|export|precompile|train|finetune|toolcall|eval|calib> ...  (got {other:?})")
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
    let mut out = "qwen.safetensors".to_string();
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
        eprintln!("usage: brain qwen import --hf <dir> --out qwen.safetensors");
        return;
    }
    match qwen3::import::import_with_block(&hf, &out, block) {
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
    // int8 KV default ON, matching the production resident's default
    // (resident_llm.rs::QwenResident::kv_int8) -- measured +0.0154 loss vs
    // fp32 on the real Qwen3-0.6B checkpoint, close enough to free that the
    // memory win is the clear default.
    // --int8 is kept as a harmless no-op (already the default) for anyone
    // with it in a script; --kv-fp32 is the real opt-out.
    let mut int8 = true;
    let mut weights_int8 = false;
    let mut kv_calib_opt_in = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--tokenizer" => tokenizer = val(args, &mut i, "--tokenizer"),
            "--prompt" => prompts.push(val(args, &mut i, "--prompt")),
            "--max-new" => max_new = val(args, &mut i, "--max-new").parse().unwrap_or(max_new),
            "--block-size" => block_size = val(args, &mut i, "--block-size").parse().unwrap_or(block_size),
            "--int8" => int8 = true,
            "--kv-fp32" => int8 = false,
            "--weights-int8" => weights_int8 = true,
            "--kv-calib" => kv_calib_opt_in = true,
            other => eprintln!("serve: ignoring {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() || tokenizer.is_empty() || prompts.is_empty() {
        eprintln!("usage: brain qwen serve --weights F --tokenizer T --prompt \"...\" [--prompt \"...\"]... [--max-new N --block-size B --kv-fp32 --weights-int8 --kv-calib]");
        eprintln!("       (int8 KV is on by default; --kv-fp32 opts out. --int8 is accepted as a no-op.");
        eprintln!("        --kv-calib opts INTO a kv_calib.json beside --weights, if one exists there.)");
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
    // A header-only peek (not a second full checkpoint load), shared by the
    // int8-KV degrade check below and the --kv-calib lookup further down.
    let header_cfg = checkpoint::weightio::WeightReader::open(&weights).ok().map(|r| QwenConfig::from_json(&r.config()));
    // `--int8` requesting the DEFAULT degrades loudly on an unsupported
    // head_dim rather than hitting `from_map_with_gpu`'s hard assert -- an
    // explicit `--int8` on the command line is, deliberately, still a
    // request by name and would panic instead; this branch only ever
    // softens the DEFAULT. See `qwen3::serve::kv_int8_supported`'s doc
    // comment.
    let int8 = int8
        && match &header_cfg {
            Some(cfg) => {
                let supported = qwen3::serve::kv_int8_supported(cfg);
                if !supported {
                    eprintln!("serve: {weights}: int8 KV requested (the default) but head_dim is not a multiple of 4; falling back to fp32 KV");
                }
                supported
            }
            None => true, // let Engine::load raise the real, specific I/O error below
        };
    let mut eng = qwen3::serve::Engine::load(&weights, block_size, num_blocks, n, blocks_per_seq, max_prompt.max(1), int8, weights_int8);
    // --kv-calib: opt IN to a kv_calib.json beside the checkpoint, if the
    // engine is int8 and one exists there with a matching shape --
    // KvCalib::from_model_dir already warns and returns None on a missing
    // file or a shape mismatch, so a caller that opts in without a real
    // file just serves uncalibrated, same as not opting in.
    if kv_calib_opt_in {
        if int8 {
            if let (Some(cfg), Some(dir)) = (&header_cfg, std::path::Path::new(&weights).parent()) {
                let calib = model::kvcalib::KvCalib::from_model_dir(dir, cfg.n_layers as usize, cfg.n_kv_heads as usize, cfg.head_dim as usize);
                eng.set_kv_calib(calib);
            }
        } else {
            eprintln!("serve: --kv-calib requested but the engine is fp32-KV (calibration only applies to int8 KV); ignoring");
        }
    }
    // Report what actually ran: the int8-weights request is capability-gated
    // and may have fallen back to fp32; --kv-calib may have found nothing.
    let weights_int8 = eng.weights_int8();
    let kv_calibrated = eng.kv_calibrated();
    let mut sched = qwen3::serve::Scheduler::new(eng, n as usize);
    let ids: Vec<u64> = toks.iter().map(|t| sched.submit(qwen3::serve::Request { prompt: t.clone(), max_new, eos })).collect();

    let t0 = std::time::Instant::now();
    let out = sched.run();
    let secs = t0.elapsed().as_secs_f64();
    let total: usize = out.values().map(|v| v.len()).sum();
    for (i, id) in ids.iter().enumerate() {
        println!("=== prompt {i}: {:?}", prompts[i]);
        println!("{}\n", tok.decode(&out[id]));
    }
    eprintln!(
        "served {n} prompts, {total} tokens in {secs:.2}s ({:.1} tok/s aggregate){}{}{}",
        total as f64 / secs,
        if int8 { " [int8 KV]" } else { " [fp32 KV]" },
        if kv_calibrated { " [calibrated]" } else { "" },
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
    let gen = qwen3::sample::generate_kv(&model, &ids, max_new, temp, top_k, 1.0, eos, &mut rng);
    let gen_ms = t_gen.elapsed().as_secs_f64() * 1e3;
    eprintln!("qwen-timing load_ms={load_ms:.1} gen_ms={gen_ms:.1} tokens={}", gen.len());
    print!("{prompt}");
    print!("{}", tok.decode(&gen));
    println!();
}

/// Shared core for `train` (fresh) and `finetune` (seeded from `--weights`).
fn train(args: &[String], base: Option<&str>) {
    let mut data_dir = String::new();
    let mut out = "out/qwen.safetensors".to_string();
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
            max_position_embeddings: block,
            tie_embeddings: true,
            qk_norm: true,
            attn_bias: false,
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
    // `--lora RANK` selects the named-adapter path (finetune_lora) over the
    // legacy full-parameter path below -- a distinct function, not a branch
    // threaded through `train`/`model::fit`, because qwen3::finetune::finetune
    // is a self-contained training loop for exactly this reason (seeding a
    // LoRA-extended param set from a base checkpoint is something
    // `model::fit`'s checkpoint-config-wins resume path cannot do; see that
    // module's own doc comment).
    if args.iter().any(|a| a == "--lora") {
        return finetune_lora(args);
    }
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

/// Resolve `--weights` to `(weights_file, its_directory, canonical_base_id)`
/// -- either a `vendor/repo[-QUANT]` model-store reference, or a direct
/// filesystem path to a `.safetensors` file (its sibling directory then
/// supplies `tokenizer.json`/`tokenizer_config.json`). Filesystem existence
/// is checked FIRST: a relative path like `out/qwen.safetensors` also parses
/// as a syntactically valid (if unlikely) `ModelRef`, so "is this a real
/// file" must win before "is this a ref" is even considered.
fn resolve_base(base: &str, store_root: Option<&Path>) -> Result<(PathBuf, PathBuf, String), String> {
    let path = Path::new(base);
    if path.is_file() {
        let dir = path.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
        // Synthesize under the "local" reserved vendor (`brain_modelref::is_reserved`)
        // so the id is always a valid `vendor/repo` -- required a moment later to
        // build `vendor/repo:owner:name:tag` for the adapter ref, which a bare
        // filename (no '/') could never parse as.
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("base");
        let id = format!("local/{stem}");
        return Ok((path.to_path_buf(), dir, id));
    }
    let r = brain_modelref::ModelRef::parse(base).map_err(|e| format!("{base}: not a file, and not a valid model ref ({e})"))?;
    let root = store_root.ok_or_else(|| "no models directory resolved (set --models-dir, BRAIN_MODELS_DIR, or HOME)".to_string())?;
    let store = brain_modelstore::Store::new(root);
    let local = store.local(&r).ok_or_else(|| format!("{base}: not found in the model store at {}", root.display()))?;
    Ok((local.weights, local.dir, r.to_string()))
}

/// `OWNER/NAME[:TAG]` -> `(owner, name, tag)`, `tag` defaulting to `"latest"`
/// (Docker-style mutable tags -- retraining with the same `--adapter`
/// overwrites that tag; a content-addressed tag scheme beyond `latest` is
/// deliberately out of scope here).
fn parse_adapter_spec(spec: &str) -> Result<(String, String, String), String> {
    let (owner_name, tag) = match spec.split_once(':') {
        Some((a, b)) => (a, b.to_string()),
        None => (spec, "latest".to_string()),
    };
    let (owner, name) = owner_name.split_once('/').ok_or_else(|| "expected \"OWNER/NAME[:TAG]\"".to_string())?;
    if owner.is_empty() || name.is_empty() || tag.is_empty() {
        return Err("OWNER, NAME, and TAG must all be non-empty".to_string());
    }
    Ok((owner.to_string(), name.to_string(), tag))
}

/// `brain qwen finetune --lora RANK --weights BASE --adapter OWNER/NAME[:TAG]
///     --dataset DIR [--alpha A --steps N --lr X --batch B --block T --seed S
///     --models-dir DIR --dataset-id ID]`
///
/// Trains a NAMED LoRA adapter and saves ONLY the adapter tensors into the
/// model store at `<models-dir>/<vendor>/<repo>/adapters/<owner>/<name>/<tag>/`
/// -- retraining with the same `--adapter` always overwrites that tag. The
/// full "retrain and overwrite" command bench's exported dataset needs:
/// `--dataset DIR` is a bench `datasets build` output directory
/// (`train.jsonl`, optionally `validation.jsonl`), rendered through the
/// BASE checkpoint's own chat template (`data::chat_template`, from its
/// `tokenizer_config.json` -- already on disk for any model fetched through
/// the normal store/plan path, no extra step).
fn finetune_lora(args: &[String]) {
    let mut base = String::new();
    let mut adapter_spec = String::new();
    let mut dataset_dir = String::new();
    let mut rank = 0u32;
    let mut alpha: Option<f32> = None;
    let mut steps = 500u32;
    let mut lr = 5e-5f32;
    let mut batch = 4u32;
    let mut block = 1024u32;
    let mut seed = 1234u64;
    let mut models_dir: Option<String> = None;
    let mut dataset_id: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => base = val(args, &mut i, "--weights"),
            "--adapter" => adapter_spec = val(args, &mut i, "--adapter"),
            "--dataset" => dataset_dir = val(args, &mut i, "--dataset"),
            "--lora" => rank = val(args, &mut i, "--lora").parse().unwrap_or(rank),
            "--alpha" => alpha = val(args, &mut i, "--alpha").parse().ok(),
            "--steps" => steps = val(args, &mut i, "--steps").parse().unwrap_or(steps),
            "--lr" => lr = val(args, &mut i, "--lr").parse().unwrap_or(lr),
            "--batch" => batch = val(args, &mut i, "--batch").parse().unwrap_or(batch),
            "--block" => block = val(args, &mut i, "--block").parse().unwrap_or(block),
            "--seed" => seed = val(args, &mut i, "--seed").parse().unwrap_or(seed),
            "--models-dir" => models_dir = Some(val(args, &mut i, "--models-dir")),
            "--dataset-id" => dataset_id = Some(val(args, &mut i, "--dataset-id")),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if base.is_empty() || adapter_spec.is_empty() || dataset_dir.is_empty() {
        eprintln!(
            "usage: brain qwen finetune --lora RANK --weights BASE --adapter OWNER/NAME[:TAG] --dataset DIR \
             [--alpha A --steps N --lr X --batch B --block T --seed S --models-dir DIR --dataset-id ID]"
        );
        return;
    }
    if rank == 0 {
        eprintln!("--lora RANK must be > 0");
        return;
    }
    let alpha = alpha.unwrap_or(rank as f32 * 2.0);

    let store_root = crate::model_dir::resolve(models_dir.as_deref());
    let (base_weights_path, base_dir, base_id) = match resolve_base(&base, store_root.as_deref()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };
    let (owner, name, tag) = match parse_adapter_spec(&adapter_spec) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("--adapter {adapter_spec:?}: {e}");
            return;
        }
    };
    let full_ref_str = format!("{base_id}:{owner}:{name}:{tag}");
    let adapter_ref = match brain_modelref::ModelRef::parse(&full_ref_str) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{full_ref_str}: {e}");
            return;
        }
    };
    let Some(store_root) = store_root else {
        eprintln!("no models directory resolved (set --models-dir, BRAIN_MODELS_DIR, or HOME)");
        return;
    };
    let store = brain_modelstore::Store::new(&store_root);
    let Some(adapter_out_path) = store.adapter_weights_path(&adapter_ref) else {
        eprintln!("{adapter_ref}: not an adapter reference (bug: parsed above with an adapter suffix)");
        return;
    };

    let dataset_dir = Path::new(&dataset_dir);
    let train_samples = match data::chat::ChatSample::from_jsonl(&dataset_dir.join("train.jsonl")) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };
    if train_samples.is_empty() {
        eprintln!("{}: no training samples", dataset_dir.join("train.jsonl").display());
        return;
    }
    let val_path = dataset_dir.join("validation.jsonl");
    let val_samples = if val_path.is_file() {
        match data::chat::ChatSample::from_jsonl(&val_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{e}");
                return;
            }
        }
    } else {
        eprintln!("note: no validation.jsonl in {} -- training with no held-out eval", dataset_dir.display());
        Vec::new()
    };

    let tokenizer_path = base_dir.join("tokenizer.json");
    let tok = match data::qwen_tokenizer::QwenBpe::from_file(tokenizer_path.to_str().unwrap_or_default()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{}: {e}", tokenizer_path.display());
            return;
        }
    };
    let chat_template = match data::chat_template::ChatTemplate::from_model_dir(&base_dir) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };

    let base_cfg_json = checkpoint::load(base_weights_path.to_str().unwrap_or_default()).header["config"].clone();
    let vocab = base_cfg_json["vocab_size"].as_u64().unwrap_or(0) as usize;
    if vocab == 0 {
        eprintln!("{}: could not read vocab_size from the base checkpoint's config", base_weights_path.display());
        return;
    }

    let scratch = std::env::temp_dir().join(format!("brain-qwen-lora-train-{}", std::process::id()));
    if let Err(e) = data::chat::prepare_chat_samples(&train_samples, &val_samples, &tok, &chat_template, vocab, &scratch) {
        eprintln!("preparing training data: {e}");
        return;
    }

    println!(
        "training LoRA adapter {full_ref_str} (rank={rank} alpha={alpha}) on {} train / {} val samples, {steps} steps...",
        train_samples.len(),
        val_samples.len()
    );
    let opts = model::FitOpts {
        steps,
        batch_size: batch,
        block_size: block,
        lr,
        min_lr: lr * 0.1,
        warmup: (steps / 20).max(1),
        decay_iters: steps,
        weight_decay: 0.1,
        grad_clip: 1.0,
        grad_accum: 1,
        eval_interval: if val_samples.is_empty() { 0 } else { (steps / 10).max(1) },
        eval_batches: 20,
        checkpoint_secs: 0,
        mask_before: None,
        mask_per_line: false,
        align_to_lines: false,
        seed,
    };
    let mode = qwen3::finetune::Mode::Lora { rank, alpha };
    let full_ckpt_out = scratch.join("full.safetensors");
    let (l0, l1) = match qwen3::finetune::finetune(
        base_weights_path.to_str().unwrap_or_default(),
        &scratch,
        &opts,
        &mode,
        full_ckpt_out.to_str().unwrap_or_default(),
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("finetune error: {e}");
            return;
        }
    };
    println!("trained: loss {l0:.4} -> {l1:.4}");

    let reloaded = Qwen::load_inference(full_ckpt_out.to_str().unwrap_or_default(), 1, block);
    if let Err(e) = qwen3::lora::save_adapter(adapter_out_path.to_str().unwrap_or_default(), &reloaded, &full_ref_str, &base_id, dataset_id.as_deref())
    {
        eprintln!("save_adapter: {e}");
        return;
    }
    let _ = std::fs::remove_dir_all(&scratch);
    println!("saved: {full_ref_str} -> {}", adapter_out_path.display());
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
    let (exact, tacc) = qwen3::toolcall_eval::score(&weights, &tok, n, tools, seq, seed);
    println!("tool-call eval: exact-match {:.1}%  token-acc {:.1}%  ({n} held-out cases)", exact * 100.0, tacc * 100.0);
}

/// `brain qwen eval --weights BASE --jsonl FILE [--adapter OWNER/NAME[:TAG]
///     --block T --models-dir DIR]`
///
/// Gate B of the Definition of Done's "a way to validate that model has
/// learned ideas from the dataset": teacher-forced held-out loss + token
/// accuracy (`qwen3::eval::score_chat`) against a REAL checkpoint and a REAL
/// bench-exported `validation.jsonl`/`test.jsonl`. Always reports the base
/// score; with `--adapter` also folds that named adapter in (the same fold
/// a resident uses to serve it -- see `qwen3::lora::fold_adapter_into`) and
/// reports it side by side, so the honest question -- does this adapter
/// actually help on data it never trained on -- has a number attached, not
/// just a training-loss curve.
fn eval_chat(args: &[String]) {
    let mut base = String::new();
    let mut jsonl = String::new();
    let mut adapter_spec: Option<String> = None;
    let mut block = 1024u32;
    let mut models_dir: Option<String> = None;
    let mut kv_modes: Vec<qwen3::eval::KvMode> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => base = val(args, &mut i, "--weights"),
            "--jsonl" => jsonl = val(args, &mut i, "--jsonl"),
            "--adapter" => adapter_spec = Some(val(args, &mut i, "--adapter")),
            "--block" => block = val(args, &mut i, "--block").parse().unwrap_or(block),
            "--models-dir" => models_dir = Some(val(args, &mut i, "--models-dir")),
            "--kv" => {
                let spec = val(args, &mut i, "--kv");
                for m in spec.split(',') {
                    match qwen3::eval::KvMode::parse(m.trim()) {
                        Ok(mode) => kv_modes.push(mode),
                        Err(e) => {
                            eprintln!("--kv: {e}");
                            return;
                        }
                    }
                }
            }
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if base.is_empty() || jsonl.is_empty() {
        eprintln!("usage: brain qwen eval --weights BASE --jsonl FILE [--adapter OWNER/NAME[:TAG] --block T --models-dir DIR] [--kv fp32,int8,int8-calib]");
        return;
    }

    let store_root = crate::model_dir::resolve(models_dir.as_deref());
    let (base_weights_path, base_dir, base_id) = match resolve_base(&base, store_root.as_deref()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };

    let samples = match data::chat::ChatSample::from_jsonl(Path::new(&jsonl)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{jsonl}: {e}");
            return;
        }
    };
    if samples.is_empty() {
        eprintln!("{jsonl}: no samples");
        return;
    }

    let tokenizer_path = base_dir.join("tokenizer.json");
    let tok = match data::qwen_tokenizer::QwenBpe::from_file(tokenizer_path.to_str().unwrap_or_default()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{}: {e}", tokenizer_path.display());
            return;
        }
    };
    let chat_template = match data::chat_template::ChatTemplate::from_model_dir(&base_dir) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };

    let base_weights_str = base_weights_path.to_str().unwrap_or_default();
    let base_score = qwen3::eval::score_chat(base_weights_str, None, &tok, &chat_template, &samples, block);
    println!(
        "base {base_id}: loss {:.4}  token-acc {:.1}%  ({}/{} samples scored, {} positions)",
        base_score.loss,
        base_score.token_accuracy * 100.0,
        base_score.samples,
        base_score.samples + base_score.skipped,
        base_score.positions
    );

    // --kv scores through the paged serving engine (qwen3::serve::Engine) --
    // the actual engine `brain serve` runs -- at each requested KV
    // representation, side by side. fp32 is included in the requested list
    // itself if the caller wants the paged-vs-legacy cross-check; it is NOT
    // implied automatically, so this stays a strict opt-in addition to the
    // unconditional base_score line above.
    for kv in &kv_modes {
        let score = qwen3::eval::score_chat_paged(base_weights_str, None, &tok, &chat_template, &samples, block, *kv);
        let delta = if base_score.loss.is_finite() { score.loss - base_score.loss } else { f32::NAN };
        println!(
            "base {base_id} [kv={}]: loss {:.4} ({delta:+.4} vs legacy fp32)  token-acc {:.1}%  ({}/{} samples scored, {} positions)",
            kv.label(),
            score.loss,
            score.token_accuracy * 100.0,
            score.samples,
            score.samples + score.skipped,
            score.positions
        );
    }

    let Some(spec) = adapter_spec else { return };
    let (owner, name, tag) = match parse_adapter_spec(&spec) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("--adapter {spec:?}: {e}");
            return;
        }
    };
    let full_ref_str = format!("{base_id}:{owner}:{name}:{tag}");
    let adapter_ref = match brain_modelref::ModelRef::parse(&full_ref_str) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{full_ref_str}: {e}");
            return;
        }
    };
    let Some(store_root) = store_root else {
        eprintln!("no models directory resolved (set --models-dir, BRAIN_MODELS_DIR, or HOME)");
        return;
    };
    let store = brain_modelstore::Store::new(&store_root);
    let Some(local) = store.local(&adapter_ref) else {
        eprintln!("{full_ref_str}: not found in the model store at {}", store_root.display());
        return;
    };
    let Some(adapter_path) = local.adapter else {
        eprintln!("{full_ref_str}: resolved but carries no adapter file (bug: parsed with an adapter suffix)");
        return;
    };
    let adapter_score = qwen3::eval::score_chat(base_weights_str, Some(adapter_path.to_str().unwrap_or_default()), &tok, &chat_template, &samples, block);
    println!(
        "{full_ref_str}: loss {:.4}  token-acc {:.1}%  ({}/{} samples scored, {} positions)",
        adapter_score.loss,
        adapter_score.token_accuracy * 100.0,
        adapter_score.samples,
        adapter_score.samples + adapter_score.skipped,
        adapter_score.positions
    );
    let verdict = if adapter_score.loss < base_score.loss { "beats" } else { "does NOT beat" };
    println!("{full_ref_str} {verdict} base on held-out loss ({:.4} vs {:.4})", adapter_score.loss, base_score.loss);
}

/// `brain qwen calib --weights BASE --jsonl PROMPTS [--report] [--out kv_calib.json]
///     [--clip-out kv_calib.json --percentile Q] [--models-dir DIR]`
///
/// The design input for a calibrated INT8 KV scale: runs `PROMPTS` through a
/// real checkpoint's fp32-KV paged engine and reports, per `(layer, K|V,
/// kv_head)`, `absmax` / `p99` / `p99.99` / `outlier_ratio` — most
/// quantization-hostile first (`qwen3::serve::Engine::calibrate_kv`,
/// `model::actstats`). `outlier_ratio` near 1 means today's per-token online
/// absmax is already close to optimal for that stream; a large ratio means a
/// rare-token outlier is setting the scale and crushing the resolution of
/// everything else.
///
/// `--clip-out FILE` (with `--percentile Q`, default `0.999`) additionally
/// writes the actual usable `KvCalib` clip table (`model::kvcalib`, shared
/// with any future paged-attention model, not Qwen-specific) — save it as
/// `kv_calib.json` next to the checkpoint and pass `--kv-calib` to `brain
/// qwen serve` (`BRAIN_QWEN_KV_CALIB=1` for `brain serve`'s resident) to opt
/// IN to it (`KvCalib::from_model_dir`); calibration is NOT picked up
/// automatically -- measurement found that
/// a small (10-prompt) calibration set makes things WORSE, so the serving
/// default stays plain online-absmax and a caller must explicitly ask for a
/// specific calibration file. This is a SEPARATE artifact from `--out`'s raw
/// diagnostic report: `--out` dumps `absmax`/`p99`/`p99.99` for humans to
/// read, `--clip-out` writes the `[layer][kv_head]` ceiling table the engine
/// actually loads.
fn calib(args: &[String]) {
    let mut weights = String::new();
    let mut jsonl = String::new();
    let mut out: Option<String> = None;
    let mut clip_out: Option<String> = None;
    let mut percentile = 0.999f32;
    let mut report = false;
    let mut models_dir: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--jsonl" => jsonl = val(args, &mut i, "--jsonl"),
            "--out" => out = Some(val(args, &mut i, "--out")),
            "--clip-out" => clip_out = Some(val(args, &mut i, "--clip-out")),
            "--percentile" => percentile = val(args, &mut i, "--percentile").parse().unwrap_or(percentile),
            "--report" => report = true,
            "--models-dir" => models_dir = Some(val(args, &mut i, "--models-dir")),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() || jsonl.is_empty() {
        eprintln!("usage: brain qwen calib --weights BASE --jsonl FILE [--report] [--out kv_calib_report.json] [--clip-out kv_calib.json --percentile Q] [--models-dir DIR]");
        return;
    }
    if !report && out.is_none() && clip_out.is_none() {
        eprintln!("brain qwen calib: nothing to do without --report, --out, or --clip-out");
        return;
    }
    if !(0.0..=1.0).contains(&percentile) {
        eprintln!("brain qwen calib: --percentile must be in [0,1], got {percentile}");
        return;
    }

    let store_root = crate::model_dir::resolve(models_dir.as_deref());
    let (weights_path, base_dir, base_id) = match resolve_base(&weights, store_root.as_deref()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };

    let samples = match data::chat::ChatSample::from_jsonl(Path::new(&jsonl)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{jsonl}: {e}");
            return;
        }
    };
    if samples.is_empty() {
        eprintln!("{jsonl}: no samples");
        return;
    }

    let tokenizer_path = base_dir.join("tokenizer.json");
    let tok = match data::qwen_tokenizer::QwenBpe::from_file(tokenizer_path.to_str().unwrap_or_default()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{}: {e}", tokenizer_path.display());
            return;
        }
    };
    let chat_template = match data::chat_template::ChatTemplate::from_model_dir(&base_dir) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };

    let total = samples.len();
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    // Keep the first few DISTINCT encode errors: the template engine's
    // carefully-worded diagnostics ("not prefix-stable … refusing to guess a
    // loss-mask boundary") used to be swallowed here — only a bare skip count
    // survived, so the user whose data was dropped never learned why.
    let mut encode_errors: Vec<String> = Vec::new();
    for s in &samples {
        match s.encode(&tok, &chat_template) {
            Ok((ids, _mask)) if !ids.is_empty() => prompts.push(ids),
            Ok(_) => {}
            Err(e) => {
                let e = e.to_string();
                if encode_errors.len() < 3 && !encode_errors.contains(&e) {
                    encode_errors.push(e);
                }
            }
        }
    }
    let report_errors = |errs: &[String]| {
        for e in errs {
            eprintln!("brain qwen calib:   encode error: {e}");
        }
    };
    if prompts.is_empty() {
        eprintln!("{jsonl}: no sample encoded to a usable prompt");
        report_errors(&encode_errors);
        return;
    }
    let skipped = total - prompts.len();
    if skipped > 0 {
        eprintln!("brain qwen calib: skipped {skipped}/{total} samples that failed to encode (first distinct errors below)");
        report_errors(&encode_errors);
    }

    // Every prompt's table must stay resident simultaneously (calibrate_kv
    // reads the pool back once per layer, after every prefill) -- size the
    // pool for the sum, not the max, of every prompt's blocks.
    let block_size = 16u32;
    let max_len = prompts.iter().map(|p| p.len()).max().unwrap_or(1) as u32;
    let max_blocks_per_seq = max_len.div_ceil(block_size).max(1);
    let total_blocks: u32 = prompts.iter().map(|p| (p.len() as u32).div_ceil(block_size).max(1)).sum();
    let num_blocks = total_blocks + prompts.len() as u32; // headroom
    let max_batch = prompts.len().max(1) as u32;
    let max_prefill = max_len.max(1);

    eprintln!("brain qwen calib: {base_id}, {} prompts", prompts.len());
    let weights_str = weights_path.to_str().unwrap_or_default();
    let mut eng = qwen3::serve::Engine::load(weights_str, block_size, num_blocks, max_batch, max_blocks_per_seq, max_prefill, false, false);
    let collector = eng.calibrate_kv(&prompts);
    let rows = collector.report();

    if report {
        println!("{:<20} {:>12} {:>12} {:>12} {:>10}", "stream", "absmax", "p99", "p99.99", "ratio");
        for r in &rows {
            println!("{:<20} {:>12.5} {:>12.5} {:>12.5} {:>10.2}", r.name, r.absmax, r.p99, r.p9999, r.outlier_ratio);
        }
        let worst = rows.first();
        let median_ratio = if rows.is_empty() { 0.0 } else { rows[rows.len() / 2].outlier_ratio };
        println!(
            "brain qwen calib: {} streams, worst ratio {:.2} ({}), median ratio {:.2}",
            rows.len(),
            worst.map(|r| r.outlier_ratio).unwrap_or(0.0),
            worst.map(|r| r.name.as_str()).unwrap_or("-"),
            median_ratio
        );
    }

    if let Some(path) = out {
        let json: Vec<serde_json::Value> = rows
            .iter()
            .map(|r| serde_json::json!({"name": r.name, "absmax": r.absmax, "p99": r.p99, "p9999": r.p9999, "outlier_ratio": r.outlier_ratio}))
            .collect();
        let doc = serde_json::json!({"model": base_id, "prompts": prompts.len(), "streams": json});
        let text = serde_json::to_string_pretty(&doc).expect("report JSON is always serializable");
        match std::fs::write(&path, text) {
            Ok(()) => println!("brain qwen calib: wrote {path}"),
            Err(e) => eprintln!("brain qwen calib: {path}: {e}"),
        }
    }

    if let Some(path) = clip_out {
        let ckpt_cfg = checkpoint::load(weights_str).header["config"].clone();
        let cfg = QwenConfig::from_json(&ckpt_cfg);
        let calib = model::kvcalib::KvCalib::from_collector(
            &base_id,
            cfg.n_layers as usize,
            cfg.n_kv_heads as usize,
            cfg.head_dim as usize,
            percentile,
            &collector,
        );
        match calib.save(Path::new(&path)) {
            Ok(()) => println!("brain qwen calib: wrote {path} (percentile {percentile})"),
            Err(e) => eprintln!("brain qwen calib: {path}: {e}"),
        }
    }
}

#[cfg(test)]
mod lora_finetune_cli_tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-qwen-cli-lora-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn parse_adapter_spec_defaults_the_tag_to_latest() {
        assert_eq!(
            parse_adapter_spec("swedishembedded-com/generic-sft").unwrap(),
            ("swedishembedded-com".to_string(), "generic-sft".to_string(), "latest".to_string())
        );
    }

    #[test]
    fn parse_adapter_spec_accepts_an_explicit_tag() {
        assert_eq!(parse_adapter_spec("owner/name:v2").unwrap(), ("owner".to_string(), "name".to_string(), "v2".to_string()));
    }

    #[test]
    fn parse_adapter_spec_rejects_a_spec_with_no_slash() {
        let err = parse_adapter_spec("no-slash-here").unwrap_err();
        assert!(err.contains("OWNER/NAME"), "{err}");
    }

    #[test]
    fn parse_adapter_spec_rejects_any_empty_segment() {
        assert!(parse_adapter_spec("/name").is_err());
        assert!(parse_adapter_spec("owner/").is_err());
        assert!(parse_adapter_spec("owner/name:").is_err());
    }

    #[test]
    fn resolve_base_prefers_a_real_file_over_parsing_it_as_a_model_ref() {
        // "out/qwen.safetensors" is ALSO a syntactically valid ModelRef
        // ("out" as vendor, "qwen" as repo) -- the file on disk must win.
        let dir = tmp("file-path");
        let weights = dir.join("qwen.safetensors");
        std::fs::write(&weights, b"not a real checkpoint, just needs to exist").unwrap();

        let (path, base_dir, id) = resolve_base(weights.to_str().unwrap(), None).unwrap();
        assert_eq!(path, weights);
        assert_eq!(base_dir, dir);
        // Synthesized under the "local" reserved vendor so it is always a
        // valid `vendor/repo` -- required a moment later to build
        // `vendor/repo:owner:name:tag` for the adapter ref.
        assert_eq!(id, "local/qwen");
        assert!(brain_modelref::ModelRef::parse(&format!("{id}:o:n:latest")).is_ok(), "id {id:?} must combine into a parseable adapter ref");
    }

    #[test]
    fn resolve_base_reports_neither_a_file_nor_a_valid_ref_by_name_not_a_panic() {
        let err = resolve_base("not-a-file-and-not-a-ref-either", None).unwrap_err();
        assert!(err.contains("not a file"), "{err}");
    }

    #[test]
    fn resolve_base_reports_a_missing_models_dir_by_name_not_a_panic() {
        let err = resolve_base("Qwen/Qwen3-0.6B", None).unwrap_err();
        assert!(err.contains("no models directory"), "{err}");
    }

    #[test]
    fn resolve_base_resolves_a_store_ref_via_the_model_store() {
        let dir = tmp("store-ref");
        let repo_dir = dir.join("Qwen").join("Qwen3-0.6B");
        std::fs::create_dir_all(&repo_dir).unwrap();
        let card = checkpoint::st::ModelCard::new("Qwen/Qwen3-0.6B", "qwen");
        checkpoint::st::save_safetensors(
            repo_dir.join("model.brain.safetensors").to_str().unwrap(),
            &[("weight".to_string(), vec![2], vec![1.0, 2.0])],
            &serde_json::json!({"vocab_size": 23}),
            Some(&card),
        )
        .unwrap();
        std::fs::write(repo_dir.join("tokenizer.json"), b"{}").unwrap();

        let (path, base_dir, id) = resolve_base("Qwen/Qwen3-0.6B", Some(&dir)).unwrap();
        assert_eq!(path, repo_dir.join("model.brain.safetensors"));
        assert_eq!(base_dir, repo_dir);
        assert_eq!(id, "Qwen/Qwen3-0.6B");
    }

    #[test]
    fn resolve_base_reports_a_ref_not_found_in_the_store_by_name_not_a_panic() {
        let dir = tmp("store-ref-missing");
        let err = resolve_base("Qwen/Qwen3-0.6B", Some(&dir)).unwrap_err();
        assert!(err.contains("not found in the model store"), "{err}");
    }
}
