// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain qwen35 ...` - run the Qwen3.8-27B dense hybrid decoder.
//!
//!   brain qwen35 infer  --weights F [--tokenizer tokenizer.json] --prompt "..."
//!                     [--max-new N --temp X --top-k K --chat]
//!
//! GGUF import lives in the GENERIC `brain import-gguf` command
//! ([`crate::gguf_import`]), which dispatches on the file's own
//! `general.architecture`; `brain qwen35 import` remains as a deprecated
//! forward to it - mirrors `qwen35moe_cli`'s own `import` exactly, though
//! this model's real checkpoint ships as safetensors, not GGUF (no GGUF
//! importer for this arch yet - an unexercised path is a gate that never
//! runs).
//!
//! No `export` subcommand: unlike `qwen35moe`, this crate has no NPU/ONNX
//! export path (`npu::qwen35_export` does not exist) - out of scope for
//! this port, matching the recorded NPU gap.
//!
//! `qwen35::model::Qwen35`'s public constructors (`new_on`) take a
//! `&HashMap<String, Vec<f32>>`, matching every other model crate's
//! simplest load path (`checkpoint::load(path).by_role("")`).

use data::rng::Rng;
use data::tokenizer::Tokenizer;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

pub fn run_qwen35(args: &[String]) {
    match args.first().map(|s| crate::args::canon_verb(s)) {
        Some("import") => import(&args[1..]),
        Some("infer") => infer(&args[1..]),
        other => eprintln!("usage: brain qwen35 <import|infer> ...  (got {other:?})"),
    }
}

fn val(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    args.get(*i).cloned().unwrap_or_else(|| {
        eprintln!("{flag} requires a value");
        std::process::exit(2);
    })
}

/// `brain qwen35 import --gguf FILE --out qwen35.safetensors` - **deprecated**
/// alias for the generic `brain import-gguf FILE [--out PATH] [--id NAME]`.
/// Mirrors `qwen35moe_cli::import` exactly.
fn import(args: &[String]) {
    eprintln!("brain qwen35 import is deprecated -- use `brain import-gguf FILE [--out PATH] [--id NAME]`");
    crate::gguf_import::run_import_gguf(args);
}

/// `brain qwen35 infer --weights F [--tokenizer T | --gguf G] --prompt "..."`:
/// single-sequence greedy/sampled generation via `Qwen35::step`, through
/// `qwen35::sample::generate_kv` (M11's own decode path). Not the paged
/// `PagedDecoder`/`Scheduler` serving path (`qwen35::serve`) - this is the
/// same "simple, direct, one request" tier `qwen3::sample::generate_kv`
/// occupies alongside `qwen3::serve::Engine`.
fn infer(args: &[String]) {
    let mut weights = String::new();
    let mut tokenizer = String::new();
    let mut gguf_for_tok = String::new();
    let mut prompt = String::new();
    let mut max_new = 32usize;
    let mut temp = 0.0f32;
    let mut top_k = 0usize;
    let mut top_p = 1.0f32;
    let mut seed = 0u64;
    let mut chat = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--tokenizer" => tokenizer = val(args, &mut i, "--tokenizer"),
            "--gguf" => gguf_for_tok = val(args, &mut i, "--gguf"),
            "--prompt" => prompt = val(args, &mut i, "--prompt"),
            "--max-new" => max_new = val(args, &mut i, "--max-new").parse().unwrap_or(max_new),
            "--temp" => temp = val(args, &mut i, "--temp").parse().unwrap_or(temp),
            "--top-k" => top_k = val(args, &mut i, "--top-k").parse().unwrap_or(top_k),
            "--top-p" => top_p = val(args, &mut i, "--top-p").parse().unwrap_or(top_p),
            "--seed" => seed = val(args, &mut i, "--seed").parse().unwrap_or(seed),
            "--chat" => chat = true,
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() || (tokenizer.is_empty() && gguf_for_tok.is_empty()) {
        eprintln!(
            "usage: brain qwen35 infer --weights F (--tokenizer tokenizer.json | --gguf original.gguf) --prompt \"...\" \
             [--max-new N --temp X --top-k K --top-p P --seed S --chat]"
        );
        return;
    }

    let tok = if !tokenizer.is_empty() {
        data::qwen_tokenizer::QwenBpe::from_file(&tokenizer)
    } else {
        checkpoint::gguf::MmapGguf::open(&gguf_for_tok)
            .map_err(|e| format!("open {gguf_for_tok}: {e}"))
            .and_then(|mg| mg.tokenizer().ok_or_else(|| format!("{gguf_for_tok}: no embedded tokenizer")))
            .and_then(|t| data::qwen_tokenizer::QwenBpe::from_gguf(&t))
    };
    let tok = match tok {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer load failed: {e}");
            return;
        }
    };

    let text = if chat { tok.apply_chat_template(&[("user", &prompt)], true) } else { prompt.clone() };
    let ids = tok.encode(&text);
    if ids.is_empty() {
        eprintln!("empty prompt");
        return;
    }

    let container = checkpoint::load(&weights);
    let cfg = Qwen35Config::from_json(&container.header["config"]);
    let init = container.by_role("");
    let cap = (ids.len() + max_new) as u32;

    let t_load = std::time::Instant::now();
    let model = Qwen35::new_on(gpu_core::Gpu::new(pipelines()), cfg, 1, cap, &init);
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    let eos_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|s| tok.encode(s).first().copied()).collect();
    let mut rng = Rng::new(seed);
    let t_gen = std::time::Instant::now();
    let gen = qwen35::sample::generate_kv(&model, &ids, max_new, temp, top_k, top_p, &eos_ids, &mut rng);
    let gen_ms = t_gen.elapsed().as_secs_f64() * 1e3;

    eprintln!("qwen35-timing load_ms={load_ms:.1} gen_ms={gen_ms:.1} tokens={}", gen.len());
    print!("{prompt}");
    print!("{}", tok.decode(&gen));
    println!();
}
