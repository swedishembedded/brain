// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain qwen35moe …` — import / run the Qwen3.5-35B-A3B hybrid decoder.
//!
//!   brain qwen35moe import --gguf FILE --out qwen35.safetensors
//!   brain qwen35moe infer  --weights F [--tokenizer tokenizer.json] --prompt "..."
//!                     [--max-new N --temp X --top-k K --chat --i8]
//!
//! Two-step flow (GGUF -> brain-native safetensors -> infer), not a direct
//! GGUF-streaming infer path: `qwen35moe::model::Qwen35`'s public
//! constructors (`new_on`/`new_on_i8`) take a `&HashMap<String, Vec<f32>>`,
//! matching every other model crate's simplest load path (`checkpoint::load
//! (path).by_role("")`); the mmap-streaming `TensorSource` path
//! `checkpoint::gguf::MmapGguf` supports is reachable today only through
//! `Qwen35`'s private constructor. Exposing that publicly (for real 35B-scale
//! serving, where materializing a HashMap defeats the point) is `serve.rs`'s
//! own concern — see `docs/models/qwen35/status.md`'s P11 entries — not
//! duplicated here.
//!
//! `import_gguf` only writes tensors -- the tokenizer is not carried through
//! the safetensors conversion -- so `infer` takes EITHER `--tokenizer
//! tokenizer.json` OR `--gguf original.gguf` (re-opens that file's embedded
//! tokenizer directly, no conversion needed for that part).

use data::rng::Rng;
use data::tokenizer::Tokenizer;
use qwen35moe::config::Qwen35Config;
use qwen35moe::model::Qwen35;

pub fn run_qwen35moe(args: &[String]) {
    match args.first().map(|s| crate::args::canon_verb(s)) {
        Some("import") => import(&args[1..]),
        Some("infer") => infer(&args[1..]),
        other => eprintln!("usage: brain qwen35moe <import|infer> ...  (got {other:?})"),
    }
}

fn val(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    args.get(*i).cloned().unwrap_or_else(|| {
        eprintln!("{flag} requires a value");
        std::process::exit(2);
    })
}

/// `brain qwen35moe import --gguf FILE --out qwen35.safetensors` — streams the
/// GGUF checkpoint into brain's native format via
/// `qwen35moe::import::import_gguf` (see that function's own module doc for
/// the full llama.cpp<->HF tensor-naming discussion; this is a thin CLI
/// wrapper, not a second implementation).
fn import(args: &[String]) {
    let mut gguf = String::new();
    let mut out = "qwen35.safetensors".to_string();
    let mut id: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => gguf = val(args, &mut i, "--gguf"),
            "--out" => out = val(args, &mut i, "--out"),
            "--id" => id = Some(val(args, &mut i, "--id")),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if gguf.is_empty() {
        eprintln!("usage: brain qwen35moe import --gguf FILE --out qwen35.safetensors [--id VENDOR/REPO]");
        return;
    }
    match qwen35moe::import::import_gguf(&gguf, &out, id.as_deref()) {
        Ok(()) => eprintln!("qwen35moe: imported {gguf} -> {out}"),
        Err(e) => {
            eprintln!("qwen35moe import failed: {e}");
            std::process::exit(1);
        }
    }
}

/// `brain qwen35moe infer --weights F [--tokenizer T | --gguf G] --prompt "..."`
/// — single-sequence greedy/sampled generation via `Qwen35::step`
/// (`qwen35moe::sample::generate_kv`, P11b's own decode path). Not the paged
/// `PagedDecoder`/`Scheduler` serving path (`qwen35moe::serve`) -- this is
/// the same "simple, direct, one request" tier `qwen3::sample::generate_kv`
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
    let mut i8 = false;
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
            "--i8" => i8 = true,
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() || (tokenizer.is_empty() && gguf_for_tok.is_empty()) {
        eprintln!(
            "usage: brain qwen35moe infer --weights F (--tokenizer tokenizer.json | --gguf original.gguf) --prompt \"...\" \
             [--max-new N --temp X --top-k K --top-p P --seed S --chat --i8]"
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
    let model = if i8 { Qwen35::new_i8(cfg, 1, cap, &init) } else { Qwen35::new(cfg, 1, cap, &init) };
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    let eos_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|s| tok.encode(s).first().copied()).collect();
    let mut rng = Rng::new(seed);
    let t_gen = std::time::Instant::now();
    let gen = qwen35moe::sample::generate_kv(&model, &ids, max_new, temp, top_k, top_p, &eos_ids, &mut rng);
    let gen_ms = t_gen.elapsed().as_secs_f64() * 1e3;

    eprintln!("qwen35moe-timing load_ms={load_ms:.1} gen_ms={gen_ms:.1} tokens={}", gen.len());
    print!("{prompt}");
    print!("{}", tok.decode(&gen));
    println!();
}
