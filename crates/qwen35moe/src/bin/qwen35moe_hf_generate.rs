// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Greedy generation from a HuggingFace Qwen3.5/3.6 MoE release read where it lies, every tensor in its stored format,
//! through the model's single-sequence incremental decode path: a bring-up and correctness tool beside the server.
//!
//! Swedish Embedded AB implements validation of quantized model releases for its clients. If your team needs expertise
//! in checking that a served checkpoint answers correctly then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Usage: qwen35moe_hf_generate --hf-dir DIR [--prompt TEXT] [--chat] [--max-new N] [--layers N] [--ctx N]
//!
//! `--chat` wraps the prompt in the ChatML template with the thinking block closed; without it the prompt is raw text
//! the model continues. Prints the continuation, the decode rate and the NVFP4 tier state.

use std::time::Instant;

use data::rng::Rng;
use data::tokenizer::Tokenizer;
use gpu_core::Gpu;
use qwen35moe::model::{pipelines, Qwen35};

fn flag(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn main() {
    let Some(hf_dir) = flag("--hf-dir") else {
        eprintln!("usage: qwen35moe_hf_generate --hf-dir DIR [--prompt TEXT] [--chat] [--max-new N] [--layers N] [--ctx N]");
        std::process::exit(2);
    };
    let dir = std::path::Path::new(&hf_dir);
    let prompt = flag("--prompt").unwrap_or_else(|| "The capital of France is".to_string());
    let max_new: usize = flag("--max-new").and_then(|s| s.parse().ok()).unwrap_or(32);
    let ctx: u32 = flag("--ctx").and_then(|s| s.parse().ok()).unwrap_or(512);
    let tok = data::qwen_tokenizer::QwenBpe::from_dir(&hf_dir).unwrap_or_else(|e| panic!("{e}"));
    let text = if std::env::args().any(|a| a == "--chat") { tok.apply_chat_template_no_think(&[("user", &prompt)]) } else { prompt.clone() };
    let tokens = tok.encode(&text);
    println!("prompt: {} tokens", tokens.len());

    let mut cfg = qwen35moe::hf_source::hf_config(dir).unwrap_or_else(|e| panic!("{e}"));
    cfg.block_size = ctx;
    if let Some(n) = flag("--layers").and_then(|s| s.parse::<u32>().ok()) {
        cfg.n_layers = n.min(cfg.n_layers);
    }
    let src = qwen35moe::hf_source::open_dir_truncated(dir, &cfg).unwrap_or_else(|e| panic!("{e}"));
    let t0 = Instant::now();
    let model = Qwen35::new_on_src(Gpu::new(pipelines()), cfg.clone(), 1, ctx, &src);
    println!("loaded in {:.1} s, backend {}, NVFP4 expert banks {}", t0.elapsed().as_secs_f64(), gpu_core::backend_name(), model.moe_nvfp4_active());

    let eos = [248046u32, 248044];
    let t1 = Instant::now();
    let out = qwen35moe::sample::generate_kv(&model, &tokens, max_new, 0.0, 0, 1.0, &eos, &mut Rng::new(data::rng::random_seed()));
    let dt = t1.elapsed().as_secs_f64();
    println!("generated {} tokens in {dt:.2} s ({:.1} tok/s including prefill): {:?}", out.len(), out.len() as f64 / dt, tok.decode(&out));
    println!("ids: {out:?}");
}
