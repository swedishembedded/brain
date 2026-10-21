// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Does a HuggingFace release served as stored compute the model the GGUF of the same release computes?
//!
//! Swedish Embedded AB implements validation of quantized model releases against independent references for its
//! clients. If your team needs expertise in proving a served checkpoint equals its upstream then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! Builds the first `--layers N` layers of a release (`--hf-dir`, every tensor in its stored format) and of the GGUF
//! of the same model (`--gguf`, int8 tier), runs both over the tokens of `--text`, and prints per position the top-1
//! token of each, the cosine of their logits and the rank of one model's top-1 in the other's. Two quantizations of
//! one model agree closely; a mismapped tensor or a wrong scale shows as a collapse of the cosine at some depth.
//!
//! Usage: qwen35moe_hf_check --hf-dir DIR --gguf FILE [--layers N] [--text "..."]

use checkpoint::gguf::MmapGguf;
use data::tokenizer::Tokenizer;
use gpu_core::Gpu;
use qwen35moe::config::Qwen35Config;
use qwen35moe::gguf_load;
use qwen35moe::model::{pipelines, Qwen35};

fn flag(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn logits_of(cfg: &Qwen35Config, src: &dyn checkpoint::TensorSource, tier: &model::ops::TierPolicy, tokens: &[u32]) -> Vec<f32> {
    let t = tokens.len() as u32;
    let m = Qwen35::new_on_tier_src(Gpu::new(pipelines()), cfg.clone(), 1, t, src, tier);
    println!("  NVFP4 expert banks: {}, int8 experts: {}", m.moe_nvfp4_active(), m.moe_int8_active());
    m.logits_all(tokens)
}

fn argmax(row: &[f32]) -> usize {
    row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0)
}

fn main() {
    let (Some(hf_dir), Some(gguf)) = (flag("--hf-dir"), flag("--gguf")) else {
        eprintln!("usage: qwen35moe_hf_check --hf-dir DIR --gguf FILE [--layers N] [--text \"...\"]");
        std::process::exit(2);
    };
    let layers: u32 = flag("--layers").and_then(|s| s.parse().ok()).unwrap_or(4);
    let text = flag("--text").unwrap_or_else(|| "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is".to_string());
    let mg = MmapGguf::open(&gguf).unwrap_or_else(|e| panic!("open {gguf}: {e}"));
    let tok = gguf_load::tokenizer(&mg).unwrap_or_else(|e| panic!("{e}"));
    let mut tokens = tok.encode(&text);
    let t = tokens.len().next_power_of_two().max(8);
    tokens.resize(t, tokens[0]);
    println!("{} tokens ({} from the text), {layers} layers", t, tokens.len());

    let dir = std::path::Path::new(&hf_dir);
    let mut hf_cfg = qwen35moe::hf_source::hf_config(dir).unwrap_or_else(|e| panic!("{e}"));
    hf_cfg.block_size = t as u32;
    hf_cfg.n_layers = layers;
    let hf_src = qwen35moe::hf_source::open_dir_truncated(dir, &hf_cfg).unwrap_or_else(|e| panic!("{e}"));
    println!("HF release as stored:");
    let hf = logits_of(&hf_cfg, &hf_src, &model::ops::TierPolicy::uniform(gpu_core::select::Dtype::F32), &tokens);

    let mut g_cfg = gguf_load::resident_config(&mg, t as u32).unwrap_or_else(|e| panic!("{e}"));
    g_cfg.n_layers = layers;
    let g_src = gguf_load::source(&mg, &g_cfg).unwrap_or_else(|e| panic!("{e}"));
    println!("GGUF (int8):");
    let gg = logits_of(&g_cfg, &g_src, &gguf_load::tier_from_env(), &tokens);

    let v = hf_cfg.vocab as usize;
    println!("{:>4} {:>8} {:>8} {:>9}  token", "pos", "hf top1", "gguf top1", "cosine");
    for pos in 0..t {
        let (a, b) = (&hf[pos * v..(pos + 1) * v], &gg[pos * v..(pos + 1) * v]);
        let (dot, na, nb) = a.iter().zip(b).fold((0f64, 0f64, 0f64), |s, (x, y)| (s.0 + *x as f64 * *y as f64, s.1 + (*x as f64).powi(2), s.2 + (*y as f64).powi(2)));
        println!("{pos:>4} {:>8} {:>8} {:>9.5}  {:?}", argmax(a), argmax(b), dot / (na.sqrt() * nb.sqrt()).max(1e-12), tok.decode(&[tokens[pos]]));
    }
}
