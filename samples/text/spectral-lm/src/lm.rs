// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SpectralLM (and its SwiGLU control) trained from scratch, scored and
//! sampled.

use std::path::{Path, PathBuf};
use std::time::Instant;

use appopts::Args;
use data::bytebpe::ByteBpe;
use data::tokenizer::Tokenizer;
use spectrallm::{Mixer, SpectralConfig, SpectralLm, Transform, TransformPath};

use crate::{die, fit_opts, read_text, tokens_of};

/// The model shape from the shared flags; `vocab` and `block_size` come from
/// the corpus and the schedule.
fn shape(a: &mut Args) -> SpectralConfig {
    let d_model = a.u32_or("--d-model", 256);
    let n_heads = a.u32_or("--heads", 8);
    let mixer = match a.str_or("--mixer", "spectral").as_str() {
        "spectral" => Mixer::Spectral { signal_len: a.u32_or("--signal-len", 2 * d_model.next_power_of_two()), channels: a.u32_or("--channels", 8) },
        "swiglu" => Mixer::SwiGlu { d_ff: a.u32_or("--d-ff", 4 * d_model) },
        other => die(&format!("--mixer {other:?} is not spectral or swiglu")),
    };
    let transform = match a.str_or("--transform", "auto").as_str() {
        "auto" => TransformPath::Auto,
        "dense" => TransformPath::Dense,
        "fft" => TransformPath::Fft,
        other => die(&format!("--transform {other:?} is not auto, dense or fft")),
    };
    SpectralConfig {
        vocab: 0,
        block_size: 0,
        n_layers: a.u32_or("--layers", 6),
        d_model,
        n_heads,
        n_kv_heads: a.u32_or("--kv-heads", (n_heads / 2).max(1)),
        head_dim: a.u32_or("--head-dim", (d_model / n_heads).max(2) & !1),
        rope_theta: a.f32_or("--rope-theta", 1.0e4),
        rms_eps: 1e-6,
        tie_embeddings: !a.take_flag("--untied"),
        mixer,
        transform,
    }
}

fn transform_path(t: Option<Transform>) -> TransformPath {
    match t {
        Some(Transform::Fft) => TransformPath::Fft,
        _ => TransformPath::Dense,
    }
}

fn param_counts(cfg: &SpectralConfig) -> (usize, usize) {
    let total: usize = cfg.param_list().iter().map(|(_, c)| c).sum();
    (cfg.non_embedding_params(), total - cfg.non_embedding_params())
}

pub fn train(a: &mut Args) {
    let Some(dir) = a.positional() else { die("usage: spectral-lm train <corpus dir> --out F [flags]") };
    let mut cfg = shape(a);
    let out = a.take_str("--out").map(PathBuf::from).unwrap_or_else(|| die("train needs --out"));
    let o = fit_opts(a, "spectral-lm train", 3e-3);
    a.finish();
    cfg.block_size = o.block_size;
    eprintln!("training {:?} on {dir}: {} steps of {} x {} tokens, seed {}", cfg.mixer, o.steps, o.batch_size * o.grad_accum.max(1), o.block_size, o.seed);
    let start = Instant::now();
    let (first, last) = model::fit::<SpectralLm>(Path::new(&dir), cfg, &o, Some(&out)).unwrap_or_else(|e| die(&e.to_string()));
    let seconds = start.elapsed().as_secs_f64();
    let out_str = out.to_string_lossy().into_owned();
    let reader = checkpoint::weightio::WeightReader::open(&out_str).unwrap_or_else(|e| die(&format!("{out_str}: {e}")));
    let cfg = SpectralConfig::from_json(&reader.config()).unwrap_or_else(|e| die(&e));
    let transform = SpectralLm::new_inference(cfg.clone(), 1, 1, &reader).transform();
    let (nonembed, embed) = param_counts(&cfg);
    let tokens = tokens_of(&o);
    println!(
        "{}",
        serde_json::json!({
            "event": "train",
            "config": cfg.to_json(),
            "seed": o.seed,
            "steps": o.steps,
            "batch": o.batch_size * o.grad_accum.max(1),
            "block": o.block_size,
            "lr": o.lr,
            "tokens": tokens,
            "initial_loss": first,
            "final_loss": last,
            "seconds": seconds,
            "tokens_per_s": tokens as f64 / seconds,
            "params_nonembed": nonembed,
            "params_embed": embed,
            "flops_per_token": cfg.forward_flops_per_token(o.block_size, transform_path(transform)),
            "transform": transform.map(|t| format!("{t:?}")),
            "checkpoint": out,
        })
    );
}

fn tokenizer(a: &mut Args) -> ByteBpe {
    let path = a.take_str("--tokenizer").unwrap_or_else(|| die("needs --tokenizer <corpus dir>/tokenizer.json"));
    ByteBpe::load(Path::new(&path)).unwrap_or_else(|e| die(&e))
}

pub fn eval(a: &mut Args) {
    let ckpt = a.take_str("--checkpoint").unwrap_or_else(|| die("eval needs --checkpoint"));
    let tok = tokenizer(a);
    let text = a.take_str("--text").unwrap_or_else(|| die("eval needs --text <documents file>"));
    let max_bytes = a.usize_or("--max-doc-bytes", usize::MAX);
    let batch = a.u32_or("--batch", 16);
    a.finish();
    let docs = model::score::byte_docs(&read_text(&text), &tok, tok.end_of_text(), max_bytes);
    let reader = checkpoint::weightio::WeightReader::open(&ckpt).unwrap_or_else(|e| die(&format!("{ckpt}: {e}")));
    let cfg = SpectralConfig::from_json(&reader.config()).unwrap_or_else(|e| die(&e));
    let longest = docs.iter().map(|d| d.tokens.len()).max().unwrap_or(2) as u32;
    if longest > cfg.block_size {
        die(&format!("the longest selected document has {longest} tokens, more than the model's {} positions; lower --max-doc-bytes", cfg.block_size));
    }
    let m = SpectralLm::new_inference(cfg, batch, longest, &reader);
    let start = Instant::now();
    let score = model::score::score_bytes(&m, &docs, batch, longest).unwrap_or_else(|e| die(&e));
    crate::qwen::print_score(&ckpt, &score, max_bytes, start.elapsed().as_secs_f64());
}

pub fn generate(a: &mut Args) {
    let ckpt = a.take_str("--checkpoint").unwrap_or_else(|| die("generate needs --checkpoint"));
    let tok = tokenizer(a);
    let prompt = a.str_or("--prompt", "Once upon a time");
    let max_new = a.usize_or("--max-new", 200);
    let temp = a.f32_or("--temp", 0.8);
    let top_k = a.usize_or("--top-k", 40);
    let seed = a.seed("spectral-lm generate");
    a.finish();
    let m = SpectralLm::load(&ckpt, 1, 1).unwrap_or_else(|e| die(&e));
    let mut rng = data::sample::Rng::new(seed);
    let mut ids = vec![tok.end_of_text()];
    ids.extend(tok.encode(&prompt));
    if ids.len() as u32 >= m.cfg.block_size {
        die(&format!("prompt of {} tokens leaves no room in the model's {} positions", ids.len(), m.cfg.block_size));
    }
    let mut logits = Vec::new();
    for id in &ids {
        logits = m.step(*id);
    }
    let start = Instant::now();
    let mut produced = 0;
    while produced < max_new {
        let next = data::sample::sample_logits(&logits, temp, top_k, &mut rng);
        if next == tok.end_of_text() {
            break;
        }
        ids.push(next);
        produced += 1;
        if m.cache_pos() == m.cfg.block_size {
            break;
        }
        logits = m.step(next);
    }
    println!("{}", tok.decode(&ids[1..]));
    eprintln!("{produced} tokens in {:.2}s, seed {seed}", start.elapsed().as_secs_f64());
}

pub fn cost(a: &mut Args) {
    let mut cfg = shape(a);
    let block = a.u32_or("--block", 256);
    cfg.vocab = a.u32_or("--vocab", data::prepare::TINYSTORIES_VOCAB as u32);
    a.finish();
    cfg.block_size = block;
    let (nonembed, embed) = param_counts(&cfg);
    println!(
        "{}",
        serde_json::json!({
            "config": cfg.to_json(),
            "params_nonembed": nonembed,
            "params_embed": embed,
            "flops_per_token_dense": cfg.forward_flops_per_token(block, TransformPath::Dense),
            "flops_per_token_fft": cfg.forward_flops_per_token(block, TransformPath::Fft),
        })
    );
}

/// Per-kernel-kind device profile of one training step at a shape: the
/// forward (trunk, head and loss) and the backward, slowest kernels first.
pub fn profile(a: &mut Args) {
    let mut cfg = shape(a);
    let (b, t) = (a.u32_or("--batch", 16), a.u32_or("--block", 512));
    cfg.vocab = a.u32_or("--vocab", data::prepare::TINYSTORIES_VOCAB as u32);
    let reps = a.usize_or("--reps", 3);
    let seed = a.seed("spectral-lm profile");
    a.finish();
    cfg.block_size = t;
    let m = SpectralLm::new_train(cfg.clone(), b, t, &spectrallm::init_weights(&cfg, seed));
    let tokens: Vec<u32> = (0..b * t).map(|i| (i * 7919) % cfg.vocab).collect();
    let targets: Vec<u32> = (0..b * t).map(|i| (i * 104_729 + 1) % cfg.vocab).collect();
    m.set_tokens_targets(&tokens, &targets);
    println!("{:?}, transform {:?}, batch {b} x {t}", cfg.mixer, m.transform());
    let (fwd, bwd) = m.profile(reps);
    fwd.print_top(None, 16);
    let bwd = bwd.expect("a training build profiles its backward");
    bwd.print_top(None, 16);
    let step = fwd.total_secs + bwd.total_secs;
    println!(
        "{}",
        serde_json::json!({
            "event": "profile",
            "config": cfg.to_json(),
            "transform": m.transform().map(|t| format!("{t:?}")),
            "batch": b,
            "block": t,
            "reps": reps,
            "forward_ms": fwd.total_secs * 1e3,
            "backward_ms": bwd.total_secs * 1e3,
            "step_ms": step * 1e3,
            "tokens_per_s": (b * t) as f64 / step,
        })
    );
}
