// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Latent sparse attention on a real Qwen3 checkpoint: distil an indexer
//! (D-LSA), evaluate indexers against the model's own attention and its
//! loss, and time prefill dense against sparse.
//!
//! Swedish Embedded AB implements long-context inference that keeps its
//! accuracy while its attention cost stops growing quadratically, for its
//! clients. If your team needs expertise in sparse attention, distillation
//! and long-context evaluation, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! Usage (every mode takes `--model REPO` - a checkout in the model store,
//! `Qwen/Qwen3-4B` - or `--checkpoint <file or dir>`, and `--device gpu1` to
//! pick the card, `gpu` by default):
//!
//! ```text
//! qwen_lsa distill --corpus DIR --out FILE [--tokens 8192] [--train-docs 48]
//!          [--epochs 3] [--queries 64] [--lr 1e-3] [--top-k 1024] [--top-p 256]
//!          [--tree SPEC] [--seed N]
//! qwen_lsa eval    --corpus DIR --indexer FILE[,FILE...] [--tokens 8192]
//!          [--held-docs 6] [--skip-docs 48] [--pca-corpus DIR] [--queries 32]
//!          [--tail 256] [--seed N]
//! qwen_lsa prefill --indexer FILE [--tokens 4096,8192,16384,32768] [--corpus DIR]
//!          [--kv-cache int8|f32|fp8] [--weights f32|int8] [--dense-until N]
//!          [--tree SPEC] [--profile]
//! ```
//!
//! `prefill --indexer random` times a random-projection indexer built for the
//! checkpoint (the speed of a read does not depend on what the projection
//! learned). `--profile` adds, per length, one more prefill of each kind with
//! the device timing every kernel: the per-kernel table, the time in the
//! index (index keys and Ward), the read, sparse and dense attention, and the
//! wall time beyond the device's (host work: page tables, node pages).
//! `--weights` holds the linears in f32 or int8; by default int8 from the
//! parameter count up that serving quantizes.
//!
//! `--tree size=4,blocks=1024:4096,beam=2048:2048,page=4096,growth=1024[,start=N]`
//! reads through node bucket levels (`model::lsa::tree`): node buckets of
//! `size` children, each level's node cluster block and beam width (one value
//! a level, `:`-separated), the page that levels grow at, the buckets level 1
//! takes at once, and the node buckets a beam may start from. `distill`
//! writes it into the indexer; `prefill` reads through it.
//!
//! `--indexer random` or `pca` in `eval` builds that training-free indexer
//! in place (PCA from the first two documents of `--pca-corpus`, by default
//! the evaluation corpus; point it at the training corpus to keep the
//! held-out documents out of every fit). The corpus is every
//! text file under `DIR` (sorted by path), tokenised and concatenated; the
//! first documents train, the last are held out.

use data::rng::Lcg;
use data::Tokenizer;
use model::lsa::distill::{retained_mass, retention, teacher_targets, CapturedLayer, Distiller, LayerSample};
use model::lsa::indexer::{IndexGeometry, Indexer};
use model::lsa::space::index_inputs;
use model::lsa::LsaConfig;
use qwen3::serve::Engine;
use std::collections::HashMap;
use std::time::Instant;

/// Keys of the teacher's distribution each query keeps (the paper's 2048).
const TEACHER_KEEP: usize = 2048;
const BLOCK: u32 = 16;

struct Args(HashMap<String, String>);

/// Flags that take no value.
const SWITCHES: [&str; 1] = ["profile"];

impl Args {
    fn parse(raw: &[String]) -> Result<Args, String> {
        let mut map = HashMap::new();
        let mut it = raw.iter();
        while let Some(k) = it.next() {
            let key = k.strip_prefix("--").ok_or_else(|| format!("expected a --flag, got {k:?}"))?;
            if SWITCHES.contains(&key) {
                map.insert(key.to_string(), String::new());
                continue;
            }
            let v = it.next().ok_or_else(|| format!("--{key} needs a value"))?;
            map.insert(key.to_string(), v.clone());
        }
        Ok(Args(map))
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.0.get(k).map(String::as_str)
    }
    fn flag(&self, k: &str) -> bool {
        self.0.contains_key(k)
    }
    fn req(&self, k: &str) -> Result<&str, String> {
        self.get(k).ok_or_else(|| format!("--{k} is required"))
    }
    fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> Result<T, String> {
        self.get(k).map_or(Ok(default), |v| v.parse().map_err(|_| format!("--{k} {v:?} is not a number")))
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().collect();
    let run = || -> Result<(), String> {
        let mode = raw.get(1).ok_or("a mode is required: distill, eval or prefill")?;
        let args = Args::parse(&raw[2..])?;
        let set = gpu_core::select_compute(args.get("device").unwrap_or("gpu"), args.get("backend"))?;
        set.apply()?;
        gpu_core::publish_compute_set(set);
        match mode.as_str() {
            "distill" => distill(&args),
            "eval" => eval(&args),
            "prefill" => prefill(&args),
            other => Err(format!("unknown mode {other:?}: distill, eval or prefill")),
        }
    };
    if let Err(e) = run() {
        eprintln!("qwen_lsa: {e}");
        std::process::exit(2);
    }
}

/// The checkpoint to load: `--checkpoint`, or `--model`'s checkout in the
/// model store. A directory holding the upstream safetensors loads as it is;
/// only a checkout that holds nothing but a converted file loads that.
fn checkpoint(args: &Args) -> Result<String, String> {
    let dir = match (args.get("checkpoint"), args.get("model")) {
        (Some(p), None) => std::path::PathBuf::from(p),
        (None, Some(m)) => {
            let reference = brain_modelref::ModelRef::parse(m).map_err(|e| format!("--model {m:?}: {e}"))?;
            let root = brain_modelstore::default_root().ok_or("no model store: pass --checkpoint")?;
            let dir = brain_modelstore::Store::new(root).repo_dir(&reference);
            if !dir.is_dir() {
                return Err(format!("--model {m}: no checkout at {}", dir.display()));
            }
            dir
        }
        _ => return Err("pass exactly one of --model and --checkpoint".into()),
    };
    let upstream = dir.join("model.safetensors").exists() || dir.join("model.safetensors.index.json").exists();
    let converted = dir.join("model.brain.safetensors");
    Ok(if !upstream && converted.exists() { converted } else { dir }.to_string_lossy().into_owned())
}

fn tokenizer_dir(path: &str) -> String {
    let p = std::path::Path::new(path);
    if p.is_dir() { path.to_string() } else { p.parent().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default() }
}

fn seed(args: &Args) -> Result<u64, String> {
    let s = match args.get("seed") {
        Some(v) => v.trim().parse::<u64>().map_err(|_| format!("--seed {v:?} is not a number"))?,
        None => data::rng::random_seed(),
    };
    println!("seed {s}");
    Ok(s)
}

/// The KV cache tier an engine stores.
#[derive(Clone, Copy, PartialEq)]
enum Kv {
    F32,
    Int8,
    Fp8,
}

fn engine(path: &str, max_tokens: u32) -> Result<(Engine, IndexGeometry), String> {
    engine_kv(path, max_tokens, Kv::F32, Some(false))
}

/// An engine for prompts of `max_tokens`, its KV cache in `kv`, its linears
/// int8 when `int8` says so - by default from the parameter count up that
/// serving quantizes (`qwen3::footprint::int8_by_default`).
fn engine_kv(path: &str, max_tokens: u32, kv: Kv, int8: Option<bool>) -> Result<(Engine, IndexGeometry), String> {
    let (cfg, src) = qwen3::open_checkpoint(path)?;
    let tier = if int8.unwrap_or_else(|| qwen3::footprint::int8_by_default(&cfg)) { qwen3::Dtype::I8 } else { qwen3::Dtype::F32 };
    let geometry = IndexGeometry { n_layers: cfg.n_layers as usize, n_heads: cfg.n_heads as usize, n_kv_heads: cfg.n_kv_heads as usize, head_dim: cfg.head_dim as usize, rope_base: cfg.rope_theta };
    let mbt = max_tokens.div_ceil(BLOCK);
    // Room for a warm-up prompt the prefix cache keeps beside the measured one.
    let mut e = Engine::from_source_tier(cfg, &*src, BLOCK, 2 * mbt + 8, 1, mbt, 512, kv != Kv::F32, tier);
    if kv == Kv::Fp8 {
        e.set_kv_fp8()?;
    }
    Ok((e, geometry))
}

/// The corpus as documents of `tokens` tokens: every text file under `dir`
/// in path order, tokenised and concatenated.
fn documents(dir: &str, tok: &dyn Tokenizer, tokens: usize, want: usize) -> Result<Vec<Vec<u32>>, String> {
    let mut files = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(dir)];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).map_err(|e| format!("{}: {e}", d.display()))? {
            let path = entry.map_err(|e| e.to_string())?.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if path.is_dir() {
                if !name.starts_with('.') && name != "target" {
                    stack.push(path);
                }
            } else if ["md", "rs", "txt", "py", "toml"].iter().any(|e| name.ends_with(&format!(".{e}"))) {
                files.push(path);
            }
        }
    }
    files.sort();
    let mut stream: Vec<u32> = Vec::new();
    let mut docs = Vec::new();
    for f in files {
        let Ok(text) = std::fs::read_to_string(&f) else { continue };
        stream.extend(tok.encode(&text));
        stream.extend(tok.encode("\n\n"));
        while stream.len() >= tokens {
            docs.push(stream.drain(..tokens).collect());
            if docs.len() == want {
                return Ok(docs);
            }
        }
    }
    Err(format!("{dir} holds only {} documents of {tokens} tokens, {want} wanted", docs.len()))
}

fn lsa_config(args: &Args) -> Result<LsaConfig, String> {
    let cfg = LsaConfig { top_k: args.num("top-k", 1024)?, top_p: args.num("top-p", 256)?, ..LsaConfig::default() };
    cfg.validate()?;
    Ok(cfg)
}

/// Query positions to sample: late enough that the read has long-range keys.
fn sample_positions(rng: &mut Lcg, tokens: usize, cfg: &LsaConfig, n: usize) -> Vec<u32> {
    let lo = (cfg.window + cfg.cluster_block + cfg.sink) as u32;
    let mut p: Vec<u32> = (0..n).map(|_| lo + rng.next_u32() % (tokens as u32 - lo)).collect();
    p.sort_unstable();
    p.dedup();
    p
}

/// Per layer: un-rotated keys, the sampled queries' rotated heads and their
/// index inputs, from one captured document.
struct Captured {
    /// Per layer `[T, kv_dim]` rotated keys.
    k: Vec<Vec<f32>>,
    /// Per layer `[S, q_dim]` rotated queries of the sampled positions.
    q: Vec<Vec<f32>>,
    /// Per layer `[T, input_dim]` index inputs of every key.
    key_inputs: Vec<Vec<f32>>,
    /// Per layer `[S, input_dim]` index inputs of the sampled queries.
    query_inputs: Vec<Vec<f32>>,
}

fn capture(e: &mut Engine, g: &IndexGeometry, doc: &[u32], positions: &[u32]) -> Result<Captured, String> {
    let layers = e.capture_attention(doc)?;
    let (q_dim, all): (usize, Vec<u32>) = (g.n_heads * g.head_dim, (0..doc.len() as u32).collect());
    let mut out = Captured { k: Vec::new(), q: Vec::new(), key_inputs: Vec::new(), query_inputs: Vec::new() };
    for qwen3::serve::LayerCapture { q, k } in layers {
        let qs: Vec<f32> = positions.iter().flat_map(|&p| q[p as usize * q_dim..][..q_dim].to_vec()).collect();
        out.key_inputs.push(index_inputs(&k, &all, g.n_kv_heads, g.n_kv_heads, g.head_dim, g.rope_base));
        out.query_inputs.push(index_inputs(&qs, positions, g.n_heads, g.n_kv_heads, g.head_dim, g.rope_base));
        out.q.push(qs);
        out.k.push(k);
    }
    Ok(out)
}

/// `--tree size=S,blocks=B1:B2..,beam=W1:W2..,page=P,growth=G[,start=N]`.
fn tree_config(args: &Args) -> Result<Option<model::lsa::tree::TreeConfig>, String> {
    let Some(spec) = args.get("tree") else { return Ok(None) };
    let mut fields = std::collections::HashMap::new();
    for part in spec.split(',') {
        let (k, v) = part.split_once('=').ok_or_else(|| format!("--tree {spec:?}: {part:?} is not key=value"))?;
        if fields.insert(k, v).is_some() {
            return Err(format!("--tree {spec:?}: {k} given twice"));
        }
    }
    let num = |k: &str| -> Result<usize, String> { fields.get(k).ok_or_else(|| format!("--tree {spec:?}: no {k}"))?.parse().map_err(|_| format!("--tree {spec:?}: {k} is not a number")) };
    let list = |k: &str| -> Result<Vec<usize>, String> { fields.get(k).ok_or_else(|| format!("--tree {spec:?}: no {k}"))?.split(':').map(|v| v.parse().map_err(|_| format!("--tree {spec:?}: {k} holds {v:?}"))).collect() };
    let tree = model::lsa::tree::TreeConfig {
        node_bucket_size: num("size")?,
        node_cluster_block: list("blocks")?,
        beam_width: list("beam")?,
        start_nodes: fields.get("start").map(|v| v.parse().map_err(|_| format!("--tree {spec:?}: start is not a number"))).transpose()?,
        page_size: num("page")?,
        growth_buckets: num("growth")?,
    };
    if let Some(k) = fields.keys().find(|k| !["size", "blocks", "beam", "start", "page", "growth"].contains(k)) {
        return Err(format!("--tree {spec:?}: unknown {k}"));
    }
    Ok(Some(tree))
}

fn distill(args: &Args) -> Result<(), String> {
    let path = checkpoint(args)?;
    let tokens: usize = args.num("tokens", 8192)?;
    let (train_docs, epochs, queries): (usize, usize, usize) = (args.num("train-docs", 48)?, args.num("epochs", 3)?, args.num("queries", 64)?);
    let lr: f32 = args.num("lr", 1e-3)?;
    let cfg = lsa_config(args)?;
    let mut rng = Lcg::new(seed(args)?);
    let tok = data::qwen_tokenizer::QwenBpe::from_dir(&tokenizer_dir(&path))?;
    let docs = documents(args.req("corpus")?, &tok, tokens, train_docs)?;
    let (mut e, geometry) = engine(&path, tokens as u32)?;

    // PCA initialisation from the first documents' keys.
    let t0 = Instant::now();
    let mut keys: Vec<Vec<f32>> = vec![Vec::new(); geometry.n_layers];
    for doc in docs.iter().take(2) {
        let c = capture(&mut e, &geometry, doc, &[])?;
        for (acc, k) in keys.iter_mut().zip(c.key_inputs) {
            acc.extend(k);
        }
    }
    let init = Indexer::pca(cfg, geometry, &keys)?;
    drop(keys);
    println!("pca init from 2 documents in {:.1}s", t0.elapsed().as_secs_f64());

    let mut d = Distiller::new(&init, 0.1)?;
    for epoch in 0..epochs {
        let mut kl_sum = 0.0;
        for (i, doc) in docs.iter().enumerate() {
            let t = Instant::now();
            let positions = sample_positions(&mut rng, tokens, &cfg, queries);
            let c = capture(&mut e, &geometry, doc, &positions)?;
            let samples: Vec<Option<LayerSample>> = (0..geometry.n_layers)
                .map(|l| {
                    let targets = teacher_targets(&c.q[l], &c.k[l], &positions, &geometry, &cfg, TEACHER_KEEP);
                    Some(LayerSample { keys: c.key_inputs[l].clone(), queries: c.query_inputs[l].clone(), positions: positions.clone(), targets })
                })
                .collect();
            let stats = d.step(&samples, lr);
            let kl = stats.iter().map(|s| s.kl).sum::<f64>() / stats.len() as f64;
            kl_sum += kl;
            println!("epoch {epoch} doc {i}: mean KL over layers {kl:.4} ({:.1}s)", t.elapsed().as_secs_f64());
        }
        println!("epoch {epoch}: mean KL {:.4}", kl_sum / docs.len() as f64);
    }
    let out = args.req("out")?;
    let indexer = Indexer { tree: tree_config(args)?, ..d.indexer() };
    indexer.save(out)?;
    println!("wrote {out}; temperatures {:?}", d.temperatures().iter().map(|t| t.map(|v| (v * 1000.0).round() / 1000.0)).collect::<Vec<_>>());
    Ok(())
}

fn eval(args: &Args) -> Result<(), String> {
    let path = checkpoint(args)?;
    let tokens: usize = args.num("tokens", 8192)?;
    let (held, queries, tail): (usize, usize, usize) = (args.num("held-docs", 6)?, args.num("queries", 32)?, args.num("tail", 256)?);
    let cfg = lsa_config(args)?;
    let mut rng = Lcg::new(seed(args)?);
    let tok = data::qwen_tokenizer::QwenBpe::from_dir(&tokenizer_dir(&path))?;
    let skip: usize = args.num("skip-docs", 48)?;
    let all = documents(args.req("corpus")?, &tok, tokens, skip + held)?;
    let held_docs = &all[skip..];
    let (mut e, geometry) = engine(&path, tokens as u32)?;
    let mut indexers: Vec<(String, Indexer)> = Vec::new();
    for spec in args.req("indexer")?.split(',') {
        let ix = match spec {
            "random" => Indexer::random(cfg, geometry, rng.next_u64())?,
            "pca" => {
                let fit = match args.get("pca-corpus") {
                    Some(dir) => documents(dir, &tok, tokens, 2)?,
                    None => all[..2].to_vec(),
                };
                let mut keys: Vec<Vec<f32>> = vec![Vec::new(); geometry.n_layers];
                for doc in &fit {
                    for (acc, k) in keys.iter_mut().zip(capture(&mut e, &geometry, doc, &[])?.key_inputs) {
                        acc.extend(k);
                    }
                }
                Indexer::pca(cfg, geometry, &keys)?
            }
            file => {
                let mut ix = Indexer::load(file)?;
                ix.cfg.top_k = cfg.top_k;
                ix.cfg.top_p = cfg.top_p;
                ix
            }
        };
        indexers.push((spec.to_string(), ix));
    }

    // Retained attention: the share of the model's own (dense) attention the
    // selection reads, against the window and sinks alone.
    let mut retained: Vec<Vec<f64>> = vec![Vec::new(); indexers.len() + 1];
    for doc in held_docs {
        let positions = sample_positions(&mut rng, tokens, &cfg, queries);
        let c = capture(&mut e, &geometry, doc, &positions)?;
        let local: Vec<Vec<u32>> = positions.iter().map(|&p| {
            let ws = (p as usize + 1).saturating_sub(cfg.window);
            (ws as u32..=p).chain(0..cfg.sink.min(ws) as u32).collect()
        }).collect();
        for l in 0..geometry.n_layers {
            retained[0].extend(retained_mass(&c.q[l], &c.k[l], &positions, &geometry, &local));
        }
        for (i, (_, ix)) in indexers.iter().enumerate() {
            let layers: Vec<CapturedLayer> = (0..geometry.n_layers).map(|l| CapturedLayer { key_inputs: &c.key_inputs[l], query_inputs: &c.query_inputs[l], q: &c.q[l], k: &c.k[l] }).collect();
            let per_layer = retention(ix, &layers, &positions)?;
            retained[i + 1].extend(per_layer.into_iter().flatten());
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    println!("\nretained share of dense attention (held-out, {} docs x {} queries x {} layers):", held_docs.len(), queries, geometry.n_layers);
    println!("  window {} + sinks {} only: {:.4}", cfg.window, cfg.sink, mean(&retained[0]));
    for (i, (name, _)) in indexers.iter().enumerate() {
        println!("  {name:<40} {:.4}  (top_k {}, top_p {})", mean(&retained[i + 1]), cfg.top_k, cfg.top_p);
    }

    // Next-token loss over each held-out document's last `tail` positions.
    let nll = |e: &mut Engine| -> f64 {
        let lp: Vec<f32> = held_docs.iter().flat_map(|d| e.position_logprobs(d, d.len() - 1 - tail)).collect();
        -lp.iter().map(|&x| x as f64).sum::<f64>() / lp.len() as f64
    };
    let dense = nll(&mut e);
    println!("\nheld-out next-token loss over the last {tail} positions of {} docs:", held_docs.len());
    println!("  dense                                    {dense:.4}  (ppl {:.3})", dense.exp());
    for (name, ix) in &indexers {
        let (mut s, _) = engine(&path, tokens as u32)?;
        s.set_lsa(ix)?;
        let v = nll(&mut s);
        println!("  {name:<40} {v:.4}  (ppl {:.3}, {:+.4} vs dense)", v.exp(), v - dense);
    }
    Ok(())
}

fn prefill(args: &Args) -> Result<(), String> {
    let path = checkpoint(args)?;
    let lens: Vec<usize> = args.get("tokens").unwrap_or("4096,8192,16384,32768").split(',').map(|s| s.parse().map_err(|_| format!("--tokens {s:?}"))).collect::<Result<_, _>>()?;
    let max = *lens.iter().max().ok_or("--tokens is empty")?;
    let cfg = lsa_config(args)?;
    let mut rng = Lcg::new(seed(args)?);
    let tok = data::qwen_tokenizer::QwenBpe::from_dir(&tokenizer_dir(&path))?;
    let doc: Vec<u32> = match args.get("corpus") {
        Some(dir) => documents(dir, &tok, max, 1)?.remove(0),
        None => (0..max).map(|_| 200 + rng.next_u32() % 20_000).collect(),
    };
    let mut ix = match args.req("indexer")? {
        "random" => {
            let (c, _) = qwen3::open_checkpoint(&path)?;
            let geometry = IndexGeometry { n_layers: c.n_layers as usize, n_heads: c.n_heads as usize, n_kv_heads: c.n_kv_heads as usize, head_dim: c.head_dim as usize, rope_base: c.rope_theta };
            Indexer::random(cfg, geometry, rng.next_u64())?
        }
        file => Indexer::load(file)?,
    };
    ix.cfg.top_k = cfg.top_k;
    ix.cfg.top_p = cfg.top_p;
    if let Some(tree) = tree_config(args)? {
        println!("through a tree: {tree:?}");
        ix.tree = Some(tree);
    }
    let weights = match args.get("weights") {
        None => None,
        Some("f32") => Some(false),
        Some("int8") => Some(true),
        Some(other) => return Err(format!("--weights {other:?}: f32 or int8")),
    };
    let kv = match args.get("kv-cache").unwrap_or("int8") {
        "f32" => Kv::F32,
        "int8" => Kv::Int8,
        "fp8" => Kv::Fp8,
        other => return Err(format!("--kv-cache {other:?}: f32, int8 or fp8")),
    };
    println!("prefill of one sequence, {path}, {} KV, top_k {} top_p {} window {}", args.get("kv-cache").unwrap_or("int8"), cfg.top_k, cfg.top_p, cfg.window);
    for &n in &lens {
        let prompt = &doc[..n];
        // One warm-up prefill of other tokens (pipelines, allocator), then the
        // timed one: a repeat of the same prompt would be served by the prefix
        // cache and time nothing.
        let warm: Vec<u32> = prompt[..prompt.len().min(1100)].iter().map(|t| (t + 7) % 20_000 + 200).collect();
        let time = |e: &mut Engine| -> f64 {
            let mut table = model::paged::BlockTable::new();
            e.prefill_for_perf(&mut table, &warm);
            e.release_for_perf(&mut table);
            let mut table = model::paged::BlockTable::new();
            let t = Instant::now();
            e.prefill_for_perf(&mut table, prompt);
            let secs = t.elapsed().as_secs_f64();
            e.release_for_perf(&mut table);
            secs
        };
        let (mut dense, _) = engine_kv(&path, n as u32, kv, weights)?;
        let td = time(&mut dense);
        drop(dense);
        let (mut sparse, _) = engine_kv(&path, n as u32, kv, weights)?;
        sparse.set_lsa(&ix)?;
        if let Some(v) = args.get("dense-until") {
            sparse.set_lsa_dense_until(v.parse().map_err(|_| format!("--dense-until {v:?} is not a number"))?);
        }
        let crossover = sparse.lsa_dense_until();
        let ts = time(&mut sparse);
        println!("T={n:>6}  dense {:>8.2} s  sparse {:>8.2} s  speedup {:>5.2}x  ({:.0} vs {:.0} tok/s)  dense below {crossover}", td, ts, td / ts, n as f64 / td, n as f64 / ts);
        if args.flag("profile") {
            // Other tokens of the same length: the timed prompt sits in the
            // prefix cache.
            let other: Vec<u32> = prompt.iter().map(|t| (t + 13) % 20_000 + 200).collect();
            profile(&mut sparse, &format!("sparse T={n}"), &other);
            drop(sparse);
            let (mut dense, _) = engine_kv(&path, n as u32, kv, weights)?;
            profile(&mut dense, &format!("dense T={n}"), &other);
        }
    }
    Ok(())
}

/// One prefill of `prompt` with the device timing every kernel: the
/// per-kernel table, then the time by part of attention and the wall time the
/// device did not account for (host work between dispatches).
fn profile(e: &mut Engine, label: &str, prompt: &[u32]) {
    if !e.gpu().set_kernel_timing(true) {
        println!("{label}: this backend does not time kernels");
        return;
    }
    // The earlier prompts' cached blocks go: the pool holds about two.
    model::serve::PagedDecoder::reclaim_prefix(e, u32::MAX);
    e.gpu().reset_kernel_times();
    let mut table = model::paged::BlockTable::new();
    let t = Instant::now();
    e.prefill_for_perf(&mut table, prompt);
    let wall_ms = t.elapsed().as_secs_f64() * 1e3;
    e.release_for_perf(&mut table);
    let rows = e.gpu().kernel_times().unwrap_or_default();
    e.gpu().set_kernel_timing(false);
    // `+ 0.0` turns an empty sum's -0.0 into 0.
    let part = |pred: &dyn Fn(&str) -> bool| rows.iter().filter(|(n, _, _)| pred(n)).map(|(_, ms, _)| ms).sum::<f64>() + 0.0;
    let device: f64 = rows.iter().map(|(_, ms, _)| ms).sum();
    let index = part(&|n| n.starts_with("lsa_index") || n.starts_with("lsa_ward"));
    let read = part(&|n| n.starts_with("lsa_read"));
    let sparse = part(&|n| n.starts_with("paged_sparse"));
    let dense = part(&|n| n.contains("flash_prefill") || n.contains("flash_decode"));
    gpu_core::profile::print_device_table(label, wall_ms, Some(rows));
    println!(
        "{label}: wall {:.0} ms, device {:.0} ms (host {:.0} ms); index {:.0} ms, read {:.0} ms, sparse attention {:.0} ms, dense attention {:.0} ms, everything else {:.0} ms",
        wall_ms,
        device,
        (wall_ms - device).max(0.0),
        index,
        read,
        sparse,
        dense,
        (device - index - read - sparse - dense).max(0.0)
    );
}
