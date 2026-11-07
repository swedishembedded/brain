// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain lsa` - latent sparse attention for any model whose engine meets
//! [`model::lsa::capture::AttentionCapture`]: distil an indexer (D-LSA),
//! evaluate indexers against the model's own attention and its loss, and time
//! prefill dense against sparse.
//!
//! Swedish Embedded AB implements long-context inference that keeps its
//! accuracy while its attention cost stops growing quadratically, for its
//! clients. If your team needs expertise in sparse attention, distillation
//! and long-context evaluation, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! The architecture is read from the checkpoint's `config.json` (or named with
//! `--arch`); `qwen3` and `ricursive` are the engines that meet the contract
//! today. The same functions distil either: Qwen3 over every layer, Ripple
//! over its global layers only (a sliding layer already reads its window).

use data::rng::Lcg;
use data::Tokenizer;
use model::lsa::capture::{self, AttentionCapture, DistillOptions};
use model::lsa::indexer::Indexer;
use model::lsa::LsaConfig;
use model::serve::PagedDecoder;
use std::collections::HashMap;
use std::time::Instant;

const USAGE: &str = "\
brain lsa distill --corpus DIR --out FILE (--model REPO | --checkpoint PATH) [options]
brain lsa eval    --corpus DIR --indexer SPEC[,SPEC...] (--model REPO | --checkpoint PATH) [options]
brain lsa prefill --indexer SPEC (--model REPO | --checkpoint PATH) [options]

  Latent sparse attention (D-LSA) for any architecture whose engine can
  capture its attention: qwen3 (every layer) and ricursive (its global layers).

  --model REPO         a checkout in the model store (vendor/repo)
  --checkpoint PATH    a checkpoint file or directory
  --arch ID            the architecture (default: what config.json names)
  --layers A,B,..      run only these layers of the checkpoint, in this order
                       (ricursive: a partial checkpoint such as layers 0 and 4)
  --device SPEC        where to run (default gpu); software for lavapipe
  --seed N             random seed (default random, printed)
  --top-k N --top-p N  the budget (1024, 256); --index-dim N --cluster-block N
                       --window N --sink N override the paper's sizes (64, 512, 512, 4)

  distill  --tokens N (8192) --train-docs N (48) --epochs N (3) --queries N (64)
           --lr X (1e-3) --top-k N (1024) --top-p N (256) [--tree SPEC]
  eval     --tokens N --held-docs N (6) --skip-docs N (48) --pca-corpus DIR
           --queries N (32) --tail N (256) --top-k N --top-p N
           SPEC is random, pca or an indexer file
  prefill  --tokens N,N,.. (4096,8192,16384,32768) [--corpus DIR] [--dense-until N]
           [--sparse-only] [--profile] [--tree SPEC]
           qwen3: --kv-cache int8|f32|fp8 --weights f32|int8 --yarn FACTOR
           ricursive: --kv-cache f32|bf16|int8|fp8 (default f32) --weights SPEC (int8, fp8, bf16,
           f32 or pattern=format; default: as the checkpoint stores them, so the NVFP4 release
           stays NVFP4). On several cards the layers are staged over the fewest that hold the
           deployment (each stage reads the global layers it owns sparsely).

  --tree size=4,blocks=1024:4096,beam=2048:2048,page=4096,growth=1024[,start=N]
           reads through node bucket levels (qwen3).

  The corpus is every text file under DIR (sorted by path), tokenised and
  concatenated; the first documents train, the last are held out.
";

const BLOCK: u32 = 16;
/// Tokens of the warm-up prefill before each timed one.
const WARM_TOKENS: usize = 1100;
const SWITCHES: [&str; 3] = ["profile", "sparse-only", "help"];

struct Args(HashMap<String, String>);

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

/// The engine surface the verb needs beyond the capture contract.
trait LsaEngine: AttentionCapture + PagedDecoder {
    fn gpu(&self) -> &gpu_core::Gpu;
    fn lsa_dense_until(&self) -> u32;
    fn set_lsa_dense_until(&mut self, position: u32);
}

impl LsaEngine for qwen3::serve::Engine {
    fn gpu(&self) -> &gpu_core::Gpu {
        qwen3::serve::Engine::gpu(self)
    }
    fn lsa_dense_until(&self) -> u32 {
        qwen3::serve::Engine::lsa_dense_until(self)
    }
    fn set_lsa_dense_until(&mut self, position: u32) {
        qwen3::serve::Engine::set_lsa_dense_until(self, position)
    }
}

impl LsaEngine for ricursive::RicursiveEngine {
    fn gpu(&self) -> &gpu_core::Gpu {
        ricursive::RicursiveEngine::gpu(self)
    }
    fn lsa_dense_until(&self) -> u32 {
        ricursive::RicursiveEngine::lsa_dense_until(self)
    }
    fn set_lsa_dense_until(&mut self, position: u32) {
        ricursive::RicursiveEngine::set_lsa_dense_until(self, position)
    }
}

pub fn run_lsa(argv: &[String]) -> ! {
    let result = (|| -> Result<(), String> {
        let mode = argv.first().ok_or("a mode is required: distill, eval or prefill")?;
        let args = Args::parse(&argv[1..])?;
        if args.flag("help") {
            print!("{USAGE}");
            return Ok(());
        }
        let set = gpu_core::select_compute(args.get("device").unwrap_or("gpu"), args.get("backend"))?;
        set.apply()?;
        gpu_core::publish_compute_set(set);
        let path = checkpoint(&args)?;
        match architecture(&args, &path)?.as_str() {
            "qwen3" => dispatch(mode, &args, &path, &QwenOpener { path: path.clone(), args: &args }),
            "ricursive" => dispatch(mode, &args, &path, &RippleOpener::new(&path, &args)?),
            other => Err(format!("brain lsa: architecture {other:?} has no engine that captures attention (qwen3 and ricursive do)")),
        }
    })();
    match result {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("brain lsa: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    }
}

/// Opens the engine of one architecture for a prompt of `max_tokens`.
trait Opener {
    type Engine: LsaEngine;
    fn open(&self, max_tokens: u32) -> Result<Self::Engine, String>;
    /// The same engine for the timed prefill: the KV tier, weights and YaRN options apply.
    fn open_timed(&self, max_tokens: u32) -> Result<Self::Engine, String> {
        self.open(max_tokens)
    }
}

fn dispatch<O: Opener>(mode: &str, args: &Args, path: &str, opener: &O) -> Result<(), String> {
    match mode {
        "distill" => distill(args, path, opener),
        "eval" => eval(args, path, opener),
        "prefill" => prefill(args, path, opener),
        other => Err(format!("unknown mode {other:?}: distill, eval or prefill")),
    }
}

struct QwenOpener<'a> {
    path: String,
    args: &'a Args,
}

impl Opener for QwenOpener<'_> {
    type Engine = qwen3::serve::Engine;
    fn open(&self, max_tokens: u32) -> Result<Self::Engine, String> {
        qwen_engine(&self.path, max_tokens, false, false, Some(false), None)
    }
    fn open_timed(&self, max_tokens: u32) -> Result<Self::Engine, String> {
        let a = self.args;
        let weights = match a.get("weights") {
            None => None,
            Some("f32") => Some(false),
            Some("int8") => Some(true),
            Some(other) => return Err(format!("--weights {other:?}: f32 or int8")),
        };
        let (int8_kv, fp8) = match a.get("kv-cache").unwrap_or("int8") {
            "f32" => (false, false),
            "int8" => (true, false),
            "fp8" => (true, true),
            other => return Err(format!("--kv-cache {other:?}: f32, int8 or fp8")),
        };
        let yarn = a.get("yarn").map(|v| v.parse::<f32>().map_err(|_| format!("--yarn {v:?} is not a factor"))).transpose()?;
        qwen_engine(&self.path, max_tokens, int8_kv, fp8, weights, yarn)
    }
}

/// An engine for prompts of `max_tokens`, its KV cache int8 (fp8) when asked, its linears int8 when `int8` says so
/// (by default from the parameter count up that serving quantizes). Past the checkpoint's native positions it
/// rotates under YaRN by `yarn`, by default the smallest power of two that covers `max_tokens`.
fn qwen_engine(path: &str, max_tokens: u32, kv_int8: bool, kv_fp8: bool, int8: Option<bool>, yarn: Option<f32>) -> Result<qwen3::serve::Engine, String> {
    let (mut cfg, src) = qwen3::open_checkpoint(path)?;
    let native = cfg.max_position_embeddings;
    if cfg.rope_scaling.is_none() && (yarn.is_some() || max_tokens > native) {
        let factor = yarn.unwrap_or_else(|| max_tokens.div_ceil(native).next_power_of_two() as f32);
        println!("YaRN x{factor} over {native} native positions");
        cfg.rope_scaling = Some(model::rope_scaling::RopeScaling::Yarn(model::yarn::YarnConfig::new(factor, native)));
    }
    let tier = if int8.unwrap_or_else(|| qwen3::footprint::int8_by_default(&cfg)) { qwen3::Dtype::I8 } else { qwen3::Dtype::F32 };
    let mbt = max_tokens.div_ceil(BLOCK);
    // Room for the warm-up prompt the prefix cache keeps beside the measured one.
    let pool = mbt + (WARM_TOKENS as u32).div_ceil(BLOCK) + 8;
    let mut e = qwen3::serve::Engine::from_source_tier(cfg, &*src, BLOCK, pool, 1, mbt, 512, kv_int8, tier);
    if kv_fp8 {
        e.set_kv_fp8()?;
    }
    Ok(e)
}

struct RippleOpener {
    path: String,
    cfg: ricursive::RicursiveConfig,
    source_layers: Vec<usize>,
    /// `--weights` and `--kv-cache`, checked against what Ripple declares (`ricursive::POLICY`).
    policy: model::policy::ServePolicy,
}

impl RippleOpener {
    fn new(path: &str, args: &Args) -> Result<RippleOpener, String> {
        for flag in ["yarn", "tree"] {
            if args.get(flag).is_some() {
                return Err(format!("--{flag} applies to qwen3 only (ricursive reads a flat index over unscaled positions)"));
            }
        }
        let mut policy = model::policy::ServePolicy::default();
        if let Some(w) = args.get("weights") {
            policy.set("--weight-format", w)?;
        }
        if let Some(kv) = args.get("kv-cache") {
            policy.set("--kv-cache", kv)?;
        }
        ricursive::POLICY.check(&policy)?;
        let config = config_path(path)?;
        let full = ricursive::RicursiveConfig::from_file(&config)?;
        let (cfg, source_layers) = match args.get("layers") {
            None => (full.clone(), (0..full.num_hidden_layers).collect()),
            Some(list) => {
                let layers: Vec<usize> = list.split(',').map(|v| v.trim().parse().map_err(|_| format!("--layers {list:?}: {v:?} is not a layer"))).collect::<Result<_, _>>()?;
                (full.truncated(&layers)?, layers)
            }
        };
        Ok(RippleOpener { path: path.to_string(), cfg, source_layers, policy })
    }

    /// The engine for prompts of `max_tokens`, its cache at `kv`, staged over the cards the device selection offers.
    fn engine(&self, max_tokens: u32, kv: model::kv_tier::KvTier) -> Result<ricursive::RicursiveEngine, String> {
        let src = ricursive::open_shared_checkpoint(&self.path, &self.cfg, &self.source_layers)?;
        let block_size = BLOCK;
        let mut serve = ricursive::ServeConfig { sparse_attention: true, kv_cache: kv, ..ricursive::ServeConfig::new(block_size, 2 * max_tokens.div_ceil(block_size) + 8, max_tokens, 1, 0, 64) };
        serve.apply_policy(&model::policy::ServePolicy { kv_cache: None, ..self.policy.clone() })?;
        let tiers = ricursive::resolve_tiers(&self.cfg, src.as_ref(), serve.weights.as_ref());
        let plan = ricursive::placement::auto_plan(&self.cfg, &serve, &tiers, ricursive::LsaCharge::default(), None)?;
        println!("ricursive: {} layers over {} stage(s) [{}], weights {}, KV {}", self.cfg.num_hidden_layers, plan.n_stages(), plan.describe(), tiers.summary(), serve.kv_cache);
        ricursive::RicursiveEngine::from_source_staged(self.cfg.clone(), src, serve, None, &plan)
    }
}

impl Opener for RippleOpener {
    type Engine = ricursive::RicursiveEngine;
    /// Distillation and evaluation read unquantized keys: an f32 cache.
    fn open(&self, max_tokens: u32) -> Result<Self::Engine, String> {
        self.engine(max_tokens, model::kv_tier::KvTier::F32)
    }
    fn open_timed(&self, max_tokens: u32) -> Result<Self::Engine, String> {
        self.engine(max_tokens, self.policy.kv_cache.unwrap_or(model::kv_tier::KvTier::F32))
    }
}

/// The checkpoint to load: `--checkpoint`, or `--model`'s checkout in the model store.
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

/// The directory holding the checkpoint's `config.json` and tokenizer: the checkpoint itself if it is one, else its parent.
fn model_dir(path: &str) -> String {
    let p = std::path::Path::new(path);
    if p.is_dir() { path.to_string() } else { p.parent().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default() }
}

fn config_path(path: &str) -> Result<String, String> {
    let config = format!("{}/config.json", model_dir(path));
    if std::path::Path::new(&config).exists() { Ok(config) } else { Err(format!("no config.json beside {path}")) }
}

/// The architecture id: `--arch`, else what the checkpoint's `config.json` names.
fn architecture(args: &Args, path: &str) -> Result<String, String> {
    if let Some(a) = args.get("arch") {
        return brain_arch::by_id(a).map(|x| x.implementation().id.to_string()).ok_or_else(|| format!("--arch {a:?} is not an architecture"));
    }
    let config = std::fs::read_to_string(config_path(path)?).map_err(|e| e.to_string())?;
    let json: serde_json::Value = serde_json::from_str(&config).map_err(|e| format!("config.json: {e}"))?;
    let classes: Vec<&str> = json["architectures"].as_array().into_iter().flatten().filter_map(|v| v.as_str()).collect();
    classes
        .iter()
        .find_map(|c| brain_arch::by_hf_config(c, |k| json.get(k).is_some()))
        .map(|a| a.implementation().id.to_string())
        .ok_or_else(|| format!("config.json names {classes:?}, which is no registered architecture; pass --arch"))
}

fn seed(args: &Args) -> Result<u64, String> {
    let s = match args.get("seed") {
        Some(v) => v.trim().parse::<u64>().map_err(|_| format!("--seed {v:?} is not a number"))?,
        None => data::rng::random_seed(),
    };
    println!("seed {s}");
    Ok(s)
}

/// The corpus as documents of `tokens` tokens: every text file under `dir` in path order, tokenised and concatenated.
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
        let Ok(bytes) = std::fs::read(&f) else { continue };
        // A file cut mid-character (`head -c` on UTF-8 text) is still a document: decode what is there and say so,
        // rather than dropping it and failing later with "holds only N documents".
        let text = String::from_utf8_lossy(&bytes);
        if matches!(text, std::borrow::Cow::Owned(_)) {
            eprintln!("brain lsa: {} is not valid UTF-8; invalid bytes are read as U+FFFD", f.display());
        }
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

fn tokenizer(path: &str) -> Result<data::qwen_tokenizer::QwenBpe, String> {
    data::qwen_tokenizer::QwenBpe::from_dir(&model_dir(path))
}

fn lsa_config(args: &Args) -> Result<LsaConfig, String> {
    let d = LsaConfig::default();
    let cfg = LsaConfig {
        top_k: args.num("top-k", 1024)?,
        top_p: args.num("top-p", 256)?,
        index_dim: args.num("index-dim", d.index_dim)?,
        cluster_block: args.num("cluster-block", d.cluster_block)?,
        window: args.num("window", d.window)?,
        sink: args.num("sink", d.sink)?,
        ..d
    };
    cfg.validate()?;
    Ok(cfg)
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

fn distill<O: Opener>(args: &Args, path: &str, opener: &O) -> Result<(), String> {
    let tokens: usize = args.num("tokens", 8192)?;
    let (train_docs, epochs, queries): (usize, usize, usize) = (args.num("train-docs", 48)?, args.num("epochs", 3)?, args.num("queries", 64)?);
    let cfg = lsa_config(args)?;
    let mut rng = Lcg::new(seed(args)?);
    let docs = documents(args.req("corpus")?, &tokenizer(path)?, tokens, train_docs)?;
    let mut e = opener.open(tokens as u32)?;

    // PCA initialisation from the first documents' keys.
    let t0 = Instant::now();
    let init = capture::pca_indexer(&mut e, cfg, &docs[..docs.len().min(2)])?;
    println!("pca init from {} documents in {:.1}s", docs.len().min(2), t0.elapsed().as_secs_f64());

    let opts = DistillOptions { epochs, queries, lr: args.num("lr", 1e-3)?, ..DistillOptions::default() };
    let mut clock = Instant::now();
    let d = capture::distill(&mut e, &init, &docs, &opts, &mut rng, |epoch, doc, kl| {
        println!("epoch {epoch} doc {doc}: mean KL over sparse layers {kl:.4} ({:.1}s)", clock.elapsed().as_secs_f64());
        clock = Instant::now();
    })?;
    let out = args.req("out")?;
    let indexer = Indexer { tree: tree_config(args)?, ..d.indexer() };
    indexer.save(out)?;
    println!("wrote {out}; temperatures {:?}", d.temperatures().iter().map(|t| t.map(|v| (v * 1000.0).round() / 1000.0)).collect::<Vec<_>>());
    Ok(())
}

fn eval<O: Opener>(args: &Args, path: &str, opener: &O) -> Result<(), String> {
    let tokens: usize = args.num("tokens", 8192)?;
    let (held, queries, tail): (usize, usize, usize) = (args.num("held-docs", 6)?, args.num("queries", 32)?, args.num("tail", 256)?);
    let cfg = lsa_config(args)?;
    let mut rng = Lcg::new(seed(args)?);
    let tok = tokenizer(path)?;
    let skip: usize = args.num("skip-docs", 48)?;
    let all = documents(args.req("corpus")?, &tok, tokens, skip + held)?;
    let held_docs = &all[skip..];
    let mut e = opener.open(tokens as u32)?;
    let geometry = e.index_geometry();
    let capable = e.sparse_capable();
    let mut indexers: Vec<(String, Indexer)> = Vec::new();
    for spec in args.req("indexer")?.split(',') {
        let ix = match spec {
            "random" => Indexer::random_for(cfg, geometry, rng.next_u64(), &capable)?,
            "pca" => {
                let fit = match args.get("pca-corpus") {
                    Some(dir) => documents(dir, &tok, tokens, 2)?,
                    None => all[..2].to_vec(),
                };
                capture::pca_indexer(&mut e, cfg, &fit)?
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

    let refs: Vec<&Indexer> = indexers.iter().map(|(_, ix)| ix).collect();
    let r = capture::retention_report(&mut e, held_docs, &refs, &cfg, queries, &mut rng)?;
    println!("\nretained share of dense attention (held-out, {} docs x {} queries x {} sparse layers):", held_docs.len(), queries, capable.iter().filter(|c| **c).count());
    println!("  window {} + sinks {} only: {:.4}", cfg.window, cfg.sink, r.local);
    for ((name, _), share) in indexers.iter().zip(&r.indexers) {
        println!("  {name:<40} {share:.4}  (top_k {}, top_p {})", cfg.top_k, cfg.top_p);
    }

    let dense = capture::tail_loss(&mut e, held_docs, tail);
    println!("\nheld-out next-token loss over the last {tail} positions of {} docs:", held_docs.len());
    println!("  dense                                    {dense:.4}  (ppl {:.3})", dense.exp());
    for (name, ix) in &indexers {
        if let Err(why) = e.set_lsa(ix) {
            println!("  {name:<40} not measured: {why}");
            continue;
        }
        let v = capture::tail_loss(&mut e, held_docs, tail);
        println!("  {name:<40} {v:.4}  (ppl {:.3}, {:+.4} vs dense)", v.exp(), v - dense);
    }
    Ok(())
}

fn prefill<O: Opener>(args: &Args, path: &str, opener: &O) -> Result<(), String> {
    let lens: Vec<usize> = args.get("tokens").unwrap_or("4096,8192,16384,32768").split(',').map(|s| s.parse().map_err(|_| format!("--tokens {s:?}"))).collect::<Result<_, _>>()?;
    let max = *lens.iter().max().ok_or("--tokens is empty")?;
    let cfg = lsa_config(args)?;
    let mut rng = Lcg::new(seed(args)?);
    let doc: Vec<u32> = match args.get("corpus") {
        Some(dir) => documents(dir, &tokenizer(path)?, max, 1)?.remove(0),
        None => (0..max).map(|_| 200 + rng.next_u32() % 20_000).collect(),
    };
    let probe = opener.open(1024)?;
    let (geometry, capable) = (probe.index_geometry(), probe.sparse_capable());
    drop(probe);
    let mut ix = match args.req("indexer")? {
        "random" => Indexer::random_for(cfg, geometry, rng.next_u64(), &capable)?,
        file => Indexer::load(file)?,
    };
    ix.cfg.top_k = cfg.top_k;
    ix.cfg.top_p = cfg.top_p;
    if let Some(tree) = tree_config(args)? {
        println!("through a tree: {tree:?}");
        ix.tree = Some(tree);
    }
    println!("prefill of one sequence, {path}, top_k {} top_p {} window {}", cfg.top_k, cfg.top_p, cfg.window);
    for &n in &lens {
        let prompt = &doc[..n];
        // One warm-up prefill of other tokens (pipelines, allocator), then the timed one: a repeat of the same
        // prompt would be served by the prefix cache and time nothing.
        let warm: Vec<u32> = prompt[..prompt.len().min(WARM_TOKENS)].iter().map(|t| (t + 7) % 20_000 + 200).collect();
        let time = |e: &mut O::Engine| -> f64 {
            let mut table = model::paged::BlockTable::new();
            e.prefill(&mut table, &warm);
            e.release_table(&mut table);
            let mut table = model::paged::BlockTable::new();
            let t = Instant::now();
            e.prefill(&mut table, prompt);
            let secs = t.elapsed().as_secs_f64();
            e.release_table(&mut table);
            secs
        };
        let skip_dense = args.flag("sparse-only");
        let td = if skip_dense { None } else { Some(time(&mut opener.open_timed(n as u32)?)) };
        let mut sparse = opener.open_timed(n as u32)?;
        sparse.set_lsa(&ix)?;
        if let Some(v) = args.get("dense-until") {
            sparse.set_lsa_dense_until(v.parse().map_err(|_| format!("--dense-until {v:?} is not a number"))?);
        }
        let crossover = sparse.lsa_dense_until();
        let ts = time(&mut sparse);
        match td {
            Some(td) => println!("T={n:>6}  dense {:>8.2} s  sparse {:>8.2} s  speedup {:>5.2}x  ({:.0} vs {:.0} tok/s)  dense below {crossover}", td, ts, td / ts, n as f64 / td, n as f64 / ts),
            None => println!("T={n:>6}  sparse {:>8.2} s  ({:.0} tok/s)  dense below {crossover}; dense not timed", ts, n as f64 / ts),
        }
        if args.flag("profile") {
            // Other tokens of the same length: the timed prompt sits in the prefix cache.
            let other: Vec<u32> = prompt.iter().map(|t| (t + 13) % 20_000 + 200).collect();
            profile(&mut sparse, &format!("sparse T={n}"), &other);
            drop(sparse);
            if !skip_dense {
                profile(&mut opener.open_timed(n as u32)?, &format!("dense T={n}"), &other);
            }
        }
    }
    Ok(())
}

/// One prefill of `prompt` with the device timing every kernel: the per-kernel table, then the time by part of
/// attention and the wall time the device did not account for (host work between dispatches).
fn profile<E: LsaEngine>(e: &mut E, label: &str, prompt: &[u32]) {
    if !e.gpu().set_kernel_timing(true) {
        println!("{label}: this backend does not time kernels");
        return;
    }
    // The earlier prompts' cached blocks go: the pool holds about two.
    e.reclaim_prefix(u32::MAX);
    e.gpu().reset_kernel_times();
    let mut table = model::paged::BlockTable::new();
    let t = Instant::now();
    e.prefill(&mut table, prompt);
    let wall_ms = t.elapsed().as_secs_f64() * 1e3;
    e.release_table(&mut table);
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
