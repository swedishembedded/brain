// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sparse-MoE Transformer: inference + generation (RMSNorm, RoPE, top-k MoE,
//! tied lm_head). Training lives in `train`. This file is the relocated former
//! `main.rs` inference engine, kept so the two transformers sit side by side.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu, Step};

use crate::train;

pub struct Config {
    pub vocab_size: u32,
    pub block_size: u32,
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub n_experts: u32,
    pub top_k: u32,
    pub d_ff: u32,
}

impl Config {
    fn head_dim(&self) -> u32 {
        self.d_model / self.n_heads
    }
}

struct Weights {
    cfg: Config,
    tensors: HashMap<String, Vec<f32>>,
}

fn load_weights(path: &str) -> Weights {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!(
            "error: cannot read MoE weights '{path}': {e}\n\
             \n\
             Specify a checkpoint with --weights <file>, or train one first:\n\
             \x20 brain train --steps 2000 --out moe.safetensors\n\
             \x20 brain generate --weights moe.safetensors --prompt 1,2,3,4 --max-new 64\n\
             \n\
             (Run `brain help` for all commands.)"
        );
        std::process::exit(1);
    });
    let json_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + json_len]).expect("bad header utf8");
    let json: serde_json::Value = serde_json::from_str(header).expect("bad header json");

    let c = &json["config"];
    let g = |k: &str| c[k].as_u64().unwrap_or_else(|| panic!("missing config.{k}")) as u32;
    let cfg = Config {
        vocab_size: g("vocab_size"),
        block_size: g("block_size"),
        n_layers: g("n_layers"),
        d_model: g("d_model"),
        n_heads: g("n_heads"),
        n_experts: g("n_experts"),
        top_k: g("top_k"),
        d_ff: g("d_ff"),
    };

    let data = &bytes[8 + json_len..];
    let mut tensors = HashMap::new();
    for t in json["tensors"].as_array().expect("tensors array") {
        let name = t["name"].as_str().unwrap().to_string();
        let offset = t["offset"].as_u64().unwrap() as usize;
        let numel = t["numel"].as_u64().unwrap() as usize;
        let floats: Vec<f32> = data[offset * 4..(offset + numel) * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        tensors.insert(name, floats);
    }
    Weights { cfg, tensors }
}

const K_EMBED: usize = 0;
const K_MATMUL: usize = 1;
const K_RMSNORM: usize = 2;
const K_ROPE: usize = 3;
const K_ATTENTION: usize = 4;
const K_ROUTER: usize = 5;
const K_SILU: usize = 6;
const K_SCALE_ADD: usize = 7;
const K_ADD: usize = 8;

pub struct Engine {
    gpu: Gpu,
    weights: HashMap<String, DeviceBuffer>,
    cfg: Config,

    tokens: DeviceBuffer,
    x: DeviceBuffer,
    xn: DeviceBuffer,
    qkv: DeviceBuffer,
    attn_out: DeviceBuffer,
    proj_out: DeviceBuffer,
    router_logits: DeviceBuffer,
    gate: DeviceBuffer,
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    h: DeviceBuffer,
    expert_out: DeviceBuffer,
    moe_acc: DeviceBuffer,
    logits: DeviceBuffer,
}

const ENGINE_PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("matmul", kernels::MATMUL),
    ("rmsnorm", kernels::RMSNORM),
    ("rope", kernels::ROPE),
    ("attention", kernels::ATTENTION),
    ("router_gate", kernels::ROUTER_GATE),
    ("silu_mul", kernels::SILU_MUL),
    ("scale_add", kernels::SCALE_ADD),
    ("add", kernels::ADD),
];

impl Engine {
    /// Load an inference [`Engine`] from a checkpoint written by the trainer
    /// (`Trainer::save` / the generic `fit`) or the inference weight format. The
    /// two share the `[u64 LE json_len][json header][f32 blob]` container, with a
    /// tied `lm_head.weight`, so a `fit`-saved MoE checkpoint loads here directly.
    pub fn load(path: &str) -> Engine {
        Engine::new(load_weights(path))
    }

    /// Vocabulary size (logits row width) this engine was built for.
    pub fn vocab_size(&self) -> u32 {
        self.cfg.vocab_size
    }

    /// The model config (layout) this engine was built from.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    fn new(w: Weights) -> Engine {
        // Shared accelerator (wgpu or native CPU) — same plumbing as the trainer,
        // GPT, and PID models. No bespoke device init here anymore.
        let gpu = Gpu::new(ENGINE_PIPELINES);

        let mut weights = HashMap::new();
        for (name, data) in &w.tensors {
            weights.insert(name.clone(), gpu.storage_init(name, data));
        }

        let c = &w.cfg;
        let bs = c.block_size as u64;
        let storage = |n: u64| gpu.storage(n);

        let tokens = gpu.storage(bs);

        let d = c.d_model as u64;
        let ff = c.d_ff as u64;
        let engine = Engine {
            weights,
            tokens,
            x: storage(bs * d),
            xn: storage(bs * d),
            qkv: storage(bs * 3 * d),
            attn_out: storage(bs * d),
            proj_out: storage(bs * d),
            router_logits: storage(bs * c.n_experts as u64),
            gate: storage(bs * c.n_experts as u64),
            gate_pre: storage(bs * ff),
            up: storage(bs * ff),
            h: storage(bs * ff),
            expert_out: storage(bs * d),
            moe_acc: storage(bs * d),
            logits: storage(bs * c.vocab_size as u64),
            cfg: w.cfg,
            gpu,
        };
        engine
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.weights
            .get(name)
            .unwrap_or_else(|| panic!("missing weight tensor: {name}"))
    }

    /// Per-position logits for one sequence, flattened row-major as
    /// `[len * vocab]` (`logits[t*vocab + i]` is token `i`'s score at position
    /// `t`). This is the full forward output the [`forward`](Self::forward)
    /// last-row convenience is sliced from; benchmark scoring needs every row.
    pub fn logits_all(&self, tokens: &[u32]) -> Vec<f32> {
        let t = tokens.len();
        let vocab = self.cfg.vocab_size as usize;
        let all = self.forward_full(tokens);
        debug_assert_eq!(all.len(), t * vocab);
        all
    }

    /// Logits for the **last** position only (`[vocab]`). The generate/eval path
    /// only needs the next-token distribution; behavior is unchanged.
    fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let vocab = self.cfg.vocab_size as usize;
        let t = tokens.len();
        let all = self.forward_full(tokens);
        all[(t - 1) * vocab..].to_vec()
    }

    /// Run the forward pass and return logits for **all** positions, row-major
    /// `[len * vocab]`. The shared core of [`forward`](Self::forward) (last row)
    /// and [`logits_all`](Self::logits_all) (every row).
    fn forward_full(&self, tokens: &[u32]) -> Vec<f32> {
        let c = &self.cfg;
        let t = tokens.len() as u32;
        assert!(t >= 1 && t <= c.block_size, "sequence length out of range");
        let d = c.d_model;
        let ff = c.d_ff;
        let e = c.n_experts;

        self.gpu.write(&self.tokens, bytemuck::cast_slice(tokens));

        let mut steps: Vec<Step> = Vec::new();

        steps.push(self.gpu.step(
            K_EMBED,
            &[&self.tokens, self.w("token_emb.weight"), &self.x],
            &[d, t],
            t * d,
        ));

        for l in 0..c.n_layers {
            let p = |s: &str| format!("blocks.{l}.{s}");
            steps.push(self.gpu.step(
                K_RMSNORM,
                &[&self.x, self.w(&p("norm1.weight")), &self.xn],
                &[d, t],
                t,
            ));
            steps.push(self.gpu.step(
                K_MATMUL,
                &[&self.xn, self.w(&p("attn.qkv.weight")), &self.qkv],
                &[t, d, 3 * d],
                t * 3 * d,
            ));
            let half = c.head_dim() / 2;
            steps.push(self.gpu.step(
                K_ROPE,
                &[&self.qkv],
                &[t, c.n_heads, c.head_dim(), 3 * d, 0],
                t * c.n_heads * half,
            ));
            steps.push(self.gpu.step(
                K_ROPE,
                &[&self.qkv],
                &[t, c.n_heads, c.head_dim(), 3 * d, d],
                t * c.n_heads * half,
            ));
            steps.push(self.gpu.step(
                K_ATTENTION,
                &[&self.qkv, &self.attn_out],
                &[t, c.n_heads, c.head_dim(), 3 * d, 0, d, 2 * d, d],
                t * c.n_heads,
            ));
            steps.push(self.gpu.step(
                K_MATMUL,
                &[&self.attn_out, self.w(&p("attn.out.weight")), &self.proj_out],
                &[t, d, d],
                t * d,
            ));
            steps.push(self.gpu.step(K_ADD, &[&self.proj_out, &self.x], &[t * d], t * d));

            steps.push(self.gpu.step(
                K_RMSNORM,
                &[&self.x, self.w(&p("norm2.weight")), &self.xn],
                &[d, t],
                t,
            ));
            steps.push(self.gpu.step(
                K_MATMUL,
                &[&self.xn, self.w(&p("moe.router.weight")), &self.router_logits],
                &[t, d, e],
                t * e,
            ));
            steps.push(self.gpu.step(
                K_ROUTER,
                &[&self.router_logits, &self.gate],
                &[t, e, c.top_k],
                t,
            ));
            for ei in 0..e {
                let ep = |s: &str| format!("blocks.{l}.moe.experts.{ei}.{s}");
                steps.push(self.gpu.step(
                    K_MATMUL,
                    &[&self.xn, self.w(&ep("w_gate.weight")), &self.gate_pre],
                    &[t, d, ff],
                    t * ff,
                ));
                steps.push(self.gpu.step(
                    K_MATMUL,
                    &[&self.xn, self.w(&ep("w_up.weight")), &self.up],
                    &[t, d, ff],
                    t * ff,
                ));
                steps.push(self.gpu.step(
                    K_SILU,
                    &[&self.gate_pre, &self.up, &self.h],
                    &[t * ff],
                    t * ff,
                ));
                steps.push(self.gpu.step(
                    K_MATMUL,
                    &[&self.h, self.w(&ep("w_down.weight")), &self.expert_out],
                    &[t, ff, d],
                    t * d,
                ));
                let accumulate = if ei == 0 { 0 } else { 1 };
                steps.push(self.gpu.step(
                    K_SCALE_ADD,
                    &[&self.gate, &self.expert_out, &self.moe_acc],
                    &[t, d, e, ei, accumulate],
                    t * d,
                ));
            }
            steps.push(self.gpu.step(K_ADD, &[&self.moe_acc, &self.x], &[t * d], t * d));
        }

        steps.push(self.gpu.step(
            K_RMSNORM,
            &[&self.x, self.w("norm.weight"), &self.xn],
            &[d, t],
            t,
        ));
        steps.push(self.gpu.step(
            K_MATMUL,
            &[&self.xn, self.w("lm_head.weight"), &self.logits],
            &[t, d, c.vocab_size],
            t * c.vocab_size,
        ));

        self.gpu.submit(&[], &steps);

        // Read the full logits buffer (gpu_core reads from offset 0).
        let vocab = c.vocab_size as usize;
        self.gpu.read(&self.logits, t as usize * vocab)
    }
}

fn xorshift(s: &mut u64) -> u64 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    x
}

fn rand_f32(s: &mut u64) -> f32 {
    (xorshift(s) >> 40) as f32 / (1u64 << 24) as f32
}

fn sample(logits: &[f32], temperature: f32, top_k: Option<usize>, rng: &mut u64) -> u32 {
    let temp = temperature.max(1e-6);
    let mut l: Vec<f32> = logits.iter().map(|&v| v / temp).collect();
    if let Some(k) = top_k {
        if k < l.len() {
            let mut sorted = l.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let thresh = sorted[k - 1];
            for v in l.iter_mut() {
                if *v < thresh {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
    }
    let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = l.iter().map(|&v| (v - mx).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let r = rand_f32(rng);
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

struct Args {
    weights: String,
    prompt: Vec<u32>,
    max_new: usize,
    temperature: f32,
    top_k: Option<usize>,
    seed: u64,
}

fn parse_args() -> Args {
    let mut a = Args {
        weights: "moe.safetensors".to_string(),
        prompt: vec![0, 1, 2, 3],
        max_new: 64,
        temperature: 0.8,
        top_k: None,
        seed: 1234,
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = if argv.get(1).map(|s| s.as_str()) == Some("generate") { 2 } else { 1 };
    while i < argv.len() {
        let next = || argv.get(i + 1).cloned().expect("missing value for flag");
        match argv[i].as_str() {
            "--weights" => a.weights = next(),
            "--prompt" => {
                a.prompt = next()
                    .split(|ch| ch == ',' || ch == ' ')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse().expect("bad prompt token"))
                    .collect()
            }
            "--max-new" => a.max_new = next().parse().expect("bad --max-new"),
            "--temperature" => a.temperature = next().parse().expect("bad --temperature"),
            "--top-k" => a.top_k = Some(next().parse().expect("bad --top-k")),
            "--seed" => a.seed = next().parse().expect("bad --seed"),
            other => panic!("unknown flag: {other}"),
        }
        i += 2;
    }
    if a.prompt.is_empty() {
        a.prompt = vec![0];
    }
    a
}

pub fn run_eval(flags: &[String]) {
    let mut weights_path = "moe_rs.safetensors".to_string();
    let mut seed: u64 = 123;
    let mut samples: usize = 400;
    let mut i = 0;
    while i < flags.len() {
        let next = || flags.get(i + 1).cloned().expect("missing value for flag");
        match flags[i].as_str() {
            "--weights" => weights_path = next(),
            "--seed" => seed = next().parse().unwrap(),
            "--samples" => samples = next().parse().unwrap(),
            other => panic!("unknown eval flag: {other}"),
        }
        i += 2;
    }

    let weights = load_weights(&weights_path);
    let vocab = weights.cfg.vocab_size;
    let bs = weights.cfg.block_size as usize;
    let engine = Engine::new(weights);

    let (corpus, table) = train::corpus_and_table(20_000, vocab, seed);
    let n_val = corpus.len() / 10;
    let (train_region, val_region) = corpus.split_at(corpus.len() - n_val);
    let new_orbit = train::orbit(&table, vocab, 1500, 41, 7);

    let acc = |seq: &[u32], ctx: usize, skip_reset: bool| -> f32 {
        let ctx = ctx.min(bs);
        let (mut correct, mut total) = (0usize, 0usize);
        for j in (ctx - 1)..(seq.len() - 1) {
            if skip_reset && (j + 1) % 257 == 0 {
                continue;
            }
            let logits = engine.forward(&seq[j - ctx + 1..=j]);
            let pred = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32;
            correct += (pred == seq[j + 1]) as usize;
            total += 1;
            if total >= samples {
                break;
            }
        }
        correct as f32 / total.max(1) as f32
    };

    let lengths: Vec<usize> = [2usize, 8, 16, 32, bs].into_iter().filter(|&l| l <= bs).collect();
    let sweep: Vec<(usize, f32)> = lengths.iter().map(|&l| (l, acc(&new_orbit, l, false))).collect();
    let (best_l, best_acc) = *sweep.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();

    println!("\n=== eval {weights_path} ===  (random = {:.1}%)", 100.0 / vocab as f32);
    println!("accuracy at best context length ({best_l} tokens):");
    println!("  train orbit       : {:5.1}%", 100.0 * acc(train_region, best_l, true));
    println!("  val   orbit       : {:5.1}%", 100.0 * acc(val_region, best_l, true));
    println!("  NEW   orbit       : {:5.1}%", 100.0 * best_acc);
    let s: Vec<String> = sweep.iter().map(|(l, a)| format!("{l}:{:.0}%", 100.0 * a)).collect();
    println!("new-orbit acc by context length: {}", s.join("  "));
}

pub fn run_train(flags: &[String]) {
    let mut a = train::TrainArgs {
        steps: 2000,
        b: 16,
        t: 64,
        lr: 3e-4,
        wd: 0.1,
        seed: 0,
        out: "moe_rs.safetensors".to_string(),
    };
    let mut i = 0;
    while i < flags.len() {
        let next = || flags.get(i + 1).cloned().expect("missing value for flag");
        match flags[i].as_str() {
            "--steps" => a.steps = next().parse().unwrap(),
            "--batch-size" => a.b = next().parse().unwrap(),
            "--block-size" | "--seq-len" => a.t = next().parse().unwrap(),
            "--lr" => a.lr = next().parse().unwrap(),
            "--weight-decay" => a.wd = next().parse().unwrap(),
            "--seed" => a.seed = next().parse().unwrap(),
            "--out" => a.out = next(),
            other => panic!("unknown train flag: {other}"),
        }
        i += 2;
    }
    train::train(a);
}

pub fn run_generate() {
    let args = parse_args();
    if std::env::var("DUMP_LOGITS").is_ok() {
        let weights = load_weights(&args.weights);
        let engine = Engine::new(weights);
        let logits = engine.forward(&args.prompt);
        println!(
            "{}",
            logits.iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>().join(",")
        );
        return;
    }
    let weights = load_weights(&args.weights);
    let block_size = weights.cfg.block_size as usize;
    let vocab = weights.cfg.vocab_size;
    for &tok in &args.prompt {
        assert!(tok < vocab, "prompt token {tok} >= vocab_size {vocab}");
    }

    let engine = Engine::new(weights);
    let mut tokens = args.prompt.clone();
    let mut rng = args.seed.max(1);
    for _ in 0..args.max_new {
        let start = tokens.len().saturating_sub(block_size);
        let logits = engine.forward(&tokens[start..]);
        let next = sample(&logits, args.temperature, args.top_k, &mut rng);
        tokens.push(next);
    }
    println!("{}", tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xorshift_deterministic() {
        let (mut a, mut b) = (12345u64, 12345u64);
        for _ in 0..50 {
            assert_eq!(xorshift(&mut a), xorshift(&mut b));
        }
    }

    // well-mixed seeds: a small seed's first xorshift output is < 2^40, so
    // rand_f32 yields exactly 0.0 (and `sample` then returns index 0). These
    // large seeds exercise the intended cumulative-sampling path.
    const SEEDS: [u64; 3] = [0x9E37_79B9_7F4A_7C15, 0xD1B5_4A32_D192_ED03, 0xCAFE_F00D_DEAD_BEEF];

    #[test]
    fn sample_picks_dominant_logit() {
        // one logit far above the rest -> argmax regardless of the rng draw
        let mut logits = vec![0.0f32; 16];
        logits[5] = 100.0;
        for seed in SEEDS {
            let mut rng = seed;
            assert_eq!(sample(&logits, 1.0, None, &mut rng), 5);
        }
    }

    #[test]
    fn sample_top_k_restricts_support() {
        // top_k=1 must always return the single highest-logit index
        let logits = vec![0.1f32, 0.2, 5.0, 0.3, 0.4];
        for seed in SEEDS {
            let mut rng = seed;
            assert_eq!(sample(&logits, 0.9, Some(1), &mut rng), 2);
        }
    }
}
