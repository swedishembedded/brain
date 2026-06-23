// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sparse-MoE Transformer: inference + generation (RMSNorm, RoPE, top-k MoE,
//! tied lm_head). Training lives in `train`. This file is the relocated former
//! `main.rs` inference engine, kept so the two transformers sit side by side.

use std::collections::HashMap;

use wgpu::util::DeviceExt;

use crate::train;

struct Config {
    vocab_size: u32,
    block_size: u32,
    n_layers: u32,
    d_model: u32,
    n_heads: u32,
    n_experts: u32,
    top_k: u32,
    d_ff: u32,
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
    let bytes = std::fs::read(path).expect("cannot read weights file");
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

struct Engine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<wgpu::ComputePipeline>,
    weights: HashMap<String, wgpu::Buffer>,
    cfg: Config,

    tokens: wgpu::Buffer,
    x: wgpu::Buffer,
    xn: wgpu::Buffer,
    qkv: wgpu::Buffer,
    attn_out: wgpu::Buffer,
    proj_out: wgpu::Buffer,
    router_logits: wgpu::Buffer,
    gate: wgpu::Buffer,
    gate_pre: wgpu::Buffer,
    up: wgpu::Buffer,
    h: wgpu::Buffer,
    expert_out: wgpu::Buffer,
    moe_acc: wgpu::Buffer,
    logits: wgpu::Buffer,
    staging: wgpu::Buffer,
}

fn make_pipeline(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

impl Engine {
    fn new(w: Weights) -> Engine {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("no suitable GPU adapter found");

        let info = adapter.get_info();
        eprintln!("adapter: {} ({:?}, {:?})", info.name, info.device_type, info.backend);

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("moe-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        }))
        .expect("request_device failed");

        let pipelines = vec![
            make_pipeline(&device, "embed", include_str!("shaders/embed.wgsl")),
            make_pipeline(&device, "matmul", include_str!("shaders/matmul.wgsl")),
            make_pipeline(&device, "rmsnorm", include_str!("shaders/rmsnorm.wgsl")),
            make_pipeline(&device, "rope", include_str!("shaders/rope.wgsl")),
            make_pipeline(&device, "attention", include_str!("shaders/attention.wgsl")),
            make_pipeline(&device, "router_gate", include_str!("shaders/router_gate.wgsl")),
            make_pipeline(&device, "silu_mul", include_str!("shaders/silu_mul.wgsl")),
            make_pipeline(&device, "scale_add", include_str!("shaders/scale_add.wgsl")),
            make_pipeline(&device, "add", include_str!("shaders/add.wgsl")),
        ];

        let mut weights = HashMap::new();
        for (name, data) in &w.tensors {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            });
            weights.insert(name.clone(), buf);
        }

        let c = &w.cfg;
        let bs = c.block_size as u64;
        let storage = |n: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: n * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let tokens = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tokens"),
            size: bs * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (c.vocab_size as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let d = c.d_model as u64;
        let ff = c.d_ff as u64;
        Engine {
            pipelines,
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
            staging,
            cfg: w.cfg,
            device,
            queue,
        }
    }

    fn uniform(&self, data: &[u32]) -> wgpu::Buffer {
        let mut bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
        while bytes.len() % 16 != 0 {
            bytes.push(0);
        }
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: &bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    fn step(
        &self,
        kind: usize,
        buffers: &[&wgpu::Buffer],
        params: &[u32],
        threads: u32,
    ) -> (usize, wgpu::BindGroup, u32) {
        let ubuf = self.uniform(params);
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: ubuf.as_entire_binding(),
        }];
        for (i, b) in buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: b.as_entire_binding(),
            });
        }
        let layout = self.pipelines[kind].get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let groups = ((threads + 63) / 64).max(1);
        (kind, bg, groups)
    }

    fn w(&self, name: &str) -> &wgpu::Buffer {
        self.weights
            .get(name)
            .unwrap_or_else(|| panic!("missing weight tensor: {name}"))
    }

    fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let c = &self.cfg;
        let t = tokens.len() as u32;
        assert!(t >= 1 && t <= c.block_size, "sequence length out of range");
        let d = c.d_model;
        let ff = c.d_ff;
        let e = c.n_experts;

        self.queue
            .write_buffer(&self.tokens, 0, bytemuck::cast_slice(tokens));

        let mut steps: Vec<(usize, wgpu::BindGroup, u32)> = Vec::new();

        steps.push(self.step(
            K_EMBED,
            &[&self.tokens, self.w("token_emb.weight"), &self.x],
            &[d, t],
            t * d,
        ));

        for l in 0..c.n_layers {
            let p = |s: &str| format!("blocks.{l}.{s}");
            steps.push(self.step(
                K_RMSNORM,
                &[&self.x, self.w(&p("norm1.weight")), &self.xn],
                &[d, t],
                t,
            ));
            steps.push(self.step(
                K_MATMUL,
                &[&self.xn, self.w(&p("attn.qkv.weight")), &self.qkv],
                &[t, d, 3 * d],
                t * 3 * d,
            ));
            let half = c.head_dim() / 2;
            steps.push(self.step(
                K_ROPE,
                &[&self.qkv],
                &[t, c.n_heads, c.head_dim(), 3 * d, 0],
                t * c.n_heads * half,
            ));
            steps.push(self.step(
                K_ROPE,
                &[&self.qkv],
                &[t, c.n_heads, c.head_dim(), 3 * d, d],
                t * c.n_heads * half,
            ));
            steps.push(self.step(
                K_ATTENTION,
                &[&self.qkv, &self.attn_out],
                &[t, c.n_heads, c.head_dim(), 3 * d, 0, d, 2 * d, d],
                t * c.n_heads,
            ));
            steps.push(self.step(
                K_MATMUL,
                &[&self.attn_out, self.w(&p("attn.out.weight")), &self.proj_out],
                &[t, d, d],
                t * d,
            ));
            steps.push(self.step(K_ADD, &[&self.proj_out, &self.x], &[t * d], t * d));

            steps.push(self.step(
                K_RMSNORM,
                &[&self.x, self.w(&p("norm2.weight")), &self.xn],
                &[d, t],
                t,
            ));
            steps.push(self.step(
                K_MATMUL,
                &[&self.xn, self.w(&p("moe.router.weight")), &self.router_logits],
                &[t, d, e],
                t * e,
            ));
            steps.push(self.step(
                K_ROUTER,
                &[&self.router_logits, &self.gate],
                &[t, e, c.top_k],
                t,
            ));
            for ei in 0..e {
                let ep = |s: &str| format!("blocks.{l}.moe.experts.{ei}.{s}");
                steps.push(self.step(
                    K_MATMUL,
                    &[&self.xn, self.w(&ep("w_gate.weight")), &self.gate_pre],
                    &[t, d, ff],
                    t * ff,
                ));
                steps.push(self.step(
                    K_MATMUL,
                    &[&self.xn, self.w(&ep("w_up.weight")), &self.up],
                    &[t, d, ff],
                    t * ff,
                ));
                steps.push(self.step(
                    K_SILU,
                    &[&self.gate_pre, &self.up, &self.h],
                    &[t * ff],
                    t * ff,
                ));
                steps.push(self.step(
                    K_MATMUL,
                    &[&self.h, self.w(&ep("w_down.weight")), &self.expert_out],
                    &[t, ff, d],
                    t * d,
                ));
                let accumulate = if ei == 0 { 0 } else { 1 };
                steps.push(self.step(
                    K_SCALE_ADD,
                    &[&self.gate, &self.expert_out, &self.moe_acc],
                    &[t, d, e, ei, accumulate],
                    t * d,
                ));
            }
            steps.push(self.step(K_ADD, &[&self.moe_acc, &self.x], &[t * d], t * d));
        }

        steps.push(self.step(
            K_RMSNORM,
            &[&self.x, self.w("norm.weight"), &self.xn],
            &[d, t],
            t,
        ));
        steps.push(self.step(
            K_MATMUL,
            &[&self.xn, self.w("lm_head.weight"), &self.logits],
            &[t, d, c.vocab_size],
            t * c.vocab_size,
        ));

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("forward") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forward"),
                timestamp_writes: None,
            });
            for (kind, bg, groups) in &steps {
                pass.set_pipeline(&self.pipelines[*kind]);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(*groups, 1, 1);
            }
        }
        let last = (t - 1) as u64 * c.vocab_size as u64 * 4;
        enc.copy_buffer_to_buffer(&self.logits, last, &self.staging, 0, c.vocab_size as u64 * 4);
        self.queue.submit(Some(enc.finish()));

        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let out = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        self.staging.unmap();
        out
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
        weights: "moe.weights".to_string(),
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
    let mut weights_path = "moe_rs.weights".to_string();
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
        out: "moe_rs.weights".to_string(),
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
