// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improvement loops that train a large
// model on its own verified work and serve the result for its clients. If your
// team needs expertise in LoRA fine-tuning of Qwen3.8-class models on one GPU
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! `brain::ChatFineTune` from the command line:
//!
//! ```text
//! chat_finetune --base DIR|FILE --dataset FILE.jsonl --out DIR --arch qwen3|qwen35
//!     [--models-dir DIR] [--tier f32|bf16|nf4|int8] [--rank R] [--alpha A] [--steps N]
//!     [--lr X] [--seed S] [--grad-accum G] [--adapter-id ID] [--device cpu|gpu0|gpu1]
//!     [--backend cuda|cpu|vulkan|wgpu]
//! ```
//!
//! Prints one line per step and, at the end, the adapter path and digest.

use brain::{ChatFineTune, Device};

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut base, mut dataset, mut out, mut arch) = (None, None, None, "qwen3".to_string());
    let (mut models_dir, mut tier, mut adapter_id, mut device, mut backend) = (None, "f32".to_string(), None, "cpu".to_string(), None);
    let (mut rank, mut alpha, mut steps, mut lr, mut seed, mut accum) = (None, None, 100u32, 3e-4f32, None, 1u32);
    while let Some(flag) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| die(&format!("{flag} needs a value")));
        match flag.as_str() {
            "--base" => base = Some(value()),
            "--dataset" => dataset = Some(value()),
            "--out" => out = Some(value()),
            "--arch" => arch = value(),
            "--models-dir" => models_dir = Some(value()),
            "--tier" => tier = value(),
            "--adapter-id" => adapter_id = Some(value()),
            "--device" => device = value(),
            "--backend" => backend = Some(value()),
            "--rank" => rank = Some(parse(&flag, &value())),
            "--alpha" => alpha = Some(parse(&flag, &value())),
            "--steps" => steps = parse(&flag, &value()),
            "--lr" => lr = parse(&flag, &value()),
            "--seed" => seed = Some(parse(&flag, &value())),
            "--grad-accum" => accum = parse(&flag, &value()),
            other => die(&format!("unknown flag {other}")),
        }
    }
    let (Some(base), Some(dataset), Some(out)) = (base, dataset, out) else { die("--base, --dataset and --out are required") };
    if let Some(backend) = backend {
        brain::select_backend(&backend).unwrap_or_else(|e| die(&format!("--backend: {e}")));
    }
    let mut run = ChatFineTune::from_pretrained(base)
        .arch(arch)
        .dataset(dataset)
        .out_dir(out)
        .steps(steps)
        .lr(lr)
        .grad_accum(accum)
        .device(Device::parse(&device).unwrap_or_else(|e| die(&format!("--device: {e}"))));
    run = match tier.as_str() {
        "f32" => run,
        "bf16" => run.bf16_base(true),
        "nf4" => run.nf4_base(true),
        "int8" => run.int8_base(true),
        other => die(&format!("--tier {other:?}: expected f32, bf16, nf4 or int8")),
    };
    if let Some(dir) = models_dir {
        run = run.models_dir(dir);
    }
    if let Some(id) = adapter_id {
        run = run.adapter_id(id);
    }
    if let Some(rank) = rank {
        run = run.rank(rank);
    }
    if let Some(alpha) = alpha {
        run = run.alpha(alpha);
    }
    if let Some(seed) = seed {
        run = run.seed(seed);
    }
    let started = std::time::Instant::now();
    let outcome = run
        .run_with(&brain::CancelToken::default(), |p| eprintln!("step {}/{} loss {:.4} lr {:.2e} t={:.1}s", p.step, p.steps, p.loss, p.lr, started.elapsed().as_secs_f64()))
        .unwrap_or_else(|e| die(&format!("fine-tune failed: {e}")));
    println!(
        "status={:?} steps={} block={} loss {:?} -> {:?} seed={} adapter={:?} digest={:?} wall={:.1}s",
        outcome.status,
        outcome.steps_completed,
        outcome.block,
        outcome.initial_loss,
        outcome.final_loss,
        outcome.seed,
        outcome.adapter,
        outcome.adapter_digest,
        started.elapsed().as_secs_f64()
    );
}

fn parse<T: std::str::FromStr>(flag: &str, v: &str) -> T {
    v.parse().unwrap_or_else(|_| die(&format!("{flag} {v:?}: not a valid value")))
}

fn die(why: &str) -> ! {
    eprintln!("chat_finetune: {why}");
    std::process::exit(2);
}
