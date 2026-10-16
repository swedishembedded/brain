// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in finding the longest sequence a model trains at on a given
// card then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Sizing probe for LoRA/QLoRA training of Qwen3.8-27B-width layers: builds
//! the first `--layers` layers of the real shape (d_model 5120, vocab 248320,
//! 3 Gated-DeltaNet layers per GQA layer) with random weights and no
//! checkpoint, runs `--steps` LoRA steps at `--tokens` tokens and prints the
//! device memory and wall time. Memory is the process's own GPU footprint as
//! the driver reports it (`nvidia-smi`), sampled every 50 ms from a helper
//! thread, so the peak is the highest sample; the steady figure is taken
//! after the build. Everything is on stdout as `PROBE key=value` lines.
//!
//! ```text
//! BRAIN_BACKEND=cuda gpu qwen35_train_probe --tokens 4096 --base nf4 \
//!     [--layers 4] [--rank 16] [--recompute on|off] [--gdn-segment N] \
//!     [--head-budget WORDS] [--flash-min-t N] [--bf16-residuals on|off] [--mlp-rows N] [--steps 2]
//! ```
//!
//! `--base` is bf16, nf4 or f32. `--gdn-segment 0` (the default) keeps a GDN
//! layer's whole-sequence intermediates; a positive value segments it.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use checkpoint::TensorSource;
use data::rng::{Lcg, Rng};
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35, TrainOpts};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn parsed<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    match arg(args, name) {
        Some(v) => v.parse().unwrap_or_else(|_| panic!("{name}: cannot parse {v:?}")),
        None => default,
    }
}

/// Random weights of the right shapes, generated one tensor at a time so the
/// host never holds the model: the probe measures the device, not the data.
struct SyntheticSource {
    sizes: HashMap<String, usize>,
    seed: u64,
}

impl TensorSource for SyntheticSource {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        let Some(&numel) = self.sizes.get(name) else { return false };
        // A per-tensor stream: the same name always gets the same values.
        let tag = name.bytes().fold(self.seed, |h, b| h.wrapping_mul(1099511628211).wrapping_add(b as u64));
        let v: Vec<f32> = if name.ends_with("norm.weight") || name.ends_with(".dt_bias") {
            vec![1.0; numel]
        } else if name.ends_with(".lora_b") {
            vec![0.0; numel]
        } else {
            // Independent streams per slab, filled in parallel: a 1.3G-element
            // table is otherwise minutes of single-threaded generation.
            let a_log = name.ends_with(".A_log");
            let mut v = vec![0f32; numel];
            let slab = numel.div_ceil(std::thread::available_parallelism().map_or(8, |n| n.get())).max(1);
            std::thread::scope(|s| {
                for (i, part) in v.chunks_mut(slab).enumerate() {
                    s.spawn(move || {
                        let mut rng = Lcg::new(tag.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
                        for x in part {
                            *x = if a_log { (rng.unit() * 16.0).max(1e-4).ln() } else { rng.scaled(0.035) };
                        }
                    });
                }
            });
            v
        };
        f(&v);
        true
    }
}

/// This process's GPU memory in MiB as `nvidia-smi` reports it.
fn own_gpu_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi").args(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).output().ok()?;
    let pid = std::process::id().to_string();
    String::from_utf8_lossy(&out.stdout).lines().find_map(|l| {
        let (p, m) = l.split_once(',')?;
        (p.trim() == pid).then(|| m.trim().parse().ok()).flatten()
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tokens: u32 = parsed(&args, "--tokens", 1024);
    let layers: u32 = parsed(&args, "--layers", 4);
    let rank: u32 = parsed(&args, "--rank", 16);
    let steps: u32 = parsed(&args, "--steps", 2);
    let base = arg(&args, "--base").unwrap_or_else(|| "bf16".into());
    let recompute = arg(&args, "--recompute").is_none_or(|v| v != "off");
    let gdn_segment: u32 = parsed(&args, "--gdn-segment", 0);
    let tier = match base.as_str() {
        "f32" => Dtype::F32,
        "bf16" => Dtype::BF16,
        "nf4" => Dtype::NF4,
        other => panic!("--base {other:?}: expected f32, bf16 or nf4"),
    };

    let mut cfg = Qwen35Config::qwen38_27b();
    cfg.n_layers = layers;
    cfg.block_size = tokens;
    cfg.lora = Some(lora_cfg(rank, 2.0 * rank as f32));
    let seed = Rng::random().next_u64();
    println!("PROBE seed={seed} layers={layers} tokens={tokens} base={base} rank={rank} recompute={recompute} gdn_segment={gdn_segment}");
    let src = SyntheticSource { sizes: cfg.param_list().into_iter().collect(), seed };
    let opts = TrainOpts {
        base: TierPolicy::uniform(tier),
        recompute,
        head_budget_words: arg(&args, "--head-budget").map(|v| v.parse().expect("--head-budget WORDS")),
        flash_min_t: arg(&args, "--flash-min-t").map(|v| v.parse().expect("--flash-min-t N")),
        gdn_segment_tokens: (gdn_segment > 0).then_some(gdn_segment),
        bf16_residuals: arg(&args, "--bf16-residuals").is_some_and(|v| v == "on"),
        mlp_rows: arg(&args, "--mlp-rows").map(|v| v.parse().expect("--mlp-rows N")),
    };

    let peak = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = {
        let (peak, stop) = (peak.clone(), stop.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if let Some(mib) = own_gpu_mib() {
                    peak.fetch_max(mib, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        })
    };

    let t0 = Instant::now();
    let m = Qwen35::new_train_opts_on(Gpu::new(pipelines()), cfg.clone(), 1, tokens, &src, &opts);
    println!("PROBE build_s={:.1}", t0.elapsed().as_secs_f64());
    let steady = own_gpu_mib();
    println!("PROBE built_mib={}", steady.map_or("n/a".to_string(), |v| v.to_string()));

    let mut rng = Lcg::new(seed ^ 0x5eed);
    let x: Vec<u32> = (0..tokens).map(|_| rng.next_u32() % cfg.vocab).collect();
    let y: Vec<u32> = (0..tokens).map(|_| rng.next_u32() % cfg.vocab).collect();
    m.set_batch(&x, &y);
    for step in 1..=steps {
        let t = Instant::now();
        m.zero_grads();
        let loss = m.forward();
        let fwd = t.elapsed().as_secs_f64();
        m.backward();
        m.adamw_step(step, 1e-4, 0.0, Default::default(), Some(1.0), 1.0);
        m.poll_wait();
        let total = t.elapsed().as_secs_f64();
        println!("PROBE step={step} loss={loss:.4} forward_s={fwd:.2} step_s={total:.2} tokens_per_s={:.0}", tokens as f64 / total);
    }
    stop.store(true, Ordering::Relaxed);
    sampler.join().expect("sampler");
    let peak = peak.load(Ordering::Relaxed);
    println!("PROBE peak_mib={peak}");
    if let Some(steady) = steady.filter(|_| peak > 0) {
        println!("PROBE step_mib_per_token={:.3}", peak.saturating_sub(steady) as f64 / tokens as f64);
    }
}
