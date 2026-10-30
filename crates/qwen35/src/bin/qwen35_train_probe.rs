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
// perf-number: the probe's own sampling cadence, not a result
//! the driver reports it (`nvidia-smi`), sampled every 50 ms from a helper
//! thread, so the peak is the highest sample; the steady figure is taken
//! after the build. Everything is on stdout as `PROBE key=value` lines.
//!
//! ```text
//! BRAIN_BACKEND=cuda gpu qwen35_train_probe --tokens 4096 --base nf4 \
//!     [--layers 4] [--rank 16] [--recompute on|off] [--gdn-segment N] \
//!     [--head-budget WORDS] [--flash-min-t N] [--residuals f32|bf16|fp8] [--mlp-rows N] [--steps 2] [--seed N] [--gemm-precision auto|fp32|bf16] [--profile]
//! ```
//!
//! `--gemm-precision` picks the arithmetic of the bf16-weight GEMMs: `auto`
//! (default) uses bf16 tensor cores where the device has them and the fp32
//! kernels otherwise, `fp32` pins the reference kernels.
//!
//! `--profile` times the last step kernel by kernel on the device and prints
//! the 30 most expensive (a time-sliced card inflates small kernels; read the
//! table next to the `other_contexts` count).
//!
//! `--seed` fixes the random weights and tokens so two runs are the same
//! workload; without it a seed is drawn and printed. Each step line carries
//! the number of dispatches the step issued, and the summary line the rate
//! over steps 2.. (step 1 pays the one-off kernel compilation) together with
//! the number of other compute processes `nvidia-smi` saw on the card at the
//! start and end: a rate measured next to another context is not the card's.
//!
//! `--base` is bf16, nf4 or f32. `--gdn-segment 0` (the default) keeps a GDN
//! layer's whole-sequence intermediates; a positive value segments it.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use checkpoint::TensorSource;
use data::rng::{Lcg, Rng};
use gpu_core::provider::library::GemmPrecision;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use model::saved::{SavedFormat, SavedPolicy};
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35, TrainOpts};

const USAGE: &str = "usage: qwen35_train_probe [--tokens N] [--layers N] [--base bf16|nf4|f32] [--rank R] [--steps N] \\
     [--recompute on|off] [--gdn-segment N] [--head-budget WORDS] [--flash-min-t N] [--residuals f32|bf16|fp8] [--mlp-rows N] \\
     [--seed N] [--gemm-precision auto|fp32|bf16] [--profile] [--device gpuN|cpu] [--backend cuda|cpu|vulkan|wgpu]
  Builds the first --layers layers of the real Qwen3.8-27B shape with random weights and times --steps LoRA steps at --tokens
  tokens. Every result is a `PROBE key=value` line on stdout. With BRAIN_PROFILE set, each step also prints the device
  time per kernel kind.";

/// Every flag that takes a value; anything else starting with `--` is refused.
const VALUE_FLAGS: [&str; 15] = [
    "--device",
    "--backend",
    "--tokens",
    "--layers",
    "--base",
    "--rank",
    "--steps",
    "--recompute",
    "--gdn-segment",
    "--head-budget",
    "--flash-min-t",
    "--residuals",
    "--mlp-rows",
    "--seed",
    "--gemm-precision",
];

/// Flags that stand alone.
const SWITCHES: [&str; 1] = ["--profile"];

/// Refuse a flag nothing here reads: a typo would otherwise measure a different configuration than the one asked for.
fn check_flags(args: &[String]) -> Result<(), String> {
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        if SWITCHES.contains(&a.as_str()) {
            continue;
        }
        if !VALUE_FLAGS.contains(&a.as_str()) {
            return Err(format!("unknown argument {a:?}"));
        }
        if it.next().is_none() {
            return Err(format!("{a} requires a value"));
        }
    }
    Ok(())
}

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

/// Compute processes on the card other than this one, as `nvidia-smi` lists them.
fn other_compute_contexts() -> Option<usize> {
    let out = std::process::Command::new("nvidia-smi").args(["--query-compute-apps=pid", "--format=csv,noheader"]).output().ok()?;
    let pid = std::process::id().to_string();
    Some(String::from_utf8_lossy(&out.stdout).lines().map(str::trim).filter(|l| !l.is_empty() && *l != pid).count())
}

/// CPU seconds (user + system, all threads) this process has used, from `/proc`.
fn cpu_seconds() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // Fields after the parenthesised command name; utime and stime are the 12th and 13th.
    let rest = stat.rsplit_once(')')?.1;
    let f: Vec<&str> = rest.split_whitespace().collect();
    let ticks = f.get(11)?.parse::<f64>().ok()? + f.get(12)?.parse::<f64>().ok()?;
    Some(ticks / 100.0)
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
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return;
    }
    if let Err(e) = check_flags(&args) {
        eprintln!("qwen35_train_probe: {e}\n{USAGE}");
        std::process::exit(2);
    }
    // The card and backend as flags (the global `--device`/`--backend` of the CLI), so a probe pinned to one card
    // stays off the others and does not fall back to the default backend of the box.
    gpu_core::select_compute(&arg(&args, "--device").unwrap_or_else(|| "gpu".to_string()), arg(&args, "--backend").as_deref()).unwrap_or_else(|e| panic!("{e}"));
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
    let seed = arg(&args, "--seed").map_or_else(|| Rng::random().next_u64(), |v| v.parse().expect("--seed N"));
    println!("PROBE seed={seed} layers={layers} tokens={tokens} base={base} rank={rank} recompute={recompute} gdn_segment={gdn_segment}");
    let src = SyntheticSource { sizes: cfg.param_list().into_iter().collect(), seed };
    let residuals = arg(&args, "--residuals").map_or(SavedFormat::F32, |v| SavedFormat::parse(&v).unwrap_or_else(|| panic!("--residuals {v:?}: expected f32, bf16 or fp8")));
    let saved = if recompute { SavedPolicy::recompute_with(residuals) } else { SavedPolicy::Keep };
    assert!(recompute || residuals == SavedFormat::F32, "--residuals bf16|fp8 needs --recompute on");
    let opts = TrainOpts {
        base: TierPolicy::uniform(tier),
        saved,
        head_budget_words: arg(&args, "--head-budget").map(|v| v.parse().expect("--head-budget WORDS")),
        flash_min_t: arg(&args, "--flash-min-t").map(|v| v.parse().expect("--flash-min-t N")),
        gdn_segment_tokens: (gdn_segment > 0).then_some(gdn_segment),
        mlp_rows: arg(&args, "--mlp-rows").map(|v| v.parse().expect("--mlp-rows N")),
        gemm_precision: match arg(&args, "--gemm-precision").as_deref() {
            None | Some("auto") => GemmPrecision::Auto,
            Some("fp32") => GemmPrecision::Fp32Reference,
            Some("bf16") => GemmPrecision::Bf16TensorCore,
            Some(other) => panic!("--gemm-precision {other:?}: expected auto, fp32 or bf16"),
        },
        ..TrainOpts::default()
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

    let others_start = other_compute_contexts();
    let t0 = Instant::now();
    let m = Qwen35::new_train_opts_on(Gpu::new(pipelines()), cfg.clone(), 1, tokens, &src, &opts);
    println!("PROBE build_s={:.1}", t0.elapsed().as_secs_f64());
    let steady = own_gpu_mib();
    println!("PROBE built_mib={}", steady.map_or("n/a".to_string(), |v| v.to_string()));
    println!("PROBE attention_flash={}", m.attention_is_flash());

    let mut rng = Lcg::new(seed ^ 0x5eed);
    let x: Vec<u32> = (0..tokens).map(|_| rng.next_u32() % cfg.vocab).collect();
    let y: Vec<u32> = (0..tokens).map(|_| rng.next_u32() % cfg.vocab).collect();
    m.set_batch(&x, &y);
    let mut warm_secs = 0.0;
    let profile = args.iter().any(|a| a == "--profile");
    for step in 1..=steps {
        // The last step of a profiled run is timed kernel by kernel on the device
        // (events around each launch, so launches are issued one at a time).
        let timed = profile && step == steps && steps > 1 && m.gpu.set_kernel_timing(true);
        m.gpu.reset_ops_counters();
        let calls0 = gpu_core::cuda_call_totals();
        let cpu0 = cpu_seconds();
        let t = Instant::now();
        let (loss, fwd) = gpu_core::profile::device_table(&m.gpu, &format!("step {step}"), || {
            m.zero_grads();
            let loss = m.forward();
            let fwd = t.elapsed().as_secs_f64();
            m.backward();
            m.adamw_step(step, 1e-4, 0.0, Default::default(), Some(1.0), 1.0);
            m.poll_wait();
            (loss, fwd)
        });
        let total = t.elapsed().as_secs_f64();
        let counters = m.gpu.ops_counters();
        let dispatches = counters.steps;
        let cpu = cpu0.zip(cpu_seconds()).map_or("n/a".to_string(), |(a, b)| format!("{:.2}", b - a));
        // Host-side driver calls the step made: allocations, single launches, graph replays. A step that is
        // host-bound shows launches near the dispatch count; one that replays shows graph_replays instead.
        let calls = gpu_core::cuda_call_totals();
        println!(
            "PROBE step={step} loss={loss:.4} forward_s={fwd:.2} step_s={total:.2} tokens_per_s={:.0} dispatches={dispatches} cpu_s={cpu} loss_bits={:08x} alloc_calls={} host_launches={} graph_replays={} submit_s={:.2} wait_s={:.2} zero_calls={} zeroed_gib={:.1} uninit_calls={} uninit_gib={:.1}",
            tokens as f64 / total,
            loss.to_bits(),
            calls.device_alloc_calls - calls0.device_alloc_calls + calls.pool_alloc_calls - calls0.pool_alloc_calls,
            calls.host_launches - calls0.host_launches,
            calls.graph_replays - calls0.graph_replays,
            (calls.submit_nanos - calls0.submit_nanos) as f64 * 1e-9,
            (calls.wait_nanos - calls0.wait_nanos) as f64 * 1e-9,
            calls.zero_calls - calls0.zero_calls,
            (calls.zeroed_bytes - calls0.zeroed_bytes) as f64 / (1u64 << 30) as f64,
            calls.uninit_calls - calls0.uninit_calls,
            (calls.uninit_bytes - calls0.uninit_bytes) as f64 / (1u64 << 30) as f64,
        );
        if step > 1 {
            warm_secs += total;
        }
        if timed {
            // Dispatches by kernel (the launch census): what a step issues, not what it costs.
            let mut census: Vec<(String, u64)> = counters.by_kernel.iter().map(|(k, v)| (k.clone(), v.calls)).chain(counters.uncovered.iter().map(|(k, c)| (k.clone(), *c))).collect();
            census.sort_by_key(|entry| std::cmp::Reverse(entry.1));
            for (name, calls) in census.iter().take(12) {
                println!("CENSUS {calls:7} {name}");
            }
            let mut rows = m.gpu.kernel_times().unwrap_or_default();
            rows.sort_by(|a, b| b.1.total_cmp(&a.1));
            let sum: f64 = rows.iter().map(|r| r.1).sum();
            println!("PROFILE device_ms_total={sum:.0} kernels={}", rows.len());
            for (name, ms, calls) in rows.iter().take(30) {
                println!("PROFILE {:6.1}% {ms:9.1} ms {calls:7} calls {:8.3} ms/call {name}", 100.0 * ms / sum, ms / *calls as f64);
            }
            m.gpu.set_kernel_timing(false);
        }
    }
    if steps > 1 {
        let rate = (steps - 1) as f64 * tokens as f64 / warm_secs;
        let n = |c: Option<usize>| c.map_or("n/a".to_string(), |c| c.to_string());
        // Which allocation sizes the cleared bytes come from (cumulative over all steps).
        for (bucket, (n_calls, bytes)) in gpu_core::cuda_zeroed_by_size().into_iter().enumerate().filter(|(_, (c, _))| *c > 0) {
            println!("PROBE zeroed size=2^{bucket}B calls={n_calls} gib={:.1}", bytes as f64 / (1u64 << 30) as f64);
        }
        println!("PROBE warm_tokens_per_s={rate:.0} other_contexts_start={} other_contexts_end={}", n(others_start), n(other_compute_contexts()));
    }
    stop.store(true, Ordering::Relaxed);
    sampler.join().expect("sampler");
    let peak = peak.load(Ordering::Relaxed);
    println!("PROBE peak_mib={peak}");
    if let Some(steady) = steady.filter(|_| peak > 0) {
        println!("PROBE step_mib_per_token={:.3}", peak.saturating_sub(steady) as f64 / tokens as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::check_flags;

    fn argv(a: &[&str]) -> Vec<String> {
        std::iter::once("probe").chain(a.iter().copied()).map(String::from).collect()
    }

    #[test]
    fn known_flags_pass_and_typos_and_missing_values_are_refused() {
        assert!(check_flags(&argv(&["--tokens", "4096", "--base", "nf4", "--mlp-rows", "256"])).is_ok());
        assert!(check_flags(&argv(&["--seed", "7", "--gemm-precision", "fp32", "--profile", "--steps", "3"])).is_ok());
        assert!(check_flags(&argv(&["--token", "4096"])).unwrap_err().contains("--token"));
        assert!(check_flags(&argv(&["--tokens"])).unwrap_err().contains("requires a value"));
    }
}
