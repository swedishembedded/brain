// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain perf …` — the performance benchmarking suite.
//!
//! Sibling to `brain bench` (which asks whether an architecture *learns* a task).
//! This asks how much correct work brain delivers per unit of hardware, memory,
//! energy and time — and is built so a single flattering number cannot be
//! reported on its own. See `docs/performance/benchmarking.md`.

use perf::scenarios::{self, Options};
use perf::target::{PerfTarget, TargetInfo};

const HELP: &str = "\
brain perf — performance benchmarking (how fast, at what cost, still correct?)

USAGE
  brain perf list
      Registered scenarios and the standard workload matrix.

  brain perf run <scenario> [options]
      scenarios: latency | throughput | serve | sweep

  brain perf compare <a.json> <b.json> ...
      Leaderboard across result artifacts. Refuses to rank across artifact
      units, excludes runs whose correctness gate failed, and warns on every
      environment/workload axis that differs between the runs.

OPTIONS
  --target <spec>       what to measure (default: fake)
                          fake                     built-in synthetic engine (no
                                                   model; validates the harness)
                          qwen-synth:<L>x<D>x<H>[xV]
                                                   the REAL paged serving engine on
                                                   random weights of that shape —
                                                   same kernels, KV traffic and
                                                   batching, no checkpoint needed.
                                                   Right for hardware/config
                                                   comparison, useless for quality.
                          qwen:<weights.brain>     the paged serving engine on a
                                                   real checkpoint
  --workload <name>     interactive | chat | rag | rag_long | agent |
                        decode_heavy | prefill_heavy | shared_prefix   (default chat)
  --concurrency <N>     fixed level for latency/serve (default 8)
  --ladder <a,b,c>      concurrency ladder for sweep (default 1,2,4,8,16,32)
  --requests <N>        measured requests per level (default 64)
  --input <N>           override the workload's prompt length
  --output <N>          override the workload's output length. THIS is what
                        bounds run time: 256 outputs at 90ms/token is 23s per
                        request no matter how few requests you run. Rates stay
                        comparable; the override is recorded in the artifact.
  --warmup <N>          warm-up requests, excluded from every statistic (default 4)
  --best-of <N>         repeat and keep the best, reporting the spread (default 1)
  --smoke               shrink everything to seconds, for CI
  --seed <S>            default 1234
  --out <file>          artifact path (default results/perf-<...>.json)
  --device cpu|gpu|vulkan  (global flag, handled by the main dispatcher)

NOTES
  * Report output artifacts/s and the latency curve, never total throughput
    alone: a huge-prompt workload posts an impressive total while delivering
    poor decode rate and bad interactive latency.
  * goodput (output meeting the SLO) is the comparison metric, not peak rate.
  * A box with no real GPU still serves --device gpu through a software
    rasteriser; artifacts record the adapter and mark such runs, and the
    reports label them. Do not report them as GPU numbers.
";

pub fn run_perf(args: &[String]) {
    match args.first().map(|s| s.as_str()) {
        Some("list") => print!("{}", perf::list()),
        Some("run") => run(&args[1..]),
        Some("compare") => compare(&args[1..]),
        Some("help") | Some("-h") | Some("--help") | None => print!("{HELP}"),
        Some(other) => {
            eprintln!("perf: unknown subcommand {other:?}\n");
            print!("{HELP}");
            std::process::exit(2);
        }
    }
}

fn val(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    args.get(*i).cloned().unwrap_or_else(|| {
        eprintln!("perf: {flag} needs a value");
        std::process::exit(2);
    })
}

fn compare(args: &[String]) {
    if args.is_empty() {
        eprintln!("perf compare: give at least one artifact (results/perf-*.json)");
        std::process::exit(2);
    }
    print!("{}", perf::report::compare(args));
}

fn run(args: &[String]) {
    let Some(scenario) = args.first().cloned() else {
        eprintln!("perf run: name a scenario (latency|throughput|serve|sweep)");
        std::process::exit(2);
    };
    let mut opt = Options { device: device_label(), ..Default::default() };
    let mut target_spec = "fake".to_string();
    let mut workload = "chat".to_string();
    let mut concurrency = 8usize;
    let mut out: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => target_spec = val(args, &mut i, "--target"),
            "--workload" => workload = val(args, &mut i, "--workload"),
            "--concurrency" => concurrency = val(args, &mut i, "--concurrency").parse().unwrap_or(concurrency),
            "--ladder" => {
                opt.concurrency = val(args, &mut i, "--ladder")
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
            }
            "--requests" => opt.num_requests = val(args, &mut i, "--requests").parse().unwrap_or(opt.num_requests),
            "--input" => opt.input_override = val(args, &mut i, "--input").parse().ok(),
            "--output" => opt.output_override = val(args, &mut i, "--output").parse().ok(),
            "--warmup" => opt.warmup_requests = val(args, &mut i, "--warmup").parse().unwrap_or(opt.warmup_requests),
            "--best-of" => opt.best_of = val(args, &mut i, "--best-of").parse().unwrap_or(opt.best_of),
            "--seed" => opt.seed = val(args, &mut i, "--seed").parse().unwrap_or(opt.seed),
            "--out" => out = Some(val(args, &mut i, "--out")),
            "--smoke" => opt = opt.smoke(),
            other => {
                eprintln!("perf run: unknown flag {other:?}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if opt.concurrency.is_empty() {
        opt.concurrency = vec![1];
    }

    let mut target = match build_target(&target_spec, &workload) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("perf: {e}");
            std::process::exit(2);
        }
    };

    let art = match scenarios::run(&scenario, target.as_mut(), &workload, concurrency, &opt) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("perf: {e}");
            std::process::exit(2);
        }
    };
    print!("{}", scenarios::render(&art));

    let path = out.unwrap_or_else(|| art.default_path(opt.seed));
    match art.write(&path) {
        Ok(()) => eprintln!("\nwrote {path}"),
        Err(e) => eprintln!("\nfailed to write {path}: {e}"),
    }
}

/// The device label recorded in the artifact. `--device` has already been
/// resolved by the main dispatcher; record the *resolved set* (`gpu[0,1]`,
/// `cpu[4 core(s)]`), not the raw flag, so an artifact says what actually ran.
fn device_label() -> String {
    match crate::compute_set() {
        Some(s) => s.to_string(),
        None => gpu_core::backend_name().to_string(),
    }
}

fn build_target(spec: &str, workload: &str) -> Result<Box<dyn PerfTarget>, String> {
    if spec == "fake" {
        return Ok(Box::new(FakeEngine::new()));
    }
    if let Some(shape) = spec.strip_prefix("qwen-synth:") {
        return build_qwen_synth(shape, workload);
    }
    if let Some(weights) = spec.strip_prefix("qwen:") {
        return build_qwen(weights, workload);
    }
    Err(format!(
        "unknown --target {spec:?} \
         (expected 'fake', 'qwen-synth:<L>x<D>x<H>[xV]', or 'qwen:<weights>')"
    ))
}

/// Build the serving engine on **randomly initialised weights** of a given
/// shape: `qwen-synth:<layers>x<d_model>x<heads>[x<vocab>]`.
///
/// Weight *values* do not affect execution cost — the same kernels, KV traffic,
/// batching and memory pressure occur whatever the numbers are — so this
/// measures the real engine without needing a checkpoint on the machine. It is
/// the right tool for hardware and configuration comparison, and the wrong tool
/// for anything about output quality: generated tokens are meaningless, so the
/// artifact records `weights: "random"` and no correctness gate can pass on it.
fn build_qwen_synth(shape: &str, workload: &str) -> Result<Box<dyn PerfTarget>, String> {
    let parts: Vec<u32> = shape.split('x').map(|p| p.trim().parse().unwrap_or(0)).collect();
    if parts.len() < 3 || parts[..3].iter().any(|&v| v == 0) {
        return Err(format!("bad shape {shape:?}, expected <layers>x<d_model>x<heads>[x<vocab>]"));
    }
    let (n_layers, d_model, n_heads) = (parts[0], parts[1], parts[2]);
    let vocab = parts.get(3).copied().unwrap_or(32_000);
    if d_model % n_heads != 0 {
        return Err(format!("d_model {d_model} must be divisible by n_heads {n_heads}"));
    }
    let head_dim = d_model / n_heads;
    // GQA with 4 query heads per kv head where that divides evenly (the usual
    // Qwen3 ratio), else MHA.
    let n_kv_heads = if n_heads % 4 == 0 { n_heads / 4 } else { n_heads };

    let cfg = qwen::QwenConfig {
        vocab,
        block_size: 4096,
        n_layers,
        d_model,
        n_heads,
        n_kv_heads,
        head_dim,
        d_ff: d_model * 4,
        rope_theta: 1_000_000.0,
        rms_eps: 1e-6,
        tie_embeddings: true,
        lora: None,
    };
    let params: usize = cfg.param_list().iter().map(|(_, n)| n).sum();
    eprintln!(
        "perf: synthetic qwen L{n_layers} D{d_model} H{n_heads} (kv {n_kv_heads}) \
         vocab {vocab} — {:.1}M params, random weights",
        params as f64 / 1e6
    );

    let weights = qwen::init_weights(&cfg, 1234);
    let (block_size, max_batch, num_blocks, per_seq, max_prefill) = pool_for(workload)?;
    let eng = qwen::serve::Engine::from_map(
        cfg, &weights, block_size, num_blocks, max_batch, per_seq, max_prefill, false,
    );
    let info = TargetInfo::new("qwen-synth", "token")
        .with("shape", format!("L{n_layers}xD{d_model}xH{n_heads}").into())
        .with("params", params.into())
        .with("weights", "random".into())
        .with("block_size", block_size.into())
        .with("max_batch", max_batch.into())
        .with("kv_dtype", "fp32".into());
    let sched = qwen::serve::Scheduler::new(eng, max_batch as usize);
    Ok(Box::new(perf::targets::PagedLlmTarget::new(sched, info, None, vocab)))
}

/// KV-pool geometry for a workload: `(block_size, max_batch, num_blocks,
/// blocks_per_seq, max_prefill)`. Sized so *admission*, not allocation failure,
/// is what limits concurrency.
fn pool_for(workload: &str) -> Result<(u32, u32, u32, u32, u32), String> {
    let w = perf::workload::standard(workload, perf::Arrival::Saturated, 1, 0)
        .ok_or_else(|| format!("unknown workload {workload:?}"))?;
    let r = &w.requests()[0];
    let (max_in, max_out) = (r.input_artifacts as u32, r.output_artifacts as u32);
    let block_size = 16u32;
    let max_batch = 32u32;
    let per_seq = (max_in + max_out + 8).div_ceil(block_size);
    let num_blocks = per_seq * max_batch + max_batch;
    Ok((block_size, max_batch, num_blocks, per_seq, max_in.max(1)))
}

/// Size the KV pool for the workload so admission, not allocation failure, is
/// what limits concurrency.
fn build_qwen(weights: &str, workload: &str) -> Result<Box<dyn PerfTarget>, String> {
    let (block_size, max_batch, num_blocks, per_seq, max_prefill) = pool_for(workload)?;
    let eng = qwen::serve::Engine::load(
        weights,
        block_size,
        num_blocks,
        max_batch,
        per_seq,
        max_prefill,
        false,
    );
    let vocab = eng.vocab() as u32;
    let sched = qwen::serve::Scheduler::new(eng, max_batch as usize);
    let info = TargetInfo::new("qwen", "token")
        .with("weights", weights.into())
        .with("block_size", block_size.into())
        .with("max_batch", max_batch.into())
        .with("kv_dtype", "fp32".into());
    Ok(Box::new(perf::targets::PagedLlmTarget::new(sched, info, None, vocab)))
}

/// A synthetic engine with no weights: enough to exercise and validate the whole
/// harness (arrival processes, percentiles, goodput, artifact schema) on any
/// machine, including one with no model checkpoint. Its *absolute* numbers mean
/// nothing and it says so in the artifact's model name.
struct FakeEngine {
    inner: Vec<(u64, usize, usize)>,
    queue: std::collections::VecDeque<(u64, perf::PerfRequest)>,
    next: u64,
}

impl FakeEngine {
    fn new() -> FakeEngine {
        FakeEngine { inner: Vec::new(), queue: Default::default(), next: 0 }
    }
}

impl PerfTarget for FakeEngine {
    fn describe(&self) -> TargetInfo {
        TargetInfo::new("fake", "token").with("note", "synthetic harness target — absolute numbers are meaningless".into())
    }
    fn submit(&mut self, req: perf::PerfRequest) -> u64 {
        let id = self.next;
        self.next += 1;
        self.queue.push_back((id, req));
        id
    }
    fn step(&mut self, out: &mut Vec<perf::target::Emission>) -> bool {
        use perf::target::{Emission, EmissionKind};
        while self.inner.len() < 32 {
            match self.queue.pop_front() {
                Some((id, req)) => {
                    // Prefill cost scales with prompt length, as a real engine's does.
                    let work: u64 = (req.input_artifacts as u64).saturating_mul(64);
                    std::hint::black_box(work.wrapping_mul(2654435761));
                    out.push(Emission { id, at: std::time::Instant::now(), kind: EmissionKind::Admitted });
                    self.inner.push((id, 0, req.output_artifacts));
                }
                None => break,
            }
        }
        let mut i = 0;
        while i < self.inner.len() {
            let (id, ref mut produced, wanted) = self.inner[i];
            *produced += 1;
            let done = *produced >= wanted;
            let now = std::time::Instant::now();
            out.push(Emission { id, at: now, kind: EmissionKind::Artifact });
            if done {
                out.push(Emission { id, at: now, kind: EmissionKind::Done });
                self.inner.remove(i);
            } else {
                i += 1;
            }
        }
        self.busy()
    }
    fn busy(&self) -> bool {
        !self.queue.is_empty() || !self.inner.is_empty()
    }
}
