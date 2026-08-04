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

  brain perf gate <candidate.json> --baseline <b.json> [--floor 0.85] [--update]
      Hard-floor regression gate: throughput floors at --floor of baseline,
      latency ceilings at baseline / --floor (generous on purpose — tight
      deltas flap on shared boxes and a flapping gate gets deleted). Refuses
      incomparable pairs (different scenario/unit/hardware/build), smoke runs
      and correctness-failed runs; unmeasured metrics skip and say so.
      --update promotes the candidate to the new baseline. Exit 1 on failure.

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
                          lfm:<weights>:<tok.json> the LFM2.5 encoder behind the
                                                   residency executor (unit: sequence)
                          flux2[:<W>x<H>x<steps>[:<precision>]]
                                                   FLUX.2 Klein (klein-4b) behind the
                                                   residency executor; weights from
                                                   BRAIN_FLUX2_* env. Default
                                                   512x512x4:fp32; unit: denoise_step.
                                                   Concurrent same-size requests share
                                                   one batched denoise loop (cap:
                                                   BRAIN_FLUX2_MAX_BATCH, default 4).
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
  --admission <p>       overload only: admission policy installed on the engine —
                        unbounded (default) | depth:<N> | deadline:<ms>. Real
                        rejections are counted; SLO attainment covers admitted
                        work only.
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

/// Target specs are a CLI concern (the perf crate only sees `PerfTarget`s), so
/// `perf list` appends them here rather than in `perf::list()`.
const TARGET_LIST: &str = "
targets (--target):
  fake                               synthetic harness self-check (numbers meaningless)
  qwen-synth:<L>x<D>x<H>[xV][:i8w]   the real paged serving engine on random weights
  qwen:<weights.brain>[:i8w]         the paged serving engine on a real checkpoint
  lfm:<weights>:<tokenizer.json>     LFM2.5 encoder via the residency executor (unit: sequence)
  kronos:<tokenizer-dir>:<decoder-dir>  Kronos OHLCV forecaster via the residency executor
                                     (unit: forecast; input_artifacts = context bars; horizon/
                                     samples from BRAIN_FORECAST_HORIZON/BRAIN_FORECAST_SAMPLES)
  chronos2:<weights>                 Chronos-2 universal forecaster (unit: forecast)
  fincast:<weights>                  FinCast financial forecaster (unit: forecast)
  flux2[:<W>x<H>x<steps>[:<prec>]]   FLUX.2 Klein via the residency executor (unit: denoise_step;
                                     weights from BRAIN_FLUX2_* env; default 512x512x4:fp32;
                                     prec = fp32|int8; batches concurrent same-key requests)
";

pub fn run_perf(args: &[String]) {
    match args.first().map(|s| s.as_str()) {
        Some("list") => print!("{}{}", perf::list(), TARGET_LIST),
        Some("run") => run(&args[1..]),
        Some("compare") => compare(&args[1..]),
        Some("gate") => gate(&args[1..]),
        Some("placement") => match crate::perf_engine::run_placement(&args[1..]) {
            Ok(art) => emit(&art, None, 0),
            Err(e) => {
                eprintln!("perf placement: {e}");
                std::process::exit(2);
            }
        },
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

/// `perf gate <candidate.json> --baseline <b.json> [--floor 0.85] [--update]`
/// — hard-floor regression gate (J2). Exit 0 = pass, 1 = fail/refused.
fn gate(args: &[String]) {
    let mut candidate = None;
    let mut baseline = None;
    let mut floor = 0.85f64;
    let mut update = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--baseline" => baseline = Some(val(args, &mut i, "--baseline")),
            "--floor" => floor = val(args, &mut i, "--floor").parse().unwrap_or(floor),
            "--update" => update = true,
            other if candidate.is_none() => candidate = Some(other.to_string()),
            other => {
                eprintln!("perf gate: unexpected argument {other:?}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let (Some(cand_path), Some(base_path)) = (candidate, baseline) else {
        eprintln!("usage: brain perf gate <candidate.json> --baseline <baseline.json> [--floor 0.85] [--update]");
        std::process::exit(2);
    };
    let cand = match perf::report::load(&cand_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("perf gate: {e}");
            std::process::exit(2);
        }
    };
    if update {
        // Promote the candidate to the new baseline — deliberate, never a
        // side effect of a passing run.
        if let Err(e) = std::fs::copy(&cand_path, &base_path) {
            eprintln!("perf gate: updating baseline: {e}");
            std::process::exit(2);
        }
        eprintln!("baseline {base_path} <- {cand_path}");
        return;
    }
    let base = match perf::report::load(&base_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("perf gate: {e} (create one with --update)");
            std::process::exit(2);
        }
    };
    let outcome = perf::gate::gate(&cand, &base, floor);
    print!("{}", perf::gate::render(&outcome, floor));
    if !outcome.passed() {
        std::process::exit(1);
    }
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
    let mut policy = "cost-aware".to_string();

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
            "--policy" => policy = val(args, &mut i, "--policy"),
            "--admission" => opt.admission = Some(val(args, &mut i, "--admission")),
            "--soak-seconds" => opt.soak_seconds = val(args, &mut i, "--soak-seconds").parse().unwrap_or(opt.soak_seconds),
            "--device-rate" => opt.device_rate = val(args, &mut i, "--device-rate").parse().ok(),
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

    // Scenarios that need the engine itself (or no target at all) are dispatched
    // before a generic target is built.
    let engine_scenario = matches!(
        scenario.as_str(),
        "startup" | "cancel" | "kvcache" | "residency" | "faults"
    );
    if engine_scenario {
        let art = match scenario.as_str() {
            "residency" => crate::perf_engine::run_residency_with(&opt, 24, 4.0, &policy),
            other => {
                let shape = target_spec
                    .strip_prefix("qwen-synth:")
                    .ok_or_else(|| format!("`{other}` needs --target qwen-synth:<L>x<D>x<H>[xV]"))
                    .and_then(|sh| SynthSpec::parse(sh, &workload));
                match shape {
                    Ok(sp) => match other {
                        "startup" => crate::perf_engine::run_startup(&sp, opt.best_of.max(2), &opt),
                        "cancel" => crate::perf_engine::run_cancel(&sp, &opt),
                        "kvcache" => crate::perf_engine::run_kvcache(&sp, &opt),
                        _ => crate::perf_engine::run_faults(&sp, &opt),
                    },
                    Err(e) => Err(e),
                }
            }
        };
        let art = match art {
            Ok(a) => a,
            Err(e) => {
                eprintln!("perf: {e}");
                std::process::exit(2);
            }
        };
        emit(&art, out, opt.seed);
        return;
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
    emit(&art, out, opt.seed);
}

/// Render an artifact and write it, printing any scenario notes so a limitation
/// is visible in the terminal and not only in the JSON.
fn emit(art: &perf::schema::Artifact, out: Option<String>, seed: u64) {
    print!("{}", scenarios::render(art));
    if let Some(n) = &art.notes {
        println!("\nnote: {n}");
    }
    let path = out.unwrap_or_else(|| art.default_path(seed));
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


/// A synthetic model shape, reusable by the engine-coupled scenarios which need
/// to build (and rebuild) the engine themselves rather than receive one.
#[derive(Clone, Debug)]
pub struct SynthSpec {
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub vocab: u32,
    pub block_size: u32,
    pub max_batch: u32,
    pub num_blocks: u32,
    pub per_seq: u32,
    pub max_prefill: u32,
}

impl SynthSpec {
    /// Parse `<L>x<D>x<H>[xV]` and size a KV pool for `workload`.
    pub fn parse(shape: &str, workload: &str) -> Result<SynthSpec, String> {
        let parts: Vec<u32> = shape.split('x').map(|p| p.trim().parse().unwrap_or(0)).collect();
        if parts.len() < 3 || parts[..3].iter().any(|&v| v == 0) {
            return Err(format!("bad shape {shape:?}, expected <layers>x<d_model>x<heads>[x<vocab>]"));
        }
        let (n_layers, d_model, n_heads) = (parts[0], parts[1], parts[2]);
        if d_model % n_heads != 0 {
            return Err(format!("d_model {d_model} must be divisible by n_heads {n_heads}"));
        }
        let (block_size, max_batch, num_blocks, per_seq, max_prefill) = pool_for(workload)?;
        Ok(SynthSpec {
            n_layers,
            d_model,
            n_heads,
            vocab: parts.get(3).copied().unwrap_or(32_000),
            block_size,
            max_batch,
            num_blocks,
            per_seq,
            max_prefill,
        })
    }

    pub fn shape(&self) -> String {
        format!("L{}xD{}xH{}", self.n_layers, self.d_model, self.n_heads)
    }
    pub fn model_name(&self) -> String {
        "qwen-synth".to_string()
    }
    pub fn prefill_tokens(&self) -> usize {
        self.max_prefill.max(8) as usize
    }

    pub fn config(&self) -> qwen::QwenConfig {
        let head_dim = self.d_model / self.n_heads;
        let n_kv_heads = if self.n_heads % 4 == 0 { self.n_heads / 4 } else { self.n_heads };
        qwen::QwenConfig {
            vocab: self.vocab,
            block_size: 4096,
            n_layers: self.n_layers,
            d_model: self.d_model,
            n_heads: self.n_heads,
            n_kv_heads,
            head_dim,
            d_ff: self.d_model * 4,
            rope_theta: 1_000_000.0,
            rms_eps: 1e-6,
            tie_embeddings: true,
            qk_norm: true,
            attn_bias: false,
            lora: None,
        }
    }

    pub fn build_weights(&self) -> (qwen::QwenConfig, std::collections::HashMap<String, Vec<f32>>) {
        let cfg = self.config();
        let w = qwen::init_weights(&cfg, 1234);
        (cfg, w)
    }

    pub fn build_engine(
        &self,
        cfg: qwen::QwenConfig,
        w: &std::collections::HashMap<String, Vec<f32>>,
    ) -> qwen::serve::Engine {
        self.build_engine_with_blocks(cfg, w, self.num_blocks)
    }

    /// Build on an existing engine's device (`Engine::from_map_on`) — the
    /// warm-start path `perf startup` measures.
    pub fn build_engine_on(
        &self,
        parent: &qwen::serve::Engine,
        cfg: qwen::QwenConfig,
        w: &std::collections::HashMap<String, Vec<f32>>,
    ) -> qwen::serve::Engine {
        qwen::serve::Engine::from_map_on(
            parent.gpu(),
            cfg,
            w,
            self.block_size,
            self.num_blocks,
            self.max_batch,
            self.per_seq,
            self.max_prefill,
            false,
            false,
        )
    }

    pub fn build_engine_with_blocks(
        &self,
        cfg: qwen::QwenConfig,
        w: &std::collections::HashMap<String, Vec<f32>>,
        num_blocks: u32,
    ) -> qwen::serve::Engine {
        qwen::serve::Engine::from_map(
            cfg,
            w,
            self.block_size,
            num_blocks,
            self.max_batch,
            self.per_seq,
            self.max_prefill,
            false,
            false,
        )
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
    if let Some(rest) = spec.strip_prefix("lfm:") {
        return build_lfm(rest);
    }
    if let Some(rest) = spec.strip_prefix("kronos:") {
        return build_kronos(rest);
    }
    if let Some(rest) = spec.strip_prefix("chronos2:") {
        return build_chronos2(rest);
    }
    if let Some(rest) = spec.strip_prefix("fincast:") {
        return build_fincast(rest);
    }
    if spec == "flux2" {
        return build_flux2("");
    }
    if let Some(rest) = spec.strip_prefix("flux2:") {
        return build_flux2(rest);
    }
    Err(format!(
        "unknown --target {spec:?} \
         (expected 'fake', 'qwen-synth:<L>x<D>x<H>[xV][:i8w]', 'qwen:<weights>[:i8w]', \
         'lfm:<weights>:<tokenizer.json>', 'kronos:<tokenizer-dir>:<decoder-dir>', \
         'chronos2:<weights>', 'fincast:<weights>', or 'flux2[:<W>x<H>x<steps>[:<precision>]]')"
    ))
}



/// `lfm:<weights>:<tokenizer.json>` — the LFM2.5 encoder behind the residency
/// EXECUTOR (scheduler + budgets + device lanes), so concurrency>1 measures
/// brain's real batching/placement rather than a synchronous provider mutex.
/// One-shot semantics: artifact_unit "sequence", ttfa == e2e, tpoa null.
/// The synthetic payload realises `input_artifacts` EXACTLY via `max_tokens`
/// truncation, so every request of a length shares one built graph.
fn build_lfm(rest: &str) -> Result<Box<dyn PerfTarget>, String> {
    let (weights, tokenizer) = rest
        .split_once(':')
        .ok_or("lfm target needs 'lfm:<weights>:<tokenizer.json>'")?;
    if !std::path::Path::new(weights).exists() {
        return Err(format!("lfm weights not found: {weights}"));
    }
    let resident = crate::resident_lfm::LfmResident::new(weights, tokenizer)?;
    // Budget ONLY the schedulable devices (`--device`/BRAIN_DEVICE narrowing,
    // exactly as `brain serve --dbus` does): budgeting an excluded GPU lets
    // placement pick an index the process cannot see, and the lane silently
    // falls back to a software adapter — a flattering-to-nobody 6× slowdown
    // that must never masquerade as a GPU number.
    let set = crate::compute_set();
    let mut budgets = residency::budget::Budgets::new();
    for (i, total) in crate::run_cli::query_gpu_mem() {
        if set.as_ref().map(|s| s.gpus.contains(&i)).unwrap_or(true) {
            budgets.set(residency::Device::Gpu(i), total, 2 << 30);
        }
    }
    if set.map(|s| s.cpu_enabled()).unwrap_or(true) {
        budgets.set(residency::Device::Cpu, crate::run_cli::query_ram_bytes(), 0);
    }
    let exec = residency::Executor::start(
        vec![std::sync::Arc::new(resident)],
        budgets,
        residency::Policy::default(),
    );
    let info = vec![
        ("weights".to_string(), serde_json::json!(weights)),
        ("batch".to_string(), serde_json::json!(std::env::var("BRAIN_LFM_BATCH").unwrap_or_else(|_| "2".into()))),
        ("engine".to_string(), serde_json::json!("residency-executor")),
    ];
    let build = Box::new(|req: &perf::target::PerfRequest| {
        // ~2N words then truncate to exactly N tokens inside the resident.
        let text = "word ".repeat(req.input_artifacts * 2);
        capability::Invocation::new()
            .set("text", serde_json::json!(text))
            .set("max_tokens", serde_json::json!(req.input_artifacts))
    });
    Ok(Box::new(perf::targets::ExecutorTarget::new(exec, lfm::caps::MODEL, "embed", "sequence", info, build)))
}

/// `BRAIN_FORECAST_HORIZON` / `BRAIN_FORECAST_SAMPLES` (defaults 64 / 1) — the
/// horizon and sample count every forecast perf target requests.
fn forecast_env() -> (i64, i64) {
    let horizon = std::env::var("BRAIN_FORECAST_HORIZON").ok().and_then(|s| s.parse().ok()).filter(|&h| h > 0).unwrap_or(64);
    let samples = std::env::var("BRAIN_FORECAST_SAMPLES").ok().and_then(|s| s.parse().ok()).filter(|&s| s > 0).unwrap_or(1);
    (horizon, samples)
}

/// Budget the schedulable devices and start a one-resident executor — the shared
/// forecast/lfm rule: never budget an excluded GPU (or placement silently drops to
/// a software adapter, a flattering-to-nobody slowdown).
fn forecast_executor<R: residency::ResidentModel + 'static>(resident: R) -> residency::Executor {
    let set = crate::compute_set();
    let mut budgets = residency::budget::Budgets::new();
    for (i, total) in crate::run_cli::query_gpu_mem() {
        if set.as_ref().map(|s| s.gpus.contains(&i)).unwrap_or(true) {
            budgets.set(residency::Device::Gpu(i), total, 2 << 30);
        }
    }
    if set.map(|s| s.cpu_enabled()).unwrap_or(true) {
        budgets.set(residency::Device::Cpu, crate::run_cli::query_ram_bytes(), 0);
    }
    residency::Executor::start(vec![std::sync::Arc::new(resident)], budgets, residency::Policy::default())
}

/// The per-request payload shared by every forecast target: a deterministic
/// trend+seasonal close series realised to exactly `input_artifacts` bars, passed
/// as the `context` f32-LE blob the residents decode, plus horizon/samples params.
fn forecast_build(horizon: i64, samples: i64) -> Box<dyn Fn(&perf::target::PerfRequest) -> capability::Invocation> {
    Box::new(move |req: &perf::target::PerfRequest| {
        let ctxlen = req.input_artifacts.max(2);
        let series: Vec<f32> = (0..ctxlen)
            .map(|i| {
                let x = i as f32;
                100.0 + (x * 0.1).sin() * 5.0 + (x * 0.02).cos() * 2.0 + x * 0.01
            })
            .collect();
        let bytes: Vec<u8> = series.iter().flat_map(|v| v.to_le_bytes()).collect();
        let blob = capability::Blob::new(capability::Media::Bytes, bytes).with_meta(serde_json::json!({ "shape": [ctxlen] }));
        capability::Invocation::new()
            .blob("context", blob)
            .set("horizon", serde_json::json!(horizon))
            .set("samples", serde_json::json!(samples))
    })
}

/// `kronos:<tokenizer-dir>:<decoder-dir>` — the Kronos OHLCV forecaster behind the
/// residency EXECUTOR (scheduler + budgets + device lanes), so concurrency>1 and
/// device placement (CPU/iGPU/NPU) are measured through brain's real serving path.
/// One forecast per request (`artifact_unit` "forecast"); `input_artifacts` is the
/// context length in bars, so a prefill/decode sweep is `--ladder` over input size.
fn build_kronos(rest: &str) -> Result<Box<dyn PerfTarget>, String> {
    let (tokenizer, decoder) = rest
        .split_once(':')
        .ok_or("kronos target needs 'kronos:<tokenizer-dir>:<decoder-dir>'")?;
    if !std::path::Path::new(tokenizer).exists() {
        return Err(format!("kronos tokenizer dir not found: {tokenizer}"));
    }
    if !std::path::Path::new(decoder).exists() {
        return Err(format!("kronos decoder dir not found: {decoder}"));
    }
    let (horizon, samples) = forecast_env();
    let exec = forecast_executor(crate::resident_forecast::KronosResident::new(tokenizer, decoder));
    let info = vec![
        ("tokenizer".to_string(), serde_json::json!(tokenizer)),
        ("decoder".to_string(), serde_json::json!(decoder)),
        ("horizon".to_string(), serde_json::json!(horizon)),
        ("samples".to_string(), serde_json::json!(samples)),
        ("engine".to_string(), serde_json::json!("residency-executor")),
    ];
    Ok(Box::new(perf::targets::ExecutorTarget::new(exec, crate::resident_forecast::KRONOS_MODEL, "forecast", "forecast", info, forecast_build(horizon, samples))))
}

/// `chronos2:<weights>` — the Chronos-2 universal forecaster behind the residency
/// executor (same measurement path as `kronos:`; `input_artifacts` = context length).
fn build_chronos2(path: &str) -> Result<Box<dyn PerfTarget>, String> {
    if !std::path::Path::new(path).exists() {
        return Err(format!("chronos2 weights not found: {path}"));
    }
    let (horizon, _samples) = forecast_env();
    let exec = forecast_executor(crate::resident_forecast::Chronos2Resident::new(path));
    let info = vec![
        ("weights".to_string(), serde_json::json!(path)),
        ("horizon".to_string(), serde_json::json!(horizon)),
        ("engine".to_string(), serde_json::json!("residency-executor")),
    ];
    Ok(Box::new(perf::targets::ExecutorTarget::new(exec, crate::resident_forecast::CHRONOS2_MODEL, "forecast", "forecast", info, forecast_build(horizon, 1))))
}

/// `fincast:<weights>` — the FinCast financial forecaster behind the residency
/// executor (same measurement path as `kronos:`; `input_artifacts` = context length).
fn build_fincast(path: &str) -> Result<Box<dyn PerfTarget>, String> {
    if !std::path::Path::new(path).exists() {
        return Err(format!("fincast weights not found: {path}"));
    }
    let (horizon, _samples) = forecast_env();
    let exec = forecast_executor(crate::resident_forecast::FincastResident::new(path));
    let info = vec![
        ("weights".to_string(), serde_json::json!(path)),
        ("horizon".to_string(), serde_json::json!(horizon)),
        ("engine".to_string(), serde_json::json!("residency-executor")),
    ];
    Ok(Box::new(perf::targets::ExecutorTarget::new(exec, crate::resident_forecast::FINCAST_MODEL, "forecast", "forecast", info, forecast_build(horizon, 1))))
}

/// `flux2[:<W>x<H>x<steps>[:<precision>]]` — FLUX.2 Klein (klein-4b, weights from the
/// `BRAIN_FLUX2_*` env like the rest of the flux2 stack) behind the residency
/// EXECUTOR (scheduler + budgets + device lanes), running [`crate::resident_flux2::Flux2Resident`]
/// — so concurrency measures brain's real scheduler, not a bare provider.
///
/// Streaming semantics: `flux2::Pipeline::generate` reports one `Progress` per
/// denoise step (message `"denoising"`, emitted at step START), plus
/// "encoding"/"decoding" bookkeeping callbacks. `ExecutorTarget::new_streaming`
/// timestamps exactly the denoising callbacks as artifacts, so
/// `artifact_unit = "denoise_step"`: TTFA = queue + prompt encode, each IAL gap
/// = one denoise step, and the final step + VAE decode land between the last
/// artifact and Done (inside e2e). All requests of one size share one resident
/// pipeline instance (the instance key is `{variant}:{w}x{h}:{nref}`), so
/// weights load once — put it in the warmup request.
fn build_flux2(rest: &str) -> Result<Box<dyn PerfTarget>, String> {
    // optional `:<precision>` tail — the DiT numeric tier the requests ask for.
    // int8 is the single-card configuration (weights ~4x smaller), so it is the
    // one a concurrency ladder can actually reach batch 8 on.
    let (rest, precision) = match rest.rsplit_once(':') {
        Some((head, p)) if flux2::Precision::from_name(p).is_ok() => (head, p.to_string()),
        _ => (rest, "fp32".to_string()),
    };
    let (w, h, steps) = if rest.is_empty() {
        (512u32, 512u32, 4u32)
    } else {
        let p: Vec<u32> = rest.split('x').map(|s| s.trim().parse().unwrap_or(0)).collect();
        if p.len() != 3 || p.iter().any(|&v| v == 0) {
            return Err(format!("bad flux2 spec {rest:?}, expected <W>x<H>x<steps>[:<precision>] (e.g. 512x512x4:int8)"));
        }
        (p[0], p[1], p[2])
    };
    if w % 16 != 0 || h % 16 != 0 {
        return Err(format!("flux2: width/height must be multiples of 16 (got {w}x{h})"));
    }
    // Fail before the run starts, naming what is missing — not at first activation.
    let paths = flux2::Paths::from_env()
        .map_err(|e| format!("flux2 target needs BRAIN_FLUX2_DIT/_VAE/_TE/_TOKENIZER: {e}"))?;
    for (var, p) in [
        ("BRAIN_FLUX2_DIT", &paths.dit),
        ("BRAIN_FLUX2_VAE", &paths.vae),
        ("BRAIN_FLUX2_TE", &paths.te),
        ("BRAIN_FLUX2_TOKENIZER", &paths.tokenizer),
    ] {
        if !std::path::Path::new(p).exists() {
            return Err(format!("flux2: {var} not found: {p}"));
        }
    }
    let resident = crate::resident_flux2::Flux2Resident::from_env()
        .ok_or("flux2: BRAIN_FLUX2_* env incomplete")?;
    // Budget ONLY the schedulable devices — same guard as `build_lfm` (its
    // ledger records a silent 6× llvmpipe regression from budgeting a GPU the
    // process could not see; the lane fell back to a software adapter and the
    // run masqueraded as a GPU number).
    let set = crate::compute_set();
    let mut budgets = residency::budget::Budgets::new();
    for (i, total) in crate::run_cli::query_gpu_mem() {
        if set.as_ref().map(|s| s.gpus.contains(&i)).unwrap_or(true) {
            budgets.set(residency::Device::Gpu(i), total, 2 << 30);
        }
    }
    if set.map(|s| s.cpu_enabled()).unwrap_or(true) {
        budgets.set(residency::Device::Cpu, crate::run_cli::query_ram_bytes(), 0);
    }
    let exec = residency::Executor::start(
        vec![std::sync::Arc::new(resident)],
        budgets,
        residency::Policy::default(),
    );
    let variant = "klein-4b";
    let cfg = flux2::Flux2Config::from_name(variant)?;
    let mut info = TargetInfo::new(&format!("flux2-{variant}"), "denoise_step");
    // The 4B MMDiT — the component a denoise step runs; TE/VAE cost sits in
    // TTFA/e2e, not in the per-step rate.
    info.params = Some(3_870_000_000);
    info.quant = Some(precision.clone());
    let info = info
        .with("width", w.into())
        .with("height", h.into())
        .with("steps", steps.into())
        .with("txt_len", (cfg.txt_len as u32).into())
        .with("precision", precision.clone().into())
        // What the instance batches into one denoise loop; `Executor::stats()`
        // reports the batch sizes actually reached.
        .with("max_batch", crate::resident_flux2::max_batch().into())
        .with("engine", "residency-executor".into());
    let build = Box::new(move |req: &perf::target::PerfRequest| {
        // Prompt values do not change the cost: text conditioning is padded to
        // the fixed txt_len. The per-request seed keeps noise deterministic.
        capability::Invocation::new()
            .set("prompt", serde_json::json!("a lighthouse on a rocky coast at sunset"))
            .set("width", serde_json::json!(w))
            .set("height", serde_json::json!(h))
            .set("steps", serde_json::json!(steps))
            .set("precision", serde_json::json!(precision))
            .set("seed", serde_json::json!(req.seed))
    });
    Ok(Box::new(perf::targets::ExecutorTarget::new_streaming(
        exec,
        flux2::caps::MODEL,
        "text2image",
        info,
        build,
        std::sync::Arc::new(|p: &capability::Progress| p.message == "denoising"),
    )))
}

/// Split an optional `:i8w` flag (int8 weights) off a target spec tail.
fn spec_flags(spec: &str) -> (&str, bool) {
    match spec.rsplit_once(':') {
        Some((head, "i8w")) => (head, true),
        _ => (spec, false),
    }
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
    let (shape, weights_int8) = spec_flags(shape);
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
        qk_norm: true,
        attn_bias: false,
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
        cfg, &weights, block_size, num_blocks, max_batch, per_seq, max_prefill, false, weights_int8,
    );
    let info = TargetInfo::new("qwen-synth", "token")
        .with("shape", format!("L{n_layers}xD{d_model}xH{n_heads}").into())
        .with("params", params.into())
        .with("weights", "random".into())
        .with("block_size", block_size.into())
        .with("max_batch", max_batch.into())
        .with("kv_dtype", "fp32".into())
        // What actually ran: the int8 request is capability-gated in the engine.
        .with("weights_dtype", if eng.weights_int8() { "int8" } else { "fp32" }.into());
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
    let (weights, weights_int8) = spec_flags(weights);
    let (block_size, max_batch, num_blocks, per_seq, max_prefill) = pool_for(workload)?;
    let eng = qwen::serve::Engine::load(
        weights,
        block_size,
        num_blocks,
        max_batch,
        per_seq,
        max_prefill,
        false,
        weights_int8,
    );
    let vocab = eng.vocab() as u32;
    let w8_effective = eng.weights_int8();
    let sched = qwen::serve::Scheduler::new(eng, max_batch as usize);
    let info = TargetInfo::new("qwen", "token")
        .with("weights", weights.into())
        .with("block_size", block_size.into())
        .with("max_batch", max_batch.into())
        .with("kv_dtype", "fp32".into())
        .with("weights_dtype", if w8_effective { "int8" } else { "fp32" }.into());
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
