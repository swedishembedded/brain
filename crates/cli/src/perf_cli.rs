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
  --target <spec>       what to measure (required — no default; there is no
                        harness-only stand-in, every target exercises a real
                        engine)
                          qwen-synth:<L>x<D>x<H>[xV[xHeadDim[xNKvHeads]]]
                                                   the REAL paged serving engine on
                                                   random weights of that shape —
                                                   same kernels, KV traffic and
                                                   batching, no checkpoint needed.
                                                   HeadDim/NKvHeads default to a
                                                   derived guess (d_model/heads,
                                                   heads/4) that does NOT match
                                                   real Qwen3 -- give them
                                                   explicitly to measure a real
                                                   model's KV shape. Right for
                                                   hardware/config
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
  qwen-synth:<L>x<D>x<H>[xV[xHeadDim[xNKvHeads]]][:i8w][:kvf32]   the real paged serving engine on random weights
  qwen:<weights.brain>[:i8w][:kvf32]         the paged serving engine on a real checkpoint
                                     (:i8w opts IN to int8 weights, off by default; :kvf32 opts
                                     OUT of int8 KV, which is ON by default -- either order, both optional.
                                     HeadDim/NKvHeads override the derived guess -- e.g.
                                     28x1024x16x151936x128x8 is Qwen3-0.6B's REAL KV shape, not the
                                     derived 64x4 -- give them to measure a real model, not a stand-in)
  http:qwen-synth:<L>x<D>x<H>[xV[xHeadDim[xNKvHeads]]]:<tokenizer.json>
                                     the REAL served path: apiserve's HTTP router, in-process,
                                     over random weights of that shape (needs a real tokenizer).
                                     The shape, incl. HeadDim/NKvHeads, is honoured -- it becomes
                                     the written checkpoint's real config; :i8w/:kvf32 are NOT --
                                     KV/weights dtype come from BRAIN_QWEN_KV_INT8 same as
                                     `brain serve`, since this measures the REAL served default.
  http:qwen:<weights.brain>:<tokenizer.json>
                                     the REAL served path over a real checkpoint -- this is what
                                     .todo/serving-performance-audit.md's 600s regression measures
  lfm:<weights>:<tokenizer.json>     LFM2.5 encoder via the residency executor (unit: sequence)
  kronos:<tokenizer-dir>:<decoder-dir>  Kronos OHLCV forecaster via the residency executor
                                     (unit: forecast; input_artifacts = context bars; horizon/
                                     samples from BRAIN_FORECAST_HORIZON/BRAIN_FORECAST_SAMPLES)
  chronos2:<weights>                 Chronos-2 universal forecaster (unit: forecast)
  fincast:<weights>                  FinCast financial forecaster (unit: forecast)
  flux2[:<W>x<H>x<steps>[:<prec>]]   FLUX.2 Klein via the residency executor (unit: denoise_step;
                                     weights from BRAIN_FLUX2_* env; default 512x512x4:fp32;
                                     prec = fp32|int8; batches concurrent same-key requests)
  gpt:<weights>                      dense char-level GPT via the residency executor (unit: token)
  glm:<weights>                      GLM-5.2-shaped decoder (MLA + sigmoid MoE) via the residency
                                     executor (unit: token)
  yolo:<weights>                     YOLOv8-style detector via the residency executor (unit: frame;
                                     one synthetic 640x640 image per request)
  depth:<weights>                    ZipDepth monocular depth via the residency executor (unit: frame;
                                     one synthetic 384x384 image per request)
  sam2:<weights-dir>[:tiny|large]    SAM 2.1 promptable segmentation via the residency executor
                                     (unit: mask; one centred point prompt per request)
  clip:<text-encoder-dir>:<eva.pt>   EVA-CLIP-L/336 image tower via the residency executor
                                     (unit: embedding; embed_image only -- no tokenizer needed)
  facenet:<weights-dir>              SCRFD detect + ArcFace embed via the residency executor
                                     (unit: embedding; align=true on a synthetic image)
  tts:<weights-dir>:<hf-ckpt-dir>    Qwen3-TTS speaker-free synthesis via the residency executor
                                     (unit: audio_chunk; --output caps codec frames)
  zimage[:<W>x<H>x<steps>]           Z-Image (Tongyi S3-DiT) via the residency executor
                                     (unit: denoise_step; weights from BRAIN_ZIMAGE_* env;
                                     default 512x512x4; large -- 13-24 GiB VRAM)
  upscale:<weights>                  Real-ESRGAN RRDBNet super-resolution via the residency
                                     executor (unit: image; one 256x256 tile, never batched)
  restore:<weights>                  CodeFormer blind face restoration via the residency executor
                                     (unit: image; fixed 512x512 graph)
  vqgan:<weights>                    CodeFormer's VQ encoder via the residency executor
                                     (unit: image; fixed 512x512 graph)
  nemotron:<checkpoint-dir>          FastConformer streaming ASR via the residency executor
                                     (unit: transcript_token, deliberately not 'token' --
                                     ASR tokens and LLM tokens are not comparable)
  qwen-asr:<checkpoint-dir>          Whisper-encoder + Qwen3-decoder offline ASR via the
                                     residency executor (unit: transcript_token)
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
    let mut target_spec: Option<String> = None;
    let mut workload = "chat".to_string();
    let mut concurrency = 8usize;
    let mut out: Option<String> = None;
    let mut policy = "cost-aware".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => target_spec = Some(val(args, &mut i, "--target")),
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
                    .as_deref()
                    .and_then(|s| s.strip_prefix("qwen-synth:"))
                    .ok_or_else(|| format!("`{other}` needs --target qwen-synth:<L>x<D>x<H>[xV]"))
                    .and_then(|sh| SynthSpec::parse(sh, &workload, opt.input_override, opt.output_override));
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

    let Some(target_spec) = target_spec else {
        eprintln!("perf run: --target is required (no default — see `brain perf list` or `brain perf --help`)");
        std::process::exit(2);
    };
    let mut target = match build_target(&target_spec, &workload, opt.input_override, opt.output_override) {
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
    /// KV heads. Defaults to the usual Qwen3 GQA ratio (`n_heads/4`) when
    /// `n_heads` divides evenly, else MHA (`n_heads`) -- but an explicit
    /// 6th shape component overrides this, because the derived guess does
    /// NOT match real Qwen3-0.6B (`n_heads=16` derives `n_kv_heads=4`; the
    /// real checkpoint is 8). A target meant to measure the REAL KV shape
    /// (not just a same-param-count stand-in) needs to say so explicitly.
    pub n_kv_heads: u32,
    /// Defaults to `d_model/n_heads`; overridden by an explicit 5th shape
    /// component for the same reason as `n_kv_heads` (real Qwen3-0.6B is
    /// head_dim=128; `d_model/n_heads` at `1024/16` derives 64).
    pub head_dim: u32,
    pub vocab: u32,
    pub block_size: u32,
    pub max_batch: u32,
    pub num_blocks: u32,
    pub per_seq: u32,
    pub max_prefill: u32,
    /// From `:i8w` -- opt IN to int8 weights, off by default.
    pub weights_int8: bool,
    /// From `:kvf32` -- opt OUT of int8 KV, which is ON by default.
    pub kv_fp32: bool,
}

impl SynthSpec {
    /// Parse `<L>x<D>x<H>[xV[xHeadDim[xNKvHeads]]][:i8w][:kvf32]` (flags in
    /// either order) and size a KV pool for `workload`, honouring `--input`/
    /// `--output` overrides the same way the actual request stream will
    /// (`crate::scenarios::workload_for`) -- see `pool_for`'s doc comment for
    /// why sizing from the un-overridden preset is a real OOM risk, not a
    /// theoretical one. The trailing `HeadDim`/`NKvHeads` are optional -- see
    /// the fields' doc comments for why a caller needs them to measure a
    /// REAL model's KV shape rather than a same-param-count stand-in.
    pub fn parse(shape: &str, workload: &str, input_override: Option<usize>, output_override: Option<usize>) -> Result<SynthSpec, String> {
        let (shape, weights_int8, kv_fp32) = spec_flags(shape);
        let parts: Vec<u32> = shape.split('x').map(|p| p.trim().parse().unwrap_or(0)).collect();
        if parts.len() < 3 || parts[..3].contains(&0) {
            return Err(format!("bad shape {shape:?}, expected <layers>x<d_model>x<heads>[x<vocab>[x<head_dim>[x<n_kv_heads>]]]"));
        }
        let (n_layers, d_model, n_heads) = (parts[0], parts[1], parts[2]);
        if d_model % n_heads != 0 {
            return Err(format!("d_model {d_model} must be divisible by n_heads {n_heads}"));
        }
        let head_dim = parts.get(4).copied().filter(|&v| v > 0).unwrap_or(d_model / n_heads);
        let n_kv_heads = parts
            .get(5)
            .copied()
            .filter(|&v| v > 0)
            .unwrap_or(if n_heads.is_multiple_of(4) { n_heads / 4 } else { n_heads });
        let (block_size, max_batch, num_blocks, per_seq, max_prefill) = pool_for(workload, input_override, output_override)?;
        Ok(SynthSpec {
            n_layers,
            d_model,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab: parts.get(3).copied().unwrap_or(32_000),
            block_size,
            max_batch,
            num_blocks,
            per_seq,
            max_prefill,
            weights_int8,
            kv_fp32,
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
        qwen::QwenConfig {
            vocab: self.vocab,
            block_size: 4096,
            n_layers: self.n_layers,
            d_model: self.d_model,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            d_ff: self.d_model * 4,
            rope_theta: 1_000_000.0,
            rms_eps: 1e-6,
            max_position_embeddings: 4096,
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
        let kv_int8 = resolve_kv_int8(&cfg, self.kv_fp32, &self.shape());
        qwen::serve::Engine::from_map_on(
            parent.gpu(),
            cfg,
            w,
            self.block_size,
            self.num_blocks,
            self.max_batch,
            self.per_seq,
            self.max_prefill,
            kv_int8,
            self.weights_int8,
        )
    }

    pub fn build_engine_with_blocks(
        &self,
        cfg: qwen::QwenConfig,
        w: &std::collections::HashMap<String, Vec<f32>>,
        num_blocks: u32,
    ) -> qwen::serve::Engine {
        let kv_int8 = resolve_kv_int8(&cfg, self.kv_fp32, &self.shape());
        qwen::serve::Engine::from_map(
            cfg,
            w,
            self.block_size,
            num_blocks,
            self.max_batch,
            self.per_seq,
            self.max_prefill,
            kv_int8,
            self.weights_int8,
        )
    }
}

fn build_target(spec: &str, workload: &str, input_override: Option<usize>, output_override: Option<usize>) -> Result<Box<dyn PerfTarget>, String> {
    if let Some(shape) = spec.strip_prefix("qwen-synth:") {
        return build_qwen_synth(shape, workload, input_override, output_override);
    }
    if let Some(weights) = spec.strip_prefix("qwen:") {
        return build_qwen(weights, workload, input_override, output_override);
    }
    if let Some(rest) = spec.strip_prefix("http:") {
        return build_http(rest, workload, input_override, output_override);
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
    if let Some(rest) = spec.strip_prefix("gpt:") {
        return build_gpt(rest);
    }
    if let Some(rest) = spec.strip_prefix("glm:") {
        return build_glm(rest);
    }
    if let Some(rest) = spec.strip_prefix("yolo:") {
        return build_yolo(rest);
    }
    if let Some(rest) = spec.strip_prefix("depth:") {
        return build_depth(rest);
    }
    if let Some(rest) = spec.strip_prefix("sam2:") {
        return build_sam2(rest);
    }
    if let Some(rest) = spec.strip_prefix("clip:") {
        return build_clip(rest);
    }
    if let Some(rest) = spec.strip_prefix("facenet:") {
        return build_facenet(rest);
    }
    if let Some(rest) = spec.strip_prefix("tts:") {
        return build_tts(rest);
    }
    if spec == "zimage" {
        return build_zimage("");
    }
    if let Some(rest) = spec.strip_prefix("zimage:") {
        return build_zimage(rest);
    }
    if let Some(rest) = spec.strip_prefix("upscale:") {
        return build_upscale(rest);
    }
    if let Some(rest) = spec.strip_prefix("restore:") {
        return build_restore(rest);
    }
    if let Some(rest) = spec.strip_prefix("vqgan:") {
        return build_vqgan(rest);
    }
    if let Some(rest) = spec.strip_prefix("nemotron:") {
        return build_nemotron(rest);
    }
    if let Some(rest) = spec.strip_prefix("qwen-asr:") {
        return build_qwen_asr(rest);
    }
    Err(format!(
        "unknown --target {spec:?} \
         (expected 'qwen-synth:<L>x<D>x<H>[xV][:i8w][:kvf32]', 'qwen:<weights>[:i8w][:kvf32]', \
         'http:qwen-synth:<L>x<D>x<H>[xV]:<tokenizer.json>', 'http:qwen:<weights>:<tokenizer.json>', \
         'lfm:<weights>:<tokenizer.json>', 'kronos:<tokenizer-dir>:<decoder-dir>', \
         'chronos2:<weights>', 'fincast:<weights>', 'flux2[:<W>x<H>x<steps>[:<precision>]]', \
         'gpt:<weights>', 'glm:<weights>', 'yolo:<weights>', 'depth:<weights>', \
         'sam2:<weights-dir>[:tiny|large]', 'clip:<text-encoder-dir>:<eva-clip.pt>', \
         'facenet:<weights-dir>', 'tts:<weights-dir>:<hf-ckpt-dir>', \
         'zimage[:<W>x<H>x<steps>]', 'upscale:<weights>', 'restore:<weights>', \
         'vqgan:<weights>', 'nemotron:<checkpoint-dir>', or 'qwen-asr:<checkpoint-dir>')"
    ))
}

/// `http:qwen-synth:<L>x<D>x<H>[xV]:<tokenizer.json>` or
/// `http:qwen:<weights.brain>:<tokenizer.json>` — the model measured through
/// the REAL served path: a real `residency::Executor` holding a real
/// `QwenResident`, behind the real `apiserve::router()`, driven over HTTP
/// in-process (`perf::targets::HttpTarget`). This is the target the serving-
/// performance audit's 600s regression (`.todo/serving-performance-audit.md`)
/// would have shown up in; the `qwen-synth:`/`qwen:` targets above measure the
/// paged engine directly and skip the whole HTTP/bridge/residency layer the
/// bug actually lived in. The tokenizer suffix is required (unlike the
/// engine-direct targets, which synthesize token ids and never tokenize) — a
/// real client sends text, and this target must too.
fn build_http(rest: &str, workload: &str, input_override: Option<usize>, output_override: Option<usize>) -> Result<Box<dyn PerfTarget>, String> {
    let (inner, tokenizer) = rest.rsplit_once(':').ok_or_else(|| {
        "http target needs 'http:qwen-synth:<L>x<D>x<H>[xV]:<tokenizer.json>' \
         or 'http:qwen:<weights>:<tokenizer.json>'"
            .to_string()
    })?;
    if !std::path::Path::new(tokenizer).exists() {
        return Err(format!("http target tokenizer not found: {tokenizer}"));
    }
    if let Some(shape) = inner.strip_prefix("qwen-synth:") {
        return build_http_qwen_synth(shape, tokenizer, workload, input_override, output_override);
    }
    if let Some(weights) = inner.strip_prefix("qwen:") {
        return build_http_qwen(weights, tokenizer);
    }
    Err(format!(
        "unknown http target {rest:?} (expected 'qwen-synth:<L>x<D>x<H>[xV]:<tok>' or 'qwen:<weights>:<tok>')"
    ))
}

/// Random weights of the given shape, written to a scratch checkpoint so
/// `QwenResident` can `mmap` them exactly as it would a real one — the point
/// is measuring the served path's OWN weight-loading/activation code, not
/// bypassing it. Content is irrelevant to serving cost; only shape matters,
/// same rationale as `SynthSpec::build_weights` for the engine-direct target.
///
/// `spec.weights_int8`/`spec.kv_fp32` are deliberately UNUSED here: this
/// target writes a real checkpoint and serves it through the real
/// `QwenResident`/residency executor, which decides KV/weights dtype from
/// `BRAIN_QWEN_KV_INT8`/`--weights-int8` same as `brain serve` -- exactly the
/// point of measuring the REAL served path rather than a direct-engine
/// stand-in. `spec.head_dim`/`n_kv_heads` ARE honoured, via `cfg.to_json()`
/// in the written checkpoint.
fn build_http_qwen_synth(shape: &str, tokenizer: &str, workload: &str, input_override: Option<usize>, output_override: Option<usize>) -> Result<Box<dyn PerfTarget>, String> {
    // Only `spec.config()` (architecture) is used below -- the pool fields
    // this parse ALSO computes are irrelevant here, since this target's real
    // serving pool comes from `QwenResident::pool_sizing` (BRAIN_QWEN_CTX),
    // not from a workload preset. Still threaded for signature consistency
    // and because a bad override should still error the same way.
    let spec = SynthSpec::parse(shape, workload, input_override, output_override)?;
    let (cfg, weights) = spec.build_weights();
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, n)| {
            let v = weights.get(&name).unwrap_or_else(|| panic!("perf: http qwen-synth missing tensor {name}")).clone();
            (name, vec![n as u64], v)
        })
        .collect();
    let dir = std::env::temp_dir().join(format!("brain-perf-http-qwen-synth-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("perf: http qwen-synth scratch dir: {e}"))?;
    let path = dir.join("model.safetensors");
    checkpoint::save(path.to_str().expect("utf8 path"), cfg.to_json(), &tensors);
    let info = TargetInfo::new("qwen-synth", "token").with("shape", spec.shape().into()).with("weights", "random".into()).with(
        "engine",
        "http".into(),
    );
    build_http_target(path.to_str().expect("utf8 path"), tokenizer, "qwen-synth", info)
}

fn build_http_qwen(weights: &str, tokenizer: &str) -> Result<Box<dyn PerfTarget>, String> {
    if !std::path::Path::new(weights).exists() {
        return Err(format!("http qwen weights not found: {weights}"));
    }
    let info = TargetInfo::new("qwen", "token").with("weights", weights.into()).with("engine", "http".into());
    build_http_target(weights, tokenizer, "qwen", info)
}

/// Shared plumbing: one `QwenResident` behind a real `residency::Executor`
/// (budgeted only over the schedulable devices — see `build_lfm`'s comment on
/// why an excluded GPU must never be budgeted), wrapped in the real
/// `apiserve` OpenAI router, driven over HTTP by `HttpTarget`.
fn build_http_target(weights: &str, tokenizer: &str, model_id: &str, info: TargetInfo) -> Result<Box<dyn PerfTarget>, String> {
    let card = checkpoint::st::ModelCard::new(model_id, "qwen");
    let resident = crate::resident_llm::QwenResident::from_card(weights, &card, Some(tokenizer), None);

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
    let exec = residency::Executor::start(vec![std::sync::Arc::new(resident)], budgets, residency::Policy::default());

    const KEY: &str = "perf-http-target-key";
    let state = apiserve::AppState::new(exec, KEY, apiserve::Provider::OpenAI);
    let router = apiserve::router(state);
    Ok(Box::new(perf::targets::HttpTarget::new(router, model_id, KEY, info)))
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
        if p.len() != 3 || p.contains(&0) {
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

/// A deterministic, non-degenerate synthetic RGB image: a diagonal gradient
/// plus a per-pixel pseudo-random speckle (an LCG seeded from the pixel
/// index, so it's reproducible without pulling in `data::rng` here). Real
/// enough that a model's forward pass sees varied, non-constant input
/// (constant input can hide a broadcast bug or trivially fold in a kernel),
/// while costing nothing to generate and needing no file on disk — exactly
/// what "random weights is fine" already licenses for the checkpoint side of
/// this suite, extended to the input side for the vision targets that have
/// no natural synthetic-shape target the way `qwen-synth:` does.
fn synth_image_blob(w: u32, h: u32) -> capability::Blob {
    let mut hwc = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let mut lcg = (y as u64 * 1_000_003 + x as u64).wrapping_mul(2_654_435_761);
            for c in 0..3u32 {
                lcg = lcg.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let speckle = ((lcg >> 33) as u32 % 32) as f32 / 255.0;
                let gradient = ((x + y + c) % (w.max(h).max(1))) as f32 / (w.max(h).max(1) as f32);
                hwc.push((gradient + speckle).min(1.0));
            }
        }
    }
    capability::blob::image_blob(&hwc, w, h, 3)
}

/// A deterministic synthetic mono waveform at 16 kHz: a few summed sine
/// tones (never pure silence, which some VAD/energy-gated front ends treat
/// as "nothing to transcribe" and short-circuit, which would measure the
/// short-circuit path rather than the model). `secs` real-time -> exactly
/// `secs * 16000` samples.
fn synth_audio_blob(secs: f32) -> capability::Blob {
    let n = (secs.max(0.1) * 16_000.0) as usize;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            0.2 * (2.0 * std::f32::consts::PI * 220.0 * t).sin() + 0.1 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();
    let bytes: Vec<u8> = pcm.iter().flat_map(|f| f.to_le_bytes()).collect();
    capability::Blob::new(capability::Media::Audio, bytes).with_meta(serde_json::json!({"sample_rate": 16000}))
}

/// `gpt:<weights>` — the dense char-level GPT baseline behind the residency
/// executor (unit: token). `--input`/`--output` control prompt length and
/// generated length, same shape as every LLM target in this file.
fn build_gpt(weights: &str) -> Result<Box<dyn PerfTarget>, String> {
    if !std::path::Path::new(weights).exists() {
        return Err(format!("gpt weights not found: {weights}"));
    }
    // SAFETY: single-threaded CLI startup, no concurrent env access.
    unsafe { std::env::set_var("BRAIN_GPT_WEIGHTS", weights) };
    let resident = crate::resident_llm::GptResident::from_env().ok_or("gpt: BRAIN_GPT_WEIGHTS did not resolve")?;
    let exec = forecast_executor(resident);
    let info = vec![("weights".to_string(), serde_json::json!(weights)), ("engine".to_string(), serde_json::json!("residency-executor"))];
    let build = Box::new(|req: &perf::target::PerfRequest| {
        capability::Invocation::new()
            .set("prompt", serde_json::json!("word ".repeat(req.input_artifacts)))
            .set("max_new", serde_json::json!(req.output_artifacts))
    });
    Ok(Box::new(perf::targets::ExecutorTarget::new(exec, "brain/gpt", "generate", "token", info, build)))
}

/// `glm:<weights>` — the GLM-5.2-shaped decoder (MLA + sigmoid-gated MoE)
/// behind the residency executor (unit: token).
fn build_glm(weights: &str) -> Result<Box<dyn PerfTarget>, String> {
    if !std::path::Path::new(weights).exists() {
        return Err(format!("glm weights not found: {weights}"));
    }
    unsafe { std::env::set_var("BRAIN_GLM_WEIGHTS", weights) };
    let resident = crate::resident_llm::GlmResident::from_env().ok_or("glm: BRAIN_GLM_WEIGHTS did not resolve")?;
    let exec = forecast_executor(resident);
    let info = vec![("weights".to_string(), serde_json::json!(weights)), ("engine".to_string(), serde_json::json!("residency-executor"))];
    let build = Box::new(|req: &perf::target::PerfRequest| {
        capability::Invocation::new()
            .set("prompt", serde_json::json!("word ".repeat(req.input_artifacts)))
            .set("max_new", serde_json::json!(req.output_artifacts))
    });
    Ok(Box::new(perf::targets::ExecutorTarget::new(exec, "brain/glm", "generate", "token", info, build)))
}

/// `yolo:<weights>` — YOLOv8-style detection behind the residency executor
/// (unit: frame). One 640x640 synthetic image per request, matching the
/// model's native training resolution.
fn build_yolo(weights: &str) -> Result<Box<dyn PerfTarget>, String> {
    if !std::path::Path::new(weights).exists() {
        return Err(format!("yolo weights not found: {weights}"));
    }
    unsafe { std::env::set_var("BRAIN_YOLO", weights) };
    let resident = crate::resident::YoloResident::from_env().ok_or("yolo: BRAIN_YOLO did not resolve")?;
    let exec = forecast_executor(resident);
    let info = vec![("weights".to_string(), serde_json::json!(weights)), ("engine".to_string(), serde_json::json!("residency-executor"))];
    let build = Box::new(|_req: &perf::target::PerfRequest| {
        capability::Invocation::new().blob("image", synth_image_blob(640, 640))
    });
    Ok(Box::new(perf::targets::ExecutorTarget::new(exec, "brain/yolo", "detect", "frame", info, build)))
}

/// `depth:<weights>` — ZipDepth monocular depth behind the residency
/// executor (unit: frame). The checkpoint's native input (0 = auto-detect,
/// typically 384) is left at its default rather than forced, so the target
/// measures what `brain depth` actually serves.
fn build_depth(weights: &str) -> Result<Box<dyn PerfTarget>, String> {
    if !std::path::Path::new(weights).exists() {
        return Err(format!("depth weights not found: {weights}"));
    }
    unsafe { std::env::set_var("BRAIN_DEPTH_WEIGHTS", weights) };
    let resident = crate::resident_depth::DepthResident::from_env().ok_or("depth: BRAIN_DEPTH_WEIGHTS did not resolve")?;
    let exec = forecast_executor(resident);
    let info = vec![("weights".to_string(), serde_json::json!(weights)), ("engine".to_string(), serde_json::json!("residency-executor"))];
    let build =
        Box::new(|_req: &perf::target::PerfRequest| capability::Invocation::new().blob("image", synth_image_blob(384, 384)));
    Ok(Box::new(perf::targets::ExecutorTarget::new(exec, "brain/depth", "depth", "frame", info, build)))
}

/// `sam2:<weights-dir>[:tiny|large]` — SAM 2.1 promptable segmentation behind
/// the residency executor (unit: mask). One centred point prompt per
/// request — a degenerate zero-prompt call is a valid API call but not a
/// representative one.
fn build_sam2(rest: &str) -> Result<Box<dyn PerfTarget>, String> {
    let (dir, variant) = rest.split_once(':').unwrap_or((rest, "tiny"));
    if !std::path::Path::new(dir).exists() {
        return Err(format!("sam2 weights dir not found: {dir}"));
    }
    sam2::caps::variant_config(variant)?;
    unsafe { std::env::set_var("BRAIN_SAM2_WEIGHTS", dir) };
    let resident = crate::resident_sam2::Sam2Resident::from_env().ok_or("sam2: BRAIN_SAM2_WEIGHTS did not resolve")?;
    let exec = forecast_executor(resident);
    let info = vec![
        ("weights".to_string(), serde_json::json!(dir)),
        ("variant".to_string(), serde_json::json!(variant)),
        ("engine".to_string(), serde_json::json!("residency-executor")),
    ];
    let variant = variant.to_string();
    let build = Box::new(move |_req: &perf::target::PerfRequest| {
        capability::Invocation::new()
            .blob("image", synth_image_blob(1024, 1024))
            .set("variant", serde_json::json!(variant))
            .set("points", serde_json::json!("512,512"))
            .set("labels", serde_json::json!("1"))
    });
    Ok(Box::new(perf::targets::ExecutorTarget::new(exec, sam2::caps::MODEL, "segment", "mask", info, build)))
}

/// `clip:<sdxl-text-encoder-dir>:<eva-clip.pt>` — the EVA-CLIP-L/336 image
/// tower behind the residency executor (unit: embedding). Text towers exist
/// on the same resident but need a real tokenizer to exercise honestly;
/// `embed_image` needs only a pixel grid.
fn build_clip(rest: &str) -> Result<Box<dyn PerfTarget>, String> {
    let (text_encoder_dir, eva_path) =
        rest.split_once(':').ok_or("clip target needs 'clip:<sdxl-text-encoder-dir>:<eva-clip.pt>'")?;
    if !std::path::Path::new(eva_path).exists() {
        return Err(format!("clip EVA-CLIP weights not found: {eva_path}"));
    }
    unsafe {
        std::env::set_var("BRAIN_CLIP_TEXT_ENCODER", text_encoder_dir);
        std::env::set_var("BRAIN_CLIP_EVA", eva_path);
    }
    let resident = crate::resident_clip::ClipResident::from_env().ok_or("clip: BRAIN_CLIP_* did not resolve")?;
    let exec = forecast_executor(resident);
    let info = vec![("weights".to_string(), serde_json::json!(eva_path)), ("engine".to_string(), serde_json::json!("residency-executor"))];
    let build = Box::new(|_req: &perf::target::PerfRequest| {
        capability::Invocation::new().blob("image", synth_image_blob(336, 336))
    });
    Ok(Box::new(perf::targets::ExecutorTarget::new(exec, clip::caps::MODEL, "embed_image", "embedding", info, build)))
}

/// `facenet:<weights-dir>` — SCRFD detect + ArcFace embed behind the
/// residency executor (unit: embedding). `align=true` (the default) runs
/// full detect+align+embed on a whole image, so a generic synthetic image
/// exercises the real served path — it will not *find* a face, but it
/// measures the same dispatch sequence a real one takes.
fn build_facenet(dir: &str) -> Result<Box<dyn PerfTarget>, String> {
    if !std::path::Path::new(dir).exists() {
        return Err(format!("facenet weights dir not found: {dir}"));
    }
    unsafe { std::env::set_var("BRAIN_FACENET_WEIGHTS", dir) };
    let resident = crate::resident_facenet::FacenetResident::from_env().ok_or("facenet: BRAIN_FACENET_WEIGHTS did not resolve")?;
    let exec = forecast_executor(resident);
    let info = vec![("weights".to_string(), serde_json::json!(dir)), ("engine".to_string(), serde_json::json!("residency-executor"))];
    let build = Box::new(|_req: &perf::target::PerfRequest| {
        capability::Invocation::new().blob("image", synth_image_blob(640, 480))
    });
    Ok(Box::new(perf::targets::ExecutorTarget::new(exec, facenet::caps::MODEL, "embed", "embedding", info, build)))
}

/// `tts:<weights-dir>:<hf-ckpt-dir>` — Qwen3-TTS speaker-free synthesis
/// behind the residency executor (unit: audio_chunk, streaming). `--output`
/// caps generated codec frames via `max_frames`.
fn build_tts(rest: &str) -> Result<Box<dyn PerfTarget>, String> {
    let (weights_dir, ckpt_dir) = rest.split_once(':').ok_or("tts target needs 'tts:<weights-dir>:<hf-ckpt-dir>'")?;
    if !std::path::Path::new(weights_dir).exists() {
        return Err(format!("tts weights dir not found: {weights_dir}"));
    }
    unsafe { std::env::set_var("BRAIN_TTS_WEIGHTS", weights_dir) };
    unsafe { std::env::set_var("BRAIN_TTS_CKPT", ckpt_dir) };
    let resident = crate::resident_tts::TtsResident::from_env().ok_or("tts: BRAIN_TTS_WEIGHTS did not resolve")?;
    let exec = forecast_executor(resident);
    let info = TargetInfo::new("tts", "audio_chunk").with("weights", weights_dir.into()).with("engine", "residency-executor".into());
    let build = Box::new(|req: &perf::target::PerfRequest| {
        capability::Invocation::new()
            .set("text", serde_json::json!("word ".repeat(req.input_artifacts)))
            .set("max_frames", serde_json::json!(req.output_artifacts))
    });
    Ok(Box::new(perf::targets::ExecutorTarget::new_streaming(
        exec,
        tts::caps::MODEL,
        "speak",
        info,
        build,
        std::sync::Arc::new(|p: &capability::Progress| p.delta.is_some()),
    )))
}

/// `zimage[:<W>x<H>x<steps>]` — Z-Image (Tongyi S3-DiT) text-to-image behind
/// the residency executor (unit: denoise_step). Mirrors `build_flux2`
/// exactly; large (13-24 GiB VRAM), so check the memory-safety protocol
/// (`docs/performance/arc.md` §0) before running on a unified-memory box.
fn build_zimage(rest: &str) -> Result<Box<dyn PerfTarget>, String> {
    let (w, h, steps) = if rest.is_empty() {
        (512u32, 512u32, 4u32)
    } else {
        let p: Vec<u32> = rest.split('x').map(|s| s.trim().parse().unwrap_or(0)).collect();
        if p.len() != 3 || p.contains(&0) {
            return Err(format!("bad zimage spec {rest:?}, expected <W>x<H>x<steps> (e.g. 512x512x20)"));
        }
        (p[0], p[1], p[2])
    };
    for var in ["BRAIN_ZIMAGE_DIT", "BRAIN_ZIMAGE_VAE", "BRAIN_ZIMAGE_QWEN", "BRAIN_ZIMAGE_TOKENIZER"] {
        if std::env::var(var).map(|v| v.is_empty()).unwrap_or(true) {
            return Err(format!("zimage target needs {var} set"));
        }
    }
    let resident = crate::resident::ZImageResident::from_env()?;
    let exec = forecast_executor(resident);
    let info = TargetInfo::new("zimage", "denoise_step")
        .with("width", w.into())
        .with("height", h.into())
        .with("steps", steps.into())
        .with("engine", "residency-executor".into());
    let build = Box::new(move |req: &perf::target::PerfRequest| {
        capability::Invocation::new()
            .set("prompt", serde_json::json!("a lighthouse on a rocky coast at sunset"))
            .set("width", serde_json::json!(w))
            .set("height", serde_json::json!(h))
            .set("steps", serde_json::json!(steps))
            .set("seed", serde_json::json!(req.seed))
    });
    Ok(Box::new(perf::targets::ExecutorTarget::new_streaming(
        exec,
        zimage::caps::MODEL,
        "text2image",
        info,
        build,
        std::sync::Arc::new(|p: &capability::Progress| p.message == "denoising"),
    )))
}

/// `upscale:<weights>` — Real-ESRGAN RRDBNet super-resolution behind the
/// residency executor (unit: image). No batching (activation-dominated —
/// `resident_upscale.rs`'s own doc explains why); one 256x256 input tile.
fn build_upscale(weights: &str) -> Result<Box<dyn PerfTarget>, String> {
    if !std::path::Path::new(weights).exists() {
        return Err(format!("upscale weights not found: {weights}"));
    }
    unsafe { std::env::set_var("BRAIN_UPSCALE_WEIGHTS", weights) };
    let resident = crate::resident_upscale::UpscaleResident::from_env().ok_or("upscale: BRAIN_UPSCALE_WEIGHTS did not resolve")?;
    let exec = forecast_executor(resident);
    let info = vec![("weights".to_string(), serde_json::json!(weights)), ("engine".to_string(), serde_json::json!("residency-executor"))];
    let build = Box::new(|_req: &perf::target::PerfRequest| {
        capability::Invocation::new().blob("image", synth_image_blob(256, 256))
    });
    Ok(Box::new(perf::targets::ExecutorTarget::new(exec, upscale::caps::MODEL, "upscale", "image", info, build)))
}

/// `restore:<weights>` — CodeFormer blind face restoration behind the
/// residency executor (unit: image). Fixed 512x512 graph.
fn build_restore(weights: &str) -> Result<Box<dyn PerfTarget>, String> {
    if !std::path::Path::new(weights).exists() {
        return Err(format!("restore weights not found: {weights}"));
    }
    unsafe { std::env::set_var("BRAIN_RESTORE_WEIGHTS", weights) };
    let resident = crate::resident_restore::RestoreResident::from_env().ok_or("restore: BRAIN_RESTORE_WEIGHTS did not resolve")?;
    let exec = forecast_executor(resident);
    let info = vec![("weights".to_string(), serde_json::json!(weights)), ("engine".to_string(), serde_json::json!("residency-executor"))];
    let build = Box::new(|_req: &perf::target::PerfRequest| {
        capability::Invocation::new().blob("image", synth_image_blob(512, 512)).set("w", serde_json::json!(0.5))
    });
    Ok(Box::new(perf::targets::ExecutorTarget::new(exec, restore::caps::MODEL, "restore_face", "image", info, build)))
}

/// `vqgan:<weights>` — the CodeFormer VQ autoencoder's encode half behind the
/// residency executor (unit: image). Fixed 512x512 graph (the released
/// checkpoints' training resolution).
fn build_vqgan(weights: &str) -> Result<Box<dyn PerfTarget>, String> {
    if !std::path::Path::new(weights).exists() {
        return Err(format!("vqgan weights not found: {weights}"));
    }
    unsafe { std::env::set_var("BRAIN_VQGAN_WEIGHTS", weights) };
    let resident = crate::resident_restore::VqganResident::from_env().ok_or("vqgan: BRAIN_VQGAN_WEIGHTS did not resolve")?;
    let exec = forecast_executor(resident);
    let info = vec![("weights".to_string(), serde_json::json!(weights)), ("engine".to_string(), serde_json::json!("residency-executor"))];
    let build = Box::new(|_req: &perf::target::PerfRequest| {
        capability::Invocation::new().blob("image", synth_image_blob(512, 512))
    });
    Ok(Box::new(perf::targets::ExecutorTarget::new(exec, vqgan::caps::MODEL, "encode", "image", info, build)))
}

/// `nemotron:<checkpoint-dir>` — FastConformer streaming ASR behind the
/// residency executor (unit: transcript_token — deliberately NOT "token":
/// ranking ASR tokens against LLM tokens is exactly the meaningless
/// cross-unit comparison the report's `artifact_unit` guard exists to
/// prevent). `--input` controls synthetic audio length in whole seconds.
fn build_nemotron(dir: &str) -> Result<Box<dyn PerfTarget>, String> {
    if !std::path::Path::new(dir).exists() {
        return Err(format!("nemotron checkpoint dir not found: {dir}"));
    }
    unsafe { std::env::set_var("BRAIN_NEMOTRON", dir) };
    let resident = crate::resident_asr::NemotronResident::from_env().ok_or("nemotron: BRAIN_NEMOTRON did not resolve")?;
    let exec = forecast_executor(resident);
    let info =
        TargetInfo::new("nemotron", "transcript_token").with("weights", dir.into()).with("engine", "residency-executor".into());
    let build = Box::new(|req: &perf::target::PerfRequest| {
        capability::Invocation::new().blob("audio", synth_audio_blob(req.input_artifacts.max(1) as f32))
    });
    Ok(Box::new(perf::targets::ExecutorTarget::new_streaming(
        exec,
        nemotron::caps::MODEL,
        "transcribe",
        info,
        build,
        std::sync::Arc::new(|p: &capability::Progress| p.delta.is_some()),
    )))
}

/// `qwen-asr:<checkpoint-dir>` — the Whisper-style-encoder + Qwen3 decoder
/// offline ASR behind the residency executor (unit: transcript_token, same
/// reasoning as `build_nemotron`).
fn build_qwen_asr(dir: &str) -> Result<Box<dyn PerfTarget>, String> {
    if !std::path::Path::new(dir).exists() {
        return Err(format!("qwen-asr checkpoint dir not found: {dir}"));
    }
    unsafe { std::env::set_var("BRAIN_QWEN_ASR", dir) };
    let resident = crate::resident_asr::QwenAsrResident::from_env().ok_or("qwen-asr: BRAIN_QWEN_ASR did not resolve")?;
    let exec = forecast_executor(resident);
    let info =
        TargetInfo::new("qwen-asr", "transcript_token").with("weights", dir.into()).with("engine", "residency-executor".into());
    let build = Box::new(|req: &perf::target::PerfRequest| {
        capability::Invocation::new().blob("audio", synth_audio_blob(req.input_artifacts.max(1) as f32))
    });
    Ok(Box::new(perf::targets::ExecutorTarget::new_streaming(
        exec,
        qwen_asr::caps::MODEL,
        "transcribe",
        info,
        build,
        std::sync::Arc::new(|p: &capability::Progress| p.delta.is_some()),
    )))
}

/// Split optional trailing target-spec flags, in either order:
/// `:i8w` (int8 weights, opt IN — off by default) and `:kvf32` (fp32 KV,
/// opt OUT of int8 KV, which perf targets default to ON — matching
/// `resident_llm.rs::QwenResident::kv_int8`'s serving default, so perf
/// numbers stay representative of what actually ships).
fn spec_flags(spec: &str) -> (&str, bool, bool) {
    let mut s = spec;
    let mut weights_int8 = false;
    let mut kv_fp32 = false;
    loop {
        if let Some(rest) = s.strip_suffix(":i8w") {
            s = rest;
            weights_int8 = true;
            continue;
        }
        if let Some(rest) = s.strip_suffix(":kvf32") {
            s = rest;
            kv_fp32 = true;
            continue;
        }
        break;
    }
    (s, weights_int8, kv_fp32)
}

/// The int8-KV boundary policy (D2), in ONE place: `:kvf32` is an explicit
/// opt-out (nothing to degrade); absent that, int8 is the default and this
/// degrades loudly on a shape it cannot support rather than hitting
/// `from_map_with_gpu`'s hard assert -- see `qwen::serve::kv_int8_supported`'s
/// doc comment. Shared by every perf target builder so the same shape always
/// gets the same answer, and the degrade warning is worded once.
fn resolve_kv_int8(cfg: &qwen::QwenConfig, kv_fp32_requested: bool, target_desc: &str) -> bool {
    let kv_int8 = !kv_fp32_requested && qwen::serve::kv_int8_supported(cfg);
    if !kv_fp32_requested && !kv_int8 {
        eprintln!("perf: {target_desc}: int8 KV requested (the default) but head_dim={} is not a multiple of 4; falling back to fp32 KV", cfg.head_dim);
    }
    kv_int8
}

/// Build the serving engine on **randomly initialised weights** of a given
/// shape: `qwen-synth:<layers>x<d_model>x<heads>[x<vocab>[x<head_dim>[x<n_kv_heads>]]]`.
///
/// Weight *values* do not affect execution cost — the same kernels, KV traffic,
/// batching and memory pressure occur whatever the numbers are — so this
/// measures the real engine without needing a checkpoint on the machine. It is
/// the right tool for hardware and configuration comparison, and the wrong tool
/// for anything about output quality: generated tokens are meaningless, so the
/// artifact records `weights: "random"` and no correctness gate can pass on it.
///
/// Shares `SynthSpec::parse`/`config`/`build_engine` with every other synth
/// caller (`startup`/`cancel`/`kvcache`/`faults`/`http:qwen-synth:`) rather
/// than a second, independent copy of the shape-to-config arithmetic
/// (`docs/lessons.md` #19: a registration/derivation split across N call
/// sites is a defect waiting for its turn — this WAS two copies until now,
/// and the derived `head_dim`/`n_kv_heads` silently did not match real
/// Qwen3-0.6B's, understating both the memory win and the append kernel's
/// cost at the shape that mattered).
fn build_qwen_synth(shape: &str, workload: &str, input_override: Option<usize>, output_override: Option<usize>) -> Result<Box<dyn PerfTarget>, String> {
    let spec = SynthSpec::parse(shape, workload, input_override, output_override)?;
    let (cfg, weights) = spec.build_weights();
    let params: usize = cfg.param_list().iter().map(|(_, n)| n).sum();
    eprintln!(
        "perf: synthetic qwen L{} D{} H{} (kv {}, head_dim {}) vocab {} — {:.1}M params, random weights",
        spec.n_layers,
        spec.d_model,
        spec.n_heads,
        spec.n_kv_heads,
        spec.head_dim,
        spec.vocab,
        params as f64 / 1e6
    );

    let eng = spec.build_engine(cfg, &weights);
    let info = TargetInfo::new("qwen-synth", "token")
        .with("shape", spec.shape().into())
        .with("params", params.into())
        .with("weights", "random".into())
        .with("block_size", spec.block_size.into())
        .with("max_batch", spec.max_batch.into())
        // Reported explicitly (not just implied by `shape`), so an artifact
        // never silently implies it ran a different model's KV geometry.
        .with("head_dim", spec.head_dim.into())
        .with("n_kv_heads", spec.n_kv_heads.into())
        // What actually ran: an explicit `:kvf32` is honoured exactly, but the
        // DEFAULT is capability-gated (see resolve_kv_int8) just like
        // weights_dtype below.
        .with("kv_dtype", if eng.kv_int8() { "int8" } else { "fp32" }.into())
        // What actually ran: the int8 request is capability-gated in the engine.
        .with("weights_dtype", if eng.weights_int8() { "int8" } else { "fp32" }.into());
    let sched = qwen::serve::Scheduler::new(eng, spec.max_batch as usize);
    Ok(Box::new(perf::targets::PagedLlmTarget::new(sched, info, None, spec.vocab)))
}

/// KV-pool geometry for a workload: `(block_size, max_batch, num_blocks,
/// blocks_per_seq, max_prefill)`. Sized so *admission*, not allocation failure,
/// is what limits concurrency.
///
/// `input_override`/`output_override` MUST be the same `--input`/`--output`
/// values `Options` carries (`crate::scenarios::workload_for` applies them to
/// the actual request stream) — this was previously sized from the workload
/// PRESET's own shape unconditionally, so `--workload prefill_heavy --input
/// 256` (a deliberately small override) still allocated a pool for
/// `prefill_heavy`'s full 32768-token preset: a real ~64 GiB single
/// allocation on this box (rejected by the allocator, not silently wrong —
/// but the box has no discrete GPU and 30 GiB total RAM, so this was one
/// workload name away from a genuine OOM). The pool must be sized from what
/// the run will ACTUALLY request, not from a name that happens to share a
/// preset with a much larger one.
fn pool_for(workload: &str, input_override: Option<usize>, output_override: Option<usize>) -> Result<(u32, u32, u32, u32, u32), String> {
    let w = perf::workload::standard(workload, perf::Arrival::Saturated, 1, 0)
        .ok_or_else(|| format!("unknown workload {workload:?}"))?;
    let r = &w.requests()[0];
    // Floored at PagedLlmTarget::fidelity's own fixed probe size: that gate
    // runs on THIS SAME pool regardless of how small --input/--output shrink
    // the measured workload (e.g. under --smoke), so a pool sized purely from
    // a tiny override can be too small to admit the probe's own requests --
    // rejected at admission, zero positions ever compared, and the gate fails
    // with a confusing "greedy_token_match 1.0000 < 1.0000" (compared == 0
    // defaults token_match to 1.0; see Fidelity::greedy) instead of ever
    // actually running. Found via a real --smoke run on the real checkpoint.
    let max_in = input_override.map(|n| n as u32).unwrap_or(r.input_artifacts as u32).max(perf::targets::FIDELITY_PROMPT_TOKENS);
    let max_out = output_override.map(|n| n as u32).unwrap_or(r.output_artifacts as u32).max(perf::targets::FIDELITY_MAX_NEW);
    let block_size = 16u32;
    let max_batch = 32u32;
    let per_seq = (max_in + max_out + 8).div_ceil(block_size);
    let num_blocks = per_seq * max_batch + max_batch;
    Ok((block_size, max_batch, num_blocks, per_seq, max_in.max(1)))
}

/// Size the KV pool for the workload so admission, not allocation failure, is
/// what limits concurrency.
fn build_qwen(weights: &str, workload: &str, input_override: Option<usize>, output_override: Option<usize>) -> Result<Box<dyn PerfTarget>, String> {
    let (weights, weights_int8, kv_fp32) = spec_flags(weights);
    let (block_size, max_batch, num_blocks, per_seq, max_prefill) = pool_for(workload, input_override, output_override)?;
    // A header-only peek (not a second full checkpoint load) to feed the
    // shared boundary policy -- see resolve_kv_int8.
    let kv_int8 = match checkpoint::weightio::WeightReader::open(weights) {
        Ok(r) => resolve_kv_int8(&qwen::QwenConfig::from_json(&r.config()), kv_fp32, weights),
        Err(_) => !kv_fp32, // let Engine::load raise the real, specific I/O error below
    };
    let eng = qwen::serve::Engine::load(
        weights,
        block_size,
        num_blocks,
        max_batch,
        per_seq,
        max_prefill,
        kv_int8,
        weights_int8,
    );
    let vocab = eng.vocab() as u32;
    let w8_effective = eng.weights_int8();
    let kv_dtype = if eng.kv_int8() { "int8" } else { "fp32" };
    let sched = qwen::serve::Scheduler::new(eng, max_batch as usize);
    let info = TargetInfo::new("qwen", "token")
        .with("weights", weights.into())
        .with("block_size", block_size.into())
        .with("max_batch", max_batch.into())
        .with("kv_dtype", kv_dtype.into())
        .with("weights_dtype", if w8_effective { "int8" } else { "fp32" }.into());
    Ok(Box::new(perf::targets::PagedLlmTarget::new(sched, info, None, vocab)))
}

