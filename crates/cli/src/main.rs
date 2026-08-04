// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `brain` native CLI — one binary over every model in the workspace.
//! The model is chosen by the subcommand (no global "model type" flag).
//!
//!   * `gpt`        — dense GPT decoder baseline (nanogpt parity).
//!   * `generate`/`train`/`eval` — the sparse-MoE Transformer.
//!   * `federated`  — sharded-MoE shard split/assemble.
//!   * `data`       — dataset generation; `gradcheck` — backprop correctness gate.
//!   * `pid`        — event/effect control Transformer (the WebGPU demo).
//!
//! Run `brain help` for the full usage with examples.

mod args;
mod caps_cli;
mod data_cli;
mod depth_cli;
mod devices_cli;
mod federated_cli;
mod fetch;
mod fetch_cli;
mod flops_cli;
mod flux2_cli;
mod forecast_cli;
mod glm_cli;
mod gpt_cli;
mod image_io;
mod imageops;
mod lfm_cli;
mod mirror_cli;
mod model_dir;
mod npu_cli;
mod perf_cli;
mod perf_engine;
mod pid_cli;
mod qwen_cli;
mod resident;
mod resident_asr;
mod resident_depth;
mod resident_flux2;
mod resident_forecast;
mod resident_lfm;
mod resident_llm;
mod resident_mock;
mod resident_tts;
mod resident_facenet;
mod resident_sam2;
mod resident_restore;
mod run_cli;
mod splat_cli;
mod supply;
mod tts_cli;
mod tts_serve;
mod wm_cli;
mod yolo_cli;

use std::sync::atomic::{AtomicBool, Ordering};

/// Set when `--device npu` is requested. The NPU is a whole-graph (OpenVINO)
/// path, not a `gpu_core` backend, so it is tracked separately and consumed by
/// the commands that support it (today: `brain yolo detect`).
static NPU_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Whether the user asked for `--device npu`.
pub(crate) fn npu_requested() -> bool {
    NPU_REQUESTED.load(Ordering::Relaxed)
}

const HELP: &str = "\
brain — train and evaluate neural nets from scratch on the GPU (Rust + WGSL).

USAGE: brain <command> [options]
The model is selected by the command.

--device selects WHICH COMPUTE IS SCHEDULABLE. Omit it and brain uses every
device on the machine — all GPUs, the CPU, and an NPU if present — scheduling
models across them. Name devices to restrict the set (BRAIN_DEVICE does the same
without a flag):

  (omitted)     every device, scheduled together
  cpu           CPU only, all cores          gpu       every GPU, nothing else
  npu           NPU only                     vulkan    every GPU, native-Vulkan backend
  gpu,cpu       GPUs and CPU together        gpu0      only physical GPU 0
  gpu0,gpu1     those two cards              cpu21     only CPU core 21
  cpu0-7        CPU cores 0..=7              gpu1,cpu0-3   one card plus four cores

Indexed CPU selections pin process affinity and size the thread pool to match.
This bounds where work EXECUTES; host RAM and disk remain available as cache and
spill tiers regardless, so --device gpu still uses RAM for weight caching.

GPU indices are CANONICAL: physical cards sorted by PCI bus id (stable across
boots), shared by --device gpuN, shard placement and residency budgets.
`brain devices` prints the table (index, PCI bus, UUID, VRAM, backends) and what
the ambient selection resolves to.

DEVICES
  brain devices                     # canonical GPU table + ambient selection

DATA
  brain data gen <name> [--out DIR --n N --seed S]
      names: calculator | reverser | wordcalc | timeseries | shakespeare_char | gpt

GPT (dense baseline)
  brain gpt train <data_dir> [--out F --steps N --batch B --block T
                              --layers L --d-model D --heads H --lr X --mask = --align]
  brain gpt eval  --weights F --data <dir> [--batches N --samples M]
  brain gpt gen   --weights F [--data <dir>] [--prompt \"...\" --max-new N --temp X --top-k K]
                              (vocab is read from the checkpoint; --data only for old ones)

FLUX.2 Klein (text-to-image + image editing; 4-step distilled flow matching)
  brain flux2 generate --prompt \"...\" --out out.ppm [--width W --height H]
      [--steps N --seed S --variant klein-4b|klein-9b|base-4b|base-9b]
      [--guidance G]              # CFG, base variants only
      [--ref in.ppm]...           # reference images => editing mode
      weights via env: BRAIN_FLUX2_DIT, BRAIN_FLUX2_VAE, BRAIN_FLUX2_TE, BRAIN_FLUX2_TOKENIZER
      served generically as model `flux2-klein`: brain caps flux2-klein,
      brain do flux2-klein text2image|edit|lora_train, and D-Bus (examples/imagegen);
      9B variants need BRAIN_FLUX2_ALLOW_NC=1 (FLUX Non-Commercial license)

World models (playable action-conditioned video models; docs/models/world-models/)
  brain wm play  --model fake|diamond [--weights F --device cpu|gpu|npu --onnx M]   # SDL window
  brain wm play  --model fake --headless --frames N [--actions FILE | --action-seq 1,2,0]
                 [--dump-ppm DIR] [--hashes]        # deterministic rollout + fnv1a hashes (CI)
  brain wm bench --model fake [--frames N]          # ms/frame + fps

WorldMirror-2 (multi-view images → 3D Gaussian Splatting scene; docs/models/mirror/)
  brain mirror import <model.safetensors|hf_dir> --out mirror.safetensors
      One-time conversion of the reference HY-WorldMirror-2.0 checkpoint (strict
      1:1, every tensor verified).
  brain mirror infer --weights F --images <dir|a.ppm,b.ppm,…> [--out DIR]
      [--ply scene.ply] [--maps] [--min-opacity X] [--max-depth X] [--prune VOXEL]
      Images → navigable 3DGS scene (scene.ply + cameras.json + depth/normal
      maps). Any aspect ratio; --prune 0.002 voxel-merges multi-view duplicates.
  brain mirror demo  --weights F --images <…> [--width N --height N --fov D]
      infer + interactive fly-through of the reconstructed world.

3D GAUSSIAN SPLATTING (scene viewer/renderer; crates/splat)
  brain splat info   <scene.ply>
  brain splat render <scene.ply> --out img.ppm [--width N --height N]
        [--eye x,y,z --target x,y,z --up x,y,z --fov D] [--depth] [--bg r,g,b]
  brain splat view   <scene.ply> [--width N --height N --fov D --bg r,g,b]
        Interactive fly-through: WASD move, Space/C up/down, Shift sprint,
        m mouse-look, arrows look, [ ] quality, v depth view, p screenshot,
        Enter reset, Esc quit.

YOLO (from-scratch anchor-free object detector)
  brain yolo train <data_dir> --out F [--steps N --batch B --lr X --nc C
                                       --input S --seed S]
  brain yolo eval  --weights F --data <dir> [--conf X --iou X]   # mAP/precision/recall
  brain yolo detect --weights F --image <P6.ppm | dataset_dir> [--conf X --iou X]
                                                                # prints [x1,y1,x2,y2,conf,class] JSON lines
  brain yolo fine-tune <data_dir> --weights <pretrained> --out F [--freeze-backbone ...]
      Trains the tiny YOLOv8 graph on a `data gen detect` dataset (CPU backend).
  brain yolo detect --weights F --image <...> --device npu     # run on the Intel NPU

INTEL NPU (OpenVINO: quantize + compile YOLO to a real NPU graph)
  brain npu export   --weights F --out model.onnx [--input S --opset N]    # fp32 ONNX
  brain npu quantize --weights F --calib <dir> --out model.int8.onnx [--input S --num-calib N]
  brain npu check    --onnx M [--device NPU]              # ONNX op histogram + compile/coverage
  brain npu run      --onnx M --image <P6|dir> [--device NPU --conf X --iou X --nc C --reg-max R]
  brain npu bench    --onnx M [--input S --device NPU --iters N --hint latency|throughput]
  brain npu sim      --weights F --data <dir> [--calib <dir>]   # fp32 vs INT8 mAP, no NPU
      export/quantize/sim are pure Rust (any machine); run/bench/check-compile need
      OpenVINO + an Intel NPU. The whole-graph NPU path is separate from --device
      cpu|gpu; see docs/models/yolo/npu.md.

SPARSE MoE
  brain train [--steps N --batch-size B --block-size T --lr X --out F]
  brain generate --weights F [--prompt 1,2,3,4 --max-new N --temperature X --top-k K]
  brain eval     --weights F [--samples N]

FEDERATED MoE (train experts separately, then assemble)
  brain federated split    <base.safetensors> <out_dir>
  brain federated verify   <dir>
  brain federated merge     <dir> --out <full.safetensors>
  brain federated assemble  <base_dir> [overlay_dir ...] --out <full.safetensors>

BENCHMARK SUITE (architecture evaluation)
  brain bench [<name>] [--seed S]          # run all benchmarks, or one by name
      names: mqar                          # prints a benchmark|score|threshold|pass table
  brain bench scaling [--seed S]           # multi-size scaling-law sweep: trains the MQAR
                                           # task at several sizes, fits L(N)=E+A*N^-alpha,
                                           # prints the size|params|flops|loss table + alpha,R2
  brain bench eval --arch <name> [--seed S --out F --smoke]
                                           # run the WHOLE battery against one architecture,
                                           # aggregate per capability axis, write
                                           # results/<arch>-<seed>.json  (archs: gpt, gpt-small, gpt-wide, moe)
                                           # + prints a 'top tuning recommendations' footer (advisor)
  brain bench scale --arch <name> [--seed S --out F]
                                           # PREDICTIVE per-capability scaling: sweep model SIZE,
                                           # fit how each axis's score scales with params N, predict
                                           # score@2x/@4x, write results/scale-<arch>-<seed>.json
  brain bench advise <eval.json> [<scale.json>]
                                           # RANKED tuning recommendations: what to tune to improve
                                           # in the best capability direction (headroom x size-slope)
  brain bench compare <a.json> <b.json> ...# side-by-side leaderboard across results artifacts

EVENT/STDIO CONTROLLER
  brain run [--gpt <ckpt>] [--yolo <ckpt>] [--conf X] [--max-new N --temp X --top-k K --seed S]
      Event-driven HFSM controller: read JSONL events on stdin, emit JSONL events
      on stdout (text streaming + object detection). With no --gpt (or BRAIN_GPT),
      a fake echo model runs; with no --yolo (or BRAIN_YOLO), a fake detector runs,
      so the loop is usable without a trained checkpoint.
      Example: printf '{\"event\":\"user_text\",\"text\":\"hi\"}\\n' | brain run

PERFORMANCE BENCHMARKING (how fast, at what cost — see docs/performance/benchmarking.md)
  brain perf list                          # scenarios + the standard workload matrix
  brain perf run <latency|throughput|serve|sweep>
      [--target fake | qwen-synth:<L>x<D>x<H>[xV] | qwen:<weights>]
      [--workload interactive|chat|rag|rag_long|agent|decode_heavy|prefill_heavy|shared_prefix]
      [--concurrency N | --ladder 1,2,4,8,16,32] [--requests N --warmup N]
      [--best-of N --smoke --seed S --out F]
                                           # writes results/perf-<...>.json
  brain perf compare results/perf-*.json   # leaderboard; refuses to rank across
                                           # artifact units, excludes runs whose
                                           # correctness gate failed
      Report output artifacts/s + the latency curve, never total throughput alone;
      goodput (output meeting the SLO) is the comparison metric, not peak rate.

FLOP/OPS ACCOUNTING (docs/performance/flops.md)
  brain flops --model qwen|gpt|lfm [--weights F] [--batch B] [--block T]
              [--train] [--i8] [--stages N] [--run]
      OFFLINE per-kernel FLOP/int-OPS/bytes for the recorded forward (and
      backward with --train) — no execution. --run also executes one pass and
      prints the ONLINE counters (accumulated at dispatch; int8 kernels count
      integer OPS). --stages N reports per-stage = per-device numbers.

QWEN3 (dense decoder; paged continuous-batching serving)
  brain qwen import <hf_dir|safetensors> --out F
  brain qwen infer  --weights F --tokenizer T --prompt \"...\" [--max-new N --temp X --top-k K]
  brain qwen serve  --weights F --tokenizer T --prompt \"...\" [--prompt ...] [--max-new N
                    --block-size B --int8]        # paged KV + continuous batching
  brain qwen train|finetune|export|precompile|toolcall ...

GLM-5.2 (MLA + sigmoid noaux_tc MoE)
  brain glm <train|finetune|infer|eval|import|export> ...

LFM2.5-ENCODER (bidirectional conv/attention encoder, MLM head, 8k context)
  brain lfm import    --hf <dir> --out lfm.safetensors
  brain lfm fill-mask --weights F --tokenizer T --text \"… <|mask|> …\" [--topk K]
  brain lfm embed     --weights F --tokenizer T (--text \"…\" | --input FILE) [--seq T]
  brain lfm data      --input corpus.txt --tokenizer T --out data/lfm
  brain lfm finetune  --weights F --tokenizer T [--data D --steps N --batch B --seq T]
  brain lfm eval      --weights F --tokenizer T [--data D]     # pseudo-ppl + masked-acc
  brain npu lfm       --weights F --seq S --out model.onnx [--int8]   # OpenVINO export
  brain do lfm <fill_mask|embed> …   # capability surface; also served over D-Bus
  make perf/lfm                      # standalone concurrency benchmark (scheduler+residency)

QWEN3-TTS (Talker + MTP + neural codec; voice cloning)
  brain tts <import|clone|synth|design|serve|sim|finetune> ...

MONOCULAR DEPTH (ZipDepth)
  brain depth --image <P6.ppm> [...]       # single image
  brain depth --camera                     # realtime V4L2 webcam (Linux)
  brain depth <calib|train> ...

FORECASTING (chronos2 / kronos / fincast behind one seam)
  brain forecast <compare|serve|import|finetune> ...

CAPABILITIES (typed actions; one dispatch path for CLI + event API)
  brain caps                               # every model's action manifest
  brain do <model> <action> [--param v ...]

OTHER
  brain gradcheck                          # finite-difference backprop check (GPT)
  brain pid <train|rollout|profile> ...
  brain help
  brain --version

EXAMPLES
  brain data gen calculator --out data/calculator --n 100000
  brain gpt train data/calculator --out out/gpt.safetensors --steps 2000 --mask =
  brain gpt eval  --weights out/gpt.safetensors --data data/calculator
  brain gpt gen   --weights out/gpt.safetensors --data data/calculator --prompt \"12+7=\" --max-new 8
  brain train --steps 2000 --out moe.safetensors
  brain generate --weights moe.safetensors --prompt 1,2,3,4 --max-new 64
  brain federated split moe.safetensors out/shards && brain federated verify out/shards
  brain gradcheck

Or drive everything via the Makefile:  make data/calculator train/gpt/calculator eval/gpt/calculator
";

/// Extract a global `--device cpu|gpu` flag from anywhere in the args and select
/// the compute backend, returning the remaining args. `BRAIN_DEVICE=cpu` does the
/// same without a flag. Both backends are compiled into every build; this only
/// chooses which one each model instantiates at runtime.
fn select_backend(argv: Vec<String>) -> Vec<String> {
    // `brain npu …` subcommands parse their OWN `--device` (the OpenVINO target
    // device, e.g. NPU/CPU/GPU as OpenVINO names them); don't consume it here.
    if argv.get(1).map(|s| s == "npu").unwrap_or(false) {
        return argv;
    }
    let mut out = Vec::with_capacity(argv.len());
    let mut spec_text: Option<String> = None;
    let mut i = 0;
    while i < argv.len() {
        if argv[i] == "--device" {
            match argv.get(i + 1) {
                Some(v) => spec_text = Some(v.clone()),
                None => {
                    eprintln!(
                        "brain: --device needs a value (cpu | gpu | npu | gpu0 | cpu0-7 | gpu,cpu)"
                    );
                    std::process::exit(2);
                }
            }
            i += 2;
        } else {
            out.push(argv[i].clone());
            i += 1;
        }
    }

    // No flag and no BRAIN_DEVICE => the empty spec, which resolves to every
    // device on the machine (GPUs + CPU + NPU), scheduled together.
    let text = spec_text
        .or_else(|| std::env::var("BRAIN_DEVICE").ok())
        .unwrap_or_default();
    let spec = match gpu_core::DeviceSpec::parse(&text) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("brain: --device: {e}");
            std::process::exit(2);
        }
    };
    let set = match spec.resolve(&gpu_core::Inventory::probe()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("brain: --device: {e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = set.apply() {
        eprintln!("brain: --device: {e}");
        std::process::exit(2);
    }
    // The NPU is a whole-graph (OpenVINO) path rather than a gpu_core backend, so
    // it is a separate flag the NPU-capable subcommands consult.
    NPU_REQUESTED.store(set.npu_enabled() && set.explicit, Ordering::Relaxed);
    // Deliberately do NOT rewrite BRAIN_DEVICE: it is recorded verbatim in perf
    // artifacts, and `gpu_core`'s own fallback parses it as a bare backend name,
    // which an indexed spec like "gpu0" is not. The resolved set is published
    // in-process instead.
    COMPUTE.set(set).ok();
    out
}

/// The resolved compute set for this process, available to every subcommand.
static COMPUTE: std::sync::OnceLock<gpu_core::ComputeSet> = std::sync::OnceLock::new();

/// What `--device` resolved to. `None` only before `select_backend` has run.
pub fn compute_set() -> Option<&'static gpu_core::ComputeSet> {
    COMPUTE.get()
}

/// `brain bench [<name>] [--seed S]` — run the architecture-evaluation suite.
/// With no name, runs every registered benchmark and prints one comparison
/// table; with a name, runs just that benchmark. Exits non-zero on any failure.
fn run_bench(args: &[String]) {
    // The turn-key architecture-eval harness and the leaderboard compare are
    // sub-subcommands with their own flag grammar; route them before the
    // single-benchmark flag parse below.
    match args.first().map(|s| s.as_str()) {
        Some("eval") => return run_bench_eval(&args[1..]),
        Some("compare") => return run_bench_compare(&args[1..]),
        Some("scale") => return run_bench_scale(&args[1..]),
        Some("advise") => return run_bench_advise(&args[1..]),
        _ => {}
    }

    let mut name: Option<&str> = None;
    let mut seed = 1337u64;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                seed = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(seed);
            }
            other if !other.starts_with("--") => name = Some(other),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }

    // `scaling` is a separate entry point (a multi-size sweep + power-law fit),
    // not a registry Benchmark — route it here before the registry lookup.
    if name == Some("scaling") {
        run_scaling(seed);
        return;
    }

    let result = match name {
        Some(n) => bench::run_one(n, seed),
        None => bench::run_all(seed),
    };
    match result {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("brain bench: one or more benchmarks FAILED their threshold");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("brain bench: {e}");
            std::process::exit(2);
        }
    }
}

/// `brain bench scaling [--seed S]` — the multi-scale scaling-law sweep: train
/// the MQAR task at several model sizes, then fit `L(N) ≈ E + A·N^(−α)` and print
/// the per-size table + fitted exponent. The foundation the later per-capability
/// predictive-scaling work builds on.
fn run_scaling(seed: u64) {
    let dir = std::env::temp_dir().join(format!("brain_scaling_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let sweep = bench::scaling::Sweep {
        seed,
        ..Default::default()
    };
    match bench::scaling::run(&sweep, &dir) {
        Ok(result) => {
            result.print();
            let _ = std::fs::remove_dir_all(&dir);
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&dir);
            eprintln!("brain bench scaling: {e}");
            std::process::exit(2);
        }
    }
}

/// `brain bench eval --arch <name> [--seed S] [--out <path>] [--smoke]` — the
/// turn-key architecture-eval harness: run EVERY registered benchmark against the
/// named architecture, aggregate per capability axis, write a results artifact
/// under `results/<arch>-<seed>.json` (or `--out`), and print the table + axes.
fn run_bench_eval(args: &[String]) {
    let mut arch = "gpt".to_string();
    let mut seed = 1337u64;
    let mut out: Option<String> = None;
    let mut smoke = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--arch" => {
                i += 1;
                arch = args.get(i).cloned().unwrap_or(arch);
            }
            "--seed" => {
                i += 1;
                seed = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(seed);
            }
            "--out" => {
                i += 1;
                out = args.get(i).cloned();
            }
            "--smoke" => smoke = true,
            other => eprintln!("brain bench eval: ignoring unknown flag {other:?}"),
        }
        i += 1;
    }

    let report = match bench::eval::run(&arch, seed, smoke) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("brain bench eval: {e}");
            std::process::exit(2);
        }
    };
    bench::eval::print_report(&report);

    let path = out
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| bench::eval::default_out_path(&arch, seed));
    if let Err(e) = bench::eval::write_artifact(&report, &path) {
        eprintln!("brain bench eval: writing {}: {e}", path.display());
        std::process::exit(2);
    }
    println!("wrote results artifact: {}", path.display());

    // The eval output itself carries the tuning breakdown the user asked for: run
    // the advisor on the just-produced artifact (enriched with the capscale
    // artifact for this arch/seed if one happens to exist) and print a short
    // ranked footer of the top levers.
    let eval_json = report.to_json();
    let capscale = bench::advisor::try_load_capscale(&arch, seed);
    let advice = bench::advisor::advise(&eval_json, capscale.as_ref());
    bench::advisor::print_footer(&advice, 3);

    // Exit non-zero if a gating benchmark failed (informational ones excluded),
    // matching `brain bench`'s pass/fail contract.
    if report.gating_passed < report.gating_total {
        eprintln!(
            "brain bench eval: {}/{} gating benchmarks passed",
            report.gating_passed, report.gating_total
        );
        std::process::exit(1);
    }
}

/// `brain bench scale --arch <name> [--seed S] [--out <path>]` — the
/// per-capability predictive-scaling sweep: train+score one representative
/// benchmark per capability axis across a small SIZE grid, fit how each axis's
/// score scales with params N, extrapolate the predicted score at 2×/4× the
/// largest N, print the per-axis curves, and write
/// `results/scale-<arch>-<seed>.json`.
fn run_bench_scale(args: &[String]) {
    let mut arch = "gpt".to_string();
    let mut seed = 1337u64;
    let mut out: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--arch" => {
                i += 1;
                arch = args.get(i).cloned().unwrap_or(arch);
            }
            "--seed" => {
                i += 1;
                seed = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(seed);
            }
            "--out" => {
                i += 1;
                out = args.get(i).cloned();
            }
            other => eprintln!("brain bench scale: ignoring unknown flag {other:?}"),
        }
        i += 1;
    }

    let cfg = bench::capscale::CapScaleConfig {
        seed,
        ..Default::default()
    };
    let report = match bench::capscale::run(&arch, &cfg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("brain bench scale: {e}");
            std::process::exit(2);
        }
    };
    bench::capscale::print_report(&report);

    let path = out
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| bench::capscale::default_out_path(&arch, seed));
    if let Err(e) = bench::capscale::write_artifact(&report, &path) {
        eprintln!("brain bench scale: writing {}: {e}", path.display());
        std::process::exit(2);
    }
    println!("wrote scaling artifact: {}", path.display());
}

/// `brain bench advise <eval.json> [<scale.json>]` — load an eval artifact (and,
/// optionally, a per-capability scaling artifact) and print a RANKED, concrete set
/// of tuning recommendations for what to tune to improve in the best capability
/// direction.
fn run_bench_advise(args: &[String]) {
    let paths: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let Some(eval_path) = paths.first() else {
        eprintln!("brain bench advise: usage: brain bench advise <eval.json> [<scale.json>]");
        std::process::exit(2);
    };
    let eval = match bench::advisor::load_eval(std::path::Path::new(eval_path.as_str())) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("brain bench advise: reading {eval_path}: {e}");
            std::process::exit(2);
        }
    };
    let capscale = paths.get(1).and_then(|p| {
        match bench::capscale::load_artifact(std::path::Path::new(p.as_str())) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("brain bench advise: reading {p}: {e} (ignoring scaling artifact)");
                None
            }
        }
    });
    let advice = bench::advisor::advise(&eval, capscale.as_ref());
    bench::advisor::print_advice(&advice);
}

/// `brain bench compare <results.json> <results.json> ...` — load ≥2 artifacts
/// and print a side-by-side leaderboard (overall pass-rate + per-axis +
/// per-benchmark scores, columns = architectures).
fn run_bench_compare(args: &[String]) {
    let paths: Vec<std::path::PathBuf> = args
        .iter()
        .filter(|a| !a.starts_with("--"))
        .map(std::path::PathBuf::from)
        .collect();
    if let Err(e) = bench::eval::compare(&paths) {
        eprintln!("brain bench compare: {e}");
        std::process::exit(2);
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if matches!(argv.get(1).map(String::as_str), Some("--version" | "-V")) {
        println!("brain {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    let argv = select_backend(argv);
    match argv.get(1).map(|s| s.as_str()) {
        Some("data") => data_cli::run_data(&argv[2..]),
        Some("devices") => devices_cli::run_devices(&argv[2..]),
        Some("fetch") => fetch_cli::run_fetch(&argv[2..]),
        Some("gpt") => gpt_cli::run_gpt(&argv[2..]),
        Some("qwen") => qwen_cli::run_qwen(&argv[2..]),
        Some("glm") => glm_cli::run_glm(&argv[2..]),
        Some("lfm") => lfm_cli::run_lfm(&argv[2..]),
        Some("tts") => tts_cli::run_tts(&argv[2..]),
        Some("wm") => wm_cli::run_wm(&argv[2..]),
        Some("yolo") => yolo_cli::run_yolo(&argv[2..]),
        Some("depth") => depth_cli::run_depth(&argv[2..]),
        Some("flux2") => flux2_cli::run_flux2(&argv[2..]),
        Some("mirror") => mirror_cli::run_mirror(&argv[2..]),
        Some("splat") => splat_cli::run_splat(&argv[2..]),
        Some("npu") => npu_cli::run_npu(&argv[2..]),
        Some("federated") => federated_cli::run_federated(&argv[2..]),
        Some("flops") => flops_cli::run_flops(&argv[2..]),
        Some("gradcheck") => {
            let report = gradcheck::check_gpt(1);
            report.print();
            let fails = report.failures(4e-3, 8e-2);
            if fails.is_empty() {
                println!("gradcheck OK ({} tensors)", report.checks.len());
            } else {
                eprintln!("gradcheck FAILED for {} tensors", fails.len());
                std::process::exit(1);
            }
        }
        Some("bench") => run_bench(&argv[2..]),
        Some("perf") => perf_cli::run_perf(&argv[2..]),
        Some("forecast") => forecast_cli::run_forecast(&argv[2..]),
        Some("caps") | Some("capabilities") => std::process::exit(caps_cli::run_caps(&argv[2..])),
        Some("do") => std::process::exit(caps_cli::run_do(&argv[2..])),
        Some("run") | Some("serve") => run_cli::run_serve(&argv[2..]),
        Some("pid") => pid_cli::run_pid(&argv[2..]),
        Some("train") => moe::run_train(&argv[2..]),
        Some("eval") => moe::run_eval(&argv[2..]),
        Some("generate") => moe::run_generate(),
        Some("help") | Some("-h") | Some("--help") | None => print!("{HELP}"),
        Some(other) => {
            eprintln!("brain: unknown command '{other}'\n");
            print!("{HELP}");
            std::process::exit(2);
        }
    }
}
