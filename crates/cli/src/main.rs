// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `brain` native CLI — one binary over every model in the workspace.
//! The model is chosen by the subcommand (no global "model type" flag).
//!
//!   * `gpt`        — dense GPT decoder baseline (nanogpt parity).
//!   * `generate`/`train`/`eval`/`validate` — the sparse-MoE Transformer.
//!   * `federated`  — sharded-MoE shard split/assemble.
//!   * `data`       — dataset generation; `gradcheck` — backprop correctness gate.
//!   * `pid`        — event/effect control Transformer (the WebGPU demo).
//!
//! Run `brain help` for the full usage with examples.

mod data_cli;
mod federated_cli;
mod gpt_cli;
mod image_io;
mod npu_cli;
mod pid_cli;
mod qwen_cli;
mod run_cli;
mod tts_cli;
mod tts_serve;
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

Add --device cpu (or set BRAIN_DEVICE=cpu) to run any command on the native CPU
backend (WGSL kernels JIT-compiled to native code across all cores, no GPU);
--device gpu (the default) uses wgpu. Both are built into the same binary.

DATA
  brain data gen <name> [--out DIR --n N --seed S]
      names: calculator | reverser | wordcalc | timeseries | shakespeare_char | gpt

GPT (dense baseline)
  brain gpt train <data_dir> [--out F --steps N --batch B --block T
                              --layers L --d-model D --heads H --lr X --mask = --align]
  brain gpt eval  --weights F --data <dir> [--batches N --samples M]
  brain gpt gen   --weights F [--data <dir>] [--prompt \"...\" --max-new N --temp X --top-k K]
                              (vocab is read from the checkpoint; --data only for old ones)

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
      cpu|gpu; see docs/yolo/NPU.md.

SPARSE MoE
  brain train [--steps N --batch-size B --block-size T --lr X --out F]
  brain generate --weights F [--prompt 1,2,3,4 --max-new N --temperature X --top-k K]
  brain eval     --weights F [--samples N]
  brain validate [ref.bin]                # gradient parity gate (if a ref file exists)

FEDERATED MoE (train experts separately, then assemble)
  brain federated split    <base.weights> <out_dir>
  brain federated verify   <dir>
  brain federated merge     <dir> --out <full.weights>
  brain federated assemble  <base_dir> [overlay_dir ...] --out <full.weights>

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

OTHER
  brain gradcheck                          # finite-difference backprop check (GPT)
  brain pid <validate|stream|train|rollout|profile> ...
  brain help

EXAMPLES
  brain data gen calculator --out data/calculator --n 100000
  brain gpt train data/calculator --out out/gpt.weights --steps 2000 --mask =
  brain gpt eval  --weights out/gpt.weights --data data/calculator
  brain gpt gen   --weights out/gpt.weights --data data/calculator --prompt \"12+7=\" --max-new 8
  brain train --steps 2000 --out moe.weights
  brain generate --weights moe.weights --prompt 1,2,3,4 --max-new 64
  brain federated split moe.weights out/shards && brain federated verify out/shards
  brain gradcheck

Or drive everything via the Makefile:  make data/calculator train/gpt/calculator eval/gpt/calculator
";

/// Extract a global `--device cpu|gpu` flag from anywhere in the args and select
/// the compute backend, returning the remaining args. `BRAIN_DEVICE=cpu` does the
/// same without a flag. Both backends are compiled into every build; this only
/// chooses which one each model instantiates at runtime.
fn select_backend(argv: Vec<String>) -> Vec<String> {
    // `brain npu …` subcommands parse their OWN `--device` (the OpenVINO target
    // device); don't consume it here.
    if argv.get(1).map(|s| s == "npu").unwrap_or(false) {
        return argv;
    }
    let mut out = Vec::with_capacity(argv.len());
    let mut i = 0;
    while i < argv.len() {
        if argv[i] == "--device" {
            match argv.get(i + 1).map(|s| s.as_str()) {
                Some("cpu") => gpu_core::set_default_backend(gpu_core::Backend::Cpu),
                Some("gpu") | Some("wgpu") => {
                    gpu_core::set_default_backend(gpu_core::Backend::Wgpu)
                }
                // Native Vulkan compute (ash + naga). Falls back to wgpu if no ICD.
                Some("vulkan") => gpu_core::set_default_backend(gpu_core::Backend::Vulkan),
                // The NPU is a whole-graph (OpenVINO) path, not a gpu_core
                // backend: record the request and leave the host backend at its
                // default (the NPU path does its own compute via OpenVINO + a
                // pure-Rust decode). Consumed by `brain yolo detect`.
                Some("npu") => NPU_REQUESTED.store(true, Ordering::Relaxed),
                other => {
                    eprintln!("brain: --device expects cpu|gpu|vulkan|npu (got {other:?})");
                    std::process::exit(2);
                }
            }
            i += 2;
        } else {
            out.push(argv[i].clone());
            i += 1;
        }
    }
    out
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
    let sweep = bench::scaling::Sweep { seed, ..Default::default() };
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

    let cfg = bench::capscale::CapScaleConfig { seed, ..Default::default() };
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
    let argv = select_backend(std::env::args().collect());
    match argv.get(1).map(|s| s.as_str()) {
        Some("data") => data_cli::run_data(&argv[2..]),
        Some("gpt") => gpt_cli::run_gpt(&argv[2..]),
        Some("qwen") => qwen_cli::run_qwen(&argv[2..]),
        Some("tts") => tts_cli::run_tts(&argv[2..]),
        Some("yolo") => yolo_cli::run_yolo(&argv[2..]),
        Some("npu") => npu_cli::run_npu(&argv[2..]),
        Some("federated") => federated_cli::run_federated(&argv[2..]),
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
        Some("run") | Some("serve") => run_cli::run_serve(&argv[2..]),
        Some("pid") => pid_cli::run_pid(&argv[2..]),
        Some("validate") => {
            let path = argv
                .get(2)
                .map(|s| s.as_str())
                .unwrap_or("scratchpad/weights/train_ref.bin");
            moe::train::validate(path);
        }
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
