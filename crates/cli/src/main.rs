// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `brain` native CLI - one binary over every model in the workspace.
//!
//! `brain <verb> <architecture>` and `brain <architecture> <verb>` are the
//! same command ([`resolve::dispatch`]) - the architecture is chosen by
//! whichever token names one of `brain_arch::ARCHS`, in either position.
//! Infrastructure commands (`data`, `devices`, `npu`, `federated`, `bench`,
//! `perf`, `forecast`, `caps`, `serve`, `gradcheck`, `flops`) are matched
//! first and never go through that resolver.
//!
//! Run `brain help` for the full usage with examples.

mod args;
mod caps_cli;
mod catalog;
mod data_cli;
mod depth_cli;
mod devices_cli;
mod federated_cli;
mod flops_cli;
mod flux2_cli;
mod forecast_cli;
mod glm_cli;
mod gguf_import;
mod gpt_cli;
mod image_io;
mod imageops;
mod lfm_cli;
mod mirror_cli;
mod model_dir;
mod npu_cli;
mod omni_cli;
mod perf_cli;
mod perf_engine;
mod pid_cli;
mod qwen35moe_cli;
mod qwen_cli;
mod resident;
mod resident_asr;
mod resident_deepseekocr;
mod resident_depth;
mod resident_flux2;
mod resident_forecast;
mod resident_lfm;
mod resident_llm;
mod resident_mock;
mod resident_omni;
mod resident_qwen35moe;
mod resident_tts;
mod resident_upscale;
mod resident_clip;
mod resident_scrfd;
mod resident_arcface;
mod resident_sam2;
mod resident_restore;
mod resolve;
mod run_cli;
mod splat_cli;
mod supply;
mod tts_cli;
mod tts_serve;
mod wm_cli;
mod yolo_cli;

/// Whether `--device`/`BRAIN_DEVICE` EXPLICITLY named the NPU. The NPU is a
/// whole-graph (OpenVINO) path, not a `gpu_core` backend, so the commands that
/// support it (`brain yolo detect`, `glm infer`, `wm play`, `qwen infer`,
/// `tts clone`/`synth`/`design`) consult this instead of a `gpu_core` backend
/// check. Reads the SAME resolved `ComputeSet` every other caller reads
/// (`gpu_core::ambient_compute_set()` - published by `select_backend` below
/// for the CLI path, lazily resolved from `BRAIN_DEVICE` otherwise) rather
/// than tracking its own process-global sidecar: `explicit` excludes the
/// ambient "everything, including an NPU that happens to be present" case, so
/// an omitted `--device` never silently triggers the whole-graph NPU path.
pub(crate) fn npu_explicit() -> bool {
    let set = gpu_core::ambient_compute_set();
    set.npu_enabled() && set.explicit
}

const HELP: &str = "\
brain - train and evaluate neural nets from scratch on the GPU (Rust + WGSL).

USAGE: brain <verb> <architecture> [options]   OR   brain <architecture> <verb> [options]
Both orders are the SAME command - `brain train gpt2 ...` and `brain gpt2
train ...` dispatch identically. `brain caps` lists every architecture brain
knows about, along with its actions (its \"verbs\"). An architecture with its
own dedicated flags (gpt2, qwen3, qwen35moe, glmdsa, lfm2, qwen3tts, yolov8,
zipdepth, flux2, worldmirror2, splat, qwen3omnimoe, diamond, toypid, toymoe)
is documented below; every other one (`brain caps` shows the full list -
s3dit, fastvlm, qwen3vl, sam2, scrfd, arcface, vqgan, codeformer, rrdbnet,
clip, deepseek2ocr, nemotronasr, qwen3asr, chronos2, fincast, kronos, ...) is
reached the same way, its verb being the exact action name `brain caps <id>`
prints (e.g. `brain scrfd detect --in image=photo.ppm`).

--device selects WHICH COMPUTE IS SCHEDULABLE. Omit it and brain uses every
device on the machine - all GPUs, the CPU, and an NPU if present - scheduling
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

GPT2 (dense baseline)
  brain gpt2 train <data_dir> [--out F --steps N --batch B --block T
                              --layers L --d-model D --heads H --lr X --mask = --align]
  brain gpt2 eval  --weights F --data <dir> [--batches N --samples M]
  brain gpt2 infer --weights F [--data <dir>] [--prompt \"...\" --max-new N --temp X --top-k K]
                              (vocab is read from the checkpoint; --data only for old ones)

FLUX.2 Klein (text-to-image + image editing; 4-step distilled flow matching)
  brain flux2 generate --prompt \"...\" --out out.ppm [--width W --height H]
      [--steps N --seed S --variant klein-4b|klein-9b|base-4b|base-9b]
      [--guidance G]              # CFG, base variants only
      [--ref in.ppm]...           # reference images => editing mode
      weights via env: BRAIN_FLUX2_DIT, BRAIN_FLUX2_VAE, BRAIN_FLUX2_TE, BRAIN_FLUX2_TOKENIZER
      served generically as model `flux2-klein`: brain caps flux2-klein,
      brain flux2 text2image|edit|lora_train, and D-Bus (examples/imagegen);
      9B variants need BRAIN_FLUX2_ALLOW_NC=1 (FLUX Non-Commercial license)

World models (playable action-conditioned video models; docs/models/world-models/)
  brain diamond play  --model fake|diamond [--weights F --device cpu|gpu|npu --onnx M]   # SDL window
  brain diamond play  --model fake --headless --frames N [--actions FILE | --action-seq 1,2,0]
                 [--dump-ppm DIR] [--hashes]        # deterministic rollout + fnv1a hashes (CI)
  brain diamond bench --model fake [--frames N]          # ms/frame + fps

WorldMirror-2 (multi-view images → 3D Gaussian Splatting scene; docs/models/mirror/)
  brain worldmirror2 import <model.safetensors|hf_dir> --out mirror.safetensors
      One-time conversion of the reference HY-WorldMirror-2.0 checkpoint (strict
      1:1, every tensor verified).
  brain worldmirror2 infer --weights F --images <dir|a.ppm,b.ppm,…> [--out DIR]
      [--ply scene.ply] [--maps] [--min-opacity X] [--max-depth X] [--prune VOXEL]
      Images → navigable 3DGS scene (scene.ply + cameras.json + depth/normal
      maps). Any aspect ratio; --prune 0.002 voxel-merges multi-view duplicates.
  brain worldmirror2 demo  --weights F --images <…> [--width N --height N --fov D]
      infer + interactive fly-through of the reconstructed world.

3D GAUSSIAN SPLATTING (scene viewer/renderer; crates/splat)
  brain splat info   <scene.ply>
  brain splat render <scene.ply> --out img.ppm [--width N --height N]
        [--eye x,y,z --target x,y,z --up x,y,z --fov D] [--depth] [--bg r,g,b]
  brain splat view   <scene.ply> [--width N --height N --fov D --bg r,g,b]
        Interactive fly-through: WASD move, Space/C up/down, Shift sprint,
        m mouse-look, arrows look, [ ] quality, v depth view, p screenshot,
        Enter reset, Esc quit.

YOLOv8 (from-scratch anchor-free object detector)
  brain yolov8 train <data_dir> --out F [--steps N --batch B --lr X --nc C
                                       --input S --seed S]
  brain yolov8 eval  --weights F --data <dir> [--conf X --iou X]   # mAP/precision/recall
  brain yolov8 detect --weights F --image <P6.ppm | dataset_dir> [--conf X --iou X]
                                                                # prints [x1,y1,x2,y2,conf,class] JSON lines
  brain yolov8 fine-tune <data_dir> --weights <pretrained> --out F [--freeze-backbone ...]
      Trains the tiny YOLOv8 graph on a `data gen detect` dataset (CPU backend).
  brain yolov8 detect --weights F --image <...> --device npu     # run on the Intel NPU

INTEL NPU (OpenVINO: quantize + compile YOLO to a real NPU graph)
  brain npu export   --weights F --out model.onnx [--input S --opset N]    # fp32 ONNX
  brain npu quantize --weights F --calib <dir> --out model.int8.onnx [--input S --num-calib N]
  brain npu check    --onnx M [--ov-device NPU]              # ONNX op histogram + compile/coverage
  brain npu run      --onnx M --image <P6|dir> [--ov-device NPU --conf X --iou X --nc C --reg-max R]
  brain npu bench    --onnx M [--input S --ov-device NPU --iters N --hint latency|throughput]
  brain npu sim      --weights F --data <dir> [--calib <dir>]   # fp32 vs INT8 mAP, no NPU
      export/quantize/sim are pure Rust (any machine); run/bench/check-compile need
      OpenVINO + an Intel NPU. --ov-device selects the OpenVINO target device
      (NPU|CPU|GPU|AUTO) - unrelated to brain's own --device cpu|gpu grammar
      (--device is still accepted here as a deprecated alias for --ov-device).

TOY ARCHITECTURES (brain's own, no upstream reference - toy tasks, not served)
  brain toymoe train [--steps N --batch-size B --block-size T --lr X --out F]
  brain toymoe infer --weights F [--prompt 1,2,3,4 --max-new N --temperature X --top-k K]
  brain toymoe eval  --weights F [--samples N]
  brain toypid <train|rollout|profile> ...   # the WebGPU browser demo's model

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
                                           # results/<arch>-<seed>.json  (archs: gpt2, gpt2-small, gpt2-wide, toymoe)
                                           # + prints a 'top tuning recommendations' footer (advisor)
  brain bench scale --arch <name> [--seed S --out F]
                                           # PREDICTIVE per-capability scaling: sweep model SIZE,
                                           # fit how each axis's score scales with params N, predict
                                           # score@2x/@4x, write results/scale-<arch>-<seed>.json
  brain bench advise <eval.json> [<scale.json>]
                                           # RANKED tuning recommendations: what to tune to improve
                                           # in the best capability direction (headroom x size-slope)
  brain bench compare <a.json> <b.json> ...# side-by-side leaderboard across results artifacts

HTTP INFERENCE APIS (brain as an OpenAI / Anthropic / OpenRouter backend)
  brain serve [--openai [PORT]] [--anthropic [PORT]] [--openrouter [PORT]] [--dbus]
              [--models-dir DIR] [--api-keys-out FILE] [--ready-file PATH]
      Serves the shared executor over localhost HTTP: one port + one freshly
      generated key per dialect (defaults: openai 8788, anthropic 8787,
      openrouter 8789). OpenAI/OpenRouter base URL: http://127.0.0.1:PORT
      *or* http://127.0.0.1:PORT/v1 -- both work.  Full reference:
      brain serve --help

EVENT/STDIO CONTROLLER
  brain serve --stdio [--gpt <ckpt>] [--yolo <ckpt>] [--conf X] [--max-new N --temp X --top-k K --seed S]
      Event-driven HFSM controller: read JSONL events on stdin, emit JSONL events
      on stdout (text streaming + object detection). With no --gpt (or BRAIN_GPT2),
      a fake echo model runs; with no --yolo (or BRAIN_YOLOV8), a fake detector runs,
      so the loop is usable without a trained checkpoint.
      Example: printf '{\"event\":\"user_text\",\"text\":\"hi\"}\\n' | brain serve --stdio

PERFORMANCE BENCHMARKING (how fast, at what cost)
  brain perf list                          # scenarios + the standard workload matrix
  brain perf run <latency|throughput|serve|sweep>
      --target qwen-synth:<L>x<D>x<H>[xV] | qwen:<weights> | ...   (required, see `brain perf list`)
      [--workload interactive|chat|rag|rag_long|agent|decode_heavy|prefill_heavy|shared_prefix]
      [--concurrency N | --ladder 1,2,4,8,16,32] [--requests N --warmup N]
      [--best-of N --smoke --seed S --out F]
                                           # writes results/perf-<...>.json
  brain perf compare results/perf-*.json   # leaderboard; refuses to rank across
                                           # artifact units, excludes runs whose
                                           # correctness gate failed
      Report output artifacts/s + the latency curve, never total throughput alone;
      goodput (output meeting the SLO) is the comparison metric, not peak rate.

FLOP/OPS ACCOUNTING
  brain flops --model qwen|gpt|lfm [--weights F] [--batch B] [--block T]
              [--train] [--i8] [--stages N] [--run]
      OFFLINE per-kernel FLOP/int-OPS/bytes for the recorded forward (and
      backward with --train) - no execution. --run also executes one pass and
      prints the ONLINE counters (accumulated at dispatch; int8 kernels count
      integer OPS). --stages N reports per-stage = per-device numbers.

QWEN3 (dense decoder; paged continuous-batching serving)
  brain qwen3 import <hf_dir|safetensors> --out F
  brain qwen3 infer  --weights F --tokenizer T --prompt \"...\" [--max-new N --temp X --top-k K]
  brain qwen3 serve  --weights F --tokenizer T --prompt \"...\" [--prompt ...] [--max-new N
                    --block-size B --kv-fp32]     # paged KV + continuous batching;
                                                  # int8 KV on by default, --kv-fp32 opts out
  brain qwen3 train|finetune|export|precompile|toolcall|eval|calib ...

GLM-5.2 (MLA + sigmoid noaux_tc MoE)
  brain glmdsa <train|finetune|infer|eval|import|export> ...

QWEN3-OMNI (text/audio/image/video in, text + speech out)
  brain qwen3omnimoe import --hf <dir> --out Qwen3-Omni-30B-A3B-Instruct-W8A16.safetensors [--id VENDOR/REPO]
      # brain-native W8A16 checkpoint for the GPU-resident sharded Thinker
      # (serve it with BRAIN_QWEN3OMNIMOE_INT8_CHECKPOINT=<out>)

LFM2.5-ENCODER (bidirectional conv/attention encoder, MLM head, 8k context)
  brain lfm2 import    --hf <dir> --out lfm.safetensors
  brain lfm2 fill-mask --weights F --tokenizer T --text \"… <|mask|> …\" [--topk K]
  brain lfm2 embed     --weights F --tokenizer T (--text \"…\" | --input FILE) [--seq T]
  brain lfm2 data      --input corpus.txt --tokenizer T --out data/lfm
  brain lfm2 finetune  --weights F --tokenizer T [--data D --steps N --batch B --seq T]
  brain lfm2 eval      --weights F --tokenizer T [--data D]     # pseudo-ppl + masked-acc
  brain npu lfm        --weights F --seq S --out model.onnx [--int8]   # OpenVINO export
  make perf/lfm                      # standalone concurrency benchmark (scheduler+residency)

QWEN3-TTS (Talker + MTP + neural codec; voice cloning)
  brain qwen3tts <import|clone|synth|design|serve|sim|finetune> ...

MONOCULAR DEPTH (ZipDepth)
  brain zipdepth --image <P6.ppm> [...]       # single image
  brain zipdepth --camera                     # realtime V4L2 webcam (Linux)
  brain zipdepth <calib|train> ...

FORECASTING (chronos2 / kronos / fincast behind one seam)
  brain forecast <compare|serve|import|finetune> ...
      chronos2/fincast/kronos are also individually listed by `brain caps`,
      but have no direct per-action CLI path yet - serve them instead.

CAPABILITIES (typed actions; one dispatch path for the CLI, D-Bus and HTTP)
  brain caps                               # every architecture's action manifest
  brain <architecture> <action> [--param v ...] [--in name=path ...] [--out name=path ...]
      Any architecture without its own dedicated flags above (`brain caps`
      lists them all) dispatches this way - the action name IS the verb.
      Example: brain scrfd detect --in image=photo.ppm --json

GGUF IMPORT (one-time conversion; dispatches on general.architecture)
  brain import FILE [--out PATH] [--id VENDOR/REPO]
      Convert a GGUF checkpoint to brain-native safetensors, choosing the
      importer by the file's own `general.architecture`. Defaults to a sibling
      <stem>.brain.safetensors, which the model-dir scan then serves on its own.
      brain import --list                # registered architectures

OTHER
  brain gradcheck                          # finite-difference backprop check (GPT2)
  brain help
  brain --version

EXAMPLES
  brain data gen calculator --out data/calculator --n 100000
  brain gpt2 train data/calculator --out out/gpt2.safetensors --steps 2000 --mask =
  brain gpt2 eval  --weights out/gpt2.safetensors --data data/calculator
  brain gpt2 infer --weights out/gpt2.safetensors --data data/calculator --prompt \"12+7=\" --max-new 8
  brain toymoe train --steps 2000 --out moe.safetensors
  brain toymoe infer --weights moe.safetensors --prompt 1,2,3,4 --max-new 64
  brain federated split moe.safetensors out/shards && brain federated verify out/shards
  brain gradcheck

Or drive everything via the Makefile:  make data/calculator train/gpt/calculator eval/gpt/calculator
";

/// Extract a global `--device cpu|gpu` flag from anywhere in the args and select
/// the compute backend, returning the remaining args. `BRAIN_DEVICE=cpu` does the
/// same without a flag. Both backends are compiled into every build; this only
/// chooses which one each model instantiates at runtime.
fn select_backend(argv: Vec<String>) -> Vec<String> {
    let mut argv = argv;
    // `brain npu …` subcommands used to have their OWN `--device` (the
    // OpenVINO target device, e.g. NPU/CPU/GPU as OpenVINO names them) -
    // a different grammar under the same flag name as brain's own `--device`.
    // That flag is now `--ov-device`; translate the deprecated `--device`
    // alias here (with a one-line note) before the generic loop below runs,
    // so an old invocation still reaches `npu_cli`'s OpenVINO parsing instead
    // of being silently swallowed as a brain compute-device request - or,
    // worse, misinterpreted as one for tokens that happen to overlap the
    // grammar (`--device cpu`/`gpu`/`npu` all parse as valid brain tokens,
    // just meaning something else entirely).
    if argv.get(1).map(|s| s == "npu").unwrap_or(false) {
        let mut warned = false;
        for a in argv.iter_mut() {
            if a == "--device" {
                if !warned {
                    eprintln!(
                        "brain npu: --device is deprecated for the OpenVINO target device; use --ov-device instead"
                    );
                    warned = true;
                }
                *a = "--ov-device".to_string();
            }
        }
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

    // An explicit `--device` flag takes precedence over `BRAIN_DEVICE` and is
    // resolved directly (still hard-exiting on a bad value - the user just
    // typed it). With no flag, `BRAIN_DEVICE` - if any - is resolved by
    // `gpu_core::ambient_compute_set()`, the ONE place `BRAIN_DEVICE` is read
    // (shared with every non-CLI caller: test binaries, library callers with
    // no CLI in the loop). `apply()` still runs here either way for the
    // CLI-only half `ambient_compute_set()` deliberately skips (rayon pool
    // sizing / CPU affinity - see `ComputeSet::apply`'s doc).
    // The resolved set is only needed for its side effects here (`apply()`,
    // `publish_compute_set`) - callers read it back via
    // `gpu_core::ambient_compute_set()`/`compute_set()`, not this binding.
    let _set = match spec_text {
        Some(text) => {
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
            // Publish explicitly: `ambient_compute_set()` was never called
            // above (an explicit --device skips its BRAIN_DEVICE read), so
            // its OnceLock still needs this process's actual resolution.
            gpu_core::publish_compute_set(set.clone());
            set
        }
        None => {
            let set = gpu_core::ambient_compute_set();
            if let Err(e) = set.apply() {
                eprintln!("brain: --device: {e}");
                std::process::exit(2);
            }
            set.clone()
        }
    };
    out
}

/// What `--device` resolved to. `None` only before `select_backend` has run.
pub fn compute_set() -> Option<&'static gpu_core::ComputeSet> {
    gpu_core::published_compute_set()
}

/// `brain bench [<name>] [--seed S]` - run the architecture-evaluation suite.
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
    // not a registry Benchmark - route it here before the registry lookup.
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

/// `brain bench scaling [--seed S]` - the multi-scale scaling-law sweep: train
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

/// `brain bench eval --arch <name> [--seed S] [--out <path>] [--smoke]` - the
/// turn-key architecture-eval harness: run EVERY registered benchmark against the
/// named architecture, aggregate per capability axis, write a results artifact
/// under `results/<arch>-<seed>.json` (or `--out`), and print the table + axes.
fn run_bench_eval(args: &[String]) {
    let mut arch = "gpt2".to_string();
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

/// `brain bench scale --arch <name> [--seed S] [--out <path>]` - the
/// per-capability predictive-scaling sweep: train+score one representative
/// benchmark per capability axis across a small SIZE grid, fit how each axis's
/// score scales with params N, extrapolate the predicted score at 2×/4× the
/// largest N, print the per-axis curves, and write
/// `results/scale-<arch>-<seed>.json`.
fn run_bench_scale(args: &[String]) {
    let mut arch = "gpt2".to_string();
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

/// `brain bench advise <eval.json> [<scale.json>]` - load an eval artifact (and,
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

/// `brain bench compare <results.json> <results.json> ...` - load ≥2 artifacts
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
        Some("caps") => std::process::exit(caps_cli::run_caps(&argv[2..])),
        Some("serve") => run_cli::run_serve(&argv[2..]),
        Some("help") | Some("-h") | Some("--help") | None => print!("{HELP}"),
        Some(_) => resolve::dispatch(&argv[1..], HELP),
    }
}

#[cfg(test)]
mod tests {
    use super::HELP;

    /// `brain --help` must at least point a reader at the HTTP serving surface
    /// and its reference (`brain serve --help`) -- before this it documented
    /// zero HTTP serving flags, so nothing told a reader brain is an OpenAI
    /// backend at all.
    #[test]
    fn global_help_advertises_the_http_serving_surface() {
        assert!(HELP.contains("brain serve --help"));
        for f in ["--openai", "--anthropic", "--openrouter", "8788"] {
            assert!(HELP.contains(f), "{f} missing from the global HTTP serving summary");
        }
    }
}
