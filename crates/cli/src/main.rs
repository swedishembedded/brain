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
mod label_cli;
mod glm_cli;
mod gguf_import;
mod gpt_cli;
mod image_io;
mod imageops;
mod lfm_cli;
mod ltxv_cli;
mod mirror_cli;
mod model_dir;
mod models_cli;
mod npu_cli;
mod omni_cli;
mod perf_cli;
mod perf_engine;
mod pid_cli;
mod placement;
mod pull_cli;
mod quantize_cli;
mod qwen35_cli;
mod qwen35moe_cli;
mod qwen_cli;
mod resident;
mod resident_asr;
mod resident_cosyvoice;
mod resident_deepseekocr;
mod resident_moondream3;
mod resident_depth;
mod resident_flux2;
mod resident_forecast;
mod resident_lfm;
mod continuous_train;
mod resident_llm;
mod resident_ltxv;
mod resident_minimaxmusic3;
mod resident_mock;
mod resident_omni;
mod resident_qwen35;
mod resident_qwen35moe;
mod resident_tts;
mod resident_upscale;
mod resident_wan;
mod resident_clip;
mod resident_t5encoder;
mod resident_sdxl;
mod resident_controlnet;
mod resident_flux1;
mod resident_pulid;
mod resident_scrfd;
mod resident_arcface;
mod resident_sam2;
mod resident_restore;
mod resident_supir;
mod resolve;
mod roofline_cli;
mod run_cli;
mod sam2_cli;
mod splat_cli;
mod supply;
mod tree;
mod tts_cli;
mod tts_serve;
mod wan_cli;
mod wan_report;
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
zipdepth, flux2, wan, worldmirror2, splat, qwen3omnimoe, diamond, toypid,
toymoe)
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

MEMORY CEILINGS (global - valid on any subcommand)
  --limit-vram-total <SIZE> hard cap on the GPU memory brain may hold, as ONE
                            total across every card (not a per-card cap).
  --limit-ram-total <SIZE>  the same, for host RAM (the CPU and any NPU).
                            SIZE is a byte count or a binary human size: 8G,
                            8GB, 8GiB, 512M, 1.5G are all accepted. An
                            allocation that would cross a ceiling is refused by
                            name, before it reaches the driver, instead of
                            OOMing the card or the box. Both apply to every
                            model automatically - they are enforced at the one
                            allocation path every model uses. Unset means no
                            ceiling (the default) and costs nothing.
                            Else $BRAIN_LIMIT_VRAM_TOTAL / $BRAIN_LIMIT_RAM_TOTAL.
  Example: brain --limit-vram-total 8G qwen3 infer --weights model.safetensors

MODEL STORE (global - valid on any subcommand)
  --brain-data-dir <DIR>   brain's data root; models live in <DIR>/models.
                           Default ~/.local/share/brain. Use it to put pulled
                           weights on another disk. Precedence, highest first:
                           --models-dir (where a subcommand has it), then
                           --brain-data-dir, then $BRAIN_MODELS_DIR, then
                           $XDG_DATA_HOME/brain/models, then
                           $HOME/.local/share/brain/models. The flag DOES
                           outrank an explicitly-set $BRAIN_MODELS_DIR, and
                           says so on stderr when it does.

DIAGNOSTIC VERBOSITY (global - valid on any subcommand, including auto-fetch)
  -v, --verbose [0-3]      0 errors only (default) | 1 +warnings | 2 +lifecycle
                           (model activate/evict, auto-fetch download progress)
                           | 3 +fine-grained detail. Repeatable (-v -v = 2);
                           bare --verbose (no number) means 1. Else $BRAIN_VERBOSE.

TRACING (structured, per-component; global - valid on any subcommand)
  --trace-<family> <0-5>   0 off (default) | 1 errors | 2 +warnings | 3 +lifecycle
                           | 4 +per-step decisions and timings | 5 everything
  --trace <family>=<level> same thing, generic - reaches every family, including
                           any that has no dedicated flag yet (repeatable)
  --trace-format text|json how to render (default text)
  --trace-output -|PATH    where to write (default -, meaning stdout)

  families:
    --trace-gpu   <0-5>    device registry, adapter enumeration, backend open/submit/wait
    --trace-ltxv  <0-5>    LTX-2.5 video: pipeline stages, denoise steps, streamed DiT blocks, the Gemma-4 text encode
  BRAIN_TRACE=ltxv=5,gpu=3 sets the same levels without a flag (any --trace* flag wins).

  Every line names the component it came from (the emitting Rust module), so a
  trace over several crates stays attributable; --trace-format json puts that
  component in a real `target` field for jq/grep rather than inside a message.
  Example: brain --trace-ltxv 5 --trace-format json --trace-output run.jsonl ltxv t2v ...

DEVICES
  brain devices                     # canonical GPU table + ambient selection

DATA
  brain data gen <name> [--out DIR --n N --seed S]
      names: calculator | reverser | wordcalc | timeseries | shakespeare_char | gpt

LABEL (caption a dataset with a vision-language model)
  brain label images <dir> [--model qwen3vl|fastvlm|llava] [--weights DIR]
      [--out FILE] [--prompt \"...\"] [--trigger PHRASE] [--max-new N] [--overwrite]
      Writes <dir>/captions.yaml - what `brain flux2 finetune` and every other
      captioned-image trainer read. Resumable and idempotent: a re-run fills in
      only what is missing and never overwrites a hand-edited caption unless
      --overwrite says so. `brain label --help` for the full flag list.

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
      served generically as model `flux2-klein`: brain caps flux2-klein, and D-Bus
      (examples/imagegen); the manifest's text2image/edit/lora_train actions are
      D-Bus/HTTP only today -- this `brain flux2` CLI reaches generate/infer, not those
      9B variants need BRAIN_FLUX2_ALLOW_NC=1 (FLUX Non-Commercial license)

WAN 2.1 (text-to-video; 81 frames at 16 fps, 480p on the 1.3B variant)
  brain wan t2v --prompt \"...\" --output-path out.mp4 [--seed S]
      [--frames N (1+4k) --width W --height H --steps N --shift S --guidance G]
      [--negative-prompt \"...\" --fps N --solver unipc|dpm++ --variant t2v-1.3B]
      [--device cpu|gpu --t5-device cpu|gpu]
      weights by flag or env: --dit/--vae/--t5/--tokenizer beat BRAIN_WAN_DIT,
      BRAIN_WAN_VAE, BRAIN_WAN_T5, BRAIN_WAN_TOKENIZER
      writing the file needs ffmpeg; without it the frames land in
      <output-path>.frames/ and the assembling command is printed

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
                                           # prints a benchmark|score|threshold|pass table
      names: mqar | toolcall | mad_recall | mad_fuzzy_recall | mad_noisy_recall |
             mad_selective_copy | mad_memorize | parity | mod_add | dyck |
             mad_compress | forecast_seasonal_trend | forecast_ar1 |
             forecast_garch_vol | forecast_regime_switch | forecast_random_walk |
             forecast_jump_diffusion
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
      --weights naming a real pulled checkpoint also writes into the shared
      pricing cache `brain models list`/`brain models profile` read - see
      MODELS below.

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

PULL MODEL WEIGHTS (fetch a model's official weights into the store)
  brain pull <model> [--brain-data-dir DIR]
      <model> is the canonical reference OR the HuggingFace page URL - a
      pasted /tree/<branch>, /blob/... or /resolve/... link resolves to the
      same repo:
        brain pull Qwen/Qwen3-0.6B
        brain pull https://huggingface.co/Qwen/Qwen3-0.6B/tree/main
      Progress goes to STDOUT: an in-place bar with throughput and ETA on a
      terminal, ten plain lines for the whole pull when piped. Re-running is
      cheap - files already in the store are skipped, so an interrupted pull
      resumes by repeating the command. `brain fetch` is an accepted alias.

MODELS (the store's own view of itself: what's local, what isn't, what it costs)
  brain models list [--arch ID] [--local] [--plain|--tui] [--json] [--reprofile]
      Architecture -> provider repo -> quantization. A terminal gets an
      interactive tree (arrow/j/k/pgup/pgdn move, enter/space expand-collapse
      a branch OR open a pulled model's tensor detail view, esc back out of
      a detail view or quit at the top, / filter, q quit); piped gets plain
      box-drawing lines whose LEAF lines carry the full canonical id, so
      `brain models list | grep Q4_K_M` returns complete lines. --reprofile
      re-measures this device's roofline and re-prices every local model
      (bandwidth tier - safe regardless of size).
  brain models list-adapters [--arch ID] [--plain|--tui] [--json]
      Architecture -> base variant -> LoRA adapter, with rank/alpha/dataset
      from the adapter's own card.
  brain models info <model> [--json]
      One checkpoint's real tensor tree: name, dtype, shape, size - adapter
      tensors merged in and marked with a leading '+'.
  brain models profile <model> [--measure [--reps N]]
      Price ONE already-pulled model now and cache the result. Errors, never
      fetches, if it is not pulled. `brain flops --weights <path>` writes into
      the same cache - see FLOP/OPS ACCOUNTING above. --measure instead
      builds the model for real and TIMES it (load time, cold pass, best of
      --reps (default 5) hot passes, achieved FLOP/s, per-layer FLOPs) -
      real execution, never cached.

ROOFLINE (measured, cross-accelerator hardware compute capacity)
  brain roofline [gpu|npu|cpu] [--reprofile] [--json]
      What can this box actually DO - raw, model-independent GFLOP/s, GOP/s
      and GB/s for every GPU (not just the ambient one), the NPU, and the
      CPU, each dtype it supports. Streamed as each accelerator's
      measurement completes - GPU first, then NPU, then CPU - so a GPU
      number lands well under 10s regardless of how long the NPU probe
      takes to conclude \"not present\". `gpu`/`npu`/`cpu` scope the report
      to just that section. --reprofile forces a fresh measurement instead
      of GPU's own cache-first path (NPU's probe has no cache to bypass, so
      it is always fresh already; CPU always measures fresh, it is fast).
      Distinct from `brain flops` (one model's cost, see FLOP/OPS ACCOUNTING
      above) and `brain perf` (empirical serving load, see PERFORMANCE
      BENCHMARKING above). Plain rows are self-contained
      (`brain roofline | grep gpu0`); --json emits the same rows as an array.

GGUF IMPORT (one-time conversion; dispatches on general.architecture)
  brain import FILE [--out PATH] [--id VENDOR/REPO]
      Convert a GGUF checkpoint to brain-native safetensors, choosing the
      importer by the file's own `general.architecture`. Defaults to a sibling
      <stem>.brain.safetensors, which the model-dir scan then serves on its own.
      brain import --list                # registered architectures

QUANTIZE (the export sibling: any checkpoint -> a quantized GGUF)
  brain quantize SRC --out PATH [--tier Q8_0] [--keep SUBSTR] [--plan]
      Convert a .safetensors file/dir or an existing .gguf to a quantized
      GGUF. Needs no per-architecture code: a tensor is quantized if it is
      rank 2 with a block-aligned row, unless --keep names it. --plan prints
      the per-tensor decision and writes nothing.

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
/// score scales with params N, extrapolate the predicted score at twice and
/// four times the largest N, print the per-axis curves, and write
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

/// Extract the global `--trace*` flags from anywhere in the args, install the
/// process's one `tracing` subscriber, and return the remaining args.
///
/// Runs FIRST in `main`, before `--device` resolution and before any
/// subcommand: device probing, adapter enumeration and checkpoint opening are
/// exactly the early work a `--trace-gpu` run is trying to see, so a
/// subscriber installed any later would miss it.
fn install_tracing(argv: Vec<String>) -> Vec<String> {
    let (cfg, rest) = match brain_trace::strip_args(argv) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("brain: {e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = brain_trace::install(&cfg) {
        eprintln!("brain: {e}");
        std::process::exit(2);
    }
    rest
}

/// Extract the global `-v`/`--verbose [0-3]` flags from anywhere in `argv`
/// (else `$BRAIN_VERBOSE`) and return `(level, remaining args)`. Pure - no
/// process-global side effect - so it's testable without racing every other
/// test in this binary over `residency::log`'s one `AtomicU8`; [`main`] is
/// the sole caller of [`residency::log::set_verbosity`] with this result.
///
/// [`main`] runs this FIRST, alongside [`install_tracing`]: auto-fetch
/// download progress (`residency::log::info`, from
/// `crate::supply::ensure_env_weights`) fires from `resolve::dispatch`,
/// before any subcommand's own argument parsing even starts, so a level
/// installed any later would miss it. This used to be `run_serve`'s own
/// local flag (`-v` only took effect on `brain serve`); every other command,
/// including a plain `brain s3dit text2image` that triggers a multi-gigabyte
/// weight fetch, had no way to ask for the progress lines that code already
/// emits.
fn parse_verbosity(argv: Vec<String>) -> (u8, Vec<String>) {
    let mut level: u8 = std::env::var("BRAIN_VERBOSE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut rest = Vec::with_capacity(argv.len());
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "-v" => level = level.saturating_add(1),
            "--verbose" => {
                level = match argv.get(i + 1).and_then(|s| s.parse::<u8>().ok()) {
                    Some(n) => {
                        i += 1;
                        n
                    }
                    None => 1, // bare --verbose (no numeric arg) means level 1
                };
            }
            _ => rest.push(argv[i].clone()),
        }
        i += 1;
    }
    (level, rest)
}

/// Extract the global `--limit-vram-total`/`--limit-ram-total <SIZE>` flags
/// from anywhere in `argv` and return `(vram, ram, remaining args)` in bytes.
/// Pure - no process-global side effect - so it is testable without racing
/// every other test in this binary over `memauth`'s one `OnceLock`; [`main`]
/// is the sole caller of [`memauth::publish_limits`] with this result, which
/// is also what applies the `$BRAIN_LIMIT_*` fallbacks for an omitted flag.
///
/// Hard-exits on an unparseable value, like [`select_backend`] does: the user
/// just typed it, so a silently ignored ceiling would be the worst outcome -
/// the run would proceed unbounded and OOM exactly as if the flag were never
/// supported.
fn parse_limits(argv: Vec<String>) -> (Option<u64>, Option<u64>, Vec<String>) {
    let (mut vram, mut ram) = (None, None);
    let mut rest = Vec::with_capacity(argv.len());
    let mut i = 0;
    while i < argv.len() {
        let slot = match argv[i].as_str() {
            "--limit-vram-total" => Some(&mut vram),
            "--limit-ram-total" => Some(&mut ram),
            _ => None,
        };
        match slot {
            Some(slot) => {
                let flag = argv[i].clone();
                let Some(value) = argv.get(i + 1) else {
                    eprintln!("brain: {flag} needs a size (e.g. 8G, 8GiB, 512M, or a byte count)");
                    std::process::exit(2);
                };
                match memauth::parse_size(value) {
                    Ok(bytes) => *slot = Some(bytes),
                    Err(e) => {
                        eprintln!("brain: {flag}: {e}");
                        std::process::exit(2);
                    }
                }
                i += 2;
            }
            None => {
                rest.push(argv[i].clone());
                i += 1;
            }
        }
    }
    (vram, ram, rest)
}

/// Extract the global `--brain-data-dir <root>` flag from anywhere in `argv`
/// and publish it, returning the remaining args.
///
/// A GLOBAL option, on the top-level parser, deliberately: it answers "where
/// does brain keep its models", which `brain pull`, a flagless `brain infer`'s
/// auto-fetch, `brain serve`'s catalog scan and every capability provider all
/// have to agree on. Scoping it to `pull` alone would mean pulling into one
/// directory and then serving from another, which is the bug the single
/// resolver ([`brain_modelstore::default_root`]) exists to prevent. It is
/// published INTO that resolver rather than passed around, so callers with no
/// CLI flag in scope see the same answer.
///
/// Hard-exits on a missing value, like [`parse_limits`] and [`select_backend`]:
/// silently ignoring a directory the user typed would download gigabytes to
/// the wrong disk.
fn parse_data_dir(argv: Vec<String>) -> (Option<String>, Vec<String>) {
    let mut root = None;
    let mut rest = Vec::with_capacity(argv.len());
    let mut i = 0;
    while i < argv.len() {
        if argv[i] == "--brain-data-dir" {
            let Some(value) = argv.get(i + 1) else {
                eprintln!("brain: --brain-data-dir needs a directory (brain's data root; models land in <DIR>/models)");
                std::process::exit(2);
            };
            root = Some(value.clone());
            i += 2;
        } else {
            rest.push(argv[i].clone());
            i += 1;
        }
    }
    (root, rest)
}

/// Publish `--brain-data-dir` into the one models-directory resolver, saying
/// so out loud when it overrules an explicitly-set `BRAIN_MODELS_DIR`.
///
/// The flag outranking the environment is the same rule `--models-dir` and
/// `--device` already follow, but a models directory is where gigabytes land:
/// an operator who exported `BRAIN_MODELS_DIR` and then inherited a
/// `--brain-data-dir` from a wrapper script deserves to be told which one won,
/// rather than discovering it by running out of disk somewhere unexpected.
fn apply_data_dir(root: Option<String>) {
    let Some(root) = root.filter(|s| !s.is_empty()) else { return };
    let root = std::path::PathBuf::from(root);
    let models = brain_modelstore::models_dir_in(&root);
    if let Some(env) = std::env::var_os("BRAIN_MODELS_DIR").filter(|s| !s.is_empty()) {
        if std::path::Path::new(&env) != models {
            eprintln!("brain: --brain-data-dir overrides BRAIN_MODELS_DIR ({}); using {}", std::path::Path::new(&env).display(), models.display());
        }
    }
    brain_modelstore::publish_data_root(Some(root));
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let argv = install_tracing(argv);
    let (verbosity, argv) = parse_verbosity(argv);
    residency::log::set_verbosity(verbosity);
    // Before `select_backend` (which probes adapters) and before any model,
    // so the ceiling is already published by the time the first byte of device
    // memory is requested - there is no window in which an allocation escapes
    // it.
    let (limit_vram, limit_ram, argv) = parse_limits(argv);
    memauth::publish_limits(limit_vram, limit_ram);
    // Before any subcommand, so every surface that resolves a models
    // directory - `pull`, auto-fetch, the served catalog scan - reads the
    // same published answer.
    let (data_dir, argv) = parse_data_dir(argv);
    apply_data_dir(data_dir);
    if matches!(argv.get(1).map(String::as_str), Some("--version" | "-V")) {
        println!("brain {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    let argv = select_backend(argv);
    // After `--device` is resolved (so the candidate set is exactly what the
    // user made schedulable) and before any model is built. This is what
    // turns `gpu_core::devices`' placement seam from inert into the
    // capacity-aware default: with no `--device`, a model lands on a card
    // that can actually hold it instead of unconditionally on card 0. An
    // explicit selection never consults it.
    placement::install();
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
        Some("label") => label_cli::run_label(&argv[2..]),
        Some("caps") => std::process::exit(caps_cli::run_caps(&argv[2..])),
        // `fetch` is an accepted alias, not a second verb: it is the spelling
        // several of this workspace's test-fixture instructions already tell a
        // reader to run, and honouring it costs one arm.
        Some("pull") | Some("fetch") => std::process::exit(pull_cli::run_pull(&argv[2..])),
        Some("models") => std::process::exit(models_cli::run_models(&argv[2..])),
        Some("roofline") => std::process::exit(roofline_cli::run_roofline(&argv[2..])),
        Some("serve") => run_cli::run_serve(&argv[2..]),
        Some("help") | Some("-h") | Some("--help") | None => print!("{HELP}"),
        Some(_) => resolve::dispatch(&argv[1..], HELP),
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_data_dir, parse_limits, parse_verbosity, HELP};

    /// The memory ceilings are stripped from anywhere in `argv`, parse human
    /// sizes, and consume exactly their own value token.
    #[test]
    fn parse_limits_strips_both_flags_and_reads_human_sizes() {
        let argv = ["brain", "--limit-vram-total", "8G", "ltxv", "t2v", "--limit-ram-total", "16GiB", "--prompt", "x"].map(String::from).to_vec();
        let (vram, ram, rest) = parse_limits(argv);
        assert_eq!(vram, Some(8 << 30));
        assert_eq!(ram, Some(16 << 30));
        assert_eq!(rest, ["brain", "ltxv", "t2v", "--prompt", "x"].map(String::from).to_vec());

        // Neither flag present: both None, args untouched -- the default path,
        // which must stay a no-op.
        let (vram, ram, rest) = parse_limits(["brain", "caps"].map(String::from).to_vec());
        assert_eq!((vram, ram), (None, None));
        assert_eq!(rest, ["brain", "caps"].map(String::from).to_vec());
    }

    /// The same two-directional discipline the trace families get: a ceiling
    /// flag that `brain help` documents but nothing parses is a flag that
    /// silently does nothing (the worst possible failure for a memory
    /// ceiling - the run proceeds unbounded), and one that parses but is
    /// undocumented is a flag nobody can discover.
    #[test]
    fn every_memory_ceiling_flag_is_documented_and_every_documented_one_is_parsed() {
        for f in ["--limit-vram-total", "--limit-ram-total", "BRAIN_LIMIT_VRAM_TOTAL", "BRAIN_LIMIT_RAM_TOTAL"] {
            assert!(HELP.contains(f), "{f} is implemented but absent from `brain help`");
        }
        for line in HELP.lines() {
            let Some(rest) = line.trim_start().strip_prefix("--limit-") else { continue };
            let name = rest.split(|c: char| !(c.is_ascii_lowercase() || c == '-')).next().unwrap_or("");
            if name.is_empty() {
                continue;
            }
            let flag = format!("--limit-{name}");
            let (vram, ram, rest) = parse_limits(["brain".to_string(), flag.clone(), "8G".to_string(), "caps".to_string()].to_vec());
            assert_eq!(rest, ["brain", "caps"].map(String::from).to_vec(), "`brain help` documents {flag}, which parse_limits does not strip");
            assert_eq!(vram.or(ram), Some(8 << 30), "`brain help` documents {flag}, which parse_limits does not turn into a ceiling");
        }
    }

    /// The data-root flag is stripped from anywhere in `argv`, consumes
    /// exactly its own value token, and leaves everything else untouched --
    /// including the default path, where it must be a complete no-op.
    #[test]
    fn parse_data_dir_strips_the_flag_from_anywhere_and_consumes_its_value() {
        let argv = ["brain", "pull", "--brain-data-dir", "/synthetic-root/brain", "Qwen/Qwen3-0.6B"].map(String::from).to_vec();
        let (root, rest) = parse_data_dir(argv);
        assert_eq!(root.as_deref(), Some("/synthetic-root/brain"));
        assert_eq!(rest, ["brain", "pull", "Qwen/Qwen3-0.6B"].map(String::from).to_vec());

        // Leading position, before the subcommand.
        let argv = ["brain", "--brain-data-dir", "/elsewhere", "serve", "--openai"].map(String::from).to_vec();
        let (root, rest) = parse_data_dir(argv);
        assert_eq!(root.as_deref(), Some("/elsewhere"));
        assert_eq!(rest, ["brain", "serve", "--openai"].map(String::from).to_vec());

        // Absent: no root, args byte-identical.
        let (root, rest) = parse_data_dir(["brain", "caps"].map(String::from).to_vec());
        assert_eq!(root, None);
        assert_eq!(rest, ["brain", "caps"].map(String::from).to_vec());
    }

    /// The same two-directional discipline the trace families and memory
    /// ceilings get, for the same reason: a global flag that `brain help`
    /// documents but nothing strips would be swallowed by a subcommand's own
    /// parser as an unknown flag, and one that is parsed but undocumented is
    /// undiscoverable.
    #[test]
    fn the_data_dir_flag_is_documented_and_the_documented_precedence_is_stated() {
        assert!(HELP.contains("--brain-data-dir"), "--brain-data-dir is implemented but absent from `brain help`");
        // The ladder it sits in must be spelled out where the flag is, so a
        // reader is never left guessing whether it beats the environment.
        for token in ["BRAIN_MODELS_DIR", "XDG_DATA_HOME", "--models-dir"] {
            assert!(HELP.contains(token), "{token} missing from the model-store precedence summary");
        }
        let (root, rest) = parse_data_dir(["brain".to_string(), "--brain-data-dir".to_string(), "/x".to_string(), "caps".to_string()].to_vec());
        assert_eq!(rest, ["brain", "caps"].map(String::from).to_vec(), "`brain help` documents --brain-data-dir, which parse_data_dir does not strip");
        assert_eq!(root.as_deref(), Some("/x"));
    }

    /// `brain pull` must be reachable and documented -- an undocumented verb
    /// is one nobody finds, and a documented verb with no dispatch arm is a
    /// command that prints the whole help text instead of pulling anything.
    #[test]
    fn the_pull_verb_is_documented_with_both_argument_spellings() {
        assert!(HELP.contains("brain pull <model>"), "`brain pull` missing from `brain help`");
        assert!(HELP.contains("https://huggingface.co/"), "`brain help` does not show that a URL is accepted");
        assert!(HELP.contains("brain fetch"), "the accepted `fetch` alias is undocumented");
    }

    /// `-v` is repeatable and additive; the flag itself is stripped from the
    /// returned args so downstream parsing never sees it. Pure function, so
    /// no `residency::log` global state to race against other tests.
    #[test]
    fn parse_verbosity_strips_repeated_v_and_sums_the_level() {
        let argv = ["brain", "-v", "-v", "s3dit", "text2image"].map(String::from).to_vec();
        let (level, rest) = parse_verbosity(argv);
        assert_eq!(level, 2);
        assert_eq!(rest, ["brain", "s3dit", "text2image"].map(String::from).to_vec());
    }

    /// `--verbose <N>` sets the level explicitly and consumes its argument;
    /// a bare `--verbose` (no following number) means level 1.
    #[test]
    fn parse_verbosity_accepts_an_explicit_level_and_a_bare_form() {
        let argv = ["brain", "--verbose", "3", "ltxv", "t2v"].map(String::from).to_vec();
        let (level, rest) = parse_verbosity(argv);
        assert_eq!(level, 3);
        assert_eq!(rest, ["brain", "ltxv", "t2v"].map(String::from).to_vec());

        let (level, rest) = parse_verbosity(vec!["brain".to_string(), "--verbose".to_string()]);
        assert_eq!(level, 1);
        assert_eq!(rest, vec!["brain".to_string()]);
    }

    /// The trace-family registry and this binary's help text are two lists of
    /// the same thing, and only one of them is compiled: a family added to
    /// the registry with no help line ships a flag nobody can discover, and a
    /// help line for a family that no longer exists documents a flag that
    /// hard-errors. Assert both directions.
    #[test]
    fn every_trace_family_is_documented_and_every_documented_one_exists() {
        for f in brain_trace::FAMILIES {
            let flag = format!("--trace-{}", f.name);
            assert!(HELP.contains(&flag), "{flag} is registered but absent from `brain help`");
            assert!(HELP.contains(f.about), "{flag}'s registry description is not what `brain help` shows");
        }
        for line in HELP.lines() {
            let Some(rest) = line.trim_start().strip_prefix("--trace-") else { continue };
            let name = rest.split(|c: char| !(c.is_ascii_lowercase() || c.is_ascii_digit())).next().unwrap_or("");
            // `--trace-format`/`--trace-output` are global options, not families.
            if name.is_empty() || name == "format" || name == "output" || rest.starts_with('<') {
                continue;
            }
            assert!(brain_trace::family(name).is_some(), "`brain help` documents --trace-{name}, which is not a registered trace family");
        }
        for f in ["--trace-format", "--trace-output", "--trace <family>=<level>", "BRAIN_TRACE"] {
            assert!(HELP.contains(f), "{f} missing from the global tracing summary");
        }
    }

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
