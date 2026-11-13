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
mod collective_profile;
mod data_cli;
mod decide_cli;
mod devices_api;
mod devices_cli;
mod doctor_cli;
mod federated_cli;
mod flops_cli;
mod forecast_cli;
mod label_cli;
mod gguf_cli;
mod image_io;
mod lsa_cli;
mod load_line;
mod import_gguf_cli;
mod models_cli;
mod npu_cli;
mod perf_cli;
mod perf_engine;
mod perf_longctx;
mod perf_qwen35;
mod perf_ricursive;
mod perf_qwen35moe;
mod placement;
mod plan_cli;
mod pull_cli;
mod finetune_cli;
mod optimize_cli;
mod prune_cli;
mod quantize_cli;
mod qwen35_cli;
mod qwen_cli;
mod resolver_cli;
mod adapter_directory;
mod continuous_train;
mod resolve;
mod roofline_cli;
mod run_cli;
mod run_record;
mod serve_daemon;
mod supply;
mod tree;
/// Whether `--device`/`BRAIN_DEVICE` EXPLICITLY named the NPU. The NPU is a
/// whole-graph (OpenVINO) path, not a `gpu_core` backend, so the commands that
/// support it (`brain yolo detect`, `qwen infer`,
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
own dedicated flags (gpt2, qwen3, qwen35, glmdsa, lfm2,
yolov8, zipdepth, flux2, wan, worldmirror2, splat, qwen3omnimoe,
toypid, toymoe)
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

--backend selects HOW THAT HARDWARE IS DRIVEN - a separate axis from --device,
which picks the hardware itself. Omitted, it is chosen from the hardware:
cuda where every card is NVIDIA's and CUDA can run (driver plus NVRTC or the
ahead-of-time kernels), wgpu on any other GPU, cpu without one. `brain devices`
prints the choice and why.

  wgpu      portable WebGPU/wgpu path
  vulkan    native Vulkan compute (ash, WGSL -> SPIR-V)
  cuda      native CUDA driver API; kernels compile through NVRTC on first
            dispatch, so it needs the CUDA toolkit's NVRTC beside the driver
  cpu       native CPU backend

An explicitly named backend is a hard contract: if it cannot be built, brain
fails and names the reason instead of quietly substituting another one.

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

CONFIG FILE (global - valid on any subcommand)
  --config <FILE>          a YAML file of settings, loaded before anything
                           else parses. Default: ~/.config/brain/config.yaml
                           ($XDG_CONFIG_HOME/brain/config.yaml) when it exists.
                           A subcommand key (`serve`, `qwen3 infer`) maps that
                           command's flags, without the dashes, to values (a
                           list for a repeatable flag, true for a switch); a
                           flag on the command line always wins. A BRAIN_* key
                           sets a variable still read from the environment,
                           filling gaps only.
  Example config.yaml:
    serve:
      api-key-file: /etc/brain/key
      checkpoint: checkpoints/qwen3.8-27b

  Per-model serving tunables (context, batching, KV storage) are command-line
  flags, not config keys - `brain serve --ctx 8192` and friends; run
  `brain serve --help` for the full list.

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

AUTO-FETCH (global - valid on any subcommand)
  --autofetch              Allow missing model weights to be downloaded on
                           demand (same as $BRAIN_AUTO_FETCH=1; the flag wins
                           over an exported =0). Default OFF: a run whose
                           weights are not pulled prints an error naming them
                           instead of downloading. `brain pull <model>`
                           fetches without the flag.

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

DOCTOR
  brain doctor [--model <vendor>/<repo>] [--models-dir DIR] [--json]
      # can a model be served here? checks the model store, the free disk, each
      # GPU's free memory and, with --model, that model's weights and tokenizer
      # and whether some card has room for it. Loads nothing; exit 1 if a check fails.

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

DECIDE (decision-model training on outcome-labelled examples)
  brain decide train --data FILE --base REF --out DIR [--freeze-encoder | --finetune-encoder]
      [--steps N --batch N --seed N --activations keep|recompute]
      Trains a decision head on the `train` split of a JSONL file and writes a
      checkpoint whose manifest records its lineage and says `unvalidated`.
  brain decide eval --model DIR|REF --data FILE --split validation|test
      [--costs FILE] [--order-seed N] [--json] [--out FILE]
      Accuracy against the majority baseline, NLL, Brier, ECE/AdaECE, coverage
      versus accuracy, option-order instability, cost-weighted regret and a
      per-family breakdown, as a table or JSON.
  brain decide promote --model DIR --data FILE --policy POLICY.json [--json]
      Re-evaluates DIR on the test split under the policy; only if every gate
      passes is it marked validated (what `auto` requires). A failed gate is
      printed with its margin, nothing changes, the exit code is 1.
  brain decide rollback --model DIR
      Makes the last-known-good checkpoint the validated one again.
  brain decide inspect --data FILE [--json]
      Validates the file (schema, leakage across splits) and prints its summary.
      `brain caps decide` lists the served `decide` action.

GPT2 (brain/gpt2: the dense baseline; samples/models/gpt2)
  brain gpt2 train --data <dir> --save F [--steps N --layers L --d-model D --mask =]
                   [--job-plan FILE --job-rank N --job-secret FILE --profile-collectives FILE]
  brain gpt2 eval --weights F --data <dir>
  brain gpt2 generate --weights F --prompt \"...\" [--max-new N --temperature X --top-k K]
      every parameter: brain gpt2 <action> --help

FLUX.2 Klein (text-to-image + image editing; 4-step distilled flow matching)
  brain flux2 text2image --prompt \"...\" --seed S --out image=out.png [--width W --height H]
  brain flux2 edit --image in.png [--image0 ref.png]... [--mask m.png | --strength S] --prompt \"...\" --seed S --out image=out.png
  brain flux2 lora_train --data <dir> --save adapter.brain --seed S
      every parameter: brain flux2 <action> --help; --dit/--vae/--text-encoder/--tokenizer name a role
      9B variants need BRAIN_FLUX2_ALLOW_NC=1 (FLUX Non-Commercial license)

WAN 2.1 (text-to-video; 81 frames at 16 fps, 480p on the 1.3B variant)
  brain --device gpu0 wan t2v --prompt \"...\" --out video=out.mp4 [--seed S --dit-dtype int8 --te-device gpu1]
  brain wan lora_train --data <dir> --save adapter.brain --seed S [--report <dir>]
      every parameter: brain wan <action> --help; --dit/--vae/--text-encoder/--tokenizer name a role

DIAMOND world model (brain/diamond; the interactive window and the ONNX/NPU export are samples/models/diamond)
  brain diamond import --source agent.pt --dest diamond.safetensors [--actions_count N]
  brain diamond rollout [--model diamond|fake] --action_seq 1,2,0 --seed S --out clip=rollout.mp4 [--record DIR --dump_ppm DIR --hashes]
  brain diamond replay --episode DIR [--verify --model diamond|fake --seed S --tolerance T --context N]
  brain diamond bench [--model diamond|fake] [--frames N --profile]       # ms/frame + fps
  brain diamond finetune --data <episode-dir> --save tuned.safetensors --seed S [--steps N --lr X --wd X --clip X]
      every parameter: brain diamond <action> --help; --weights names the checkpoint

WorldMirror-2 (brain/worldmirror2: multi-view images to a 3D Gaussian Splatting scene; samples/models/worldmirror2)
  brain worldmirror2 import --source <model.safetensors|hf_dir> --dest mirror.safetensors
  brain worldmirror2 reconstruct --images <dir|clip.mp4> --out scene=scene.ply --out cameras=cameras.json
      every parameter: brain worldmirror2 <action> --help

3D GAUSSIAN SPLATTING (brain/splat; the fly-through window is samples/models/splat)
  brain splat render --scene s.ply --out image=view.png [--width W --height H --eye x,y,z --target x,y,z]
  brain splat fit --scene s.ply --cameras cams.json --images <dir> --out scene=fitted.ply
  brain splat train --images <dir> --out scene=s.ply --out cameras=cams.json
      also info, orient, merge, prune, sfm; every parameter: brain splat <action> --help

YOLOv8 (brain/yolov8: detection, training, evaluation; NPU export and run in samples/models/yolov8)
  brain yolov8 detect --image photo.png --json [--conf X --iou X]
  brain yolov8 train --data <dir> --save yolo.safetensors --seed S
  brain yolov8 finetune --data <dir> --save ft.safetensors [--nc C --freeze-backbone ...]
  brain yolov8 eval --data <dir> [--split val]          # mAP@0.5, mAP@0.5:0.95, per-class AP, P/R
      every parameter: brain yolov8 <action> --help

INTEL NPU (OpenVINO)
  brain npu check    --onnx M [--ov-device NPU]              # ONNX op histogram + compile/coverage
  brain npu bench    --onnx M [--input S --ov-device NPU --iters N --hint latency|throughput]
      --ov-device selects the OpenVINO target device (NPU|CPU|GPU|AUTO), unrelated to
      brain's own --device; check without OpenVINO decodes the graph only.

TOY ARCHITECTURES (brain's own, no upstream reference; samples/models/toymoe, toypid)
  brain toymoe train --save F [--steps N --batch B --block T --lr X]
  brain toymoe generate --weights F [--prompt 1,2,3,4 --max-new N --temperature X --top-k K]
  brain toymoe eval --weights F [--samples N]
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
              [--admit-deadline-ms N] [--cold-build-admit-deadline-ms N]
      Serves the shared executor over localhost HTTP: one port + one freshly
      generated key per dialect (defaults: openai 8788, anthropic 8787,
      openrouter 8789). OpenAI/OpenRouter base URL: http://127.0.0.1:PORT
      *or* http://127.0.0.1:PORT/v1 -- both work.  Full reference:
      brain serve --help

CAPACITY (will this run here, and what will it cost me)
  brain plan <model> <action> [--params JSON] [--json]
             [--bus session|system|<address>] [--name <bus name>]
      Asks a running `brain serve --dbus` daemon what would happen if this
      action ran on it NOW: memory required, whether it is already resident and
      where, which device it would be placed on, exactly what would be evicted
      to make room, and every device it could run on at all. Reserves nothing
      and loads nothing. Exits 0 when it would run, 1 when it would not, so a
      script can branch on the answer without parsing the text.
      Needs the daemon because a plan describes a LIVE host: a fresh process
      has nothing resident and would always report room it does not have.

EVENT/STDIO CONTROLLER
  brain serve --stdio [--gpt <ckpt>] [--yolo <ckpt>] [--conf X] [--max-new N --temp X --top-k K --seed S]
      Event-driven HFSM controller: read JSONL events on stdin, emit JSONL events
      on stdout (text streaming + object detection). With no --gpt, a fake echo
      model runs; with no --yolo, a fake detector runs,
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
  brain glmdsa train --data <dir> [--save F --size tiny|small|base --steps N --seed $RANDOM]
  brain glmdsa finetune --data <dir> --save F [--weights BASE --steps N --lr X]
  brain glmdsa eval --data <dir> [--weights F --batches N]
  brain glmdsa generate --prompt \"...\" [--weights F --max-new N --temp X --top-k K]
  brain glmdsa import --source <hf_dir> --dest glm.safetensors
      # the OpenVINO export and NPU decode are actions of samples/models/glmdsa

QWEN3.5-35B-A3B (brain/qwen35moe: hybrid Gated-DeltaNet/GQA sparse-MoE decoder)
  brain import-gguf Qwen3.5-35B-A3B-Q8_0.gguf             # a brain checkpoint the store finds
  brain qwen35moe generate --prompt \"...\" --out text=out.txt [--chat true --precision int8 --seed $RANDOM]
      every parameter: brain qwen35moe generate --help; --weights/--tokenizer name a role
      # the ONNX export is an action of samples/models/qwen35moe

QWEN3-OMNI (brain/qwen3omnimoe: text/audio/image/video in, text + speech out)
  brain qwen3omnimoe generate --prompt \"...\" --out text=out.txt     # the store's published checkpoint
  brain qwen3omnimoe speak --prompt \"...\" --out audio=reply.wav
  brain qwen3omnimoe import --source <hf_dir> --dest Qwen3-Omni-30B-A3B-Instruct-W8A16.safetensors [--id VENDOR/REPO]
      # brain serve serves the published directory and, GPU-sharded, the int8 Thinker over
      # the checkpoint import wrote; every parameter: brain qwen3omnimoe <action> --help

LFM2.5-ENCODER (bidirectional conv/attention encoder, MLM head, 8k context)
  brain lfm2 fill_mask --text \"… <|mask|> …\" [--topk K]
  brain lfm2 embed --text \"…\" --out embeddings=emb.f32
  brain lfm2 tokenize --corpus corpus.txt --dest data/lfm
  brain lfm2 finetune --data data/lfm --save ft.safetensors [--steps N --batch B --seq T]
  brain lfm2 eval --data data/lfm                       # pseudo-ppl + masked-acc
  brain lfm2 import --source <hf_dir> --dest lfm.safetensors
      every parameter: brain lfm2 <action> --help; --weights/--tokenizer name a role
  brain npu lfm        --weights F --seq S --out model.onnx [--int8]   # OpenVINO export
  make perf/model/lfm                      # standalone concurrency benchmark (scheduler+residency)

QWEN3-TTS (brain/qwen3tts: Talker + MTP + neural codec; NPU actions are samples/models/qwen3tts)
  brain qwen3tts synth --text \"Hello.\" --out audio=hello.wav
  brain qwen3tts clone --ref ref.wav --ref-text \"...\" --text \"...\" --out audio=me.wav
  brain qwen3tts design --instruct \"a calm low voice\" --text \"...\" --out audio=d.wav
  brain qwen3tts finetune --data data/tts --save talker.safetensors [--steps N --full true]
  brain qwen3tts <batch|sim|import> ...
      every parameter: brain qwen3tts <action> --help

MONOCULAR DEPTH (brain/zipdepth; image and camera windows are samples/models/zipdepth)
  brain zipdepth infer --image photo.png --out depth=depth.png [--view side --out view=side.png]
  brain zipdepth train --save zipdepth.safetensors [--steps N --fresh true]
  brain zipdepth calib --images <dir>
      every parameter: brain zipdepth <action> --help

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

GGUF INSPECT (look at a downloaded .gguf FILE before importing/pulling it)
  brain gguf inspect PATH [--json]
      PATH is a real filesystem path, opened directly - never a model-store
      reference. Prints the KV metadata plus a name/dtype/shape/size tensor
      tree; --json emits the same information as data instead.

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

LSA (latent sparse attention for any model whose engine captures its attention)
  brain lsa distill --corpus DIR --out FILE (--model REPO | --checkpoint PATH) [--tokens N] [--epochs N] [--top-k N] [--top-p N]
  brain lsa eval    --corpus DIR --indexer random|pca|FILE[,..] (--model REPO | --checkpoint PATH)
  brain lsa prefill --indexer random|FILE (--model REPO | --checkpoint PATH) [--tokens N,N,..]
      Distil, evaluate and time a latent-sparse-attention indexer. The
      architecture is read from the checkpoint's config.json (--arch names
      it): qwen3 over every layer, ricursive over its global layers. brain lsa
      --help lists every option.

OPTIMIZE (the weight format of every linear, searched within a byte budget)
  brain optimize BASE --dataset FILE --out DIR [--budget FORMAT | --bits-per-weight B | --max-bytes N]
      Rank every linear and candidate format by its estimated loss increase
      on the dataset, solve for the formats that fit the budget, and accept
      the plan only if it measures better on held-out records, on the
      deployed kernels, than every uniform format that fits. Prints the
      --weight-format policy; `brain optimize --help` for every option.
  brain prune BASE --dataset FILE --out DIR [--keep F | --keep-channels N]
      Remove the least important channels of every prunable dimension,
      ranked on the dataset, and write the narrower model as a checkpoint
      directory; `brain prune --help` for every option.

OTHER
  brain gradcheck                          # finite-difference backprop check (GPT2)
  brain help
  brain --version

EXAMPLES
  brain data gen calculator --out data/calculator --n 100000
  brain gpt2 train --data data/calculator --save out/gpt2.safetensors --steps 2000 --mask =
  brain gpt2 eval --weights out/gpt2.safetensors --data data/calculator
  brain gpt2 generate --weights out/gpt2.safetensors --prompt \"12+7=\" --max-new 8
  brain toymoe train --steps 2000 --save moe.safetensors
  brain toymoe generate --weights moe.safetensors --prompt 1,2,3,4 --max-new 64
  brain federated split moe.safetensors out/shards && brain federated verify out/shards
  brain gradcheck

Or drive everything via the Makefile:  make gpt/calculator/data gpt/calculator/train gpt/calculator/eval
";

/// Extract the two global compute flags from anywhere in the args and select
/// the compute set, returning the remaining args.
///
/// * `--device` (`BRAIN_DEVICE`) - **which hardware** is schedulable:
///   `cpu | gpu | npu | gpu0 | cpu0-7 | gpu1,cpu0-3`, unions of those.
/// * `--backend` - **how that hardware is driven**:
///   `wgpu | vulkan | cuda | cpu`, chosen from the hardware when omitted
///   (`gpu_core::defaults::BACKEND`).
///
/// Two axes, and only the first one picks hardware. `--backend` writes the
/// `backend` field the resolved `ComputeSet` has always carried, leaving the
/// device set alone, so `--device gpu1 --backend vulkan` is expressible and
/// means what it reads as. Every backend is compiled into every build; both
/// flags only choose what each model instantiates at run time.
///
/// `--device vulkan`/`--device wgpu` still parse, as backend-setting device
/// tokens, and are unaffected by this flag. That overlap goes away in a change
/// of its own - `BRAIN_DEVICE=vulkan` is load-bearing in the parity gate and
/// documented as a performance instruction, so it cannot be retired as a side
/// effect of adding `--backend`.
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
    let mut backend_text: Option<String> = None;
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
        } else if argv[i] == "--backend" {
            match argv.get(i + 1) {
                Some(v) => backend_text = Some(v.clone()),
                None => {
                    eprintln!("brain: --backend needs a value (wgpu | vulkan | cuda | cpu)");
                    std::process::exit(2);
                }
            }
            i += 2;
        } else if argv[i] == "--vram-budget" {
            // The VRAM this process was granted, in GiB: every card it budgets
            // from is capped at it (the card helper sets this from the shares it
            // reserved). Set before any device is probed.
            match argv.get(i + 1).and_then(|v| v.parse::<f64>().ok()).filter(|g| *g > 0.0) {
                Some(gib) => gpu_core::capacity::set_vram_budget((gib * (1u64 << 30) as f64) as u64),
                None => {
                    eprintln!("brain: --vram-budget needs a positive number of GiB");
                    std::process::exit(2);
                }
            }
            i += 2;
        } else if argv[i] == "--no-native-kernels" {
            // Every dispatch stays on the generated tier: the A/B switch the
            // native kernels are measured with. Set before any device is built.
            gpu_core::set_native_kernels(false);
            i += 1;
        } else if argv[i] == "--no-kernel-upgrade" {
            // Every model keeps the kernel it registered (no transparent upgrade): the A/B switch and the driver fallback.
            // Set before any device is built.
            gpu_core::set_kernel_upgrade(false);
            i += 1;
        } else if argv[i] == "--no-provider" {
            // `--no-provider <name>[,<name>]` disables compute providers by name (`cuda`, ...): the bisecting switch for a
            // suspected provider-specific bug. Set before any device is built.
            match argv.get(i + 1).filter(|v| !v.trim().is_empty()) {
                Some(v) => gpu_core::provider::set_disabled_providers(&v.split(',').collect::<Vec<_>>()),
                None => {
                    eprintln!("brain: --no-provider needs one or more provider names, e.g. --no-provider cuda");
                    std::process::exit(2);
                }
            }
            i += 2;
        } else if argv[i] == "--no-gdn-fast" {
            // The Gated DeltaNet prefill stays on the portable dispatch sequence: the A/B switch of its cooperative kernels.
            model::gdn::set_fast_kernels(false);
            i += 1;
        } else if argv[i] == "--roof-probe" {
            // Whether the roofline probe runs: `auto` (every backend but the CPU one), `off`, or `on` (the CPU backend too).
            let probe = match argv.get(i + 1).map(String::as_str) {
                Some("auto") => gpu_core::roof::Probe::Auto,
                Some("off") => gpu_core::roof::Probe::Off,
                Some("on") => gpu_core::roof::Probe::On,
                _ => {
                    eprintln!("brain: --roof-probe takes auto, off or on");
                    std::process::exit(2);
                }
            };
            gpu_core::roof::set_probe(probe);
            i += 2;
        } else if argv[i] == "--no-flash-cross" {
            // Text cross-attention stays on the materialized scores/softmax/apply trio: the fused kernel's A/B switch.
            model::block::set_flash_cross(false);
            i += 1;
        } else if argv[i] == "--no-stream-shard" {
            // A streaming checkpoint keeps one stage instead of splitting over the device set's cards (a split is
            // bit-identical; this bisects a driver problem or leaves a card free).
            model::shard::set_stream_shard(false);
            i += 1;
        } else if argv[i] == "--conv-col-mib" {
            // Caps the im2col scratch of every lowered convolution, in MiB.
            match argv.get(i + 1).and_then(|v| v.parse::<u64>().ok()).filter(|&m| m > 0) {
                Some(mib) => gpu_core::lower::set_col_budget_mib(Some(mib)),
                None => {
                    eprintln!("brain: --conv-col-mib needs a positive number of MiB");
                    std::process::exit(2);
                }
            }
            i += 2;
        } else if argv[i] == "--no-coop-ln" {
            // LayerNorm stays on the per-element kernels: the cooperative variant's A/B switch.
            gpu_core::select::set_coop_layernorm(false);
            i += 1;
        } else if argv[i] == "--no-coop-gradnorm" {
            // The optimiser's grad-norm stays single-threaded: the cooperative reduction's A/B switch.
            gpu_core::select::set_coop_gradnorm(false);
            i += 1;
        } else if argv[i] == "--conv1d-gemm" {
            // `auto` (the measured threshold), `off` (the direct 1D convolutions) or `force` (the GEMM lowering below it).
            let pin = match argv.get(i + 1).map(String::as_str) {
                Some("auto") => None,
                Some("off") => Some(false),
                Some("force") => Some(true),
                _ => {
                    eprintln!("brain: --conv1d-gemm takes auto, off or force");
                    std::process::exit(2);
                }
            };
            gpu_core::select::set_conv1d_gemm(pin);
            i += 2;
        } else if argv[i] == "--no-graphs" {
            // Every dispatch is launched on its own instead of captured into graphs and replayed (bisects a capture bug).
            gpu_core::runtime::set_graph_capture(false);
            i += 1;
        } else if argv[i] == "--no-mempool" {
            // Scratch blocks come from the exact-size cache of freed blocks instead of the stream-ordered pool.
            gpu_core::runtime::set_stream_pool(false);
            i += 1;
        } else if argv[i] == "--block-cache-mb" {
            // The most idle MiB of freed device blocks a backend keeps (`0` keeps none); unset sizes it from the card.
            match argv.get(i + 1).and_then(|v| v.parse::<u64>().ok()) {
                Some(mb) => gpu_core::runtime::set_block_cache_bytes(Some(mb << 20)),
                None => {
                    eprintln!("brain: --block-cache-mb needs a number of MiB (0 keeps none)");
                    std::process::exit(2);
                }
            }
            i += 2;
        } else if argv[i] == "--gpu-wait-s" {
            // The longest one wait on a device may take before the device counts as wedged (seconds, finite and positive).
            match argv.get(i + 1).and_then(|v| gpu_core::hardware::parse_wait_seconds(v)) {
                Some(t) => gpu_core::hardware::set_wait_timeout(Some(t)),
                None => {
                    eprintln!("brain: --gpu-wait-s needs a finite, positive number of seconds (a bad value must not lift the bound)");
                    std::process::exit(2);
                }
            }
            i += 2;
        } else if argv[i] == "--max-unsynced-workgroups" {
            // Workgroups a backend may queue before a host-synchronised flush: lower it on a driver that times out under
            // long unsynchronised runs.
            match argv.get(i + 1).and_then(|v| v.parse::<u64>().ok()).filter(|&w| w > 0) {
                Some(w) => gpu_core::hardware::set_max_unsynced_workgroups(Some(w)),
                None => {
                    eprintln!("brain: --max-unsynced-workgroups needs a positive count (0 would serialise every dispatch)");
                    std::process::exit(2);
                }
            }
            i += 2;
        } else if argv[i] == "--cuda-aot-dir" {
            // The directory of ahead-of-time CUDA kernel images (`make cuda/aot`); default `cuda-aot` under the cache.
            match argv.get(i + 1).filter(|v| !v.is_empty()) {
                Some(v) => gpu_core::cuda_aot::set_dir(Some(v.into())),
                None => {
                    eprintln!("brain: --cuda-aot-dir needs a directory");
                    std::process::exit(2);
                }
            }
            i += 2;
        } else if argv[i] == "--no-cuda-aot" {
            // Kernels never look for ahead-of-time images: every one is compiled by NVRTC.
            gpu_core::cuda_aot::set_enabled(false);
            i += 1;
        } else if argv[i] == "--nvrtc" {
            // The exact NVRTC library to load (a path or a SONAME) instead of searching CUDA_PATH and the loader path.
            match argv.get(i + 1).filter(|v| !v.is_empty()) {
                Some(v) => gpu_core::cuda_nvrtc::set_library(Some(v.clone())),
                None => {
                    eprintln!("brain: --nvrtc needs a library path or SONAME");
                    std::process::exit(2);
                }
            }
            i += 2;
        } else if argv[i] == "--cache-dir" {
            // Where derived data persists (pipeline blobs, autotune records, kernel images); default the platform's cache.
            match argv.get(i + 1).filter(|v| !v.is_empty()) {
                Some(v) => gpu_core::set_cache_dir(Some(v.into())),
                None => {
                    eprintln!("brain: --cache-dir needs a directory");
                    std::process::exit(2);
                }
            }
            i += 2;
        } else if argv[i] == "--kernel-profile" {
            // Per-kernel dispatch timing printed at exit, and the host-side stage timings.
            gpu_core::profile::set_profile(true);
            i += 1;
        } else if argv[i] == "--no-autotune" {
            // Every kernel slot keeps its default candidate and the selector its static policy: reproducible benchmarking
            // and A/B runs.
            gpu_core::set_native_autotune(false);
            gpu_core::select::set_autotune(false);
            i += 1;
        } else if argv[i] == "--kernel" {
            // `--kernel <slot>=<candidate>` pins one kernel slot to one of its candidates (repeatable), wherever the
            // candidate can serve the dispatch - the debugging and A/B override of the measured selection.
            match argv.get(i + 1).and_then(|v| v.split_once('=')).filter(|(s, c)| !s.is_empty() && !c.is_empty()) {
                Some((slot, candidate)) => gpu_core::force_candidate(slot, candidate),
                None => {
                    eprintln!("brain: --kernel takes <slot>=<candidate>, e.g. --kernel matmul_i8_gemv=matmul_i8_mma_m16");
                    std::process::exit(2);
                }
            }
            i += 2;
        } else if argv[i] == "--profile-replays" {
            // Passes that support it print a min-of-N-replays kernel table.
            match argv.get(i + 1).and_then(|v| v.parse::<u32>().ok()) {
                Some(n) => gpu_core::profile::set_program_replays(n),
                None => {
                    eprintln!("brain: --profile-replays needs a replay count (a non-negative integer)");
                    std::process::exit(2);
                }
            }
            i += 2;
        } else {
            out.push(argv[i].clone());
            i += 1;
        }
    }

    let backend = backend_text.map(|text| match gpu_core::devices::Backend::parse(&text) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("brain: --backend: {e}");
            std::process::exit(2);
        }
    });

    // An explicit `--device` flag is resolved directly (still hard-exiting on a
    // bad value - the user just typed it). With no flag the CLI runs on every
    // device the machine has: it is configured by its flags alone, never by
    // `BRAIN_DEVICE`/`BRAIN_BACKEND` left in the environment
    // (`gpu_core::devices::resolve_cli_default_compute_set` says so when one is
    // set). `apply()` runs here either way for the CLI-only half
    // `ambient_compute_set()` deliberately skips (rayon pool sizing / CPU
    // affinity - see `ComputeSet::apply`'s doc).
    //
    // One resolution, then one optional `--backend` override on top of it,
    // then publish - in that order, because `--backend` must reach the set
    // BEFORE it is published or the process's ambient set and the flag the
    // user typed disagree (that `OnceLock` is first-writer-wins by design).
    // Publishing is what makes the resolution readable again: callers get it
    // back through `gpu_core::ambient_compute_set()`/`compute_set()`, never
    // from this binding.
    let mut set = match spec_text {
        Some(text) => {
            let spec = match gpu_core::DeviceSpec::parse(&text) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("brain: --device: {e}");
                    std::process::exit(2);
                }
            };
            match spec.resolve(&gpu_core::Inventory::probe()) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("brain: --device: {e}");
                    std::process::exit(2);
                }
            }
        }
        None => gpu_core::devices::resolve_cli_default_compute_set(),
    };

    // An unsatisfiable `--backend` is a hard error, never a quiet demotion to
    // whatever else happens to be present.
    if let Some(b) = backend {
        if let Err(e) = set.set_backend(b) {
            eprintln!("brain: {e}");
            std::process::exit(2);
        }
    }

    if let Err(e) = set.apply() {
        eprintln!("brain: --device: {e}");
        std::process::exit(2);
    }
    // An NVIDIA card driven through the portable backend because CUDA cannot
    // load is a large, silent slowdown: say so once, with the loader's reason.
    if set.gpu_enabled() {
        if let Some(notice) = gpu_core::defaults::Hardware::detected().cuda_fallback_notice(set.backend) {
            eprintln!("brain: warning: {notice}");
        }
    }
    gpu_core::publish_compute_set(set);
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

    let (seed, rest) = crate::args::split_seed(args, "brain bench");
    let args = &rest[..];
    let mut name: Option<&str> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
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
    let (seed, rest) = crate::args::split_seed(args, "brain bench eval");
    let args = &rest[..];
    let mut arch = "gpt2".to_string();
    let mut out: Option<String> = None;
    let mut smoke = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--arch" => {
                i += 1;
                arch = args.get(i).cloned().unwrap_or(arch);
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
    let (seed, rest) = crate::args::split_seed(args, "brain bench scale");
    let args = &rest[..];
    let mut arch = "gpt2".to_string();
    let mut out: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--arch" => {
                i += 1;
                arch = args.get(i).cloned().unwrap_or(arch);
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

/// Extract the global `--autofetch` switch from anywhere in `argv` and
/// return `(present, remaining args)`. Pure - no process-global side effect -
/// so it's testable like [`parse_verbosity`] and [`parse_limits`]; [`main`]
/// is the sole caller of [`apply_autofetch`] with this result.
fn parse_autofetch(argv: Vec<String>) -> (bool, Vec<String>) {
    let mut on = false;
    let mut rest = Vec::with_capacity(argv.len());
    for a in argv {
        if a == "--autofetch" {
            on = true;
        } else {
            rest.push(a);
        }
    }
    (on, rest)
}

/// Publish `--autofetch` into the one variable every fetch site gates on
/// ([`crate::supply::auto_fetch_enabled`]): resolve's weight resolution, the
/// `--model` resolver and `brain serve`'s supplier. Setting the variable
/// (rather than threading a bool through every entry point) is what makes
/// one gate the single source of truth, and an explicitly typed flag
/// outranking an inherited `BRAIN_AUTO_FETCH=0` is the same flag-beats-env
/// rule `--brain-data-dir` applies to `BRAIN_MODELS_DIR`.
fn apply_autofetch(on: bool) {
    if on {
        std::env::set_var("BRAIN_AUTO_FETCH", "1");
    }
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

/// Extract the global `--config <FILE>` flag from anywhere in `argv` and
/// return `(path, remaining args)`. With no flag, the user's own config file
/// is used when one exists: `$XDG_CONFIG_HOME/brain/config.yaml`, else
/// `~/.config/brain/config.yaml`. Pure but for that existence check, so it is
/// testable like [`parse_data_dir`]; [`main`] is the sole caller of
/// [`apply_config`] with this result.
fn parse_config(argv: Vec<String>) -> (Option<String>, Vec<String>) {
    let mut path = None;
    let mut rest = Vec::with_capacity(argv.len());
    let mut i = 0;
    while i < argv.len() {
        if argv[i] == "--config" {
            let Some(value) = argv.get(i + 1) else {
                eprintln!("brain: --config needs a file path");
                std::process::exit(2);
            };
            path = Some(value.clone());
            i += 2;
        } else {
            rest.push(argv[i].clone());
            i += 1;
        }
    }
    let path = path.or_else(|| user_config_path().filter(|p| p.is_file()).map(|p| p.to_string_lossy().into_owned()));
    (path, rest)
}

/// Where the user's own config file lives (it need not exist).
fn user_config_path() -> Option<std::path::PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME").filter(|v| !v.is_empty()).map(std::path::PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".config")))?;
    Some(base.join("brain").join("config.yaml"))
}

/// Variables brain no longer reads, each with the flag (or config-section
/// key) that replaced it. Naming one in a config file is refused, so a file
/// written for an older brain does not silently stop doing what it says.
const RETIRED_VARIABLES: &[Retired] = &[
    Retired { name: "BRAIN_API_KEY", replacement: "serve: {api-key-file: FILE}" },
    Retired { name: "BRAIN_QWEN_WEIGHTS", replacement: "serve: {checkpoint: PATH}" },
    Retired { name: "BRAIN_QWEN_TOKENIZER", replacement: "serve: {tokenizer: PATH}" },
    Retired { name: "BRAIN_RUNTIME_DIR", replacement: "serve: {runtime-dir: DIR}" },
    Retired { name: "BRAIN_CONTROLNET_DIR", replacement: "serve: {pin: [controlnet.control=DIR]}" },
    Retired { name: "BRAIN_T5ENCODER_DIR", replacement: "serve: {pin: [t5encoder.root=DIR]}" },
    Retired { name: "BRAIN_ESRGAN_WEIGHTS", replacement: "serve: {pin: [rrdbnet.weights=FILE]}" },
    Retired { name: "BRAIN_NEMOTRONASR", replacement: "serve: {pin: [nemotronasr.weights=DIR]}" },
    Retired { name: "BRAIN_QWEN3ASR", replacement: "serve: {pin: [qwen3asr.weights=DIR]}" },
    Retired { name: "BRAIN_QWEN3ASR_WINDOW", replacement: "qwen3asr transcribe: {window_secs: S}" },
    Retired { name: "BRAIN_QWEN3ASR_MAXNEW", replacement: "qwen3asr transcribe: {max_new: N}" },
    Retired { name: "BRAIN_GLMDSA_WEIGHTS", replacement: "serve: {pin: [glmdsa.weights=FILE]}" },
    Retired { name: "BRAIN_QWEN35MOE_WEIGHTS", replacement: "serve: {pin: [qwen35moe.weights=FILE]}" },
    Retired { name: "BRAIN_QWEN35MOE_TOKENIZER", replacement: "serve: {pin: [qwen35moe.tokenizer=FILE]}" },
    Retired { name: "BRAIN_QWEN35MOE_CTX", replacement: "serve: {ctx: N}" },
    Retired { name: "BRAIN_QWEN35MOE_MAX_BATCH", replacement: "serve: {max-batch: N}" },
    Retired { name: "BRAIN_QWEN3OMNIMOE_HF_DIR", replacement: "serve: {pin: [qwen3omnimoe.weights=DIR]}" },
    Retired { name: "BRAIN_QWEN3OMNIMOE_INT8_CHECKPOINT", replacement: "serve: {pin: [qwen3omnimoe-int8.weights=FILE]}" },
    Retired { name: "BRAIN_QWEN3OMNIMOE_INT8_TOKENIZER_DIR", replacement: "serve: {pin: [qwen3omnimoe-int8.tokenizer=FILE]}" },
    Retired { name: "BRAIN_GPT2_WEIGHTS", replacement: "serve: {models-dir: DIR} (the model-dir scan serves every GPT checkpoint)" },
    Retired { name: "BRAIN_MOCK", replacement: "serve: {mock: true}" },
    Retired { name: "BRAIN_MOCK_DELAY_MS", replacement: "serve: {mock-delay-ms: MS}" },
    Retired { name: "BRAIN_WM_RENDERER", replacement: "diamond play: {renderer: auto|software|accelerated}" },
];

/// One entry of [`RETIRED_VARIABLES`].
struct Retired {
    name: &'static str,
    replacement: &'static str,
}

/// A subcommand's flags as a config file sets them: `--flag value` pairs (a
/// `true` switch is a bare `--flag`) appended to that subcommand's command
/// line, in file order, for every flag the command line does not name itself.
#[derive(Debug, Default, PartialEq)]
struct ConfigSections(Vec<ConfigSection>);

/// One subcommand's section: the command words (`["qwen3", "infer"]`) and
/// each flag's name with its values (none for a switch).
type ConfigSection = (Vec<String>, Vec<(String, Vec<String>)>);

/// Load `path` and apply what it holds.
///
/// The file is a YAML mapping of two kinds of key:
///
/// * a subcommand path, lowercase (`serve`, `qwen3 infer`): a mapping of that
///   command's long flag names (without `--`) to a value, a list of values
///   (a repeatable flag) or a bool (a switch). Returned as [`ConfigSections`]
///   for [`inject_config_flags`]; a flag on the command line always wins.
/// * a `BRAIN_*` variable name: a setting that is still read from the
///   environment, published into it here. It fills gaps only: a variable the
///   caller already exported always wins. A variable brain no longer reads
///   ([`RETIRED_VARIABLES`]) is refused, naming its replacement.
///
/// Must run before every other `parse_*`/`apply_*` call in [`main`] that
/// reads a `BRAIN_*` variable as its own default (`parse_verbosity` reads
/// `BRAIN_VERBOSE` directly) - so this is `main`'s FIRST call, not merely an
/// early one.
///
/// Hard-exits on a missing/unreadable/unparseable file or a value of the
/// wrong shape: the operator named this file, so silently ignoring a mistake
/// in it would start a run configured differently than they believe.
fn apply_config(path: Option<String>) -> ConfigSections {
    let Some(path) = path else { return ConfigSections::default() };
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("brain: --config {path}: {e}");
        std::process::exit(2);
    });
    parse_config_text(&text).unwrap_or_else(|e| {
        eprintln!("brain: --config {path}: {e}");
        std::process::exit(2);
    })
}

/// The scalar a config value stands for, `None` for an explicit `null`.
fn config_scalar(value: &serde_norway::Value) -> Result<Option<String>, String> {
    Ok(Some(match value {
        serde_norway::Value::String(s) => s.clone(),
        serde_norway::Value::Bool(b) => b.to_string(),
        serde_norway::Value::Number(n) => n.to_string(),
        // An explicit `null` means "leave this unset", not "set to the empty
        // string" - a caller who wants a real empty value writes `""`.
        serde_norway::Value::Null => return Ok(None),
        other => return Err(format!("expected a plain string, number or bool, got {other:?}")),
    }))
}

/// [`apply_config`] on the file's text: publishes the variables, returns the
/// sections.
fn parse_config_text(text: &str) -> Result<ConfigSections, String> {
    let map: std::collections::BTreeMap<String, serde_norway::Value> = serde_norway::from_str(text).map_err(|e| e.to_string())?;
    let mut sections = ConfigSections::default();
    for (key, value) in map {
        if key.starts_with("BRAIN_") {
            if let Some(retired) = RETIRED_VARIABLES.iter().find(|r| r.name == key) {
                return Err(format!("{key} is no longer read; set `{}` instead", retired.replacement));
            }
            if std::env::var_os(&key).is_some() {
                continue; // an inherited/exported value always outranks the file
            }
            if let Some(value) = config_scalar(&value).map_err(|e| format!("{key}: {e}"))? {
                std::env::set_var(key, value);
            }
            continue;
        }
        let command: Vec<String> = key.split_whitespace().map(str::to_string).collect();
        if command.is_empty() || !command.iter().all(|w| w.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')) {
            return Err(format!("{key:?} is neither a BRAIN_* variable nor a subcommand (e.g. `serve`)"));
        }
        let serde_norway::Value::Mapping(flags) = value else {
            return Err(format!("{key}: expected a mapping of flag names to values"));
        };
        let mut parsed = Vec::new();
        for (flag, value) in flags {
            let Some(flag) = flag.as_str().map(str::to_string) else {
                return Err(format!("{key}: a flag name must be a string"));
            };
            if flag.starts_with('-') || flag.is_empty() {
                return Err(format!("{key}: write the flag {flag:?} without its leading dashes"));
            }
            let values = match &value {
                serde_norway::Value::Bool(false) | serde_norway::Value::Null => continue,
                serde_norway::Value::Bool(true) => Vec::new(),
                serde_norway::Value::Sequence(items) => {
                    items.iter().map(|v| config_scalar(v).and_then(|v| v.ok_or_else(|| "null in a list".to_string()))).collect::<Result<_, _>>().map_err(|e| format!("{key}.{flag}: {e}"))?
                }
                v => vec![config_scalar(v).map_err(|e| format!("{key}.{flag}: {e}"))?.unwrap_or_default()],
            };
            parsed.push((flag, values));
        }
        sections.0.push((command, parsed));
    }
    Ok(sections)
}

/// Global flags that take a value and may stand before the subcommand when
/// [`subcommand_index`] runs (the others are stripped before it).
const GLOBAL_VALUE_FLAGS: &[&str] = &["--device", "--backend"];

/// Where the subcommand starts in `argv`: past the program name and any
/// leading global flag with its value.
fn subcommand_index(argv: &[String]) -> usize {
    let mut i = 1;
    while let Some(arg) = argv.get(i) {
        if GLOBAL_VALUE_FLAGS.contains(&arg.as_str()) {
            i += 2;
        } else {
            break;
        }
    }
    i
}

/// Append the flags of the config section that names this command (the
/// longest match, so `qwen3 infer` wins over `qwen3`) for every flag the
/// command line does not already give. A repeatable flag given on the command
/// line replaces the file's list rather than adding to it.
fn inject_config_flags(mut argv: Vec<String>, sections: &ConfigSections) -> Vec<String> {
    let start = subcommand_index(&argv);
    let words = &argv[start.min(argv.len())..];
    let Some((_, flags)) = sections.0.iter().filter(|(command, _)| words.starts_with(command)).max_by_key(|(command, _)| command.len()) else {
        return argv;
    };
    let mut extra = Vec::new();
    for (flag, values) in flags {
        let flag = format!("--{flag}");
        if argv.contains(&flag) {
            continue;
        }
        if values.is_empty() {
            extra.push(flag);
        } else {
            for value in values {
                extra.push(flag.clone());
                extra.push(value.clone());
            }
        }
    }
    argv.extend(extra);
    argv
}

/// The full "about to terminate" sequence every one-shot CLI path that may
/// have touched a real GPU device runs instead of a bare
/// `std::process::exit`, then actually terminates with `code` - never
/// returns.
///
/// See `gpu_core::drain_all_devices`'s own doc for the measured segfault this
/// exists to avoid: `brain qwen3vl generate` already prints its correct
/// answer, then segfaults on exit ~2-4 out of 10 real runs (int8, a real 8B
/// perf-number: hardware count (two cards) and a crash frequency in the bug report, not a timing
/// GGUF checkpoint, 2x Tesla P40/Vulkan) - `gdb` shows the crashing thread
/// has no Rust frames at all, a live NVIDIA driver worker thread killed
/// mid-flight during process termination.
///
/// Two independent mitigations, in order, because the first ALONE measured
/// as not enough: draining every live device to idle (bounded, 5s - generous
/// for a device that is actually idle, since the CLI already blocked on
/// reading its own last result back, and a bounded give-up rather than a
/// hang if one genuinely wedged) confirmed both this run's devices report
/// clean (no error, no timeout) and the process still segfaulted moments
/// later - so the race is not "pending GPU work", and `gpu_core::hard_exit`
/// (raw `_exit`, skipping glibc's own `exit()`/`atexit` sequence entirely,
/// see its own doc) is the second, address the crash directly rather than
/// trying to out-wait it.
pub(crate) fn drain_before_exit(code: i32) -> ! {
    gpu_core::drain_all_devices(std::time::Duration::from_secs(5));
    gpu_core::set_process_exiting();
    gpu_core::hard_exit(code);
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    // First call, deliberately: every other parse_*/apply_* below (and
    // every model/backend it touches) reads a BRAIN_* variable as its own
    // default, so the config file has to be in the environment before any
    // of them run - see apply_config's own doc.
    let (config_path, argv) = parse_config(argv);
    let config_sections = apply_config(config_path);
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
    // Before any subcommand, so every fetch site -- resolve's weight
    // resolution, the `--model` resolver, `brain serve`'s supplier -- reads
    // the same published opt-in.
    let (autofetch, argv) = parse_autofetch(argv);
    apply_autofetch(autofetch);
    if matches!(argv.get(1).map(String::as_str), Some("--version" | "-V")) {
        println!("brain {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    // Before `select_backend` and every other probe: detaching after a Vulkan
    // instance exists produces a daemon that inherits the cached device
    // registry but not the instance behind it, and silently runs every model
    // on a software rasteriser. See `serve_daemon::lifecycle`.
    // The config file's subcommand flags join the command line before
    // anything parses it, lifecycle included (`serve: {runtime-dir: …}`).
    let argv = inject_config_flags(argv, &config_sections);
    let subcommand = subcommand_index(&argv);
    let argv = serve_daemon::lifecycle(argv, subcommand);
    let argv = select_backend(argv);
    // After `--device` is resolved (so the candidate set is exactly what the
    // user made schedulable) and before any model is built. This is what
    // turns `gpu_core::devices`' placement seam from inert into the
    // capacity-aware default: with no `--device`, a model lands on a card
    // that can actually hold it instead of unconditionally on card 0. An
    // explicit selection never consults it.
    placement::install();
    // A one-shot CLI verb (every arm below except `serve`) has nothing else
    // running in the process that needs to survive a panic - unlike
    // `serve`'s own request-isolating `catch_unwind`
    // (`crates/residency/src/executor.rs::lane_loop`, untouched: this hook
    // is never installed for that arm). The default unwind is actively
    // dangerous here instead: it runs every live GPU device's `Drop` while
    // the panicking thread's own driver call may still be on the stack -
    // exactly the teardown race `drain_before_exit` exists to avoid (see its
    // own doc) - and a `wgpu::Error::OutOfMemory` from
    // `device.on_uncaptured_error` (`crates/backend-wgpu/src/lib.rs`) panics
    // by design. Measured production failure: `brain pulid text2image
    // --precision fp32` panicked on a genuine T5-XXL OOM and the unwind
    // segfaulted (exit 139) instead of the clean, non-zero exit this hook
    // produces. The default hook still runs first, so `RUST_BACKTRACE`
    // keeps working exactly as before.
    if argv.get(1).map(String::as_str) != Some("serve") {
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            default_hook(info);
            drain_before_exit(101);
        }));
    }
    match argv.get(1).map(|s| s.as_str()) {
        Some("data") => data_cli::run_data(&argv[2..]),
        Some("devices") => devices_cli::run_devices(&argv[2..]),
        Some("doctor") => std::process::exit(doctor_cli::run_doctor(&argv[2..])),
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
        Some("decide") if decide_cli::handles(argv.get(2).map(String::as_str)) => std::process::exit(decide_cli::run_decide(&argv[2..])),
        Some("caps") => std::process::exit(caps_cli::run_caps(&argv[2..])),
        // `fetch` is an accepted alias, not a second verb: it is the spelling
        // several of this workspace's test-fixture instructions already tell a
        // reader to run, and honouring it costs one arm.
        Some("pull") | Some("fetch") => std::process::exit(pull_cli::run_pull(&argv[2..])),
        Some("models") => std::process::exit(models_cli::run_models(&argv[2..])),
        Some("plan") => std::process::exit(plan_cli::run_plan(&argv[2..])),
        Some("gguf") => std::process::exit(gguf_cli::run_gguf(&argv[2..])),
        Some("roofline") => std::process::exit(roofline_cli::run_roofline(&argv[2..])),
        Some("serve") => run_cli::run_serve(&argv[2..]),
        Some("help") | Some("-h") | Some("--help") | None => print!("{HELP}"),
        Some(_) => resolve::dispatch(&argv[1..], HELP),
    }
    // Every arm above either already called `drain_before_exit` itself (in
    // which case this line is unreachable, which is fine - see the matching
    // call sites in `resolve::dispatch`/`caps_cli::run_do`) or returned normally, about to fall off `main` - in
    // which case THIS is what actually terminates the process (`main`
    // returning normally never reaches `drain_before_exit`'s `hard_exit`
    // unless something calls it, so falling through to a bare return here
    // would skip the fix entirely for every subcommand above that does not
    // exit itself, e.g. `data`/`npu`/`federated`/`bench`).
    drain_before_exit(0);
}

#[cfg(test)]
mod tests {
    use super::{apply_autofetch, apply_config, inject_config_flags, parse_autofetch, parse_config, parse_config_text, parse_data_dir, parse_limits, parse_verbosity, ConfigSections, HELP, RETIRED_VARIABLES};
    use brain_testutil::env_lock;

    /// The opt-in fetch flag is stripped from anywhere in `argv`, consumes
    /// no value (it is a switch, not a dial), and leaves everything else
    /// untouched -- including the default path, where it must be a complete
    /// no-op.
    #[test]
    fn parse_autofetch_strips_the_flag_from_anywhere() {
        let (on, rest) = parse_autofetch(["brain", "--autofetch", "flux2", "generate"].map(String::from).to_vec());
        assert!(on);
        assert_eq!(rest, ["brain", "flux2", "generate"].map(String::from).to_vec());

        let (on, rest) = parse_autofetch(["brain", "serve", "--autofetch", "--openai", "8787"].map(String::from).to_vec());
        assert!(on);
        assert_eq!(rest, ["brain", "serve", "--openai", "8787"].map(String::from).to_vec());

        let (on, rest) = parse_autofetch(["brain", "caps"].map(String::from).to_vec());
        assert!(!on, "absent must be off");
        assert_eq!(rest, ["brain", "caps"].map(String::from).to_vec());
    }

    /// Publishing is the flag setting the one variable every fetch site
    /// gates on; an explicitly typed flag outranks an inherited opt-out,
    /// the same flag-beats-env rule `--brain-data-dir` applies.
    #[test]
    fn apply_autofetch_publishes_the_opt_in_over_an_inherited_opt_out() {
        let _serial = env_lock();
        std::env::set_var("BRAIN_AUTO_FETCH", "0");
        apply_autofetch(false);
        assert_eq!(std::env::var("BRAIN_AUTO_FETCH").as_deref(), Ok("0"), "no flag must not touch the environment");
        apply_autofetch(true);
        assert_eq!(std::env::var("BRAIN_AUTO_FETCH").as_deref(), Ok("1"), "the flag outranks an inherited =0");
        std::env::remove_var("BRAIN_AUTO_FETCH");
    }

    /// The same two-directional discipline the memory ceilings and the
    /// data-dir flag get: documented AND stripped.
    #[test]
    fn the_autofetch_flag_is_documented_and_stripped() {
        assert!(HELP.contains("--autofetch"), "--autofetch is implemented but absent from `brain help`");
        let (on, rest) = parse_autofetch(["brain".to_string(), "--autofetch".to_string(), "caps".to_string()].to_vec());
        assert!(on);
        assert_eq!(rest, ["brain", "caps"].map(String::from).to_vec(), "`brain help` documents --autofetch, which parse_autofetch does not strip");
    }

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

    /// `--config` strips from anywhere and consumes its value; with no flag
    /// the user's own file is read when it exists, and `$BRAIN_CONFIG` names
    /// nothing.
    #[test]
    fn parse_config_strips_the_flag_and_falls_back_to_the_user_file() {
        let argv = ["brain", "serve", "--config", "/etc/brain/config.yaml", "--openai"].map(String::from).to_vec();
        let (path, rest) = parse_config(argv);
        assert_eq!(path.as_deref(), Some("/etc/brain/config.yaml"));
        assert_eq!(rest, ["brain", "serve", "--openai"].map(String::from).to_vec());

        let _serial = brain_testutil::env_lock();
        let home = std::env::temp_dir().join(format!("brain-config-home-{}", std::process::id()));
        std::fs::create_dir_all(home.join("brain")).unwrap();
        let saved = std::env::var_os("XDG_CONFIG_HOME");
        std::env::set_var("XDG_CONFIG_HOME", &home);
        // The variable that once named the config file, set to show it names nothing.
        const RETIRED_CONFIG_VARIABLE: &str = "BRAIN_CONFIG";
        std::env::set_var(RETIRED_CONFIG_VARIABLE, "/from/env.yaml");
        let (path, _) = parse_config(["brain", "caps"].map(String::from).to_vec());
        assert_eq!(path, None, "no user file and no flag: nothing, whatever $BRAIN_CONFIG says");
        std::fs::write(home.join("brain/config.yaml"), "serve: {}\n").unwrap();
        let (path, rest) = parse_config(["brain", "caps"].map(String::from).to_vec());
        assert_eq!(path.map(std::path::PathBuf::from), Some(home.join("brain/config.yaml")));
        assert_eq!(rest, ["brain", "caps"].map(String::from).to_vec());
        std::env::remove_var(RETIRED_CONFIG_VARIABLE);
        match saved {
            Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
        std::fs::remove_dir_all(&home).ok();
    }

    /// A subcommand section supplies that command's flags, after the command
    /// line's own and never in place of one it gives; the longest matching
    /// section wins, and a leading global flag does not hide the subcommand.
    #[test]
    fn a_config_section_supplies_the_flags_its_command_line_does_not_give() {
        let sections = parse_config_text("serve:\n  api-key-file: /k\n  openai: 9000\n  autofetch: true\n  stdio: false\n  decision-model: [a, b]\nqwen3:\n  seed: 1\nqwen3 infer:\n  seed: 2\n").unwrap();
        let argv = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            inject_config_flags(argv(&["brain", "--device", "cpu", "serve", "--openai", "8000"]), &sections),
            argv(&["brain", "--device", "cpu", "serve", "--openai", "8000", "--api-key-file", "/k", "--autofetch", "--decision-model", "a", "--decision-model", "b"])
        );
        assert_eq!(inject_config_flags(argv(&["brain", "qwen3", "infer"]), &sections), argv(&["brain", "qwen3", "infer", "--seed", "2"]));
        assert_eq!(inject_config_flags(argv(&["brain", "caps"]), &sections), argv(&["brain", "caps"]));
    }

    /// A retired variable, a key that is neither a variable nor a command, or
    /// a flag written with its dashes is a mistake the file's author made:
    /// refused, naming what to write instead.
    #[test]
    fn a_config_file_written_for_a_retired_variable_is_refused_naming_its_replacement() {
        let retired = RETIRED_VARIABLES[1].name;
        let e = parse_config_text(&format!("{retired}: /w\n")).unwrap_err();
        assert!(e.contains(retired) && e.contains("checkpoint"), "{e}");
        let e = parse_config_text("BRAIN_API_KEY: sk\n").unwrap_err();
        assert!(e.contains("api-key-file"), "{e}");
        assert!(parse_config_text("Serve: {}\n").is_err());
        assert!(parse_config_text("serve:\n  --openai: 1\n").unwrap_err().contains("dashes"));
        assert!(parse_config_text("serve: 3\n").is_err());
    }

    /// The file fills gaps in the environment but never overrides an
    /// already-exported variable - the core precedence guarantee
    /// [`apply_config`]'s own doc promises.
    #[test]
    fn apply_config_fills_gaps_but_never_overrides_an_inherited_value() {
        let _serial = brain_testutil::env_lock();
        std::env::remove_var("BRAIN_TEST_CONFIG_FILLED");
        std::env::set_var("BRAIN_TEST_CONFIG_INHERITED", "from-shell");

        let dir = std::env::temp_dir().join(format!("brain-config-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.yaml");
        std::fs::write(&path, "BRAIN_TEST_CONFIG_FILLED: from-file\nBRAIN_TEST_CONFIG_INHERITED: from-file\nBRAIN_TEST_CONFIG_NUMBER: 16\nBRAIN_TEST_CONFIG_BOOL: true\n").unwrap();

        assert_eq!(apply_config(Some(path.to_string_lossy().into_owned())), ConfigSections::default());

        assert_eq!(std::env::var("BRAIN_TEST_CONFIG_FILLED").as_deref(), Ok("from-file"), "an unset variable must be filled from the file");
        assert_eq!(std::env::var("BRAIN_TEST_CONFIG_INHERITED").as_deref(), Ok("from-shell"), "an inherited value must outrank the file");
        assert_eq!(std::env::var("BRAIN_TEST_CONFIG_NUMBER").as_deref(), Ok("16"));
        assert_eq!(std::env::var("BRAIN_TEST_CONFIG_BOOL").as_deref(), Ok("true"));

        std::env::remove_var("BRAIN_TEST_CONFIG_FILLED");
        std::env::remove_var("BRAIN_TEST_CONFIG_INHERITED");
        std::env::remove_var("BRAIN_TEST_CONFIG_NUMBER");
        std::env::remove_var("BRAIN_TEST_CONFIG_BOOL");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Same documented-and-stripped discipline every other global flag gets.
    #[test]
    fn the_config_flag_is_documented_and_stripped() {
        assert!(HELP.contains("--config"), "--config is implemented but absent from `brain help`");
        assert!(HELP.contains("~/.config/brain/config.yaml"), "the default config file is absent from `brain help`");
        let (path, rest) = parse_config(["brain".to_string(), "--config".to_string(), "/x.yaml".to_string(), "caps".to_string()].to_vec());
        assert_eq!(rest, ["brain", "caps"].map(String::from).to_vec(), "`brain help` documents --config, which parse_config does not strip");
        assert_eq!(path.as_deref(), Some("/x.yaml"));
    }

    /// `brain plan` must be reachable and documented -- see the `pull` test below for why
    /// both halves matter. The exit-code contract is documented too: it is the whole reason
    /// a script would call this rather than reading `stats_snapshot` and guessing.
    #[test]
    fn the_plan_verb_is_documented_including_its_exit_code_contract() {
        assert!(HELP.contains("brain plan <model> <action>"), "`brain plan` missing from `brain help`");
        assert!(HELP.contains("Exits 0 when it would run"), "`brain help` does not document the exit code a script branches on");
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
