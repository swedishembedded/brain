# The CLI

Every capability of brain is reachable through the single `brain` binary.
Most models get their own dedicated subcommand (`brain qwen`, `brain yolo`,
…), and every model — regardless of whether it has a dedicated subcommand —
is also reachable through two general-purpose entry points, `brain caps` and
`brain do`, which are what make brain's model surface uniform.

## The uniform entry points

### `brain caps`

Lists every currently loaded model and its action manifest — what it can do,
and what parameters each action takes:

```bash
brain caps                 # every loaded model
brain caps <model>         # one model's manifest in detail
```

This is the discovery mechanism: it reflects a model's real, declared
capabilities, not documentation that can drift out of sync with the code.

### `brain do <model> <action> [--flag value ...]`

Runs any action on any model through one uniform invocation, regardless of
whether that model also has its own dedicated subcommand:

```bash
brain do brain/upscale upscale --in image=photo.ppm --out image=big.ppm
brain do brain/facenet detect --in image=photo.ppm --json
```

`brain do` dispatches through the same generic `(model, action)` mechanism
that HTTP and D-Bus use internally — so what you can do with `brain do`, you
can also do over the network, with the same parameters and the same
semantics. Note that `brain do` only reaches models already registered
locally by name; unlike the HTTP/D-Bus serving surfaces, it does not
auto-fetch a model's weights for you (see
[Models & weights](models-and-weights.md)).

## Top-level subcommands

| Command | Purpose |
|---|---|
| `brain data` | dataset generation and tokenizers |
| `brain devices` | canonical GPU table (index, PCI bus, UUID, VRAM) + ambient device selection — see [Hardware](../introduction/hardware.md) |
| `brain gpt` | GPT decoder: `train`, `gen`, `eval` |
| `brain qwen` | Qwen3 LLM: `import`, `infer`, `export`, `precompile`, `train`, `finetune` |
| `brain glm` | GLM-5.2 decoder: `train`, `finetune`, `infer`, `eval`, `import`, `export` |
| `brain lfm` | LFM2.5-Encoder: `import`, `fill-mask`, `embed`, `data`, `finetune`, `eval` |
| `brain tts` | Qwen3-TTS: `import`, `clone`, `synth`, `design`, `serve`, `sim`, `finetune` |
| `brain yolo` | YOLOv8 detector: `train`, `fine-tune`, `eval`, `detect` |
| `brain depth` | ZipDepth monocular depth: `image`, `camera`, `calib`, `train` |
| `brain flux2` | FLUX.2 Klein text-to-image + editing: `generate` |
| `brain mirror` | WorldMirror-2 3D reconstruction: `import`, `infer`, `demo` |
| `brain splat` | 3D Gaussian Splatting: `info`, `render`, `view`, `fit` |
| `brain wm` | playable world models (DIAMOND): `play`, `replay`, `bench`, `finetune` |
| `brain forecast` | Chronos-2 / Kronos / FinCast forecasting: `compare`, `serve`, `import`, `finetune` |
| `brain npu` | OpenVINO/NPU: `export`, `quantize`, `check`, `run`, `bench`, `sim` |
| `brain federated` | sharded MoE: `split`, `verify`, `merge`, `assemble`, `train-expert` |
| `brain pid` | PID control transformer |
| `brain bench` | architecture-evaluation harness: `eval`, `scale`, `advise`, `compare` |
| `brain perf` | performance benchmarking: latency/throughput/serve/sweep, vs. a baseline |
| `brain flops` | offline/online FLOP and int-OPS accounting for a forward/backward pass |
| `brain import-gguf` | one-time GGUF → brain-native conversion, dispatched by the file's `general.architecture` |
| `brain caps` | every model's action manifest — see above |
| `brain do` | run one action on one model, uniformly — see above |
| `brain gradcheck` | finite-difference backprop correctness gate |
| `brain serve` (alias `brain run`) | serve models over HTTP/D-Bus, or run the event-driven stdio controller — see below |

Run `brain <cmd> --help` for any subcommand's full flag list — this table is
the map, not the exhaustive reference.

## Serving

`brain serve` and `brain run` are the same command under two names.

- With a surface flag — `brain serve [--openai [PORT]] [--anthropic [PORT]]
  [--openrouter [PORT]] [--dbus]` — it makes configured models resident and
  answers requests over one or more transports at once. See
  [Serving](serving.md) for the full flag reference, access control, and
  admission behavior.
- With no surface flag, it instead runs an event-driven stdio controller: it
  reads JSONL events on stdin (e.g. `user_text`, `camera_frame`) and streams
  JSONL events back on stdout (e.g. `brain_text_chunk`, `object_detected`),
  driven by a `--gpt`/`--yolo` checkpoint if you give one.

## See also

- [Models & weights](models-and-weights.md) — model ids, auto-fetch, and
  local checkpoints.
- [Configuration](configuration.md) — every `BRAIN_*` environment variable.
- [Hardware](../introduction/hardware.md) — device selection (`--device`,
  `brain devices`).
