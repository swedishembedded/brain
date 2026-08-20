# The CLI

Every capability of brain is reachable through the single `brain` binary,
using one grammar for every architecture:

```
brain <verb> <architecture> [options]
brain <architecture> <verb> [options]
```

Both orders are the SAME command - `brain train gpt2 ...` and `brain gpt2
train ...` dispatch identically. There is no separate alias table to drift
out of sync: one resolver (`crates/cli/src/resolve.rs`) reads both orders
against the one canonical architecture registry (`crates/arch`).

## Discovering what's there

### `brain caps`

Lists every architecture and its action manifest - what it can do, and what
parameters each action takes:

```bash
brain caps                 # every architecture brain knows about
brain caps <architecture>  # one architecture's manifest in detail
```

This is the discovery mechanism: it reflects an architecture's real,
declared capabilities, not documentation that can drift out of sync with the
code. Toy architectures (`toymoe`, `toypid`, `toyseq2seq`,
`toyautoencoder` - brain's own tasks, no upstream reference) are excluded
from this listing; they're real, gradient-checked, and reachable by name,
just not part of the public model surface.

### Standard verbs

Every architecture that supports it accepts these, regardless of whether it
also has arch-specific long-tail flags:

| Verb | What |
|---|---|
| `infer` | run a forward pass - the canonical inference verb. Auto-fetches default weights when none are given, for architectures with a known small default checkpoint (`brain caps <id>` shows whether one exists). |
| `train` / `finetune` / `eval` | training lifecycle |
| `bench` | the architecture's own benchmark |
| `import` / `export` | one-time checkpoint conversion |

An architecture with no dedicated CLI module dispatches the action name
directly as the verb - `brain caps` prints the exact list. For example:

```bash
brain scrfd detect --in image=photo.ppm --json
brain infer scrfd --in image=photo.ppm --json     # identical, verb-first order
```

Architectures with their own dedicated flags and a longer verb vocabulary
(`gpt2`, `qwen3`, `qwen35moe`, `glmdsa`, `lfm2`, `qwen3tts`, `yolov8`,
`zipdepth`, `flux2`, `worldmirror2`, `splat`, `qwen3omnimoe`, `diamond`,
`toypid`, `toymoe`) are documented in full by `brain help` and
`brain <architecture> --help`; every other architecture (`brain caps` lists
them all - `s3dit`, `fastvlm`, `qwen3vl`, `sam1`, `sam2`, `scrfd`, `arcface`,
`vqgan`, `codeformer`, `rrdbnet`, `clip`, `t5encoder`, `sdxlunet`,
`controlnet`, `pulid`, `instantid`, `autoencoderkl`, `deepseek2ocr`,
`nemotronasr`, `qwen3asr`, `mimi`, `ecapatdnn`, `chronos2`, `fincast`,
`kronos`, `genieredux`, ...) is reached the same uniform way.

## Infrastructure verbs (unchanged, not per-architecture)

| Command | Purpose |
|---|---|
| `brain devices` | canonical GPU table (index, PCI bus, UUID, VRAM) + ambient device selection - see [Hardware](../introduction/hardware.md) |
| `brain data` | dataset generation and tokenizers |
| `brain caps` | every architecture's action manifest - see above |
| `brain serve` | serve models over HTTP/D-Bus, or (with `--stdio`) run the event-driven stdio controller - see below |
| `brain perf` | performance benchmarking: latency/throughput/serve/sweep, vs. a baseline |
| `brain flops` | offline/online FLOP and int-OPS accounting for a forward/backward pass |
| `brain gradcheck` | finite-difference backprop correctness gate |
| `brain npu` | OpenVINO/NPU: `export`, `quantize`, `check`, `run`, `bench`, `sim` |
| `brain federated` | sharded MoE: `split`, `verify`, `merge`, `assemble`, `train-expert` |
| `brain bench` | cross-architecture evaluation harness: `eval`, `scale`, `advise`, `compare` |
| `brain import FILE` | GGUF import with no architecture token - dispatches on the file's own `general.architecture` header instead of the command line |
| `brain quantize SRC` | the export direction: any safetensors/GGUF checkpoint to a quantized GGUF, with no per-architecture code at all |

Run `brain <cmd> --help` (or `brain <architecture> --help`) for any
subcommand's full flag list - `brain help` and this page are the map, not the
exhaustive reference.

## What used to be here and isn't anymore

This CLI had a hard break, with no aliases kept, from an earlier one-command-
per-model-port shape:

- `brain do <model> <action>` is now `brain infer <architecture> ...` (or any
  other verb - `do` always meant "dispatch the generic typed action", which
  is now just what every architecture's own verb does).
- `brain run` (the event-driven stdio controller) is now `brain serve
  --stdio`.
- `brain import-gguf` is now `brain import`.
- `brain pid ...` is now `brain toypid ...`.
- `brain capabilities` is now `brain caps`.
- The bare `brain train|eval|generate` (which meant the sparse-MoE toy task)
  is now namespaced: `brain train toymoe`, `brain toymoe eval`, etc.

## Serving

`brain serve` covers two distinct things depending on the flags given:

- With a surface flag - `brain serve [--openai [PORT]] [--anthropic [PORT]]
  [--openrouter [PORT]] [--dbus]` - it makes configured models resident and
  answers requests over one or more transports at once. See
  [Serving](serving.md) for the full flag reference, access control, and
  admission behavior.
- With `--stdio`, it instead runs an event-driven stdio controller: it reads
  JSONL events on stdin (e.g. `user_text`, `camera_frame`) and streams JSONL
  events back on stdout (e.g. `brain_text_chunk`, `object_detected`), driven
  by a `--gpt`/`--yolo` checkpoint if you give one.

## See also

- [Models & weights](models-and-weights.md) - model ids, auto-fetch, and
  local checkpoints.
- [Configuration](configuration.md) - every `BRAIN_*` environment variable.
- [Hardware](../introduction/hardware.md) - device selection (`--device`,
  `brain devices`).
