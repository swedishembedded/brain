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
`kronos`, `minimaxmusic3`, `genieredux`, ...) is reached the same uniform
way.

## Infrastructure verbs (unchanged, not per-architecture)

| Command | Purpose |
|---|---|
| `brain devices` | canonical GPU table (index, PCI bus, UUID, VRAM) + ambient device selection - see [Hardware](../introduction/hardware.md) |
| `brain roofline` | measured, cross-accelerator hardware compute-capacity report: every GPU, the NPU, and the CPU, each dtype it supports, streamed as each finishes |
| `brain data` | dataset generation and tokenizers |
| `brain label` | caption a dataset with any vision-language model - see [Labeling a dataset](../training/labeling.md) |
| `brain caps` | every architecture's action manifest - see above |
| `brain serve` | serve models over HTTP/D-Bus, or (with `--stdio`) run the event-driven stdio controller - see below |
| `brain perf` | performance benchmarking: latency/throughput/serve/sweep, vs. a baseline |
| `brain flops` | offline/online FLOP and int-OPS accounting for a forward/backward pass, or for a whole image/video generation stage by stage (`--model flux2\|ltxv`) |
| `brain gradcheck` | finite-difference backprop correctness gate |
| `brain npu` | OpenVINO/NPU: `export`, `quantize`, `check`, `run`, `bench`, `sim` |
| `brain federated` | sharded MoE: `split`, `verify`, `merge`, `assemble`, `train-expert` |
| `brain bench` | cross-architecture evaluation harness: `eval`, `scale`, `advise`, `compare` |
| `brain pull <model>` | fetch a model's official weights into the model store, by canonical id or HuggingFace URL - see below |
| `brain models list` | architecture → provider repo → quantization: what's local, what's declared-but-not-pulled, size/fit/cost - see [Seeing what you have](models-and-weights.md#seeing-what-you-have) |
| `brain models list-adapters` | architecture → base variant → LoRA adapter, with rank/alpha/dataset from the adapter's own card |
| `brain models info <model>` | one checkpoint's real tensor tree - name, dtype, shape, size; adapter tensors merged in |
| `brain models profile <model>` | price one already-pulled model now and cache the result - errors, never fetches, if it isn't pulled |
| `brain import FILE` | GGUF import with no architecture token - dispatches on the file's own `general.architecture` header instead of the command line |
| `brain quantize SRC` | the export direction: any safetensors/GGUF checkpoint to a quantized GGUF, with no per-architecture code at all |

`models`, `flops`, `perf`, `bench`, `devices` and `roofline` sound similar but
answer different questions, and none of them subsumes another: `models` is
what you HAVE (an inventory, cache-only, fast even over a large store);
`flops` prices ONE model's forward/backward pass in detail, on demand (and -
naming a real checkpoint via `--weights` - feeds the very cache `models
list`/`models profile` read, so the two can never disagree); `perf` MEASURES
real latency/throughput against a regression baseline, correctness-gated;
`bench` asks whether an architecture learns at all, no hardware axis;
`devices` is just the GPU table, no timing; `roofline` answers "what can this
hardware do" - raw, model-INDEPENDENT compute capacity (GFLOP/s, GOP/s,
GB/s) for every accelerator on the box - which is the number `flops`' own
"roofs" line and `perf`'s utilisation figures are both graded against.
`models`'s size/cost columns read `devices`' device data and `flops`'s
pricing engine directly - never a second copy of either.

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

## Pulling weights

`brain pull <model>` fetches a model's official weights into the model store
and makes them servable, as an explicit up-front step rather than the
on-demand fetch a first `infer`/serve request makes when `--autofetch` is
passed. It is the same operation, spelled out loud - one plan, one download,
one finish step.

The argument is the canonical reference or a HuggingFace URL, with or without
a scheme, a `www.` host, a trailing slash, a query string or a fragment. A
repo page and a `/tree/<revision>` branch view both name the whole repo, at
the revision the URL names:

```bash
brain pull Qwen/Qwen3-0.6B
brain pull https://huggingface.co/Qwen/Qwen3-0.6B
brain pull https://huggingface.co/Qwen/Qwen3-0.6B/tree/main
```

Anything that is not a model reference - a dataset or space URL, a link to
another site, a name with no vendor - is refused by name rather than sent to
the hub as a repo id. `brain fetch` is an accepted alias for the same verb.

### Pulling one file

A `/blob/<revision>/<path>` or `/resolve/<revision>/<path>` URL - the two
spellings of a file's page, one from the address bar and one from the download
button - pulls **exactly that one file**, whatever its extension and from
whatever revision the URL names. Nothing is inferred, because the file is
named. The command prints the path the file landed at, which is what a flag
like `brain flux2 generate --text-encoder <path>` is then pointed at:

```bash
brain pull https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/blob/main/flux-2-klein-9b-Q8_0.gguf
```

A URL that names neither the whole repo nor one file - a subdirectory view, a
`/commits/` or `/discussions/` page, a `/blob/<rev>/` with no filename - is
refused by name. Pulling the whole repo because a directory was named would be
doing something adjacent to what was asked, which is worse than an error.

### Pulling one quantization from a GGUF repo

A `*-GGUF` repo publishes many quantizations of the *same* model - fifteen of
them, over 100 GB in total, for `unsloth/FLUX.2-klein-9B-GGUF`. Exactly one is
ever fetched. Name it with the reference grammar's own quantization suffix:

```bash
brain pull unsloth/FLUX.2-klein-9B-GGUF-Q4_K_M
```

Name none, and brain picks the highest-fidelity quantization the repo offers
(`Q8_0` whenever it is published) **and prints the choice** along with how to
ask for a different one. Asking for a quantization the repo does not publish
fails with the list of the ones it does, rather than falling back to
downloading a base checkpoint to quantize locally.

Either spelling lands the file as `<QUANT>.gguf` in the repo's store
directory, so pulling `<repo>-Q8_0` and pulling that same file by its URL are
one artifact in one place, not two copies.

Re-running a pull is cheap for what already landed: a file already complete in
the store is not fetched again, and a pull of a model that is already complete
does no network I/O at all and says so.

**Resume is per file, not per byte.** A transfer interrupted part-way through a
file restarts *that file* from the beginning - the partial download is
discarded, not continued. For a sharded checkpoint that costs one shard. For a
GGUF repo, where the whole artifact is a single multi-gigabyte file, it costs
the entire transfer, so prefer a connection you can leave alone.

### Progress

Progress is written to **stdout**, and the shape depends on whether stdout is
a terminal:

- **On a terminal** - one line, redrawn in place, showing the bar, the
  fraction complete, throughput and ETA for the whole pull.
- **Piped or redirected** - ten plain lines for the whole pull, bracketed by
  a "need N in M files" header and a completion line. The budget is spent
  over the *total bytes of every file in the plan*, so a model that ships as
  six shards still costs ten lines, not sixty. No carriage returns, no escape
  sequences, one greppable fact per line.

### Where the weights land

`--brain-data-dir <DIR>` sets brain's data root; models live in
`<DIR>/models`. It is a **global** option, valid on any subcommand, because
`pull`, auto-fetch and `brain serve`'s catalog scan all have to agree on where
models are - see [Configuration](configuration.md#paths) for the full
precedence ladder.

### Naming weights on a model command: `--model`

Model commands grow the same `--model` argument for their primary weights
(the DiT, for an image or video model) instead of a per-command flag:
`brain flux2 generate --model <ARG>` overrides `BRAIN_FLUX2_DIT`. The
command prints the model it is about to load before anything is loaded.

`ARG` is read against one ladder, whichever way you mean it:

- **An explicit `.gguf`/`.safetensors` extension** names the file outright.
  It is taken literally - a missing file is an error, nothing is probed or
  fetched.
- **Without an extension**, `<ARG>.gguf` then `<ARG>.safetensors` are probed
  beside the path.
- **A `<vendor>/<repo>[-<QUANT>]` id** resolves through the model store: a
  local copy wins; otherwise the model is announced and downloaded, with
  per-file progress. A compound checkpoint (a diffusers pipeline) hands the
  command its named role. For `flux2`, the remaining components - VAE, text
  encoder, tokenizer - still come from their `BRAIN_FLUX2_*` variables, so
  mix sources with that in mind.

Anything else is refused with what was probed and what a model id looks
like, rather than silently fetching something adjacent.

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
