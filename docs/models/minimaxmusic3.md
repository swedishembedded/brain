# MiniMax Music 3 (lyrics + caption-conditioned music generation)

Give it lyrics and a structured music description (genre, BPM, vocal
timbre, instrumentation, arrangement) and it generates a full song - up to
five minutes, 44.1 kHz stereo, with vocals, evolving arrangement and a
coherent structure (intro/verse/chorus/bridge/outro). Reach for it when you
need a first-draft song from a text brief, not for editing or remixing
existing audio (there is no audio-conditioned path).

**Status: fully wired, unvalidated end-to-end on this machine.** All five
components - condition encoder, vocoder, RVQ depth decoder, the
flow-matching DiT, and the Global LLM (a real Qwen3-8B, reused verbatim
from `crates/qwen3`) - have import + forward, each verified at real
weights (cosine 1.0 vs the reference; the Global LLM via a single
streamed real decoder layer, not whole-model residency - see below). The
vocoder, depth decoder, and DiT are also fully trainable (full fine-tune
with gradcheck, LoRA); the DiT additionally has an INT8 storage tier and
`model::Shardable` pipeline sharding (single-device validated only - no
discrete GPU on the machine this was built on). The full pipeline -
prompt assembly, the CFG-guided AR sampling loop, chunked DiT denoising,
vocoder crop-and-stitch - is implemented, unit/structurally tested, and
wired all the way through the CLI/D-Bus serving contract (`brain
minimaxmusic3 generate`), but a real, real-checkpoint end-to-end run
could **not** be validated on this development machine: whole-8B-model
residency exceeds what either of this machine's backends can hold (see
"Hardware and limits" below) - a measured, diagnosed gap, not an
unwritten feature. See this repo's own roadmap ledger
(`.agents/roadmap/minimaxmusic3.md`) for the live milestone history.

## Support

| Capability | Supported |
|---|---|
| Inference              | [x] |
| Training from scratch  | [x] (per-component, library-level - vocoder/depth decoder/DiT; no CLI verb) |
| LoRA fine-tune         | [x] (per-component, library-level; no CLI verb) |
| INT8                   | [x] (DiT storage tier; the Global LLM's own int8 path does not actually shrink memory on this machine's CPU backend - see "Hardware and limits") |
| CLI (`brain minimaxmusic3 generate`) | [x] |
| HTTP API               | [ ] (D-Bus and the event API are wired; a dedicated HTTP route is not) |
| D-Bus                  | [x] |
| Batched serving        | [ ] (single-sequence AR decode only - `crates/qwen3`'s own decode API is `b=1`) |
| Multi-device sharding  | [x] (DiT `model::Shardable`, single-device validated only) |
| NPU                    | [ ] |

## Getting the weights

Model id: `brain/minimaxmusic3`. Five components, one role each - no
combined single-file checkpoint exists upstream, so each is its own env
var:

| Env var | Role |
|---|---|
| `BRAIN_MINIMAXMUSIC3_LM` | Global LLM (Qwen3-8B architecture) |
| `BRAIN_MINIMAXMUSIC3_DEPTH` | RVQ depth decoder |
| `BRAIN_MINIMAXMUSIC3_CONDITION` | Condition encoder |
| `BRAIN_MINIMAXMUSIC3_DIT` | Flow-matching DiT |
| `BRAIN_MINIMAXMUSIC3_VOCODER` | Vocoder |
| `BRAIN_MINIMAXMUSIC3_TOKENIZER` | Tokenizer (Qwen2Tokenizer-compatible) |

## Running it

```bash
BRAIN_MINIMAXMUSIC3_LM=/path/to/language_model \
BRAIN_MINIMAXMUSIC3_DEPTH=/path/to/rvq_depth_decoder \
BRAIN_MINIMAXMUSIC3_CONDITION=/path/to/condition_encoder \
BRAIN_MINIMAXMUSIC3_DIT=/path/to/transformer \
BRAIN_MINIMAXMUSIC3_VOCODER=/path/to/vocoder \
BRAIN_MINIMAXMUSIC3_TOKENIZER=/path/to/qwen3-8B-tokenizer-music \
BRAIN_DEVICE=cpu \
brain minimaxmusic3 generate \
    --lyrics "$(printf '[verse]\nquiet morning light\n[chorus]\nhold on to this feeling\n')" \
    --caption "warm acoustic ballad, gentle piano, soft vocals, 80 BPM" \
    --duration_seconds 10 \
    --out audio=song.wav
```

Or over D-Bus: see `examples/musicgen/generate_song.py` and its own
README for the full `dbus-run-session` invocation.

`BRAIN_DEVICE=cpu` matters on a machine whose GPU cannot hold the Global
LLM's ~3.28 GB embedding/`lm_head` tensors as single buffers (an Intel
integrated GPU, for instance).

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--lyrics` | (required) | song lyrics, with `[verse]`/`[chorus]`/etc structural tags |
| `--caption` | (required) | structured music description: genre, BPM, vocal timbre, instrumentation, arrangement |
| `--duration_seconds` | `10` | target song length in seconds (the AR stage may stop earlier if it samples the end token) |
| `--num_inference_steps` | `30` | Euler steps per denoise chunk |
| `--seed` | `0` | RNG seed (reproducible run) |

## Hardware and limits

Whole-8B-model residency (the Global LLM) does not fit on the machine
this port was built on, on either of its backends: the CPU-JIT backend's
`int8` request silently promotes to fp32 (no backend in this workspace
executes real int8 compute yet - `backend-cpu`'s own
`caps().numeric.int8_dot` is `false`), and this machine's actual GPU (an
Intel integrated Vulkan device, not a discrete card) caps single buffers
at 2047 MiB, below the ~3.28 GB embedding/`lm_head` tensors regardless of
dtype. Measured directly (a single Global LLM instance OOM-kills on this
machine's ~26 GB available RAM). Neither is a defect in this port - both
are pre-existing framework limits this port's own testing was the first
to exercise at whole-8B-model scale; see this repo's own roadmap ledger's
Phase 10 for the full diagnosis. A machine with more RAM, a real
int8-capable CPU path, or a discrete GPU can run `brain minimaxmusic3
generate` (or `crates/minimaxmusic3/tests/e2e_short_generation.rs`
directly) as-is - every piece of the pipeline is real, tested code, not a
stub.

Generation is single-sequence only (`crates/qwen3`'s own incremental
decode API is `b=1`): no batched multi-request AR serving.

License: MiniMax-Music3 Community License (attribution + a UI credit
requirement for commercial use, a >$20M/yr revenue registration clause; no
non-commercial restriction). See the checkpoint's own `LICENSE` file.
